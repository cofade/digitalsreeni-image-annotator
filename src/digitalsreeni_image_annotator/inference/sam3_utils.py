"""
SAM 3 text-prompt utilities — runs Ultralytics ``SAM3SemanticPredictor``
in-process (issue #50, ADR-038).

Where it sits
-------------
SAM 3 is a **new producer** wired into the exact spot the two-stage
"DINO boxes → SAM 2 masks" pipeline occupies today. Instead of DINO
detecting boxes and SAM 2 refining them, SAM 3 takes a text phrase and
emits masks + boxes + scores directly. Everything downstream (the
temp-annotation review overlay, Enter/Escape accept-reject, batch over
images + slices, auto-accept, ``.iap`` persistence) is reused verbatim —
see ``controllers/dino_controller.py``. Grounding-DINO stays as a
fallback and its path is behaviourally unchanged.

Deliberately parallel to ``SAMUtils`` / ``DINOUtils``. The shared
threading scaffolding (``_run_sync``, the module-level
``_inference_in_flight`` busy flag, ``InferenceBusyError``) and the
geometry / marshalling helpers (``_qimage_to_numpy``, ``_mask_to_polygon``)
are imported from ``sam_utils`` — NOT forked — so SAM 3 serialises
against SAM 2 and DINO on the one busy flag (desirable: they'd race on
the GPU otherwise) and shares the exact contour/area-floor rules.

Threading model
---------------
Same as ``SAMUtils``: model load and inference run on a worker thread
via ``_run_sync`` while the caller's (GUI) thread pumps its event loop.
The public API looks synchronous. Call it from the GUI thread only —
the busy-flag tripwire in ``_run_sync`` enforces that.

Weights
-------
The real ``sam3.pt`` checkpoint is ~3.45 GB and gated on Hugging Face.
We never auto-download it (mirrors DINO's gated-download UX): the
controller calls :meth:`weights_available` to decide between "Ready"
and a "request access + place sam3.pt" status. ``ensure_loaded`` only
flips :attr:`loaded` / :attr:`_predictor` AFTER a successful construct —
never a half-loaded state.

``ultralytics.models.sam.SAM3SemanticPredictor`` is imported lazily
inside the load function (ADR-012 / ADR-016): keeps app startup fast,
keeps the app importable on older ultralytics that predate SAM 3, and
avoids pulling torch in early.
"""

from __future__ import annotations

import os

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage

from .sam_utils import (
    InferenceBusyError,  # noqa: F401 — re-exported for callers/tests
    SAM_MODELS_DIR,
    _mask_to_polygon,
    _qimage_to_numpy,
    _run_sync,
)

from ..core.logging_config import get_logger

logger = get_logger(__name__)


# Human-readable entry shown in the DINO model dropdown. Single source of
# truth: imported by ui/sidebar.py (the addItem) and dino_controller.py
# (the "is SAM 3 selected?" check) so the label can't drift between them.
SAM3_MODEL_LABEL = "SAM 3 (text prompt)"

# Bare checkpoint filename. Ultralytics resolves a bare name against cwd
# and its own weights dir; we additionally look under the app models dir
# (parallel to the SAM 2 weights) and an explicit env override.
SAM3_WEIGHTS_FILENAME = "sam3.pt"

# Construction-time confidence floor. SAM 3's ONLY threshold knob is
# ``conf`` (set once at construct time), but the review UI carries a
# per-class ``box_thr``. We construct with this low floor so the
# predictor doesn't pre-filter above any class threshold, then apply
# each class's ``box_thr`` in Python (authoritative). A user box_thr
# below this floor can't recover instances the predictor already
# dropped — acceptable; defaults are 0.25.
SAM3_CONF_FLOOR = 0.05


class SAM3Utils(QObject):
    """In-process Ultralytics SAM 3 text-prompt wrapper with a cached model.

    Public API (kept stable — #51 extends this with ``track``):

    - ``ensure_loaded() -> None``
    - ``detect_text(image: QImage, class_configs: list[dict]) -> list[dict] | None``
    - ``unload() -> None``
    - ``weights_available() -> bool``
    - attributes ``loaded: bool`` and ``_predictor`` (None until loaded)
    """

    model_changed = pyqtSignal(str)

    def __init__(self, conf: float = SAM3_CONF_FLOOR):
        super().__init__()
        self.loaded = False
        self._predictor = None       # SAM3SemanticPredictor once loaded
        self._device: str | None = None
        self._conf = float(conf)

    # ── weights resolution (no download) ───────────────────────────────

    def _candidate_weight_paths(self) -> list[str]:
        """Locations we accept a pre-placed ``sam3.pt`` from (no download)."""
        paths = []
        env = os.environ.get("SAM3_MODEL_PATH")
        if env:
            paths.append(env)
        paths.append(os.path.join(os.getcwd(), SAM3_WEIGHTS_FILENAME))
        paths.append(os.path.join(SAM_MODELS_DIR, SAM3_WEIGHTS_FILENAME))
        return paths

    def _resolve_weights_path(self) -> str | None:
        """First existing candidate path, or None if none is present."""
        for p in self._candidate_weight_paths():
            if p and os.path.exists(p):
                return p
        return None

    def weights_available(self) -> bool:
        """True if ``sam3.pt`` is already on disk somewhere resolvable.

        Never triggers a download — the checkpoint is gated on HF. The
        controller uses this to choose between enabling detection and
        showing a "request access + place sam3.pt" status.
        """
        return self._resolve_weights_path() is not None

    # ── model lifecycle ────────────────────────────────────────────────

    def ensure_loaded(self) -> None:
        """Construct the predictor if not already loaded (lazy — ADR-012).

        Runs the construct on a worker thread via ``_run_sync`` (torch /
        model build can stall the UI). Only marks ``loaded`` / sets
        ``_predictor`` on success; on failure leaves ``loaded=False`` /
        ``_predictor=None`` and re-raises so the controller can surface
        the error and reset the picker.
        """
        if self.loaded and self._predictor is not None:
            return
        _run_sync(self._load_model_blocking)
        self.model_changed.emit(SAM3_MODEL_LABEL)
        logger.info("SAM 3 predictor loaded")

    def _load_model_blocking(self) -> None:
        # Lazy import (ADR-016): inside the function so the app stays
        # importable on ultralytics builds without SAM 3 and torch isn't
        # pulled in at startup. Constructs the SAM 3 semantic predictor.
        from ultralytics.models.sam import SAM3SemanticPredictor
        from ..core.torch_utils import resolve_torch_device

        self._device, _ = resolve_torch_device()

        weights = self._resolve_weights_path() or SAM3_WEIGHTS_FILENAME
        # NOTE (ADR-038): pass model/task/conf/device only. `quantize=` raises
        # "'quantize' is not a valid YOLO argument" on ultralytics 8.4.51, and
        # `mode=` is redundant. conf is the single threshold knob. `device` is
        # REQUIRED (mirrors SAMUtils): resolve_torch_device() may force "cpu"
        # even when CUDA is present (unsupported compute capability / probe
        # failure); without it Ultralytics auto-picks cuda:0 and crashes on the
        # exact GPU the fallback rejected.
        overrides = dict(
            model=weights, task="segment", conf=self._conf, device=self._device
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)

        # Flip state only after a clean construct — never half-loaded.
        self._predictor = predictor
        self.loaded = True

    def unload(self) -> None:
        """Free GPU/CPU memory held by the loaded predictor.

        Mirrors ``SAMUtils.unload`` verbatim: move the nn.Module to CPU,
        drop references, ``gc.collect()``, then
        ``empty_cache()`` + ``ipc_collect()`` + ``synchronize()``. Caveat
        (same as SAM 2 / DINO): PyTorch keeps a per-process CUDA context
        that survives unload; full reclaim needs an app restart.
        """
        import gc
        try:
            model = getattr(self._predictor, "model", None)
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            logger.warning(
                "unload: moving SAM 3 model to CPU failed; GPU memory may not "
                "be fully released", exc_info=True,
            )
        self._predictor = None
        self.loaded = False
        self._device = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            logger.warning(
                "CUDA cache cleanup failed during SAM 3 unload; GPU memory may "
                "not be fully released", exc_info=True,
            )
        logger.info("SAM 3 unload complete")

    # ── inference ──────────────────────────────────────────────────────

    def detect_text(self, image: QImage, class_configs: list[dict]):
        """Run text-prompted segmentation for each class config.

        ``class_configs`` share the DINO shape (from
        ``_build_dino_class_configs``): keys ``name``, ``phrases``,
        ``box_thr``, ``txt_thr``, ``nms_thr``. For SAM 3 only ``name``,
        ``phrases`` and ``box_thr`` are used — ``box_thr`` is the
        confidence filter; ``txt_thr`` / ``nms_thr`` are **ignored**
        (SAM 3 exposes a single confidence knob).

        Returns a COMBINED per-instance list::

            {"class_name": str, "score": float,
             "segmentation": [x1, y1, x2, y2, ...],
             "bbox": [x1, y1, x2, y2]}

        or ``None`` if the model is not loaded (never half-work). An
        instance whose mask falls below ``_mask_to_polygon``'s area-10
        contour floor yields no entry. Runs entirely through
        ``_run_sync`` (GUI-thread tripwire + shared in-flight guard).
        """
        if not self.loaded or self._predictor is None:
            logger.warning("detect_text: SAM 3 not loaded")
            return None
        return _run_sync(
            self._detect_text_blocking,
            _qimage_to_numpy(image),
            list(class_configs),
        )

    def _detect_text_blocking(self, image_np, class_configs: list[dict]):
        # Worker thread. set_image once, then one predictor call per class
        # with that class's phrase list. Filter each instance by the
        # class's box_thr, convert its mask to a polygon.
        if not self.loaded or self._predictor is None:
            return None

        self._predictor.set_image(image_np)

        out = []
        for cfg in class_configs:
            name = cfg["name"]
            phrases = self._clean_phrases(cfg, name)
            box_thr = cfg.get("box_thr", 0.25)

            results = self._predictor(text=phrases)
            res = results[0] if isinstance(results, (list, tuple)) else results

            masks = getattr(res, "masks", None)
            boxes = getattr(res, "boxes", None)
            if masks is None or boxes is None:
                continue

            mask_arr = masks.data.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            box_arr = boxes.xyxy.cpu().numpy()

            for i in range(len(mask_arr)):
                score = float(conf_arr[i]) if i < len(conf_arr) else 0.0
                if score < box_thr:
                    continue
                polygon = _mask_to_polygon(mask_arr[i])
                if polygon is None:
                    continue
                bbox = (
                    [float(v) for v in box_arr[i]]
                    if i < len(box_arr) else [0.0, 0.0, 0.0, 0.0]
                )
                out.append({
                    "class_name": name,
                    "score": score,
                    "segmentation": polygon,
                    "bbox": bbox,
                })

        logger.debug("detect_text: %d instance(s) across %d class(es)",
                     len(out), len(class_configs))
        return out

    @staticmethod
    def _clean_phrases(cfg: dict, name: str) -> list[str]:
        """Sanitise the phrase list; fall back to the class name."""
        phrases = list(cfg.get("phrases") or [name])
        clean = [p.strip().rstrip(".") for p in phrases if p and p.strip()]
        return clean or [name]
