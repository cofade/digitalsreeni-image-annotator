"""
Grounding DINO utilities — runs HF Transformers DINO in-process.

History
-------
The previous version delegated to ``dino_worker.py`` over subprocess to
dodge the same Windows + Python 3.14 + PyQt5 DLL conflict that motivated
the SAM worker (ADR-011). With PyQt6 in place the conflict is gone and
we run inference directly — saves a process spawn per detection call
and lets the model stay resident in memory between calls.

Threading model
---------------
Same as ``sam_utils.SAMUtils``: inference runs on a worker thread; the
caller's thread pumps its event loop while waiting via ``_run_sync``.
torch + transformers are imported lazily on first detect call.
"""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QImage

from .sam_utils import _qimage_to_numpy, _run_sync
from .utils import models_base_dir


GDINO_MODEL_NAMES = [
    "grounding-dino-base",
    "grounding-dino-tiny",
]

# Area filter: discard DINO boxes that cover more than this fraction of
# the image. Catches degenerate "whole image" detections from generic
# phrases ("object", "thing", etc.). Same value used in the old worker.
MAX_AREA_FRAC = 0.70

# Default IoU threshold for cross-class NMS across all classes.
DEFAULT_CROSS_CLASS_NMS_THR = 0.50


def _gdino_local_path(model_name: str) -> str:
    """Canonical local install path for a Grounding DINO model."""
    return os.path.join(models_base_dir(), model_name)


GDINO_MODEL_PATHS = {
    "grounding-dino-base": _gdino_local_path("grounding-dino-base"),
    "grounding-dino-tiny": _gdino_local_path("grounding-dino-tiny"),
}

GDINO_REPO_IDS = {
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
}


class DINOUtils(QObject):
    """In-process Grounding DINO wrapper with a cached model."""

    model_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._proc = None             # AutoProcessor instance
        self._model = None            # AutoModelForZeroShotObjectDetection
        self._loaded_model_path: str | None = None
        self._device: str | None = None  # set on first load

    # ── model lifecycle ───────────────────────────────────────────────

    def _resolve_device(self) -> str:
        """Pick CUDA if available; honour DINO_DEVICE env override."""
        env = os.environ.get("DINO_DEVICE")
        if env:
            return env
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_model_blocking(self, model_path: str) -> None:
        """Load (cache) the Grounding DINO model for ``model_path``."""
        # Lazy imports so app startup doesn't pay the torch+transformers
        # tax for users who never run detection.
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        device = self._resolve_device()
        print(f"[DINO] Loading from {model_path} on {device} ...")
        if not Path(model_path).exists():
            print(f"[DINO] Local path missing; will attempt HF hub download.")

        proc = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
        model.eval().to(device)

        self._proc = proc
        self._model = model
        self._loaded_model_path = model_path
        self._device = device
        print("[DINO] Model loaded successfully.")

    def unload(self) -> None:
        """Drop the cached model so its GPU/CPU memory comes back."""
        self._proc = None
        self._model = None
        self._loaded_model_path = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── inference ─────────────────────────────────────────────────────

    def detect(
        self,
        image: QImage,
        class_configs: list[dict],
        model_name: str = "grounding-dino-base",
        custom_model_path: str | None = None,
        cross_class_nms_thr: float | None = None,
    ):
        """Run text-prompted detection. Returns list of dicts:

            {"class_name": str, "bbox": [x1, y1, x2, y2],
             "score": float,  "label": str}

        Returns ``None`` on error (model resolution failure or runtime
        exception). An empty list means "ran, no boxes survived
        filtering".
        """
        model_path = custom_model_path or GDINO_MODEL_PATHS.get(model_name)
        if model_path is None:
            print(f"Unknown DINO model: {model_name}")
            return None

        # Marshal to numpy on the calling thread so the worker doesn't
        # touch the QImage (Qt objects are not designed to cross threads).
        image_np = _qimage_to_numpy(image)

        return _run_sync(
            self._detect_blocking,
            image_np,
            list(class_configs),
            model_path,
            cross_class_nms_thr,
        )

    def _detect_blocking(
        self,
        image_np,
        class_configs: list[dict],
        model_path: str,
        cross_class_nms_thr: float | None,
    ):
        # We're already on a worker thread (called via _run_sync). Load
        # the model directly here when needed — calling _run_sync from
        # within would deadlock against the outer QEventLoop.
        if self._loaded_model_path != model_path or self._model is None:
            try:
                self._load_model_blocking(model_path)
            except Exception:
                import traceback
                traceback.print_exc()
                return None

        import torch
        from PIL import Image as PILImage
        from torchvision.ops import nms

        image_pil = PILImage.fromarray(image_np).convert("RGB")
        device = self._device or "cpu"

        all_boxes, all_scores, all_labels = [], [], []
        # Ensure model is on the active device for this call (cheap if
        # already there) — guards against an earlier off-load.
        self._model.to(device)

        for cfg in class_configs:
            boxes, scores, labels = self._run_for_class(image_pil, cfg, device)
            if len(boxes):
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.extend(labels)

        # Off-load to CPU between batch calls; harmless if device is CPU.
        self._model.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()

        if not all_boxes:
            return []

        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)

        # Cross-class NMS — drop boxes that overlap heavily across
        # classes so the user doesn't get two near-identical masks
        # for one object.
        cc_thr = (
            cross_class_nms_thr
            if cross_class_nms_thr is not None
            else DEFAULT_CROSS_CLASS_NMS_THR
        )
        cross_keep = nms(all_boxes, all_scores, cc_thr).tolist()
        all_boxes = all_boxes[cross_keep]
        all_scores = all_scores[cross_keep]
        all_labels = [all_labels[i] for i in cross_keep]

        results = []
        for i in range(len(all_boxes)):
            box = all_boxes[i].numpy().tolist()
            results.append({
                "class_name": all_labels[i],
                "bbox": [float(v) for v in box],
                "score": float(all_scores[i].item()),
                "label": all_labels[i],
            })
        return results

    def _run_for_class(self, image_pil, class_cfg, device):
        """Single DINO inference for one class. Returns (boxes, scores, labels)."""
        import torch
        from torchvision.ops import nms

        phrases = class_cfg.get("phrases", [class_cfg["name"]])
        if class_cfg["name"] not in phrases:
            phrases = [class_cfg["name"]] + list(phrases)

        clean_phrases = [p.strip().rstrip(".") for p in phrases if p.strip()]
        prompt = " . ".join(clean_phrases) + " ."

        box_thr = class_cfg.get("box_thr", 0.25)
        txt_thr = class_cfg.get("txt_thr", 0.25)
        nms_thr = class_cfg.get("nms_thr", 0.50)

        print(
            f'[DINO]   Class: "{class_cfg["name"]}" '
            f'({len(clean_phrases)} phrase(s), '
            f'box={box_thr:.2f} txt={txt_thr:.2f} nms={nms_thr:.2f})'
        )

        inputs = self._proc(
            images=image_pil,
            text=prompt,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        det = self._proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_thr,
            text_threshold=txt_thr,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        boxes = det["boxes"].cpu()
        scores = det["scores"].cpu()
        raw_labels = det.get("text_labels", det.get("labels", []))

        if len(boxes) == 0:
            return torch.zeros((0, 4)), torch.zeros(0), []

        # Area filter
        iw, ih = image_pil.size
        area = iw * ih
        keep = [
            i for i, b in enumerate(boxes)
            if ((b[2] - b[0]) * (b[3] - b[1])).item() / area < MAX_AREA_FRAC
        ]
        if not keep:
            return torch.zeros((0, 4)), torch.zeros(0), []

        boxes = boxes[keep]
        scores = scores[keep]
        raw_labels = [raw_labels[i] for i in keep]

        # Per-class NMS
        keep2 = nms(boxes, scores, nms_thr).tolist()
        boxes = boxes[keep2]
        scores = scores[keep2]
        raw_labels = [raw_labels[i] for i in keep2]

        # Override DINO's free-text labels to our canonical class name
        norm_labels = [class_cfg["name"]] * len(raw_labels)
        return boxes, scores, norm_labels

    # ── model download ────────────────────────────────────────────────

    def download_model(self, model_name: str):
        """Download model from Hugging Face Hub into the canonical local path.

        Returns the absolute local path on success, or None on error.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface_hub not installed. Cannot download models.")
            return None

        repo_id = GDINO_REPO_IDS.get(model_name)
        if not repo_id:
            print(f"No repo ID for model: {model_name}")
            return None

        local_path = GDINO_MODEL_PATHS.get(model_name) or _gdino_local_path(model_name)
        if os.path.exists(local_path):
            print(f"Model already exists at {local_path}")
            return local_path

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {repo_id} -> {local_path} ...")
        snapshot_download(repo_id, local_dir=local_path)
        print(f"Done. Model at {local_path}")
        return local_path
