"""SAM 2 / 2.1 fine-tuning engine — Ultralytics-native custom training loop.

Why this exists
---------------
Ultralytics has **no SAM trainer**: ``SAM.task_map`` registers only a
*predictor* for the ``segment`` task, so ``SAM(...).train()`` raises
``NotImplementedError`` (verified on ultralytics 8.4.51). We therefore
cannot mirror the YOLO ``model.train(data=yaml, ...)`` path.

What we do instead
------------------
``SAM(...).model`` is a plain ``nn.Module`` (``SAM2Model``) exposing
``image_encoder`` / ``sam_prompt_encoder`` / ``sam_mask_decoder``, and the
Ultralytics ``SAM2Predictor`` already implements the forward path in reusable
pieces — ``get_im_features`` (image encoder) and ``prompt_inference`` /
``_inference_features`` (prompt encoder + mask decoder). We call those methods
**under autograd** (they are not wrapped in ``inference_mode`` unless reached
via the public ``__call__``), add a focal+dice loss and an AdamW step, and
save a checkpoint that reloads through the existing ``SAM(path)`` inference
path. No extra dependency, and the result drops straight into the app's SAM
model selector.

This keeps our exposure confined to a thin adapter over already-exercised
predictor methods. The spike that validated the mechanic lives in the PR for
issue bnsreenu#73.

Threading
---------
``train()`` is **blocking and CPU/GPU-bound**; the controller runs it on a
dedicated ``QThread`` (never the GUI thread). Unlike SAM inference it does
*not* go through ``sam_utils._run_sync`` — that helper's re-entry guard is
GUI-thread-local.

The trainer loads its **own** ``SAM`` instance (see ``_build_predictor``); it
never touches ``SAMUtils._model``, so this is not "two threads driving one
model". The hazard it *does* create is two SAM models (the resident inference
one and this training one) competing for the same GPU/CUDA context. The
controller therefore locks the SAM inference UI (tools + model selector + the
fine-tune menu) for the duration so no concurrent inference or model swap can
be triggered while a run is in flight.
"""

from __future__ import annotations

import os
import random

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from ..inference.sam_utils import MODEL_FILES, MODEL_NAMES, SAM_MODELS_DIR

# Fine-tuned checkpoints live alongside the base weights, namespaced so they
# never collide with Ultralytics' auto-downloaded base files.
SAM_CUSTOM_DIR = os.path.join(SAM_MODELS_DIR, "custom")


def make_custom_filename(base_model: str, name: str) -> str:
    """Build a fine-tuned checkpoint path under ``SAM_CUSTOM_DIR``.

    Ultralytics' ``build_sam`` selects the architecture by ``ckpt.endswith(token)``
    where ``token`` is the base file name (e.g. ``sam2_t.pt``). A fine-tuned file
    therefore **must** keep that suffix or ``SAM(path)`` raises "not a supported
    SAM model". We sanitise the user label and append ``_<base_token>``.
    """
    token = MODEL_FILES.get(base_model, os.path.basename(base_model))
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    safe = safe or "finetuned"
    return os.path.join(SAM_CUSTOM_DIR, f"{safe}_{token}")


def list_custom_models() -> dict:
    """``{display_name: path}`` for fine-tuned checkpoints, for the SAM selector."""
    out = {}
    if os.path.isdir(SAM_CUSTOM_DIR):
        for fn in sorted(os.listdir(SAM_CUSTOM_DIR)):
            if fn.endswith(".pt"):
                out[f"★ {os.path.splitext(fn)[0]}"] = os.path.join(SAM_CUSTOM_DIR, fn)
    return out


# ── geometry: annotation → (mask, prompt) ───────────────────────────────────

def polygon_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Flat ``[x1,y1,x2,y2,...]`` polygon → bool mask (inverse of
    ``sam_utils._mask_to_polygon``)."""
    pts = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def bbox_to_mask(bbox, height: int, width: int) -> np.ndarray:
    """``[x, y, w, h]`` → bool mask."""
    x, y, w, h = bbox
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), 1, thickness=-1)
    return mask.astype(bool)


def mask_to_xyxy(mask: np.ndarray):
    """Tight ``[x1, y1, x2, y2]`` bounding box of a bool mask, or None if empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def mask_to_point(mask: np.ndarray):
    """A single foreground point well inside the mask.

    Uses the distance transform's argmax so the point sits near the medial
    axis rather than on an edge — a more stable positive prompt than a random
    interior pixel.
    """
    m = mask.astype(np.uint8)
    if m.sum() == 0:
        return None
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [float(x), float(y)]


# ── loss ────────────────────────────────────────────────────────────────────

def _focal_dice_loss(logits, target, focal_weight: float = 20.0):
    """SAM's mask supervision: focal + dice, ≈20:1 (focal:dice).

    ``logits`` and ``target`` are ``(1, 1, H, W)`` on the same device; target
    is float {0,1}. Matches the recipe used across the SAM fine-tuning
    literature (focal for hard pixels, dice for region overlap).
    """
    import torch
    import torch.nn.functional as F

    prob = torch.sigmoid(logits)
    # Focal (binary), gamma=2.
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    focal = (bce * (1 - p_t).pow(2)).mean()
    # Dice.
    inter = (prob * target).sum()
    dice = 1 - (2 * inter + 1) / (prob.sum() + target.sum() + 1)
    return focal_weight * focal + dice


# ── dataset ──────────────────────────────────────────────────────────────────

class SampleGroup:
    """One image plus the specs for its instances, loaded lazily.

    ``image_loader`` returns an RGB ``uint8`` array (matching what
    ``sam_utils._qimage_to_numpy`` feeds inference — channel-order consistency
    between train and predict matters more than absolute order). ``specs`` are
    raw annotations (``{"segmentation": [...]}`` or ``{"bbox": [x,y,w,h]}``)
    rasterised to bool masks **at load time** using the actual image size, so
    masks are never held in RAM between epochs and always match the image.
    """

    def __init__(self, image_loader, specs):
        self._image_loader = image_loader
        self.specs = specs
        self.n_instances = len(specs)

    def load(self):
        """Return ``(image_rgb, [{"mask": bool HxW}, ...])``."""
        image = self._image_loader()
        h, w = image.shape[:2]
        instances = []
        for spec in self.specs:
            if spec.get("segmentation"):
                mask = polygon_to_mask(spec["segmentation"], h, w)
            elif spec.get("bbox"):
                mask = bbox_to_mask(spec["bbox"], h, w)
            else:
                continue
            if mask.any():
                instances.append({"mask": mask})
        return image, instances


# ── engine ───────────────────────────────────────────────────────────────────

class SAMFineTuner(QObject):
    """Fine-tunes a SAM 2 mask decoder (optionally image encoder) on
    user instances. Mirrors ``YOLOTrainer``'s signal/stop surface so the
    controller and progress dialog wiring is identical."""

    progress_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.stop_training = False

    def stop_training_signal(self):
        self.stop_training = True
        self.progress_signal.emit("Stopping after current step…")

    # -- model setup ---------------------------------------------------------

    def _build_predictor(self, base_model):
        """Return a ready ``SAM2Predictor`` for ``base_model`` (a registry name
        like ``"SAM 2 tiny"`` or a path to a ``.pt``).

        Forces predictor creation with one throwaway predict so ``set_image`` /
        ``prompt_inference`` are usable. Pins the device via
        ``resolve_torch_device`` so an incompatible GPU (which Ultralytics would
        otherwise pick blindly and crash on) is honoured as CPU — the same
        device decision SAM/DINO/YOLO inference already share."""
        from ultralytics import SAM

        from ..core.torch_utils import resolve_torch_device

        if base_model in MODEL_NAMES:
            model_file = os.path.join(SAM_MODELS_DIR, MODEL_FILES[base_model])
        else:
            model_file = base_model
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Base SAM weights not found: {model_file}")

        device, _ = resolve_torch_device()
        model = SAM(model_file)
        warm = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        # device= forces the warmup (and predictor model placement) onto the
        # resolved device rather than Ultralytics' default cuda-if-present.
        model(warm, bboxes=[[8, 8, 56, 56]], device=device, verbose=False)
        return model, model.predictor, model_file

    @staticmethod
    def _apply_freeze(net, freeze_image_encoder: bool):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        for p in net.sam_mask_decoder.parameters():
            p.requires_grad_(True)
        if not freeze_image_encoder:
            for p in net.image_encoder.parameters():
                p.requires_grad_(True)
            net.image_encoder.train()
        trainable = [p for p in net.parameters() if p.requires_grad]
        n = sum(p.numel() for p in trainable)
        return trainable, n

    # -- training ------------------------------------------------------------

    def train(
        self,
        base_model,
        groups,
        *,
        epochs: int = 10,
        lr: float = 1e-4,
        batch_size: int = 1,
        freeze_image_encoder: bool = True,
        prompt_type: str = "bbox",
        out_path: str,
    ) -> dict:
        """Fine-tune and save. Returns a small result dict; raises on failure.

        ``groups`` is an iterable of :class:`SampleGroup`. ``batch_size`` is
        treated as a gradient-accumulation count (SAM prompts are per-object,
        so we step the optimizer every ``batch_size`` instances).
        """
        import torch

        groups = list(groups)
        if not groups:
            raise ValueError("No annotated instances to train on.")
        if prompt_type not in ("bbox", "point"):
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        model, pred, base_file = self._build_predictor(base_model)
        net = pred.model
        device = next(net.parameters()).device
        trainable, n_trainable = self._apply_freeze(net, freeze_image_encoder)
        self.progress_signal.emit(
            f"Base: {os.path.basename(base_file)} | device: {device} | "
            f"images: {len(groups)} | trainable params: {n_trainable:,} | "
            f"encoder {'TRAINED' if not freeze_image_encoder else 'frozen'}"
        )

        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.1)
        self.stop_training = False
        total_instances = sum(g.n_instances for g in groups)
        if total_instances == 0:
            raise ValueError("No annotated instances to train on.")

        for epoch in range(1, epochs + 1):
            if self.stop_training:
                break
            random.shuffle(groups)
            epoch_loss, seen, accum = 0.0, 0, 0
            optimizer.zero_grad()

            for group in groups:
                if self.stop_training:
                    break
                image, instances = group.load()
                h, w = image.shape[:2]
                im_t = self._set_image(pred, image, freeze_image_encoder)

                # Accumulate all of THIS image's instance losses and backward
                # ONCE per image. When the image encoder is trainable, every
                # instance's decoder graph hangs off the one shared encoder
                # feature graph; a per-instance backward() would free that
                # shared graph and make the next instance raise "backward
                # through the graph a second time". One backward per image
                # keeps the shared graph alive for exactly one pass.
                inst_losses = []
                for inst in instances:
                    bbox = mask_to_xyxy(inst["mask"])
                    if bbox is None:
                        continue
                    prompt = self._prompt_kwargs(prompt_type, inst["mask"], bbox)
                    with torch.enable_grad():
                        pm, _ = pred.prompt_inference(
                            im_t, multimask_output=False, **prompt
                        )
                    logits = torch.nn.functional.interpolate(
                        pm[:1].unsqueeze(0).float(), size=(h, w),
                        mode="bilinear", align_corners=False,
                    )
                    target = torch.from_numpy(inst["mask"].astype(np.float32))[None, None].to(device)
                    inst_losses.append(_focal_dice_loss(logits, target))

                del image  # bound memory: encoder features recomputed per epoch
                if not inst_losses:
                    continue

                # batch_size = number of IMAGES to accumulate before an
                # optimizer step (gradient accumulation over images).
                image_loss = torch.stack(inst_losses).mean() / max(1, batch_size)
                image_loss.backward()
                epoch_loss += image_loss.detach().item() * max(1, batch_size)
                seen += 1
                accum += 1
                if accum >= batch_size:
                    optimizer.step()
                    optimizer.zero_grad()
                    accum = 0

            if accum > 0:  # flush partial accumulation
                optimizer.step()
                optimizer.zero_grad()
            avg = epoch_loss / max(1, seen)
            self.progress_signal.emit(f"Epoch {epoch}/{epochs}  loss={avg:.4f}")

        result = self._save_and_verify(net, base_file, out_path)
        result.update(stopped=self.stop_training, instances=total_instances)
        self.progress_signal.emit(
            f"Saved fine-tuned model: {out_path}" if not self.stop_training
            else f"Stopped early — saved current state to {out_path}"
        )
        return result

    def _set_image(self, pred, image: np.ndarray, freeze_image_encoder: bool):
        """Preprocess ``image`` and compute encoder features into the predictor.

        Sets ``pred.batch`` explicitly so ``_prepare_prompts`` maps bbox/point
        prompts from this image's original pixel size into model coordinates —
        a stale ``batch`` from a different-sized image would mis-scale prompts.
        """
        import torch

        pred.setup_source(image)
        im_t = None
        for batch in pred.dataset:
            im_t = pred.preprocess(batch[1])
            break
        # (paths, im0s, ...) — prompt_inference reads self.batch[1][0].shape for orig H,W.
        pred.batch = (None, [image], None)
        if freeze_image_encoder:
            with torch.no_grad():
                pred.features = pred.get_im_features(im_t)
        else:
            pred.features = pred.get_im_features(im_t)
        return im_t

    @staticmethod
    def _prompt_kwargs(prompt_type: str, mask: np.ndarray, bbox):
        if prompt_type == "bbox":
            return {"bboxes": [bbox]}
        pt = mask_to_point(mask)
        return {"points": [pt], "labels": [1]}

    # -- checkpoint ----------------------------------------------------------

    def _save_and_verify(self, net, base_file: str, out_path: str) -> dict:
        """Save fine-tuned weights as ``{"model": state_dict}`` and prove they
        reload through ``SAM(out_path)``.

        Ultralytics' ``_load_checkpoint`` reads only the nested ``"model"`` key
        (a tensor dict) and rebuilds the architecture from the filename suffix,
        so a pure state_dict is all we need — no need to ``torch.load`` (and
        unpickle) the base file. Because ``net`` is the same ``SAM2Model`` class
        Ultralytics instantiates, the keys match exactly, sidestepping the
        "Unexpected key(s)" reload failures (facebookresearch/sam2#337).
        """
        import torch
        from ultralytics import SAM

        token = os.path.basename(base_file)
        if not os.path.basename(out_path).endswith(token):
            raise ValueError(
                f"Fine-tuned checkpoint name must end with '{token}' so "
                f"Ultralytics can pick the right architecture; got "
                f"'{os.path.basename(out_path)}'. Use make_custom_filename()."
            )

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        net_cpu_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        torch.save({"model": net_cpu_state}, out_path)

        # Round-trip verification: load + one forward. Failing here is loud by design.
        verify = SAM(out_path)
        verify(
            (np.random.rand(64, 64, 3) * 255).astype(np.uint8),
            bboxes=[[8, 8, 56, 56]], verbose=False,
        )
        return {"out_path": out_path, "verified": True}
