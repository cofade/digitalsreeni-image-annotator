"""
Standalone Grounding DINO worker --- runs in an isolated subprocess.

No PyQt5 imports. Loads torch/transformers in a clean process.
Communication:  stdin -> JSON request, stdout -> JSON response.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

# --- constants --------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_DEVICE = "cpu"
MAX_AREA_FRAC = 0.70
DEFAULT_CROSS_CLASS_NMS_THR = 0.50

# cached models (loaded once per worker lifetime)
_gdino_proc = None
_gdino_model = None
_loaded_model_path = None


# --- helpers ----------------------------------------------------------------

def _log(msg: str):
    print(f"[DINO] {msg}", flush=True)


def _load_models(model_path: str):
    """Load (cache) Grounding DINO model."""
    global _gdino_proc, _gdino_model, _loaded_model_path

    if _loaded_model_path == model_path and _gdino_model is not None:
        return _gdino_proc, _gdino_model

    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
    )

    _log(f"Loading Grounding DINO from {model_path} ...")
    if not Path(model_path).exists():
        _log("Local path missing; will attempt HF hub download.")

    proc = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
    model.eval().to(DEVICE)

    _gdino_proc = proc
    _gdino_model = model
    _loaded_model_path = model_path
    _log("Model loaded successfully.")
    return proc, model


def _run_dino_for_class(image_pil, class_cfg, gdino_proc, gdino_model):
    """Single DINO inference for one class. Returns (boxes, scores, labels)."""
    phrases = class_cfg.get("phrases", [class_cfg["name"]])
    if class_cfg["name"] not in phrases:
        phrases = [class_cfg["name"]] + list(phrases)

    clean_phrases = [p.strip().rstrip(".") for p in phrases if p.strip()]
    prompt = " . ".join(clean_phrases) + " ."

    box_thr = class_cfg.get("box_thr", 0.25)
    txt_thr = class_cfg.get("txt_thr", 0.25)
    nms_thr = class_cfg.get("nms_thr", 0.50)

    _log(
        f'  Class: "{class_cfg["name"]}" '
        f'({len(clean_phrases)} phrase(s), '
        f'box={box_thr:.2f} txt={txt_thr:.2f} nms={nms_thr:.2f})'
    )

    inputs = gdino_proc(
        images=image_pil,
        text=prompt,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    det = gdino_proc.post_process_grounded_object_detection(
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

    # Area filter: discard boxes covering > MAX_AREA_FRAC of image
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

    # Single-class pass: override all labels to canonical name
    norm_labels = [class_cfg["name"]] * len(raw_labels)

    return boxes, scores, norm_labels


def run_dino_detection(image_path: str, class_configs: list[dict],
                       model_path: str, cross_class_nms_thr: float | None = None) -> list[dict]:
    """
    Run DINO detection. Returns list of:
        {"class_name": str, "bbox": [x1, y1, x2, y2], "score": float, "label": str}
    """
    gdino_proc, gdino_model = _load_models(model_path)
    image_pil = Image.open(image_path).convert("RGB")

    all_boxes, all_scores, all_labels = [], [], []

    gdino_model.to(DEVICE)
    for cfg in class_configs:
        boxes, scores, labels = _run_dino_for_class(image_pil, cfg, gdino_proc, gdino_model)
        if len(boxes):
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.extend(labels)

    gdino_model.to(CPU_DEVICE)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    if not all_boxes:
        return []

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)

    # Cross-class NMS
    cc_thr = cross_class_nms_thr if cross_class_nms_thr is not None else DEFAULT_CROSS_CLASS_NMS_THR
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


# --- main -------------------------------------------------------------------

def main():
    raw = sys.stdin.read()
    if not raw.strip():
        return

    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc}"}))
        return

    action = request.get("action")
    if action != "detect":
        print(json.dumps({"error": f"Unknown action: {action}"}))
        return

    image_path = request.get("image_path")
    class_configs = request.get("class_configs", [])
    model_path = request.get("model_path", "models/grounding-dino-base")
    cc_nms = request.get("cross_class_nms_thr")

    if not image_path or not class_configs:
        print(json.dumps({"error": "Missing image_path or class_configs."}))
        return

    env_device = os.environ.get("DINO_DEVICE")
    if env_device:
        global DEVICE
        DEVICE = env_device
        _log(f"Using device override: {DEVICE}")
    else:
        _log(f"Using device: {DEVICE}")

    try:
        results = run_dino_detection(
            image_path, class_configs, model_path,
            cross_class_nms_thr=cc_nms
        )
        print(json.dumps({"results": results}))
    except Exception:
        print(json.dumps({"error": traceback.format_exc()}))


if __name__ == "__main__":
    main()
