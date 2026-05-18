"""
Standalone SAM worker — runs in an isolated subprocess.

This script is intentionally free of PyQt5 imports so it can load
torch/ultralytics in a clean process where the parent GUI's loaded
DLLs do not interfere.

Communication:
  stdin  -> JSON request (image path + model + prompts)
  stdout -> JSON response (polygon + score or error)
"""

from __future__ import annotations

import io
import json
import os
import sys
import traceback

import cv2
import numpy as np
from PIL import Image


# SAM weights live under <project_root>/models/sam/, parallel to the
# DINO models directory (e.g. models/grounding-dino-base/).
SAM_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "sam",
)

MODELS = {
    "SAM 2 tiny": os.path.join(SAM_MODELS_DIR, "sam2_t.pt"),
    "SAM 2 small": os.path.join(SAM_MODELS_DIR, "sam2_s.pt"),
    "SAM 2 base": os.path.join(SAM_MODELS_DIR, "sam2_b.pt"),
    "SAM 2 large": os.path.join(SAM_MODELS_DIR, "sam2_l.pt"),
    "SAM 2.1 tiny": os.path.join(SAM_MODELS_DIR, "sam2.1_t.pt"),
    "SAM 2.1 small": os.path.join(SAM_MODELS_DIR, "sam2.1_s.pt"),
    "SAM 2.1 base": os.path.join(SAM_MODELS_DIR, "sam2.1_b.pt"),
    "SAM 2.1 large": os.path.join(SAM_MODELS_DIR, "sam2.1_l.pt"),
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _log_device():
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            print(f"[SAM] Using CUDA: {torch.version.cuda} — {dev}")
        else:
            print("[SAM] No GPU available, running on CPU")
    except Exception:
        pass


def load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def mask_to_polygon(mask: np.ndarray) -> list | None:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            polygon = contour.flatten().tolist()
            if len(polygon) >= 6:
                polygons.append(polygon)
    if not polygons:
        return None
    # Return the polygon with the largest area (ignore tiny noise holes)
    biggest = max(polygons, key=lambda p: cv2.contourArea(np.array(p).reshape(-1, 2)))
    return biggest


def _bbox_of_contour(contour: list) -> tuple[float, float, float, float]:
    pts = np.array(contour).reshape(-1, 2)
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def _bbox_area(bbox: list) -> float:
    return float(max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]))


def _check_points(contour: list, positive: list, negative: list) -> bool:
    """Return True iff all positive points are inside and all negative outside."""
    cnt = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
    for x, y in positive:
        if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) < 0:
            return False
    for x, y in negative:
        if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
            return False
    return True


def _predicted_bbox_area_ratio(pred_contour: list, user_bbox: list) -> float:
    """Ratio of predicted contour bbox area over user-drawn bbox area."""
    px1, py1, px2, py2 = _bbox_of_contour(pred_contour)
    user_area = _bbox_area(user_bbox)
    if user_area == 0:
        return 0.0
    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    return pred_area / user_area


# ── core SAM runner ──────────────────────────────────────────────────────────

def _is_single_bbox(value) -> bool:
    """Return True if value is a single bbox [x1,y1,x2,y2] (list of 4 numbers)."""
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    )


def _filter_and_rank_masks(
    masks, confidences, user_bbox, positive_pts, negative_pts
) -> dict | None:
    """Apply hard constraints and return the best single mask as a dict."""
    best_result = None
    best_score = -1.0

    for i, mask in enumerate(masks):
        contour = mask_to_polygon(mask)
        if contour is None:
            continue

        score = float(confidences[i]) if i < len(confidences) else 0.0
        mask_pixels = int(mask.sum())

        if user_bbox is not None:
            ratio = _predicted_bbox_area_ratio(contour, user_bbox)
            if ratio < 0.20:
                continue
            ux, uy, ux2, uy2 = user_bbox
            uw, uh = ux2 - ux, uy2 - uy
            px, py, px2, py2 = _bbox_of_contour(contour)
            pw, ph = px2 - px, py2 - py
            if pw < 0.5 * uw or ph < 0.5 * uh:
                continue
            if pw > 1.5 * uw or ph > 1.5 * uh:
                continue

        if positive_pts is not None and negative_pts is not None:
            if not _check_points(contour, positive_pts, negative_pts):
                continue

        if mask_pixels > best_score:
            best_score = mask_pixels
            best_result = {
                "segmentation": contour,
                "score": score,
                "mask_pixels": mask_pixels,
            }

    return best_result


def run_sam(
    image_path: str,
    model_name: str,
    bboxes: list | None = None,
    points: dict | None = None,
) -> dict | list[dict]:
    from ultralytics import SAM

    _log_device()

    model_file = MODELS[model_name]
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    sam_model = SAM(model_file)
    image_np = load_image(image_path)

    if points is not None:
        pos = points.get("positive", [])
        neg = points.get("negative", [])
        all_points = [pos + neg]
        all_labels = [([1] * len(pos)) + ([0] * len(neg))]
        if not all_points[0]:
            return {"error": "No points provided."}
        results = sam_model(image_np, points=all_points, labels=all_labels)
        user_bbox = None
        positive_pts = pos
        negative_pts = neg

        masks = results[0].masks.data.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        best = _filter_and_rank_masks(masks, confidences, user_bbox, positive_pts, negative_pts)

        if best is None:
            return {"error": "No SAM mask matches the given constraints. Try repositioning positive/negative points."}
        return {"segmentation": best["segmentation"], "score": best["score"]}

    elif bboxes is not None:
        is_batch = not _is_single_bbox(bboxes)
        sam_bboxes = bboxes if is_batch else [bboxes]

        # Ultralytics always returns [Results] (single Results object)
        results = sam_model(image_np, bboxes=sam_bboxes)
        res = results[0]

        if not (hasattr(res, "masks") and res.masks is not None):
            return [{"error": "No mask generated."}] * len(sam_bboxes) if is_batch else {"error": "No mask generated."}

        masks = res.masks.data.cpu().numpy()  # (N, H, W)
        confidences = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.zeros(len(masks))

        output = []
        for i in range(len(masks)):
            mask = masks[i]
            score = float(confidences[i]) if i < len(confidences) else 0.0
            contour = mask_to_polygon(mask)
            if contour is None:
                output.append({"error": "No valid mask polygon."})
                continue

            mask_pixels = int(mask.sum())
            user_bbox = sam_bboxes[i]

            # hard constraints per bbox
            ratio = _predicted_bbox_area_ratio(contour, user_bbox)
            if ratio < 0.20:
                output.append({"error": "Mask too small relative to box."})
                continue
            ux, uy, ux2, uy2 = user_bbox
            uw, uh = ux2 - ux, uy2 - uy
            px, py, px2, py2 = _bbox_of_contour(contour)
            pw, ph = px2 - px, py2 - py
            if pw < 0.5 * uw or ph < 0.5 * uh:
                output.append({"error": "Mask dimensions too small."})
                continue
            if pw > 1.5 * uw or ph > 1.5 * uh:
                output.append({"error": "Mask dimensions too large."})
                continue

            output.append({"segmentation": contour, "score": score})

        return output if is_batch else output[0]

    else:
        return {"error": "No prompts provided."}


def main():
    raw = sys.stdin.read()
    if not raw.strip():
        return

    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON: {exc}"}))
        return

    image_path = request.get("image_path")
    model_name = request.get("model_name", "SAM 2 tiny")
    bboxes = request.get("bboxes")
    points = request.get("points")

    try:
        result = run_sam(image_path, model_name, bboxes=bboxes, points=points)
    except Exception:
        result = {"error": traceback.format_exc()}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
