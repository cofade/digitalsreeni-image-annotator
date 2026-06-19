"""Build SAM fine-tuning :class:`SampleGroup`s from either the live project
annotations or a prepared on-disk dataset folder.

The project path mirrors the image-resolution logic in
``io.export_formats.export_yolo_v5plus`` (slice lookup via ``slices`` /
``image_slices``; regular images via ``image_paths`` with exact-then-substring
match; TIFF/CZI source files skipped in favour of their extracted slices), so a
dataset that exports cleanly to YOLO also trains cleanly here.
"""

from __future__ import annotations

import json
import os

from PyQt6.QtGui import QImage

from .sam_trainer import SampleGroup
from ..inference.sam_utils import _qimage_to_numpy


def _specs_for(annotations) -> list:
    """Flatten ``{class: [ann, ...]}`` into raw instance specs the
    :class:`SampleGroup` rasterises lazily."""
    specs = []
    for _class_name, class_annotations in (annotations or {}).items():
        for ann in class_annotations:
            if ann.get("segmentation"):
                specs.append({"segmentation": ann["segmentation"]})
            elif ann.get("bbox"):
                specs.append({"bbox": ann["bbox"]})
    return specs


def build_groups_from_project(all_annotations, image_paths, slices, image_slices):
    """Live project annotations → ``list[SampleGroup]``.

    Images load lazily (one at a time during training) to bound memory; in-RAM
    slice QImages are reused directly.
    """
    slice_map = {name: qimage for name, qimage in slices}
    groups = []

    for image_name, image_annotations in all_annotations.items():
        specs = _specs_for(image_annotations)
        if not specs:
            continue

        if image_name in slice_map or ("_" in image_name and "." not in image_name):
            qimage = slice_map.get(image_name)
            if qimage is None:
                for stack_slices in image_slices.values():
                    qimage = next((s[1] for s in stack_slices if s[0] == image_name), None)
                    if qimage is not None:
                        break
            if qimage is None:
                print(f"[SAM dataset] skip slice {image_name!r}: no image data")
                continue
            groups.append(SampleGroup(lambda q=qimage: _qimage_to_numpy(q), specs))
            continue

        image_path = image_paths.get(image_name)
        if image_path is None:
            image_path = next(
                (p for name, p in image_paths.items() if image_name in name), None
            )
        if not image_path:
            print(f"[SAM dataset] skip {image_name!r}: no image_paths entry")
            continue
        if image_path.lower().endswith((".tif", ".tiff", ".czi")):
            print(f"[SAM dataset] skip TIFF/CZI source {image_name!r} (use slices)")
            continue
        groups.append(SampleGroup(lambda p=image_path: _qimage_to_numpy(QImage(p)), specs))

    return groups


# ── prepared folder ──────────────────────────────────────────────────────────

def build_groups_from_folder(folder: str):
    """Read a folder produced by ``export_sam_dataset`` → ``list[SampleGroup]``.

    Expects ``<folder>/manifest.json`` with entries
    ``{"image": "images/x.png", "instances": [{"bbox": [...]}|{"segmentation": [...]}]}``.
    """
    manifest_path = os.path.join(folder, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest.json in {folder}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    groups = []
    for entry in manifest.get("images", []):
        img_rel = entry["image"]
        img_path = os.path.join(folder, img_rel)
        specs = entry.get("instances", [])
        if not specs or not os.path.exists(img_path):
            continue
        groups.append(SampleGroup(lambda p=img_path: _qimage_to_numpy(QImage(p)), specs))
    return groups
