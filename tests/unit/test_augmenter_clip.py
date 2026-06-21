"""Augmenter bounds-clip integration at the method level (issue #36).

ImageAugmenterDialog.augment_annotation must clip a transformed polygon to the
(augmented) image rectangle and drop one that ends up fully outside, so the
exported COCO JSON never carries out-of-image coordinates.
"""

import pytest

from src.digitalsreeni_image_annotator.dialogs.image_augmenter import (
    ImageAugmenterDialog,
)


@pytest.fixture
def dialog(qtbot):
    dlg = ImageAugmenterDialog(None)
    qtbot.addWidget(dlg)
    return dlg


def _flip(flip_code=1):
    # Horizontal flip mirrors x -> width - x; no matrix needed by flip_polygon.
    return {"type": "flip", "flip_code": flip_code}


def test_augment_clips_partially_out_of_bounds(dialog):
    # A 50..150 square horizontally flipped in a 100x100 image lands at -50..50,
    # i.e. partly outside; the clip must trim it back inside [0, 100].
    ann = {"segmentation": [[50, 50, 150, 50, 150, 150, 50, 150]], "category_name": "cell"}
    out = dialog.augment_annotation(ann, _flip(), (100, 100))
    assert out is not None
    xs = out["segmentation"][0][0::2]
    ys = out["segmentation"][0][1::2]
    assert min(xs) >= 0 and max(xs) <= 100
    assert min(ys) >= 0 and max(ys) <= 100
    x, y, w, h = out["bbox"]
    assert x + w <= 100 and y + h <= 100           # recomputed bbox stays inside


def test_augment_drops_fully_out_of_bounds(dialog):
    # A 200..300 square flipped horizontally in a 100x100 image lands fully in
    # negative x → nothing remains inside → annotation dropped (None).
    ann = {"segmentation": [[200, 10, 300, 10, 300, 90, 200, 90]], "category_name": "cell"}
    assert dialog.augment_annotation(ann, _flip(), (100, 100)) is None
