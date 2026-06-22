"""Annotations table + reversible polygon simplification (issue #24).

The bottom-left Annotations panel is a QTableWidget (ID | Class | Area |
Detail %). Each row's Detail % spinbox re-simplifies that annotation's polygon
from a lazily-captured raw copy; 100 % restores raw exactly. The simplified
polygon persists (.iap) and is what exports emit; the raw is kept for reverting.

One real offscreen ImageAnnotator; no model weights.
"""

import copy
import json
import math

import pytest
from PyQt6.QtWidgets import QSpinBox

from digitalsreeni_image_annotator.core import image_utils
from digitalsreeni_image_annotator.core.constants import (
    ANNOT_COL_AREA,
    ANNOT_COL_CLASS,
    ANNOT_COL_DETAIL,
    ANNOT_COL_ID,
)
from digitalsreeni_image_annotator.utils import calculate_area


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def _circle(cx, cy, r, n):
    seg = []
    for i in range(n):
        a = 2 * math.pi * i / n
        seg += [round(cx + r * math.cos(a)), round(cy + r * math.sin(a))]
    return seg


def _seed(window, anns, monkeypatch):
    monkeypatch.setattr(window, "auto_save", lambda: None)
    window.image_file_name = "img.png"
    window.current_slice = None
    window.class_mapping = {"cell": 1}
    window.all_annotations = {"img.png": {"cell": copy.deepcopy(anns)}}
    window.image_label.annotations = copy.deepcopy(window.all_annotations["img.png"])
    window.update_annotation_list()
    return window.image_label.annotations["cell"]


def _dense(number=1):
    return {"segmentation": _circle(50, 50, 30, 60), "category_name": "cell",
            "category_id": 1, "number": number}


def test_table_populates_columns(window, monkeypatch):
    _seed(window, [_dense(1)], monkeypatch)
    tbl = window.annotation_list
    assert tbl.rowCount() == 1
    assert tbl.item(0, ANNOT_COL_ID).text() == "1"
    assert tbl.item(0, ANNOT_COL_CLASS).text() == "cell"
    float(tbl.item(0, ANNOT_COL_AREA).text())            # area is numeric
    spin = tbl.cellWidget(0, ANNOT_COL_DETAIL)
    assert isinstance(spin, QSpinBox) and spin.value() == 100


def test_detail_change_simplifies_live_and_refreshes_area(window, monkeypatch):
    (live,) = _seed(window, [_dense(1)], monkeypatch)
    tbl = window.annotation_list
    raw_len = len(live["segmentation"])

    tbl.cellWidget(0, ANNOT_COL_DETAIL).setValue(30)      # fires the handler

    assert len(live["segmentation"]) < raw_len            # thinned
    assert live["segmentation_raw"] and len(live["segmentation_raw"]) == raw_len
    assert live["detail_pct"] == 30
    # Area cell tracks the (now simplified) polygon.
    assert tbl.item(0, ANNOT_COL_AREA).text() == f"{calculate_area(live):.2f}"


def test_back_to_100_restores_raw_exactly(window, monkeypatch):
    (live,) = _seed(window, [_dense(1)], monkeypatch)
    spin = window.annotation_list.cellWidget(0, ANNOT_COL_DETAIL)
    spin.setValue(25)
    assert len(live["segmentation"]) < len(live["segmentation_raw"])

    spin.setValue(100)
    assert live["segmentation"] == live["segmentation_raw"]  # raw restored exactly
    assert live["detail_pct"] == 100


def test_simplified_persists_and_exports_simplified(window, monkeypatch):
    (live,) = _seed(window, [_dense(1)], monkeypatch)
    window.annotation_list.cellWidget(0, ANNOT_COL_DETAIL).setValue(40)

    # Persistence: project save does ann.copy() -> convert_to_serializable ->
    # json. Both new keys must survive the round-trip.
    roundtripped = json.loads(json.dumps(image_utils.convert_to_serializable(live)))
    assert roundtripped["detail_pct"] == 40
    assert roundtripped["segmentation_raw"] == live["segmentation_raw"]

    # Export emits the *effective* (simplified) polygon, not the raw.
    coco = window.annotation_controller.create_coco_annotation(live, 1, 1)
    assert coco["segmentation"] == [live["segmentation"]]
    assert coco["segmentation"][0] != live["segmentation_raw"]


def test_bbox_only_row_has_disabled_spinbox(window, monkeypatch):
    _seed(window, [{"bbox": [10, 10, 40, 40], "segmentation": None,
                    "category_name": "cell", "category_id": 1, "number": 1}],
          monkeypatch)
    spin = window.annotation_list.cellWidget(0, ANNOT_COL_DETAIL)
    assert isinstance(spin, QSpinBox) and not spin.isEnabled()
