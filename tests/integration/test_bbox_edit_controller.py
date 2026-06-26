"""Bbox edit ↔ persistence integration tests (bnsreenu #40 / #32).

A canvas bbox resize/move mutates the annotation in place, clamps it to the
image on release, and commit_bbox_edit pushes the new coords into
all_annotations (value-equality preserved) and keeps the box selected.

One real offscreen ImageAnnotator; no model weights, no worker thread.
"""

import copy

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


class _FakeEvent:
    def modifiers(self):
        return Qt.KeyboardModifier.NoModifier


def _bbox(x, y, w, h, number):
    return {"bbox": [x, y, w, h], "category_name": "cell", "number": number}


def _seg(x0, y0, side, number):
    return {
        "segmentation": [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side],
        "category_name": "cell", "number": number,
    }


def _seed(window, anns):
    window.image_file_name = "img.png"
    window.current_slice = None
    window.all_annotations = {"img.png": {"cell": list(anns)}}
    window.image_label.annotations = copy.deepcopy(window.all_annotations["img.png"])
    window.update_annotation_list()
    il = window.image_label
    il.original_pixmap = QPixmap(100, 100)
    il.zoom_factor = 1.0
    il.ui_scale = 1.0
    return il.annotations["cell"]      # the live copies the canvas mutates


def _selected_data(window):
    # The annotations widget is now a QTableWidget; selection is per-row, with
    # the annotation dict in column 0's UserRole. Dedupe selected cells to rows.
    tbl = window.annotation_list
    rows = sorted({idx.row() for idx in tbl.selectedIndexes()})
    return [tbl.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]


def test_resize_persists_into_all_annotations(window, monkeypatch):
    monkeypatch.setattr(window, "auto_save", lambda: None)
    (live,) = _seed(window, [_bbox(10, 10, 40, 40, 1)])
    il = window.image_label
    window.annotation_controller.apply_canvas_selection([live], "replace")

    # Drag the bottom-right handle out to (80, 70) — fully in bounds.
    il._begin_shape_edit(live, "resize", "br", (50, 50))
    il._update_bbox_drag((80, 70))
    il._commit_bbox_drag((80, 70), _FakeEvent())  # emits bboxEditCommitted → commit

    assert live["bbox"] == [10, 10, 70, 60]
    # all_annotations now holds the mutated box (value-equality), and the list
    # rebuilt + re-selected it.
    assert window.all_annotations["img.png"]["cell"][0]["bbox"] == [10, 10, 70, 60]
    assert _selected_data(window) == [live]


def test_list_selected_bbox_resize_persists(window, monkeypatch):
    """Selecting via the annotation LIST (not the canvas) puts a value-equal
    copy in highlighted_annotations; the handle drag must still mutate the live
    object so the edit is saved, not silently lost."""
    monkeypatch.setattr(window, "auto_save", lambda: None)
    (live,) = _seed(window, [_bbox(10, 10, 40, 40, 1)])
    il = window.image_label

    # Select through the list widget — drives update_highlighted_annotations,
    # which stores item.data(UserRole) (a copy, distinct from `live`).
    window.annotation_list.selectRow(0)
    window.annotation_controller.update_highlighted_annotations()
    assert il.highlighted_annotations and il.highlighted_annotations[0] is not live

    shape = il._single_selected_shape()            # the list copy (geometry)
    live_obj = il._live_annotation(shape)          # press handler resolves to live
    assert live_obj is live
    il._begin_shape_edit(live_obj, "resize", "br", (50, 50))
    il._update_bbox_drag((80, 70))
    il._commit_bbox_drag((80, 70), _FakeEvent())

    assert window.all_annotations["img.png"]["cell"][0]["bbox"] == [10, 10, 70, 60]
    assert _selected_data(window) == [live]        # still selected after rebuild


def test_resize_out_of_bounds_is_clamped_on_commit(window, monkeypatch):
    monkeypatch.setattr(window, "auto_save", lambda: None)
    (live,) = _seed(window, [_bbox(10, 10, 40, 40, 1)])
    il = window.image_label
    window.annotation_controller.apply_canvas_selection([live], "replace")

    il._begin_shape_edit(live, "resize", "br", (50, 50))
    il._update_bbox_drag((300, 300))               # way past the image edge
    il._commit_bbox_drag((300, 300), _FakeEvent())

    x, y, w, h = window.all_annotations["img.png"]["cell"][0]["bbox"]
    assert x >= 0 and y >= 0
    assert x + w <= 100 and y + h <= 100           # clamped inside the image


def test_polygon_resize_persists_into_all_annotations(window, monkeypatch):
    """The common case: a segmentation mask (no bbox key) resized via its
    handles scales the vertices and persists — the path the bbox-only feature
    missed."""
    monkeypatch.setattr(window, "auto_save", lambda: None)
    (live,) = _seed(window, [_seg(20, 20, 40, 1)])   # square 20..60
    il = window.image_label
    window.annotation_controller.apply_canvas_selection([live], "replace")

    il._begin_shape_edit(live, "resize", "br", (60, 60))
    il._update_bbox_drag((100, 100))                 # scale the square 2x
    il._commit_bbox_drag((100, 100), _FakeEvent())

    saved = window.all_annotations["img.png"]["cell"][0]["segmentation"]
    assert saved == [20, 20, 100, 20, 100, 100, 20, 100]
    assert _selected_data(window) == [live]
