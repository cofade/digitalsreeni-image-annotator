"""Unit tests for direct-manipulation bbox editing on ImageLabel (issue #40).

Exercises the pure helpers (handle hit-test, resize math) and the drag
lifecycle (_update_bbox_drag / _commit_bbox_drag / _cancel_bbox_drag) that
mutates a selected bbox in place, clamps it on commit, and emits
bboxEditCommitted. No main window, no model — just the widget.
"""

import pytest

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent, QPixmap

from src.digitalsreeni_image_annotator.widgets.image_label import ImageLabel


@pytest.fixture
def label(qtbot):
    lbl = ImageLabel(None)
    qtbot.addWidget(lbl)
    lbl.zoom_factor = 1.0
    lbl.ui_scale = 1.0
    return lbl


def _bbox(x, y, w, h, name="cell"):
    return {"bbox": [x, y, w, h], "category_name": name}


def _seg(x0, y0, side, name="cell"):
    return {
        "segmentation": [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side],
        "category_name": name,
    }


class _FakeEvent:
    """Minimal QMouseEvent stand-in — only modifiers() is read by the helpers."""

    def __init__(self, shift=False):
        self._shift = shift

    def modifiers(self):
        if self._shift:
            return Qt.KeyboardModifier.ShiftModifier
        return Qt.KeyboardModifier.NoModifier


# --- handle geometry / hit-testing -----------------------------------------

def test_handle_points_has_eight_positions():
    pts = ImageLabel._bbox_handle_points((10, 10, 90, 90))
    assert set(pts) == {"tl", "tm", "tr", "ml", "mr", "bl", "bm", "br"}
    assert pts["tl"] == (10, 10)
    assert pts["br"] == (90, 90)
    assert pts["tm"] == (50, 10)
    assert pts["mr"] == (90, 50)


def test_bbox_handle_at_corners_and_edges(label):
    box = _bbox(10, 10, 80, 80)            # bb = (10, 10, 90, 90)
    assert label._bbox_handle_at(box, (10, 10)) == "tl"
    assert label._bbox_handle_at(box, (90, 90)) == "br"
    assert label._bbox_handle_at(box, (50, 10)) == "tm"
    assert label._bbox_handle_at(box, (10, 50)) == "ml"
    assert label._bbox_handle_at(box, (50, 50)) is None   # interior, no handle
    assert label._bbox_handle_at(box, (300, 300)) is None  # far away


def test_single_selected_bbox(label):
    box = _bbox(10, 10, 80, 80)
    seg = _seg(0, 0, 20)
    label.highlighted_annotations = [box]
    assert label._single_selected_bbox() is box
    label.highlighted_annotations = [seg]          # segmentation, not a bbox
    assert label._single_selected_bbox() is None
    label.highlighted_annotations = [box, _bbox(0, 0, 5, 5)]  # multi-select
    assert label._single_selected_bbox() is None


# --- resize math -----------------------------------------------------------

def test_resize_corner_grows_from_opposite():
    # bb = (10,10,90,90); drag br to (120,120) -> tl anchored.
    assert ImageLabel._resize_bbox([10, 10, 80, 80], "br", (120, 120)) == [10, 10, 110, 110]


def test_resize_corner_tl_moves_origin():
    assert ImageLabel._resize_bbox([10, 10, 80, 80], "tl", (0, 0)) == [0, 0, 90, 90]


def test_resize_edge_changes_one_dimension():
    # right-middle handle moves only the right edge.
    assert ImageLabel._resize_bbox([10, 10, 80, 80], "mr", (120, 999)) == [10, 10, 110, 80]
    # top-middle handle moves only the top edge.
    assert ImageLabel._resize_bbox([10, 10, 80, 80], "tm", (999, 0)) == [10, 0, 80, 90]


def test_resize_past_anchor_normalises_without_negative_size():
    out = ImageLabel._resize_bbox([10, 10, 80, 80], "br", (0, 0))
    x, y, w, h = out
    assert w >= 1 and h >= 1
    assert [x, y, w, h] == [0, 0, 10, 10]


# --- drag lifecycle: move ---------------------------------------------------

def _arm(label, box, mode, handle, start):
    label.original_pixmap = QPixmap(100, 100)
    label.annotations = {box["category_name"]: [box]}
    label.highlighted_annotations = [box]
    label.bbox_edit = {
        "annotation": box, "mode": mode, "handle": handle,
        "orig_bbox": list(box["bbox"]), "start_pos": start, "moved": False,
    }


def test_pending_move_promotes_only_after_threshold(label):
    box = _bbox(20, 20, 40, 40)
    _arm(label, box, "pending_move", None, (40, 40))
    label._update_bbox_drag((41, 41))           # within click threshold
    assert label.bbox_edit["mode"] == "pending_move"
    assert box["bbox"] == [20, 20, 40, 40]
    label._update_bbox_drag((60, 50))           # clears threshold -> move
    assert label.bbox_edit["mode"] == "move"
    assert box["bbox"] == [40, 30, 40, 40]       # translated by (+20, +10)


def test_commit_move_clamps_and_emits(label, qtbot):
    box = _bbox(80, 80, 30, 30)
    _arm(label, box, "move", None, (90, 90))
    label.bbox_edit["mode"] = "move"
    label._update_bbox_drag((130, 130))          # push far past the bottom-right
    assert box["bbox"][0] + box["bbox"][2] > 100  # out of bounds before commit

    with qtbot.waitSignal(label.bboxEditCommitted, timeout=500):
        label._commit_bbox_drag((130, 130), _FakeEvent())
    x, y, w, h = box["bbox"]
    assert x + w <= 100 and y + h <= 100         # clamped into the image
    assert label.bbox_edit is None


def test_commit_resize_clamps(label, qtbot):
    box = _bbox(10, 10, 80, 80)
    _arm(label, box, "resize", "br", (90, 90))
    label._update_bbox_drag((200, 200))          # drag corner way outside
    with qtbot.waitSignal(label.bboxEditCommitted, timeout=500):
        label._commit_bbox_drag((200, 200), _FakeEvent())
    x, y, w, h = box["bbox"]
    assert x + w <= 100 and y + h <= 100


def test_commit_without_drag_falls_through_to_selection(label):
    box = _bbox(20, 20, 40, 40)
    _arm(label, box, "pending_move", None, (40, 40))
    captured = []
    label.canvasSelectionChanged.connect(lambda anns, mode: captured.append((anns, mode)))
    # No _update_bbox_drag call -> moved stays False -> commit behaves as a click.
    label._commit_bbox_drag((40, 40), _FakeEvent())
    assert captured == [([box], "replace")]
    assert label.bbox_edit is None


def test_hover_cursor_reflects_handle_interior_and_outside(label):
    box = _bbox(10, 10, 80, 80)
    label.annotations = {"cell": [box]}
    label.highlighted_annotations = [box]
    label._update_select_cursor((10, 10))        # top-left corner handle
    assert label.cursor().shape() == Qt.CursorShape.SizeFDiagCursor
    label._update_select_cursor((50, 50))        # interior
    assert label.cursor().shape() == Qt.CursorShape.SizeAllCursor
    label._update_select_cursor((300, 300))      # empty space
    assert label.cursor().shape() == Qt.CursorShape.ArrowCursor


def test_escape_cancels_and_restores(label):
    box = _bbox(20, 20, 40, 40)
    _arm(label, box, "move", None, (40, 40))
    label.bbox_edit["mode"] = "move"
    label._update_bbox_drag((80, 80))            # box now moved
    assert box["bbox"] != [20, 20, 40, 40]
    label.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    )
    assert label.bbox_edit is None
    assert box["bbox"] == [20, 20, 40, 40]       # restored to original
