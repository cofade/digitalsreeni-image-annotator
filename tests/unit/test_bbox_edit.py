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


def test_single_selected_shape(label):
    box = _bbox(10, 10, 80, 80)
    seg = _seg(0, 0, 20)
    label.highlighted_annotations = [box]
    assert label._single_selected_shape() is box
    label.highlighted_annotations = [seg]          # polygons are editable too
    assert label._single_selected_shape() is seg
    label.highlighted_annotations = [box, seg]     # multi-select
    assert label._single_selected_shape() is None


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

def _arm(label, ann, mode, handle, start):
    label.original_pixmap = QPixmap(100, 100)
    label.annotations = {ann["category_name"]: [ann]}
    label.highlighted_annotations = [ann]
    label._begin_shape_edit(ann, mode, handle, start)


def test_pending_move_promotes_only_after_threshold(label):
    box = _bbox(20, 20, 40, 40)
    _arm(label, box, "pending_move", None, (40, 40))
    label._update_bbox_drag((41, 41))           # within click threshold
    assert label.bbox_edit["mode"] == "pending_move"
    assert box["bbox"] == [20, 20, 40, 40]
    label._update_bbox_drag((60, 50))           # clears threshold -> move
    assert label.bbox_edit["mode"] == "move"
    assert box["bbox"] == [40, 30, 40, 40]       # translated by (+20, +10)


def test_commit_move_slides_inside_preserving_size(label, qtbot):
    box = _bbox(80, 80, 30, 30)
    _arm(label, box, "move", None, (90, 90))
    label.bbox_edit["mode"] = "move"
    label._update_bbox_drag((130, 130))          # push far past the bottom-right
    assert box["bbox"][0] + box["bbox"][2] > 100  # out of bounds before commit

    with qtbot.waitSignal(label.bboxEditCommitted, timeout=500):
        label._commit_bbox_drag((130, 130), _FakeEvent())
    assert box["bbox"] == [70, 70, 30, 30]       # slid inside, size preserved
    assert label.bbox_edit is None


def test_commit_move_off_top_left_preserves_size(label, qtbot):
    # The clamp_bbox-collapse bug: moving past the top-left must keep 40x40.
    box = _bbox(20, 20, 40, 40)
    _arm(label, box, "move", None, (40, 40))
    label.bbox_edit["mode"] = "move"
    label._update_bbox_drag((-30, -30))          # drag past the top-left corner
    with qtbot.waitSignal(label.bboxEditCommitted, timeout=500):
        label._commit_bbox_drag((-30, -30), _FakeEvent())
    assert box["bbox"] == [0, 0, 40, 40]         # not collapsed to a sliver


def test_live_annotation_resolves_to_object_not_highlighted_copy(label):
    # List-driven selection puts a value-equal COPY in highlighted_annotations
    # (item.data(UserRole) round-trips as a copy). The handle drag must mutate
    # the live object inside label.annotations, or the edit is lost — so the
    # press handler resolves the selection via _live_annotation before arming.
    box = _bbox(10, 10, 40, 40)
    copy = dict(box)
    label.annotations = {"cell": [box]}
    label.highlighted_annotations = [copy]        # a copy, different identity
    assert label._single_selected_shape() is copy  # geometry entry (the copy)
    assert label._live_annotation(copy) is box     # resolves to the live object


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


# --- polygon (segmentation) transform: resize scales, move translates --------

def test_resize_scales_polygon(label):
    seg = _seg(20, 20, 40)                         # square 20..60
    _arm(label, seg, "resize", "br", (60, 60))
    label._update_bbox_drag((100, 100))            # drag br to (100,100)
    # br anchored at the opposite (top-left) corner; the square scales 2x.
    assert seg["segmentation"] == [20, 20, 100, 20, 100, 100, 20, 100]


def test_move_translates_polygon(label):
    seg = _seg(20, 20, 40)
    _arm(label, seg, "move", None, (40, 40))
    label.bbox_edit["mode"] = "move"
    label._update_bbox_drag((50, 55))              # dx=10, dy=15
    assert seg["segmentation"] == [30, 35, 70, 35, 70, 75, 30, 75]


def test_commit_polygon_move_off_edge_slides_inside(label, qtbot):
    seg = _seg(70, 70, 20)                          # 70..90
    _arm(label, seg, "move", None, (80, 80))
    label.bbox_edit["mode"] = "move"
    label._update_bbox_drag((120, 120))            # push fully past bottom-right
    with qtbot.waitSignal(label.bboxEditCommitted, timeout=500):
        label._commit_bbox_drag((120, 120), _FakeEvent())
    xs, ys = seg["segmentation"][0::2], seg["segmentation"][1::2]
    assert min(xs) >= 0 and min(ys) >= 0 and max(xs) <= 100 and max(ys) <= 100
    assert max(xs) - min(xs) == 20 and max(ys) - min(ys) == 20  # shape preserved


def test_resize_polygon_syncs_bbox_key(label):
    # An imported annotation carries both segmentation and bbox; editing the
    # polygon must keep the bbox key consistent (it feeds export/training).
    seg = {
        "segmentation": [20, 20, 60, 20, 60, 60, 20, 60],
        "bbox": [20, 20, 40, 40], "category_name": "cell",
    }
    label.original_pixmap = QPixmap(100, 100)
    label.annotations = {"cell": [seg]}
    label.highlighted_annotations = [seg]
    label._begin_shape_edit(seg, "resize", "br", (60, 60))
    label._update_bbox_drag((100, 100))
    assert seg["bbox"] == [20, 20, 80, 80]         # recomputed from scaled verts


def test_escape_restores_polygon(label):
    seg = _seg(20, 20, 40)
    orig = list(seg["segmentation"])
    _arm(label, seg, "move", None, (40, 40))
    label.bbox_edit["mode"] = "move"
    label._update_bbox_drag((80, 80))
    assert seg["segmentation"] != orig
    label.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    )
    assert label.bbox_edit is None
    assert seg["segmentation"] == orig


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
