"""
Unit tests for idle-mode canvas mask selection (bnsreenu issue #75).

Covers the pure hit-testing helpers on ImageLabel (annotation_at /
annotations_in_rect) and the press→release gesture resolution
(_finish_selection) that emits canvasSelectionChanged. No main window,
no model — just the widget.
"""

import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap

from src.digitalsreeni_image_annotator.widgets.image_label import ImageLabel


@pytest.fixture
def label(qtbot):
    lbl = ImageLabel(None)
    qtbot.addWidget(lbl)
    return lbl


def _square(x0, y0, side, name):
    return {
        "segmentation": [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side],
        "category_name": name,
    }


def _bbox(x, y, w, h, name):
    return {"bbox": [x, y, w, h], "category_name": name}


class _FakeEvent:
    """Minimal stand-in for QMouseEvent — only modifiers() is read."""

    def __init__(self, shift=False):
        self._shift = shift

    def modifiers(self):
        if self._shift:
            return Qt.KeyboardModifier.ShiftModifier
        return Qt.KeyboardModifier.NoModifier


class _FakeCtx:
    def __init__(self, hidden=()):
        self._hidden = set(hidden)

    def is_class_visible(self, name):
        return name not in self._hidden


# --- annotation_at ---------------------------------------------------------

def test_annotation_at_smallest_of_nested(label):
    outer = _square(0, 0, 100, "outer")     # area 10000
    inner = _square(40, 40, 20, "inner")    # area 400, fully inside outer
    label.annotations = {"cell": [outer, inner]}
    assert label.annotation_at((50, 50)) is inner   # inside both → smallest
    assert label.annotation_at((10, 10)) is outer   # inside outer only
    assert label.annotation_at((500, 500)) is None  # empty space


def test_annotation_at_hits_bbox(label):
    box = _bbox(0, 0, 100, 100, "box")
    label.annotations = {"cell": [box]}
    assert label.annotation_at((50, 50)) is box
    assert label.annotation_at((150, 150)) is None


def test_annotation_at_smallest_across_seg_and_bbox(label):
    big_box = _bbox(0, 0, 200, 200, "box")   # area 40000
    small_seg = _square(40, 40, 20, "seg")   # area 400
    label.annotations = {"cell": [big_box, small_seg]}
    assert label.annotation_at((45, 45)) is small_seg


def test_annotation_at_skips_hidden_class(label):
    visible = _square(0, 0, 100, "visible")
    hidden = _square(40, 40, 20, "hidden")   # smaller, but its class is hidden
    label.annotations = {"visible": [visible], "hidden": [hidden]}
    label.set_context(_FakeCtx(hidden={"hidden"}))
    # Without the visibility guard, the smaller hidden square would win.
    assert label.annotation_at((50, 50)) is visible


# --- annotations_in_rect ---------------------------------------------------

def test_annotations_in_rect_returns_intersecting(label):
    a = _square(0, 0, 20, "a")
    b = _square(50, 50, 20, "b")
    far = _square(500, 500, 20, "far")
    bbox_in = _bbox(10, 10, 30, 30, "c")
    label.annotations = {"cell": [a, b, far, bbox_in]}
    hit = label.annotations_in_rect((0, 0, 80, 80))
    assert a in hit and b in hit and bbox_in in hit
    assert far not in hit


def test_annotations_in_rect_any_corner_order(label):
    a = _square(0, 0, 20, "a")
    label.annotations = {"cell": [a]}
    # Rect given bottom-right → top-left must still match.
    assert a in label.annotations_in_rect((80, 80, 0, 0))


def test_annotations_in_rect_skips_hidden(label):
    a = _square(0, 0, 20, "a")
    label.annotations = {"hidden": [a]}
    label.set_context(_FakeCtx(hidden={"hidden"}))
    assert label.annotations_in_rect((0, 0, 80, 80)) == []


# --- _finish_selection gesture resolution ----------------------------------

@pytest.fixture
def captured(label):
    events = []
    label.canvasSelectionChanged.connect(lambda anns, mode: events.append((anns, mode)))
    return events


def test_click_on_mask_emits_replace(label, captured):
    inner = _square(40, 40, 20, "inner")
    label.annotations = {"cell": [inner]}
    label.selection_origin = (45, 45)
    label.selecting = False
    label._finish_selection((45, 45), _FakeEvent(shift=False))
    assert captured == [([inner], "replace")]
    # gesture state reset
    assert label.selection_origin is None and label.selection_rect is None


def test_click_empty_emits_replace_empty(label, captured):
    label.annotations = {"cell": [_square(40, 40, 20, "inner")]}
    label.selection_origin = (300, 300)
    label._finish_selection((300, 300), _FakeEvent(shift=False))
    assert captured == [([], "replace")]


def test_shift_click_mask_emits_toggle(label, captured):
    inner = _square(40, 40, 20, "inner")
    label.annotations = {"cell": [inner]}
    label.selection_origin = (45, 45)
    label._finish_selection((45, 45), _FakeEvent(shift=True))
    assert captured == [([inner], "toggle")]


def test_shift_click_empty_emits_nothing(label, captured):
    label.annotations = {"cell": [_square(40, 40, 20, "inner")]}
    label.selection_origin = (300, 300)
    label._finish_selection((300, 300), _FakeEvent(shift=True))
    assert captured == []


def test_drag_emits_add_when_shift(label, captured):
    a = _square(0, 0, 20, "a")
    b = _square(50, 50, 20, "b")
    label.annotations = {"cell": [a, b]}
    label.selection_origin = (0, 0)
    label.selecting = True
    label.selection_rect = (0, 0, 80, 80)
    label._finish_selection((80, 80), _FakeEvent(shift=True))
    anns, mode = captured[-1]
    assert mode == "add"
    assert a in anns and b in anns


def test_drag_emits_replace_without_shift(label, captured):
    a = _square(0, 0, 20, "a")
    label.annotations = {"cell": [a]}
    label.selection_origin = (0, 0)
    label.selecting = True
    label.selection_rect = (0, 0, 80, 80)
    label._finish_selection((80, 80), _FakeEvent(shift=False))
    assert captured[-1][1] == "replace"
    assert a in captured[-1][0]


def test_escape_cancels_rubber_band(label):
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QKeyEvent

    label.current_tool = None
    label.selection_origin = (0, 0)
    label.selecting = True
    label.selection_rect = (0, 0, 50, 50)
    label.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    )
    assert label.selection_rect is None
    assert label.selecting is False
    assert label.selection_origin is None


# --- selection rendering: overlay, not recolor ----------------------------

def _setup_canvas(label, class_color):
    label.set_context(_FakeCtx())
    px = QPixmap(100, 100)
    px.fill(QColor("white"))
    label.original_pixmap = px
    label.class_colors = {"cell": QColor(class_color)}


def _render_center(label):
    img = QImage(100, 100, QImage.Format.Format_RGB32)
    img.fill(QColor("white"))
    p = QPainter(img)
    label.draw_annotations(p)
    p.end()
    return img.pixelColor(50, 50)


def test_selection_does_not_recolor_fill(label):
    # A mask's interior fill must look identical selected vs. unselected — the
    # old code turned it red, which was invisible on a red-class mask.
    _setup_canvas(label, "#1F77B4")
    mask = {"segmentation": [20, 20, 80, 20, 80, 80, 20, 80], "category_name": "cell"}
    label.annotations = {"cell": [mask]}

    label.highlighted_annotations = []
    unselected = _render_center(label)
    label.highlighted_annotations = [mask]
    selected = _render_center(label)
    assert selected == unselected  # selection adds an overlay, never recolors


def test_selection_overlay_runs_for_seg_and_bbox(label):
    # Red class = the worst case; outline + marquee must still render fine.
    _setup_canvas(label, "#D62728")
    seg = {"segmentation": [10, 10, 40, 10, 40, 40, 10, 40], "category_name": "cell"}
    box = {"bbox": [50, 50, 30, 30], "category_name": "cell"}
    label.annotations = {"cell": [seg, box]}
    label.highlighted_annotations = [seg, box]

    img = QImage(100, 100, QImage.Format.Format_RGB32)
    img.fill(QColor("white"))
    p = QPainter(img)
    label.draw_annotations(p)  # must not raise
    p.end()
