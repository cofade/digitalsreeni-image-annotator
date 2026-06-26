"""Canvas-selection ↔ annotation-list integration tests (bnsreenu #75).

apply_canvas_selection must update image_label.highlighted_annotations,
mirror that onto the annotation list selection (so Delete / Merge /
Change-Class operate on the same set), and toggle the merge / change-class
buttons. The canvas Delete path then reuses delete_selected_annotations.

One real offscreen ImageAnnotator; no model weights, no worker thread.
"""

import copy

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def _square(x0, y0, side, number):
    return {
        "segmentation": [x0, y0, x0 + side, y0, x0 + side, y0 + side, x0, y0 + side],
        "category_name": "cell",
        "number": number,
    }


def _seed(window, anns):
    window.image_file_name = "img.png"
    window.current_slice = None
    window.all_annotations = {"img.png": {"cell": list(anns)}}
    window.image_label.annotations = copy.deepcopy(window.all_annotations["img.png"])
    window.update_annotation_list()


def _selected_data(window):
    # The annotations widget is now a QTableWidget; selection is per-row, with
    # the annotation dict in column 0's UserRole. Dedupe selected cells to rows.
    tbl = window.annotation_list
    rows = sorted({idx.row() for idx in tbl.selectedIndexes()})
    return [tbl.item(r, 0).data(Qt.ItemDataRole.UserRole) for r in rows]


def test_replace_selects_one(window):
    a1, a2, a3 = _square(0, 0, 10, 1), _square(50, 0, 10, 2), _square(100, 0, 10, 3)
    _seed(window, [a1, a2, a3])

    window.annotation_controller.apply_canvas_selection([a1], "replace")

    assert window.image_label.highlighted_annotations == [a1]
    assert _selected_data(window) == [a1]
    assert not window.merge_button.isEnabled()       # needs ≥2
    assert window.change_class_button.isEnabled()    # needs ≥1


def test_add_then_toggle(window):
    a1, a2, a3 = _square(0, 0, 10, 1), _square(50, 0, 10, 2), _square(100, 0, 10, 3)
    _seed(window, [a1, a2, a3])
    ac = window.annotation_controller

    ac.apply_canvas_selection([a1], "replace")
    ac.apply_canvas_selection([a2], "add")
    assert window.image_label.highlighted_annotations == [a1, a2]
    # The list mirrors by value-equality (PyQt round-trips UserRole dicts as
    # copies), so compare by membership, not identity.
    sel = _selected_data(window)
    assert len(sel) == 2 and a1 in sel and a2 in sel
    assert window.merge_button.isEnabled()

    # Toggling a1 off leaves only a2.
    ac.apply_canvas_selection([a1], "toggle")
    assert window.image_label.highlighted_annotations == [a2]
    assert _selected_data(window) == [a2]
    assert not window.merge_button.isEnabled()


def test_replace_empty_clears(window):
    a1, a2 = _square(0, 0, 10, 1), _square(50, 0, 10, 2)
    _seed(window, [a1, a2])
    ac = window.annotation_controller

    ac.apply_canvas_selection([a1, a2], "replace")
    assert window.merge_button.isEnabled()

    ac.apply_canvas_selection([], "replace")
    assert window.image_label.highlighted_annotations == []
    assert _selected_data(window) == []
    assert not window.merge_button.isEnabled()
    assert not window.change_class_button.isEnabled()


def test_canvas_delete_removes_selected_set(window, monkeypatch):
    a1, a2, a3 = _square(0, 0, 10, 1), _square(50, 0, 10, 2), _square(100, 0, 10, 3)
    _seed(window, [a1, a2, a3])

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(window, "auto_save", lambda: None)

    window.annotation_controller.apply_canvas_selection([a1, a2], "replace")
    window.delete_selected_annotations()

    remaining = window.image_label.annotations.get("cell", [])
    assert a3 in remaining
    assert a1 not in remaining and a2 not in remaining
    assert window.all_annotations["img.png"]["cell"] == remaining
    assert window.image_label.highlighted_annotations == []


def test_canvas_delete_gated_on_list_selection(window, monkeypatch):
    """Canvas Delete fires only when the annotation list actually has a
    selection — not when only the red highlight is (stale) populated, e.g.
    after a sort rebuilds the list. Otherwise it pops a spurious warning."""
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QKeyEvent

    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(window, "auto_save", lambda: None)

    a1, a2, a3 = _square(0, 0, 10, 1), _square(50, 0, 10, 2), _square(100, 0, 10, 3)
    _seed(window, [a1, a2, a3])
    il = window.image_label
    il.current_tool = None

    def press_delete():
        il.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Delete,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    emitted = []
    il.deleteSelectionRequested.connect(lambda: emitted.append(True))

    # In sync (canvas selection mirrored to the list) → Delete fires.
    window.annotation_controller.apply_canvas_selection([a1], "replace")
    assert il._ctx.has_annotation_selection()
    press_delete()
    assert emitted == [True]
    assert a1 not in il.annotations.get("cell", [])

    # Construct the divergence the gate guards against: a stale red highlight
    # with no list selection. The canvas Delete keys off the list (the
    # controller's source of truth), so it must NOT fire here — no spurious
    # "nothing selected" warning.
    emitted.clear()
    il.highlighted_annotations = [a2]
    window.annotation_list.clearSelection()
    assert il.highlighted_annotations                # highlight is stale
    assert not il._ctx.has_annotation_selection()    # but the list isn't selected
    press_delete()
    assert emitted == []
