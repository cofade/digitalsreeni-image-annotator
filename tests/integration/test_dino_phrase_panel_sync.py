"""Regression test for issue #63 -- the DINO phrase panel must follow the
top class-list selection, not only the threshold-table row.

The phrase panel ("Phrases for: X") was bound only to the DINO threshold
table's ``itemSelectionChanged``. Selecting a different class in the *top*
class list retargeted the annotation tools' ``current_class`` but left the
phrase editor pointing at the previously-selected class, so you couldn't
add/edit phrases for the top-list class. The fix makes the top class list
the single source of truth: ``ClassController.on_class_selected`` selects
the matching threshold-table row, which cascades to the phrase panel via
the existing signal.

One real offscreen ImageAnnotator so the actual sidebar signal wiring
(built in ui.sidebar.build_sidebar) is exercised end-to-end.
"""

import pytest
from PyQt6.QtCore import Qt


@pytest.fixture
def window(qt_application, monkeypatch):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    # add_class / on_class_selected call auto_save; on an unsaved project that
    # pops a modal. Stub it out so the test never blocks on a dialog.
    monkeypatch.setattr(w, "auto_save", lambda: None)
    yield w
    w.deleteLater()


def _select_in_top_list(window, name):
    """Drive the top class-list selection exactly as an itemClicked would."""
    item = window.class_list.findItems(name, Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.on_class_selected(item)


def test_phrase_panel_follows_top_class_list_selection(window):
    for name in ("Drone", "rotor", "camera"):
        window.add_class(name)

    # After the adds, the last-added class drives both selectors (add_class
    # selects the freshly-added threshold row, which cascades to the panel).
    assert window.dino_class_table.selected_class_name() == "camera"
    assert window.dino_phrase_panel._active_class == "camera"

    # Select a DIFFERENT class in the TOP class list -- the #63 path.
    _select_in_top_list(window, "Drone")

    assert window.current_class == "Drone"
    # Both the threshold table and the phrase panel now track the top list.
    assert window.dino_class_table.selected_class_name() == "Drone"
    assert window.dino_phrase_panel._active_class == "Drone"


def test_phrase_panel_add_targets_top_list_class(window):
    """Phrases added after a top-list selection land on that class, not the
    stale threshold-table row (the concrete user-visible symptom in #63)."""
    for name in ("Drone", "camera"):
        window.add_class(name)

    _select_in_top_list(window, "Drone")
    window.dino_phrase_panel._phrases["Drone"].append("quadcopter")

    assert "quadcopter" in window.dino_phrase_panel.get_phrases_for("Drone")
    assert "quadcopter" not in window.dino_phrase_panel.get_phrases_for("camera")


def test_select_class_by_name_is_noop_for_unknown_class(window):
    """Selecting a class absent from the threshold table (e.g. a Temp-* review
    class) leaves the current selection intact rather than clearing it."""
    for name in ("Drone", "camera"):
        window.add_class(name)
    _select_in_top_list(window, "Drone")

    assert window.dino_class_table.select_class_by_name("Temp-camera") is False
    # Selection unchanged by the failed lookup.
    assert window.dino_class_table.selected_class_name() == "Drone"
    assert window.dino_phrase_panel._active_class == "Drone"
