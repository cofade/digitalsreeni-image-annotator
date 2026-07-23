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

One real offscreen ImageAnnotator, driven through the real ``class_list``
``itemClicked`` signal, so the actual sidebar wiring (built in
ui.sidebar.build_sidebar) is exercised end-to-end -- deleting that ``connect``
must fail these tests, not just bypass them.
"""

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QInputDialog


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
    """Click a class in the top list, exactly as the user does.

    Goes through ``class_list.itemClicked`` (the signal ui.sidebar connects to
    ``on_class_selected``) rather than calling the handler directly, so the
    wiring itself is under test.
    """
    item = window.class_list.findItems(name, Qt.MatchFlag.MatchExactly)[0]
    window.class_list.setCurrentItem(item)
    window.class_list.itemClicked.emit(item)


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


def test_phrase_panel_add_targets_top_list_class(window, monkeypatch):
    """A phrase added via the Add Phrase button after a top-list selection
    lands on that class, not the stale threshold-table row -- the concrete
    user-visible symptom in #63.

    Drives the real ``_add_phrase`` (which reads ``_active_class``) with a
    stubbed input dialog; asserting on a hardcoded dict key instead would be a
    tautology that passes even with the fix reverted.
    """
    for name in ("Drone", "camera"):
        window.add_class(name)
    # add_class leaves "camera" active in the panel; now pick "Drone" up top.
    _select_in_top_list(window, "Drone")

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("quadcopter", True))
    window.dino_phrase_panel._add_phrase()

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


def test_rename_class_carries_dino_threshold_row_and_phrases(window, monkeypatch):
    """Renaming a class must retarget its DINO threshold row and phrase list.

    Both registries are keyed by class name, so a rename that skips them leaves
    detection running under a dead class name and makes the next project load
    silently discard that class's phrases and thresholds. Found while asserting
    the #63 "top class list is the single source of truth" invariant.
    """
    window.add_class("Drone")
    window.add_class("camera")
    _select_in_top_list(window, "Drone")

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("rotor blade", True))
    window.dino_phrase_panel._add_phrase()
    window.dino_class_table.set_thresholds("Drone", 0.4, 0.35, 0.6)

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("UAV", True))
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.rename_class(item)

    assert "UAV" in window.dino_class_table.get_class_names()
    assert "Drone" not in window.dino_class_table.get_class_names()
    # Thresholds ride along with the row rather than resetting to defaults.
    assert window.dino_class_table.get_thresholds_dict()["UAV"]["box"] == pytest.approx(0.4)
    # Phrases re-key; the custom phrase survives and the class-name phrase
    # (row 0, untouched by the user) follows the rename.
    phrases = window.dino_phrase_panel.get_phrases_for("UAV")
    assert phrases[0] == "UAV"
    assert "rotor blade" in phrases
    assert window.dino_phrase_panel._active_class == "UAV"


def test_rename_class_keeps_user_customised_first_phrase(window, monkeypatch):
    """Row 0 is renameable independently of the class, so a rename must not
    clobber a prompt the user deliberately customised."""
    window.add_class("Drone")
    window.dino_phrase_panel._phrases["Drone"][0] = "small quadcopter"

    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("UAV", True))
    item = window.class_list.findItems("Drone", Qt.MatchFlag.MatchExactly)[0]
    window.class_controller.rename_class(item)

    assert window.dino_phrase_panel.get_phrases_for("UAV") == ["small quadcopter"]
