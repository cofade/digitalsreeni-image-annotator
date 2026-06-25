"""Mutually-exclusive tool activation + Esc → selection mode.

activate_tool is the single choke-point that keeps current_tool, the SAM
flags, and the toolbar button checks in sync, so a SAM tool can never be
active alongside a manual tool. Esc on the canvas asks for selection mode via
selectModeRequested → activate_tool(None).

One real offscreen ImageAnnotator.
"""

import pytest


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def test_activating_tool_unchecks_all_others(window):
    window.activate_tool("polygon")
    assert window.image_label.current_tool == "polygon"
    assert window.polygon_button.isChecked()

    window.activate_tool("sam_box")
    assert window.image_label.current_tool == "sam_box"
    assert window.image_label.sam_box_active
    assert not window.image_label.sam_points_active
    assert not window.polygon_button.isChecked()       # exclusivity (the bug)
    assert window.sam_box_button.isChecked()


def test_switching_from_sam_clears_sam_flags(window):
    window.activate_tool("sam_points")
    assert window.image_label.sam_points_active

    window.activate_tool("rectangle")
    assert window.image_label.current_tool == "rectangle"
    assert not window.image_label.sam_points_active
    assert not window.image_label.sam_box_active
    assert not window.sam_points_button.isChecked()


def test_activate_none_returns_to_select_mode(window):
    window.activate_tool("paint_brush")
    window.activate_tool(None)
    assert window.image_label.current_tool is None
    assert not window.image_label.sam_box_active
    assert not window.image_label.sam_points_active
    assert window.image_label._is_select_mode()
    assert not any(
        b.isChecked() for b in window._tool_buttons().values()
    )


def test_select_mode_signal_deactivates_tool(window):
    # Esc on the canvas emits selectModeRequested; the window deactivates the
    # active tool and returns to selection mode.
    window.activate_tool("polygon")
    window.image_label.selectModeRequested.emit()
    assert window.image_label.current_tool is None
    assert window.image_label._is_select_mode()


def test_sam_toggle_routes_through_activate_tool(window):
    # toggle_sam_box mirrors a user click: the button check flips first, then
    # the handler routes through activate_tool.
    window.sam_box_button.setChecked(True)
    window.sam_controller.toggle_sam_box()
    assert window.image_label.current_tool == "sam_box"
    assert window.image_label.sam_box_active

    window.sam_box_button.setChecked(False)
    window.sam_controller.toggle_sam_box()
    assert window.image_label.current_tool is None
    assert not window.image_label.sam_box_active
