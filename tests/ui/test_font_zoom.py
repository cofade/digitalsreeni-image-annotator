"""Tests for the continuous UI font zoom (low-vision mode).

Uses a minimal QMainWindow stub carrying exactly the state
theme.set_font_pt touches, instead of constructing the full
ImageAnnotator (which would dominate suite runtime). save_ui_prefs is
patched out so tests never write the real per-user settings.
"""

import pytest
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMainWindow

from digitalsreeni_image_annotator.app_settings import (
    FONT_PT_DEFAULT,
    FONT_PT_MAX,
    FONT_PT_MIN,
)
from digitalsreeni_image_annotator.ui import theme
from digitalsreeni_image_annotator.widgets.image_label import ImageLabel


class _StubWindow(QMainWindow):
    """Just enough ImageAnnotator surface for theme.set_font_pt."""

    def __init__(self):
        super().__init__()
        self.font_sizes = {"Small": 8, "Medium": 10, "Large": 12, "XL": 14, "XXL": 16}
        self.ui_font_pt = FONT_PT_DEFAULT
        self.dark_mode = True
        self.image_label = ImageLabel()
        self._font_preset_actions = {}
        for name in self.font_sizes:
            action = QAction(name, self)
            action.setCheckable(True)
            self._font_preset_actions[name] = action


@pytest.fixture
def window(qt_application, monkeypatch):
    saved = []
    monkeypatch.setattr(
        theme, "save_ui_prefs", lambda pt, dark, settings=None: saved.append((pt, dark))
    )
    w = _StubWindow()
    w._saved_prefs = saved
    yield w
    w.image_label.deleteLater()
    w.deleteLater()


def test_step_up_increments_and_scales_canvas(window):
    theme.step_font_pt(window, 1)
    assert window.ui_font_pt == FONT_PT_DEFAULT + 1
    assert window.image_label.ui_scale == pytest.approx(
        (FONT_PT_DEFAULT + 1) / FONT_PT_DEFAULT
    )


def test_step_clamps_at_bounds(window):
    theme.set_font_pt(window, FONT_PT_MAX)
    theme.step_font_pt(window, 1)
    assert window.ui_font_pt == FONT_PT_MAX

    theme.set_font_pt(window, FONT_PT_MIN)
    theme.step_font_pt(window, -1)
    assert window.ui_font_pt == FONT_PT_MIN


def test_reset_returns_to_default(window):
    theme.set_font_pt(window, 20)
    theme.reset_font_pt(window)
    assert window.ui_font_pt == FONT_PT_DEFAULT
    assert window.image_label.ui_scale == pytest.approx(1.0)


def test_preset_entry_point_sets_value(window):
    theme.change_font_size(window, "XXL")
    assert window.ui_font_pt == 16


def test_preset_checkmark_follows_value(window):
    theme.change_font_size(window, "Large")
    assert window._font_preset_actions["Large"].isChecked()
    assert not window._font_preset_actions["Medium"].isChecked()

    # Stepping to an in-between size unchecks every preset.
    theme.step_font_pt(window, 1)  # 13pt — between Large and XL
    assert not any(a.isChecked() for a in window._font_preset_actions.values())


def test_every_change_is_persisted(window):
    theme.set_font_pt(window, 12)
    theme.step_font_pt(window, 1)
    assert window._saved_prefs == [(12, True), (13, True)]


def test_default_scale_renders_identical_to_legacy(window):
    """At the default 10pt, ui_scale is exactly 1.0 — overlay rendering
    must be pixel-identical to the pre-feature code paths."""
    theme.set_font_pt(window, FONT_PT_DEFAULT)
    assert window.image_label.ui_scale == 1.0
    assert window.image_label._pen_w(2) == pytest.approx(2.0)


def test_stylesheet_contains_scaled_overrides(window):
    theme.set_font_pt(window, 20)
    sheet = window.styleSheet()
    assert "QWidget { font-size: 20pt; }" in sheet
    # Header/indicator overrides scale the legacy px values by 2x at 20pt.
    assert "QLabel.section-header { font-size: 28px; }" in sheet
    assert "width: 28px; height: 28px;" in sheet
    assert "QRadioButton::indicator { border-radius: 16px; }" in sheet


def test_default_stylesheet_overrides_match_legacy_px(window):
    """At the 10pt default the appended overrides must reproduce the
    static stylesheets' values exactly (14px header, 14px indicators,
    8px radio radius) — the zoom feature must be invisible until used."""
    theme.set_font_pt(window, FONT_PT_DEFAULT)
    sheet = window.styleSheet()
    assert "QLabel.section-header { font-size: 14px; }" in sheet
    assert "width: 14px; height: 14px;" in sheet
    assert "QRadioButton::indicator { border-radius: 8px; }" in sheet
