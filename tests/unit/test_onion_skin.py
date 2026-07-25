"""Onion-skinning: neighbour resolution, rendering and persistence (issue #67).

The neighbour arithmetic is Qt-free and gets the bulk of the coverage here,
because index arithmetic that silently wraps at the ends of a stack is a bug a
user would only notice as "the last frame ghosts the first" — exactly the kind
of thing that should die in a unit test.

The rendering side asserts two things: that the ghost is drawn at all, and
— more importantly — that it stays *decorative*. A ghost that leaked into
hit-testing, SAM input or export would be a correctness bug, not a cosmetic one.
"""

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColor, QPixmap

from src.digitalsreeni_image_annotator import app_settings
from src.digitalsreeni_image_annotator.core import onion
from tests.canvas_fixtures import RecordingPainter, make_label, square

NAMES = [f"stack_Z{i}" for i in range(1, 6)]  # Z1..Z5


# --- neighbour resolution --------------------------------------------------


def test_previous_is_the_default_neighbour():
    assert onion.neighbour_names(NAMES, "stack_Z3") == ["stack_Z2"]


def test_next_mode():
    assert onion.neighbour_names(NAMES, "stack_Z3", mode=onion.MODE_NEXT) == [
        "stack_Z4"
    ]


def test_both_mode_returns_them_in_order():
    assert onion.neighbour_names(NAMES, "stack_Z3", mode=onion.MODE_BOTH) == [
        "stack_Z2",
        "stack_Z4",
    ]


def test_offset_reaches_further():
    assert onion.neighbour_names(NAMES, "stack_Z3", offset=2) == ["stack_Z1"]


def test_first_slice_has_no_previous_ghost():
    assert onion.neighbour_names(NAMES, "stack_Z1") == []


def test_last_slice_has_no_next_ghost():
    assert onion.neighbour_names(NAMES, "stack_Z5", mode=onion.MODE_NEXT) == []


def test_ends_never_wrap_around():
    """The failure this guards against: Z5 ghosting Z1, which looks plausible
    and is completely wrong."""
    assert onion.neighbour_names(NAMES, "stack_Z5", mode=onion.MODE_BOTH) == [
        "stack_Z4"
    ]
    assert onion.neighbour_names(NAMES, "stack_Z1", mode=onion.MODE_BOTH) == [
        "stack_Z2"
    ]


def test_offset_past_the_end_yields_nothing_rather_than_clamping():
    """Clamping would ghost the *wrong* slice while looking like it worked."""
    assert onion.neighbour_names(NAMES, "stack_Z2", offset=4) == []


def test_a_single_image_has_no_neighbours():
    assert onion.neighbour_names(["only.png"], "only.png") == []


def test_an_unknown_current_slice_yields_nothing():
    assert onion.neighbour_names(NAMES, "not_in_the_stack") == []


def test_empty_collection_is_safe():
    assert onion.neighbour_names([], None) == []
    assert onion.neighbour_names(None, "x") == []


def test_is_available_needs_more_than_one_slice():
    assert onion.is_available(NAMES) is True
    assert onion.is_available(["one"]) is False
    assert onion.is_available([]) is False
    assert onion.is_available(None) is False


# --- clamps ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [(0.0, 0.05), (1.0, 0.95), (-5, 0.05), (0.4, 0.4), ("nonsense", 0.35)],
)
def test_opacity_is_clamped_away_from_both_extremes(raw, expected):
    """Never 0 (invisible ghost, decode cost still paid) and never 1 (the ghost
    would completely hide the current slice)."""
    assert onion.clamp_opacity(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [(0, 1), (99, 5), (3, 3), (None, 1)])
def test_offset_is_clamped(raw, expected):
    assert onion.clamp_offset(raw) == expected


def test_unknown_mode_falls_back_to_the_default():
    assert onion.normalise_mode("sideways") == onion.MODE_PREVIOUS
    assert onion.normalise_mode(None) == onion.MODE_PREVIOUS
    assert onion.normalise_mode(onion.MODE_BOTH) == onion.MODE_BOTH


# --- persistence (ADR-020) -------------------------------------------------


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)


def test_onion_prefs_round_trip(settings):
    app_settings.save_onion_prefs(True, 0.6, 3, onion.MODE_BOTH, settings)
    enabled, opacity, offset, mode = app_settings.load_onion_prefs(settings)
    assert enabled is True
    assert opacity == pytest.approx(0.6)
    assert offset == 3
    assert mode == onion.MODE_BOTH


def test_onion_prefs_default_to_off(settings):
    enabled, opacity, offset, mode = app_settings.load_onion_prefs(settings)
    assert enabled is False
    assert opacity == pytest.approx(onion.DEFAULT_OPACITY)
    assert offset == onion.DEFAULT_OFFSET
    assert mode == onion.MODE_PREVIOUS


def test_garbage_in_settings_does_not_crash_startup(settings):
    """A hand-edited INI must not be able to break the app or produce an
    invisible-but-expensive ghost."""
    settings.setValue("ui/onion_opacity", "not a number")
    settings.setValue("ui/onion_offset", "lots")
    settings.setValue("ui/onion_mode", "diagonally")
    _enabled, opacity, offset, mode = app_settings.load_onion_prefs(settings)
    assert opacity == pytest.approx(onion.DEFAULT_OPACITY)
    assert offset == onion.DEFAULT_OFFSET
    assert mode == onion.MODE_PREVIOUS


# --- rendering -------------------------------------------------------------


@pytest.fixture
def label(qtbot):
    return make_label(qtbot)


def _ghost(width=200, height=200, colour="#00FF00"):
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(colour))
    return pixmap


def test_no_ghost_means_no_painter_calls(label):
    painter = RecordingPainter()
    label.renderer.draw_onion_skin(painter)
    assert painter.calls == []


def test_ghost_is_blitted_at_the_configured_opacity(label):
    label.onion_pixmaps = [_ghost()]
    label.onion_opacity = 0.4

    painter = RecordingPainter()
    label.renderer.draw_onion_skin(painter)

    assert painter.count("drawPixmap") == 1
    opacities = [args[0] for name, args in painter.calls if name == "setOpacity"]
    assert opacities[0] == pytest.approx(0.4)


def test_opacity_is_restored_so_later_layers_are_not_faded(label):
    """Leaving the painter at 0.4 would wash out every annotation drawn after."""
    label.onion_pixmaps = [_ghost()]
    label.onion_opacity = 0.4

    painter = RecordingPainter()
    label.renderer.draw_onion_skin(painter)

    opacities = [args[0] for name, args in painter.calls if name == "setOpacity"]
    assert opacities[-1] == 1.0


def test_both_neighbours_are_drawn(label):
    label.onion_pixmaps = [_ghost(), _ghost(colour="#0000FF")]
    painter = RecordingPainter()
    label.renderer.draw_onion_skin(painter)
    assert painter.count("drawPixmap") == 2


def test_a_null_ghost_is_skipped_silently(label):
    """A failed decode is not an error — it is simply no ghost."""
    label.onion_pixmaps = [QPixmap(), None, _ghost()]
    painter = RecordingPainter()
    label.renderer.draw_onion_skin(painter)
    assert painter.count("drawPixmap") == 1


def test_ghost_uses_the_same_origin_as_the_image(label):
    """Pan and zoom must keep ghost and current slice in lockstep."""
    label.offset_x, label.offset_y = 37, 11
    label.zoom_factor = 2.0
    label.onion_pixmaps = [_ghost(100, 100)]

    painter = RecordingPainter()
    label.renderer.draw_onion_skin(painter)

    name, args = next(c for c in painter.calls if c[0] == "drawPixmap")
    assert (args[0], args[1]) == (37, 11)
    assert args[2].width() == 200, "ghost scales with zoom like the image"


def test_ghost_renders_between_the_image_and_the_annotations(qtbot):
    """The layer order that makes the feature both visible and usable: over the
    opaque raster, under every annotation."""
    label = make_label(qtbot)
    label.annotations = {"cell": [square(10, 10, 40)]}
    label.onion_pixmaps = [_ghost()]

    order = []
    label.renderer.draw_onion_skin = lambda *a, **k: order.append("onion")
    label.renderer.draw_annotations = lambda *a, **k: order.append("annotations")

    label.grab()

    assert order == ["onion", "annotations"]


# --- the ghost stays decorative -------------------------------------------


def test_ghost_does_not_become_the_current_image(label):
    """``original_pixmap`` is what SAM, export and every measurement read. If
    the ghost ever landed there the consequences would be silent and wrong."""
    before = label.original_pixmap
    label.onion_pixmaps = [_ghost(colour="#FF00FF")]
    assert label.original_pixmap is before


def test_ghost_is_not_hit_testable(label):
    """Selection reads ``annotations``; the ghost contributes nothing to it."""
    label.annotations = {}
    label.onion_pixmaps = [_ghost()]
    assert label.annotation_at((100, 100)) is None


def test_clearing_the_canvas_drops_the_ghost(label):
    label.set_onion_pixmaps([_ghost()])
    label.clear()
    assert label.onion_pixmaps == []
    assert label.scaled_onion_pixmaps() == []


# --- the scaled cache ------------------------------------------------------
#
# Scaling in the paint pass meant two full-resolution SmoothTransformation
# rescales per repaint on the GUI thread throughout a pan -- the exact cost the
# main image already avoids with `scaled_pixmap`. These pin the cache contract.


def test_the_scaled_cache_is_reused_within_one_zoom_level(label):
    label.set_onion_pixmaps([_ghost(100, 100)])
    first = label.scaled_onion_pixmaps()
    assert label.scaled_onion_pixmaps() is first, "rescaled on an unchanged zoom"


def test_changing_the_zoom_invalidates_the_cache(label):
    label.zoom_factor = 1.0
    label.set_onion_pixmaps([_ghost(100, 100)])
    assert label.scaled_onion_pixmaps()[0].width() == 100

    label.zoom_factor = 2.0
    assert label.scaled_onion_pixmaps()[0].width() == 200


def test_replacing_the_ghosts_invalidates_the_cache(label):
    """The whole reason set_onion_pixmaps exists rather than assigning the list
    directly: a stale scaled copy would keep drawing the previous slice."""
    label.set_onion_pixmaps([_ghost(100, 100)])
    assert len(label.scaled_onion_pixmaps()) == 1

    label.set_onion_pixmaps([_ghost(100, 100), _ghost(100, 100, "#FF0000")])
    assert len(label.scaled_onion_pixmaps()) == 2


def test_null_ghosts_are_dropped_by_the_cache_not_the_paint_pass(label):
    label.set_onion_pixmaps([QPixmap(), None, _ghost()])
    assert len(label.scaled_onion_pixmaps()) == 1
