"""Default class-colour palette (issue #75 follow-up).

Red must no longer be the first auto-assigned class colour (it collided with
the selection highlight); it stays in the palette but at the back.
"""

from src.digitalsreeni_image_annotator.core.constants import (
    DEFAULT_CLASS_COLORS,
    default_class_color,
)

_RED = "#D62728"


def test_first_default_color_is_not_red():
    assert default_class_color(0).upper() != _RED
    assert default_class_color(0) == DEFAULT_CLASS_COLORS[0]


def test_palette_cycles_modulo_length():
    n = len(DEFAULT_CLASS_COLORS)
    assert default_class_color(n) == default_class_color(0)
    assert default_class_color(n + 3) == default_class_color(3)


def test_red_present_but_last():
    upper = [c.upper() for c in DEFAULT_CLASS_COLORS]
    assert _RED in upper
    assert upper[-1] == _RED


def test_all_entries_distinct_and_valid_hex():
    assert len(set(DEFAULT_CLASS_COLORS)) == len(DEFAULT_CLASS_COLORS)
    for c in DEFAULT_CLASS_COLORS:
        assert c.startswith("#") and len(c) == 7
