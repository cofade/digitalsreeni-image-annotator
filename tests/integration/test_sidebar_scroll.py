"""Integration test for the scrollable left sidebar (upstream issue #88).

The left sidebar packs Import, Classes, Annotation tools, the DINO panel,
the Annotations table and Export into one column. On small screens or at
large UI font sizes an expanded DINO panel used to squeeze the Annotations
table down to just its header row. The fix wraps the whole sidebar in a
QScrollArea and gives each competing section a usable minimum height, so
the sidebar scrolls vertically instead of collapsing a section.

Constructs one full offscreen ImageAnnotator because the wiring under test
is created in ui.sidebar.build_sidebar during setup_ui.
"""

import pytest


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def test_sidebar_wrapped_in_scroll_area(window):
    from PyQt6.QtWidgets import QScrollArea

    assert isinstance(window.sidebar_scroll, QScrollArea)
    assert window.sidebar_scroll.widgetResizable()
    # The sidebar content widget is the scroll area's inner widget.
    assert window.sidebar_scroll.widget() is window.sidebar


def test_sidebar_has_no_horizontal_scroll(window):
    from PyQt6.QtCore import Qt

    assert (
        window.sidebar_scroll.horizontalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )


def test_key_sections_keep_a_usable_minimum_height(window):
    # Each vertically-competing section keeps a minimum so it can't be
    # squeezed to a header row; the scroll area supplies scrolling instead.
    assert window.class_list.minimumHeight() > 0
    assert window.annotation_list.minimumHeight() > 0
    assert window.dino_class_table.minimumHeight() > 0
    assert window.dino_phrase_panel.minimumHeight() > 0
