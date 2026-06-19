"""
Unit tests for nested-polygon edit selection (upstream issue #33).

start_polygon_edit must enter edit mode on the *smallest* polygon that
contains the click, so an annotation fully nested inside another is
reachable instead of always grabbing the outer one.
"""

import pytest

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


@pytest.fixture
def nested(label):
    outer = _square(0, 0, 100, "outer")      # area 10000
    inner = _square(40, 40, 20, "inner")     # area 400, fully inside outer
    # Insert outer first so the old "return first match" behavior would
    # have returned outer — the test fails unless smallest-area wins.
    label.annotations = {"cell": [outer, inner]}
    return label, outer, inner


def test_click_in_nested_region_selects_inner(nested):
    label, outer, inner = nested
    result = label.start_polygon_edit((50, 50))  # inside both
    assert result is inner
    assert label.editing_polygon is inner


def test_click_only_in_outer_selects_outer(nested):
    label, outer, inner = nested
    result = label.start_polygon_edit((10, 10))  # inside outer only
    assert result is outer
    assert label.editing_polygon is outer


def test_click_outside_all_returns_none(nested):
    label, outer, inner = nested
    label.editing_polygon = None
    result = label.start_polygon_edit((500, 500))
    assert result is None


def test_insertion_order_does_not_matter(label):
    # Inner listed first: result must still be the smallest, not the first.
    outer = _square(0, 0, 100, "outer")
    inner = _square(40, 40, 20, "inner")
    label.annotations = {"cell": [inner, outer]}
    assert label.start_polygon_edit((50, 50)) is inner


def test_bbox_only_annotation_is_ignored(label):
    # start_polygon_edit only handles "segmentation"; bbox editing is #40.
    bbox_ann = {"bbox": [0, 0, 100, 100], "category_name": "box"}
    label.annotations = {"cell": [bbox_ann]}
    assert label.start_polygon_edit((50, 50)) is None
