"""Unit tests for the bounds-enforcement geometry helpers (issues #32 / #36).

Pure-Python (no Qt): clamp_segmentation / clamp_bbox snap manual-edit coords
into the image rectangle; clip_polygon_to_bounds geometrically trims an
augmented polygon (and drops one that falls fully outside).
"""

from src.digitalsreeni_image_annotator.utils import (
    clamp_bbox,
    clamp_segmentation,
    clip_polygon_to_bounds,
    fit_bbox_inside,
)


# --- clamp_segmentation ----------------------------------------------------

def test_clamp_segmentation_in_bounds_unchanged():
    seg = [10, 10, 90, 10, 90, 90, 10, 90]
    assert clamp_segmentation(seg, 100, 100) == seg


def test_clamp_segmentation_snaps_out_of_bounds():
    # Negative and over-size coords snap to the [0, w] x [0, h] edges.
    seg = [-5, -20, 150, 30, 50, 200]
    assert clamp_segmentation(seg, 100, 100) == [0, 0, 100, 30, 50, 100]


def test_clamp_segmentation_preserves_vertex_count():
    seg = [-1, -1, 200, 200, 50, -10, -10, 50]
    out = clamp_segmentation(seg, 100, 100)
    assert len(out) == len(seg)  # per-coordinate clamp never adds/drops vertices


def test_clamp_segmentation_does_not_mutate_input():
    seg = [-5, 5, 5, 5, 5, -5]
    clamp_segmentation(seg, 100, 100)
    assert seg == [-5, 5, 5, 5, 5, -5]


# --- clamp_bbox ------------------------------------------------------------

def test_clamp_bbox_in_bounds_unchanged():
    assert clamp_bbox([10, 10, 30, 30], 100, 100) == [10, 10, 30, 30]


def test_clamp_bbox_partly_outside_top_left():
    assert clamp_bbox([-10, -10, 50, 50], 100, 100) == [0, 0, 40, 40]


def test_clamp_bbox_larger_than_image():
    assert clamp_bbox([-10, -10, 200, 200], 100, 100) == [0, 0, 100, 100]


def test_clamp_bbox_fully_outside_keeps_min_size_inside():
    out = clamp_bbox([200, 10, 30, 30], 100, 100)
    x, y, w, h = out
    assert w >= 1 and h >= 1
    assert 0 <= x and x + w <= 100        # stays inside even after min-size bump
    assert 0 <= y and y + h <= 100


# --- fit_bbox_inside (move path: translate inside, preserve size) ----------

def test_fit_bbox_inside_in_bounds_unchanged():
    assert fit_bbox_inside([10, 10, 30, 30], 100, 100) == [10, 10, 30, 30]


def test_fit_bbox_inside_off_bottom_right_preserves_size():
    # A move that overshot the bottom-right slides back in, keeping 30x30.
    assert fit_bbox_inside([120, 120, 30, 30], 100, 100) == [70, 70, 30, 30]


def test_fit_bbox_inside_off_top_left_preserves_size():
    # The bug clamp_bbox had: crossing the top/left edge must NOT collapse.
    assert fit_bbox_inside([-50, -50, 40, 40], 100, 100) == [0, 0, 40, 40]
    assert fit_bbox_inside([10, -50, 40, 40], 100, 100) == [10, 0, 40, 40]


def test_fit_bbox_inside_larger_than_image_shrinks_to_fit():
    assert fit_bbox_inside([-10, -10, 200, 200], 100, 100) == [0, 0, 100, 100]


# --- clip_polygon_to_bounds ------------------------------------------------

def _bounds(seg):
    xs, ys = seg[0::2], seg[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def test_clip_polygon_in_bounds_unchanged_area():
    seg = [10, 10, 90, 10, 90, 90, 10, 90]
    out = clip_polygon_to_bounds(seg, 100, 100)
    assert out is not None
    x0, y0, x1, y1 = _bounds(out)
    assert (x0, y0, x1, y1) == (10, 10, 90, 90)


def test_clip_polygon_partly_outside_is_trimmed():
    # A 50..150 square in a 100x100 image clips to 50..100.
    seg = [50, 50, 150, 50, 150, 150, 50, 150]
    out = clip_polygon_to_bounds(seg, 100, 100)
    assert out is not None
    x0, y0, x1, y1 = _bounds(out)
    assert x0 >= 0 and y0 >= 0 and x1 <= 100 and y1 <= 100
    assert x1 == 100 and y1 == 100        # cut exactly at the image edge


def test_clip_polygon_fully_outside_returns_none():
    seg = [200, 200, 300, 200, 300, 300, 200, 300]
    assert clip_polygon_to_bounds(seg, 100, 100) is None


def test_clip_polygon_degenerate_returns_none():
    assert clip_polygon_to_bounds([10, 10, 20, 20], 100, 100) is None  # <3 pts


def test_clip_polygon_self_intersecting_is_repaired():
    # A bow-tie (self-intersecting) polygon: shapely .buffer(0) splits it into a
    # MultiPolygon; clip must still return a valid in-bounds ring (largest part).
    bowtie = [0, 0, 100, 100, 100, 0, 0, 100]
    out = clip_polygon_to_bounds(bowtie, 100, 100)
    assert out is not None
    x0, y0, x1, y1 = _bounds(out)
    assert x0 >= 0 and y0 >= 0 and x1 <= 100 and y1 <= 100
