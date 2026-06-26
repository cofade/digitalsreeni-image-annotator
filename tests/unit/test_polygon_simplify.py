"""Unit tests for reversible polygon simplification (issue #24).

simplify_polygon (Douglas-Peucker via cv2.approxPolyDP) thins a dense polygon
to a Detail % of its vertices, where 100 = the raw polygon unchanged. Pure, no
Qt.
"""

import math

from src.digitalsreeni_image_annotator.utils import simplify_polygon


def _circle(cx, cy, r, n):
    """A flat [x1,y1,...] polygon approximating a circle with n vertices."""
    seg = []
    for i in range(n):
        a = 2 * math.pi * i / n
        seg += [cx + r * math.cos(a), cy + r * math.sin(a)]
    return seg


def _npts(seg):
    return len(seg) // 2


def test_detail_100_returns_raw_unchanged():
    seg = _circle(50, 50, 30, 60)
    assert simplify_polygon(seg, 100) == seg


def test_lower_detail_reduces_vertex_count_within_budget():
    seg = _circle(50, 50, 30, 60)
    out = simplify_polygon(seg, 50)
    assert _npts(out) <= 30                 # <= round(60 * 50/100)
    assert _npts(out) >= 3
    assert len(out) % 2 == 0 and len(out) >= 6


def test_more_aggressive_keeps_fewer_points():
    seg = _circle(50, 50, 30, 60)
    coarse = simplify_polygon(seg, 15)
    fine = simplify_polygon(seg, 60)
    assert _npts(coarse) <= _npts(fine)


def test_does_not_mutate_input():
    seg = _circle(50, 50, 30, 40)
    snapshot = list(seg)
    simplify_polygon(seg, 30)
    assert seg == snapshot                   # raw is preserved for reversibility


def test_tiny_polygon_unchanged():
    tri = [0, 0, 10, 0, 5, 10]               # 3 points — nothing to simplify
    assert simplify_polygon(tri, 10) == tri
    square = [0, 0, 10, 0, 10, 10, 0, 10]    # 4 points
    assert simplify_polygon(square, 10) == square


def test_result_is_a_valid_subset_shape():
    # A coarse simplification of a circle should still be a sensible polygon
    # bounded by the original extent.
    seg = _circle(50, 50, 30, 60)
    out = simplify_polygon(seg, 20)
    xs, ys = out[0::2], out[1::2]
    assert min(xs) >= 19 and max(xs) <= 81   # within the circle's bounds (+/- 1)
    assert min(ys) >= 19 and max(ys) <= 81
