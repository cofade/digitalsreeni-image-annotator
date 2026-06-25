"""Unit tests for AnnotationHistory (pure, no Qt).

Exercises the symmetric record/undo/redo model, per-key isolation, the
deep-equal dedup, depth capping, and snapshot independence (mutating a
returned snapshot must not corrupt stored history).
"""

from digitalsreeni_image_annotator.controllers.annotation_history import (
    AnnotationHistory,
)


def _snap(*names):
    """A tiny annotation-dict snapshot keyed by class name."""
    return {n: [{"segmentation": [0, 0, 1, 1], "category_name": n}] for n in names}


def test_record_then_undo_returns_previous_state():
    h = AnnotationHistory()
    s0, s1 = _snap(), _snap("a")
    h.record("img", s0)            # before the first edit
    # (live is now s1 outside the history object)
    assert h.can_undo("img")
    restored = h.undo("img", s1)
    assert restored == s0


def test_redo_returns_to_post_edit_state():
    h = AnnotationHistory()
    s0, s1 = _snap(), _snap("a")
    h.record("img", s0)
    h.undo("img", s1)             # back to s0
    assert h.can_redo("img")
    assert h.redo("img", s0) == s1


def test_record_clears_redo():
    h = AnnotationHistory()
    s0, s1, s2 = _snap(), _snap("a"), _snap("b")
    h.record("img", s0)
    h.undo("img", s1)            # redo now holds s1
    assert h.can_redo("img")
    h.record("img", s0)         # a fresh edit must invalidate redo
    assert not h.can_redo("img")
    # live becomes s2 after the new edit
    assert h.undo("img", s2) == s0


def test_dedup_skips_equal_consecutive():
    h = AnnotationHistory()
    s0 = _snap("a")
    h.record("img", s0)
    h.record("img", _snap("a"))   # value-equal -> skipped
    assert h.undo("img", _snap("a")) == s0
    assert not h.can_undo("img")   # only one entry existed


def test_per_key_isolation():
    h = AnnotationHistory()
    h.record("A", _snap())
    assert h.can_undo("A")
    assert not h.can_undo("B")
    assert h.undo("B", _snap("x")) is None


def test_depth_cap_drops_oldest():
    h = AnnotationHistory(max_depth=3)
    for i in range(5):
        h.record("img", {"k": [i]})
    # Only the last 3 before-states survive.
    stack = h._stacks["img"]["undo"]
    assert [s["k"][0] for s in stack] == [2, 3, 4]


def test_snapshot_independence():
    h = AnnotationHistory()
    s0 = _snap("a")
    h.record("img", s0)
    s0["a"].append("mutated")      # external mutation after recording
    restored = h.undo("img", _snap("a"))
    assert restored == _snap("a")  # stored copy unaffected


def test_clear_and_drop():
    h = AnnotationHistory()
    h.record("A", _snap())
    h.record("B", _snap())
    h.drop("A")
    assert not h.can_undo("A")
    assert h.can_undo("B")
    h.clear()
    assert not h.can_undo("B")
