"""Unit tests for KeypointSchemaDialog's stateful bits (issue #35).

Focus: the point-reorder must remap skeleton edges AND flip partners through the
swap permutation, not just the display names (senior-review P1), and a plain
build round-trips through _collect/get_schema.
"""

import pytest

from src.digitalsreeni_image_annotator.dialogs.keypoint_schema_dialog import (
    KeypointSchemaDialog,
)


@pytest.fixture
def dialog(qtbot):
    def _make(schema):
        d = KeypointSchemaDialog(None, class_name="person", schema=schema)
        qtbot.addWidget(d)
        return d

    return _make


def _accept(d):
    d._on_accept()
    return d.get_schema()


def test_build_roundtrips(dialog):
    schema = {"names": ["a", "b", "c"], "skeleton": [[0, 2]], "flip_idx": [0, 2, 1]}
    d = dialog(schema)
    assert _accept(d) == schema


def test_move_point_remaps_skeleton_and_flip(dialog):
    # a=self, b<->c ; edge a-c. Move row 0 (a) down past b (swap 0,1).
    d = dialog({"names": ["a", "b", "c"], "skeleton": [[0, 2]], "flip_idx": [0, 2, 1]})
    d.points_table.setCurrentCell(0, 0)
    d._move_point(1)
    out = _accept(d)
    # New order: b, a, c. b<->c must now be 0<->2; a self; edge a-c now 1-2.
    assert out["names"] == ["b", "a", "c"]
    assert out["flip_idx"] == [2, 1, 0]
    assert out["skeleton"] == [[1, 2]]


def test_move_point_is_reversible(dialog):
    schema = {"names": ["a", "b", "c"], "skeleton": [[0, 2]], "flip_idx": [0, 2, 1]}
    d = dialog(schema)
    d.points_table.setCurrentCell(0, 0)
    d._move_point(1)   # a down
    d.points_table.setCurrentCell(1, 0)
    d._move_point(-1)  # a back up
    assert _accept(d) == schema


def test_reject_duplicate_names(dialog, monkeypatch):
    d = dialog({"names": ["a", "b"], "skeleton": [], "flip_idx": [0, 1]})
    d.points_table.item(1, 0).setText("a")  # make them duplicate
    warned = []
    monkeypatch.setattr(
        "digitalsreeni_image_annotator.dialogs.keypoint_schema_dialog.QMessageBox.warning",
        lambda *a, **k: warned.append(a),
    )
    d._on_accept()
    assert d.get_schema() is None and warned
