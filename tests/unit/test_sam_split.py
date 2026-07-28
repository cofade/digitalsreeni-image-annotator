"""
Unit tests for the SAM fine-tuner's deterministic per-image train/val split
(issue #85).

`split_groups` partitions a list of `SampleGroup`s into (train, val), reusing
the YOLO export's stable-hash `assign_train_val` so the SAM split is
reproducible. 100% train (or a single image) yields an empty val set.
"""

from src.digitalsreeni_image_annotator.training.sam_dataset import split_groups
from src.digitalsreeni_image_annotator.training.sam_trainer import SampleGroup


def _groups(n):
    return [
        SampleGroup(lambda: None, [{"bbox": [0, 0, 1, 1]}], name=f"img{i}.png")
        for i in range(n)
    ]


def test_split_80_20_counts():
    train, val = split_groups(_groups(10), 80)
    assert len(train) == 8 and len(val) == 2


def test_split_is_deterministic_across_calls():
    groups = _groups(20)
    a = [g.name for g in split_groups(groups, 75)[1]]
    b = [g.name for g in split_groups(groups, 75)[1]]
    assert a == b


def test_100_pct_has_no_val():
    train, val = split_groups(_groups(10), 100)
    assert len(train) == 10 and val == []


def test_single_group_is_all_train():
    train, val = split_groups(_groups(1), 80)
    assert len(train) == 1 and val == []


def test_frames_of_one_recording_stay_on_one_side():
    """The SAM split is group-aware too (ADR-044) — and it matters more here
    than for YOLO, because val loss drives early stopping."""
    groups = [
        SampleGroup(lambda: None, [{"bbox": [0, 0, 1, 1]}], name=name)
        for base in ("clipA", "clipB")
        for name in (f"{base}_F{i:05d}" for i in range(10))
    ]
    train, val = split_groups(groups, 50)
    train_bases = {g.name.split("_F")[0] for g in train}
    val_bases = {g.name.split("_F")[0] for g in val}
    assert train_bases.isdisjoint(val_bases)


def test_a_dataset_folder_groups_by_recording_too(tmp_path):
    """`export_sam_dataset` writes manifest paths like `images/clip_F00042.png`.

    The dot in that made `derive_groups` give every frame its own group, so
    "Fine-Tune SAM from Dataset Folder" silently got no grouping at all while
    the project path was correctly grouped. `build_groups_from_folder` now
    normalises the name to the ext-stripped basename.
    """
    import json

    from src.digitalsreeni_image_annotator.training.sam_dataset import (
        build_groups_from_folder,
    )

    images = tmp_path / "images"
    images.mkdir()
    entries = []
    for base in ("clipA", "clipB"):
        for i in range(10):
            name = f"{base}_F{i:05d}.png"
            (images / name).write_bytes(b"")
            entries.append(
                {"image": f"images/{name}", "instances": [{"bbox": [0, 0, 1, 1]}]}
            )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"images": entries}), encoding="utf-8"
    )

    groups = build_groups_from_folder(str(tmp_path))
    assert len(groups) == 20
    assert all("." not in g.name for g in groups), [g.name for g in groups]

    train, val = split_groups(groups, 50)
    train_bases = {g.name.split("_F")[0] for g in train}
    val_bases = {g.name.split("_F")[0] for g in val}
    assert train_bases and val_bases
    assert train_bases.isdisjoint(val_bases)


def test_a_supplied_grouping_overrides_the_derived_one():
    """The GUI warns about a grouping refined by curation clusters (ADR-045),
    then hands that exact mapping over. Re-deriving it on the worker thread
    would drop the refinement, so the dialog would describe one split and the
    run would perform another."""
    groups = _groups(10)
    keyed_groups = {f"{index}:img{index}.png": "one" for index in range(10)}

    train, val = split_groups(groups, 80, keyed_groups)

    # Everything is one group now, so a group-aware split cannot hold anything
    # out -- it falls back to the per-name split rather than emptying val.
    assert len(train) + len(val) == 10
    assert val, "an empty val set silently disables early stopping"


def test_keys_the_supplied_grouping_omits_fall_back_to_the_derived_one():
    """A stale mapping degrades to the structural split, not to nonsense."""
    groups = [
        SampleGroup(lambda: None, [{"bbox": [0, 0, 1, 1]}], name=name)
        for base in ("clipA", "clipB")
        for name in (f"{base}_F{i:05d}" for i in range(10))
    ]
    train, val = split_groups(groups, 50, {"nonsense:key": "whatever"})

    for base in ("clipA", "clipB"):
        held = {g.name for g in val if g.name.startswith(base)}
        assert len(held) in (0, 10), base
    assert len(train) + len(val) == 20


def test_split_is_disjoint_and_complete():
    groups = _groups(13)
    train, val = split_groups(groups, 70)
    names = {g.name for g in train} | {g.name for g in val}
    assert len(names) == 13 and len(train) + len(val) == 13
