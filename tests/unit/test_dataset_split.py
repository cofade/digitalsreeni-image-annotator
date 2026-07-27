"""Group-aware train/val splitting (issue #80, ADR-044).

The bug these tests exist for is invisible at runtime. A name-keyed split over
a video's frames produces a perfectly ordinary-looking dataset, trains without
a warning, and reports validation metrics that are simply too good — the more
redundant the data, the better they look. Nothing fails; the numbers just stop
meaning what everyone reads them as.

So the load-bearing test here is :func:`test_no_group_ever_straddles_the_split`.
The rest establish that the grouping is derived correctly in the first place.
"""

import subprocess
import sys

from src.digitalsreeni_image_annotator.core import dataset_split
from src.digitalsreeni_image_annotator.core.dataset_split import (
    assign_train_val,
    derive_groups,
    merge_groups,
    plan_split,
)


class _FakeSliceList:
    """Stand-in for ``LazySliceList``: only ``.names`` is ever touched."""

    def __init__(self, names):
        self.names = list(names)


def _frames(base, count):
    return [f"{base}_F{i:05d}" for i in range(count)]


def _buckets(names, groups):
    """``{group_key: {name, ...}}`` for asserting on whole groups."""
    buckets = {}
    for name in names:
        buckets.setdefault(groups[name], set()).add(name)
    return buckets


# --- Qt-free guarantee -----------------------------------------------------


def test_the_split_imports_without_qt():
    """``io.export_formats`` imports this module, and the headless CLI imports
    that — a stray Qt import here would make a CI export need a display.

    Specifically it must not reach ``core.slice_cache``, which pulls in
    ``core.image_utils`` and therefore ``QImage``.
    """
    code = (
        "import sys;"
        "sys.path.insert(0, 'src');"
        "import digitalsreeni_image_annotator.core.dataset_split as m;"
        "qt = [n for n in sys.modules if n.startswith('PyQt6')];"
        "assert not qt, qt;"
        "print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# --- deriving the grouping -------------------------------------------------


def test_slices_group_by_their_stack():
    """``image_slices`` is keyed by the ext-stripped base name, so the mapping
    is exact — no parsing, no pixel work."""
    image_slices = {"stack": _FakeSliceList(["stack_T1_Z1", "stack_T1_Z2"])}
    groups = derive_groups(["stack_T1_Z1", "stack_T1_Z2", "photo.png"], image_slices)
    assert groups["stack_T1_Z1"] == "stack"
    assert groups["stack_T1_Z2"] == "stack"
    assert groups["photo.png"] == "photo.png"


def test_a_plain_list_slice_collection_also_groups():
    """Legacy call sites and several tests still hand in ``[(name, qimage)]``."""
    image_slices = {"stack": [("stack_Z1", None), ("stack_Z2", None)]}
    groups = derive_groups(["stack_Z1", "stack_Z2"], image_slices)
    assert groups["stack_Z1"] == groups["stack_Z2"] == "stack"


def test_slice_names_group_without_any_image_slices():
    """The fallback that protects the CLI, which passes an empty mapping, and
    an .iap whose stack was never materialised this session."""
    names = ["stack_T1_Z1", "stack_T2_Z1", "other_T1_Z1"]
    groups = derive_groups(names)
    assert groups["stack_T1_Z1"] == groups["stack_T2_Z1"] == "stack"
    assert groups["other_T1_Z1"] == "other"


def test_video_frames_group_by_recording():
    names = _frames("clip", 3) + _frames("other", 2)
    groups = derive_groups(names)
    assert {groups[n] for n in _frames("clip", 3)} == {"clip"}
    assert {groups[n] for n in _frames("other", 2)} == {"other"}


def test_a_regular_image_is_its_own_group():
    """Regular names keep their extension; the dot is what tells them apart
    from a slice name, the same signal the exporters already use."""
    names = ["a_T1.png", "b.jpg", "c.png"]
    groups = derive_groups(names)
    assert groups == {"a_T1.png": "a_T1.png", "b.jpg": "b.jpg", "c.png": "c.png"}


def test_an_exact_mapping_beats_the_name_heuristic():
    """A stack whose base name itself looks like a slice suffix is grouped by
    what ``image_slices`` says, not by what the regex guesses."""
    image_slices = {"run_T1": _FakeSliceList(["run_T1_Z1", "run_T1_Z2"])}
    groups = derive_groups(["run_T1_Z1", "run_T1_Z2"], image_slices)
    assert groups["run_T1_Z1"] == groups["run_T1_Z2"] == "run_T1"


# --- folding in near-duplicate clusters ------------------------------------


def test_merge_groups_is_transitive():
    """Names already sharing a stack plus a cluster linking one of them to a
    third name must all end up together, or the refinement reintroduces the
    very straddle it was meant to close."""
    groups = {"a_Z1": "a", "a_Z2": "a", "loose.png": "loose.png"}
    merged = merge_groups(groups, [["a_Z2", "loose.png"]])
    assert len({merged["a_Z1"], merged["a_Z2"], merged["loose.png"]}) == 1


def test_a_cluster_member_outside_the_grouping_still_bridges():
    """An unannotated image is not in the split, but it is evidence that the
    two images it resembles are near-duplicates of each other."""
    groups = {"left.png": "left.png", "right.png": "right.png"}
    merged = merge_groups(groups, [["left.png", "unannotated.png", "right.png"]])
    assert merged["left.png"] == merged["right.png"]


def test_merging_no_clusters_changes_nothing():
    groups = {"a.png": "a.png", "b.png": "b.png"}
    assert merge_groups(groups, []) == groups
    assert merge_groups(groups, None) == groups


# --- the split itself ------------------------------------------------------


def test_no_group_ever_straddles_the_split():
    """THE test. Two recordings plus loose photos: every frame of a recording
    has to land on one side, whichever side that is."""
    names = _frames("clipA", 30) + _frames("clipB", 30) + [
        f"photo{i}.png" for i in range(10)
    ]
    groups = derive_groups(names)
    train, val, fell_back = plan_split(names, 20, groups)

    assert not fell_back
    for members in _buckets(names, groups).values():
        assert members <= train or members <= val, members


def test_a_video_project_holds_out_a_whole_recording():
    names = _frames("clipA", 20) + _frames("clipB", 20)
    groups = derive_groups(names)
    train, val, _ = plan_split(names, 50, groups)
    assert train and val
    assert train.isdisjoint(val)
    assert train | val == set(names)


def test_a_single_recording_falls_back_and_says_so():
    """One video is the case where no honest split exists. Returning an empty
    val set would be truthful but makes the trainer silently drop validation
    and early stopping (ADR-028), so the flag carries the news instead."""
    names = _frames("clip", 20)
    groups = derive_groups(names)
    train, val, fell_back = plan_split(names, 20, groups)
    assert fell_back
    assert len(val) == 4 and len(train) == 16


def test_neither_side_is_ever_empty_even_with_lopsided_groups():
    """A group is indivisible, so a single huge one could otherwise swallow
    the whole dataset and leave train empty — which is not a split at all."""
    names = _frames("big", 18) + ["a.png", "b.png"]
    groups = derive_groups(names)
    train, val, _ = plan_split(names, 80, groups)
    assert train and val
    assert train | val == set(names)


def test_the_group_split_is_deterministic():
    names = _frames("clipA", 10) + _frames("clipB", 10) + ["x.png", "y.png"]
    groups = derive_groups(names)
    first = plan_split(names, 30, groups)
    second = plan_split(list(reversed(names)), 30, groups)
    assert first[:2] == second[:2]


def test_grouping_by_identity_matches_the_ungrouped_split():
    """The compatibility guarantee: every name in its own group is exactly the
    historical per-name split, so `groups=None` callers are unaffected."""
    names = [f"img_{i:03d}.png" for i in range(37)]
    assert assign_train_val(names, 30) == assign_train_val(
        names, 30, {name: name for name in names}
    )


def test_zero_split_never_reports_a_fallback():
    """With no val set requested there is nothing to warn about."""
    names = _frames("clip", 10)
    assert plan_split(names, 0, derive_groups(names)) == (set(names), set(), False)


def test_the_split_covers_and_partitions_every_name():
    names = _frames("clipA", 7) + _frames("clipB", 5) + ["solo.png"]
    groups = derive_groups(names)
    train, val, _ = plan_split(names, 25, groups)
    assert train.isdisjoint(val)
    assert train | val == set(names)


def test_slice_base_leaves_a_dotted_name_alone():
    """Guards the one heuristic in the module against widening."""
    assert dataset_split._slice_base("photo_T1.png") is None
    assert dataset_split._slice_base("stack_T1_Z2") == "stack"
    assert dataset_split._slice_base("plain_name") is None


# --- the UI-facing warning -------------------------------------------------
#
# `group_split_warning` is a pure text function; importing its module pulls in
# Qt in-process, which the subprocess purity test above is immune to.


def test_no_warning_when_the_grouping_works():
    from src.digitalsreeni_image_annotator.controllers.io_controller import (
        group_split_warning,
    )

    names = _frames("clipA", 10) + _frames("clipB", 10)
    assert group_split_warning(names, None, 20) is None


def test_no_warning_when_no_validation_set_was_asked_for():
    from src.digitalsreeni_image_annotator.controllers.io_controller import (
        group_split_warning,
    )

    assert group_split_warning(_frames("clip", 10), None, 0) is None


def test_a_single_recording_warns_that_the_metrics_are_optimistic():
    from src.digitalsreeni_image_annotator.controllers.io_controller import (
        group_split_warning,
    )

    message = group_split_warning(_frames("clip", 10), None, 20)
    assert message is not None
    assert "optimistic" in message


def test_annotated_image_names_matches_what_the_exporter_splits():
    """The preview has to be computed over the same set the export uses, or the
    warning is about a different split than the one that happens."""
    from src.digitalsreeni_image_annotator.controllers.io_controller import (
        annotated_image_names,
    )

    all_annotations = {
        "a.png": {"cell": [{"bbox": [0, 0, 1, 1]}]},
        "b.png": {},
        # Truthy-but-empty, and therefore counted -- because the exporter
        # counts it too. Mirroring the exporter's own filter is the whole job;
        # a "better" filter here would preview a different split than the one
        # that runs.
        "c.png": {"cell": []},
    }
    assert annotated_image_names(all_annotations) == ["a.png", "c.png"]
    assert annotated_image_names(None) == []
