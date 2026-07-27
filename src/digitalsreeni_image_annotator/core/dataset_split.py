"""Group-aware train/val splitting (issue #80, ADR-044).

The split used to be keyed by the **image name**. That is correct only when
every name is an independent observation, and in this app it routinely is not:
a multi-dimensional stack contributes one name per slice (``stack_T1_Z5_C1``)
and a video one per frame (``video_F00042``). Consecutive frames of one
recording are near-identical, so a name-keyed split scatters them across train
and val by construction -- the model is validated on frames it effectively
trained on, and every reported validation metric comes back optimistic. The
numbers look better the more redundant the data is, which is precisely
backwards.

So the split key is the **group**, not the name. A group is "one source of
observations", and the whole group lands on one side.

**Groups are derived from structure, not from a model.** The primary source is
``image_slices``, already keyed by the ext-stripped base name, so the mapping is
exact and free. Embedding clusters from the curation feature (#72) can *refine*
it through :func:`merge_groups`, but they are never required: the worst leakage
-- a 200-frame video -- is fixed without a model, a GPU or a curation run, and
therefore on every path including the headless CLI.

Qt-free (ADR-041), and deliberately importing nothing from ``slice_cache``:
that module reaches ``core.image_utils``, which imports ``QImage``. The
three-line ``.names`` accessor is inlined below for the same reason
``core.slice_index`` inlines it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence

# A slice name is the ext-stripped base plus one `<DimLetter><index>` component
# per non-spatial dimension (`SliceProvider._build_index`), or `_F#####` for a
# video frame (ADR-037). Regular image names keep their extension, which is why
# the dot check below is enough to tell the two apart -- the same signal the
# exporters already use (`'_' in name and '.' not in name`).
_SLICE_SUFFIX = re.compile(r"^(?P<base>.+?)(?:_[A-Z]\d+)+$")


def _collection_names(collection: Any) -> list[str]:
    """Slice names of a collection, decoding nothing.

    ``.names`` on a ``LazySliceList``; a plain ``[(name, qimage), ...]``
    otherwise, which legacy call sites and several tests still hand in.
    """
    names = getattr(collection, "names", None)
    if names is not None:
        return list(names)
    return [name for name, _ in (collection or [])]


def _slice_base(name: str) -> str | None:
    """The stack/video base name a slice name belongs to, or ``None``.

    A best-effort fallback for names with no ``image_slices`` entry -- the CLI
    passes an empty mapping (``cli.commands._export_dispatch``) and an ``.iap``
    can carry slice names whose stack was never materialised in this session.

    Over-grouping is the safe direction here, and the non-greedy base makes the
    failure land that way: a stack literally named ``run_T1`` yields base
    ``run``, merging it with ``run_T2``. That costs some split granularity; the
    opposite error would reintroduce the leak this module exists to close.
    """
    if "." in name:
        return None
    match = _SLICE_SUFFIX.match(name)
    return match.group("base") if match else None


def derive_groups(
    names: Iterable[str], image_slices: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """``{name: group_key}`` for every name in ``names``.

    ``image_slices`` is the main window's ``{ext_stripped_base: collection}``
    mapping; it gives an exact answer with no parsing and no pixel work. Names
    it does not cover fall back to :func:`_slice_base`, and anything left is its
    own group -- a plain image is a group of one.
    """
    exact: dict[str, str] = {}
    for base, collection in (image_slices or {}).items():
        for slice_name in _collection_names(collection):
            exact[slice_name] = base

    groups: dict[str, str] = {}
    for name in names:
        groups[name] = exact.get(name) or _slice_base(name) or name
    return groups


def merge_groups(
    groups: Mapping[str, str], clusters: Iterable[Sequence[str]] | None
) -> dict[str, str]:
    """Fold near-duplicate ``clusters`` into an existing grouping.

    Union-find over the *group keys*, so the two sources compose transitively:
    if names ``a`` and ``b`` already share a stack and a cluster links ``b`` to
    ``c``, all three end up in one group. A cluster member that is not in
    ``groups`` (an image with no annotations, say) still bridges the groups on
    either side of it, which is the behaviour that makes the refinement worth
    anything.
    """
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for key in groups.values():
        find(key)
    for cluster in clusters or []:
        keys = [groups.get(name, name) for name in cluster]
        for key in keys[1:]:
            union(keys[0], key)

    return {name: find(key) for name, key in groups.items()}


def _split_by_group(
    names: list[str], groups: Mapping[str, str], val_count: int
) -> tuple[set[str], set[str]]:
    """Route whole groups into val until ``val_count`` images are held out.

    Groups are ordered by a stable MD5 of the group key -- the same device the
    name-keyed split used, so the result is reproducible across runs and
    machines (unlike ``hash()``, which is salted per process).

    A group is indivisible, so the requested count is a target rather than a
    guarantee: the last group added usually overshoots it. It is dropped again
    when that lands closer to what was asked for.
    """
    members: dict[str, list[str]] = {}
    for name in names:
        members.setdefault(groups.get(name, name), []).append(name)

    ordered = sorted(
        members, key=lambda key: hashlib.md5(key.encode("utf-8")).hexdigest()
    )

    chosen: list[str] = []
    held_out = 0
    for key in ordered:
        if chosen and held_out >= val_count:
            break
        chosen.append(key)
        held_out += len(members[key])

    if len(chosen) > 1:
        without = held_out - len(members[chosen[-1]])
        if abs(without - val_count) < abs(held_out - val_count):
            chosen.pop()

    # Never drain train. With singleton groups the count clamp already
    # guarantees this; with real groups a single large one can swallow
    # everything, and an empty train set is not a split at all.
    if len(chosen) == len(ordered) and len(ordered) > 1:
        chosen.pop()

    val = {name for key in chosen for name in members[key]}
    return set(names) - val, val


def plan_split(
    names: Iterable[str],
    val_pct: float,
    groups: Mapping[str, str] | None = None,
) -> tuple[set[str], set[str], bool]:
    """``(train, val, fell_back)`` for ``names`` at ``val_pct`` percent.

    ``fell_back`` is True when grouping was requested but could not be applied
    because everything belongs to a single group -- a project that is one video,
    typically. There the honest split does not exist: any val set shares a
    recording with train. Returning an empty val set would be the more truthful
    answer, but it makes the trainer silently skip validation and early stopping
    (ADR-028), which surfaces as a regression rather than as information. So the
    name-keyed split is used and the flag says so, leaving the UI to state
    plainly that the validation numbers will be optimistic.
    """
    ordered_names = list(names)
    total = len(ordered_names)
    if val_pct <= 0 or total < 2:
        return set(ordered_names), set(), False

    # Nearest integer, clamped so neither side is ever empty. round() is
    # half-to-even; the clamp makes that irrelevant at the boundaries.
    val_count = max(1, min(total - 1, round(total * val_pct / 100)))

    if not groups:
        return (*_split_by_group(ordered_names, {}, val_count), False)

    distinct = {groups.get(name, name) for name in ordered_names}
    if len(distinct) < 2:
        return (*_split_by_group(ordered_names, {}, val_count), True)

    return (*_split_by_group(ordered_names, groups, val_count), False)


def assign_train_val(
    image_names: Iterable[str],
    val_pct: float,
    groups: Mapping[str, str] | None = None,
) -> tuple[set[str], set[str]]:
    """Deterministically partition image names into ``(train, val)``.

    ``val_pct`` in ``[0, 100]``; 0 keeps everything in train. Without ``groups``
    this is the historical per-name split, unchanged. Re-exported from
    ``io.export_formats``, where it used to live.
    """
    train, val, _fell_back = plan_split(image_names, val_pct, groups)
    return train, val
