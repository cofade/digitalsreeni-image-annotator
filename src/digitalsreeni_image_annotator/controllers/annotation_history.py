"""Snapshot-based undo/redo history for annotations.

Each undoable edit pushes a deep copy of one image's entire per-class
annotation dict (the same structure stored at
``all_annotations[image_key]``) *before* the edit mutates it. Undo restores
a whole snapshot wholesale, which sidesteps the value-equality /
renumbering / selection-rehoming / ``segmentation_raw`` subtleties that a
fine-grained command pattern would have to reproduce (see ADR-022/024/025).

History is kept **per image key** (``current_slice or image_file_name``):
Ctrl+Z acts on the image you are looking at and never reaches back to an
image you can't see. Stacks are retained across navigation for the session
and cleared on new-project / clear-all / project-open.

Model (symmetric, no separate baseline needed). The live state lives in
``all_annotations`` outside this object; the undo stack holds prior states:

    record(before)  -> undo.append(before); redo.clear()
    undo(current)   -> redo.append(current); return undo.pop()
    redo(current)   -> undo.append(current); return redo.pop()

This class holds no Qt state and imports no PyQt, so it is unit testable in
isolation. The owning controller deep-copies snapshots in and applies a
returned snapshot back onto the live model.
"""

import copy


class AnnotationHistory:
    """Per-image-key undo/redo stacks of annotation-dict snapshots."""

    def __init__(self, max_depth=50):
        # image_key -> {"undo": [snapshot, ...], "redo": [snapshot, ...]}
        self._stacks = {}
        self._max_depth = max_depth

    def _stack(self, key):
        return self._stacks.setdefault(key, {"undo": [], "redo": []})

    def _cap(self, lst):
        while len(lst) > self._max_depth:
            lst.pop(0)

    def record(self, key, before_snapshot):
        """Push the pre-mutation snapshot; clear redo; cap depth.

        Skips the push when ``before_snapshot`` equals the current undo top
        (deep value-equality). That dedup keeps a begin/commit pair from
        recording the same state twice and drops genuine no-op edits.
        """
        stack = self._stack(key)
        if stack["undo"] and stack["undo"][-1] == before_snapshot:
            return
        stack["undo"].append(copy.deepcopy(before_snapshot))
        stack["redo"].clear()
        self._cap(stack["undo"])

    def can_undo(self, key):
        stack = self._stacks.get(key)
        return bool(stack) and bool(stack["undo"])

    def can_redo(self, key):
        stack = self._stacks.get(key)
        return bool(stack) and bool(stack["redo"])

    def undo(self, key, current_snapshot):
        """Step back one edit. Returns the snapshot to restore, or None.

        ``current_snapshot`` (the live state) is moved onto the redo stack so
        a subsequent redo can return to it.
        """
        if not self.can_undo(key):
            return None
        stack = self._stack(key)
        stack["redo"].append(copy.deepcopy(current_snapshot))
        self._cap(stack["redo"])
        return stack["undo"].pop()

    def redo(self, key, current_snapshot):
        """Step forward one edit. Returns the snapshot to restore, or None."""
        if not self.can_redo(key):
            return None
        stack = self._stack(key)
        stack["undo"].append(copy.deepcopy(current_snapshot))
        self._cap(stack["undo"])
        return stack["redo"].pop()

    def drop(self, key):
        self._stacks.pop(key, None)

    def clear(self):
        self._stacks.clear()
