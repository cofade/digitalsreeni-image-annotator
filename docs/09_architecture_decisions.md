# Architecture Decisions

## ADR-001: GUI Framework Choice

**Status**: Superseded by [ADR-014](#adr-014-migrate-from-pyqt5-to-pyqt6)

**Original decision (historical)**: Use PyQt5 5.15.11. Chosen because the upstream project used PyQt5, PyQt5's ecosystem was more mature at the time, and migration carried risk.

**Superseding decision**: The project migrated to PyQt6 6.7+ in the same PR that introduced in-process AI inference. See [ADR-014](#adr-014-migrate-from-pyqt5-to-pyqt6) for the rationale (mainly: PyQt6 eliminated the WinError 1114 DLL load-order conflict that motivated ADR-011, unblocking the subprocess removal in ADR-013).

---

## ADR-002: Use Ultralytics for SAM Integration

**Status**: Accepted

**Context**: Need to integrate Segment Anything Model 2 for semi-automated annotation

**Decision**: Use Ultralytics library instead of direct SAM2 installation

**Rationale**:
- Simplifies SAM model loading (single line)
- Includes PyTorch dependencies
- Automatic model caching
- No manual model download required
- Supports both SAM 2.0 and SAM 2.1 variants

**Consequences**:
- ✅ Simplified installation (no separate SAM2 setup)
- ✅ Automatic model management
- ✅ Consistent API
- ⚠️ Dependency on Ultralytics release cycle

---

## ADR-003: Store Absolute Paths in Project Files

**Status**: Accepted

**Context**: Project files need to reference image locations

**Decision**: Store absolute paths to images in project JSON

**Rationale**:
- Images can be anywhere on filesystem
- No requirement to keep images with project file
- Simplifies project structure

**Consequences**:
- ✅ Flexible image locations
- ❌ Projects not portable between machines
- ❌ Moving images breaks projects

**Mitigation**: Export functions copy images to output directory

---

## ADR-004: No Automated Testing Framework

**Status**: Accepted (Technical Debt)

**Context**: Application is GUI-heavy with complex interactions

**Decision**: Rely on manual testing only

**Rationale**:
- PyQt testing requires significant setup (pytest-qt, fixtures)
- Visual nature of tool makes automated testing difficult
- Small development team
- Rapid iteration on features

**Consequences**:
- ❌ Risk of regressions
- ❌ Manual testing required for all changes
- ❌ Slower development velocity for large refactors
- ✅ Lower initial development overhead

**Future Consideration**: Add unit tests for utility functions (calculate_area, conversions)

---

## ADR-005: Disable Autosave During Project Loading

**Status**: Accepted

**Context**: Projects were getting corrupted when application terminated during loading (v0.8.9 bug)

**Decision**: Set `is_loading_project` flag to disable autosave during load

**Rationale**:
- Autosave triggered with partially loaded state corrupts file
- Loading large projects is slow, increases risk
- Simple flag prevents the issue

**Consequences**:
- ✅ Prevents project corruption
- ✅ Minimal code change
- ⚠️ Users lose autosave protection during load window

---

## ADR-006: Use Shapely for Polygon Operations

**Status**: Accepted

**Context**: Need to merge, validate, and manipulate polygon geometries

**Decision**: Use Shapely library for all polygon operations

**Rationale**:
- Industry-standard computational geometry library
- Handles invalid polygons gracefully
- Efficient union/intersection operations
- Well-tested algorithms

**Consequences**:
- ✅ Robust polygon handling
- ✅ Easy merge operations
- ✅ Automatic polygon validation
- ⚠️ Additional dependency

---

## ADR-007: Flatten Polygon Coordinates in Storage

**Status**: Accepted

**Context**: Need to store polygon annotations

**Decision**: Store as flattened list `[x1, y1, x2, y2, ...]` instead of nested `[[x1, y1], [x2, y2], ...]`

**Rationale**:
- Compatible with COCO JSON format
- Smaller file size
- Standard in annotation tools

**Consequences**:
- ✅ COCO compatibility
- ✅ Compact representation
- ⚠️ Must convert to/from paired format for some operations

---

## ADR-008: Support Multiple Export Formats

**Status**: Accepted

**Context**: Users need annotations in different formats for various ML frameworks

**Decision**: Implement exporters for COCO, YOLO, Pascal VOC, labeled images, semantic labels

**Rationale**:
- Different frameworks have different input requirements
- YOLO and COCO are most common
- Labeled images useful for visual verification
- Semantic labels needed for segmentation models

**Consequences**:
- ✅ Wide compatibility
- ✅ Flexible workflow
- ⚠️ More code to maintain
- ⚠️ Must keep up with format changes (e.g., YOLOv11)

---

## ADR-009: Use Per-Slice Annotation Storage for Multi-dimensional Images

**Status**: Accepted

**Context**: TIFF stacks and CZI files have multiple slices that need individual annotations

**Decision**: Store annotations per slice with naming convention `{filename}_T{t}_Z{z}_C{c}`

**Rationale**:
- Each slice is effectively a separate 2D image
- Simple extension of existing single-image annotation
- User can navigate and annotate independently

**Consequences**:
- ✅ Simple mental model (each slice = image)
- ✅ Reuses existing annotation code
- ⚠️ Large stacks create many entries in annotations dict
- ⚠️ No 3D annotation support

---

## ADR-010: Normalize 16-bit Images to 8-bit for Display

**Status**: Accepted

**Context**: SAM and display require 8-bit images, but microscopy often uses 16-bit

**Decision**: Normalize 16-bit to 8-bit using percentile clipping

**Rationale**:
- SAM models trained on 8-bit RGB images
- Displays only show 8-bit effectively
- Percentile clipping (2nd-98th) provides better contrast than linear

**Consequences**:
- ✅ Better visual contrast
- ✅ SAM compatibility
- ⚠️ Information loss (quantization)
- ⚠️ Different normalization per image/slice

---

## ADR-011: Run Torch-based Workers in Isolated Subprocesses

**Status**: Superseded by [ADR-013](#adr-013-in-process-inference-with-qthread-wrapping)

**Context**: Both SAM 2 (via Ultralytics) and Grounding DINO (via transformers) load PyTorch into the process. On Windows + Python 3.14, importing PyQt5 first and then loading PyTorch causes `WinError 1114` (DLL load order conflict between Qt and Torch native dependencies). The application is fundamentally PyQt5-based, so we cannot reorder these imports.

**Decision**: Run each ML model in its own subprocess script that has no PyQt5 imports — `sam_worker.py` for SAM and `dino_worker.py` for DINO. The parent GUI process speaks to each worker over stdin/stdout with JSON requests and responses.

**Rationale**:
- The DLL conflict only manifests when both libraries are loaded in the same process. Splitting them across processes avoids the issue entirely.
- Keeps the GUI responsive: heavy model loading doesn't block PyQt's event loop in the same address space.
- Lets us swap or upgrade torch/transformers/ultralytics versions without worrying about Qt interactions.
- The JSON-over-stdio protocol is simple, language-agnostic, and easy to debug — just inspect what the worker prints.

**Consequences**:
- ✅ Works reliably on Windows + Python 3.14 (the original motivating bug)
- ✅ Worker scripts are PyQt-free; they can be tested independently
- ⚠️ Per-inference subprocess spawn cost (~1-2 s startup + first model load)
- ⚠️ Need UTF-8 forced on both ends of the pipe (`PYTHONIOENCODING=utf-8` in env, `encoding="utf-8", errors="replace"` on parent) — Windows cp1252 default crashes on non-ASCII bytes in torch warnings
- ⚠️ Two near-identical worker scripts to maintain (`sam_worker.py` mirrors the pattern from `dino_worker.py`)

**Superseded by**: Migrating to PyQt6 (ADR-013) eliminated the underlying DLL conflict. The subprocess hop, JSON marshalling, and `check_worker_isolation.py` tooling were removed in the same PR.

**Related**:
- Implementation (historical): `sam_utils.py` / `sam_worker.py`, `dino_utils.py` / `dino_worker.py`
- Original SAM-only version landed in #65 (Python 3.14 support)
- DINO subprocess pattern landed alongside the DINO feature

---

## ADR-012: Lazy Model Load on Dropdown Selection

**Status**: Accepted

**Context**: Both SAM and DINO model weights are large (SAM 2 tiny ~80 MB up to large ~400 MB; Grounding DINO base ~1.9 GB) and may not exist on first run. An earlier DINO flow required an explicit "Load" button click that did the resolve-or-download dance synchronously before the user could detect anything.

**Decision**: Selecting a model from the dropdown only updates state. Actual downloads happen on first use (first Detect call). UI feedback in the status label distinguishes "Ready: <model>" (weights present) from "<model> — will download on first detection".

**Rationale**:
- Matches the existing SAM behaviour (`change_sam_model` just stores the name; download happens in the worker).
- Removes a redundant click — one fewer thing for users to discover.
- Selecting a model the user picked by mistake is now free; only confirmed Detect triggers the (potentially heavy) download.

**Consequences**:
- ✅ Consistent UX between the SAM and DINO panels
- ✅ Faster perceived startup; no spurious downloads from idle browsing
- ⚠️ First Detect after selection blocks the UI while download runs (~1 min for DINO base); the status label shows progress but the dialog is otherwise unresponsive
- ⚠️ No async download progress dialog — `huggingface_hub` prints to stdout

---

## ADR-013: In-process Inference with QThread Wrapping

**Status**: Accepted

**Context**: ADR-011 introduced a subprocess hop for every SAM and DINO inference call to work around a PyQt5 + Torch DLL load-order conflict on Windows + Python 3.14. The workaround cost a fresh `python sam_worker.py` / `dino_worker.py` spawn per inference (~1-2 s warm latency, model reloaded from disk on every call) plus a temp-PNG marshal of the image.

Migrating the GUI from PyQt5 to PyQt6 (same PR) was expected to eliminate the DLL conflict — initially verified by `tools/check_pyqt6_torch_coexistence.py` importing PyQt6 packages → torch cleanly. However, further testing (see [ADR-017](#adr-017-eager-torch-import-in-mainpy-before-qapplication-creation)) discovered that the conflict resurfaces when Qt's **platform plugin** is loaded before torch, which happens inside `QApplication()`. The practical workaround is to import torch eagerly before creating the QApplication.

**Decision**: Run SAM and DINO inference directly inside the main Python process. Keep the model objects on the `SAMUtils` / `DINOUtils` singletons so they persist across calls. Wrap each inference in a short-lived `QThread` to keep the UI thread responsive; the public API blocks the caller via a nested `QEventLoop` so call sites in `annotator_window.py` stay synchronous-looking.

**Rationale**:
- The latency win is the whole point. Subprocess spawn + Python startup + model reload was ~1-2 s every call; in-process with a cached model is ~50-500 ms.
- Threading via a nested `QEventLoop` (the `_run_sync` helper in `sam_utils.py`) lets the calling thread keep pumping events — timers, repaints, progress dialog cancels still work — while inference runs on the QThread. Existing call sites need no refactor.
- Torch and transformers are imported lazily on first inference, so app startup stays fast for users who never touch SAM/DINO.
- `_qimage_to_numpy` already exists; converting the QImage on the calling thread (not on the worker) keeps Qt objects single-threaded as required.

**Consequences**:
- ✅ Each inference is ~1-2 s faster on Windows; less dramatic on macOS/Linux but still smoother.
- ✅ Cached model survives between calls — opening a DINO model once costs once. The DINO model stays on its compute device (CPU or CUDA) for its full lifetime; the old worker shuffled CPU↔GPU per call, defeating the caching gain on PCIe. Call `DINOUtils.unload()` / `SAMUtils.unload()` to free GPU memory explicitly.
- ✅ UI stays responsive during batch DINO+SAM runs (the calling thread's `QEventLoop` still processes events).
- ✅ One source of truth per model — no more keeping `sam_utils.py` and `sam_worker.py` aligned.
- ✅ Exceptions from the inference worker (model load failures, CUDA errors) propagate out of `_run_sync` rather than being printed and silently turned into `None`. The `change_sam_model` error path in `annotator_window.py` actually catches now.
- ⚠️ A crash in torch (CUDA OOM, segfault) now takes the app down where the subprocess used to absorb it. Mitigation: inference is wrapped in `try/except` at the `_run_sync` boundary; the user sees an error dialog instead of a frozen UI.
- ⚠️ Model RAM stays resident until the user closes the app (or invokes the `unload()` method).
- ⚠️ Re-entrancy is a real hazard, addressed with belt-and-braces:
   - `_run_sync` sets a module-level `_inference_in_flight` flag and raises `InferenceBusyError` if re-entered. Same-thread re-entry can happen because the calling thread pumps its event loop while waiting (a timer fire, a click on an un-disabled widget, etc.). A `QMutex` would not help — same-thread re-acquisition deadlocks on a non-recursive mutex and is meaningless on a recursive one.
   - The known re-entry vector — the SAM debounce timer firing during an in-flight inference — is guarded at the call site: `apply_sam_prediction` in `annotator_window.py` carries its own `_sam_inference_in_flight` flag and skips. Batch DINO already disables its trigger buttons.
   - The two-layer design is intentional: the call-site flag handles the common case quietly; the `_run_sync` flag is the safety net that surfaces unknown re-entry vectors as a real exception rather than corrupting the model with concurrent `.forward()` calls (torch / ultralytics / transformers model objects are not thread-safe).

**Related**:
- Implementation: `sam_utils.py`, `dino_utils.py` (both refactored in the same PR that retires ADR-011).
- Smoke test: `tools/check_pyqt6_torch_coexistence.py` (gate that gated this whole change).
- Supersedes: [ADR-011](#adr-011-run-torch-based-workers-in-isolated-subprocesses).

---

## ADR-014: Migrate from PyQt5 to PyQt6

**Status**: Accepted

**Context**: The project shipped on PyQt5 5.15+ (ADR-001) from inception. Two pressures combined to motivate a migration:
1. The PyQt5 + Torch DLL load-order conflict on Windows + Python 3.14 (ADR-011) forced an entire subprocess isolation layer. It was hypothesised that Qt6's packaging would eliminate the conflict entirely, but real-world testing (see [ADR-017](#adr-017-eager-torch-import-in-mainpy-before-qapplication-creation)) showed the conflict persists when Qt's platform plugin is loaded before torch, regardless of whether PyQt5 or PyQt6 is the binding. The migration still removes PyQt5-specific issues (XCB plugin paths, enum namespacing drift).
2. PyQt5 is in maintenance mode. PyQt6 is the actively developed line, gets new Qt6.x features, and has better Linux native integration (XCB plugin paths in particular).

**Decision**: Migrate the GUI binding from PyQt5 (`>=5.15.0`) to PyQt6 (`>=6.7.0`). Land in a single PR alongside the subprocess-removal work (ADR-013), gated behind `tools/check_pyqt6_torch_coexistence.py` to confirm the DLL conflict is actually gone on Windows + Python 3.14.

**Rationale**:
- Two coupled changes share most of their cost (touching every file that imports PyQt5) so doing them in one PR avoids paying the migration tax twice.
- Most PyQt5→PyQt6 differences are enum namespacing (`Qt.AlignCenter` → `Qt.AlignmentFlag.AlignCenter`) and module relocations (`QAction` moves from `QtWidgets` to `QtGui`) — mechanical, codemod-able. The behavioural risk is in event APIs (`event.pos()` → `event.position()`, returning `QPointF` not `QPoint`) and a handful of removed widgets (`QDesktopWidget` → `QGuiApplication.primaryScreen()`).
- The existing test suite (65 pytest-qt tests, mostly exercising coordinate transforms) serves as the regression safety net.

**Consequences**:
- ✅ Subprocess workers retired; inference is in-process with cached models (see [ADR-013](#adr-013-in-process-inference-with-qthread-wrapping)).
- ✅ Cleaner Linux story — `libxcb-cursor0` is required by Qt 6 (was optional under Qt 5), but the platform plugin path mess is gone.
- ✅ Long support runway: PyQt6 is the maintained binding.
- ⚠️ One-time migration cost: ~30 files touched, enum namespacing across `annotator_window.py` (300+ references), `event.pos()` → `event.position()` rewrite in `image_label.py`.
- ⚠️ PyQt6 is GPLv3 / commercial like PyQt5. Switching to PySide6 (LGPL) was considered and rejected to stay close to the existing `pyqtSignal`/`pyqtSlot` API.
- ✅ All `.exec_()` call sites in `src/` migrated to `.exec()` in the v0.9.0 fix-pack — the PyQt5 alias is gone from this codebase.

**Verification**:
- `tools/check_pyqt6_torch_coexistence.py` tests both import orders. The production order (torch first, then `QApplication`) must pass. The Qt-first order is the known-failing case and is checked only to document the environment. Run before merging on the Windows + Python 3.14 target.
- 65 tests pass on the new binding under `QT_QPA_PLATFORM=offscreen`.
- Full app constructs and renders headlessly; snake-game easter egg validates the `QDesktopWidget` → `QGuiApplication.primaryScreen()` replacement.

**Related**:
- Supersedes: [ADR-001](#adr-001-gui-framework-choice).
- Unblocks: [ADR-013](#adr-013-in-process-inference-with-qthread-wrapping).

---

## ADR-015: Application-wide Event Filter for DINO Review Shortcuts

**Status**: Accepted (v0.9.0)

**Context**: During DINO batch / single-image review, the user has
to accept (Enter) or reject (Escape) pending masks. The keyboard
handling was originally in `ImageLabel.keyPressEvent`, which only
fires when the canvas has focus. In practice the user clicks slice
entries, image entries, or buttons during review — focus moves to
those widgets and Enter is consumed locally (e.g. `QListWidget`
emits `itemActivated`), never reaching the canvas. The result: Enter
and Escape silently failed during the most common review workflow.

Three options were considered:

1. **Force focus back to the canvas on every UI interaction** —
   intrusive, breaks normal navigation (Tab/Arrow keys on lists), and
   fragile because Qt's focus chain is not always predictable.
2. **Global `QShortcut` with ApplicationShortcut context** — fires
   regardless of focus but unconditionally hijacks Enter / Escape,
   breaking modal dialogs (Enter activates default button) and inline
   editing in `QLineEdit` / `QInputDialog`.
3. **Application-wide `QObject` event filter** that intercepts only
   when DINO temp_annotations are pending, and only when the focused
   widget is not a text input and no modal dialog is active.

**Decision**: Option 3. Implement `DINOReviewEventFilter`, install it
on `QApplication.instance()` once at startup, and gate the
interception on three conditions: pending DINO temp_annotations,
no active modal widget, focus not on `QLineEdit`/`QTextEdit`.

**Consequences**:
- ✅ Enter/Escape works regardless of which widget holds focus during
  DINO review.
- ✅ Modal dialogs and text-input fields are unaffected.
- ✅ Pattern is reusable for any future "review pending state" feature.
- ⚠️ Adds a per-key-press function call cost to the entire app. The
  filter short-circuits in three cheap checks before any work, so the
  overhead is negligible (≤ a few μs per keystroke).
- ⚠️ Single global filter means future review-state features must
  share it or layer additional filters; if more review modes appear,
  collapse them into a strategy registry rather than installing
  multiple top-level filters.

**Related**:
- Implementation: `DINOReviewEventFilter` class in
  `controllers/dino_controller.py` (moved there in Phase 4b);
  `installEventFilter` call in `ui/shortcuts.py:install_event_filters`,
  invoked from `ImageAnnotator.__init__` (moved there in Phase 8).
- Cross-cuts: documented in
  [Cross-cutting Concepts → DINO Temp Annotations](08_crosscutting_concepts.md#dino-temp-annotations--single-field-many-images).

---

## ADR-018: Decouple ImageLabel from ImageAnnotator via Signals + CanvasContext

**Status**: Accepted (Phase 6 of the modular refactor)

**Context**: Before Phase 6, `ImageLabel.set_main_window(main_window)`
injected the orchestrator into the canvas widget, and the widget poked
~50 sites on `main_window` directly — both reading state
(`paint_brush_size`, `class_mapping`, `current_class`, `scroll_area`,
`current_slice`, `image_file_name`) and mutating it
(`all_annotations[name] = …`, `add_class(…)`,
`update_annotation_list()`, `save_current_annotations()`,
`update_slice_list_colors()`, `schedule_sam_prediction()`,
`zoom_in()`, `enable_tools()`, etc.). The coupling made:
- ImageLabel impossible to test in isolation without a
  whole-`ImageAnnotator` fixture.
- Every controller extraction (Phases 3–5) leak through `main_window`
  delegation pass-throughs, because deleting them would break the
  widget.
- The Phase 7 per-tool split (paint / eraser / polygon / rectangle
  handler classes) impractical, because each handler would need the
  same `main_window` reference and would multiply the coupling.

Three options were considered:

1. **Protocol / duck-typed callback object** — pass a small protocol
   with the methods ImageLabel needs. Strict, type-safe, but writes
   are still synchronous direct calls; the widget still knows the
   exact method names on the orchestrator.
2. **Defer the fix** — leave `main_window` for one more phase, accept
   the debt. Cheapest, but each subsequent refactor pays the cost.
3. **Qt signals for every write + a narrow read accessor object** —
   ImageLabel emits typed signals; the orchestrator connects each to
   a controller slot during `__init__`. Reads go through a
   `CanvasContext` object with method-style accessors.

**Decision**: Option 3. ImageLabel declares ~20 `pyqtSignal`s covering
annotation lifecycle, SAM, class, tool/UI state, navigation, and
batch finalisation. Reads go via a `CanvasContext` instance passed in
through `set_context(ctx)`. The previous `set_main_window` /
`self.main_window` field is removed entirely.

The connection block lives in `ImageAnnotator._connect_image_label_signals`,
called once at the end of `__init__` after every controller exists.
`CanvasContext` wraps the main window rather than copying state, so
the source of truth stays on `ImageAnnotator` and controllers see
their writes reflected on the next read.

**Consequences**:
- ✅ ImageLabel has zero `main_window` references; signals form the
  documented public write surface at the top of the class.
- ✅ ImageLabel is now testable in isolation by connecting signals
  to stub slots; no controller fixture needed.
- ✅ Phase 7 (per-tool handlers) can carve `mousePressEvent` /
  `mouseMoveEvent` etc. without each handler needing the orchestrator.
- ✅ Signal connections are explicit and grep-able — searching for
  `il.annotationCommitted.connect` finds the single wiring site.
- ⚠️ Two parallel mechanisms (signals for writes, `CanvasContext` for
  reads) need to be kept in step. The widget's signal block and
  `_connect_image_label_signals` must stay in sync; a missing
  connection is a silent no-op write.
- ⚠️ Signal connections rely on Qt's default `AutoConnection` semantics,
  which is synchronous within a single thread. Consumers that depend
  on a write taking effect before the next read (e.g. `classRequested`
  emit followed by `_ctx.class_id(name)` read) must stay on the GUI
  thread.
- ⚠️ The synchronous batch-save signal (`annotationsBatchSaved`)
  preserves the original O(1)-save-per-batch behaviour. Replacing it
  with per-annotation save would silently turn paint commits into
  O(N) saves. Future refactors must keep the batch boundary.

**Pattern for adding a new ImageLabel → orchestrator interaction**:

1. Add a `pyqtSignal(<args>)` to `ImageLabel`.
2. Add a slot method on a controller (or main window) with matching
   signature.
3. Wire it in `_connect_image_label_signals`.
4. Replace the previous direct call site in ImageLabel with
   `self.<signal>.emit(<args>)`.

**Pattern for adding a new read accessor**:

1. Add a method on `CanvasContext` returning the value.
2. Use `self._ctx.<accessor>()` at the read site in ImageLabel.

**Related**:
- Implementation: `widgets/canvas_context.py`,
  `widgets/image_label.py` (signal block lines 42–70),
  `annotator_window.py:_connect_image_label_signals`.
- Cross-cuts: documented in
  [Cross-cutting Concepts → Canvas Decoupling](08_crosscutting_concepts.md#canvas-decoupling--signals--canvascontext).
- Predecessor pattern: ADR-015 (DINO event filter) showed that
  ImageLabel can't reliably observe global keyboard state without
  help; ADR-018 generalises "explicit interaction surface, narrow
  read surface" to all canvas ↔ orchestrator traffic.

---

## ADR-019: Per-Tool Handler Classes inside ImageLabel

**Status**: Accepted (Phase 7 of the modular refactor)

**Context**: After Phase 6, `ImageLabel` no longer held a back-reference
to `ImageAnnotator`, but it still embedded four distinct annotation
tools (polygon, rectangle, paint_brush, eraser) as if/elif branches
spread across six event methods (`mousePressEvent`, `mouseMoveEvent`,
`mouseReleaseEvent`, `mouseDoubleClickEvent`, `keyPressEvent`,
`paintEvent`). Each tool also owned helper methods on the widget
(`start_painting`, `commit_paint_annotation`, `commit_eraser_changes`,
`finish_polygon`, `cancel_current_annotation`, …). Adding a new tool
meant touching all six event methods plus the widget's helper layer,
and the file had reached ~1,240 LOC.

Three options were considered:

1. **Keep tools as if/elif branches** — cheapest, but the widget keeps
   accruing every new tool's behaviour.
2. **Per-tool widget subclass** (one `QWidget` per tool, swap on tool
   change) — too heavy: tool switches would require teardown of the
   pixmap, scroll context, zoom factor, and the SAM/DINO/edit-mode
   sub-states that cut across tool selection.
3. **Per-tool handler classes** with a thin dispatcher on the widget.
   Plain Python objects (not QObjects); the widget keeps a
   `_tools: dict[str, ToolHandler]` and routes events to
   `active_tool_handler`. Tools emit through the widget's existing
   Phase 6 signals.

**Decision**: Option 3. Each tool becomes a subclass of `ToolHandler`
in `widgets/tools/`. The contract:

- Event hooks return `True` when consumed: `on_mouse_press`,
  `on_mouse_move`, `on_mouse_release`, `on_double_click`, `on_enter`,
  `on_escape`.
- `paint_overlay(painter)` renders in-progress state (paint mask,
  eraser mask, polygon-in-progress, rectangle preview).
- `has_unsaved_state()` / `commit()` / `discard()` participate in
  the widget's `check_unsaved_changes()` dialog.
- `deactivate()` runs when the user switches away from this tool;
  default is no-op (matches the pre-Phase-7 "silently drop temp state
  mid-stroke" behaviour).

**Deliberate non-decision: state ownership.** Tool handlers contain
only *behaviour*; the temp-state fields (`current_rectangle`,
`current_annotation`, `temp_paint_mask`, `temp_eraser_mask`,
`drawing_polygon`, `drawing_rectangle`, `is_painting`, `is_erasing`)
remain on `ImageLabel`. Reason: `AnnotationController.finish_rectangle`
and `finish_polygon` (Phase 5a) read `mw.image_label.current_rectangle`
and `mw.image_label.current_annotation` directly. Moving the state
onto the handlers would have required a parallel controller refactor.
Handlers mutate `self.label.X` for those fields; pure-tool state
(e.g. future tool-internal counters) can live on the handler. See
the architectural-smell note below.

**What stays on `ImageLabel` (intentional non-extraction)**:

- Navigation (zoom, pan, offset, scaled pixmap) — cross-cutting.
- SAM bbox / points state — activates from any tool via the SAM-box /
  SAM-points toggles, cuts across the main tools.
- Polygon edit mode (`editing_polygon`, `handle_editing_click`,
  `handle_editing_move`, `draw_editing_polygon`) — modal state
  orthogonal to tool selection; sets `current_tool = None` while
  active. Promoting this to a handler would tangle the modal flow.
- DINO `temp_annotations` + `accept_temp_annotations` —
  cross-cutting; already touched by ADR-015's event filter.
- `draw_tool_size_indicator` — small enough that splitting it across
  paint/eraser handlers buys nothing.

**`paintEvent` overlay pass**. Iterates **all** handlers'
`paint_overlay()`, not just the active one. Reason: pre-Phase-7 the
temp paint mask, temp eraser mask, and polygon-in-progress rendered
whenever their state was populated, regardless of `current_tool`.
Each handler's `paint_overlay` short-circuits when its state is empty,
so the iteration is cheap and the user can switch tools mid-stroke
without losing visual feedback.

**Consequences**:
- ✅ `image_label.py` shrinks from 1,239 to ~960 LOC. Adding a new
  tool now means: create one file in `widgets/tools/`, register it
  in `_tools`, wire a button in `annotator_window.py`. No event-method
  edits.
- ✅ Each tool can be unit-tested by instantiating the handler with
  a stub `label` carrying signals and `_ctx` — no controller fixture
  needed.
- ✅ Phase 6's signal contract (ADR-018) is unchanged: handlers emit
  via `self.label.<signal>.emit(...)`.
- ⚠️ **State leak across the widget boundary.** Handlers reach into
  `self.label.X` for state. The contract drifts toward "handler is a
  namespaced function bag." Mitigation: revisit if/when controllers
  are updated to ask the handler (e.g. `polygon_tool.points()`)
  instead of reading the widget's field.
- ⚠️ `deactivate()` is no-op by default. If you make it
  `discard()` later, audit the three call sites that still write
  `current_tool = None` directly (`ImageLabel.clear()`,
  `ImageLabel.start_polygon_edit`, three locations in
  `SAMController`) — they bypass `set_active_tool` and therefore the
  hook.
- ⚠️ `check_unsaved_changes` now iterates all handlers, not just
  paint/eraser. Polygon participates via `has_unsaved_state() = len > 2`
  (sub-3-point polygons are silently discarded on switch — they
  can't be saved anyway).

**Pattern for adding a new mouse-driven tool**:

1. Create `widgets/tools/foo_tool.py` with `class FooTool(ToolHandler):`.
2. Override the event hooks you need; emit via
   `self.label.<signal>.emit(...)` and read via `self.label._ctx.X()`.
3. Register in `ImageLabel.__init__`'s `_tools = {…, "foo": FooTool(self)}`.
4. Add a button in `ui/sidebar.py:build_sidebar` next to the existing
   tool buttons, register it in `window.tool_group`, and connect
   `clicked` to `window.toggle_tool`. Then add a branch in
   `ImageAnnotator.toggle_tool` that calls
   `self.image_label.set_active_tool("foo")` for that button (since
   Phase 8 the UI building lives in `ui/sidebar.py`, not on the
   orchestrator).

**Related**:
- Implementation: `widgets/tools/base.py`,
  `widgets/tools/{rectangle,polygon,paint,eraser}_tool.py`,
  `widgets/image_label.py:set_active_tool`,
  `widgets/image_label.py:paintEvent` overlay-iteration block.
- Predecessor: ADR-018 (Phase 6 signal decoupling) made this safe by
  removing the `main_window` reference; handlers don't need an
  orchestrator handle.
- Cross-cuts: documented in
  [Cross-cutting Concepts → Canvas Decoupling](08_crosscutting_concepts.md#canvas-decoupling--signals--canvascontext)
  (extended to describe the tool dispatcher).

---

## ADR-016: Static AST Inspection of Inline Imports as Quality Gate for Refactor PRs

**Status**: Accepted

**Context**: During Phase 1 of the modular refactoring (2025-06-10), 25 modules were moved into `core/`, `dialogs/`, `inference/`, `io/`, `ui/`, `widgets/` subpackages. The smoke tests (`test_smoke.py`) verified that every module could be imported at top-level. All 30 smoke tests passed. However, four stale **inline imports** inside method bodies were missed:

```python
# annotator_window.py — inside function bodies, NOT top-level
from .dino_utils import GDINO_MODEL_PATHS        # moved to .inference.dino_utils
from .annotation_statistics import ...           # moved to .dialogs.annotation_statistics
from .project_details import ...                 # moved to .dialogs.project_details
from .project_search import ...                  # moved to .dialogs.project_search
```

These imports were deferred until the specific UI action triggered the function (e.g. picking a DINO model from the dropdown). The smoke tests, which only import modules, never execute function bodies and therefore never resolved the inline `from .dino_utils` reference. The bug surfaced only in manual QA when selecting a DINO model.

**Decision**: Add a static AST analysis test (`test_annotator_window_inline_imports_are_resolvable`) that parses `annotator_window.py`, extracts every bare relative import (`from .module`), and asserts the module still exists in the package root. The test fails with the exact line number for any stale import, preventing silent runtime-only regressions from reaching CI.

**Rationale**:
- Top-level import rewrites are mechanical and easy to verify via module import.
- Inline imports inside method bodies are invisible to module-level import tests.
- Manual QA is the fallback for behaviour, not for mechanical import correctness.
- AST inspection is cheap (~1 ms), zero false positives for this codebase, and runs in every CI build along with smoke tests.

**Consequences**:
- 🛑 Regression now impossible: the 30th smoke test would have failed the PR before merge.
- 🔧 No runtime cost — purely static analysis.
- ⚠️ Only covers `annotator_window.py`. If other files use the same inline-import pattern, the test should be generalized (or each file that contains inline imports gets its own AST check). In this codebase, `annotator_window.py` is the only file with significant inline imports.
- ⚠️ Doesn't catch dynamic imports (`__import__`, `importlib.import_module`), but we don't use those.

**Related**:
- Implementation: `tests/integration/test_smoke.py` (`test_annotator_window_inline_imports_are_resolvable`).
- Cross-cuts: `CLAUDE.md` "Testing Checklist" updated to reference this test as a mandatory CI gate.

---

## ADR-017: Eager Torch Import in `main.py` before `QApplication` Creation

**Status**: Accepted

**Context**: ADR-011 and ADR-014 both discussed a DLL load-order conflict on Windows when PyQt and PyTorch share a process. The conflict was first observed with PyQt5 (ADR-011) and later claimed to be resolved by migrating to PyQt6 (ADR-014):

> "Qt6's packaging reshuffle eliminates it." — ADR-014
>
> "...verified by `tools/check_pyqt6_torch_coexistence.py` importing PyQt6 → torch → transformers → ultralytics cleanly in one process..." — ADR-013

This claim was based on testing at the time, but it tested the **wrong order**: importing PyQt6 *packages* before torch works even in Qt5. The actual failure mode is triggered only when Qt's **native platform plugin** is loaded, which happens inside `QApplication.__init__()`, not at `import PyQt6`. The earlier verification script did not call `QApplication()`, so it never exercised the real failure path.

Real-world testing with `torch 2.11.0+cu126 + PyQt6 6.10.2 + Python 3.14.2` on Windows 11 shows the conflict **still surfaces** when Qt's platform DLLs (e.g. `qwindows.dll`) are loaded BEFORE torch's `c10.dll`. The error is `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed`.

**Root cause analysis**: Qt and torch both ship native DLLs that load into the same process. On Windows the DLL load order and address-space layout matter. When Qt's platform plugin claims certain memory slots or loads conflicting CRT libraries before torch does, torch's `c10.dll` init fails. The conflict is NOT between PyQt5 and torch per se — it is between Qt platform plugins and torch, regardless of whether the binding is PyQt5 or PyQt6.

**Decision**: Two complementary changes:

1. In `main.py`, eagerly `import torch` (with an `ImportError` fallback) **before** importing `QApplication` and creating the app. This ensures torch's DLLs claim their slot first.
2. In `__init__.py`, replace eager toplevel imports of `annotator_window`, `image_label`, and `sam_utils` with a `__getattr__`-based lazy loader. The package init runs before `main.py` when launched via the `sreeni` console script (`digitalsreeni_image_annotator.main:main`). If `__init__.py` eagerly imports modules that transitively import PyQt6 (e.g. `annotator_window`), Qt loads first and the `import torch` in `main.py` crashes with the same WinError 1114. Lazy loading defers the Qt import until someone actually accesses `pkg.ImageAnnotator`, which only happens after the torch-first guard has run.

**Verification**:
- `tools/check_pyqt6_torch_coexistence.py` now tests both orders:
  1. `torch` → `QApplication` (production order) — **PASS**.
  2. `QApplication` → `torch` (the claimed-safe order) — **FAIL** on Windows with torch 2.11.0.
- Exit code 0 means production order works; exit code 1 means even torch-first fails and subprocess isolation (ADR-011) must be restored.
- Smoke test `test_public_api_exports` passes: `__getattr__` correctly resolves all five public names.

**Consequences**:
- ✅ SAM and DINO model loading works on Windows + Python 3.14 + PyQt6 without subprocess overhead.
- ✅ App startup cost is negligible — torch import adds ~0.5-1 s before the splash window appears, which is acceptable for a desktop annotation tool.
- ⚠️ `tests/integration/test_smoke.py` cannot import `main.py` because the pytest-qt test process already has Qt loaded; importing torch afterward triggers the same WinError 1114. `main.py` is therefore **excluded** from the module-import list and is validated by CLI smoke tests instead.
- ⚠️ Future Qt upgrades may change DLL packaging and make this unnecessary, but `check_pyqt6_torch_coexistence.py` will detect that automatically.
- ⚠️ Any new public name added to `__init__.py` must also be wired through `__getattr__` or it will transitively pull in PyQt6 and break the torch-first guard.

**Related**:
- Supersedes (in spirit): ADR-014's claim that PyQt6 eliminates the conflict.
- Unblocks: ADR-013 in-process inference on the affected Windows environment.
- Implementation: `src/digitalsreeni_image_annotator/main.py`.
- Gate: `tools/check_pyqt6_torch_coexistence.py`.

## ADR-020: App-Global UI Preferences via QSettings; Canvas Overlays Scale with `ui_font_pt`

**Status**: Accepted

**Context**: The low-vision accessibility feature (continuous UI font
zoom, 8–24pt) needed (a) the chosen size to survive app restarts and
(b) canvas overlay elements — annotation labels, SAM point markers,
pen widths — to grow with the setting. UI preferences were previously
reset on every launch, and the `.iap` project file was the only
persistence mechanism in the app.

**Decision**:
1. Introduce the app's first QSettings usage
   (`QSettings("DigitalSreeni", "ImageAnnotator")`, module
   `app_settings.py`) for `ui/font_pt` and `ui/dark_mode`. These are
   per-user preferences, so they do **not** go into the `.iap` file —
   a project opened by a different user must not impose a font size.
2. A single integer `ImageAnnotator.ui_font_pt` is the source of
   truth; the named presets and the step shortcuts both funnel
   through `theme.set_font_pt` (clamp → apply → persist → menu sync).
3. Canvas overlay sizes derive from `ui_scale = ui_font_pt / 10.0`
   (10 = the legacy default, so the default renders pixel-identical
   to the pre-feature code). `ImageLabel` receives the value via a
   plain setter from `apply_theme_and_font`, not via CanvasContext —
   consistent with the existing direct `image_label.setFont` call,
   and avoids a paint-before-context-set window.

**Alternatives considered**:
- Storing prefs in the `.iap` file — rejected: project files are
  shared artifacts; accessibility settings are personal.
- Templating the static stylesheets per font size — rejected:
  appended QSS override rules (later rules win at equal specificity)
  achieve the same with zero churn in the two stylesheet strings.

**Consequences**:
- ✅ Font size and dark mode persist across restarts.
- ✅ Tests stay hermetic: every `app_settings` function accepts an
  injectable `QSettings` (INI temp file) instance.
- ⚠️ Any new scalable UI metric should use `ImageLabel._pen_w` /
  `_overlay_font` or the appended-override block in
  `theme.apply_theme_and_font` — hardcoded px values won't follow the
  setting (see "UI Font Zoom" in `08_crosscutting_concepts.md`).
- ⚠️ Known debt: standalone dialogs still carry hardcoded
  `font-size:Npx` tokens (`dino_phrase_editor.py`,
  `dino_merge_dialog.py` — the latter also a `color:#444` dark-mode
  contrast issue) and therefore don't scale. Tracked, not an
  oversight; fix when those dialogs are next touched.

---

## Decisions Under Consideration

### Consider pytest-qt for Utility Testing

**Status**: Under Consideration

**Proposal**: Add unit tests for non-GUI utilities (calculate_area, coordinate conversions, export functions)

**Pros**:
- Catch regressions in utility functions
- Build confidence for refactoring
- Document expected behavior

**Cons**:
- Setup overhead
- Maintenance burden
- May not catch most bugs (which are in GUI)

---

### Consider Relative Paths with Image Copying

**Status**: Under Consideration

**Proposal**: Copy images to project folder, store relative paths

**Pros**:
- Portable projects
- Self-contained

**Cons**:
- Disk space duplication
- Slow for large image sets
- Export already copies images
