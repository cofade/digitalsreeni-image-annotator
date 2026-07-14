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
- ⚠️ Deliberately-compact widgets (DINO threshold table / phrase
  panel) don't hardcode their small font inline; the appended block
  owns it via type/objectName selectors (`ClassThresholdTable`,
  `PhraseEditorPanel …`, `#dino_phrase_hint`) so compact ≠ unscaled.
  Follow that pattern for new compact widgets.
- ⚠️ Known debt: `dino_merge_dialog.py` still carries hardcoded
  `font-size:Npx` tokens and a `color:#444` dark-mode contrast issue,
  so it doesn't scale. Tracked, not an oversight; fix when that
  dialog is next touched.

---

## ADR-021: SAM Fine-Tuning via a Custom Loop over the Ultralytics SAM2 Module

**Status**: Accepted

**Context**: Users annotating domain-specific imagery (microscopy,
medical, materials) get generic SAM masks that need heavy correction.
We want to let them fine-tune SAM 2 / 2.1 on their own annotations and
reuse the result in the existing SAM-box / SAM-points workflow
(upstream issue bnsreenu#73).

The obvious approach — mirror the YOLO trainer's `model.train(...)` —
**does not work**: Ultralytics registers only a *predictor* for SAM's
`segment` task (`SAM.task_map`), so `SAM(...).train()` raises
`NotImplementedError` (verified on ultralytics 8.4.51).

**Decision**: Fine-tune with a custom PyTorch loop that **reuses
Ultralytics' own forward path**. `SAM(...).model` is a plain
`SAM2Model` `nn.Module`; its `SAM2Predictor` exposes the forward in
reusable pieces — `get_im_features` (image encoder) and
`prompt_inference` / `_inference_features` (prompt encoder + mask
decoder). These are *not* wrapped in `inference_mode` unless reached
via the public `__call__`, so calling them directly under
`torch.enable_grad()` yields differentiable mask logits. The engine
(`training/sam_trainer.py`) adds focal+dice loss (≈20:1) + AdamW +
backward. Default freeze policy: train only `sam_mask_decoder`
(image + prompt encoders frozen); an optional flag also unfreezes the
image encoder.

Checkpoints are saved as `{"model": state_dict}` — the exact shape
Ultralytics' `_load_checkpoint` reads (it rebuilds the architecture
from the filename suffix and `load_state_dict`s the nested `model`
key). Consequently a fine-tuned file **must keep its base token in the
name** (e.g. `myrun_sam2_t.pt`), enforced by `make_custom_filename`;
`build_sam` selects the architecture by `ckpt.endswith(token)`. Every
save is round-trip-verified by reloading through `SAM(out_path)` and
running one forward — failing loudly rather than producing a file that
won't reload (cf. facebookresearch/sam2#337 key-mismatch failures).

**Alternatives considered**:
- *facebookresearch/sam2 training code* — rejected: heavy extra
  dependency overlapping Ultralytics' bundled SAM2, and its checkpoints
  need state-dict conversion to reload into our `SAM()` inference path.
- *Export dataset + train externally* — rejected as the default (less
  "integrated"), though `Prepare SAM Dataset` + folder training give a
  similar offline path for users who want it.

**Consequences**:
- ✅ No new runtime dependency; fine-tuned models drop straight into
  the existing SAM selector and inference path.
- ✅ Exposure to Ultralytics internals is confined to a few
  already-exercised predictor methods, guarded by
  `test_sam_finetuning.py::TestUltralyticsAPI` (fails on an upgrade
  that renames them).
- ⚠️ The trainer loads its **own** `SAM` instance on its `QThread`
  (it does not touch `SAMUtils._model`), and must **not** use
  `sam_utils._run_sync` (its re-entry guard is GUI-thread-local). The
  real hazard is two SAM models (resident inference + training) on one
  CUDA context, so `SAMTrainController` locks the SAM inference UI
  (tools + model selector + the fine-tune menu) for the duration —
  re-enabled in `training_finished` on both the success and error
  paths.
- ⚠️ Decoder fine-tuning is realistically GPU-only; a CPU-only box is
  hard-warned before a run (`resolve_torch_device`), and the device is
  pinned so an incompatible GPU is honoured as CPU instead of crashing.
- ⚠️ Encoder features are recomputed per epoch (bounded memory) rather
  than cached across epochs; revisit if large datasets need the speedup.
- ⚠️ **Loss must use the inference coordinate frame.** SAM2 letterboxes
  the image (`LetterBox(1024, center=False)`, pad bottom/right) and
  inference maps masks back with `ops.scale_masks(..., padding=False)`,
  which crops that padding before upsampling. The training loss therefore
  runs the decoder logits through the *same* `ops.scale_masks` before
  comparing to the GT mask — a naive `F.interpolate` over the full
  low-res mask bakes the padding into the target and the decoder learns
  masks shifted by the pad (a downward shift on non-square images, caught
  only during GUI testing because the e2e tests used square images). The
  landscape regression test (`test_landscape_no_mask_shift`) and the
  `ops.scale_masks` API-drift guard protect this.
- ℹ️ The custom loop later gained a train/val split, a no-grad validation pass
  (`val_loss`), a warmup→cosine LR schedule, and early stopping with best-checkpoint
  selection — see **ADR-028**.

---

## ADR-022: Canvas Mask Selection Unified with the Annotation List

**Status**: Accepted (issue bnsreenu#75)

**Context**: Selecting an existing annotation was only possible through the
bottom-left annotation list (already `ExtendedSelection`) or by *double*-clicking
a mask on the canvas — which immediately enters vertex-edit mode. There was no
single-click select, no box/multi-select on the image, and canvas `Delete` worked
only while in vertex-edit mode. Issue #75 asked for single-click select (without
entering edit), rubber-band box select, modifier multi-select, and multi-delete —
all directly on the canvas.

**Decision**: Add an **idle-mode selection layer** to `ImageLabel` and route it
through the *existing* annotation-list selection so delete/merge/change-class are
reused unchanged:

- **Idle activation.** Selection is live only in `_is_select_mode()` — no drawing
  tool, not editing, not SAM, no temp review. Picking any tool restores drawing.
  No new tool button (matches the user's "a single click should select" ask).
- **Gestures.** Plain click selects the smallest mask under the cursor (covers
  segmentation *and* bbox); click on empty space clears; drag draws a rubber band
  and selects every annotation whose bounds intersect it; **Shift** makes a click
  toggle and a drag additive. Double-click is unchanged (still vertex edit).
- **Ctrl stays pan.** Ctrl+drag pan (with its carefully tuned reference frame) is
  left untouched; multi-select uses Shift instead of Ctrl.
- **One selection, two surfaces.** The canvas emits
  `canvasSelectionChanged(annotations, mode)`; `AnnotationController.apply_canvas_selection`
  computes the new set (replace/add/toggle), sets `image_label.highlighted_annotations`,
  and **mirrors it onto the list** with signals blocked. `Delete` on the canvas
  reuses `delete_selected_annotations` (which reads the list selection).

**Consequences**:
- Delete / Merge / Change-Class need no new logic — they already operate on the
  list selection, which the canvas now drives.
- ⚠️ Matching between the canvas and list relies on **dict value-equality**, like
  the rest of the selection code (`image_label.annotations` is a deepcopy of
  `all_annotations`, and PyQt round-trips `UserRole` dicts as copies, so identity
  is never stable). Value-equal duplicate masks would select together — a
  pre-existing, accepted limitation. See the crosscutting "Canvas selection ↔
  list selection" section.
- ⚠️ The list mirror must block `itemSelectionChanged` while selecting, or it
  recurses back through `update_highlighted_annotations` and overwrites the set.

**Selection is rendered class-colour-independent (amendment).** The first cut
drew the selected mask in solid **red** — invisible on a red-class mask, and the
default palette assigned red as the *first* class colour. Selection is now an
overlay drawn in a final pass on top of every mask, independent of class colour
and modelled on the sibling open-garden-planner app's CAD selection: a dashed
selection-blue **bounding-box marquee** (`_SELECTION_COLOR = QColor(0, 120, 215,
220)`) plus bright opaque-blue **handle squares** at the 4 corners + 4 edge
midpoints, white-cased and fixed on-screen size (`_draw_selection_overlay` in
`widgets/image_label.py`). The handles are what make selection unmistakable
regardless of mask colour (a single thin dashed outline was too faint; an earlier
marching-ants + marquee was too busy). The handles are now grab targets for
resize/move of any selected shape (see ADR-023). The mask keeps its
class colour; the rubber-band rect uses the same blue dashed style. Separately,
the default class palette
(`core/constants.py::DEFAULT_CLASS_COLORS` / `default_class_color`) was reordered
so red is **last** (no fresh project starts on red) and muted, and the default
fill opacity dropped to `0.2` (`DEFAULT_FILL_OPACITY`) so masks don't bury the
image. Existing projects keep their persisted class colours.

---

## ADR-023: Direct-Manipulation Shape Editing on the Selection Handles

**Status**: Accepted (issue bnsreenu#40)

**Context**: `"bbox"`-keyed annotations (from COCO/YOLO import and detectors)
were **not editable at all** — `start_polygon_edit` only matches `"segmentation"`,
so double-click vertex edit skipped them. ADR-022 draws 8 handle squares around
*any* selected annotation, but they were visual-only. A first cut wired them up
for `"bbox"`-typed annotations only — but almost everything in this app is stored
as `"segmentation"` (drawn rectangles, polygons, SAM/DINO masks all are), so the
handles looked grabbable on every shape yet did nothing on the shapes users
actually have. The handles must act on **any** selected shape.

**Decision**: Wire the handles up as **direct-manipulation** resize/move of the
single selected shape, modelled on the sibling open-garden-planner app's
`ResizeHandle`. No new mode, no double-click — it works off the existing idle-mode
selection:

- **Single-shape, any kind.** Handles are draggable when exactly one annotation
  with a bounding box is selected (`_single_selected_shape()`); a multi-select
  leaves them visual. The press handler resolves to the live object
  (`_live_annotation`) and records `kind` — `"seg"` (polygon/mask) or `"bbox"`
  (box-only import) — which picks the geometry the handles drive
  (`_begin_shape_edit`).
- **Anchor-from-handle.** A corner/edge drag computes the new bounding box
  (`_resize_bbox`: replaces the dragged coordinate, opposite side fixed,
  normalised, ≥ 1px). A `"bbox"` shape sets `[x, y, w, h]` directly; a polygon
  **scales every vertex** from the old box to the new one
  (`_scale_segmentation`), so the outline resizes proportionally. Per-handle
  resize cursors match OGP (`_BBOX_HANDLE_CURSORS`; `SizeAll` over the interior).
- **Move is drag-gated.** A press inside the shape starts a *pending* move that
  promotes only once the drag clears the `3px/zoom` threshold — so a plain click
  still falls through to selection (preserving nested-mask click-through). Move
  translates the box (`[x,y,w,h]`) or all vertices (`_translate_segmentation`).
  The geometry mutates **in place** so the canvas + overlay redraw live.
- **Bbox key stays in sync.** Imported annotations carry both `segmentation` and
  `bbox`; editing the polygon recomputes the `bbox` key (`_sync_bbox_key`) so
  export/training stay consistent. Drawn shapes have no bbox key and gain none.
- **Commit / cancel.** Release clamps into the image (ADR-024 — move slides the
  intact shape back inside, resize trims/clamps) and emits `bboxEditCommitted` →
  `AnnotationController.commit_bbox_edit` (save + list rebuild + re-mirror the
  selection). Escape restores the original geometry.

**Consequences**:
- The handles you see are exactly the grab targets — `_draw_selection_overlay`
  and `_bbox_handle_at` share `_bbox_handle_points`, so visual and hit geometry
  can't drift — and now they work on every selected shape, not just imported boxes.
- Resizing a polygon **scales** it (handles drive the bounding box); reshaping a
  polygon vertex-by-vertex is still double-click vertex edit. A `"bbox"` shape
  stays rectangular by construction.
- ⚠️ The shape-drag branches sit **before** the rubber-band branch in the
  idle-mode mouse dispatch; both are gated on `_is_select_mode()` so a
  tool/edit/SAM state still wins. (Internal names keep the `bbox_edit` /
  `bboxEditCommitted` prefix — they denote editing via the bounding-box handles,
  whatever the underlying geometry.)

---

## ADR-024: Bounds Enforcement — Clamp Manual Edits, Clip Augmented Data

**Status**: Accepted (issues bnsreenu#32, bnsreenu#36)

**Context**: Annotation coordinates could be persisted outside the image
rectangle and silently poison training data. *Drawn* shapes were already safe
(`finish_polygon`/`finish_rectangle` shapely-intersect with the image boundary),
but two paths weren't: **manual edits** (polygon vertex drag; the new bbox drag)
clamped nothing, and the **Image Augmenter** wrote rotated/zoomed/flipped polygons
verbatim.

**Decision**: Add three pure helpers in `utils.py` and apply the right one per
path:

- **Clamp manual edits** with `clamp_segmentation` / `clamp_bbox` — per-coordinate
  snap into `[0, w] × [0, h]`. Per-coordinate (not a shapely cut) is deliberate:
  it **preserves the vertex count and ordering**, so a polygon being dragged never
  loses or splits points mid-edit. Applied in place at edit commit (polygon Enter;
  bbox release), persisting through the existing save-by-reference path.
- **Clip augmented data** with `clip_polygon_to_bounds` — a shapely intersection
  (largest resulting polygon; `buffer(0)` first to repair self-intersections an
  affine augmentation can introduce). Geometric trimming is correct here because an
  augmented shape genuinely extends past the frame and should be cut at the edge,
  not have stray vertices snapped onto it. A polygon left fully outside returns
  `None` and is **dropped** by the augmenter loop.

**Consequences**:
- One vocabulary, two semantics: *clamp* (cheap, count-preserving, for live edits)
  vs *clip* (exact, may drop/split, for batch augmentation). The choice is about
  whether vertex correspondence must survive, not about which is "more correct".
- ⚠️ `clip_polygon_to_bounds` can return fewer/more vertices than the input and may
  return `None`; callers must handle the drop (the augmenter `continue`s).
- The existing `finish_polygon`/`finish_rectangle` inline clips were left as-is to
  keep the diff contained; they could later delegate to `clip_polygon_to_bounds`.

---

## ADR-025: Reversible Per-Annotation Polygon Simplification (Detail %)

**Status**: Accepted (issue bnsreenu#24)

**Context**: SAM/DINO masks are stored as raw dense polygons — `_mask_to_polygon`
returns the flattened `cv2.findContours` boundary with no simplification, so a
single mask can carry hundreds of vertices, bloating label files. Issue #24 asked
for a "mask complexity — less ↔ more points" control. The point add/remove half of
#24 was already covered by the SAM-points tool; this is the remaining piece.

**Decision**: A **per-annotation, reversible Detail %** control, surfaced as a
column in the Annotations panel:

- **Detail % (1–100, 100 = raw).** `utils.simplify_polygon(raw, pct)` thins via
  Douglas-Peucker (`cv2.approxPolyDP`), binary-searching the epsilon for the
  richest polygon whose vertex count is still ≤ `round(raw_count × pct/100)`.
- **Reversible via a preserved raw.** The dense original is **lazy-captured** into
  `segmentation_raw` the first time a mask is thinned (nothing simplifies it
  before that, so the live `segmentation` *is* the raw at capture). 100 % copies
  `segmentation_raw` back into `segmentation` exactly. No edits to the SAM/DINO/
  manual accept paths were needed.
- **Two new annotation keys** (`segmentation_raw`, `detail_pct`) ride along: they
  round-trip through `.iap` for free (project save does `ann.copy()` →
  `convert_to_serializable` → JSON), and exports read only the effective
  `segmentation`, so the *simplified* polygon is what's exported. Imported/old
  annotations have neither key → handled by lazy-init.
- **Live + in place.** The change handler resolves the selected row to the live
  drawn object by value-equality (`image_label._live_annotation`, reused from
  #40), mutates `segmentation` in place, refreshes the Area cell + the row's
  UserRole, redraws, and saves. The `bbox` key (if present) is recomputed.

**The Annotations panel became a `QTableWidget`** (ID | Class | Area | Detail %),
mirroring `dialogs/dino_phrase_editor.ClassThresholdTable` (per-row spinbox via
`setCellWidget`, `SelectRows`, `NoEditTriggers`, stylesheet-only header). This
re-homes the #75 canvas↔list selection bridge onto a table:

- The annotation dict lives in **column 0's UserRole** (the value-equality marker).
- `count()/item(i)/selectedItems()` → `rowCount()/item(r, 0)/row-deduped
  selectedIndexes()`; the mirror uses **`setRangeSelected` (additive)** because
  `selectRow()` *replaces* the selection in ExtendedSelection mode and would drop
  all but the last row. `blockSignals` + value-equality are preserved verbatim.

**Consequences**:
- Closing #24 with a small, contained change: the feature is the table UI + one
  controller handler + one pure util; the accept paths are untouched.
- ✅ Fully reversible per annotation: Detail %=100 restores `segmentation_raw`
  exactly. **Exception:** reshaping a polygon with the #40 handles invalidates the
  baseline — `_clamp_edited_shape` drops `segmentation_raw` and resets
  `detail_pct=100`, so the *edited* geometry becomes the new raw (the old dense
  outline no longer describes the reshaped polygon, and a later 100 % must not
  silently revert the edit). The detail handler also re-points
  `highlighted_annotations` at the mutated object so the overlay + a subsequent
  handle drag stay coherent.
- ⚠️ The spinbox `valueChanged` is connected **after** the initial `setValue`, so
  building/rebuilding the table never fires the simplification handler.
- ⚠️ The dead `core/annotation_utils.py` still references the old QListWidget API
  but is unimported (confirmed) — left as-is to keep the diff contained.

---

## ADR-026: Snapshot-Based Undo/Redo for Annotation Edits

**Status**: Accepted

**Context**: Annotation edits (create, delete, merge, move/scale, change class,
detail %, paint, eraser, SAM/DINO accept) were all irreversible. The only safety
net was a confirmation dialog on delete and a keep/delete prompt on merge — both
of which broke flow (delete also popped a success dialog). The justification for
those dialogs was "you can't undo," so removing them required a real undo/redo.

The mutation surface is wide and subtle: every operation writes both
`image_label.annotations` (the live working copy) and `all_annotations[key]`, and
the two share inner list objects via the shallow-copy save. Annotations are
matched by **value-equality**, not identity (ADR-022/025), numbers are reassigned
on most edits (`renumber_annotations`), and Detail % carries a lazily-captured
`segmentation_raw` (ADR-025). A fine-grained command-per-operation design would
have to reproduce every one of these invariants in its undo path.

**Decision**: **Snapshot the whole per-image annotation dict** before each edit;
undo restores a snapshot wholesale. Restoring the entire dict sidesteps all the
value-equality / renumbering / selection-rehoming / `segmentation_raw`
subtleties — there is nothing to reconcile, only a deep copy to install.

- `controllers/annotation_history.AnnotationHistory` holds **per-image-key**
  undo/redo stacks (key = `current_slice or image_file_name`), so Ctrl+Z acts on
  the image on screen and never reaches an image you can't see. Stacks are
  retained across navigation and cleared on clear-all / new-project / project
  open. Depth is capped (50) and the symmetric model needs no separate baseline:
  `record(before)` pushes the pre-edit state, `undo(current)` swaps current onto
  redo and returns the popped before-state, `redo(current)` is the mirror.
- **One choke-point**, `AnnotationController.record_history()`, called *before*
  each synchronous mutation (finish polygon/rectangle, delete, merge, change
  class, eraser replace, SAM accept, DINO accept). It is **not** hooked onto
  `save_current_annotations()` — that also fires on navigation and runs *after*
  mutation, so it can neither be filtered to real edits nor capture a clean
  "before."
- **Deferred gestures** (bbox move/scale, paint stroke, polygon vertex edit)
  notify the controller only *after* mutating in place. They capture the baseline
  at gesture **start** via a new `ImageLabel.editBaselineRequested` signal →
  `capture_edit_baseline`, and push it at commit (`commit_edit_baseline`, called
  from `commit_bbox_edit`, `commit_polygon_edit`, and the `annotationsBatchSaved`
  handler). A **deep-equality dedup** in `record()` drops aborted gestures (Esc'd
  drag, empty stroke) so they leave no entry.
  - *Vertex edit also got a save-discipline fix.* Its Enter-commit historically
    only refreshed the list and relied on a later save to persist (and **Esc did
    not revert** the in-place drags). `commit_polygon_edit` now calls
    `save_current_annotations`, and Esc restores the segmentation from a snapshot
    taken at edit-mode entry — so the commit is both persisted and undoable, and
    Esc truly cancels.
- **Detail-% coalescing.** The spinbox fires `valueChanged` per step; a whole
  drag on one annotation records once (token = key + number + class), so one
  Ctrl+Z reverts the entire drag including `detail_pct` and `segmentation_raw`.
- **Shortcuts** are `QShortcut`s with `ApplicationShortcut` context (Ctrl+Z; Ctrl+Y
  and Ctrl+Shift+Z for redo) — the annotation-list `QTableWidget` would otherwise
  consume Ctrl+Z. Undo/redo are no-ops during project load, while a modal is open,
  while a text field has focus, or while a draw/edit gesture is in flight
  (`_undo_blocked`). Undo persists via `auto_save` — the net must survive reopen.

**Delete and merge dialogs removed.** With undo as the net, `delete_selected_annotations`
drops both the confirmation and the success dialog; `merge_annotations` drops the
keep/delete prompt (originals are always replaced by the union) and the success
dialog. Validation warnings stay.

**Consequences**:
- ✅ Every annotation edit is reversible; destructive ops are instant and flow-friendly.
- ✅ Robust against the value-equality/renumber/raw subtleties because it restores
  whole dicts rather than replaying operations.
- ⚠️ Memory is a bounded deep copy per edit per image (annotations are small;
  depth-capped at 50). ⚠️ Undo clears the current selection rather than trying to
  re-resolve it by value across a list rebuild — the safe, predictable choice.

---

## ADR-027: Mandatory MLflow Experiment Tracking (SAM-explicit / YOLO-native)

**Status**: Accepted (issue bnsreenu#74); amended — tracking is now a core,
always-on feature rather than an opt-in extra (see "Amendment" below).

**Context**: Training output (SAM fine-tuning and YOLO training) vanished after a
session — no record linked a saved checkpoint to its hyperparameters, no run-to-run
comparison, no persisted loss curves. The only feedback was live strings in
`TrainingInfoDialog`. Issue #74 asked for MLflow tracking. This fork's owner
decided every training run must be tracked — the app already hard-requires torch /
ultralytics / transformers, so MLflow's footprint is negligible and an opt-out only
invites "why is there no run?" confusion.

**Decision**:

- **Core dependency.** MLflow is in `install_requires` (and uncommented in
  `requirements.txt`), not an extra — a fresh `pip install` always has it. `import
  mlflow` still happens only inside the methods of `training/mlflow_tracker.py`
  (never at module top), so app startup stays fast and a *broken* install can't stop
  the GUI from launching — but tracking is never *expected* to be absent.
- **One small wrapper, `MLflowTracker`, always on.** There is no "disabled" mode and
  no user toggle. Every live mlflow call is wrapped so a tracking error logs a status
  line but **never aborts training** — pure crash-safety, not an opt-out. A separate
  `_NullTracker` no-op stands in only when a trainer is called *without* a tracker
  (direct/programmatic calls, tests); the GUI always supplies a real tracker.
- **Two integration styles, by trainer shape:**
  - **SAM** has a *custom* training loop, so it logs **explicitly** through a
    tracker passed into `SAMFineTuner.train(..., tracker=...)`. The run is
    started/logged/ended **inside `train()` on the worker thread** because MLflow
    runs are thread-bound. The controller builds the (unstarted) tracker; the
    trainer wires the tracker's status `log` to its own thread-safe
    `progress_signal` (never a direct cross-thread QTextEdit write).
  - **YOLO** uses Ultralytics' *built-in* MLflow callback — armed every run by
    setting `MLFLOW_TRACKING_URI` / `MLFLOW_EXPERIMENT_NAME` and
    `ultralytics.settings.update({"mlflow": True})`. It logs richer metrics
    (box/cls/dfl loss, mAP, the model) for free.
- **Local file store by default.** `resolve_tracking_uri()` precedence: a non-empty
  QSettings override → `<project>/mlruns` when a project is open → `<cwd>/mlruns`.
  Two cross-version/cross-platform hazards are handled at the mlflow boundary
  (`to_mlflow_uri()` + an env flag), not in the resolver (which keeps returning a
  plain path for display and directory use):
  - **Windows path → `file://` URI.** mlflow validates the URI *scheme*, so a bare
    `C:\…\mlruns` is read as scheme `c` and rejected — local tracking would silently
    degrade to untracked. `to_mlflow_uri()` converts local paths to `file://` URIs
    (genuine `http`/`sqlite`/`databricks` URIs pass through) at every mlflow call
    site: `MLflowTracker.start()`, the YOLO `MLFLOW_TRACKING_URI` env var, and the
    `mlflow ui` launch.
  - **mlflow 3.x file-store opt-out.** mlflow ≥3 raises on the local file store
    unless `MLFLOW_ALLOW_FILE_STORE=true`; we `setdefault` it before touching mlflow
    so the documented file-store default keeps working on both 2.x and 3.x without
    overriding an explicit user setting.
- **Config surface — destination only, no on/off.** A dedicated **Settings →
  Experiment Tracking** dialog (`MLflowSettingsDialog`) edits the tracking-store
  URI and experiment name (the only knobs), and an **Open MLflow UI** action shells
  out to `<python> -m mlflow ui`. The training dialogs have **no** "track this run"
  checkbox — every run is tracked.
- **Live run link + auto-open.** When a SAM run opens, `MLflowTracker.start()`
  captures `run_id`/`experiment_id` and fires a `set_run_url_callback` with the UI
  deep link (`run_ui_url()` → `http://localhost:5000/#/experiments/<id>/runs/<id>`).
  The trainer relays it via its `mlflow_run_url` signal (worker → GUI thread); the
  controller posts a **clickable link** into the Fine-Tuning Progress dialog (now a
  `QTextBrowser` with `setOpenExternalLinks`), starts the `mlflow ui` server **once**
  (`start_mlflow_ui_server`, split out of `launch_mlflow_ui`), and **opens the run
  in the browser** (deferred ~2.5 s via `QTimer` on first launch so the cold server
  is ready). All of this is best-effort and never disturbs the run.

**Amendment (always-on)**: The feature originally shipped as an opt-in extra with a
per-dialog checkbox and an `enabled` QSettings flag (matching issue #74's "optional"
framing). That was reversed: MLflow moved to `install_requires`, the checkboxes and
the `enabled` pref were removed, `MLflowTracker` lost its `enabled` parameter, and
`load/save_mlflow_prefs` now carry only `(uri, experiment)`. The lazy import and the
never-abort-training error handling remain — as robustness, not as an opt-out.

**Consequences**:
- Every training run produces an MLflow run with no user action; there is no path
  that trains untracked except a genuinely broken MLflow install (which degrades
  safely rather than crashing the run).
- ⚠️ Two logging styles (explicit for SAM, native for YOLO) means runs from the two
  trainers are organized by their respective conventions; both honor the same
  tracking URI / experiment name, but their param/metric keys differ.
- ⚠️ MLflow run thread-affinity is load-bearing for SAM: the run **must** open and
  close on the worker thread that trains, not the GUI thread that builds the tracker.

---

## ADR-028: Train/Val Split, Val-Loss, Warmup→Cosine LR & Early Stopping for Training

**Status**: Accepted (issue bnsreenu#85)

**Context**: The two trainers were asymmetric and neither reported *generalization*.
SAM fine-tuning (ADR-021) logged only a per-epoch `train_loss` over all annotated
instances — no held-out set, so the curve always trended down and couldn't reveal
overfitting (the main reason to track experiments). Its LR was fixed for the whole
run and only the last epoch's weights were saved. YOLO already got val metrics + mAP
from Ultralytics natively, but the app surfaced only a single `trainer.loss` line and
exposed none of Ultralytics' LR-schedule / early-stop knobs.

**Decision**: Give both trainers a configurable train/val split, both-loss tracking,
a linear-warmup → cosine-to-floor LR schedule, and patience-based early stopping with
best-checkpoint selection. The schedule shape is a **fixed smart default** (warmup =
first 10% of steps, cosine floor = 10% of peak); only the *peak* LR, train %, and
patience are user-editable (the literature says the peak LR matters more than the
shape).

- **Deterministic per-image split.** A new `sam_dataset.split_groups(groups,
  train_pct, seed)` reuses the YOLO export's stable-MD5 `assign_train_val` (ADR for
  #83) so SAM and YOLO split identically and reproducibly. `SampleGroup` gained a
  `name` used only as the split key. At 100% train (or a single image) the val set is
  empty and the val pass / early stopping are skipped (the UI says so; the SAM dialog
  also disables OK at 0% train). **YOLO's split stays at "Prepare Dataset" time** —
  it's baked into `images/train` vs `images/val` folders at export, so the Train
  dialog only adds schedule/early-stop knobs, never a re-export.
- **SAM val pass + both losses.** The per-image loss body was extracted into
  `_image_instance_losses(..., train=bool)` so the same forward serves training
  (`enable_grad`, backprops once per image) and a no-grad validation pass
  (`_validation_loss`, run under `net.eval()` then encoder train-mode restored). Each
  epoch logs `train_loss`, `val_loss`, and the current `lr` to MLflow and the progress
  window. YOLO surfaces its native val loss + mAP via a new `on_fit_epoch_end`
  callback (fires *after* validation, so `trainer.metrics` is populated; keys read
  defensively) — MLflow already gets them natively.
- **LR schedule.** SAM uses `torch.optim.lr_scheduler.LambdaLR` driven by a pure
  `lr_schedule.warmup_cosine_lambda(total_steps)`, stepped once per optimizer step
  (`~ceil(train_images / batch_size)` per epoch; the lambda clamps if the real count
  drifts). A checkbox reverts to constant LR. When on, YOLO forwards `cos_lr=True`,
  `lr0`, `lrf=0.1`, `warmup_epochs=round(0.1·epochs)` to Ultralytics' `train()`; when
  off it forwards `cos_lr=False`, `lrf=1.0`, `warmup_epochs=0` so both toggles' "off"
  state means a genuinely constant LR (not a linear-decay-with-warmup).
- **Early stopping + best checkpoint.** SAM uses a tiny pure `EarlyStopper(patience)`
  (default 20; `0` disables). On each val improvement the trainer snapshots the
  weights (CPU clone); `_save_and_verify(..., state=best_state)` saves that snapshot
  rather than the last epoch (falls back to the live net when there's no val /
  improvement, preserving the original behaviour). YOLO forwards `patience` and keeps
  Ultralytics' `best.pt`.

**Why pure helpers** (`lr_schedule.py`, `early_stop.py`, `split_groups`): the tricky
math/bookkeeping is unit-tested without torch, Qt, or a real run
(`test_lr_schedule.py`, `test_early_stop.py`, `test_sam_split.py`); the val/best/early
-stop wiring is covered by driving `_run_epochs` with light stubs
(`test_sam_finetuning.py::TestValPassAndBestCheckpoint`), and the YOLO passthrough by
`test_yolo_training_args.py`.

**Alternatives considered**:
- *A shared trainer base class for both paths* — rejected: the two loops are too
  different (custom PyTorch vs Ultralytics-native). Only the pure pieces +
  `assign_train_val` are shared; a unification refactor isn't warranted.
- *Exposing every schedule knob* (warmup fraction, floor, lr0 for SAM) — deferred:
  the simplified surface (peak LR + train % + patience + a schedule toggle) keeps the
  dialogs legible; the fixed 10%/10% recipe is the modern default.
- *Stepping the SAM schedule per epoch* — rejected in favour of per-optimizer-step so
  short runs still get a real warmup ramp.

**Consequences**:
- ✅ Both trainers now show generalization (val_loss / mAP) and a tracked LR curve;
  SAM saves its best-val checkpoint instead of the last epoch.
- ⚠️ SAM's saved checkpoint changing from "last" to "best-val" is a behaviour change
  (only when a val set exists); 100% train preserves the old last-epoch save.
- ⚠️ The SAM val pass recomputes encoder features for the held-out images each epoch
  (same bounded-memory tradeoff as training, ADR-021); a larger split costs more time.
- ⚠️ `on_fit_epoch_end` reads Ultralytics metric keys (`val/box_loss`,
  `metrics/mAP50(B)`, …) which vary by task/version, so every read is guarded and a
  miss just omits that field rather than breaking the run.

---

## ADR-029: Keypoint / Pose Annotation — Per-Class Schema, COCO Instance Model, 3-State Visibility

**Status**: Accepted (issue bnsreenu#35, PR-1 + PR-2 of 3)

**Context**: The app annotated polygons, rectangles, and paint masks (+ SAM/DINO) but
had no way to place **keypoints** — the primitive for pose estimation and landmark
detection. "Keypoint annotation" can mean standalone points or full COCO/YOLO-pose
instances; the maximal target was chosen: per-class ordered named keypoints + a
skeleton, one annotation = one K-point instance tied to a bounding box, COCO 3-state
visibility, and (later PRs) COCO/YOLO-pose export-import + YOLO-pose training. This PR
(PR-1) covers annotate + persist + render only.

**Decision**:

- **One instance = one annotation, flat `[x,y,v]*K` + stored `bbox`.** A pose instance
  is stored like any other annotation in `all_annotations[image][class][]`:
  `{"keypoints": [x1,y1,v1, …], "num_keypoints": <v>0 count>, "bbox": [x,y,w,h],
  "category_id", "category_name", "number"}`. The flat triple list mirrors COCO
  exactly, so it round-trips through `.iap` via `image_utils.convert_to_serializable`
  with **zero** save/load code. **Absence of a `segmentation` key is the load-bearing
  discriminator**: `calculate_area` falls to the bbox branch, the Detail-% spin
  auto-disables, and `draw_annotations` routes to the keypoint branch (added *before*
  the `bbox` branch, since an instance also carries a bbox).
- **`v` is the COCO 3-state enum** (0 = not labelled, 1 = labelled+occluded,
  2 = labelled+visible) — identical to YOLO-pose, so no remap on export. The bbox is
  **stored, not derived**, so `_annotation_bbox`, `calculate_area`, click-selection,
  and the #40 resize handles all work unchanged; `_keypoint_bounds` is a fallback only
  for imports that omit a box. `num_keypoints` is recomputed on every edit.
- **Per-class schema, not per-instance.** COCO requires all instances of a category to
  share one keypoint set, so the schema lives in `main_window.keypoint_schemas`
  (`{class_name: {"names", "skeleton", "flip_idx"}}`), **not** on the annotation. A
  class is a "pose class" iff it has a schema. The schema is embedded on each `classes[]`
  entry in `.iap` (mirroring `dino_config`'s validate-on-load robustness — malformed
  schemas are dropped with a print, never crash; old projects load unchanged with an
  empty store). `flip_idx` (h-flip partner per point) is app-only for COCO but required
  by YOLO-pose; pure validation lives in `core/keypoint_schema.py` so it's unit-testable.
- **Guided in-order placement tool.** `KeypointTool(ToolHandler)` places points in
  schema order: left-click = visible (v2), right-click / Shift+left = occluded (v1),
  auto-finish at K, Enter finishes early (remaining points padded v0), Backspace goes
  back, Esc discards. Because the manual-tool dispatch is left-button-only, the tool
  short-circuits `mousePressEvent` to accept both buttons (mirroring `sam_points`). The
  in-progress overlay renders for free (paintEvent iterates all handlers).
- **Editing reuses the #40 selection-handle path, not double-click** (double-click is
  segmentation-specific). A new edit `kind="kpt"` makes the instance **box transform the
  whole pose** (scale/translate points + box together, like a polygon's box) via the
  existing `bbox_edit` machinery — it commits via `bboxEditCommitted` → `commit_bbox_edit`,
  same as a bbox/segmentation resize. A **separate** single-point drag (`editing_keypoint`)
  moves one keypoint, and a right-click on a committed point toggles its visibility
  (2↔1) — both of *these* push an undo baseline at gesture start and commit via
  `keypointEditCommitted` → `commit_keypoint_edit` (ADR-026). Not-labelled (v=0) points
  are skipped by `_scale_keypoints`/`_translate_keypoints` and stay at `(0,0)`, since
  COCO/YOLO-pose require that invariant.
- **Guards.** Keypoint instances are rejected from **merge** (no mergeable geometry —
  they'd be silently deleted) and from cross-schema **change-class** (a normal
  annotation can't become a pose instance and vice versa; a keypoint instance only
  moves to a pose class with an identical `names` list). The schema-definition dialog
  locks the keypoint count K once instances exist (only K can corrupt them).
- **Area = bbox area (not 0).** `calculate_area` returns the stored box's `w*h` for a
  keypoint instance — deliberate, so sort-by-area behaves consistently with imported
  bbox annotations rather than dumping all poses to the end.
- **Export/import (PR-2).** COCO categories gain `keypoints`/`skeleton` (**1-based**,
  per spec) plus an app-only `flip_idx` extension key (kept **0-based** — no COCO
  precedent, and it's consumed only by our own importer / the PR-3 trainer, both
  0-based; converting it would just add a pointless round-trip). `create_coco_annotation`
  and `export_yolo_v5plus`'s per-annotation writer both check `"keypoints" in ann`
  *before* segmentation/bbox, mirroring the rendering dispatch order — a pose instance
  also carries a `bbox`, so checking bbox first would make the keypoints branch
  unreachable. YOLO-pose (`data.yaml`: `kpt_shape: [K, 3]`, `flip_idx`) is
  **dataset-global**, so `export_yolo_v5plus` refuses (`ValueError` → `QMessageBox`, via
  `_pose_export_check`) a mix of >1 distinct `(K, flip_idx)` schema or a mix of pose and
  non-pose classes among the annotations actually being written; detection is
  data-driven (based on which annotations carry `keypoints`), not solely on
  `keypoint_schemas`, so a caller that doesn't thread schemas through (e.g. the not-yet-
  migrated PR-3 training call site) still gets a correct K. All four `io.import_formats`
  entry points (`import_coco_json`, `import_yolo_v4`, `import_yolo_v5plus`, and
  `process_import_format`'s pass-through) now uniformly return
  `(annotations, image_info, keypoint_schemas)` — `{}` where nothing was recovered — so
  `io_controller.py` has one contract regardless of format. YOLO-pose import applies its
  one recovered schema (generic `kp0..kp{K-1}` names, no skeleton) to **every** class
  declared in `data.yaml`'s `names`, not just classes observed with pose-shaped lines,
  since `kpt_shape`/`flip_idx` are dataset-global. **The rebuild step in
  `io_controller.import_annotations` (`_rebuild_imported_annotation`) builds a fully
  separate dict shape for keypoint vs. non-keypoint annotations** — it must never attach
  a `None`-valued `segmentation`/`type` key to a keypoint annotation. Several
  existence-only checks (`"segmentation" in annotation`, not a truthiness/None guard) in
  `image_label.py::draw_annotations`, `image_label.py::start_polygon_edit` (the
  double-click handler, which iterates every annotation across every class), and
  `widgets/tools/eraser_tool.py` would otherwise misfire: a pose instance would render
  nothing, and any double-click anywhere on the canvas would raise
  `TypeError: 'NoneType' object is not subscriptable` as soon as one keypoint instance
  exists in the current image. `AnnotationController._keypoint_instance_bbox` was
  relocated to `utils.keypoint_instance_bbox` (delegate kept for `finish_keypoint`) so
  COCO import can reuse the same bbox-from-labelled-points fallback instead of a second
  copy.

**Why pure helpers** (`core/keypoint_schema.py`, `utils.clamp_keypoints`,
`ImageLabel._keypoint_bounds/_scale_keypoints/_translate_keypoints`): schema
validation, clamping (ADR-024 bounds enforcement extended to keypoints), and the affine
geometry are unit-tested without Qt or a model (`test_keypoint_schema.py`,
`test_keypoint_geometry.py`, `test_utils.py`); the tool logic via a fake-context
`ImageLabel` (`test_keypoint_tool.py`); the controller/persistence end-to-end on a real
offscreen window (`test_keypoint_controller.py`).

**Alternatives considered**:
- *Standalone labeled points* — rejected for this issue: the user wanted COCO/YOLO-pose,
  which needs the ordered, fixed-K, skeleton-bearing instance model.
- *Per-instance schema* — rejected: COCO mandates one schema per category; per-class
  storage enforces it and keeps instances small.
- *Double-click vertex editing (as polygons use)* — rejected: it's bound to
  `start_polygon_edit`; the #40 handle path generalizes cleanly to a `kpt` kind.
- *Graphical skeleton editor* — deferred; a list-based dialog ships first (lowest risk).

**Consequences**:
- ✅ Pose classes can be defined, annotated, edited, saved/reloaded; the data model is
  COCO/YOLO-pose-shaped, and PR-2 confirms it round-trips through both formats losslessly
  (mod point names, which YOLO-pose doesn't carry). Training (PR-3) is still pending.
- ⚠️ Finishing early pads not-yet-placed points with v=0 at the origin; they don't
  render and (in PR-1) can't be relabelled via right-click (only v>0 points are
  hit-testable). Acceptable — v=0 means "not labelled" per COCO.
- ⚠️ Old builds opening a project with keypoint instances preserve but don't render them
  (the old `if/elif` ignores the key); the schema and instances survive a save.
- ⚠️ A YOLO-pose dataset built outside this app that genuinely mixes pose and non-pose
  classes, or hand-edits per-class `kpt_shape`, is out of scope — import applies one
  recovered schema uniformly to every declared class, and export refuses to mix them.

This builds on **ADR-022/023** (canvas selection + #40 handle editing), **ADR-024**
(bounds clamping), and **ADR-026** (snapshot undo) — see those for the machinery reused.

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
