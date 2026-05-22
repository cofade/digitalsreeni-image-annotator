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

Migrating the GUI from PyQt5 to PyQt6 (same PR) eliminates the DLL conflict — verified by `tools/check_pyqt6_torch_coexistence.py` importing PyQt6 → torch → transformers → ultralytics cleanly in one process on Windows+Py3.14 (the original failure case) and the Linux/macOS test matrix.

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
1. The PyQt5 + Torch DLL load-order conflict on Windows + Python 3.14 (ADR-011) forced an entire subprocess isolation layer (`sam_worker.py`, `dino_worker.py`, `check_worker_isolation.py`) that added ~1-2 s latency per inference. The conflict only manifests on PyQt5 — Qt6's packaging reshuffle eliminates it.
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
- `tools/check_pyqt6_torch_coexistence.py` imports PyQt6 → torch → torchvision → transformers → ultralytics in that order. Run before merging on the Windows + Python 3.14 target.
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

**Decision**: Option 3. Implement `_DINOReviewEventFilter`, install it
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
- Implementation: `annotator_window.py` (`_DINOReviewEventFilter`
  class, `installEventFilter` call in `__init__`).
- Cross-cuts: documented in
  [Cross-cutting Concepts → DINO Temp Annotations](08_crosscutting_concepts.md#dino-temp-annotations--single-field-many-images).

---

## ADR-016: Decouple ImageLabel from ImageAnnotator via Signals + CanvasContext

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
  help; ADR-016 generalises "explicit interaction surface, narrow
  read surface" to all canvas ↔ orchestrator traffic.

---

## ADR-017: Per-Tool Handler Classes inside ImageLabel

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
- SAM bbox / points / magic-wand state — activates from any tool via
  the magic-wand toggle, cuts across the main tools.
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
- ✅ Phase 6's signal contract (ADR-016) is unchanged: handlers emit
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
4. Add a button in `ImageAnnotator.setup_tool_buttons()` and a branch
   in the tool-toggle handler that calls
   `self.image_label.set_active_tool("foo")`.

**Related**:
- Implementation: `widgets/tools/base.py`,
  `widgets/tools/{rectangle,polygon,paint,eraser}_tool.py`,
  `widgets/image_label.py:set_active_tool`,
  `widgets/image_label.py:paintEvent` overlay-iteration block.
- Predecessor: ADR-016 (Phase 6 signal decoupling) made this safe by
  removing the `main_window` reference; handlers don't need an
  orchestrator handle.
- Cross-cuts: documented in
  [Cross-cutting Concepts → Canvas Decoupling](08_crosscutting_concepts.md#canvas-decoupling--signals--canvascontext)
  (extended to describe the tool dispatcher).

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
