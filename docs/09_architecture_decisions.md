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
