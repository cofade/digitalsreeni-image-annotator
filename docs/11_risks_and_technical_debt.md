# Risks and Technical Debt

## Technical Risks

### Linux Platform Support

**Risk Level**: Medium

**Description**: Application has limited testing on Linux, known XCB plugin issues

**Impact**:
- May not work correctly on Linux systems
- Potential crashes or rendering issues
- Limits user base

**Mitigation**:
- Environment variable workaround in main.py (removes `QT_QPA_PLATFORM_PLUGIN_PATH`)
- Document Windows/macOS as primary platforms
- Community testing and feedback

**Future Action**: Dedicated Linux testing and fixes

---

### SAM 2 Large Model Crashes

**Risk Level**: Medium

**Description**: SAM 2 large model can crash application on systems with limited RAM

**Impact**:
- Application termination
- Loss of unsaved work
- Poor user experience

**Mitigation**:
- Documentation recommends tiny/small models
- UI warns about large model
- Autosave reduces data loss
- Out-of-memory on model load now shows an actionable "pick a smaller model"
  dialog instead of a generic error (`core/torch_utils._is_oom` +
  `SAMController.change_sam_model`, issue #34)

**Future Action**:
- Add RAM detection and warning
- Catch OOM exceptions gracefully

---

### Project File Portability

**Status**: Resolved (#42, ADR-033)

**Risk Level**: Low-Medium (historical)

**Description**: ~~Projects store absolute paths, not portable between machines.~~
`.iap` now stores portable `image_paths_rel` (POSIX separators) alongside the absolutes;
`resolve_image_path()` resolves relative-first, so a moved or shared project folder opens
without a missing-images prompt. v1 projects still resolve via the `images/` convention.

**Original impact** (pre-#42):
- Cannot share projects easily
- Moving images breaks projects
- Collaboration difficult

**Mitigation**:
- Export functions copy images
- Users can manually update paths in JSON

**Future Action**: Consider relative paths or image embedding option

---

### Large Image Memory Usage

**Risk Level**: Medium

**Description**: Loading very large images or many slices can exhaust memory

**Impact**:
- Application slowdown
- Potential crashes
- Poor performance

**Mitigation**:
- Slice-by-slice loading for multi-dimensional images
- Image downsampling for display (future)
- Lazy loading (future)

**Current Limitation**: All slices loaded into memory

---

## Technical Debt

### Low Test Coverage of Interactive Paths

**Debt Level**: Medium

**Description**: A pytest + pytest-qt suite of 94 tests now exists
(boot smoke, coordinate conversions, export-format round-trips,
utility functions). Coverage is ~15% by line — the gap is the
canvas event flow (mouse events → tool handler → signal emission →
controller slot) and the SAM/DINO/YOLO inference paths.

**Impact**:
- Phase 6/7/8 refactors had to lean on manual QA checklists for the
  canvas flow because no automated test exercises it end-to-end.
- Inference paths are exercised only via the smoke boot, not under
  real model loads (those would slow CI prohibitively).

**Effort to Resolve**: Medium

**Priority**: Medium

**Plan**:
1. Per-tool unit tests under `widgets/tools/` — each handler can be
   tested by instantiating with a stub `label` carrying signals
   and a fake `CanvasContext`, then feeding `QMouseEvent`s.
2. Integration test that loads a tiny project, draws a polygon,
   asserts the `.iap` round-trip restores state.
3. Mock SAMUtils / DINOUtils inference returns to exercise the
   controller signal paths without needing model weights.

---

### Limited Coverage — Inline Imports Not Caught by Module Tests

**Debt Level**: Medium

**Description**: Smoke tests verify modules import cleanly at top-level, but inline `from .module` imports inside function bodies are deferred and only fail when the function is called. Phase 1 modular refactoring moved 25 modules; four stale inline imports (`from .dino_utils`, `.annotation_statistics`, `.project_details`, `.project_search`) were missed and only surfaced in manual QA.

**Impact**:
- Subpackage refactor PRs require functional QA paths (not just module import CI) to verify inline imports
- Silent regressions until user clicks the specific button/dialog that triggers the stale import

**Mitigation**:
- Added AST-based static smoke test (ADR-016) that parses `annotator_window.py` and asserts every bare relative import resolves to an existing module in the package root
- The test now catches inline import drift in CI before merge

**Future Action**: Extend the AST check to any other file that uses inline deferred imports (currently only `annotator_window.py` has them).

---

### Inconsistent Error Handling

**Status**: ✅ Resolved with a written convention (issue #34)

**Debt Level**: Medium (historical)

**Resolution**: A single error-handling convention now governs the codebase —
core/inference/io/training raise; controllers/dialogs catch, `logger.exception`,
and surface a `QMessageBox`; catch the narrowest type; never `pass` silently;
bare `except:` banned. Seven silent `except: pass` sites were fixed and the one
bare `except:` removed. See ADR-031 and the Error-Handling Convention in
[docs/08](08_crosscutting_concepts.md#error-handling-convention-issue-34).

**Description (historical)**: Mix of exceptions, return values, and UI warnings

**Examples**:
```python
# Some functions raise exceptions
raise ValueError("Invalid dimension")

# Some show message boxes
QMessageBox.warning(self, "Error", "...")

# Some return None
return None
```

**Impact**:
- Inconsistent user experience
- Hard to predict error behavior
- Difficult to add global error handling

**Effort to Resolve**: Medium (weeks)

**Priority**: Low

**Plan**: Standardize on exception-based approach with top-level handler

---

### Print Statements for Logging

**Status**: ✅ Resolved (issue #33)

**Description**: Historically used `print()` instead of a logging framework.
All ~307 `print()` calls and 12 `traceback.print_exc()` sites in `src/` were
migrated to the stdlib `logging` module: one package-level logger tree rooted
at `digitalsreeni_image_annotator`, configured once in
`core/logging_config.py`, with a `--debug` / `IMAGE_ANNOTATOR_DEBUG` level
switch. `print()` is now banned in `src/` (ADR-030). See the
"Logging and Debug Output" section in
[docs/08](08_crosscutting_concepts.md#logging-and-debug-output).

**Plan**: Replace with `logging` module

---

### Tight Coupling Between ImageAnnotator and ImageLabel — Resolved (Phase 6)

**Status**: Resolved. `ImageLabel.main_window` and `set_main_window()`
were removed; every write path is now a `pyqtSignal` emission and every
read goes through a narrow `CanvasContext` accessor.

**Pattern**: see `widgets/canvas_context.py` and
`ImageAnnotator._connect_image_label_signals`. ImageLabel emits ~20
signals (annotation lifecycle, SAM, class, tool/UI state, navigation);
the orchestrator wires each to the matching controller slot.

**ADR**: see ADR-018 in `09_architecture_decisions.md`.

---

### Duplicate Code in Export Functions

**Debt Level**: Low

**Description**: Export formats share similar code (image copying, directory creation)

**Impact**:
- Bug fixes must be applied multiple times
- Inconsistent behavior across formats
- More maintenance

**Effort to Resolve**: Low (extract common functions)

**Priority**: Low

---

### No Type Hints

**Debt Level**: Medium

**Description**: Python code lacks type hints

**Impact**:
- No static type checking
- Harder to understand function contracts
- More runtime errors

**Effort to Resolve**: High (add gradually)

**Priority**: Low

**Plan**: Add type hints to new code, gradually backfill

---

### Hardcoded UI Strings

**Debt Level**: Low

**Description**: No internationalization (i18n) support

**Impact**:
- Cannot translate to other languages
- Limits international user base

**Effort to Resolve**: Medium (Qt has i18n support)

**Priority**: Very Low (no current demand)

---

## Known Issues

### YOLO Training Not Supported for Multi-dimensional Images

**Status**: Known Limitation

**Description**: YOLO training only works with single images, not TIFF/CZI slices

**Workaround**: Export slices as individual images first

**Priority**: Low (niche use case)

---

### Annotation Merge Only Works for Connected Regions

**Status**: Known Limitation

**Description**: Merge tool requires annotations to overlap or touch

**Workaround**: Use paint brush to connect regions first

**Priority**: Low

---

### Keypoint / Pose Constraints (issue #35)

**Status**: Known Limitations (ADR-029)

**Description**:
- The keypoint count **K is locked** once a pose class has instances (changing K
  would corrupt existing instances). Renaming points / editing the skeleton / flip
  stays allowed.
- A schema is **per class** (the COCO rule); all instances of a class share it.
- A point set to *not labelled* (v=0) via "finish early" doesn't render and can't be
  relabelled with a right-click in PR-1 (only v>0 points are hit-testable).
- ~~**Defining a schema on a class that already holds normal (polygon/bbox) annotations
  is unguarded**~~ **— Resolved (#44).** The UI now blocks *new* mixing in both
  directions (schema-on-plain-class; shape/SAM-tool-on-pose-class;
  pose-class-selection-while-a-tool-is-active; DINO skips pose classes); see ADR-029
  Guards. Legacy-mixed classes still load/render/save, and `_pose_export_check` remains
  the export backstop.
- **Forthcoming** (PR-2/PR-3): YOLO-pose export requires a **single `kpt_shape` per
  dataset**, so a project mixing pose classes of different K (or pose + non-pose) can't
  export to YOLO-pose (COCO has no such limit). YOLO-pose *training* stays unsupported
  for multi-dimensional stacks (same constraint as detect/segment training).

**Priority**: Low (documented constraints, not bugs)

---

### SAM Point Mode Requires Manual Confirmation

**Status**: By Design

**Description**: User must press Enter to accept SAM prediction

**Rationale**: Allows user to add more points or reject prediction

**Priority**: N/A (intentional)

---

### Autosave Doesn't Ask for File Location

**Status**: Resolved (#41, ADR-032)

**Description**: ~~Autosave only works after first manual save.~~ Before the first save,
`auto_save()` now writes a silent recovery snapshot (no dialog) that the app offers to
restore on next launch; a real save clears it. New projects are protected from the first
mutation.

---

## Upstream Fork Divergence

**Risk Level**: Medium

**Description**: This is a fork of https://github.com/bnsreenu/digitalsreeni-image-annotator

**Impact**:
- May miss upstream features
- May miss upstream bug fixes
- Merge conflicts on updates

**Mitigation**:
- Document fork-specific changes
- Periodically review upstream
- Consider contributing changes back

**Current Fork-Specific Changes** (derived from the merge history and ADR index):
- PyQt6 migration replacing PyQt5 (ADR-014), with a torch-before-Qt DLL
  load-order guard in `main.py` (ADR-017).
- In-process SAM 2 / Grounding-DINO inference on a `QThread` with a re-entrancy
  guard, replacing the old subprocess workers (ADR-013).
- Grounding-DINO text-prompted detection — single image and batch — with an
  Enter/Escape review-and-accept overlay.
- SAM 2 fine-tuning via a custom loop over the Ultralytics SAM2 module (ADR-021),
  with always-on MLflow experiment tracking (ADR-027).
- YOLO training + prediction for detection, segmentation, and pose (issue #35).
- Keypoint / pose annotation: per-class named schema + skeleton (COCO instance
  model), with COCO-keypoints and YOLO-pose export/import (ADR-029).
- Undo / redo via per-image annotation snapshots (ADR-026).
- Canvas selection unified with the annotations table + handle-based shape
  editing and vertex editing (ADR-022 / 023 / 025), with bounds clamping and
  augmentation clipping (ADR-024).
- Modular architecture: thin `ImageAnnotator` orchestrator + per-responsibility
  controllers + per-tool handlers (ADR-018 / 019).
- Central stdlib `logging` framework (ADR-030) and a written error-handling
  convention (ADR-031).
- A pytest + pytest-qt automated test suite run in CI on 3 OS × Python
  3.10-3.14, superseding the original manual-testing-only decision (ADR-004).

---

## Security Considerations

### No Input Validation on JSON Loading

**Risk Level**: Low

**Description**: Project JSON files loaded without strict schema validation

**Impact**:
- Malformed files can crash application
- Potential for malicious project files

**Mitigation**:
- Projects are local files (user-controlled)
- Try-catch around JSON loading

**Priority**: Low (desktop app, local files)

---

### Arbitrary File Paths in Projects

**Risk Level**: Low

**Description**: Project files can reference any file path

**Impact**:
- Could load unintended files
- Path traversal (theoretical)

**Mitigation**:
- Desktop app (user has filesystem access anyway)
- File existence checks before loading

**Priority**: Very Low
