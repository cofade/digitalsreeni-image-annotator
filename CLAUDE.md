# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

DigitalSreeni Image Annotator - PyQt6 desktop app for image annotation with SAM 2 integration and multi-dimensional image support.

**Fork of**: https://github.com/bnsreenu/digitalsreeni-image-annotator

## Quick Reference

```bash
# Install
pip install -e .

# Run
python -m src.digitalsreeni_image_annotator.main
# or: digitalsreeni-image-annotator
# or: sreeni
```

## Tech Stack

Python 3.10+ | PyQt6 6.7+ | Ultralytics 8.3.27 (SAM 2) | NumPy | OpenCV | Shapely

**Test suite**: `tests/` (pytest + pytest-qt). 94 tests pass on PyQt6.

## Documentation

For detailed architecture and design information, see **[docs/](docs/)**:

- **[Building Block View](docs/05_building_block_view.md)** - Components, data model, class responsibilities
- **[Runtime View](docs/06_runtime_view.md)** - Workflows and key scenarios
- **[Cross-cutting Concepts](docs/08_crosscutting_concepts.md)** - Coordinate systems, conversions, patterns
- **[Architecture Decisions](docs/09_architecture_decisions.md)** - Why we made key choices
- **[Glossary](docs/12_glossary.md)** - Terms, acronyms, data structures

See [docs/README.md](docs/README.md) for full documentation index.

## Project Structure

```
src/digitalsreeni_image_annotator/
├── main.py                       # Entry point
├── annotator_window.py           # ImageAnnotator - thin orchestrator
├── app_settings.py               # QSettings UI prefs: ui_font_pt, dark_mode (ADR-020)
├── utils.py                      # Utility functions (calculate_area, …)
├── __init__.py                   # Public API re-exports
│
├── core/                         # constants, annotation_utils, image_utils
├── controllers/                  # 8 controllers (project, image, sam,
│                                 #   sam_train, dino, yolo, annotation,
│                                 #   class) + io_controller
├── widgets/
│   ├── image_label.py            # ImageLabel canvas widget (dispatcher)
│   ├── canvas_context.py         # CanvasContext read accessor (ADR-018)
│   └── tools/                    # Per-tool handlers (ADR-019): rectangle,
│                                 #   polygon, paint, eraser
├── inference/                    # sam_utils.py, dino_utils.py
├── training/                     # SAM fine-tuning (ADR-021): sam_trainer.py
│                                 #   (SAMFineTuner), sam_dataset.py
├── io/                           # export_formats.py, import_formats.py
├── ui/                           # menu_bar, sidebar, shortcuts, theme, stylesheets
└── dialogs/                      # Standalone tool dialogs (statistics,
                                  #   splitter, augmenter, … 16 files)
```

## Key Classes

| Class | File | Responsibility |
|-------|------|----------------|
| `ImageAnnotator` | annotator_window.py | Thin orchestrator — holds controllers, wires signals, delegates almost everything |
| `ImageLabel` | widgets/image_label.py | Canvas display, zoom/pan, event dispatch to tool handlers |
| `CanvasContext` | widgets/canvas_context.py | Narrow read view of main-window state for ImageLabel (ADR-018) |
| `ToolHandler` (+ 4 subclasses) | widgets/tools/ | Per-tool mouse/key handling (rectangle, polygon, paint, eraser) (ADR-019) |
| `ProjectController` | controllers/project_controller.py | `.iap` save/load, auto-save, `is_loading_project` guard |
| `ImageController` | controllers/image_controller.py | TIFF/CZI loading, multi-dim slicing, image/slice switching |
| `AnnotationController` | controllers/annotation_controller.py | Annotation CRUD, sort, edit-mode, finish_polygon/rectangle |
| `ClassController` | controllers/class_controller.py | Class add/delete/rename/colour/visibility |
| `SAMController` | controllers/sam_controller.py | SAM model picker, debounce, in-flight guard (ADR-013) |
| `DINOController` | controllers/dino_controller.py | DINO single/batch detection, batch review, temp-class workflow |
| `YOLOController` | controllers/yolo_controller.py | YOLO training menu + prediction wiring |
| `SAMUtils` | inference/sam_utils.py | Load SAM models (built-in + fine-tuned), run inference |
| `DINOUtils` | inference/dino_utils.py | Grounding-DINO model load + inference |
| `SAMFineTuner` | training/sam_trainer.py | Fine-tune SAM 2 decoder/encoder via custom loop over Ultralytics SAM2Model (ADR-021) |
| `SAMTrainController` | controllers/sam_train_controller.py | SAM fine-tune menu, GPU gate, training thread, selector registration |

See [Building Block View](docs/05_building_block_view.md) for detailed class documentation.

## Common Development Tasks

### Adding a New Annotation Tool

1. Add button in `ImageAnnotator.create_tool_section()`
2. Set `image_label.current_tool` on click
3. Handle mouse events in `ImageLabel` (mousePressEvent, mouseMoveEvent)
4. Render in `ImageLabel.paintEvent()`
5. Commit via `self.annotationCommitted.emit(annotation_dict)` — the
   orchestrator routes it to `AnnotationController.add_annotation_to_list`
   (see ADR-018)

### Working with Annotations

```python
# Annotation storage: dict[image_filename, list[annotation_dict]]
self.all_annotations[self.image_file_name].append({
    "segmentation": [x1, y1, x2, y2, ...],  # Polygon
    # OR "bbox": [x, y, width, height],     # Rectangle
    "category": class_name
})
```

### SAM Integration

SAM runs in-process; the Ultralytics model object lives on `SAMUtils`
and persists across calls. Inference runs on a background QThread but
the public API is synchronous — see ADR-013 in
`docs/09_architecture_decisions.md`.

```python
# Load model on first selection (downloads weights if missing, ~40-400MB)
self.sam_utils.change_sam_model("SAM 2 tiny")  # blocks UI thread via QEventLoop spin

# Run inference (also runs on worker thread, returns when done)
prediction = self.sam_utils.apply_sam_points(
    qimage,
    positive_points=[(x1, y1)],
    negative_points=[(x2, y2)]
)
# Returns: {"segmentation": [...], "score": float}
```

## Important Notes

### Platform Support
- ✅ Windows, macOS, Linux supported (PyQt6 native integration improved over PyQt5)
- Linux runtime needs libxcb-cursor0 (Qt6 requires this; was optional under Qt5)

### Critical: Project Loading
**Always check `is_loading_project` flag before saving!** Autosave during load corrupts files (v0.8.12 fix).

```python
def save_project(self):
    if self.is_loading_project:
        return  # Skip during load
    # ... save logic
```

### Coordinate Systems
- **Mouse events**: Screen coordinates → must convert to image coordinates
- **Annotations stored**: Image coordinates (unzoomed, absolute pixels)
- Account for `zoom_factor`, `offset_x`, `offset_y`

See [Cross-cutting Concepts](docs/08_crosscutting_concepts.md#coordinate-systems) for details.

### Multi-dimensional Images
- User assigns dimensions (T, Z, C, H, W) via dialog
- Slices extracted with names like `stack.tif_T0_Z5_C0`
- Each slice annotated independently
- Stored in `image_slices` dict
- TIFF axis hint: `load_tiff` reads `tifffile.series[0].axes` and pre-fills the dimension dialog; ndim≥5 had a `[-ndim:]` slice bug that produced 2560 wrong slices on a 5D `TZCYX` file — see arc42 if you touch this

See [Runtime View](docs/06_runtime_view.md#multi-dimensional-image-loading) for workflow.

### Patterns introduced in v0.9.0 (read before touching these areas)

| Area | Pattern | Why |
|------|---------|-----|
| Pan / zoom-to-cursor in scroll area | Use `event.globalPosition()` for pan; derive post-zoom offset from `viewport().width()`, not `self.width()` | Widget-local coords absorb half the pan delta as the widget shifts; `self.width()` is stale during zoom-out before layout settles. See [Pan + Zoom Reference Frames](docs/08_crosscutting_concepts.md#pan--zoom-reference-frames). |
| Dark mode contrast | No hardcoded `background:` / `color:` in widget `setStyleSheet(...)` | Hardcoded greys override `soft_dark_stylesheet.py` and punch bright boxes into the sidebar. Add a global rule first, then write the widget. See [No Hardcoded Colors Rule](docs/08_crosscutting_concepts.md#dark-mode--no-hardcoded-colors-rule). |
| DINO review state | `image_label.temp_annotations` is a single field, **not** per-image — must be re-synced from `dino_batch_results` on every image/slice switch via `_refresh_dino_temp_for_current` | Otherwise the first image's masks bleed onto every subsequent slice during navigation. See [DINO Temp Annotations](docs/08_crosscutting_concepts.md#dino-temp-annotations--single-field-many-images). |
| DINO batch over stacks | Use `_collect_dino_batch_work_items()` to flatten regular images + every loaded slice; don't iterate `self.all_images` directly | Multi-dim images appear in `all_images` as a single entry — slices live in `self.image_slices[base_name]` and were silently skipped. |
| DINO Enter/Escape during review | Application-wide `DINOReviewEventFilter`, gated on pending temp_annotations + no modal + no text input | `QListWidget` consumes Enter for `itemActivated` before `ImageLabel.keyPressEvent` sees it. See [ADR-015](docs/09_architecture_decisions.md#adr-015-application-wide-event-filter-for-dino-review-shortcuts). |
| Auto-accept dropdown | Honored by **both** `run_dino_detection_single` and `run_dino_detection_batch` | Easy to forget in the single path because the combo is labeled "batch". |
| GPU model unload | `model.cpu()` → `gc.collect()` → `torch.cuda.empty_cache()` + `ipc_collect()` + `synchronize()` — full reclaim requires app restart due to per-process CUDA context | Setting refs to None alone leaves circular refs pinned and shows zero Task Manager drop. See [Releasing Model GPU Memory](docs/08_crosscutting_concepts.md#releasing-model-gpu-memory). |
| Export image-path lookup | Exact-key match first, substring fallback only | `"bee.jpg" in "honeybee.jpg"` is True — substring-only matching writes the wrong file. See [Export Format Filename Matching](docs/08_crosscutting_concepts.md#export-format-filename-matching). |
| F2 / global shortcuts | Use `QShortcut` with `Qt.ShortcutContext.ApplicationShortcut`, not `keyPressEvent` | `QTableWidget` consumes F2 for in-cell edit before it bubbles up. |

## Development Workflow

**CRITICAL: Always use feature branches — NEVER commit directly to master.**

| Step | Action | Notes |
|------|--------|-------|
| 1 | Create branch: `git checkout -b feature/short-description` | Before any changes |
| 2 | Implement feature | Follow patterns below |
| 3 | Run manual tests | See Testing Checklist |
| 4 | Update arc42 docs if behavior changed | `docs/` — see Documentation section |
| 5 | **Run senior reviewer agent** | `.claude/agents/senior-reviewer.md` — mandatory quality gate before every PR |
| 6 | Commit: `feat: Description` or `fix: Description` | Clear, descriptive messages |
| 7 | Push & create PR | `git push origin feature/branch` |

### Testing Checklist

Before opening a PR, verify at minimum:

1. **Smoke tests pass** — `pytest tests/integration/test_smoke.py -v`. This includes the AST-based `test_annotator_window_inline_imports_are_resolvable` which catches stale relative imports inside function bodies after any module move (see ADR-016). A launch that "looks clean" is NOT sufficient — inline imports fail only when the function is called at runtime.
2. **Launch the app** — no import errors, main window renders
3. **Golden path** — perform the new feature's primary workflow end-to-end
4. **Edge cases** — empty state, cancel/escape, large images, missing model files
5. **Dark mode** — toggle and check rendering of new UI elements
6. **Save/load roundtrip** — if the feature touches `.iap` project files, save, close, reopen, verify state restored
7. **Adjacent features** — verify no regression in SAM, annotation tools, export formats
8. **Inference features** — if touching `sam_utils.py` or `dino_utils.py`, verify the model loads end-to-end (no silent load failure), returns masks/boxes, and the UI stays responsive during inference (timers, redraws, progress dialog cancels keep firing — see ADR-013)

### arc42 Documentation Update Rules

When a change affects behavior documented in `docs/`, update the docs in the same PR:

| Change Type | Update Target |
|-------------|---------------|
| New component/module | `05_building_block_view.md` — add to component table |
| Changed runtime behavior | `06_runtime_view.md` — update workflow description |
| New UI pattern or concept | `08_crosscutting_concepts.md` — add section |
| Architecture decision | `09_architecture_decisions.md` — record ADR |
| New domain term | `12_glossary.md` — add definition |

### Quality Gate — Senior Reviewer

Before every PR, run the senior reviewer agent (`.claude/agents/senior-reviewer.md`).

This is **mandatory** — the agent performs an independent end-of-implementation review:
- Reads the actual diff, not commit messages
- Ranks issues P0 (blocks merge) / P1 (should fix) / P2 (nit)
- Checks CLAUDE.md compliance (feature branches, coordinate systems, `is_loading_project` guards, DINO config persistence, in-process inference re-entrancy guards)

**Run it in the foreground** — never `run_in_background: true`. The review is a blocking quality gate: the next steps (address P0s, push, open PR) depend on its findings. Launch the agent and wait for the result before doing anything else, then iterate until clean.

Address all P0s before merging. Address P1s unless there's explicit justification.

## Known Constraints

- No type hints (gradual addition encouraged)
- Print statements instead of logging (acceptable)
- Absolute paths in projects (not portable)
- SAM 2 large crashes on limited RAM
- YOLO training not supported for multi-dimensional images

See [Risks and Technical Debt](docs/11_risks_and_technical_debt.md) for full list.

## Keyboard Shortcuts

| Global | Action |
|--------|--------|
| Ctrl+N/O/S | New/Open/Save Project |
| Ctrl+Shift+= / Ctrl+Shift+- | UI font bigger/smaller (8-24pt, persisted via QSettings) |
| Ctrl+Shift+0 | Reset UI font size |
| F1 | Help |

| Canvas | Action |
|--------|--------|
| Ctrl+Wheel | Zoom |
| Ctrl+Drag | Pan |
| Enter | Finish/Accept |
| Esc | Cancel |
| Up/Down | Navigate slices |
| -/= | Brush size |

## Quick Tips

- SAM models cache in working directory (Ultralytics)
- Recommend SAM 2 tiny/small (avoid large)
- Polygon area uses shoelace formula (utils.py)
- Export formats copy images to output directory
- Dark mode changes annotation colors for visibility
- Snake game is hidden Easter egg 🐍
