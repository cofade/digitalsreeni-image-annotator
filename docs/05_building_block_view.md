# Building Block View

## Level 1: System Overview

```
┌─────────────────────────────────────────────┐
│   DigitalSreeni Image Annotator            │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │   GUI    │  │  SAM 2   │  │  YOLO    │ │
│  │ (PyQt6)  │  │(Ultraly.)│  │ Trainer  │ │
│  └──────────┘  └──────────┘  └──────────┘ │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │   Image Processing                   │  │
│  │   (NumPy, OpenCV, Shapely)          │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
         │                  │
         ▼                  ▼
   File System        SAM Model Cache
```

## Level 2: Main Components

### Core Application

```
src/digitalsreeni_image_annotator/
├── main.py                        # Entry point, initializes QApplication
├── annotator_window.py            # ImageAnnotator - main window
├── utils.py                       # Utility functions
│
├── core/                          # Shared constants and helpers
│   ├── constants.py
│   └── annotation_utils.py
│
├── widgets/                       # Custom display widgets
│   └── image_label.py             # ImageLabel - display, zoom, annotation interaction
│
├── inference/                     # AI model management (SAM / DINO)
│   ├── sam_utils.py               # SAMUtils - SAM model loading and inference
│   └── dino_utils.py              # DINOUtils - DINO detection + SAM-to-mask
│
├── io/                            # Import / export format handlers
│   ├── export_formats.py          # COCO, YOLO, Pascal VOC, semantic / labeled images
│   └── import_formats.py          # COCO and YOLO importers
│
├── ui/                            # Stylesheets for light / dark themes
│   ├── default_stylesheet.py
│   └── soft_dark_stylesheet.py
│
└── dialogs/                       # Standalone tool windows
    ├── annotation_statistics.py
    ├── coco_json_combiner.py
    ├── dataset_splitter.py
    ├── dicom_converter.py
    ├── dino_merge_dialog.py
    ├── dino_phrase_editor.py
    ├── help_window.py
    ├── image_augmenter.py
    ├── image_patcher.py
    ├── project_details.py
    ├── project_search.py
    ├── slice_registration.py
    ├── snake_game.py
    ├── stack_interpolator.py
    ├── stack_to_slices.py
    └── yolo_trainer.py
```

### ImageAnnotator (annotator_window.py)

**Responsibility**: Main application window and state management

**Key Attributes**:
```python
all_annotations: dict[str, list]    # {filename: [annotation_dicts]}
all_images: list[str]               # List of loaded image filenames
class_mapping: dict[str, int]       # {class_name: class_id}
image_paths: dict[str, str]         # {filename: absolute_path}
image_dimensions: dict              # Multi-dimensional image metadata
image_slices: dict                  # Extracted slices from stacks
current_slice: str                  # Currently displayed slice
```

**Key Methods**:
- `save_project()`: Save project state to JSON
- `load_project_data()`: Load project from JSON
- `add_annotation()`: Add annotation to current image
- `export_annotations()`: Export to various formats
- `import_annotations()`: Import from COCO/YOLO

### ImageLabel (image_label.py)

**Responsibility**: Image display and annotation interaction

**Key Attributes**:
```python
current_tool: str                   # Active annotation tool
zoom_factor: float                  # Current zoom level
annotations: dict                   # Displayed annotations
class_colors: dict                  # Class color mapping
temp_paint_mask: np.ndarray         # Temporary paint strokes
sam_positive_points: list           # SAM positive points
sam_negative_points: list           # SAM negative points
```

**Key Methods**:
- `mousePressEvent()`: Handle mouse clicks for annotation
- `mouseMoveEvent()`: Handle mouse dragging
- `paintEvent()`: Render image and annotations
- `zoom_in()`, `zoom_out()`: Zoom controls
- `start_painting()`, `start_erasing()`: Brush tools

### SAMUtils (sam_utils.py)

**Responsibility**: SAM model loading and inference (in-process).

**Key state** (on the `SAMUtils` instance):
- `sam_models: dict` — available SAM model variants (class-level, exposed for the UI dropdown)
- `current_sam_model: str | None` — name of the currently loaded model; `None` if unloaded
- `_model: ultralytics.SAM | None` — the loaded model object (private)

**Key public methods**:
- `change_sam_model(model_name)` — load a SAM model. Blocks the calling thread (with the UI's event loop pumping) until weights are downloaded and the model is in memory. Raises on load failure.
- `apply_sam_points(image, positive_points, negative_points)` — point-prompted segmentation.
- `apply_sam_prediction(image, bbox)` — single bbox-prompted segmentation.
- `apply_sam_predictions_batch(image, bboxes)` — multi-bbox segmentation in one model call (used by the DINO pipeline).
- `unload()` — drop the cached model and free GPU/CPU memory. Wired to the Tools → "Unload AI Models" menu entry.

**Module-level helpers** (not class methods):
- `_qimage_to_numpy(qimage)` — convert a `QImage` to an owned numpy array (always copies; see ADR-013 on lifetime safety).
- `_mask_to_polygon(mask)` — convert a SAM mask tensor into polygon contour vertices.
- `_run_sync(fn, *args, **kwargs)` — run `fn` on a worker `QThread`, pump the calling thread's event loop until done, re-raise any exception. Serialises concurrent calls via the `_inference_in_flight` flag; re-entry raises `InferenceBusyError`.

Inference runs in-process on a background `QThread`. `SAMUtils._run_sync()`
spawns the thread, pumps the caller's event loop until done, and returns
the result — keeping the API synchronous-looking from call sites while
the UI stays responsive. Model objects (Ultralytics `SAM`) live on the
`SAMUtils` singleton and persist across calls. See
[ADR-013](09_architecture_decisions.md#adr-013-in-process-inference-with-qthread-wrapping).
The earlier subprocess approach is documented as
[ADR-011](09_architecture_decisions.md#adr-011-run-torch-based-workers-in-isolated-subprocesses)
(Superseded).

### DINO Subsystem (Grounding DINO + SAM pipeline)

LLM-assisted detection: the user gives free-form text phrases per class,
Grounding DINO produces bounding boxes, and SAM 2 refines them into
segmentation masks.

| Module | Responsibility |
|--------|----------------|
| `dino_utils.py` | `DINOUtils` — in-process Grounding DINO wrapper. Resolves model paths via `models_base_dir()`, loads `transformers.AutoModelForZeroShotObjectDetection` lazily on first use, caches it across calls, runs inference on a worker `QThread` (same `_run_sync` pattern as `SAMUtils`). |
| `dino_phrase_editor.py` | Two widgets: `ClassThresholdTable` (per-class box/text/NMS thresholds) and `PhraseEditorPanel` (per-class phrase list). These widgets are the **single source of truth** for phrases and thresholds; project save/load reads/writes them via `get_all_phrases()` / `set_phrases()` and `get_thresholds_dict()` / `set_thresholds()`. |
| `dino_merge_dialog.py` | Standalone dialog: merges accumulated DINO+SAM annotations across images into a training-ready COCO JSON. |

**Detection call signature** (in-process):
```python
DINOUtils().detect(
    qimage,                                # PyQt6.QtGui.QImage
    class_configs=[
        {"name": "drone", "phrases": ["drone", "quadcopter"],
         "box_thr": 0.10, "txt_thr": 0.25, "nms_thr": 0.50},
        ...
    ],
    model_name="grounding-dino-base",      # or custom_model_path=...
)
```

**Detection return value**:
```python
[
    {"class_name": "drone", "bbox": [x1, y1, x2, y2], "score": 0.93, "label": "drone"},
    ...
]
# or [] if no boxes survived filtering, or None on error
```

DINO's xyxy boxes feed directly into `SAMUtils.apply_sam_predictions_batch()`,
which returns segmentation polygons (xywh bbox is derived from the polygon at
export time — see [Cross-cutting Concepts](08_crosscutting_concepts.md)).

## Level 3: Export/Import Subsystem

### Export Formats (export_formats.py)

**Functions**:
- `export_coco_json()`: COCO format with images directory
- `export_yolo_v5plus()`: YOLOv11-compatible structure
- `export_yolo_v4()`: Legacy YOLO format
- `export_labeled_images()`: Colored overlay visualizations
- `export_semantic_labels()`: Single-channel label images
- `export_pascal_voc_bbox()`: Pascal VOC XML (bounding boxes)

**Data Flow**:
```
all_annotations (internal format)
    │
    ├──> convert_to_coco() ──> COCO JSON
    ├──> convert to YOLO ──> YOLO txt files + data.yaml
    ├──> render_annotations() ──> Labeled images (PNG)
    ├──> mask_to_labels() ──> Semantic labels (PNG)
    └──> convert to Pascal VOC ──> XML files
```

### Import Formats (import_formats.py)

**Functions**:
- `import_coco_json()`: Parse COCO format
- `import_yolo_v5plus()`: Parse YOLO v8/v11 format
- `import_yolo_v4()`: Parse legacy YOLO format
- `process_import_format()`: Unified import dispatcher

## Level 4: Tool Dialogs

Each tool is a standalone dialog/window:

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `annotation_statistics.py` | Statistics display | Count, area per class, plotly charts |
| `coco_json_combiner.py` | Merge datasets | Combine multiple COCO JSON files |
| `dataset_splitter.py` | Train/val/test split | Stratified splitting, configurable ratios |
| `image_patcher.py` | Create patches | Sliding window with overlap |
| `image_augmenter.py` | Data augmentation | Rotation, flip, brightness, preview |
| `slice_registration.py` | Align slices | Multiple registration algorithms (pystackreg) |
| `stack_interpolator.py` | Z-spacing adjustment | Interpolation methods, memory-efficient |
| `dicom_converter.py` | DICOM to TIFF | Preserve metadata, export to JSON |
| `yolo_trainer.py` | Model training | Train YOLO, load predictions |

## Data Model

### Project JSON Structure

```json
{
  "images": ["image1.png", "image2.jpg"],
  "image_paths": {
    "image1.png": "/absolute/path/to/image1.png"
  },
  "classes": ["cell", "nucleus"],
  "class_colors": {
    "cell": [255, 0, 0],
    "nucleus": [0, 255, 0]
  },
  "annotations": {
    "image1.png": [
      {
        "segmentation": [x1, y1, x2, y2, ...],
        "category": "cell"
      },
      {
        "bbox": [x, y, width, height],
        "category": "nucleus"
      }
    ]
  },
  "image_dimensions": {
    "stack.tif": "TZCYX"
  },
  "image_shapes": {
    "stack.tif": [10, 50, 3, 512, 512]
  }
}
```

### Annotation Formats

**Polygon** (segmentation):
```python
{
    "segmentation": [x1, y1, x2, y2, x3, y3, ...],  # Flattened coordinates
    "category": "class_name"
}
```

**Rectangle** (bounding box):
```python
{
    "bbox": [x, y, width, height],  # COCO format
    "category": "class_name"
}
```

### Multi-dimensional Image Handling

**Slice Naming Convention**:
```
{filename}_T{t}_Z{z}_C{c}_S{s}
Example: stack.tif_T0_Z5_C0_S0
```

**Dimension Labels**: T (Time), Z (Depth), C (Channel), S (Scene), H (Height), W (Width)

## Dependencies Between Components

```
ImageAnnotator (main window)
    ├── uses ──> ImageLabel (display/interaction)
    ├── uses ──> SAMUtils (model inference)
    ├── uses ──> export_formats (export)
    ├── uses ──> import_formats (import)
    ├── uses ──> yolo_trainer (training)
    └── launches ──> Tool Dialogs (utilities)

ImageLabel
    ├── references ──> ImageAnnotator (callbacks)
    └── uses ──> utils (area, bbox calculations)

SAMUtils
    └── depends on ──> ultralytics.SAM

Tool Dialogs
    └── operate independently (standalone windows)
```
