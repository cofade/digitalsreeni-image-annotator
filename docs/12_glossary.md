# Glossary

## Terms and Definitions

### Annotation
A marked region on an image, either a polygon (segmentation) or rectangle (bounding box), associated with a class label.

### Bounding Box (bbox)
A rectangular annotation defined by `[x, y, width, height]` in COCO format. Stored in annotation as `"bbox"` key.

### Class
A category label for annotations (e.g., "cell", "nucleus", "mitochondria"). Each class has an ID and color.

### COCO Format
Common Objects in Context - a standardized JSON format for object detection and segmentation annotations. Includes images, categories, and annotations with segmentation polygons or bounding boxes.

### CZI File
Carl Zeiss Image file format for multi-dimensional microscopy images. Contains metadata and multi-channel Z-stacks.

### DINO / Grounding DINO
"DINO" in this codebase refers specifically to **Grounding DINO** (IDEA-Research, 2023) — an open-set object detector that takes a natural-language phrase ("drone", "wing of an aircraft") and returns bounding boxes for matching regions of an image. Not to be confused with the self-supervised vision-only DINOv1/v2 backbones (similar name, different model). Models live under `models/grounding-dino-base/` and `models/grounding-dino-tiny/`.

### Fine-Tuning (SAM)
Continuing training of a pre-trained SAM 2 / 2.1 model on the user's own annotations so the assisted tools work better on their imagery. Because Ultralytics ships no SAM trainer, the app uses a custom loop over the Ultralytics `SAM2Model` (see [ADR-021](09_architecture_decisions.md#adr-021-sam-fine-tuning-via-a-custom-loop-over-the-ultralytics-sam2-module)). **Decoder-only** (default) trains just the mask decoder, freezing the image and prompt encoders — fast, low-VRAM, robust on modest data; optionally the image encoder is also unfrozen for heavily domain-shifted data.

### Focal + Dice Loss
The mask-supervision loss used during SAM fine-tuning: a focal term (down-weights easy pixels, emphasises hard ones) plus a dice term (region overlap), combined ≈20:1. Standard across the SAM fine-tuning literature.

### Mask Decoder
The lightweight SAM head that turns image embeddings + prompt embeddings into mask logits. The default fine-tuning target (`sam_mask_decoder`, ~4.2M params for the tiny model) since it is small and adapts quickly.

### Multi-dimensional Image
An image with more than 2 dimensions, typically from microscopy. Dimensions include T (time), Z (depth), C (channel), S (scene), H (height), W (width).

### NMS (Non-Maximum Suppression)
Post-processing step that removes redundant overlapping boxes. After Grounding DINO scores many candidate boxes, NMS keeps only the highest-scoring one per cluster — controlled per-class via the **NMS thr** column in the DINO panel (higher = more aggressive de-duplication).

### Paint Brush Tool
Drawing tool that creates freeform annotations by painting a mask with adjustable brush size. Converted to polygon contours when finished.

### Pascal VOC
Visual Object Classes dataset format. XML-based annotation format primarily for bounding boxes.

### Phrase (DINO)
A free-form text description used by Grounding DINO to find objects. Each annotation class has a list of phrases — for example a "drone" class might use phrases `["drone", "quadcopter", "octocopter", "helicopter"]`. The class name itself is always the first phrase and cannot be removed.

### Polygon / Segmentation
A closed shape annotation defined by a list of vertex coordinates `[x1, y1, x2, y2, ...]`. Stored in annotation as `"segmentation"` key.

### Project
A saved workspace containing images, classes, and annotations. Stored as a `.json` file with absolute paths to images.

### SAM / SAM 2
Segment Anything Model - Meta's foundation model for image segmentation. Version 2 (SAM 2) is used in this application.

### SAM Point Mode
Annotation mode where user clicks positive points (inside object) and negative points (outside object) to guide SAM segmentation.

### Select Mode (Canvas)
The idle canvas state (no drawing/SAM tool active, not editing, no temp review) in
which clicks and drags select existing masks instead of drawing. Single-click
selects, Shift toggles/adds, drag box-selects; double-click still enters vertex
edit. See ADR-022.

### Rubber-Band Selection
A dashed selection rectangle dragged on the canvas in Select Mode; every annotation
whose bounds intersect it is selected. Shift+drag adds to the current selection.

### Semantic Labels
Single-channel image where each pixel value represents the class ID. Used for semantic segmentation training.

### Slice
A 2D image extracted from a multi-dimensional image stack. Named with format `{filename}_T{t}_Z{z}_C{c}_S{s}`.

### Stack
A multi-dimensional image, typically a TIFF or CZI file with multiple 2D slices in Z-dimension (depth).

### Subprocess Worker (historical)
A standalone Python script (`sam_worker.py`, `dino_worker.py`) that ran ML model inference in its own process to dodge a PyQt5 + Torch DLL load-order conflict on Windows + Python 3.14. Removed once the codebase migrated to PyQt6 (the conflict no longer manifests). See [ADR-011](09_architecture_decisions.md#adr-011-run-torch-based-workers-in-isolated-subprocesses) (Superseded) and [ADR-013](09_architecture_decisions.md#adr-013-in-process-inference-with-qthread-wrapping).

### TIFF Stack
Multi-page TIFF file containing multiple 2D images, often used for Z-stacks in microscopy.

### YOLO Format
You Only Look Once - object detection format. Uses `.txt` files with normalized coordinates: `class_id x_center y_center width height`.

### Z-Stack
A series of 2D images taken at different focal depths (Z positions), used in microscopy to capture 3D structure.

### CanvasContext
Narrow read-only view of main-window state exposed to `ImageLabel`. Introduced in Phase 6 (ADR-018) to replace the old `image_label.main_window` back-reference. Method-style accessors (`paint_brush_size()`, `current_class()`, `is_class_visible(name)`, `scroll_area()`, …) so future state migrations can re-route reads without touching the widget. Constructed once in `ImageAnnotator.__init__` and passed via `image_label.set_context(ctx)`.

### Controller
Architectural pattern used across `controllers/*`. A controller is a `QObject` subclass holding `self.mw = main_window` that owns a single responsibility cluster carved out of the old monolithic `ImageAnnotator` — project I/O, image loading, annotations, classes, SAM, DINO, or YOLO. The orchestrator delegates to the controllers via thin pass-through methods, keeping external entry points (menu actions, signal connections) stable across refactors. Seven controllers exist as of Phase 8.

### ToolHandler
Base class for per-tool mouse / key / paint behaviour inside `ImageLabel`. Plain Python object (not a `QObject`); holds a back-reference to the widget for signal emission and `CanvasContext` reads. Subclasses (`RectangleTool`, `PolygonTool`, `PaintBrushTool`, `EraserTool`) live in `widgets/tools/` and are dispatched to by `ImageLabel.active_tool_handler`. Introduced in Phase 7 (ADR-019).

### Tool subclasses (`RectangleTool`, `PolygonTool`, `PaintBrushTool`, `EraserTool`)
Concrete `ToolHandler` implementations, one per mouse-driven annotation tool. Each overrides the event hooks defined on the base class (`on_mouse_press`, `on_mouse_move`, `on_mouse_release`, `on_double_click`, `on_enter`, `on_escape`, `paint_overlay`, `deactivate`) and participates in the `has_unsaved_state()` / `commit()` / `discard()` contract used by the `check_unsaved_changes` dialog.

### UI builders (`build_menu_bar`, `build_sidebar`, `build_image_area`, `build_image_list`)
Functions under `ui/` that construct widget trees at startup. Each takes the `ImageAnnotator` instance as `window`, attaches widgets as `window.X = QWidget(...)` so other modules can read them, and wires signals to `window.<method>` delegate methods. Replaced the equivalent `setup_*` methods on `ImageAnnotator` in Phase 8.

## Acronyms

| Acronym | Full Term |
|---------|-----------|
| ADR | Architecture Decision Record |
| API | Application Programming Interface |
| bbox | Bounding Box |
| COCO | Common Objects in Context |
| CZI | Carl Zeiss Image |
| DICOM | Digital Imaging and Communications in Medicine |
| GUI | Graphical User Interface |
| JSON | JavaScript Object Notation |
| ML | Machine Learning |
| OOM | Out Of Memory |
| PNG | Portable Network Graphics |
| PyQt | Python bindings for Qt framework |
| RGB | Red Green Blue (color model) |
| SAM | Segment Anything Model |
| TIFF | Tagged Image File Format |
| UI | User Interface |
| VOC | Visual Object Classes |
| XML | eXtensible Markup Language |
| YOLO | You Only Look Once |

## File Extensions

| Extension | Description |
|-----------|-------------|
| `.json` | Project file or COCO annotation file |
| `.tif`, `.tiff` | TIFF image, possibly multi-dimensional stack |
| `.czi` | Carl Zeiss microscopy image |
| `.png`, `.jpg`, `.jpeg` | Standard image formats |
| `.txt` | YOLO annotation file |
| `.xml` | Pascal VOC annotation file |
| `.yaml`, `.yml` | YOLO data configuration file |
| `.pt` | PyTorch model file (SAM weights) |
| `.dcm` | DICOM medical image file |

## Key Classes (Code)

| Class | Module | Description |
|-------|--------|-------------|
| `ImageAnnotator` | annotator_window.py | Thin orchestrator (QMainWindow). Holds controllers, wires signals, delegates almost everything. |
| `ImageLabel` | widgets/image_label.py | Canvas widget — display, zoom/pan, event dispatch to tool handlers. |
| `CanvasContext` | widgets/canvas_context.py | Narrow read view of main-window state for ImageLabel (ADR-018). |
| `ToolHandler` | widgets/tools/base.py | Base class for per-tool mouse/key handlers (ADR-019). |
| `RectangleTool` / `PolygonTool` / `PaintBrushTool` / `EraserTool` | widgets/tools/ | Per-tool handler subclasses. |
| `ProjectController` | controllers/project_controller.py | `.iap` save/load, auto-save, `is_loading_project` guard. |
| `ImageController` | controllers/image_controller.py | TIFF/CZI loading, multi-dim slicing, image/slice switching. |
| `AnnotationController` | controllers/annotation_controller.py | Annotation CRUD, sort, edit-mode, finish_polygon/rectangle. |
| `ClassController` | controllers/class_controller.py | Class add/delete/rename/colour/visibility. |
| `SAMController` | controllers/sam_controller.py | SAM model picker + debounce + ADR-013 re-entrancy guard. |
| `DINOController` | controllers/dino_controller.py | DINO single + batch detection, batch review, temp-class workflow. |
| `YOLOController` | controllers/yolo_controller.py | YOLO training menu + prediction wiring. |
| `SAMUtils` | inference/sam_utils.py | SAM model loading and inference. |
| `DINOUtils` | inference/dino_utils.py | Grounding-DINO model loading and inference. |
| `DimensionDialog` | controllers/image_controller.py | Dialog for assigning dimensions to multi-dim stacks. |
| `TrainingThread` | controllers/yolo_controller.py | Background thread for YOLO training. |
| `YOLOTrainer` | dialogs/yolo_trainer.py | YOLO model training and prediction dialog. |
| `DINOReviewEventFilter` | controllers/dino_controller.py | App-wide Enter/Escape filter during DINO review (ADR-015). |

## Data Structure Keys

### Project JSON
- `images`: List of image filenames
- `image_paths`: Dict mapping filename to absolute path
- `classes`: List of class names
- `class_colors`: Dict mapping class name to RGB tuple
- `annotations`: Dict mapping filename/slice to list of annotation dicts
- `image_dimensions`: Dict mapping filename to dimension string (e.g., "TZCYX")
- `image_shapes`: Dict mapping filename to shape tuple

### Annotation Dict
- `segmentation`: Flattened polygon coordinates `[x1, y1, x2, y2, ...]`
- `bbox`: Rectangle `[x, y, width, height]` (mutually exclusive with segmentation)
- `category`: Class name string

### COCO JSON
- `images`: List of image metadata dicts
- `categories`: List of class dicts with id and name
- `annotations`: List of annotation dicts with id, image_id, category_id, segmentation/bbox

## UI Components

| Component | Description |
|-----------|-------------|
| Tool Section | Buttons for Polygon, Rectangle, Paint Brush, Eraser, SAM tools |
| Class List | QListWidget showing all classes with colors |
| Annotation List | QListWidget showing all annotations for current image |
| Image Label | Central QLabel displaying image with zoom/pan |
| Slice Slider | Navigate through multi-dimensional image slices |
| Menu Bar | File, Edit, View, Tools, Help menus |

## Coordinate Systems

| System | Origin | Units | Used For |
|--------|--------|-------|----------|
| Image Coordinates | Top-left (0,0) | Pixels | Annotation storage, calculations |
| Screen Coordinates | Top-left of window | Pixels | Mouse events, rendering |
| Normalized Coordinates | Top-left (0,0) to (1,1) | Fractional | YOLO export format |
