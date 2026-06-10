# Cross-cutting Concepts

## Coordinate Systems

### Screen Coordinates vs Image Coordinates

All mouse events are in screen coordinates and must be converted to image coordinates:

```python
# In ImageLabel
def screen_to_image_coords(self, screen_pos):
    # Account for offset (centering)
    image_x = screen_pos.x() - self.offset_x
    image_y = screen_pos.y() - self.offset_y

    # Account for zoom
    original_x = image_x / self.zoom_factor
    original_y = image_y / self.zoom_factor

    return (original_x, original_y)
```

### Annotation Storage Format

Annotations are stored in image coordinates (unzoomed, absolute pixels):
- **Polygon**: Flattened list `[x1, y1, x2, y2, ...]`
- **Rectangle**: COCO format `[x, y, width, height]`

### Pan + Zoom Reference Frames

Two non-obvious gotchas live in `ImageLabel.mouseMoveEvent` /
`wheelEvent`:

- **Pan must use `event.globalPosition()`, not `event.position()`.**
  Widget-local coords absorb half the cursor delta during a scrollbar
  move (the widget shifts under the cursor mid-drag) → effective
  half-speed pan. The global frame is stable.
- **Zoom-to-cursor must compute the post-zoom `offset_x/y`
  analytically from the viewport, not read `self.offset_x` after the
  zoom call.** `update_scaled_pixmap()` only *relaxes* the minimum
  size on zoom-out; the widget hasn't shrunk by the time
  `update_offset()` runs, so `self.width()` is stale and the offset
  comes out wrong. Use `viewport().width()` + `scaled_pixmap.width()`
  to derive the offset directly. Zoom-in worked by accident because
  the widget grows immediately when `setMinimumSize` enlarges it.

## Image Format Conversions

### QImage ↔ NumPy Array

**QImage to NumPy** (for SAM inference):
```python
def qimage_to_numpy(qimage):
    width = qimage.width()
    height = qimage.height()
    fmt = qimage.format()

    if fmt == QImage.Format_Grayscale16:
        # 16-bit → normalize to 8-bit → RGB
        buffer = qimage.constBits().asarray(height * width * 2)
        image = np.frombuffer(buffer, dtype=np.uint16)
        image_8bit = normalize_16bit_to_8bit(image)
        return np.stack((image_8bit,) * 3, axis=-1)

    elif fmt == QImage.Format_RGB888:
        # Direct conversion
        buffer = qimage.constBits().asarray(height * width * 3)
        return np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))

    # ... handle other formats
```

**16-bit Normalization**:
```python
def normalize_16bit_to_8bit(image):
    # Percentile-based normalization for better contrast
    p2, p98 = np.percentile(image, (2, 98))
    image_clipped = np.clip(image, p2, p98)
    return ((image_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
```

## Polygon Operations

### Shapely for Geometry

**Merge Annotations**:
```python
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# Convert segmentation lists to Shapely Polygons
polygons = []
for ann in selected_annotations:
    coords = [(ann["segmentation"][i], ann["segmentation"][i+1])
              for i in range(0, len(ann["segmentation"]), 2)]
    poly = Polygon(coords)
    poly = make_valid(poly)  # Fix invalid polygons
    polygons.append(poly)

# Merge
merged = unary_union(polygons)

# Convert back to segmentation format
coords = list(merged.exterior.coords)
segmentation = [coord for point in coords for coord in point]
```

### Minimum Area Threshold

Paint brush annotations filter out small artifacts:
```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    if cv2.contourArea(contour) > 10:  # 10 pixels minimum
        # Accept annotation
```

## Autosave and Project Corruption Prevention

### Critical: Disable Autosave During Load

**Problem**: Autosave triggered during loading can corrupt project files

**Solution** (v0.8.12):
```python
class ImageAnnotator:
    def load_project_data(self, project_data):
        self.is_loading_project = True  # Disable autosave
        try:
            # ... load all data
        finally:
            self.is_loading_project = False  # Re-enable

    def save_project(self, show_message=True):
        if self.is_loading_project:
            return  # Skip save during load
        # ... normal save logic
```

## SAM Model Management

### Model Caching

First use downloads models, subsequent uses load from cache:
```python
# Ultralytics automatically caches in:
# - Working directory (current implementation)
# - Or ~/.cache/ultralytics/ (default)

sam_model = SAM("sam2_t.pt")  # Downloads if not present
```

### Releasing Model GPU Memory

`SAMUtils.unload()` and `DINOUtils.unload()` must do **three** things,
in order:

1. Drop the cached Python references (`self._model = None`, etc.).
2. **`gc.collect()`** to break circular references inside Ultralytics
   / Transformers model objects (config ↔ model, processor ↔
   tokenizer). Without this, the C++/CUDA backing memory stays pinned
   until Python's cyclic GC runs on its own schedule, which can be
   many seconds or never. Task Manager / `nvidia-smi` will show zero
   drop in GPU memory.
3. **`torch.cuda.empty_cache()`** (plus `torch.cuda.ipc_collect()`) so
   the PyTorch allocator returns the freed blocks to the OS / driver.

Skipping step 2 was the cause of "Tools → Unload AI Models does
nothing visible" in v0.9.0 manual testing.

### Model Size Recommendations

| Model | Size | RAM Usage | Speed | Recommendation |
|-------|------|-----------|-------|----------------|
| SAM 2 tiny | ~40MB | Low | Fast | ✅ Recommended for most users |
| SAM 2 small | ~90MB | Medium | Medium | ✅ Good balance |
| SAM 2 base | ~150MB | Medium-High | Slow | ⚠️ Use with caution |
| SAM 2 large | ~400MB | High | Very Slow | ❌ Not recommended (crashes on limited resources) |

## Dark Mode Support

### Stylesheet Switching

```python
# In ImageAnnotator
if dark_mode_enabled:
    self.setStyleSheet(soft_dark_stylesheet)
    self.image_label.set_dark_mode(True)
else:
    self.setStyleSheet(default_stylesheet)
    self.image_label.set_dark_mode(False)
```

**Dark Mode Considerations**:
- Annotation rendering uses inverted colors for visibility
- Text labels use high-contrast colors
- Background grid adjusted for dark backgrounds

### Dark Mode — No Hardcoded Colors Rule

**Do not hardcode `background`, `color`, or other palette-dependent
values in widget `setStyleSheet(...)` calls.** They override both the
default OS look *and* `soft_dark_stylesheet.py`, leaving bright
rectangles on the dark sidebar. Past offenders that bit us:

- `ClassThresholdTable` header had `background: #e0e0e0;` → bright bar
  across the top of the DINO panel in dark mode.
- `lbl_dino_status` had `background: #f5f5f5;` → bright box where the
  "No DINO model loaded" status sat.

Either leave the property out of the inline stylesheet so the global
sheet wins, or use Qt's palette role functions (`palette(base)`,
`palette(mid)`, `palette(text)`, …) which resolve at paint time
against the active palette. Inline hardcoded greys are an anti-pattern.

When introducing a new widget type that doesn't have a rule in
`soft_dark_stylesheet.py` yet — add the rule there *first*, then build
the widget. Otherwise the widget uses the OS default in dark mode,
which on Windows means barely-visible radio-button indicators and
white-on-white headers (the dataset splitter radio buttons hit this
before they were styled).

## Thread Safety for YOLO Training

### Training Thread

```python
class TrainingThread(QThread):
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(object)

    def run(self):
        try:
            results = self.yolo_trainer.train_model(
                epochs=self.epochs,
                imgsz=self.imgsz
            )
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit(str(e))
```

**UI Update**:
- Training runs in background thread
- Progress updates via Qt signals
- UI remains responsive during training

## Error Handling

### YOLO Model/Data Mismatch

**Problem**: Loading YOLO model trained on different classes

**Solution**:
```python
try:
    model = YOLO(model_path)
    model_classes = model.names
    yaml_classes = data_yaml['names']

    if model_classes != yaml_classes:
        QMessageBox.warning(
            self,
            "Class Mismatch",
            f"Model classes: {model_classes}\n"
            f"Data classes: {yaml_classes}"
        )
        return
except Exception as e:
    # Handle gracefully instead of crashing
```

## Multi-dimensional Image Slicing

### Dimension Assignment

User assigns meaning to each dimension:
```
TIFF shape: (10, 50, 3, 512, 512)
User assigns: T   Z   C   H    W

Result: 10 timepoints × 50 Z-slices × 3 channels = 1500 slices
Each slice: 512×512 pixels
```

### Slice Naming Convention

```python
def generate_slice_name(filename, t, z, c, s):
    parts = []
    if t is not None:
        parts.append(f"T{t}")
    if z is not None:
        parts.append(f"Z{z}")
    if c is not None:
        parts.append(f"C{c}")
    if s is not None:
        parts.append(f"S{s}")

    return f"{filename}_{'_'.join(parts)}"

# Example: "stack.tif_T0_Z5_C0"
```

## Keyboard Shortcuts

### Global Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New Project |
| Ctrl+O | Open Project |
| Ctrl+S | Save Project |
| Ctrl+W | Close Project |
| Ctrl+Shift+S | Annotation Statistics |
| F1 | Help Window |

### Canvas Shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+Wheel | Zoom In/Out |
| Ctrl+Drag | Pan |
| Esc | Cancel Current Annotation |
| Enter | Finish/Accept Annotation |
| Up/Down | Navigate Slices (multi-dimensional) |
| -/= | Adjust Brush/Eraser Size |

## Logging and Debug Output

### Print Statements

Current implementation uses `print()` for debugging:
```python
print(f"Changed SAM model to: {model_name}")
print(f"SAM input points: {all_points}, labels: {all_labels}")
print(f"Loading project from: {project_path}")
```

**Note**: No formal logging framework is used. Output goes to console.

## DINO Temp Annotations — Single Field, Many Images

`ImageLabel.temp_annotations` is a **single list on the image_label**,
not a per-image cache. It holds the pending DINO+SAM masks shown as
an overlay while the user decides accept/reject. The per-image batch
cache is `ImageAnnotator.dino_batch_results` (a dict keyed by image
name) — `image_label.temp_annotations` is only ever set to one image's
slice of that dict at a time.

Consequences this codebase has tripped over:

- **Image/slice switches must re-sync** `temp_annotations` from
  `dino_batch_results` for the new image (load if pending, clear if
  not). Otherwise masks from the previously-viewed image visually
  bleed onto every slice the user navigates to. See
  `_refresh_dino_temp_for_current()`.
- **Enter / Escape during review** must work even when the focus is on
  slice_list / image_list / a button — `QListWidget` consumes
  Enter for itemActivated before `ImageLabel.keyPressEvent` ever sees
  it. Solved with an application-wide event filter
  (`DINOReviewEventFilter`) that fires only while
  `temp_annotations` has DINO items and skips modal dialogs and text
  inputs. Setting `image_label.setFocus()` synchronously inside
  `_show_dino_batch_review` was not enough — Qt's focus handling
  raced the click event that opened the review and the canvas
  often didn't end up focused. `QTimer.singleShot(0, …)` defers until
  the current event chain settles.
- **Auto-accept dropdown applies to both paths.** The batch-mode
  combo ("Review before accepting" / "Auto-accept all detections")
  controls **both** "Detect Current Image" and "Detect All Images".
  Only checking it in `run_dino_detection_batch` and not
  `run_dino_detection_single` produced a confusing "auto-accept
  doesn't actually auto-accept for single image" bug.
- **Batch detection must enumerate slices, not just `all_images`.**
  Multi-dim images live in `all_images` as a single entry with
  `is_multi_slice=True`, and their actual slice QImages live under
  `self.image_slices[base_name]`. The first cut of
  `run_dino_detection_batch` iterated `all_images` and skipped the
  multi-slice entries with a console log — leaving stack-based
  projects unable to use "Detect All Images" at all. Batch jobs go
  through `_collect_dino_batch_work_items()` which flattens regular
  images + every loaded slice into a `(name, QImage)` list.
- **Review navigation must handle slice names.** Slice names like
  `stack_T1_Z1_C1` are not in `image_list`. After collecting batch
  results for slices, `_navigate_to_image_or_slice()` finds the
  parent image via `os.path.splitext` matching and then activates
  the specific row in `slice_list`. Without this, batch review on
  slices either silently no-op'd or showed the first regular
  image's masks on a slice.

## Multi-dimensional TIFF Axis Defaults

`load_tiff` extracts `tif.series[0].axes` (e.g. `"TZCYX"`) and maps
it through `{T:T, Z:Z, C:C, S:S, Y:H, X:W}` to populate the
`DimensionDialog` combo boxes. This is what lets a user open an
ImageJ-style 5D TIFF and just click OK.

When the metadata is missing or unfamiliar, fall back to the
hand-crafted defaults keyed on `ndim`:

| ndim | default labels |
|------|---------------|
| 3 | `Z H W` |
| 4 | `T Z H W` |
| 5 | `T Z C H W` |
| 6 | `T Z C S H W` |

**Do not** use `default_dimensions[-ndim:]` of a shorter list to
"extend" defaults — that silently degrades for `ndim ≥ 5`: the final
combo gets no default and inherits the first item ("T"), which is
the wrong axis. The 5D TZCYX bug that produced 2560 one-row slices
on a `(2,5,2,256,256)` file came from exactly this.

## Export Format Filename Matching

`export_formats.py` historically looked up image paths via substring
match:

```python
image_path = next(
    (path for name, path in image_paths.items() if image_name in name),
    None,
)
```

That is fragile — `"bee.jpg" in "honeybee.jpg"` returns True and you
write the wrong file. The COCO, YOLO v4, and YOLO v5+ exports all
share this code path.

**Always try the exact key first; fall back to substring only if no
exact key matches.** Pattern:

```python
image_path = image_paths.get(image_name)
if image_path is None:
    image_path = next(
        (path for name, path in image_paths.items() if image_name in name),
        None,
    )
```

The substring fallback is kept for backward compatibility with old
projects that may have stored normalised image names (e.g. without
extension); new code should prefer the exact-key path.

## Canvas Decoupling — Signals + CanvasContext

`ImageLabel` (the canvas widget) does **not** hold a reference to
`ImageAnnotator`. Communication is split:

- **Writes** (committing an annotation, requesting a SAM prediction,
  asking for tools to be re-enabled, etc.) leave the widget as Qt
  `pyqtSignal` emissions. The signal block at the top of `ImageLabel`
  documents every outbound interaction. `ImageAnnotator` connects
  each signal to the right controller slot once, in
  `_connect_image_label_signals` (called at the end of
  `ImageAnnotator.__init__`).
- **Reads** (`paint_brush_size`, `current_class`, `class_mapping`,
  `is_class_visible`, `scroll_area`, etc.) go through a
  `CanvasContext` object passed in via
  `image_label.set_context(CanvasContext(self))`.
  `CanvasContext` wraps the main window rather than copying state,
  so updates made by controllers are visible on the next read.

**Why both mechanisms.** Signals are inherently one-way (fire and
forget); a synchronous read like "is this class visible" needs a
return value, which signals don't provide. Trying to express reads
as request/response signals adds latency and ordering bugs. The
`CanvasContext` accessor list is small (~10 methods) and stable.

**Rules for adding traffic in either direction**:

- New write from canvas → orchestrator: declare a `pyqtSignal` on
  `ImageLabel`, add a slot on a controller, wire it in
  `_connect_image_label_signals`. Do not add a back-reference to
  `ImageAnnotator`.
- New read from canvas → orchestrator: add a method on
  `CanvasContext`. Do not expose `_ctx._mw` directly.

**Synchronous-emit ordering**. Qt's default `AutoConnection` runs the
slot synchronously when the sender and receiver share a thread (true
for everything on the GUI thread). Code that emits a signal and then
reads state expected to be updated by it is correct — the slot has
already run by the time `.emit()` returns. This is load-bearing for
`accept_temp_annotations`, where `classRequested` must complete
before the subsequent class lookup.

**Batch save signal**. Paint commits and accept-temp commits emit
`annotationCommitted` per annotation but `annotationsBatchSaved` only
once at the end. The single batch save preserves O(1) `.iap` writes
per user action; replacing it with a per-annotation save would turn
paint commits into O(N). See ADR-018.

See ADR-018 in `09_architecture_decisions.md` for the rationale and
the full pattern.
