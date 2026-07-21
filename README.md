# DigitalSreeni Image Annotator (cofade fork)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A PyQt6 desktop application for annotating images — manual tools plus AI-assisted
segmentation (SAM 2), text-prompted detection (Grounding DINO), keypoint/pose
annotation, and in-app model training. This is an actively-developed **fork** of
[bnsreenu/digitalsreeni-image-annotator](https://github.com/bnsreenu/digitalsreeni-image-annotator)
that adds in-process inference, DINO detection, SAM 2 fine-tuning with MLflow
tracking, undo/redo, an annotations table with per-mask simplification,
keypoint/pose annotation, canvas selection + handle editing, and a pytest +
pytest-qt test suite running in CI. See
[Fork attribution & upstream](#fork-attribution--upstream).

![DigitalSreeni Image Annotator Demo](screenshots/digitalsreeni-image-annotator-demo.gif)

> Original tool and video walkthroughs by **@DigitalSreeni** — Dr. Sreenivas
> Bhattiprolu (upstream author). The demo GIF above shows an earlier upstream
> release; this fork's UI has since gained the features listed below.

## What is this

A scientific image-annotation desktop app: draw polygons / rectangles / paint
masks, place keypoints, and let SAM 2 or Grounding DINO propose masks from a box,
a few clicks, or a text prompt. It handles multi-dimensional images (TIFF stacks,
CZI), round-trips COCO / YOLO / Pascal VOC, and can fine-tune SAM 2 or train YOLO
directly from your annotations.

## Feature highlights

**Annotation tools**
- Polygon, rectangle, paint brush, and eraser (adjustable size with `-` / `=`)
- Keypoint / pose annotation with a per-class named schema + skeleton (COCO
  instance model, 3-state visibility)
- Merge annotations; vertex editing (double-click) and handle-based resize/move
  for any selected shape
- Undo / redo (Ctrl+Z / Ctrl+Y)

**AI-assisted**
- SAM 2 box and point prompts (pick tiny / small / large — tiny or small
  recommended)
- Grounding DINO text-prompted detection — single image or batch, with a
  review / accept overlay
- SAM 2 fine-tuning with MLflow experiment tracking
- YOLO training and prediction (detection, segmentation, and pose)

**Images**
- TIFF / CZI stacks with a dimension-assignment dialog and slice navigation
- DICOM converter, slice registration, stack interpolation

**Data management**
- Export: COCO JSON, YOLO v8/v11, Pascal VOC, labeled images, semantic labels
- Import: COCO JSON, YOLO datasets (detection + pose)
- Dataset splitter, image patcher, image augmenter
- Annotations table with Area and per-mask **Detail %** simplification
- Multi-project search with AND/OR queries

**UI**
- Dark mode, on-the-fly font scaling, image-list filter / sort

## Operating system requirements

Built with PyQt6; runs on macOS, Windows, and Linux (CI exercises the test suite
on all three). On Linux you need the standard Qt 6 runtime libraries (notably
`libxcb-cursor0`, `libegl1`, `libgl1`, and the XCB plugin set) —
`sudo apt install libxcb-cursor0 libegl1 libgl1 libxcb-xinerama0 libxkbcommon-x11-0`
covers the common ones on Debian/Ubuntu.

## Installation

This fork is **not published to PyPI** — install from source:

```bash
git clone https://github.com/cofade/digitalsreeni-image-annotator.git
cd digitalsreeni-image-annotator
pip install -e .
```

The Ultralytics library pulls in SAM 2 and PyTorch; SAM 2 weights download
automatically on first use — no separate SAM2 / PyTorch install needed.

### GPU acceleration (NVIDIA)

The PyTorch wheel installed by default from PyPI is **CPU-only** on Windows. If
you have an NVIDIA GPU, SAM and Grounding DINO will run dramatically faster on
CUDA — reinstall PyTorch from the CUDA index:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If `cu128` errors as "no matching distribution", try `cu124` instead. Verify the
install picked up your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see `True` and your GPU name. For other platforms or driver
combinations, use the official selector at <https://pytorch.org/get-started/locally/>.

#### Older NVIDIA GPUs (Pascal / Maxwell)

PyTorch ≥ 2.8 wheels no longer include kernels for GPUs older than Volta (compute
capability < 7.0), e.g. the GTX 10xx series (sm_61). On such cards the app detects
the mismatch, warns once, and automatically runs inference on the CPU instead of
crashing with `CUDA error: no kernel image is available`. To keep using the GPU,
install an older PyTorch that still supports it:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

## Quick start

Run the app:

```bash
digitalsreeni-image-annotator      # or:  sreeni      # or:  python -m digitalsreeni_image_annotator.main
```

Golden path:

1. **New Project** (Ctrl+N).
2. **Add Images / Videos** — images (including TIFF stacks and CZI files) or videos (MP4/AVI/MOV).
3. **Add a class** and pick its colour.
4. **Annotate** — draw manually, or pick a SAM model and use **SAM-box** /
   **SAM-points** (Enter accepts the mask, Esc discards it). For text prompts,
   use Grounding DINO detection.
5. **Export Annotations** — COCO / YOLO / Pascal VOC / labeled / semantic.

For SAM: prefer **SAM 2 tiny** or **small**; the large model can exhaust memory
on modest machines (an out-of-memory load now shows a "pick a smaller model"
dialog rather than crashing).

## Keyboard shortcuts

| Global | Action |
|--------|--------|
| Ctrl+N/O/S | New / Open / Save Project |
| Ctrl+Z / Ctrl+Y (or Ctrl+Shift+Z) | Undo / redo annotation edit |
| Ctrl+Shift+= / Ctrl+Shift+- | UI font bigger / smaller (8-24pt, persisted) |
| Ctrl+Shift+0 | Reset UI font size |
| F1 | Help |

| Canvas | Action |
|--------|--------|
| Ctrl+Wheel | Zoom |
| Ctrl+Drag | Pan |
| Click / Shift+Click (no tool) | Select / toggle mask |
| Drag / Shift+Drag (no tool) | Rubber-band select / add |
| Drag handle / inside (one shape selected) | Resize (scale) / move the shape |
| Double-click | Vertex-edit mode |
| Delete | Delete selected mask(s) — instant, undoable (no confirm dialog) |
| Enter | Finish / Accept (keypoint tool: finish pose early, padding unplaced points v=0) |
| Esc | Cancel in-progress shape **and** return to selection mode (deactivates the tool) |
| Left / Right-click (keypoint tool) | Place next keypoint visible / occluded |
| Backspace (keypoint tool) | Remove the last placed keypoint |
| Right-click a selected pose's point | Toggle its visibility (visible ↔ occluded) |
| Up/Down | Navigate slices |
| -/= | Brush size |

## Documentation

- **Architecture (arc42)**: [docs/README.md](docs/README.md) — building-block view,
  runtime scenarios, cross-cutting concepts, architecture decisions, glossary.
- **Contributor guide**: [CLAUDE.md](CLAUDE.md).
- **Testing**: [TESTING.md](TESTING.md).

## Development

```bash
git clone https://github.com/cofade/digitalsreeni-image-annotator.git
cd digitalsreeni-image-annotator
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"                            # runtime + test extras
pytest tests/ -v                                  # headless: QT_QPA_PLATFORM=offscreen
python -m src.digitalsreeni_image_annotator.main
```

Use feature branches (never commit to `master`), keep the arc42 docs in sync, and
run the senior-reviewer quality gate before opening a PR — see
[CLAUDE.md](CLAUDE.md) for the full workflow.

## Contributing

Contributions are welcome. Fork, branch (`git checkout -b feature/your-feature`),
commit, push, and open a Pull Request against this fork.

## Fork attribution & upstream

This repository is a fork of
[bnsreenu/digitalsreeni-image-annotator](https://github.com/bnsreenu/digitalsreeni-image-annotator)
by **Dr. Sreenivas Bhattiprolu** (@DigitalSreeni on
[YouTube](http://www.youtube.com/c/DigitalSreeni)), who authored the original
polygon / rectangle + SAM annotation tool. This fork diverges substantially:
PyQt6 migration, in-process inference, Grounding-DINO text detection, SAM 2
fine-tuning + MLflow, undo/redo, an annotations table with Detail %,
keypoint/pose annotation, canvas selection + handle editing, and an automated
test suite in CI. Please credit the upstream author for the original work (see
Citing below).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- Dr. Sreenivas Bhattiprolu and his
  [YouTube](http://www.youtube.com/c/DigitalSreeni) community for the original
  tool and tutorials.
- Inspired by the need for efficient image annotation in computer-vision tasks.

## Citing

If you use this software in your research, please cite the upstream project:

Bhattiprolu, S. (2024). DigitalSreeni Image Annotator [Computer software].
https://github.com/bnsreenu/digitalsreeni-image-annotator

```bibtex
@software{digitalsreeni_image_annotator,
  author = {Bhattiprolu, Sreenivas},
  title = {DigitalSreeni Image Annotator},
  year = {2024},
  url = {https://github.com/bnsreenu/digitalsreeni-image-annotator}
}
```
