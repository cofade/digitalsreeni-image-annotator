"""Training and navigating a video's frames (regressions found in manual use).

Two bugs, one theme: a video's frames are ordinary slices everywhere else in
the app, and two places forgot it.

1. ``multidimensional_blockers`` refused to train on any stack or video, citing
   a constraint that predated slice-aware export. Annotating 368 polygons
   across a video's frames produced a perfectly good dataset that the training
   dialog then declined to use.
2. The slice list never switched the canvas on Up/Down, because only
   ``itemClicked`` was connected and a focused ``QListWidget`` consumes the
   arrow keys itself.

These use a REAL video written by the ``make_test_video`` fixture and the real
``load_video`` path, because both bugs were in the wiring between the video
machinery and something that assumed a plain image.
"""

import os

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from src.digitalsreeni_image_annotator.io.export_formats import export_yolo_v5plus


@pytest.fixture
def window(qt_application, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    monkeypatch.setattr(w, "auto_save", lambda *a, **k: None)
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
    yield w
    w.deleteLater()


@pytest.fixture
def video_window(window, make_test_video, tmp_path):
    """A real video added through the real add path.

    ``add_images_to_list`` is what registers the ``is_video`` entry in
    ``all_images`` that the training dialog reads -- calling ``load_video``
    directly loads the frames but leaves the project with no images at all,
    which would make the dialog assertions vacuous.
    """
    path = make_test_video(tmp_path, frames=6)
    window.add_images_to_list([path])
    assert window.all_images, "video was not registered"
    return window, path


def _polygon():
    return {
        "segmentation": [1.0, 1.0, 10.0, 1.0, 10.0, 8.0],
        "category_name": "bee",
        "category_id": 1,
    }


# --- arrow-key navigation --------------------------------------------------


def test_arrow_keys_move_the_canvas_through_the_frames(video_window, qtbot):
    window, _ = video_window
    first = window.current_slice
    assert window.slice_list.count() == 6

    window.slice_list.setFocus()
    qtbot.keyClick(window.slice_list, Qt.Key.Key_Down)

    assert window.current_slice != first, (
        "the row moved but the canvas did not follow it"
    )
    assert window.slice_list.currentItem().text() == window.current_slice
    assert window.current_image is not None


def test_arrow_keys_walk_further_than_one_frame(video_window, qtbot):
    window, _ = video_window
    window.slice_list.setFocus()
    for _ in range(3):
        qtbot.keyClick(window.slice_list, Qt.Key.Key_Down)

    assert window.slice_list.currentRow() == 3
    assert window.current_slice == window.slice_list.item(3).text()


def test_reselecting_the_current_frame_is_a_no_op(video_window):
    """The guard that lets currentRowChanged drive navigation without
    blockSignals at a dozen programmatic-selection call sites."""
    window, _ = video_window
    saves = []
    window.save_current_annotations = lambda *a, **k: saves.append(1)

    window.image_controller.switch_slice(window.slice_list.currentItem())

    assert saves == []


# --- training on a video's frames ------------------------------------------


def test_the_train_dialog_accepts_annotated_video_frames(video_window, qtbot):
    from digitalsreeni_image_annotator.dialogs.train_dialog import TrainDialog

    window, _ = video_window
    window.add_class("bee", QColor("#ffa500"))
    frame = window.slice_list.item(1).text()
    window.all_annotations[frame] = {"bee": [_polygon()]}

    dialog = TrainDialog(window)
    qtbot.addWidget(dialog)

    assert dialog.train_button.isEnabled() is True, dialog.blocker_label.text()
    # The frames are the images, not the video file.
    assert "1 of 6 image(s)" in dialog.data_label.text()


def test_video_frames_are_written_by_the_yolo_export(video_window, tmp_path):
    """The export half. A frame resolves through the video's LazySliceList
    exactly like a stack slice does."""
    window, _ = video_window
    window.add_class("bee", QColor("#ffa500"))
    frames = [window.slice_list.item(i).text() for i in (1, 3)]
    for frame in frames:
        window.all_annotations[frame] = {"bee": [_polygon()]}

    out = tmp_path / "dataset"
    export_yolo_v5plus(
        window.all_annotations,
        {"bee": 1},
        window.image_paths,
        window.slices,
        window.image_slices,
        str(out),
        val_split=0,
    )

    images = os.listdir(out / "images" / "train")
    labels = os.listdir(out / "labels" / "train")
    assert sorted(images) == sorted(f"{name}.png" for name in frames)
    assert sorted(labels) == sorted(f"{name}.txt" for name in frames)
    for label in labels:
        assert (out / "labels" / "train" / label).read_text().strip()


def test_the_export_does_not_materialise_every_frame(video_window, tmp_path):
    """Building a ``{name: qimage}`` map over the whole collection decoded and
    held every frame at once, defeating the bounded lazy cache (#45). Only the
    annotated frames should ever be decoded."""
    window, _ = video_window
    window.add_class("bee", QColor("#ffa500"))
    frame = window.slice_list.item(2).text()
    window.all_annotations[frame] = {"bee": [_polygon()]}

    decoded = []
    real_get = window.slices.get

    def counting_get(name):
        decoded.append(name)
        return real_get(name)

    window.slices.get = counting_get
    export_yolo_v5plus(
        window.all_annotations,
        {"bee": 1},
        window.image_paths,
        window.slices,
        window.image_slices,
        str(tmp_path / "ds"),
        val_split=0,
    )

    assert decoded == [frame], f"decoded {len(decoded)} frame(s), wanted 1"
