"""
Unit tests for graceful handling of compressed TIFFs missing imagecodecs
(upstream issue #56).

An LZW TIFF read raises ValueError when imagecodecs is absent; the app
must skip the file with a dialog instead of crashing, and must not leave
a half-added entry.
"""

import pytest
from PyQt6.QtWidgets import QWidget, QListWidget, QComboBox

import src.digitalsreeni_image_annotator.controllers.image_controller as ic_module
from src.digitalsreeni_image_annotator.controllers.image_controller import (
    ImageController,
)

LZW_ERROR = "<COMPRESSION.LZW: 5> requires the 'imagecodecs' package"


class FakeMainWindow(QWidget):
    pass


@pytest.fixture
def mw(qtbot):
    window = FakeMainWindow()
    qtbot.addWidget(window)
    window.image_list = QListWidget(window)
    window.image_filter_combo = QComboBox(window)
    window.image_filter_combo.addItems(
        ["All images", "Without annotations", "With annotations"]
    )
    window.all_images = []
    window.all_annotations = {}
    window.image_slices = {}
    window.image_paths = {}
    window.is_loading_project = False
    window.auto_save = lambda: None
    window.image_controller = ImageController(window)
    return window


def test_lzw_tiff_without_codec_is_skipped_with_dialog(mw, monkeypatch):
    dialogs = []
    monkeypatch.setattr(
        ic_module.QMessageBox, "critical",
        lambda *a, **k: dialogs.append(a),
    )

    def raise_lzw(_path):
        raise ValueError(LZW_ERROR)

    mw.image_controller.load_multi_slice_image = raise_lzw

    # Must not raise.
    mw.image_controller.add_images_to_list(["C:/data/scan.tif"])

    assert len(dialogs) == 1          # one critical dialog shown
    assert mw.all_images == []        # no half-added entry
    assert mw.image_list.count() == 0
    assert "scan.tif" not in mw.image_paths


def test_is_missing_codec_error_matches():
    assert ImageController._is_missing_codec_error(ValueError(LZW_ERROR))
    assert ImageController._is_missing_codec_error(
        ValueError("requires the 'imagecodecs' package")
    )
    # A bare "compression" mention must NOT match — that would swallow
    # unrelated errors behind a misleading "install imagecodecs" dialog.
    assert not ImageController._is_missing_codec_error(
        ValueError("unsupported compression scheme")
    )


def test_unrelated_value_error_is_reraised(mw, monkeypatch):
    monkeypatch.setattr(ic_module.QMessageBox, "critical", lambda *a, **k: None)

    def raise_other(_path):
        raise ValueError("corrupt dimension metadata")

    mw.image_controller.load_multi_slice_image = raise_other

    with pytest.raises(ValueError, match="corrupt dimension metadata"):
        mw.image_controller.add_images_to_list(["C:/data/scan.tif"])
