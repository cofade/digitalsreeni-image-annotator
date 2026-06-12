"""
Unit tests for the image-list annotation-status filter (upstream issue #27).

Covers ImageController.image_has_annotations and apply_image_filter.
"""

import pytest
from PyQt6.QtWidgets import QWidget, QListWidget, QComboBox

from src.digitalsreeni_image_annotator.controllers.image_controller import (
    ImageController,
)


FILTER_ALL = 0
FILTER_WITHOUT = 1
FILTER_WITH = 2


class FakeMainWindow(QWidget):
    """Minimal stand-in for ImageAnnotator with the state the filter reads."""


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
    window.image_controller = ImageController(window)
    return window


def _add_image(mw, file_name, is_multi_slice=False):
    mw.all_images.append(
        {"file_name": file_name, "is_multi_slice": is_multi_slice}
    )
    mw.image_list.addItem(file_name)


class TestImageHasAnnotations:
    def test_regular_image_without_annotations(self, mw):
        _add_image(mw, "plain.png")
        assert not mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_regular_image_with_empty_class_lists(self, mw):
        _add_image(mw, "plain.png")
        mw.all_annotations["plain.png"] = {"cell": []}
        assert not mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_regular_image_with_annotations(self, mw):
        _add_image(mw, "plain.png")
        mw.all_annotations["plain.png"] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }
        assert mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_multi_slice_with_annotated_slice(self, mw):
        _add_image(mw, "stack.tif", is_multi_slice=True)
        mw.image_slices["stack"] = [("stack_T1_Z1", None), ("stack_T1_Z2", None)]
        mw.all_annotations["stack_T1_Z2"] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }
        assert mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_multi_slice_without_annotated_slices(self, mw):
        _add_image(mw, "stack.tif", is_multi_slice=True)
        mw.image_slices["stack"] = [("stack_T1_Z1", None)]
        mw.all_annotations["stack_T1_Z1"] = {"cell": []}
        assert not mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_multi_slice_prefix_fallback_when_slices_not_loaded(self, mw):
        # Project annotations exist under slice keys, but the slices were
        # never extracted (e.g. load cancelled) — prefix fallback applies.
        _add_image(mw, "stack.tif", is_multi_slice=True)
        mw.all_annotations["stack_T1_Z5_C1"] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }
        assert mw.image_controller.image_has_annotations(mw.all_images[0])

    def test_multi_slice_no_substring_false_positive(self, mw):
        # "bee" must not match keys of "honeybee" (and vice versa).
        _add_image(mw, "bee.tif", is_multi_slice=True)
        mw.all_annotations["honeybee_T1_Z1"] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }
        assert not mw.image_controller.image_has_annotations(mw.all_images[0])


class TestApplyImageFilter:
    @pytest.fixture
    def populated(self, mw):
        _add_image(mw, "annotated.png")
        _add_image(mw, "empty.png")
        mw.all_annotations["annotated.png"] = {
            "cell": [{"segmentation": [0, 0, 1, 0, 1, 1]}]
        }
        # Select the annotated image so the "never hide current" rule is
        # exercised by a dedicated test, not by accident here.
        mw.image_list.setCurrentRow(-1)
        return mw

    def _hidden(self, mw):
        return [
            mw.image_list.isRowHidden(i) for i in range(mw.image_list.count())
        ]

    def test_all_images_shows_everything(self, populated):
        populated.image_filter_combo.setCurrentIndex(FILTER_ALL)
        populated.image_controller.apply_image_filter()
        assert self._hidden(populated) == [False, False]

    def test_without_annotations_hides_annotated(self, populated):
        populated.image_filter_combo.setCurrentIndex(FILTER_WITHOUT)
        populated.image_controller.apply_image_filter()
        assert self._hidden(populated) == [True, False]

    def test_with_annotations_hides_unannotated(self, populated):
        populated.image_filter_combo.setCurrentIndex(FILTER_WITH)
        populated.image_controller.apply_image_filter()
        assert self._hidden(populated) == [False, True]

    def test_current_row_is_never_hidden(self, populated):
        populated.image_list.setCurrentRow(0)  # annotated.png
        populated.image_filter_combo.setCurrentIndex(FILTER_WITHOUT)
        populated.image_controller.apply_image_filter()
        assert self._hidden(populated) == [False, False]

    def test_no_combo_is_a_noop(self, mw):
        _add_image(mw, "plain.png")
        del mw.image_filter_combo
        mw.image_controller.apply_image_filter()  # must not raise
        assert not mw.image_list.isRowHidden(0)

    def test_switching_back_to_all_unhides(self, populated):
        populated.image_filter_combo.setCurrentIndex(FILTER_WITH)
        populated.image_controller.apply_image_filter()
        populated.image_filter_combo.setCurrentIndex(FILTER_ALL)
        populated.image_controller.apply_image_filter()
        assert self._hidden(populated) == [False, False]
