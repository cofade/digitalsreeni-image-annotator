"""
Unit tests for alphabetical image-list sorting (upstream issue #60).

sort_image_list must order the list case-insensitively, keep the
all_images model aligned with the list rows (positional invariant used
by COCO import), and never fire a spurious switch_image.
"""

import pytest
from PyQt6.QtWidgets import QWidget, QListWidget, QComboBox

from src.digitalsreeni_image_annotator.controllers.image_controller import (
    ImageController,
)


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
    window.image_controller = ImageController(window)
    return window


def _populate(mw, names):
    # Populate out of order, mimicking the model+view pairing that
    # add_images_to_list produces before a sort.
    for n in names:
        mw.all_images.append({"file_name": n, "is_multi_slice": False})
        mw.image_list.addItem(n)


def _list_texts(mw):
    return [mw.image_list.item(i).text() for i in range(mw.image_list.count())]


def test_sorts_alphabetically(mw):
    _populate(mw, ["banana.png", "apple.png", "cherry.png"])
    mw.image_controller.sort_image_list()
    assert _list_texts(mw) == ["apple.png", "banana.png", "cherry.png"]


def test_sort_is_case_insensitive(mw):
    _populate(mw, ["Zebra.png", "apple.png", "Banana.png"])
    mw.image_controller.sort_image_list()
    assert _list_texts(mw) == ["apple.png", "Banana.png", "Zebra.png"]


def test_model_and_view_stay_aligned(mw):
    _populate(mw, ["d.png", "a.png", "c.png", "b.png"])
    mw.image_controller.sort_image_list()
    assert _list_texts(mw) == [info["file_name"] for info in mw.all_images]


def test_sort_fires_no_switch_image(mw):
    _populate(mw, ["b.png", "a.png"])
    calls = []
    mw.switch_image = lambda item: calls.append(item)
    mw.image_controller.switch_image = lambda item: calls.append(item)
    mw.image_list.setCurrentRow(0)
    calls.clear()
    mw.image_controller.sort_image_list()  # no select_name, do_switch=False
    assert calls == []


def test_selection_preserved_across_sort(mw):
    _populate(mw, ["b.png", "a.png", "c.png"])
    mw.image_list.setCurrentRow(0)  # b.png
    mw.image_controller.sort_image_list()
    assert mw.image_list.currentItem().text() == "b.png"


def test_select_name_and_switch(mw):
    _populate(mw, ["b.png", "a.png"])
    calls = []
    mw.switch_image = lambda item: calls.append(item.text())
    mw.image_controller.switch_image = lambda item: calls.append(item.text())
    mw.image_controller.sort_image_list(select_name="a.png", do_switch=True)
    assert mw.image_list.currentItem().text() == "a.png"
    assert calls == ["a.png"]
