"""Image / multi-dimensional slice loading and navigation controller.

Extracted from `ImageAnnotator` to give image I/O its own home. Owns:

- Loading from disk (PNG/JPG, TIFF, CZI)
- Multi-dimensional image handling: dimension assignment dialog,
  per-axis slicing, slice list population
- Image / slice navigation (switch_image, switch_slice, activate_slice)
- Display and per-image lifecycle (remove_image, delete_selected_image,
  redefine_dimensions)

State (`current_image`, `current_slice`, `slices`, `image_paths`,
`image_slices`, `image_dimensions`, `image_shapes`, `all_images`,
`image_file_name`, etc.) still lives on the main window and is read here
via `self.mw`. A future phase may migrate ownership of selected
attributes to the controller — for now this is pure method relocation.

The `DimensionDialog` widget lives here too — it is only used by
`process_multidimensional_image`.
"""

import os

import numpy as np
from czifile import CziFile
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from tifffile import TiffFile

from ..core import image_utils

from ..core.logging_config import get_logger

logger = get_logger(__name__)


class DimensionDialog(QDialog):
    def __init__(self, shape, file_name, parent=None, default_dimensions=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Dimensions")
        layout = QVBoxLayout(self)

        file_name_label = QLabel(f"File: {file_name}")
        file_name_label.setWordWrap(True)
        layout.addWidget(file_name_label)

        dim_widget = QWidget()
        dim_layout = QGridLayout(dim_widget)
        self.combos = []
        self.shape = shape
        dimensions = ["T", "Z", "C", "S", "H", "W"]
        for i, dim in enumerate(shape):
            dim_layout.addWidget(QLabel(f"Dimension {i} (size {dim}):"), i, 0)
            combo = QComboBox()
            combo.addItems(dimensions)
            if default_dimensions and i < len(default_dimensions):
                combo.setCurrentText(default_dimensions[i])
            dim_layout.addWidget(combo, i, 1)
            self.combos.append(combo)
        layout.addWidget(dim_widget)

        self.button = QPushButton("OK")
        self.button.clicked.connect(self.accept)
        layout.addWidget(self.button)

        self.setMinimumWidth(300)

    def get_dimensions(self):
        return [combo.currentText() for combo in self.combos]


class ImageController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window

    def update_image_list(self):
        # Rebuild (and sort) the list, preserving the current selection
        # without switching images.
        self.sort_image_list()

    def sort_image_list(self, select_name=None, do_switch=False):
        """Populate image_list in alphabetical order (upstream issue #60).

        Sorts the model (`all_images`) and the view together so the
        `all_images[i]` ↔ `image_list.item(i)` positional invariant holds
        (relied on by COCO import reconciliation). `setSortingEnabled` is
        deliberately NOT used: `currentRowChanged` is wired to
        `switch_image`, so a live re-sort would fire spurious image
        switches. We rebuild with signals blocked instead, then re-select
        explicitly.

        select_name: file to select after the rebuild (defaults to the
        previously-current item). do_switch: call switch_image once for
        the selected item (used when adding new images).
        """
        current = None
        if self.mw.image_list.currentItem() is not None:
            current = self.mw.image_list.currentItem().text()

        self.mw.all_images.sort(
            key=lambda info: (
                info.get("file_name", "").casefold(),
                info.get("file_name", ""),
            )
        )

        self.mw.image_list.blockSignals(True)
        self.mw.image_list.clear()
        for info in self.mw.all_images:
            self.mw.image_list.addItem(info["file_name"])
        self.mw.image_list.blockSignals(False)

        self.apply_image_filter()

        target = select_name if select_name is not None else current
        if target is not None:
            items = self.mw.image_list.findItems(
                target, Qt.MatchFlag.MatchExactly
            )
            if items:
                self.mw.image_list.blockSignals(True)
                self.mw.image_list.setCurrentItem(items[0])
                self.mw.image_list.blockSignals(False)
                if do_switch:
                    self.switch_image(items[0])

    def image_has_annotations(self, image_info):
        """True if the image (or, for multi-dim images, any of its slices)
        has at least one annotation."""

        def _non_empty(by_class):
            return bool(by_class) and any(by_class.values())

        file_name = image_info["file_name"]
        if _non_empty(self.mw.all_annotations.get(file_name, {})):
            return True

        if image_info.get("is_multi_slice", False):
            base_name = os.path.splitext(file_name)[0]
            slices = self.mw.image_slices.get(base_name)
            if slices:
                return any(
                    _non_empty(self.mw.all_annotations.get(slice_name, {}))
                    for slice_name, _ in slices
                )
            # Slices not extracted yet (e.g. load cancelled) — slice keys
            # are f"{base_name}_T1_Z5_..." so a "{base_name}_" prefix match
            # is exact enough; a bare substring match would not be. Caveat:
            # this also matches "{base_name}_8bit" artifact keys, which
            # redefine_dimensions deliberately excludes — acceptable here
            # since an _8bit key with annotations still means "this image
            # has annotations".
            prefix = base_name + "_"
            return any(
                key.startswith(prefix) and _non_empty(by_class)
                for key, by_class in self.mw.all_annotations.items()
            )

        return False

    def apply_image_filter(self):
        """Hide image-list rows that don't match the annotation-status
        filter (upstream issue #27).

        Rows are hidden via setRowHidden, never removed: other code
        (DINO batch navigation, COCO import) iterates the list by row
        index, and hiding fires no currentRowChanged so it cannot
        trigger a spurious switch_image.

        A non-matching row is hidden even when it is the current
        selection — hiding does not change `current_image`, so the canvas
        keeps showing the worked-on image while its row leaves the list
        (e.g. the image just gained its first annotation under the
        "Without annotations" filter). Keyboard nav skips hidden rows.
        """
        combo = getattr(self.mw, "image_filter_combo", None)
        if combo is None:
            return
        mode = combo.currentIndex()  # 0 = all, 1 = without, 2 = with
        if mode == 0:
            # Default case runs on every update_slice_list_colors —
            # keep it a plain unhide pass with no annotation scans.
            for i in range(self.mw.image_list.count()):
                self.mw.image_list.setRowHidden(i, False)
            return
        infos = {info["file_name"]: info for info in self.mw.all_images}
        for i in range(self.mw.image_list.count()):
            info = infos.get(self.mw.image_list.item(i).text())
            annotated = bool(info) and self.image_has_annotations(info)
            hide = annotated if mode == 1 else not annotated
            self.mw.image_list.setRowHidden(i, hide)

    def setup_slice_list(self):
        self.mw.slice_list = QListWidget()
        self.mw.slice_list.itemClicked.connect(self.switch_slice)
        self.mw.image_list_layout.addWidget(QLabel("Slices:"))
        self.mw.image_list_layout.addWidget(self.mw.slice_list)

    def open_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(
            self.mw,
            "Open Images",
            "",
            "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)",
        )
        if file_names:
            self.mw.image_list.clear()
            self.mw.image_paths.clear()
            self.mw.all_images.clear()
            self.mw.slice_list.clear()
            self.mw.slices.clear()
            self.mw.current_stack = None
            self.mw.current_slice = None
            self.add_images_to_list(file_names)

    def add_images_to_list(self, file_names):
        first_added_name = None
        for file_name in file_names:
            base_name = os.path.basename(file_name)
            if base_name not in self.mw.image_paths:
                image_info = {
                    "file_name": base_name,
                    "height": 0,
                    "width": 0,
                    "id": len(self.mw.all_images) + 1,
                    "is_multi_slice": False,
                }

                if file_name.lower().endswith((".tif", ".tiff", ".czi")):
                    try:
                        self.load_multi_slice_image(file_name)
                    except ValueError as e:
                        # LZW/compressed TIFFs need the optional imagecodecs
                        # package; without it tifffile raises ValueError and
                        # the app used to crash (#56). Skip the file with an
                        # actionable message instead of a half-added entry.
                        if self._is_missing_codec_error(e):
                            QMessageBox.critical(
                                self.mw,
                                "Cannot open TIFF",
                                f"'{base_name}' uses a compression that requires "
                                "the 'imagecodecs' package, which is not "
                                "installed.\n\nInstall it with:\n"
                                "    pip install imagecodecs\n\n"
                                "then reopen the image.",
                            )
                            continue
                        raise
                    base_name_without_ext = os.path.splitext(base_name)[0]
                    if (
                        base_name_without_ext in self.mw.image_slices
                        and self.mw.image_slices[base_name_without_ext]
                    ):
                        first_slice_name, first_slice = self.mw.image_slices[
                            base_name_without_ext
                        ][0]
                        image_info["height"] = first_slice.height()
                        image_info["width"] = first_slice.width()
                        image_info["is_multi_slice"] = True
                        image_info["dimensions"] = self.mw.image_dimensions.get(
                            base_name_without_ext, []
                        )
                        image_info["shape"] = self.mw.image_shapes.get(
                            base_name_without_ext, []
                        )
                else:
                    image = QImage(file_name)
                    image_info["height"] = image.height()
                    image_info["width"] = image.width()

                self.mw.all_images.append(image_info)
                if first_added_name is None:
                    first_added_name = base_name

                self.mw.image_paths[base_name] = file_name

        # Rebuild the list in sorted order and select/switch to the first
        # newly added image. Skipped during project load (the list is
        # rebuilt once via update_ui afterwards, and load picks row 0) to
        # avoid an O(n^2) re-sort per image.
        if first_added_name is not None and not self.mw.is_loading_project:
            self.sort_image_list(select_name=first_added_name, do_switch=True)
        else:
            self.apply_image_filter()

        if not self.mw.is_loading_project:
            self.mw.auto_save()

    @staticmethod
    def _is_missing_codec_error(exc):
        """True if a tifffile read failed because the imagecodecs package
        is unavailable for the TIFF's compression — e.g. LZW (#56).

        Matches only the reliable 'imagecodecs' token: tifffile names the
        package in every such message. A broader 'compression' match would
        silently swallow unrelated ValueErrors behind a misleading dialog.
        """
        return "imagecodecs" in str(exc).lower()

    def update_all_images(self, new_image_info):
        for info in new_image_info:
            if not any(
                img["file_name"] == info["file_name"] for img in self.mw.all_images
            ):
                self.mw.all_images.append(info)

    def switch_slice(self, item):
        if item is None:
            return
        # check_unsaved_changes prompts the user and commits/discards
        # all dirty tool handlers; returns False on Cancel.
        if not self.mw.image_label.check_unsaved_changes():
            return

        self.mw.save_current_annotations()
        self.mw.image_label.clear_temp_sam_prediction()
        # Exit vertex-edit mode (as switch_image does) so editing_polygon /
        # _editing_polygon_orig don't linger onto the next slice (ADR-026).
        self.mw.image_label.exit_editing_mode()
        self.mw.annotation_controller.reset_coalesce()

        slice_name = item.text()
        for name, qimage in self.mw.slices:
            if name == slice_name:
                self.mw.current_image = qimage
                self.mw.current_slice = name
                self.display_image()
                self.mw.load_image_annotations()
                self.mw.update_annotation_list()
                self.mw.clear_highlighted_annotation()
                self.mw.image_label.reset_annotation_state()
                self.mw.image_label.clear_current_annotation()
                self.mw.update_image_info()
                break

        self.mw.image_label.update()
        self.mw.update_slice_list_colors()

        self.mw.set_zoom(1.0)

        self.mw._refresh_dino_temp_for_current()

    def switch_image(self, item):
        if item is None:
            return
        if not self.mw.image_label.check_unsaved_changes():
            return

        current_item = self.mw.image_list.currentItem()

        if not self.mw.check_temp_annotations():
            self.mw.image_list.setCurrentItem(current_item)
            return

        self.mw.save_current_annotations()
        self.mw.image_label.clear_temp_sam_prediction()
        self.mw.image_label.exit_editing_mode()
        self.mw.annotation_controller.reset_coalesce()

        file_name = item.text()
        logger.debug(f"Switching to image: {file_name}")

        image_info = next(
            (img for img in self.mw.all_images if img["file_name"] == file_name), None
        )

        if image_info:
            self.mw.image_file_name = file_name
            image_path = self.mw.image_paths.get(file_name)

            if not image_path:
                image_path = os.path.join(
                    self.mw.current_project_dir, "images", file_name
                )

            if image_path and os.path.exists(image_path):
                if image_info.get("is_multi_slice", False):
                    base_name = os.path.splitext(file_name)[0]
                    if base_name in self.mw.image_slices:
                        self.mw.slices = self.mw.image_slices[base_name]
                        if self.mw.slices:
                            self.mw.current_image = self.mw.slices[0][1]
                            self.mw.current_slice = self.mw.slices[0][0]
                            self.update_slice_list()
                            self.activate_slice(self.mw.current_slice)
                    else:
                        self.load_multi_slice_image(
                            image_path,
                            image_info.get("dimensions"),
                            image_info.get("shape"),
                        )
                else:
                    self.load_regular_image(image_path)
                    self.display_image()
                    self.clear_slice_list()

                self.mw.load_image_annotations()
                self.mw.update_annotation_list()
                self.mw.clear_highlighted_annotation()
                self.mw.image_label.update()
                self.mw.image_label.reset_annotation_state()
                self.mw.image_label.clear_current_annotation()
                self.mw.update_image_info()

                self.mw.adjust_zoom_to_fit()
            else:
                self.mw.current_image = None
                self.mw.image_label.clear()
                self.mw.load_image_annotations()
                self.mw.update_annotation_list()
                self.mw.update_image_info()

            self.mw.image_list.setCurrentItem(item)
            self.mw.image_label.update()
            self.mw.update_slice_list_colors()
        else:
            self.mw.current_image = None
            self.mw.current_slice = None
            self.mw.image_label.clear()
            self.mw.update_image_info()
            self.clear_slice_list()

        self.mw._refresh_dino_temp_for_current()

    def activate_current_slice(self):
        if self.mw.current_slice:
            items = self.mw.slice_list.findItems(
                self.mw.current_slice, Qt.MatchFlag.MatchExactly
            )
            if items:
                self.mw.slice_list.setCurrentItem(items[0])

            self.mw.load_image_annotations()
            self.mw.image_label.update()
            self.mw.update_annotation_list()

    def load_image(self, image_path):
        extension = os.path.splitext(image_path)[1].lower()
        if extension in [".tif", ".tiff"]:
            self.load_tiff(image_path)
        elif extension == ".czi":
            self.load_czi(image_path)
        else:
            self.load_regular_image(image_path)

    def load_tiff(
        self, image_path, dimensions=None, shape=None, force_dimension_dialog=False
    ):
        logger.debug(f"Loading TIFF file: {image_path}")
        axes_hint = None
        with TiffFile(image_path) as tif:
            logger.debug(f"TIFF tags: {tif.pages[0].tags}")

            try:
                metadata = tif.pages[0].tags["ImageDescription"].value
                logger.debug(f"TIFF metadata: {metadata}")
            except KeyError:
                logger.debug("No ImageDescription metadata found")

            try:
                series_axes = tif.series[0].axes if tif.series else None
                if series_axes:
                    axis_map = {
                        "T": "T", "Z": "Z", "C": "C", "S": "S",
                        "Y": "H", "X": "W",
                    }
                    mapped = [axis_map.get(a) for a in series_axes]
                    if all(a is not None for a in mapped):
                        axes_hint = mapped
                        logger.debug(f"TIFF series axes: {series_axes} → dimension hint: {axes_hint}")
                    else:
                        unknown = [a for a in series_axes if axis_map.get(a) is None]
                        logger.debug(f"TIFF series axes had unknown labels {unknown}, no hint applied")
            except Exception:
                logger.exception("Could not read TIFF series axes")

            if len(tif.pages) > 1:
                logger.debug(f"Multi-page TIFF detected. Number of pages: {len(tif.pages)}")
                image_array = tif.asarray()
            else:
                logger.debug("Single-page TIFF detected.")
                image_array = tif.pages[0].asarray()

            logger.debug(f"Image array shape: {image_array.shape}")
            logger.debug(f"Image array dtype: {image_array.dtype}")
            logger.debug(f"Image min: {image_array.min()}, max: {image_array.max()}")

        if dimensions and shape and not force_dimension_dialog:
            logger.debug(f"Using stored dimensions: {dimensions}")
            logger.debug(f"Using stored shape: {shape}")
            image_array = image_array.reshape(shape)
        else:
            logger.debug("Processing as new image or forcing dimension dialog.")
            dimensions = None

        self.process_multidimensional_image(
            image_array, image_path, dimensions, force_dimension_dialog,
            axes_hint=axes_hint,
        )

    def load_czi(
        self, image_path, dimensions=None, shape=None, force_dimension_dialog=False
    ):
        logger.debug(f"Loading CZI file: {image_path}")
        with CziFile(image_path) as czi:
            image_array = czi.asarray()
            logger.debug(f"CZI array shape: {image_array.shape}")
            logger.debug(f"CZI array dtype: {image_array.dtype}")
            logger.debug(f"CZI array min: {image_array.min()}, max: {image_array.max()}")

        if dimensions and shape and not force_dimension_dialog:
            logger.debug(f"Using stored dimensions: {dimensions}")
            logger.debug(f"Using stored shape: {shape}")
            image_array = image_array.reshape(shape)
        else:
            logger.debug("Processing as new image or forcing dimension dialog.")
            dimensions = None

        self.process_multidimensional_image(
            image_array, image_path, dimensions, force_dimension_dialog
        )

    def load_regular_image(self, image_path):
        self.mw.current_image = QImage(image_path)
        self.mw.slices = []
        self.mw.slice_list.clear()
        self.mw.current_slice = None

    def load_multi_slice_image(self, image_path, dimensions=None, shape=None):
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        logger.debug(f"Loading multi-slice image: {image_path}")
        logger.debug(f"Base name: {base_name}")

        if dimensions and shape:
            logger.debug(f"Using stored dimensions: {dimensions}")
            logger.debug(f"Using stored shape: {shape}")
            self.mw.image_dimensions[base_name] = dimensions
            self.mw.image_shapes[base_name] = shape
            if image_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(image_path, dimensions, shape)
            elif image_path.lower().endswith(".czi"):
                self.load_czi(image_path, dimensions, shape)
        else:
            logger.debug("No stored dimensions or shape, loading as new image")
            if image_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(image_path)
            elif image_path.lower().endswith(".czi"):
                self.load_czi(image_path)

        logger.debug(f"Loaded multi-slice image: {file_name}")
        logger.debug(f"Dimensions: {self.mw.image_dimensions.get(base_name, 'Not found')}")
        logger.debug(f"Shape: {self.mw.image_shapes.get(base_name, 'Not found')}")
        logger.debug(f"Number of slices: {len(self.mw.slices)}")

        if self.mw.slices:
            self.mw.current_image = self.mw.slices[0][1]
            self.mw.current_slice = self.mw.slices[0][0]

            self.update_slice_list()
            self.mw.slice_list.setCurrentRow(0)
            self.activate_slice(self.mw.current_slice)
            logger.debug(f"Activated first slice: {self.mw.current_slice}")
        else:
            logger.warning("No slices were loaded")
            self.mw.current_image = None
            self.mw.current_slice = None

        self.update_slice_list()
        self.mw.image_label.update()

    def process_multidimensional_image(
        self, image_array, image_path, dimensions=None,
        force_dimension_dialog=False, axes_hint=None,
    ):
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        logger.debug(f"Processing file: {file_name}")
        logger.debug(f"Image array shape: {image_array.shape}")
        logger.debug(f"Image array dtype: {image_array.dtype}")

        if dimensions is None or force_dimension_dialog:
            if image_array.ndim > 2:
                # ndim≥5 had a `[-ndim:]` slice bug that produced 2560 wrong
                # slices on a 5D TZCYX file — see arc42.
                if axes_hint and len(axes_hint) == image_array.ndim:
                    default_dimensions = list(axes_hint)
                    logger.debug(f"Applying axes hint as default dims: {default_dimensions}")
                else:
                    if axes_hint and len(axes_hint) != image_array.ndim:
                        logger.debug(
                            f"Ignoring axes hint (length {len(axes_hint)} "
                            f"vs ndim {image_array.ndim})"
                        )
                    ndim_defaults = {
                        3: ["Z", "H", "W"],
                        4: ["T", "Z", "H", "W"],
                        5: ["T", "Z", "C", "H", "W"],
                        6: ["T", "Z", "C", "S", "H", "W"],
                    }
                    default_dimensions = ndim_defaults.get(
                        image_array.ndim,
                        ["T"] * max(0, image_array.ndim - 2) + ["H", "W"],
                    )

                progress = QProgressDialog(
                    "Assigning dimensions...", "Cancel", 0, 100, self.mw
                )
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(10)
                QApplication.processEvents()

                while True:
                    dialog = DimensionDialog(
                        image_array.shape, file_name, self.mw, default_dimensions
                    )
                    progress.setValue(50)
                    QApplication.processEvents()
                    if dialog.exec():
                        dimensions = dialog.get_dimensions()
                        logger.debug(f"Assigned dimensions: {dimensions}")
                        if "H" in dimensions and "W" in dimensions:
                            self.mw.image_dimensions[base_name] = dimensions
                            break
                        else:
                            QMessageBox.warning(
                                self.mw,
                                "Invalid Dimensions",
                                "You must assign both H and W dimensions.",
                            )
                    else:
                        progress.close()
                        return
                progress.setValue(100)
                progress.close()
            else:
                dimensions = ["H", "W"]
                self.mw.image_dimensions[base_name] = dimensions

        self.mw.image_shapes[base_name] = image_array.shape
        logger.debug(f"Final assigned dimensions: {self.mw.image_dimensions[base_name]}")
        logger.debug(f"Image shape: {self.mw.image_shapes[base_name]}")

        if self.mw.image_dimensions[base_name]:
            self.create_slices(
                image_array, self.mw.image_dimensions[base_name], image_path
            )
        else:
            rgb_image = image_utils.convert_to_8bit_rgb(image_array)
            self.mw.current_image = image_utils.array_to_qimage(rgb_image)
            self.mw.slices = []
            self.mw.slice_list.clear()

        if self.mw.slices:
            self.mw.current_image = self.mw.slices[0][1]
            self.mw.current_slice = self.mw.slices[0][0]
            self.mw.slice_list.setCurrentRow(0)
            self.mw.load_image_annotations()
            self.mw.image_label.update()

        self.mw.update_image_info()

        self.update_slice_list()
        self.mw.update_annotation_list()
        self.mw.image_label.update()

    def create_slices(self, image_array, dimensions, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        slices = []
        self.mw.slice_list.clear()

        logger.debug(f"Creating slices for {base_name}")
        logger.debug(f"Dimensions: {dimensions}")
        logger.debug(f"Image array shape: {image_array.shape}")

        progress = QProgressDialog("Loading slices...", "Cancel", 0, 100, self.mw)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        if image_array.ndim == 2:
            progress.setValue(50)
            QApplication.processEvents()
            normalized_array = image_utils.normalize_array(image_array)
            qimage = image_utils.array_to_qimage(normalized_array)
            slice_name = f"{base_name}"
            slices.append((slice_name, qimage))
            self.add_slice_to_list(slice_name)
        else:
            slice_indices = [
                i for i, dim in enumerate(dimensions) if dim not in ["H", "W"]
            ]

            total_slices = np.prod([image_array.shape[i] for i in slice_indices])
            for idx, _ in enumerate(
                np.ndindex(tuple(image_array.shape[i] for i in slice_indices))
            ):
                if progress.wasCanceled():
                    break

                full_idx = [slice(None)] * len(dimensions)
                for i, val in zip(slice_indices, _):
                    full_idx[i] = val

                slice_array = image_array[tuple(full_idx)]
                rgb_slice = image_utils.convert_to_8bit_rgb(slice_array)
                qimage = image_utils.array_to_qimage(rgb_slice)

                slice_name = f"{base_name}_{'_'.join([f'{dimensions[i]}{val+1}' for i, val in zip(slice_indices, _)])}"
                slices.append((slice_name, qimage))

                self.add_slice_to_list(slice_name)

                progress_value = int((idx + 1) / total_slices * 100)
                progress.setValue(progress_value)
                QApplication.processEvents()

        progress.setValue(100)

        self.mw.image_slices[base_name] = slices
        self.mw.slices = slices

        if slices:
            self.mw.current_image = slices[0][1]
            self.mw.current_slice = slices[0][0]
            self.mw.slice_list.setCurrentRow(0)

            self.activate_slice(self.mw.current_slice)

            slice_info = f"Total slices: {len(slices)}"
            for dim, size in zip(dimensions, image_array.shape):
                if dim not in ["H", "W"]:
                    slice_info += f", {dim}: {size}"
            self.mw.update_image_info(additional_info=slice_info)
        else:
            logger.warning("No slices were created")

        logger.info(f"Created {len(slices)} slices for {base_name}")
        return slices

    def add_slice_to_list(self, slice_name):
        item = QListWidgetItem(slice_name)

        if self.mw.dark_mode:
            item.setBackground(QColor(40, 40, 40))
            if slice_name in self.mw.all_annotations:
                item.setForeground(QColor(235, 235, 235))
                item.setBackground(QColor(58, 95, 140))
            else:
                item.setForeground(QColor(200, 200, 200))
        else:
            item.setBackground(QColor(240, 240, 240))
            if slice_name in self.mw.all_annotations:
                item.setForeground(QColor(255, 255, 255))
                item.setBackground(QColor(70, 130, 180))
            else:
                item.setForeground(QColor(0, 0, 0))

        self.mw.slice_list.addItem(item)

    def activate_slice(self, slice_name):
        self.mw.current_slice = slice_name
        self.mw.image_file_name = slice_name
        self.mw.load_image_annotations()
        self.mw.update_annotation_list()

        for name, qimage in self.mw.slices:
            if name == slice_name:
                self.mw.current_image = qimage
                self.display_image()
                break

        self.mw.image_label.update()

        items = self.mw.slice_list.findItems(slice_name, Qt.MatchFlag.MatchExactly)
        if items:
            self.mw.slice_list.setCurrentItem(items[0])

    def update_slice_list(self):
        self.mw.slice_list.clear()
        for slice_name, _ in self.mw.slices:
            item = QListWidgetItem(slice_name)
            if slice_name in self.mw.all_annotations:
                item.setForeground(QColor(Qt.GlobalColor.green))
            else:
                item.setForeground(
                    QColor(Qt.GlobalColor.black)
                    if not self.mw.dark_mode
                    else QColor(Qt.GlobalColor.white)
                )
            self.mw.slice_list.addItem(item)

        if self.mw.current_slice:
            items = self.mw.slice_list.findItems(
                self.mw.current_slice, Qt.MatchFlag.MatchExactly
            )
            if items:
                self.mw.slice_list.setCurrentItem(items[0])

    def clear_slice_list(self):
        self.mw.slice_list.clear()
        self.mw.slices = []
        self.mw.current_slice = None

    def is_multi_dimensional(self, file_name):
        return file_name.lower().endswith((".tif", ".tiff", ".czi"))

    def redefine_dimensions(self, file_name):
        file_path = self.mw.image_paths.get(file_name)
        if not file_path or not file_path.lower().endswith((".tif", ".tiff", ".czi")):
            return

        reply = QMessageBox.warning(
            self.mw,
            "Redefine Dimensions",
            "Redefining dimensions will cause all associated annotations to be lost. "
            "Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            base_name = os.path.splitext(file_name)[0]

            logger.debug(f"Removing annotations for image: {base_name}")

            keys_to_remove = [
                key
                for key in self.mw.all_annotations.keys()
                if key == base_name
                or (
                    key.startswith(f"{base_name}_")
                    and not key.startswith(f"{base_name}_8bit")
                )
            ]

            logger.debug(f"Keys to remove: {keys_to_remove}")

            for key in keys_to_remove:
                del self.mw.all_annotations[key]

            if base_name in self.mw.image_slices:
                del self.mw.image_slices[base_name]

            if self.mw.image_file_name == file_name:
                self.mw.current_image = None
                self.mw.image_label.clear()

            if file_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(file_path, force_dimension_dialog=True)
            elif file_path.lower().endswith(".czi"):
                self.load_czi(file_path, force_dimension_dialog=True)

            self.update_slice_list()
            self.mw.update_annotation_list()
            self.mw.image_label.update()

            QMessageBox.information(
                self.mw,
                "Dimensions Redefined",
                "The dimensions have been redefined and the image reloaded. "
                "All previous annotations for this image have been removed.",
            )

    def remove_image(self):
        current_item = self.mw.image_list.currentItem()
        if current_item:
            file_name = current_item.text()

            self.mw.image_list.takeItem(self.mw.image_list.row(current_item))
            self.mw.image_paths.pop(file_name, None)
            self.mw.all_images = [
                img for img in self.mw.all_images if img["file_name"] != file_name
            ]

            self.mw.all_annotations.pop(file_name, None)

            base_name = os.path.splitext(file_name)[0]
            if base_name in self.mw.image_slices:
                for slice_name, _ in self.mw.image_slices[base_name]:
                    self.mw.all_annotations.pop(slice_name, None)
                del self.mw.image_slices[base_name]

                self.mw.slice_list.clear()

            if self.mw.image_file_name == file_name:
                self.mw.current_image = None
                self.mw.image_file_name = ""
                self.mw.current_slice = None
                self.mw.image_label.clear()
                self.mw.annotation_list.setRowCount(0)

            if self.mw.image_list.count() > 0:
                next_item = self.mw.image_list.item(0)
                self.mw.image_list.setCurrentItem(next_item)
                self.switch_image(next_item)
            else:
                self.mw.current_image = None
                self.mw.image_file_name = ""
                self.mw.current_slice = None
                self.mw.image_label.clear()
                self.mw.annotation_list.setRowCount(0)
                self.mw.slice_list.clear()

            self.mw.update_ui()
            self.mw.auto_save()

    def delete_selected_image(self):
        current_item = self.mw.image_list.currentItem()
        if current_item:
            file_name = current_item.text()
            reply = QMessageBox.question(
                self.mw,
                "Delete Image",
                f"Are you sure you want to delete the image '{file_name}'?\n\n"
                "This will remove the image and all its associated annotations.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.mw.image_list.takeItem(self.mw.image_list.row(current_item))
                self.mw.image_paths.pop(file_name, None)
                self.mw.all_images = [
                    img for img in self.mw.all_images if img["file_name"] != file_name
                ]

                self.mw.all_annotations.pop(file_name, None)

                base_name = os.path.splitext(file_name)[0]
                if base_name in self.mw.image_slices:
                    for slice_name, _ in self.mw.image_slices[base_name]:
                        self.mw.all_annotations.pop(slice_name, None)
                    del self.mw.image_slices[base_name]

                    self.mw.slice_list.clear()

                if self.mw.image_file_name == file_name:
                    self.mw.current_image = None
                    self.mw.image_file_name = ""
                    self.mw.current_slice = None
                    self.mw.image_label.clear()
                    self.mw.annotation_list.setRowCount(0)

                if self.mw.image_list.count() > 0:
                    next_item = self.mw.image_list.item(0)
                    self.mw.image_list.setCurrentItem(next_item)
                    self.switch_image(next_item)
                else:
                    self.mw.current_image = None
                    self.mw.image_file_name = ""
                    self.mw.current_slice = None
                    self.mw.image_label.clear()
                    self.mw.annotation_list.setRowCount(0)
                    self.mw.slice_list.clear()

                self.mw.update_ui()

                QMessageBox.information(
                    self.mw,
                    "Image Deleted",
                    f"The image '{file_name}' has been deleted.",
                )

    def display_image(self):
        if self.mw.current_image:
            if isinstance(self.mw.current_image, QImage):
                pixmap = QPixmap.fromImage(self.mw.current_image)
            elif isinstance(self.mw.current_image, QPixmap):
                pixmap = self.mw.current_image
            else:
                logger.warning(f"Unexpected image type: {type(self.mw.current_image)}")
                return

            if not pixmap.isNull():
                self.mw.image_label.setPixmap(pixmap)
                self.mw.image_label.adjustSize()
            else:
                logger.warning("Null pixmap")
        else:
            self.mw.image_label.clear()
            logger.debug("No current image to display")
