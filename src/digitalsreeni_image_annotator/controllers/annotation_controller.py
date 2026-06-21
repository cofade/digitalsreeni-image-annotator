"""Annotation CRUD + UI list management controller.

Extracted from `ImageAnnotator`. Owns the annotation list widget
plumbing, the per-image annotation cache sync (`load_image_annotations`
/ `save_current_annotations`), polygon and rectangle commit paths,
merge/delete/change-class workflows, sort & renumber, the COCO-load
path, and the edit-mode lifecycle.

This is the cluster `ImageLabel` mutates most directly (via
`main_window.add_annotation_to_list(...)` etc.). Phase 5 keeps the
delegation pattern on `ImageAnnotator`; Phase 6 will replace
`ImageLabel`'s `main_window.*` calls with Qt signals targeting these
controller methods.

State stays on the main window:
- `all_annotations` (dict[image_name, dict[class_name, list[ann]]])
- `image_label.annotations` (per-image working copy)
- `editing_mode`, `loaded_json`, `current_sort_method`
- All Qt widgets (`annotation_list`, `merge_button`, `change_class_button`)
"""

import copy
import json

from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
)
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from ..core.constants import default_class_color
from ..utils import calculate_area, calculate_bbox


class AnnotationController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window

    # --- COCO conversion helper ---

    def create_coco_annotation(self, ann, image_id, annotation_id):
        coco_ann = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": ann["category_id"],
            "area": calculate_area(ann),
            "iscrowd": 0,
        }

        if "segmentation" in ann:
            coco_ann["segmentation"] = [ann["segmentation"]]
            coco_ann["bbox"] = calculate_bbox(ann["segmentation"])
        elif "bbox" in ann:
            coco_ann["bbox"] = ann["bbox"]

        return coco_ann

    # --- List widget updates ---

    def update_all_annotation_lists(self):
        for image_name in self.mw.all_annotations.keys():
            self.update_annotation_list(image_name)
        self.update_annotation_list()

    def update_annotation_list(self, image_name=None):
        self.mw.annotation_list.clear()
        current_name = image_name or self.mw.current_slice or self.mw.image_file_name
        annotations = self.mw.all_annotations.get(current_name, {})
        for class_name, class_annotations in annotations.items():
            if not class_name.startswith("Temp-"):
                color = self.mw.image_label.class_colors.get(
                    class_name, QColor(Qt.GlobalColor.white)
                )
                for annotation in class_annotations:
                    number = annotation.get("number", 0)
                    area = calculate_area(annotation)
                    item_text = f"{class_name} - {number:<3} Area: {area:.2f}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, annotation)
                    item.setForeground(color)
                    self.mw.annotation_list.addItem(item)

        self.mw.annotation_list.repaint()

    def update_annotation_list_colors(self, class_name=None, color=None):
        for i in range(self.mw.annotation_list.count()):
            item = self.mw.annotation_list.item(i)
            annotation = item.data(Qt.ItemDataRole.UserRole)
            if class_name is None or annotation["category_name"] == class_name:
                item_color = (
                    color
                    if class_name
                    else self.mw.image_label.class_colors.get(
                        annotation["category_name"], QColor(Qt.GlobalColor.white)
                    )
                )
                item.setForeground(item_color)

    def update_annotation_list_with_sorted(self, sorted_annotations):
        self.mw.annotation_list.clear()
        for annotation in sorted_annotations:
            class_name = annotation["category_name"]
            if not class_name.startswith("Temp-"):
                number = annotation.get("number", 0)
                area = calculate_area(annotation)
                item_text = f"{class_name} - {number:<3} Area: {area:.2f}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, annotation)
                color = self.mw.image_label.class_colors.get(
                    class_name, QColor(Qt.GlobalColor.white)
                )
                item.setForeground(color)
                self.mw.annotation_list.addItem(item)

        self.mw.image_label.update()

    # --- Per-image annotation cache sync ---

    def load_image_annotations(self):
        self.mw.image_label.annotations.clear()
        current_name = self.mw.current_slice or self.mw.image_file_name
        if current_name in self.mw.all_annotations:
            self.mw.image_label.annotations = copy.deepcopy(
                self.mw.all_annotations[current_name]
            )
        else:
            print(f"No annotations found for {current_name}")
        self.mw.image_label.update()

    def save_current_annotations(self):
        if self.mw.current_slice:
            current_name = self.mw.current_slice
        elif self.mw.image_file_name:
            current_name = self.mw.image_file_name
        else:
            return

        if self.mw.image_label.annotations:
            self.mw.all_annotations[current_name] = (
                self.mw.image_label.annotations.copy()
            )
        elif current_name in self.mw.all_annotations:
            del self.mw.all_annotations[current_name]

        self.mw.update_slice_list_colors()

    def replace_annotations(self, image_key: str, annotations: dict) -> None:
        """Replace the full per-class annotation dict for one image.
        Used by the eraser path which has already cut polygons in
        ImageLabel.annotations. Triggers list refresh, save, and slice
        colour update atomically."""
        self.mw.all_annotations[image_key] = annotations
        self.update_annotation_list()
        self.save_current_annotations()
        self.mw.class_controller.update_slice_list_colors()

    # --- Sorting ---

    def sort_annotations_by_class(self):
        current_name = self.mw.current_slice or self.mw.image_file_name
        if current_name not in self.mw.all_annotations:
            QMessageBox.information(
                self.mw,
                "No Annotations",
                "There are no annotations to sort for this image.",
            )
            return

        annotations = self.mw.all_annotations[current_name]
        sorted_annotations = []
        for class_name in sorted(annotations.keys()):
            if not class_name.startswith("Temp-"):
                class_annotations = sorted(
                    annotations[class_name], key=lambda x: x.get("number", 0)
                )
                sorted_annotations.extend(class_annotations)

        self.update_annotation_list_with_sorted(sorted_annotations)

    def sort_annotations_by_area(self):
        current_name = self.mw.current_slice or self.mw.image_file_name
        if current_name not in self.mw.all_annotations:
            QMessageBox.information(
                self.mw,
                "No Annotations",
                "There are no annotations to sort for this image.",
            )
            return

        annotations = self.mw.all_annotations[current_name]
        sorted_annotations = []
        for class_name in annotations.keys():
            if not class_name.startswith("Temp-"):
                class_annotations = sorted(
                    annotations[class_name],
                    key=lambda x: calculate_area(x),
                    reverse=True,
                )
                sorted_annotations.extend(class_annotations)

        self.update_annotation_list_with_sorted(sorted_annotations)

    # --- COCO JSON load (independent of project save/load) ---

    def load_annotations(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self.mw, "Load Annotations", "", "JSON Files (*.json)"
        )
        if not file_name:
            return

        with open(file_name, "r", encoding='utf-8') as f:
            self.mw.loaded_json = json.load(f)

        self.mw.class_list.clear()
        self.mw.image_label.class_colors.clear()
        self.mw.class_mapping.clear()
        for category in self.mw.loaded_json["categories"]:
            class_name = category["name"]
            self.mw.class_mapping[class_name] = category["id"]

            if class_name not in self.mw.image_label.class_colors:
                color = QColor(
                    default_class_color(len(self.mw.image_label.class_colors))
                )
                self.mw.image_label.class_colors[class_name] = color

            item = QListWidgetItem(class_name)
            self.mw.update_class_item_color(
                item, self.mw.image_label.class_colors[class_name]
            )
            self.mw.class_list.addItem(item)

        image_id_to_filename = {
            img["id"]: img["file_name"] for img in self.mw.loaded_json["images"]
        }

        json_images = {img["file_name"]: img for img in self.mw.loaded_json["images"]}

        updated_all_images = []
        for i in range(self.mw.image_list.count()):
            item = self.mw.image_list.item(i)
            file_name = item.text()
            if file_name in json_images:
                updated_image = self.mw.all_images[i].copy()
                updated_image.update(json_images[file_name])
                updated_all_images.append(updated_image)
                del json_images[file_name]
            else:
                updated_all_images.append(self.mw.all_images[i])

        for img in json_images.values():
            updated_all_images.append(img)

        self.mw.all_images = updated_all_images
        # Rebuild the list in sorted order (issue #60). The reconciliation
        # loop above already consumed the pre-sort row/index alignment.
        self.mw.update_image_list()

        self.mw.all_annotations.clear()
        for annotation in self.mw.loaded_json["annotations"]:
            image_id = annotation["image_id"]
            file_name = image_id_to_filename.get(image_id)
            if file_name:
                if file_name not in self.mw.all_annotations:
                    self.mw.all_annotations[file_name] = {}

                category = next(
                    (
                        cat
                        for cat in self.mw.loaded_json["categories"]
                        if cat["id"] == annotation["category_id"]
                    ),
                    None,
                )
                if category:
                    category_name = category["name"]
                    if category_name not in self.mw.all_annotations[file_name]:
                        self.mw.all_annotations[file_name][category_name] = []

                    ann = {
                        "category_id": annotation["category_id"],
                        "category_name": category_name,
                    }

                    if "segmentation" in annotation:
                        ann["segmentation"] = annotation["segmentation"][0]
                        ann["type"] = "polygon"
                    elif "bbox" in annotation:
                        ann["bbox"] = annotation["bbox"]
                        ann["type"] = "bbox"

                    if "number" not in ann:
                        ann["number"] = (
                            len(self.mw.all_annotations[file_name][category_name]) + 1
                        )

                    self.mw.all_annotations[file_name][category_name].append(ann)

        missing_images = [
            img["file_name"]
            for img in self.mw.loaded_json["images"]
            if img["file_name"] not in self.mw.image_paths
        ]
        if missing_images:
            self.mw.show_warning(
                "Missing Images",
                "The following images are missing:\n" + "\n".join(missing_images),
            )

        if self.mw.image_file_name and self.mw.image_file_name in self.mw.all_annotations:
            self.mw.switch_image(
                self.mw.image_list.findItems(
                    self.mw.image_file_name, Qt.MatchFlag.MatchExactly
                )[0]
            )
        elif self.mw.all_images:
            self.mw.switch_image(self.mw.image_list.item(0))

        self.mw.image_label.highlighted_annotations = []
        self.update_annotation_list()
        self.mw.image_label.update()

    # --- Highlighting / selection ---

    def clear_highlighted_annotation(self):
        self.mw.image_label.highlighted_annotations.clear()
        # Selection is gone — Merge/Change Class must follow, or they linger
        # enabled against an empty list selection after an image/slice switch.
        self._sync_selection_buttons(0)
        self.mw.image_label.update()

    def update_highlighted_annotations(self):
        selected_items = self.mw.annotation_list.selectedItems()
        self.mw.image_label.highlighted_annotations = [
            item.data(Qt.ItemDataRole.UserRole) for item in selected_items
        ]
        self.mw.image_label.update()
        self._sync_selection_buttons(len(selected_items))

    def _sync_selection_buttons(self, count):
        """Merge needs ≥2 annotations; Change Class needs ≥1. Shared by the
        list-driven and canvas-driven (issue #75) selection paths."""
        self.mw.merge_button.setEnabled(count >= 2)
        self.mw.change_class_button.setEnabled(count > 0)

    def apply_canvas_selection(self, annotations, mode):
        """Apply a selection change that originated on the canvas (issue #75)
        and mirror it onto the annotation list so Delete / Merge / Change
        Class operate on the same set. Matching uses dict value-equality,
        consistent with the rest of the selection code.

        ``mode`` is one of ``"replace"``, ``"add"``, ``"toggle"``.
        """
        current = list(self.mw.image_label.highlighted_annotations)

        def contains(seq, ann):
            return any(a == ann for a in seq)

        if mode == "replace":
            new = list(annotations)
        elif mode == "add":
            new = current + [a for a in annotations if not contains(current, a)]
        elif mode == "toggle":
            new = list(current)
            for a in annotations:
                match = next((x for x in new if x == a), None)
                if match is not None:
                    new.remove(match)
                else:
                    new.append(a)
        else:
            return

        self.mw.image_label.highlighted_annotations = new

        # Mirror onto the list widget. Block signals so the programmatic
        # selection doesn't retrigger itemSelectionChanged →
        # update_highlighted_annotations, which would overwrite `new` with
        # the list items' own (all_annotations) object identities.
        lst = self.mw.annotation_list
        lst.blockSignals(True)
        lst.clearSelection()
        for i in range(lst.count()):
            item = lst.item(i)
            if contains(new, item.data(Qt.ItemDataRole.UserRole)):
                item.setSelected(True)
        lst.blockSignals(False)

        self._sync_selection_buttons(len(new))
        self.mw.image_label.update()

    def commit_bbox_edit(self):
        """Persist a bbox resize/move performed directly on the canvas
        (issue #40). ImageLabel already mutated + clamped the bbox in place, so
        we save (pushing the new coords into all_annotations), rebuild the list
        so the displayed area refreshes, then re-mirror the selection so the
        edited box stays selected and list/canvas stay in sync."""
        selected = list(self.mw.image_label.highlighted_annotations)
        self.save_current_annotations()
        self.update_annotation_list()
        self.apply_canvas_selection(selected, "replace")
        self.mw.auto_save()

    def highlight_annotation_in_list(self, annotation):
        for i in range(self.mw.annotation_list.count()):
            item = self.mw.annotation_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == annotation:
                self.mw.annotation_list.setCurrentItem(item)
                break

    def select_annotation_in_list(self, annotation):
        for i in range(self.mw.annotation_list.count()):
            item = self.mw.annotation_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == annotation:
                self.mw.annotation_list.setCurrentItem(item)
                break

    # --- Annotation numbering ---

    def renumber_annotations(self):
        current_name = self.mw.current_slice or self.mw.image_file_name
        if current_name in self.mw.all_annotations:
            for class_name, annotations in self.mw.all_annotations[
                current_name
            ].items():
                for i, ann in enumerate(annotations, start=1):
                    ann["number"] = i
        self.update_annotation_list()

    # --- Delete / merge / change-class ---

    def delete_annotation(self):
        current_item = self.mw.annotation_list.currentItem()
        if current_item:
            annotation = current_item.data(Qt.ItemDataRole.UserRole)
            category_name = annotation["category_name"]
            self.mw.image_label.annotations[category_name].remove(annotation)
            self.mw.annotation_list.takeItem(
                self.mw.annotation_list.row(current_item)
            )
            self.mw.image_label.highlighted_annotations.clear()
            self.mw.image_label.update()

    def delete_selected_annotations(self):
        selected_items = self.mw.annotation_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self.mw, "No Selection", "Please select an annotation to delete."
            )
            return

        reply = QMessageBox.question(
            self.mw,
            "Delete Annotations",
            f"Are you sure you want to delete {len(selected_items)} annotation(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            annotations_to_remove = []
            for item in selected_items:
                annotation = item.data(Qt.ItemDataRole.UserRole)
                annotations_to_remove.append((annotation["category_name"], annotation))

            for category_name, annotation in annotations_to_remove:
                if category_name in self.mw.image_label.annotations:
                    if annotation in self.mw.image_label.annotations[category_name]:
                        self.mw.image_label.annotations[category_name].remove(
                            annotation
                        )

            current_name = self.mw.current_slice or self.mw.image_file_name
            self.mw.all_annotations[current_name] = self.mw.image_label.annotations

            if self.mw.current_sort_method == "area":
                self.sort_annotations_by_area()
            else:
                self.sort_annotations_by_class()

            self.mw.image_label.highlighted_annotations.clear()
            # Selection is now empty — Merge/Change Class must follow.
            self._sync_selection_buttons(0)
            self.mw.image_label.update()

            self.mw.update_slice_list_colors()

            QMessageBox.information(
                self.mw,
                "Annotations Deleted",
                f"{len(selected_items)} annotation(s) have been deleted.",
            )
            self.mw.auto_save()

    def merge_annotations(self):
        if self.mw.image_label.editing_polygon is not None:
            QMessageBox.warning(
                self.mw,
                "Edit Mode Active",
                "Please exit the annotation edit mode before merging annotations.",
            )
            return

        selected_items = self.mw.annotation_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(
                self.mw,
                "Not Enough Annotations",
                "Please select at least two annotations to merge.",
            )
            return

        class_name = selected_items[0].data(Qt.ItemDataRole.UserRole)["category_name"]
        if not all(
            item.data(Qt.ItemDataRole.UserRole)["category_name"] == class_name
            for item in selected_items
        ):
            QMessageBox.warning(
                self.mw,
                "Mixed Classes",
                "All selected annotations must be from the same class.",
            )
            return

        polygons = []
        original_annotations = []
        for item in selected_items:
            annotation = item.data(Qt.ItemDataRole.UserRole)
            original_annotations.append(annotation)
            if "segmentation" in annotation:
                points = zip(
                    annotation["segmentation"][0::2], annotation["segmentation"][1::2]
                )
                polygon = Polygon(points)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                polygons.append(polygon)

        def are_all_polygons_connected(polygons):
            if len(polygons) < 2:
                return True

            connected = set([0])
            to_check = set(range(1, len(polygons)))

            while to_check:
                newly_connected = set()
                for i in connected:
                    for j in to_check:
                        if polygons[i].intersects(polygons[j]) or polygons[i].touches(
                            polygons[j]
                        ):
                            newly_connected.add(j)

                if not newly_connected:
                    return False

                connected.update(newly_connected)
                to_check -= newly_connected

            return True

        if not are_all_polygons_connected(polygons):
            QMessageBox.warning(
                self.mw,
                "Disconnected Polygons",
                "Not all selected annotations are connected. Please select only connected annotations to merge.",
            )
            return

        try:
            merged_polygon = unary_union(polygons)
        except Exception as e:
            QMessageBox.warning(
                self.mw,
                "Merge Error",
                f"Unable to merge the selected annotations due to an error: {str(e)}",
            )
            return

        new_annotation = {
            "segmentation": [],
            "category_id": self.mw.class_mapping[class_name],
            "category_name": class_name,
        }

        if isinstance(merged_polygon, Polygon):
            new_annotation["segmentation"] = [
                coord for point in merged_polygon.exterior.coords for coord in point
            ]
        elif isinstance(merged_polygon, MultiPolygon):
            largest_polygon = max(merged_polygon.geoms, key=lambda p: p.area)
            new_annotation["segmentation"] = [
                coord for point in largest_polygon.exterior.coords for coord in point
            ]

        msg_box = QMessageBox(self.mw)
        msg_box.setWindowTitle("Merge Annotations")
        msg_box.setText("Do you want to keep the original annotations?")
        msg_box.setIcon(QMessageBox.Icon.Question)

        msg_box.addButton("Keep", QMessageBox.ButtonRole.YesRole)
        delete_button = msg_box.addButton("Delete", QMessageBox.ButtonRole.NoRole)
        cancel_button = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

        msg_box.setDefaultButton(cancel_button)
        msg_box.setEscapeButton(cancel_button)

        msg_box.exec()

        if msg_box.clickedButton() == cancel_button:
            return

        if msg_box.clickedButton() == delete_button:
            for annotation in original_annotations:
                if annotation in self.mw.image_label.annotations[class_name]:
                    self.mw.image_label.annotations[class_name].remove(annotation)

        self.mw.image_label.annotations.setdefault(class_name, []).append(new_annotation)

        current_name = self.mw.current_slice or self.mw.image_file_name
        self.mw.all_annotations[current_name] = self.mw.image_label.annotations

        self.renumber_annotations()
        self.update_annotation_list()
        self.save_current_annotations()
        self.mw.update_slice_list_colors()
        self.mw.image_label.update()

        QMessageBox.information(
            self.mw, "Merge Complete", "Annotations have been merged successfully."
        )
        self.mw.auto_save()

    def change_annotation_class(self):
        selected_items = self.mw.annotation_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self.mw,
                "No Selection",
                "Please select one or more annotations to change class.",
            )
            return

        class_dialog = QDialog(self.mw)
        class_dialog.setWindowTitle("Change Class")
        layout = QVBoxLayout(class_dialog)

        class_combo = QComboBox()
        for class_name in self.mw.class_mapping.keys():
            class_combo.addItem(class_name)
        layout.addWidget(class_combo)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(class_dialog.accept)
        button_box.rejected.connect(class_dialog.reject)
        layout.addWidget(button_box)

        if class_dialog.exec() == QDialog.DialogCode.Accepted:
            new_class = class_combo.currentText()
            current_name = self.mw.current_slice or self.mw.image_file_name

            max_number = max(
                [
                    ann.get("number", 0)
                    for ann in self.mw.image_label.annotations.get(new_class, [])
                ]
                + [0]
            )

            for item in selected_items:
                annotation = item.data(Qt.ItemDataRole.UserRole)
                old_class = annotation["category_name"]

                self.mw.image_label.annotations[old_class].remove(annotation)
                if not self.mw.image_label.annotations[old_class]:
                    del self.mw.image_label.annotations[old_class]

                annotation["category_name"] = new_class
                annotation["category_id"] = self.mw.class_mapping[new_class]
                max_number += 1
                annotation["number"] = max_number
                if new_class not in self.mw.image_label.annotations:
                    self.mw.image_label.annotations[new_class] = []
                self.mw.image_label.annotations[new_class].append(annotation)

            self.mw.all_annotations[current_name] = self.mw.image_label.annotations

            self.renumber_annotations()

            self.update_annotation_list()
            self.mw.image_label.update()
            self.save_current_annotations()
            self.mw.update_slice_list_colors()
            self.mw.auto_save()

            QMessageBox.information(
                self.mw,
                "Class Changed",
                f"Selected annotations have been changed to class '{new_class}'.",
            )

    # --- Commit paths for the drawing tools ---

    def finish_polygon(self):
        if (
            self.mw.image_label.current_tool == "polygon"
            and len(self.mw.image_label.current_annotation) > 2
        ):
            if self.mw.current_class is None:
                QMessageBox.warning(
                    self.mw,
                    "No Class Selected",
                    "Please select a class before finishing the annotation.",
                )
                return

            polygon = Polygon(self.mw.image_label.current_annotation)

            image_boundary = Polygon(
                [
                    (0, 0),
                    (self.mw.current_image.width(), 0),
                    (self.mw.current_image.width(), self.mw.current_image.height()),
                    (0, self.mw.current_image.height()),
                ]
            )

            clipped_polygon = polygon.intersection(image_boundary)

            if clipped_polygon.is_empty:
                QMessageBox.warning(
                    self.mw,
                    "Invalid Annotation",
                    "The annotation is completely outside the image boundaries.",
                )
                self.mw.image_label.clear_current_annotation()
                self.mw.image_label.update()
                return

            if isinstance(clipped_polygon, Polygon):
                segmentation = [
                    coord
                    for point in clipped_polygon.exterior.coords
                    for coord in point
                ]
            elif isinstance(clipped_polygon, MultiPolygon):
                largest_polygon = max(clipped_polygon.geoms, key=lambda p: p.area)
                segmentation = [
                    coord
                    for point in largest_polygon.exterior.coords
                    for coord in point
                ]
            else:
                QMessageBox.warning(
                    self.mw,
                    "Invalid Annotation",
                    "The annotation could not be processed.",
                )
                return

            new_annotation = {
                "segmentation": segmentation,
                "category_id": self.mw.class_mapping[self.mw.current_class],
                "category_name": self.mw.current_class,
            }
            self.mw.image_label.annotations.setdefault(
                self.mw.current_class, []
            ).append(new_annotation)
            self.add_annotation_to_list(new_annotation)
            self.mw.image_label.clear_current_annotation()
            self.mw.image_label.drawing_polygon = False
            self.mw.image_label.reset_annotation_state()
            self.mw.image_label.update()

            self.save_current_annotations()

            self.mw.update_slice_list_colors()
            self.mw.auto_save()

    def finish_rectangle(self):
        if self.mw.image_label.current_rectangle:
            x1, y1, x2, y2 = self.mw.image_label.current_rectangle

            rectangle = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            image_boundary = Polygon(
                [
                    (0, 0),
                    (self.mw.current_image.width(), 0),
                    (self.mw.current_image.width(), self.mw.current_image.height()),
                    (0, self.mw.current_image.height()),
                ]
            )

            clipped_rectangle = rectangle.intersection(image_boundary)

            if clipped_rectangle.is_empty:
                QMessageBox.warning(
                    self.mw,
                    "Invalid Annotation",
                    "The annotation is completely outside the image boundaries.",
                )
                self.mw.image_label.current_rectangle = None
                self.mw.image_label.update()
                return

            if isinstance(clipped_rectangle, Polygon):
                segmentation = [
                    coord
                    for point in clipped_rectangle.exterior.coords
                    for coord in point
                ]
            elif isinstance(clipped_rectangle, MultiPolygon):
                largest_polygon = max(clipped_rectangle.geoms, key=lambda p: p.area)
                segmentation = [
                    coord
                    for point in largest_polygon.exterior.coords
                    for coord in point
                ]
            else:
                QMessageBox.warning(
                    self.mw,
                    "Invalid Annotation",
                    "The annotation could not be processed.",
                )
                return

            new_annotation = {
                "segmentation": segmentation,
                "category_id": self.mw.class_mapping[self.mw.current_class],
                "category_name": self.mw.current_class,
            }
            self.mw.image_label.annotations.setdefault(
                self.mw.current_class, []
            ).append(new_annotation)
            self.add_annotation_to_list(new_annotation)
            self.mw.image_label.start_point = None
            self.mw.image_label.end_point = None
            self.mw.image_label.current_rectangle = None
            self.mw.image_label.update()

            self.save_current_annotations()

            self.mw.update_slice_list_colors()
            self.mw.auto_save()

    def add_annotation_to_list(self, annotation):
        class_name = annotation["category_name"]
        color = self.mw.image_label.class_colors.get(
            class_name, QColor(Qt.GlobalColor.white)
        )
        annotations = self.mw.image_label.annotations.get(class_name, [])
        number = max([ann.get("number", 0) for ann in annotations] + [0]) + 1
        annotation["number"] = number
        area = calculate_area(annotation)
        item_text = f"{class_name} - {number:<3} Area: {area:.2f}"

        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, annotation)
        item.setForeground(color)
        self.mw.annotation_list.addItem(item)

        self.mw.annotation_list.clearSelection()
        self.mw.image_label.highlighted_annotations.clear()
        self.mw.image_label.update()

    # --- Edit mode ---

    def enter_edit_mode(self, annotation):
        self.mw.editing_mode = True
        self.mw.disable_tools()

        QMessageBox.information(
            self.mw,
            "Edit Mode",
            "You are now in edit mode. Click and drag points to move them, Shift+Click to delete points, or click on edges to add new points.",
        )

    def exit_edit_mode(self):
        self.mw.editing_mode = False
        self.mw.enable_tools()

        self.mw.image_label.editing_polygon = None
        self.mw.image_label.editing_point_index = None
        self.mw.image_label.hover_point_index = None
        self.update_annotation_list()
        self.mw.image_label.update()
