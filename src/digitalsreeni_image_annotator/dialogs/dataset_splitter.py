import os
import json
import shutil
import random
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, 
                             QLabel, QSpinBox, QRadioButton, QButtonGroup, QMessageBox, QComboBox)
from PyQt6.QtCore import Qt
import yaml
from PIL import Image

class DatasetSplitterTool(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dataset Splitter")
        self.setGeometry(100, 100, 500, 300)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Option selection
        options_layout = QVBoxLayout()
        self.images_only_radio = QRadioButton("Images Only")
        options_layout.addWidget(self.images_only_radio)

        images_annotations_layout = QHBoxLayout()
        self.images_annotations_radio = QRadioButton("Images and Annotations")
        images_annotations_layout.addWidget(self.images_annotations_radio)
        self.select_json_button = QPushButton("Upload COCO JSON File")
        self.select_json_button.clicked.connect(self.select_json_file)
        self.select_json_button.setEnabled(False)
        images_annotations_layout.addWidget(self.select_json_button)
        options_layout.addLayout(images_annotations_layout)

        layout.addLayout(options_layout)
        
        option_group = QButtonGroup(self)
        option_group.addButton(self.images_only_radio)
        option_group.addButton(self.images_annotations_radio)
        
        self.images_only_radio.setChecked(True)

        # Percentage inputs
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("Train %:"))
        self.train_percent = QSpinBox()
        self.train_percent.setRange(0, 100)
        self.train_percent.setValue(70)
        train_layout.addWidget(self.train_percent)
        layout.addLayout(train_layout)

        val_layout = QHBoxLayout()
        val_layout.addWidget(QLabel("Validation %:"))
        self.val_percent = QSpinBox()
        self.val_percent.setRange(0, 100)
        self.val_percent.setValue(30)
        val_layout.addWidget(self.val_percent)
        layout.addLayout(val_layout)

        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("Test %:"))
        self.test_percent = QSpinBox()
        self.test_percent.setRange(0, 100)
        self.test_percent.setValue(0)
        test_layout.addWidget(self.test_percent)
        layout.addLayout(test_layout)

        # Keep the three percentages summing to 100 by auto-balancing the
        # other two whenever one changes. _rebalancing guards against the
        # programmatic setValue calls re-triggering these handlers.
        self._rebalancing = False
        self.train_percent.valueChanged.connect(self._on_train_changed)
        self.val_percent.valueChanged.connect(self._on_val_changed)
        self.test_percent.valueChanged.connect(self._on_test_changed)

        # Format selection
        self.format_selection_layout = QHBoxLayout()
        self.format_label = QLabel("Output Format:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["COCO JSON", "YOLO"])
        self.format_combo.setEnabled(False)
        self.format_selection_layout.addWidget(self.format_label)
        self.format_selection_layout.addWidget(self.format_combo)
        options_layout.addLayout(self.format_selection_layout)

        # Buttons
        self.select_input_button = QPushButton("Select Input Directory")
        self.select_input_button.clicked.connect(self.select_input_directory)
        layout.addWidget(self.select_input_button)

        self.select_output_button = QPushButton("Select Output Directory")
        self.select_output_button.clicked.connect(self.select_output_directory)
        layout.addWidget(self.select_output_button)

        self.split_button = QPushButton("Split Dataset")
        self.split_button.clicked.connect(self.split_dataset)
        layout.addWidget(self.split_button)

        self.setLayout(layout)

        self.input_directory = ""
        self.output_directory = ""
        self.json_file = ""

        # Connect radio buttons to enable/disable JSON selection
        self.images_only_radio.toggled.connect(self.toggle_json_selection)
        self.images_annotations_radio.toggled.connect(self.toggle_json_selection)

    def toggle_json_selection(self):
        is_annotations = self.images_annotations_radio.isChecked()
        self.select_json_button.setEnabled(is_annotations)
        self.format_combo.setEnabled(is_annotations)

    def select_input_directory(self):
        self.input_directory = QFileDialog.getExistingDirectory(self, "Select Input Directory")

    def select_output_directory(self):
        self.output_directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")

    def select_json_file(self):
        self.json_file, _ = QFileDialog.getOpenFileName(self, "Select COCO JSON File", "", "JSON Files (*.json)")

    def _set_silently(self, spin, value):
        """Set a spin box without re-entering the rebalance handlers."""
        spin.blockSignals(True)
        spin.setValue(max(0, min(100, value)))
        spin.blockSignals(False)

    def _on_train_changed(self, train):
        # Changing train pushes the delta into val; if val can't absorb it
        # all (would go negative), the remainder spills into test.
        if self._rebalancing:
            return
        self._rebalancing = True
        try:
            train = max(0, min(100, train))
            test = self.test_percent.value()
            val = 100 - train - test
            if val < 0:
                val, test = 0, 100 - train
            self._set_silently(self.train_percent, train)
            self._set_silently(self.val_percent, val)
            self._set_silently(self.test_percent, test)
        finally:
            self._rebalancing = False

    def _on_val_changed(self, val):
        # Changing val is absorbed by test first; once test is exhausted,
        # the rest comes out of train.
        if self._rebalancing:
            return
        self._rebalancing = True
        try:
            val = max(0, min(100, val))
            train = self.train_percent.value()
            test = 100 - train - val
            if test < 0:
                test, train = 0, 100 - val
            self._set_silently(self.train_percent, train)
            self._set_silently(self.val_percent, val)
            self._set_silently(self.test_percent, test)
        finally:
            self._rebalancing = False

    def _on_test_changed(self, test):
        # Changing test is absorbed by val first; once val is exhausted,
        # the rest comes out of train.
        if self._rebalancing:
            return
        self._rebalancing = True
        try:
            test = max(0, min(100, test))
            train = self.train_percent.value()
            val = 100 - train - test
            if val < 0:
                val, train = 0, 100 - test
            self._set_silently(self.train_percent, train)
            self._set_silently(self.val_percent, val)
            self._set_silently(self.test_percent, test)
        finally:
            self._rebalancing = False

    @staticmethod
    def compute_split_counts(n, train_pct, val_pct, test_pct):
        """Return (train, val, test) image counts that sum to n.

        A subset whose percentage is 0 gets exactly 0 images (this is the
        fix for the off-by-one where test=0% still produced one image). The
        flooring remainder is handed to the subsets with the largest
        fractional part, but only to those with pct > 0. Assumes the three
        percentages sum to 100 (validated in split_dataset).
        """
        if n <= 0:
            return 0, 0, 0
        counts = {
            "train": int(n * train_pct / 100),
            "val": int(n * val_pct / 100),
            "test": int(n * test_pct / 100),
        }
        fracs = sorted(
            (
                (n * pct / 100 - counts[key], key)
                for pct, key in (
                    (train_pct, "train"),
                    (val_pct, "val"),
                    (test_pct, "test"),
                )
                if pct > 0
            ),
            reverse=True,
        )
        remainder = n - sum(counts.values())
        i = 0
        while remainder > 0 and fracs:
            counts[fracs[i % len(fracs)][1]] += 1
            remainder -= 1
            i += 1
        return counts["train"], counts["val"], counts["test"]

    def split_dataset(self):
        if not self.input_directory or not self.output_directory:
            QMessageBox.warning(self, "Error", "Please select input and output directories.")
            return

        if self.images_annotations_radio.isChecked() and not self.json_file:
            QMessageBox.warning(self, "Error", "Please select a COCO JSON file.")
            return

        train_percent = self.train_percent.value()
        val_percent = self.val_percent.value()
        test_percent = self.test_percent.value()

        if train_percent + val_percent + test_percent != 100:
            QMessageBox.warning(self, "Error", "Percentages must add up to 100%.")
            return

        try:
            if self.images_only_radio.isChecked():
                self.split_images_only()
            else:
                self.split_images_and_annotations()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Dataset split failed:\n{e}")

    def split_images_only(self):
        image_files = [f for f in os.listdir(self.input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        random.shuffle(image_files)

        n = len(image_files)
        train_n, val_n, test_n = self.compute_split_counts(
            n, self.train_percent.value(), self.val_percent.value(),
            self.test_percent.value())

        train_images = image_files[:train_n]
        val_images = image_files[train_n:train_n + val_n]
        test_images = image_files[train_n + val_n:train_n + val_n + test_n]

        for subset, images in [("train", train_images), 
                             ("val", val_images), 
                             ("test", test_images)]:
            if images:  # Only create directories and copy images if there are images for this split
                subset_dir = os.path.join(self.output_directory, subset)
                os.makedirs(subset_dir, exist_ok=True)
                self.copy_images(images, subset, images_only=True)

        QMessageBox.information(self, "Success", "Dataset split successfully!")

    def split_images_and_annotations(self):
        with open(self.json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        image_files = [img['file_name'] for img in coco_data['images']]

        # The JSON lists filenames; nothing guarantees they exist in the
        # chosen input directory. A partial split would silently produce a
        # broken dataset, so refuse to start if anything is missing.
        missing = [f for f in image_files
                   if not os.path.exists(os.path.join(self.input_directory, f))]
        if missing:
            preview = "\n".join(missing[:10])
            if len(missing) > 10:
                preview += f"\n... and {len(missing) - 10} more"
            QMessageBox.warning(
                self, "Images Not Found",
                f"{len(missing)} of {len(image_files)} image(s) listed in the "
                f"COCO JSON were not found in the selected input directory:\n\n"
                f"{preview}\n\n"
                "Please select the directory that contains these images."
            )
            return

        random.shuffle(image_files)

        n = len(image_files)
        train_n, val_n, test_n = self.compute_split_counts(
            n, self.train_percent.value(), self.val_percent.value(),
            self.test_percent.value())

        train_images = image_files[:train_n]
        val_images = image_files[train_n:train_n + val_n]
        test_images = image_files[train_n + val_n:train_n + val_n + test_n]

        # Create main directories
        os.makedirs(self.output_directory, exist_ok=True)
        
        if self.format_combo.currentText() == "COCO JSON":
            self.split_coco_format(coco_data, train_images, val_images, test_images)
        else:  # YOLO format
            self.split_yolo_format(coco_data, train_images, val_images, test_images)

    def copy_images(self, image_list, subset, images_only=False):
        if not image_list:
            return
            
        if images_only:
            subset_dir = os.path.join(self.output_directory, subset)
        else:
            subset_dir = os.path.join(self.output_directory, subset, "images")
        os.makedirs(subset_dir, exist_ok=True)
        
        for image in image_list:
            src = os.path.join(self.input_directory, image)
            dst = os.path.join(subset_dir, image)
            shutil.copy2(src, dst)

    def create_subset_annotations(self, coco_data, subset_images):
        subset_images_data = [img for img in coco_data['images'] if img['file_name'] in subset_images]
        subset_image_ids = [img['id'] for img in subset_images_data]
        
        return {
            "images": subset_images_data,
            "annotations": [ann for ann in coco_data['annotations'] if ann['image_id'] in subset_image_ids],
            "categories": coco_data['categories']
        }

    def split_coco_format(self, coco_data, train_images, val_images, test_images):
        # Only create directories and save annotations for non-empty splits
        for subset, images in [("train", train_images), 
                             ("val", val_images), 
                             ("test", test_images)]:
            if images:  # Only process if there are images in this split
                subset_dir = os.path.join(self.output_directory, subset)
                os.makedirs(subset_dir, exist_ok=True)  # Create the subset directory first
                os.makedirs(os.path.join(subset_dir, "images"), exist_ok=True)
                self.copy_images(images, subset, images_only=False)
                
                # Create and save annotations for this subset
                subset_data = self.create_subset_annotations(coco_data, images)
                self.save_coco_annotations(subset_data, subset)

        QMessageBox.information(self, "Success", "Dataset and COCO annotations split successfully!")

    def save_coco_annotations(self, data, subset):
        subset_dir = os.path.join(self.output_directory, subset)
        os.makedirs(subset_dir, exist_ok=True)
        output_file = os.path.join(subset_dir, f"{subset}_annotations.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def split_yolo_format(self, coco_data, train_images, val_images, test_images):
        # Create directories only for non-empty splits
        yaml_paths = {}
        for subset, images in [("train", train_images), 
                             ("val", val_images), 
                             ("test", test_images)]:
            if images:  # Only create directories if there are images for this split
                subset_dir = os.path.join(self.output_directory, subset)
                os.makedirs(os.path.join(subset_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(subset_dir, "labels"), exist_ok=True)
                yaml_paths[subset] = f'./{subset}/images'

        # Create class mapping (COCO to YOLO indices)
        categories = {cat["id"]: i for i, cat in enumerate(coco_data["categories"])}

        # Process each non-empty subset
        for subset, images in [("train", train_images), 
                             ("val", val_images), 
                             ("test", test_images)]:
            if not images:  # Skip if no images in this split
                continue
                
            images_dir = os.path.join(self.output_directory, subset, "images")
            labels_dir = os.path.join(self.output_directory, subset, "labels")
            
            for image_file in images:
                # Copy image
                src = os.path.join(self.input_directory, image_file)
                shutil.copy2(src, os.path.join(images_dir, image_file))
                
                # Get image dimensions
                img = Image.open(src)
                img_width, img_height = img.size
                
                # Get annotations for this image
                image_id = next(img["id"] for img in coco_data["images"] if img["file_name"] == image_file)
                annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
                
                # Create YOLO format labels
                label_file = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")
                with open(label_file, "w", encoding='utf-8') as f:
                    for ann in annotations:
                        # Convert COCO class id to YOLO class id
                        yolo_class = categories[ann["category_id"]]
                        
                        # Convert COCO bbox to YOLO format
                        x, y, w, h = ann["bbox"]
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        w = w / img_width
                        h = h / img_height
                        
                        f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # Create data.yaml with only the relevant paths
        yaml_data = {
            'nc': len(categories),
            'names': [cat["name"] for cat in sorted(coco_data["categories"], key=lambda x: categories[x["id"]])]
        }
        yaml_data.update(yaml_paths)  # Add only paths for non-empty splits

        with open(os.path.join(self.output_directory, 'data.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)

        QMessageBox.information(self, "Success", "Dataset and YOLO annotations split successfully!")

    def show_centered(self, parent):
        parent_geo = parent.geometry()
        self.move(parent_geo.center() - self.rect().center())
        self.show()