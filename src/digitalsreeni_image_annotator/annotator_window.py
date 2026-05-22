import copy
import json
import os
import shutil
import traceback
import warnings
from datetime import datetime

import cv2
import numpy as np
import shapely
from czifile import CziFile
from PyQt6.QtCore import QEvent, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPalette,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from tifffile import TiffFile

from .annotation_statistics import show_annotation_statistics
from .coco_json_combiner import show_coco_json_combiner
from .dino_phrase_editor import ClassThresholdTable, PhraseEditorPanel
from .dino_utils import DINOUtils
from .dataset_splitter import DatasetSplitterTool
from .default_stylesheet import default_stylesheet
from .dicom_converter import DicomConverter
from .dino_merge_dialog import show_dino_merge_dialog
from .export_formats import (
    export_coco_json,
    export_labeled_images,
    export_pascal_voc_bbox,
    export_pascal_voc_both,
    export_semantic_labels,
    export_yolo_v4,
    export_yolo_v5plus,
)
from .help_window import HelpWindow
from .image_augmenter import show_image_augmenter
from .image_label import ImageLabel
from .image_patcher import show_image_patcher
from .import_formats import (
    import_coco_json,
    import_yolo_v4,
    import_yolo_v5plus,
    process_import_format,
)
from .sam_utils import InferenceBusyError, SAMUtils
from .slice_registration import SliceRegistrationTool
from .snake_game import SnakeGame
from .soft_dark_stylesheet import soft_dark_stylesheet
from .stack_interpolator import StackInterpolator
from .stack_to_slices import show_stack_to_slices
from .utils import calculate_area, calculate_bbox
from .yolo_trainer import LoadPredictionModelDialog, TrainingInfoDialog, YOLOTrainer

warnings.filterwarnings("ignore", category=UserWarning)


class TrainingThread(QThread):
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(object)

    def __init__(self, yolo_trainer, epochs, imgsz):
        super().__init__()
        self.yolo_trainer = yolo_trainer
        self.epochs = epochs
        self.imgsz = imgsz

    def run(self):
        try:
            results = self.yolo_trainer.train_model(
                epochs=self.epochs, imgsz=self.imgsz
            )
            self.finished.emit(results)
        except Exception as e:
            self.finished.emit(str(e))


class DimensionDialog(QDialog):
    def __init__(self, shape, file_name, parent=None, default_dimensions=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Dimensions")
        layout = QVBoxLayout(self)

        # Add file name label
        file_name_label = QLabel(f"File: {file_name}")
        file_name_label.setWordWrap(True)
        layout.addWidget(file_name_label)

        # Add dimension assignment widgets
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


class _DINOReviewEventFilter(QObject):
    """Application-wide event filter that lets Enter / Escape accept or
    reject pending DINO temp_annotations regardless of which widget has
    focus. Without this, clicking a slice/image entry in a list moves
    focus there and Enter is consumed by the list's itemActivated
    handler before it can reach ImageLabel.keyPressEvent.

    Suppressed when a modal dialog is active or focus is on a text-input
    widget so we don't break dialog default-button behaviour or
    in-cell editing.
    """

    def __init__(self, main_window: "ImageAnnotator"):
        super().__init__(main_window)
        self.main_window = main_window

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.KeyPress:
            return False
        key = event.key()
        if key not in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            return False
        app = QApplication.instance()
        if app is None or app.activeModalWidget() is not None:
            return False
        focused = app.focusWidget()
        if isinstance(focused, (QLineEdit, QTextEdit)):
            return False
        temp = self.main_window.image_label.temp_annotations
        if not temp or not any(a.get("source") == "dino" for a in temp):
            return False
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.main_window.accept_dino_results()
        else:
            self.main_window.reject_dino_results()
        return True


class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.is_loading_project = False
        self.backup_project_path = None

        self.setWindowTitle("Image Annotator")
        self.setGeometry(100, 100, 1400, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.create_menu_bar()

        # Initialize image_label early
        self.image_label = ImageLabel()

        self.image_label.sam_box_active = False
        self.image_label.sam_points_active = False
        self.image_label.sam_positive_points = []
        self.image_label.sam_negative_points = []
        self.image_label.set_main_window(self)

        # Initialize attributes
        self.current_image = None
        self.current_class = None
        self.image_file_name = ""
        self.all_annotations = {}
        self.all_images = []
        self.image_paths = {}
        self.loaded_json = None
        self.class_mapping = {}
        self.editing_mode = False
        self.current_slice = None
        self.slices = []
        self.current_stack = None
        self.image_dimensions = {}
        self.image_slices = {}
        self.image_shapes = {}

        
        # For paint brush and eraser
        self.paint_brush_size = 10
        self.eraser_size = 10
        # Initialize SAM utils
        self.current_sam_model = None
        self.sam_utils = SAMUtils()

        # Initialize DINO utils for LLM-assisted detection.
        # Phrases and thresholds are owned by the widgets (PhraseEditorPanel
        # and ClassThresholdTable); the project save/load reads/writes them
        # through the widget APIs, not through a shadow dict on self.
        self.dino_utils = DINOUtils()
        self.dino_model_loaded = False
        self.dino_custom_model_path = None

        # Debounce timer for SAM points: wait 1s after last click before inference
        self.sam_inference_timer = QTimer(self)
        self.sam_inference_timer.setSingleShot(True)
        self.sam_inference_timer.timeout.connect(self.apply_sam_prediction)

        # Guards against re-entrant `apply_sam_prediction` calls — the
        # debounce timer can fire while an earlier inference is still
        # pumping inside _run_sync. See apply_sam_prediction().
        self._sam_inference_in_flight = False

        # Create sam_magic_wand_button
        self.sam_magic_wand_button = QPushButton("Magic Wand")
        self.sam_magic_wand_button.setCheckable(True)
        self.sam_magic_wand_button.setEnabled(False)  # Initially disable the button

        # Initialize tool group
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)

        # Font size control
        self.font_sizes = {
            "Small": 8,
            "Medium": 10,
            "Large": 12,
            "XL": 14,
            "XXL": 16,
        }  # Also, add the options in create_menu_bar method
        self.current_font_size = "Medium"

        # Dark mode control. Default on — matches the look most users
        # expect from a 2025-era desktop annotation tool; toggle with
        # Settings → Toggle Dark Mode (Ctrl+D).
        self.dark_mode = True

        # Default annotations sorting
        self.current_sort_method = "class"  # Default sorting method

        # DINO batch review state. Initialised eagerly here so the
        # consumers don't each carry a `hasattr` check (one forgotten
        # check would crash with AttributeError).
        self.dino_batch_results: dict[str, list] = {}

        # Setup UI components
        self.setup_ui()

        # Apply theme and font (this includes stylesheet and font size application)
        self.apply_theme_and_font()

        # Connect sam_magic_wand_button
        self.sam_magic_wand_button.clicked.connect(self.toggle_tool)

        self.class_list.itemChanged.connect(self.toggle_class_visibility)

        # YOLO Trainer
        self.yolo_trainer = None
        self.setup_yolo_menu()

        # F2 → Snake game (Easter egg). Registered as a global QShortcut
        # so it fires regardless of which widget has focus — putting it
        # in keyPressEvent didn't work because QTableWidget (DINO
        # threshold table) and other focusable children consume F2
        # before it bubbles up to the main window.
        self._snake_shortcut = QShortcut(QKeySequence("F2"), self)
        self._snake_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._snake_shortcut.activated.connect(self.launch_snake_game)

        # Enter/Escape for DINO temp_annotations need to work even when
        # focus is on slice_list / image_list / a button — none of which
        # forward the key to ImageLabel.keyPressEvent. Application-wide
        # event filter intercepts these keys but only when DINO results
        # are pending review, and skips modal dialogs + text inputs.
        self._dino_review_filter = _DINOReviewEventFilter(self)
        QApplication.instance().installEventFilter(self._dino_review_filter)
        
        # Start in maximized mode
        self.showMaximized()

        # Start in maximized mode
        self.showMaximized()

    def setup_ui(self):
        # Initialize the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # Initialize tool group
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)

        # Setup UI components
        self.setup_sidebar()
        self.setup_image_area()
        self.setup_image_list()
        self.setup_slice_list()
        self.update_ui_for_current_tool()

    def update_window_title(self):
        base_title = "Image Annotator"
        if hasattr(self, "current_project_file"):
            project_name = os.path.basename(self.current_project_file)
            project_name = os.path.splitext(project_name)[
                0
            ]  # Remove the file extension
            self.setWindowTitle(f"{base_title} - {project_name}")
        else:
            self.setWindowTitle(base_title)

    def new_project(self):
        self.remove_all_temp_annotations()  # Remove temp annotations from the previous project
        project_file, _ = QFileDialog.getSaveFileName(
            self, "Create New Project", "", "Image Annotator Project (*.iap)"
        )
        if project_file:
            # Ensure the file has the correct extension
            if not project_file.lower().endswith(".iap"):
                project_file += ".iap"

            self.current_project_file = project_file
            self.current_project_dir = os.path.dirname(project_file)

            # Create the images directory
            images_dir = os.path.join(self.current_project_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            # Clear existing data without showing messages
            self.clear_all(new_project=True, show_messages=False)

            # Prompt for initial project notes
            notes, ok = QInputDialog.getMultiLineText(
                self, "Project Notes", "Enter initial project notes:"
            )
            if ok:
                self.project_notes = notes
            else:
                self.project_notes = ""

            self.project_creation_date = datetime.now().isoformat()

            # Save the empty project without showing a message
            self.save_project(show_message=False)

            # Keep only this message
            self.show_info(
                "New Project", f"New project created at {self.current_project_file}"
            )
            self.initialize_yolo_trainer()
            self.update_window_title()

    def show_project_search(self):
        from .project_search import show_project_search

        show_project_search(self)

    def open_project(self):
        print("open_project method called")  # Debug print
        self.remove_all_temp_annotations()  # Remove temp annotations from the previous project
        project_file, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Image Annotator Project (*.iap)"
        )
        print(f"Selected project file: {project_file}")  # Debug print
        if project_file:
            try:
                self.backup_project_before_open(project_file)
                self.open_specific_project(project_file)
            except Exception as e:
                self.restore_project_from_backup()
                QMessageBox.critical(
                    self,
                    "Error",
                    f"An error occurred while opening the project: {str(e)}\n"
                    f"The project file has been restored from backup.",
                )
        else:
            print("No project file selected")  # Debug print

    def backup_project_before_open(self, project_file):
        """Create a backup of the project file before opening it."""
        import os
        import shutil

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(project_file), ".project_backups")
        os.makedirs(backup_dir, exist_ok=True)

        self.backup_project_path = os.path.join(
            backup_dir, f"{os.path.basename(project_file)}.{timestamp}.backup"
        )
        shutil.copy2(project_file, self.backup_project_path)

    def restore_project_from_backup(self):
        """Restore the project file from its backup if available."""
        if self.backup_project_path and os.path.exists(self.backup_project_path):
            try:
                shutil.copy2(self.backup_project_path, self.current_project_file)
                print(f"Project restored from backup: {self.backup_project_path}")
            except Exception as e:
                print(f"Failed to restore from backup: {str(e)}")

    def open_specific_project(self, project_file):
        print(f"Opening specific project: {project_file}")  # Debug print
        if os.path.exists(project_file):
            try:
                self.is_loading_project = True  # Set loading flag

                with open(project_file, "r") as f:
                    project_data = json.load(f)

                self.clear_all(show_messages=False)
                self.current_project_file = project_file
                self.current_project_dir = os.path.dirname(project_file)

                # Load project notes and metadata
                self.project_notes = project_data.get("notes", "")
                self.project_creation_date = project_data.get("creation_date", "")
                self.last_modified = project_data.get("last_modified", "")

                # Parse dates
                if self.project_creation_date:
                    self.project_creation_date = datetime.fromisoformat(
                        self.project_creation_date
                    ).strftime("%Y-%m-%d %H:%M:%S")
                if self.last_modified:
                    self.last_modified = datetime.fromisoformat(
                        self.last_modified
                    ).strftime("%Y-%m-%d %H:%M:%S")

                # Load all data without triggering auto-saves
                self.load_project_data(project_data)

                # Now save once after everything is loaded
                self.is_loading_project = False  # Clear loading flag
                # Reveal the phrase editor if any classes exist — the
                # per-class selectRow inside add_class was skipped during
                # load (see add_class). Selecting row 0 is enough; the
                # user can switch rows freely afterwards.
                if self.dino_class_table.rowCount() > 0:
                    self.dino_class_table.selectRow(0)
                self.save_project(show_message=False)  # Save once after loading

                self.initialize_yolo_trainer()
                self.update_window_title()

                print(f"Project opened successfully: {project_file}")
                QMessageBox.information(
                    self,
                    "Project Opened",
                    f"Project opened successfully: {os.path.basename(project_file)}",
                )

            except Exception as e:
                self.is_loading_project = False  # Make sure to clear flag on error
                raise e
        else:
            print(f"Project file not found: {project_file}")
            QMessageBox.critical(
                self, "Error", f"Project file not found: {project_file}"
            )

    def load_project_data(self, project_data):
        """Load project data without triggering auto-saves."""
        # Load classes
        self.class_mapping.clear()
        self.image_label.class_colors.clear()
        for class_info in project_data.get("classes", []):
            self.add_class(class_info["name"], QColor(class_info["color"]))

        # Load images
        self.all_images = project_data.get("images", [])
        self.image_paths = project_data.get("image_paths", {})

        # Load all annotations first
        self.all_annotations.clear()
        for image_info in project_data["images"]:
            if image_info.get("is_multi_slice", False):
                for slice_info in image_info.get("slices", []):
                    self.all_annotations[slice_info["name"]] = slice_info["annotations"]
            else:
                self.all_annotations[image_info["file_name"]] = image_info.get(
                    "annotations", {}
                )

        # Handle missing images
        missing_images = []
        for image_info in project_data["images"]:
            image_path = os.path.join(
                self.current_project_dir, "images", image_info["file_name"]
            )

            if not os.path.exists(image_path):
                missing_images.append(image_info["file_name"])
                continue

            # Update image_paths
            self.image_paths[image_info["file_name"]] = image_path

            if image_info.get("is_multi_slice", False):
                dimensions = image_info.get("dimensions", [])
                shape = image_info.get("shape", [])
                self.load_multi_slice_image(image_path, dimensions, shape)
            else:
                self.add_images_to_list([image_path])

        # Restore DINO configuration if present. Classes were created above
        # via add_class(), so the threshold table already has rows for them;
        # we just push the saved values into the existing widgets. Filter
        # out any keys that reference classes no longer in the project
        # (hand-edited .iap, class deleted between sessions) so stale state
        # doesn't get round-tripped on the next save.
        dino_cfg = project_data.get("dino_config", {})
        valid_classes = set(self.class_mapping.keys())

        phrases = dino_cfg.get("phrases", {})
        if phrases:
            kept = {k: v for k, v in phrases.items() if k in valid_classes}
            for orphan in phrases.keys() - kept.keys():
                print(f"  Skipped saved DINO phrases for unknown class "
                      f"'{orphan}' — class is not in the current project.")
            self.dino_phrase_panel.set_phrases(kept)

        for cls_name, thr in dino_cfg.get("thresholds", {}).items():
            ok = self.dino_class_table.set_thresholds(
                cls_name,
                thr.get("box", 0.25),
                thr.get("txt", 0.25),
                thr.get("nms", 0.50),
            )
            if not ok:
                print(f"  Skipped saved DINO thresholds for unknown class "
                      f"'{cls_name}' — class is not in the current project.")

        # Update UI
        self.update_ui()

        # Handle missing images if any
        if missing_images:
            self.handle_missing_images(missing_images)

        # Select the first image if available
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
            first_item = self.image_list.item(0)
            if first_item:
                self.switch_image(first_item)

        # Select the first class if available
        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self.on_class_selected()

    def handle_missing_images(self, missing_images):
        message = "The following images have annotations but were not found in the project directory:\n\n"
        message += "\n".join(missing_images[:10])  # Show first 10 missing images
        if len(missing_images) > 10:
            message += f"\n... and {len(missing_images) - 10} more."
        message += "\n\nWould you like to locate these images now?"

        reply = QMessageBox.question(
            self,
            "Missing Images",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.load_missing_images(missing_images)
        else:
            self.remove_missing_images(missing_images)

    def remove_missing_images(self, missing_images):
        for image_name in missing_images:
            # Remove from all_images
            self.all_images = [
                img for img in self.all_images if img["file_name"] != image_name
            ]

            # Remove from image_paths
            self.image_paths.pop(image_name, None)

            # Remove from all_annotations
            self.all_annotations.pop(image_name, None)

            # If it's a multi-slice image, remove all related slices
            base_name = os.path.splitext(image_name)[0]
            if base_name in self.image_slices:
                for slice_name, _ in self.image_slices[base_name]:
                    self.all_annotations.pop(slice_name, None)
                del self.image_slices[base_name]

        self.update_ui()
        QMessageBox.information(
            self,
            "Images Removed",
            f"{len(missing_images)} missing images and their annotations have been removed from the project.",
        )

    def prompt_load_missing_images(self, missing_images):
        message = "The following images have annotations but were not found in the project directory:\n\n"
        message += "\n".join(missing_images[:10])  # Show first 10 missing images
        if len(missing_images) > 10:
            message += f"\n... and {len(missing_images) - 10} more."
        message += "\n\nWould you like to locate these images now?"

        reply = QMessageBox.question(
            self,
            "Load Missing Images",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.load_missing_images(missing_images)

    def load_missing_images(self, missing_images):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Missing Images",
            "",
            "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)",
        )
        if files:
            images_loaded = 0
            for file_path in files:
                file_name = os.path.basename(file_path)
                if file_name in missing_images:
                    dst_path = os.path.join(
                        self.current_project_dir, "images", file_name
                    )
                    shutil.copy2(file_path, dst_path)
                    self.image_paths[file_name] = dst_path

                    # Add the image to all_images if it's not already there
                    if not any(
                        img["file_name"] == file_name for img in self.all_images
                    ):
                        self.all_images.append(
                            {
                                "file_name": file_name,
                                "height": 0,
                                "width": 0,
                                "id": len(self.all_images) + 1,
                                "is_multi_slice": False,
                            }
                        )
                    images_loaded += 1
                    missing_images.remove(file_name)

            self.update_image_list()
            if images_loaded > 0:
                self.image_list.setCurrentRow(0)  # Select the first image
                self.switch_image(self.image_list.item(0))  # Display the first image
            QMessageBox.information(
                self,
                "Images Loaded",
                f"Successfully copied and loaded {images_loaded} out of {len(files)} selected images.",
            )

            # If there are still missing images, prompt again
            if missing_images:
                self.prompt_load_missing_images(missing_images)

    def update_image_list(self):
        self.image_list.clear()
        for image_info in self.all_images:
            self.image_list.addItem(image_info["file_name"])

    def select_class(self, index):
        if 0 <= index < self.class_list.count():
            item = self.class_list.item(index)
            self.class_list.setCurrentItem(item)
            self.current_class = item.text()
            print(f"Selected class: {self.current_class}")
        else:
            print("Invalid class index")

    def close_project(self):
        if hasattr(self, "current_project_file"):
            reply = QMessageBox.question(
                self,
                "Close Project",
                "Do you want to save the current project before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.remove_all_temp_annotations()  # Remove temp annotations before saving
                self.save_project(show_message=False)  # Save without showing a message
            elif reply == QMessageBox.StandardButton.Cancel:
                return  # User cancelled the operation

        # Clear all data
        self.clear_all(new_project=True, show_messages=False)

        # Reset project-related attributes
        if hasattr(self, "current_project_file"):
            del self.current_project_file
        if hasattr(self, "current_project_dir"):
            del self.current_project_dir

        # Update the window title
        self.update_window_title()

    def delete_selected_class(self):
        selected_items = self.class_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "No Selection", "Please select a class to delete."
            )
            return

        class_name = selected_items[0].text()
        reply = QMessageBox.question(
            self,
            "Delete Class",
            f"Are you sure you want to delete the class '{class_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_class(
                class_name
            )  # Sreeni note: Implement this method to handle class deletion

    def check_missing_images(self):
        missing_images = [
            img["file_name"]
            for img in self.all_images
            if img["file_name"] not in self.image_paths
            or not os.path.exists(self.image_paths[img["file_name"]])
        ]
        if missing_images:
            self.prompt_load_missing_images(missing_images)

    def convert_to_serializable(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {
                key: self.convert_to_serializable(value) for key, value in obj.items()
            }
        else:
            return obj

    def save_project(self, show_message=True):
        if not hasattr(self, "current_project_file") or not self.current_project_file:
            self.current_project_file, _ = QFileDialog.getSaveFileName(
                self, "Save Project", "", "Image Annotator Project (*.iap)"
            )
            if not self.current_project_file:
                return  # User cancelled the save dialog

        self.current_project_dir = os.path.dirname(self.current_project_file)

        # Check if images are in the correct directory structure
        images_dir = os.path.join(self.current_project_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        images_to_copy = []
        for file_name, src_path in self.image_paths.items():
            dst_path = os.path.join(images_dir, file_name)
            if os.path.abspath(src_path) != os.path.abspath(dst_path):
                if not os.path.exists(dst_path):
                    images_to_copy.append((file_name, src_path, dst_path))

        if images_to_copy:
            reply = QMessageBox.question(
                self,
                "Image Directory Structure",
                f"The project structure requires all images to be in an 'images' subdirectory. "
                f"{len(images_to_copy)} images need to be copied to the correct location. "
                f"Do you want to copy these images?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )

            if reply == QMessageBox.StandardButton.Yes:
                for file_name, src_path, dst_path in images_to_copy:
                    try:
                        shutil.copy2(src_path, dst_path)
                        self.image_paths[file_name] = dst_path
                    except Exception as e:
                        QMessageBox.warning(
                            self, "Copy Failed", f"Failed to copy {file_name}: {str(e)}"
                        )
                        return
            else:
                QMessageBox.warning(
                    self,
                    "Save Cancelled",
                    "Project cannot be saved without the correct directory structure.",
                )
                return

        # Prepare image data
        images_data = []
        for image_info in self.all_images:
            file_name = image_info["file_name"]
            image_data = {
                "file_name": file_name,
                "width": image_info["width"],
                "height": image_info["height"],
                "is_multi_slice": image_info["is_multi_slice"],
            }

            if image_data["is_multi_slice"]:
                base_name_without_ext = os.path.splitext(file_name)[0]
                image_data["slices"] = []
                for slice_name, _ in self.image_slices.get(base_name_without_ext, []):
                    slice_data = {
                        "name": slice_name,
                        "annotations": self.convert_to_serializable(
                            self.all_annotations.get(slice_name, {})
                        ),
                    }
                    image_data["slices"].append(slice_data)

                image_data["dimensions"] = self.convert_to_serializable(
                    self.image_dimensions.get(base_name_without_ext, [])
                )
                image_data["shape"] = self.convert_to_serializable(
                    self.image_shapes.get(base_name_without_ext, [])
                )
            else:
                image_data["annotations"] = {}
                for class_name, annotations in self.all_annotations.get(
                    file_name, {}
                ).items():
                    image_data["annotations"][class_name] = [
                        ann.copy() for ann in annotations
                    ]

            images_data.append(image_data)

        # Create project data
        project_data = {
            "classes": [
                {"name": name, "color": color.name()}
                for name, color in self.image_label.class_colors.items()
            ],
            "images": images_data,
            "image_paths": {
                k: v for k, v in self.image_paths.items() if os.path.exists(v)
            },
            "notes": getattr(self, "project_notes", ""),
            "creation_date": getattr(
                self, "project_creation_date", datetime.now().isoformat()
            ),
            "last_modified": datetime.now().isoformat(),
        }

        # Persist DINO configuration by snapshotting the widgets that own it.
        dino_cfg = {
            "phrases": self.dino_phrase_panel.get_all_phrases(),
            "thresholds": self.dino_class_table.get_thresholds_dict(),
        }
        if dino_cfg["phrases"] or dino_cfg["thresholds"]:
            project_data["dino_config"] = dino_cfg

        # Save project data
        with open(self.current_project_file, "w") as f:
            json.dump(self.convert_to_serializable(project_data), f, indent=2)

        if show_message:
            self.show_info(
                "Project Saved", f"Project saved to {self.current_project_file}"
            )

        # Update the window title
        self.update_window_title()

        # Update image_paths to reflect the correct locations
        for file_name in self.image_paths.keys():
            self.image_paths[file_name] = os.path.join(images_dir, file_name)

    def save_project_as(self):
        new_project_file, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "Image Annotator Project (*.iap)"
        )
        if new_project_file:
            # Ensure the file has the correct extension
            if not new_project_file.lower().endswith(".iap"):
                new_project_file += ".iap"

            # Store the original project file
            original_project_file = getattr(self, "current_project_file", None)

            # Set the new project file as the current one
            self.current_project_file = new_project_file
            self.current_project_dir = os.path.dirname(new_project_file)

            # Save the project with the new name
            self.save_project(show_message=False)

            # Update the window title
            self.update_window_title()

            # Show a success message
            QMessageBox.information(
                self, "Project Saved As", f"Project saved as:\n{new_project_file}"
            )

            # If this was originally a new unsaved project, update the original project file
            if original_project_file is None:
                self.current_project_file = new_project_file

    def auto_save(self):
        if self.is_loading_project:
            return  # Skip auto-save during project loading

        if not hasattr(self, "current_project_file"):
            reply = QMessageBox.question(
                self,
                "No Project",
                "You need to save the project before auto-saving. Would you like to save now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.save_project()
            else:
                return

        if hasattr(self, "current_project_file"):
            self.save_project(show_message=False)
            print("Project auto-saved.")

    def show_project_details(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        from .annotation_statistics import AnnotationStatisticsDialog
        from .project_details import ProjectDetailsDialog

        # Generate annotation statistics
        stats_dialog = AnnotationStatisticsDialog(self)
        stats_dialog.generate_statistics(self.all_annotations)

        dialog = ProjectDetailsDialog(self, stats_dialog)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.were_changes_made():
                self.project_notes = dialog.get_notes()
                self.save_project(show_message=False)
                QMessageBox.information(
                    self, "Project Details", "Project details have been updated."
                )
            else:
                print("No changes made to project details.")

    def load_multi_slice_image(self, image_path, dimensions=None, shape=None):

        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        print(f"Loading multi-slice image: {image_path}")
        print(f"Base name: {base_name}")

        if dimensions and shape:
            print(f"Using stored dimensions: {dimensions}")
            print(f"Using stored shape: {shape}")
            self.image_dimensions[base_name] = dimensions
            self.image_shapes[base_name] = shape
            if image_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(image_path, dimensions, shape)
            elif image_path.lower().endswith(".czi"):
                self.load_czi(image_path, dimensions, shape)
        else:
            print("No stored dimensions or shape, loading as new image")
            if image_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(image_path)
            elif image_path.lower().endswith(".czi"):
                self.load_czi(image_path)

        print(f"Loaded multi-slice image: {file_name}")
        print(f"Dimensions: {self.image_dimensions.get(base_name, 'Not found')}")
        print(f"Shape: {self.image_shapes.get(base_name, 'Not found')}")
        print(f"Number of slices: {len(self.slices)}")

        if self.slices:
            self.current_image = self.slices[0][1]
            self.current_slice = self.slices[0][0]

            self.update_slice_list()
            self.slice_list.setCurrentRow(0)
            self.activate_slice(self.current_slice)
            print(f"Activated first slice: {self.current_slice}")
        else:
            print("No slices were loaded")
            self.current_image = None
            self.current_slice = None

        self.update_slice_list()
        self.image_label.update()

    # print(f"Loaded slices: {[slice_name for slice_name, _ in self.slices]}")

    def activate_sam_magic_wand(self):
        # Uncheck all other tools
        for button in self.tool_group.buttons():
            if button != self.sam_magic_wand_button:
                button.setChecked(False)

        # Set the current tool
        self.image_label.current_tool = "sam_magic_wand"
        self.image_label.sam_magic_wand_active = True
        self.image_label.setCursor(Qt.CursorShape.CrossCursor)

        # Update UI based on the current tool
        self.update_ui_for_current_tool()

        # If a class is not selected, select the first one (if available)
        if self.current_class is None and self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self.current_class = self.class_list.currentItem().text()
        elif self.class_list.count() == 0:
            QMessageBox.warning(
                self,
                "No Class Selected",
                "Please add a class before using annotation tools.",
            )
            self.sam_magic_wand_button.setChecked(False)
            self.deactivate_sam_magic_wand()

    def deactivate_sam_magic_wand(self):
        self.image_label.current_tool = None
        self.image_label.sam_magic_wand_active = False
        self.sam_magic_wand_button.setChecked(False)
        self.sam_magic_wand_button.setEnabled(False)  # Disable the button
        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

        # Clear any SAM-related temporary data
        self.image_label.sam_bbox = None
        self.image_label.drawing_sam_bbox = False
        self.image_label.temp_sam_prediction = None

        # Update UI based on the current tool
        self.update_ui_for_current_tool()

    def toggle_sam_assisted(self):
        if not self.current_sam_model:
            QMessageBox.warning(
                self,
                "No SAM Model Selected",
                "Please pick a SAM model before using the SAM-Assisted tool.",
            )
            self.sam_magic_wand_button.setChecked(False)
            return

        if self.sam_magic_wand_button.isChecked():
            self.activate_sam_magic_wand()
        else:
            self.deactivate_sam_magic_wand()

        self.image_label.clear_temp_sam_prediction()  # Clear temporary prediction

    def toggle_sam_magic_wand(self):
        if self.sam_magic_wand_button.isChecked():
            if self.current_class is None:
                QMessageBox.warning(
                    self,
                    "No Class Selected",
                    "Please select a class before using SAM2 Magic Wand.",
                )
                self.sam_magic_wand_button.setChecked(False)
                return
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.image_label.sam_magic_wand_active = True
        else:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.image_label.sam_magic_wand_active = False
            self.image_label.sam_bbox = None

        self.image_label.clear_temp_sam_prediction()  # Clear temporary prediction

    def schedule_sam_prediction(self):
        """Restart the debounce timer; inference fires 1s after last click."""
        self.sam_inference_timer.stop()
        self.sam_inference_timer.start(1000)

    def apply_sam_prediction(self):
        # Re-entry guard: if a previous SAM call is still in flight, the
        # event-loop pump inside _run_sync can deliver this timer fire
        # before the first call returns. Bail and rely on the user
        # clicking again (which restarts the debounce) to issue a fresh
        # inference with the up-to-date point set.
        if self._sam_inference_in_flight:
            return
        self._sam_inference_in_flight = True
        try:
            try:
                if self.image_label.current_tool == "sam_box":
                    if self.image_label.sam_bbox is None:
                        print("SAM bbox is None")
                        return
                    x1, y1, x2, y2 = self.image_label.sam_bbox
                    bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    prediction = self.sam_utils.apply_sam_prediction(self.current_image, bbox)
                    self.image_label.sam_bbox = None
                elif self.image_label.current_tool == "sam_points":
                    # Always use all points!
                    pos_points = self.image_label.sam_positive_points
                    neg_points = self.image_label.sam_negative_points
                    print(
                        f"[SAM-POINTS] Predicting with {len(pos_points)} positive points: {pos_points} "
                        f"and {len(neg_points)} negative points: {neg_points}"
                    )
                    if not pos_points:
                        print("No positive points for SAM-points")
                        return
                    prediction = self.sam_utils.apply_sam_points(
                        self.current_image,
                        pos_points,
                        neg_points,
                    )
                else:
                    return
            except InferenceBusyError:
                # Re-entry safety net from sam_utils. The call-site flag
                # above should catch this first, but if a different
                # caller drives inference concurrently we just skip —
                # the user keeps interacting; their next click will
                # restart the debounce.
                return
            except Exception as exc:
                traceback.print_exc()
                QMessageBox.critical(
                    self,
                    "SAM Error",
                    f"SAM inference failed:\n\n{exc}\n\n"
                    "See the log for details.",
                )
                return

            if prediction:
                temp_annotation = {
                    "segmentation": prediction["segmentation"],
                    "category_id": self.class_mapping[self.current_class],
                    "category_name": self.current_class,
                    "score": prediction["score"],
                }
                self.image_label.temp_sam_prediction = temp_annotation
                self.image_label.update()
            elif prediction is None:
                QMessageBox.information(
                    self,
                    "SAM",
                    "No mask matches the given constraints. "
                    "Try adjusting the box or point positions."
                )
            else:
                print("Failed to generate prediction")

            # Only clear box/points for box mode, not for points mode!
            if self.image_label.current_tool == "sam_box":
                self.image_label.sam_bbox = None
                self.image_label.update()
        finally:
            self._sam_inference_in_flight = False

    def accept_sam_prediction(self):
        if self.image_label.temp_sam_prediction:
            new_annotation = self.image_label.temp_sam_prediction
            self.image_label.annotations.setdefault(
                new_annotation["category_name"], []
            ).append(new_annotation)
            self.add_annotation_to_list(new_annotation)
            self.save_current_annotations()
            self.update_slice_list_colors()
            self.image_label.temp_sam_prediction = None
            # --- Clear points after accepting
            self.image_label.sam_positive_points = []
            self.image_label.sam_negative_points = []
            self.image_label.update()
            print("SAM prediction accepted, points cleared, and added to annotations.")

    def setup_slice_list(self):
        self.slice_list = QListWidget()
        self.slice_list.itemClicked.connect(self.switch_slice)
        self.image_list_layout.addWidget(QLabel("Slices:"))
        self.image_list_layout.addWidget(self.slice_list)

    def open_images(self):
        file_names, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Images",
            "",
            "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)",
        )
        if file_names:
            self.image_list.clear()
            self.image_paths.clear()
            self.all_images.clear()
            self.slice_list.clear()
            self.slices.clear()
            self.current_stack = None
            self.current_slice = None
            self.add_images_to_list(file_names)

    def convert_to_8bit_rgb(self, image_array):
        if image_array.ndim == 2:
            # Grayscale image
            image_8bit = self.normalize_array(image_array)
            return np.stack((image_8bit,) * 3, axis=-1)
        elif image_array.ndim == 3:
            if image_array.shape[2] == 3:
                # Already RGB, just normalize
                return self.normalize_array(image_array)
            elif image_array.shape[2] > 3:
                # Multi-channel image, use first three channels
                rgb_array = image_array[:, :, :3]
                return self.normalize_array(rgb_array)

        raise ValueError(f"Unsupported image shape: {image_array.shape}")

    def add_images_to_list(self, file_names):
        first_added_item = None
        for file_name in file_names:
            base_name = os.path.basename(file_name)
            if base_name not in self.image_paths:
                image_info = {
                    "file_name": base_name,
                    "height": 0,
                    "width": 0,
                    "id": len(self.all_images) + 1,
                    "is_multi_slice": False,
                }

                # Detect multi-slice images and set dimensions
                if file_name.lower().endswith((".tif", ".tiff", ".czi")):
                    self.load_multi_slice_image(file_name)
                    base_name_without_ext = os.path.splitext(base_name)[0]
                    if (
                        base_name_without_ext in self.image_slices
                        and self.image_slices[base_name_without_ext]
                    ):
                        first_slice_name, first_slice = self.image_slices[
                            base_name_without_ext
                        ][0]
                        image_info["height"] = first_slice.height()
                        image_info["width"] = first_slice.width()
                        image_info["is_multi_slice"] = True
                        image_info["dimensions"] = self.image_dimensions.get(
                            base_name_without_ext, []
                        )
                        image_info["shape"] = self.image_shapes.get(
                            base_name_without_ext, []
                        )
                else:
                    # For regular images
                    image = QImage(file_name)
                    image_info["height"] = image.height()
                    image_info["width"] = image.width()

                self.all_images.append(image_info)
                item = QListWidgetItem(base_name)
                self.image_list.addItem(item)
                if first_added_item is None:
                    first_added_item = item

                # Update image_paths with the original file path
                self.image_paths[base_name] = file_name

        if first_added_item:
            self.image_list.setCurrentItem(first_added_item)
            self.switch_image(first_added_item)

        if not self.is_loading_project:
            self.auto_save()

    def update_all_images(self, new_image_info):
        for info in new_image_info:
            if not any(
                img["file_name"] == info["file_name"] for img in self.all_images
            ):
                self.all_images.append(info)

    def closeEvent(self, event):
        if not self.image_label.check_unsaved_changes():
            event.ignore()
            return
        event.accept()

        if (
            self.image_label.temp_paint_mask is not None
            or self.image_label.temp_eraser_mask is not None
        ):
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save them before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Yes:
                if self.image_label.temp_paint_mask is not None:
                    self.image_label.commit_paint_annotation()
                if self.image_label.temp_eraser_mask is not None:
                    self.image_label.commit_eraser_changes()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

        # Perform any other cleanup or saving operations here
        event.accept()

    def switch_slice(self, item):
        if item is None:
            return
        if not self.image_label.check_unsaved_changes():
            return

        # Check for unsaved changes
        if (
            self.image_label.temp_paint_mask is not None
            or self.image_label.temp_eraser_mask is not None
        ):
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Yes:
                if self.image_label.temp_paint_mask is not None:
                    self.image_label.commit_paint_annotation()
                if self.image_label.temp_eraser_mask is not None:
                    self.image_label.commit_eraser_changes()
            elif reply == QMessageBox.StandardButton.Cancel:
                return
            else:
                self.image_label.discard_paint_annotation()
                self.image_label.discard_eraser_changes()

        self.save_current_annotations()
        self.image_label.clear_temp_sam_prediction()

        slice_name = item.text()
        for name, qimage in self.slices:
            if name == slice_name:
                self.current_image = qimage
                self.current_slice = name
                self.display_image()
                self.load_image_annotations()
                self.update_annotation_list()
                self.clear_highlighted_annotation()
                self.image_label.highlighted_annotations.clear()  # Add this line
                self.image_label.reset_annotation_state()
                self.image_label.clear_current_annotation()
                self.update_image_info()
                break

        # Ensure the UI is updated
        self.image_label.update()
        self.update_slice_list_colors()

        # Reset zoom level to default (1.0)
        self.set_zoom(1.0)

        # Sync DINO temp_annotations to the new slice (carry over masks
        # from the previous slice was a reported bug).
        self._refresh_dino_temp_for_current()

    def switch_image(self, item):
        if item is None:
            return
        if not self.image_label.check_unsaved_changes():
            return

        # Store the current item before checking temp annotations
        current_item = self.image_list.currentItem()

        if not self.check_temp_annotations():
            # If the user chooses not to discard temp annotations, revert the selection
            self.image_list.setCurrentItem(current_item)
            return

        self.save_current_annotations()
        self.image_label.clear_temp_sam_prediction()
        self.image_label.exit_editing_mode()

        file_name = item.text()
        print(f"\nSwitching to image: {file_name}")

        image_info = next(
            (img for img in self.all_images if img["file_name"] == file_name), None
        )

        if image_info:
            self.image_file_name = file_name
            image_path = self.image_paths.get(file_name)

            if not image_path:
                image_path = os.path.join(self.current_project_dir, "images", file_name)

            if image_path and os.path.exists(image_path):
                if image_info.get("is_multi_slice", False):
                    base_name = os.path.splitext(file_name)[0]
                    if base_name in self.image_slices:
                        self.slices = self.image_slices[base_name]
                        if self.slices:
                            self.current_image = self.slices[0][1]
                            self.current_slice = self.slices[0][0]
                            self.update_slice_list()
                            self.activate_slice(self.current_slice)
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

                self.load_image_annotations()
                self.update_annotation_list()
                self.clear_highlighted_annotation()

                self.image_label.highlighted_annotations.clear()
                self.image_label.update()
                self.image_label.reset_annotation_state()
                self.image_label.clear_current_annotation()
                self.update_image_info()

                self.adjust_zoom_to_fit()
            else:
                self.current_image = None
                self.image_label.clear()
                self.load_image_annotations()
                self.update_annotation_list()
                self.update_image_info()

            self.image_list.setCurrentItem(item)
            self.image_label.update()
            self.update_slice_list_colors()
        else:
            self.current_image = None
            self.current_slice = None
            self.image_label.clear()
            self.update_image_info()
            self.clear_slice_list()

        # Sync DINO temp_annotations to the new image (mask carry-over
        # bug from single-image review and batch review).
        self._refresh_dino_temp_for_current()

    def adjust_zoom_to_fit(self):
        if not self.current_image:
            return

        # Get the dimensions of the image and the display area
        image_width = self.current_image.width()
        image_height = self.current_image.height()
        display_width = self.scroll_area.viewport().width()
        display_height = self.scroll_area.viewport().height()

        # Calculate and apply the zoom factor to fit the longest side
        zoom_factor = min(display_width / image_width, display_height / image_height)
        self.set_zoom(zoom_factor)

    def activate_current_slice(self):
        if self.current_slice:
            # Ensure the current slice is selected in the slice list
            items = self.slice_list.findItems(self.current_slice, Qt.MatchFlag.MatchExactly)
            if items:
                self.slice_list.setCurrentItem(items[0])

            # Load annotations for the current slice
            self.load_image_annotations()

            # Update the image label
            self.image_label.update()

            # Update the annotation list
            self.update_annotation_list()

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
        print(f"Loading TIFF file: {image_path}")
        axes_hint = None
        with TiffFile(image_path) as tif:
            print(f"TIFF tags: {tif.pages[0].tags}")

            # Try to access metadata if available
            try:
                metadata = tif.pages[0].tags["ImageDescription"].value
                print(f"TIFF metadata: {metadata}")
            except KeyError:
                print("No ImageDescription metadata found")

            # Try to read axis labels from the tifffile series. ImageJ /
            # OME-TIFF stores axes like "TZCYX" — we can prefill the
            # dimension dialog with the right labels so the user just
            # clicks OK instead of guessing per axis. Map tifffile's
            # axes vocabulary (T,Z,C,S,Y,X) to the app's (T,Z,C,S,H,W).
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
                        print(f"TIFF series axes: {series_axes} → dimension hint: {axes_hint}")
                    else:
                        unknown = [a for a in series_axes if axis_map.get(a) is None]
                        print(f"TIFF series axes had unknown labels {unknown}, no hint applied")
            except Exception as e:
                print(f"Could not read TIFF series axes: {e}")

            # Check if it's a multi-page TIFF
            if len(tif.pages) > 1:
                print(f"Multi-page TIFF detected. Number of pages: {len(tif.pages)}")
                # Read all pages into a 3D array
                image_array = tif.asarray()
            else:
                print("Single-page TIFF detected.")
                image_array = tif.pages[0].asarray()

            print(f"Image array shape: {image_array.shape}")
            print(f"Image array dtype: {image_array.dtype}")
            print(f"Image min: {image_array.min()}, max: {image_array.max()}")

        if dimensions and shape and not force_dimension_dialog:
            # Use stored dimensions and shape
            print(f"Using stored dimensions: {dimensions}")
            print(f"Using stored shape: {shape}")
            image_array = image_array.reshape(shape)
        else:
            # Process as before for new images or when forcing dimension dialog
            print("Processing as new image or forcing dimension dialog.")
            dimensions = None

        self.process_multidimensional_image(
            image_array, image_path, dimensions, force_dimension_dialog,
            axes_hint=axes_hint,
        )

    def load_czi(
        self, image_path, dimensions=None, shape=None, force_dimension_dialog=False
    ):
        print(f"Loading CZI file: {image_path}")
        with CziFile(image_path) as czi:
            image_array = czi.asarray()
            print(f"CZI array shape: {image_array.shape}")
            print(f"CZI array dtype: {image_array.dtype}")
            print(f"CZI array min: {image_array.min()}, max: {image_array.max()}")

        if dimensions and shape and not force_dimension_dialog:
            # Use stored dimensions and shape
            print(f"Using stored dimensions: {dimensions}")
            print(f"Using stored shape: {shape}")
            image_array = image_array.reshape(shape)
        else:
            # Process as before for new images or when forcing dimension dialog
            print("Processing as new image or forcing dimension dialog.")
            dimensions = None

        self.process_multidimensional_image(
            image_array, image_path, dimensions, force_dimension_dialog
        )

    def load_regular_image(self, image_path):
        self.current_image = QImage(image_path)
        self.slices = []
        self.slice_list.clear()
        self.current_slice = None

    def process_multidimensional_image(
        self, image_array, image_path, dimensions=None,
        force_dimension_dialog=False, axes_hint=None,
    ):
        file_name = os.path.basename(image_path)
        base_name = os.path.splitext(file_name)[0]
        print(f"Processing file: {file_name}")
        print(f"Image array shape: {image_array.shape}")
        print(f"Image array dtype: {image_array.dtype}")

        if dimensions is None or force_dimension_dialog:
            if image_array.ndim > 2:
                # Prefer the loader's metadata-derived hint (e.g. ImageJ
                # TIFF axes='TZCYX'). Fall back to a hand-crafted default
                # that covers ndim 3..6 so a user clicking OK without
                # tweaking the combos gets a sensible result. The earlier
                # `default_dimensions[-ndim:]` slice silently degraded for
                # ndim≥5: one axis ended up unset and inherited the combo
                # box's first item ("T"), producing 2560 wrong slices for
                # a 5D TZCYX file.
                if axes_hint and len(axes_hint) == image_array.ndim:
                    default_dimensions = list(axes_hint)
                    print(f"Applying axes hint as default dims: {default_dimensions}")
                else:
                    if axes_hint and len(axes_hint) != image_array.ndim:
                        print(
                            f"Ignoring axes hint (length {len(axes_hint)} "
                            f"vs ndim {image_array.ndim})"
                        )
                    ndim_defaults = {
                        3: ["Z", "H", "W"],
                        4: ["T", "Z", "H", "W"],
                        5: ["T", "Z", "C", "H", "W"],
                        6: ["T", "Z", "C", "S", "H", "W"],
                    }
                    # ndim ≥ 7 falls into the generic case: pad with
                    # "T" at the front so H / W are still the last two
                    # axes — that way "click OK" still produces a
                    # sensible 2D slice even on exotic inputs.
                    default_dimensions = ndim_defaults.get(
                        image_array.ndim,
                        ["T"] * max(0, image_array.ndim - 2) + ["H", "W"],
                    )

                # Show a progress dialog
                progress = QProgressDialog(
                    "Assigning dimensions...", "Cancel", 0, 100, self
                )
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(10)
                QApplication.processEvents()

                while True:
                    dialog = DimensionDialog(
                        image_array.shape, file_name, self, default_dimensions
                    )
                    # Qt6 no longer shows the "?" help button by default;
                    # the old WindowContextHelpButtonHint clear is gone.
                    progress.setValue(50)
                    QApplication.processEvents()
                    if dialog.exec():
                        dimensions = dialog.get_dimensions()
                        print(f"Assigned dimensions: {dimensions}")
                        if "H" in dimensions and "W" in dimensions:
                            self.image_dimensions[base_name] = dimensions
                            break
                        else:
                            QMessageBox.warning(
                                self,
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
                self.image_dimensions[base_name] = dimensions

        self.image_shapes[base_name] = image_array.shape
        print(f"Final assigned dimensions: {self.image_dimensions[base_name]}")
        print(f"Image shape: {self.image_shapes[base_name]}")

        if self.image_dimensions[base_name]:
            self.create_slices(
                image_array, self.image_dimensions[base_name], image_path
            )
        else:
            rgb_image = self.convert_to_8bit_rgb(image_array)
            self.current_image = self.array_to_qimage(rgb_image)
            self.slices = []
            self.slice_list.clear()

        if self.slices:
            self.current_image = self.slices[0][1]
            self.current_slice = self.slices[0][0]
            self.slice_list.setCurrentRow(0)
            self.load_image_annotations()
            self.image_label.update()

        self.update_image_info()

        # Update UI
        self.update_slice_list()
        self.update_annotation_list()
        self.image_label.update()

    def create_slices(self, image_array, dimensions, image_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        slices = []
        self.slice_list.clear()

        print(f"Creating slices for {base_name}")
        print(f"Dimensions: {dimensions}")
        print(f"Image array shape: {image_array.shape}")

        # Create and show progress dialog
        progress = QProgressDialog("Loading slices...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately

        # Handle 2D images
        if image_array.ndim == 2:
            progress.setValue(50)  # Update progress
            QApplication.processEvents()  # Allow GUI to update
            normalized_array = self.normalize_array(image_array)
            qimage = self.array_to_qimage(normalized_array)
            slice_name = f"{base_name}"
            slices.append((slice_name, qimage))
            self.add_slice_to_list(slice_name)
        else:
            # For 3D or higher dimensional arrays
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
                rgb_slice = self.convert_to_8bit_rgb(slice_array)
                qimage = self.array_to_qimage(rgb_slice)

                slice_name = f"{base_name}_{'_'.join([f'{dimensions[i]}{val+1}' for i, val in zip(slice_indices, _)])}"
                slices.append((slice_name, qimage))

                self.add_slice_to_list(slice_name)

                # Update progress
                progress_value = int((idx + 1) / total_slices * 100)
                progress.setValue(progress_value)
                QApplication.processEvents()  # Allow GUI to update

        progress.setValue(100)  # Ensure progress reaches 100%

        self.image_slices[base_name] = slices
        self.slices = slices

        if slices:
            self.current_image = slices[0][1]
            self.current_slice = slices[0][0]
            self.slice_list.setCurrentRow(0)

            self.activate_slice(self.current_slice)

            slice_info = f"Total slices: {len(slices)}"
            for dim, size in zip(dimensions, image_array.shape):
                if dim not in ["H", "W"]:
                    slice_info += f", {dim}: {size}"
            self.update_image_info(additional_info=slice_info)
        else:
            print("No slices were created")

        print(f"Created {len(slices)} slices for {base_name}")
        return slices

    def add_slice_to_list(self, slice_name):
        item = QListWidgetItem(slice_name)

        if self.dark_mode:
            # Dark mode
            item.setBackground(
                QColor(40, 40, 40)
            )  # Very dark gray background for all items
            if slice_name in self.all_annotations:
                # Muted steel-blue + light text; the prior light-blue
                # (173, 216, 230) bg + dark-gray text was painfully
                # bright on a dark sidebar.
                item.setForeground(QColor(235, 235, 235))
                item.setBackground(QColor(58, 95, 140))
            else:
                item.setForeground(QColor(200, 200, 200))  # Light gray text
        else:
            # Light mode
            item.setBackground(
                QColor(240, 240, 240)
            )  # Very light gray background for all items
            if slice_name in self.all_annotations:
                item.setForeground(QColor(255, 255, 255))  # White text
                item.setBackground(QColor(70, 130, 180))  # Medium-dark blue background
            else:
                item.setForeground(QColor(0, 0, 0))  # Black text

        self.slice_list.addItem(item)

    def normalize_array(self, array):
        # print(f"Normalizing array. Shape: {array.shape}, dtype: {array.dtype}")
        # print(f"Array min: {array.min()}, max: {array.max()}, mean: {array.mean()}")

        array_float = array.astype(np.float32)

        if array.dtype == np.uint16:
            array_normalized = (array_float - array.min()) / (array.max() - array.min())
        elif array.dtype == np.uint8:
            # For 8-bit images, use a simple contrast stretching
            p_low, p_high = np.percentile(
                array_float, (0, 100)
            )  # Change these to 1, 99 or something to stretch the contrast for visualizing 8 bit images
            array_normalized = np.clip(array_float, p_low, p_high)
            array_normalized = (array_normalized - p_low) / (p_high - p_low)
        else:
            array_normalized = (array_float - array.min()) / (array.max() - array.min())

        # Apply gamma correction
        gamma = 1.0  # Adjust this value to fine-tune brightness (> 1 for darker, < 1 for brighter)
        array_normalized = np.power(array_normalized, gamma)

        return (array_normalized * 255).astype(np.uint8)

    def adjust_contrast(self, image, low_percentile=1, high_percentile=99):
        if image.dtype != np.uint8:
            p_low, p_high = np.percentile(image, (low_percentile, high_percentile))
            image_adjusted = np.clip(image, p_low, p_high)
            image_adjusted = (image_adjusted - p_low) / (p_high - p_low)
            return (image_adjusted * 255).astype(np.uint8)
        return image

    def activate_slice(self, slice_name):
        self.current_slice = slice_name
        self.image_file_name = slice_name
        self.load_image_annotations()
        self.update_annotation_list()

        for name, qimage in self.slices:
            if name == slice_name:
                self.current_image = qimage
                self.display_image()
                break

        self.image_label.update()

        items = self.slice_list.findItems(slice_name, Qt.MatchFlag.MatchExactly)
        if items:
            self.slice_list.setCurrentItem(items[0])

    def array_to_qimage(self, array):
        if array.ndim == 2:
            height, width = array.shape
            return QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
        elif array.ndim == 3 and array.shape[2] == 3:
            height, width, _ = array.shape
            bytes_per_line = 3 * width
            return QImage(
                array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            )
        else:
            raise ValueError(
                f"Unsupported array shape {array.shape} for conversion to QImage"
            )

    def update_slice_list(self):
        self.slice_list.clear()
        for slice_name, _ in self.slices:
            item = QListWidgetItem(slice_name)
            if slice_name in self.all_annotations:
                item.setForeground(QColor(Qt.GlobalColor.green))
            else:
                item.setForeground(
                    QColor(Qt.GlobalColor.black) if not self.dark_mode else QColor(Qt.GlobalColor.white)
                )
            self.slice_list.addItem(item)

        # Select the current slice
        if self.current_slice:
            items = self.slice_list.findItems(self.current_slice, Qt.MatchFlag.MatchExactly)
            if items:
                self.slice_list.setCurrentItem(items[0])

    def clear_slice_list(self):
        self.slice_list.clear()
        self.slices = []
        self.current_slice = None

    def reset_tool_buttons(self):
        for button in self.tool_group.buttons():
            button.setChecked(False)

    def keyPressEvent(self, event):
        # Check if the current focus is on a text editing widget
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, (QLineEdit, QTextEdit)):
            super().keyPressEvent(event)
            return

        # F2 (Snake game) is wired as a global QShortcut in __init__
        # so it works when child widgets have focus. Don't re-handle here.
        if event.key() == Qt.Key.Key_Delete:
            # Handle deletions
            if self.class_list.hasFocus() and self.class_list.currentItem():
                self.delete_class(self.class_list.currentItem())
            elif (
                self.annotation_list.hasFocus() and self.annotation_list.selectedItems()
            ):
                self.delete_selected_annotations()
            elif self.image_list.hasFocus() and self.image_list.currentItem():
                self.delete_selected_image()
        elif event.key() == Qt.Key.Key_Up or event.key() == Qt.Key.Key_Down:
            # Handle slice navigation
            if self.slice_list.hasFocus():
                current_row = self.slice_list.currentRow()
                if event.key() == Qt.Key.Key_Up and current_row > 0:
                    self.slice_list.setCurrentRow(current_row - 1)
                elif (
                    event.key() == Qt.Key.Key_Down
                    and current_row < self.slice_list.count() - 1
                ):
                    self.slice_list.setCurrentRow(current_row + 1)
                self.switch_slice(self.slice_list.currentItem())
            else:
                # Pass the event to the parent for default handling
                super().keyPressEvent(event)
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # Handle accepting visible temporary classes
            if self.has_visible_temp_classes():
                self.accept_visible_temp_classes()
            else:
                super().keyPressEvent(event)
        elif event.key() == Qt.Key.Key_Escape:
            # Handle rejecting visible temporary classes
            if self.has_visible_temp_classes():
                self.reject_visible_temp_classes()
            else:
                super().keyPressEvent(event)
        else:
            # Pass any other key events to the parent for default handling
            super().keyPressEvent(event)

    def has_visible_temp_classes(self):
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.text().startswith("Temp-") and item.checkState() == Qt.CheckState.Checked:
                return True
        return False

    def launch_snake_game(self):
        # print("Launching Snake game")
        if not hasattr(self, "snake_game") or not self.snake_game.isVisible():
            self.snake_game = SnakeGame()
        self.snake_game.show()
        self.snake_game.setFocus()

    def import_annotations(self):
        if not self.image_label.check_unsaved_changes():
            return
        print("Starting import_annotations")
        import_format = self.import_format_selector.currentText()
        print(f"Import format: {import_format}")

        if import_format == "COCO JSON":
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Import COCO JSON Annotations", "", "JSON Files (*.json)"
            )
            if not file_name:
                print("No file selected, returning")
                return

            print(f"Selected file: {file_name}")
            json_dir = os.path.dirname(file_name)
            images_dir = os.path.join(json_dir, "images")
            imported_annotations, image_info = import_coco_json(
                file_name, self.class_mapping
            )

        elif import_format in ["YOLO (v4 and earlier)", "YOLO (v5+)"]:
            yaml_file, _ = QFileDialog.getOpenFileName(
                self, "Select YOLO Dataset YAML", "", "YAML Files (*.yaml *.yml)"
            )
            if not yaml_file:
                print("No YAML file selected, returning")
                return

            print(f"Selected YAML file: {yaml_file}")
            try:
                imported_annotations, image_info = process_import_format(
                    import_format, yaml_file, self.class_mapping
                )
                yaml_dir = os.path.dirname(yaml_file)
                if import_format == "YOLO (v4 and earlier)":
                    images_dir = os.path.join(yaml_dir, "train", "images")
                else:  # YOLO (v5+)
                    images_dir = os.path.join(
                        yaml_dir, "images", "train"
                    )  # Preferring train over val
            except ValueError as e:
                QMessageBox.warning(self, "Import Error", str(e))
                return

        else:
            QMessageBox.warning(
                self,
                "Unsupported Format",
                f"The selected format '{import_format}' is not implemented for import.",
            )
            return

        print(
            f"JSON/YOLO directory: {json_dir if import_format == 'COCO JSON' else os.path.dirname(yaml_file)}"
        )
        print(f"Images directory: {images_dir}")
        print(f"Imported annotations count: {len(imported_annotations)}")
        print(f"Image info count: {len(image_info)}")

        images_loaded = 0
        images_not_found = []

        for info in image_info.values():
            print(f"Processing image: {info['file_name']}")
            image_path = os.path.join(images_dir, info["file_name"])

            if os.path.exists(image_path):
                print(f"Image found at: {image_path}")
                self.image_paths[info["file_name"]] = image_path
                self.all_images.append(
                    {
                        "file_name": info["file_name"],
                        "height": info["height"],
                        "width": info["width"],
                        "id": info["id"],
                        "is_multi_slice": False,
                    }
                )
                images_loaded += 1
            else:
                print(f"Image not found at: {image_path}")
                images_not_found.append(info["file_name"])

        print(f"Images loaded: {images_loaded}")
        print(f"Images not found: {len(images_not_found)}")

        if images_not_found:
            message = f"The following {len(images_not_found)} images were not found in the 'images' directory:\n\n"
            message += "\n".join(images_not_found[:10])
            if len(images_not_found) > 10:
                message += f"\n... and {len(images_not_found) - 10} more."
            message += "\n\nDo you want to proceed and ignore annotations for these missing images?"
            reply = QMessageBox.question(
                self,
                "Missing Images",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                print("Import cancelled due to missing images")
                QMessageBox.information(
                    self,
                    "Import Cancelled",
                    "Import cancelled. Please ensure all images are in the 'images' directory and try again.",
                )
                return

        # Update annotations (only for found images)
        for image_name, annotations in imported_annotations.items():
            if image_name not in self.image_paths:
                continue
            self.all_annotations[image_name] = {}
            for category_name, category_annotations in annotations.items():
                self.all_annotations[image_name][category_name] = []
                for i, ann in enumerate(category_annotations, start=1):
                    new_ann = {
                        "segmentation": ann.get("segmentation"),
                        "bbox": ann.get("bbox"),
                        "category_id": ann["category_id"],
                        "category_name": category_name,
                        "number": i,
                        "type": ann.get("type", "polygon"),
                    }
                    self.all_annotations[image_name][category_name].append(new_ann)

        # Update class mapping and colors
        for annotations in self.all_annotations.values():
            for category_name in annotations.keys():
                if category_name not in self.class_mapping:
                    new_id = len(self.class_mapping) + 1
                    self.class_mapping[category_name] = new_id
                    self.image_label.class_colors[category_name] = QColor(
                        Qt.GlobalColor(new_id % 16 + 7)
                    )

        print("Updating UI")
        # Update UI
        self.update_class_list()
        self.update_image_list()
        self.update_annotation_list()

        # Highlight and display the first image
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)
            self.switch_image(self.image_list.item(0))

        # Select the first class if available
        if self.class_list.count() > 0:
            self.class_list.setCurrentRow(0)
            self.on_class_selected()

        self.image_label.update()

        message = f"Annotations have been imported successfully from {file_name if import_format == 'COCO JSON' else yaml_file}.\n"
        message += f"{images_loaded} images were loaded from the 'images' directory.\n"
        if images_not_found:
            message += (
                f"Annotations for {len(images_not_found)} missing images were ignored."
            )

        print("Import complete, showing message")
        QMessageBox.information(self, "Import Complete", message)
        self.auto_save()  # Auto-save after importing annotations

    def export_annotations(self):
        if not self.image_label.check_unsaved_changes():
            return
        export_format = self.export_format_selector.currentText()

        supported_formats = [
            "COCO JSON",
            "YOLO (v4 and earlier)",
            "YOLO (v5+)",
            "Labeled Images",
            "Semantic Labels",
            "Pascal VOC (BBox)",
            "Pascal VOC (BBox + Segmentation)",
        ]

        if export_format not in supported_formats:
            QMessageBox.warning(
                self,
                "Unsupported Format",
                f"The selected format '{export_format}' is not implemented.",
            )
            return

        if export_format == "COCO JSON":
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Export COCO JSON Annotations", "", "JSON Files (*.json)"
            )
        else:
            file_name = QFileDialog.getExistingDirectory(
                self, f"Select Output Directory for {export_format} Export"
            )

        if not file_name:
            return

        self.save_current_annotations()

        if export_format == "COCO JSON":
            output_dir = os.path.dirname(file_name)
            json_filename = os.path.basename(file_name)
            json_file, images_dir = export_coco_json(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                output_dir,
                json_filename,
            )
            message = (
                "Annotations have been exported successfully in COCO JSON format.\n"
            )
            message += f"JSON file: {json_file}\nImages directory: {images_dir}"

        elif export_format == "YOLO (v4 and earlier)":
            labels_dir, yaml_path = export_yolo_v4(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = "Annotations have been exported successfully in YOLO (v4 and earlier) format.\n"
            message += f"Labels: {labels_dir}\nYAML: {yaml_path}"

        elif export_format == "YOLO (v5+)":
            output_dir, yaml_path = export_yolo_v5plus(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = (
                "Annotations have been exported successfully in YOLO (v5+) format.\n"
            )
            message += f"Output directory: {output_dir}\nYAML: {yaml_path}"

        elif export_format == "Labeled Images":
            labeled_images_dir = export_labeled_images(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = f"Labeled images have been exported successfully.\nLabeled Images: {labeled_images_dir}\n"
            message += f"A class summary has been saved in: {os.path.join(labeled_images_dir, 'class_summary.txt')}"

        elif export_format == "Semantic Labels":
            semantic_labels_dir = export_semantic_labels(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = f"Semantic labels have been exported successfully.\nSemantic Labels: {semantic_labels_dir}\n"
            message += f"A class-pixel mapping has been saved in: {os.path.join(semantic_labels_dir, 'class_pixel_mapping.txt')}"

        elif export_format == "Pascal VOC (BBox)":
            voc_dir = export_pascal_voc_bbox(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = "Annotations have been exported successfully in Pascal VOC format (BBox only).\n"
            message += f"Pascal VOC Annotations: {voc_dir}"

        elif export_format == "Pascal VOC (BBox + Segmentation)":
            voc_dir = export_pascal_voc_both(
                self.all_annotations,
                self.class_mapping,
                self.image_paths,
                self.slices,
                self.image_slices,
                file_name,
            )
            message = "Annotations have been exported successfully in Pascal VOC format (BBox + Segmentation).\n"
            message += f"Pascal VOC Annotations: {voc_dir}"

        QMessageBox.information(self, "Export Complete", message)

    def save_slices(self, directory):
        slices_saved = False
        for image_file, image_slices in self.image_slices.items():
            for slice_name, qimage in image_slices:
                if (
                    slice_name in self.all_annotations
                    and self.all_annotations[slice_name]
                ):
                    file_path = os.path.join(directory, f"{slice_name}.png")
                    qimage.save(file_path, "PNG")
                    slices_saved = True

        return slices_saved

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

    def update_all_annotation_lists(self):
        for image_name in self.all_annotations.keys():
            self.update_annotation_list(image_name)
        self.update_annotation_list()  # Update for the current image/slice

    def update_annotation_list(self, image_name=None):
        self.annotation_list.clear()
        current_name = image_name or self.current_slice or self.image_file_name
        annotations = self.all_annotations.get(current_name, {})
        for class_name, class_annotations in annotations.items():
            if not class_name.startswith(
                "Temp-"
            ):  # Only show non-temporary annotations
                color = self.image_label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
                for annotation in class_annotations:
                    number = annotation.get("number", 0)
                    area = calculate_area(annotation)
                    item_text = f"{class_name} - {number:<3} Area: {area:.2f}"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, annotation)
                    item.setForeground(color)
                    self.annotation_list.addItem(item)

        # Force the annotation list to repaint
        self.annotation_list.repaint()

    def update_slice_list_colors(self):
        # Set the background color of the entire list widget
        if self.dark_mode:
            self.slice_list.setStyleSheet(
                "QListWidget { background-color: rgb(40, 40, 40); }"
            )
        else:
            self.slice_list.setStyleSheet(
                "QListWidget { background-color: rgb(240, 240, 240); }"
            )

        for i in range(self.slice_list.count()):
            item = self.slice_list.item(i)
            slice_name = item.text()

            if self.dark_mode:
                # Dark mode (annotated colors match add_slice_to_list —
                # muted steel-blue, light text; not the prior glaring
                # light-blue bg)
                if slice_name in self.all_annotations and any(
                    self.all_annotations[slice_name].values()
                ):
                    item.setForeground(QColor(235, 235, 235))
                    item.setBackground(QColor(58, 95, 140))
                else:
                    item.setForeground(QColor(200, 200, 200))  # Light gray text
                    item.setBackground(QColor(40, 40, 40))  # Very dark gray background
            else:
                # Light mode
                if slice_name in self.all_annotations and any(
                    self.all_annotations[slice_name].values()
                ):
                    item.setForeground(QColor(255, 255, 255))  # White text
                    item.setBackground(
                        QColor(70, 130, 180)
                    )  # Medium-dark blue background
                else:
                    item.setForeground(QColor(0, 0, 0))  # Black text
                    item.setBackground(
                        QColor(240, 240, 240)
                    )  # Very light gray background

        # Force the list to repaint
        self.slice_list.repaint()

    def update_annotation_list_colors(self, class_name=None, color=None):
        for i in range(self.annotation_list.count()):
            item = self.annotation_list.item(i)
            annotation = item.data(Qt.ItemDataRole.UserRole)
            # Update only the item for the specific class if class_name is provided
            if class_name is None or annotation["category_name"] == class_name:
                item_color = (
                    color
                    if class_name
                    else self.image_label.class_colors.get(
                        annotation["category_name"], QColor(Qt.GlobalColor.white)
                    )
                )
                item.setForeground(item_color)

    def load_image_annotations(self):
        # print(f"Loading annotations for: {self.current_slice or self.image_file_name}")
        self.image_label.annotations.clear()
        current_name = self.current_slice or self.image_file_name
        # print(f"Current name for annotations: {current_name}")
        # print(f"All annotations keys: {list(self.all_annotations.keys())}")
        if current_name in self.all_annotations:
            self.image_label.annotations = copy.deepcopy(
                self.all_annotations[current_name]
            )
            # print(f"Loaded annotations: {self.image_label.annotations}")
        else:
            print(f"No annotations found for {current_name}")
        self.image_label.update()

    def save_current_annotations(self):
        if self.current_slice:
            current_name = self.current_slice
        elif self.image_file_name:
            current_name = self.image_file_name
        else:
            # print("Error: No current slice or image file name set")
            return

        # print(f"Saving annotations for: {current_name}")
        if self.image_label.annotations:
            self.all_annotations[current_name] = self.image_label.annotations.copy()
            # print(f"Saved {len(self.image_label.annotations)} annotations for {current_name}")
        elif current_name in self.all_annotations:
            del self.all_annotations[current_name]
            # print(f"Removed annotations for {current_name}")

        self.update_slice_list_colors()

        # print(f"All annotations now: {self.all_annotations.keys()}")
        # print(f"Current slice: {self.current_slice}")
        # print(f"Current image_file_name: {self.image_file_name}")

    def setup_class_list(self):
        """Set up the class list widget."""
        self.class_list = QListWidget()
        self.class_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self.show_class_context_menu)
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.sidebar_layout.addWidget(QLabel("Classes:"))
        self.sidebar_layout.addWidget(self.class_list)

    def setup_tool_buttons(self):
        """Set up the tool buttons with grouped manual and automated tools."""
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)

        # Create a widget for manual tools
        manual_tools_widget = QWidget()
        manual_layout = QVBoxLayout(manual_tools_widget)
        manual_layout.setSpacing(5)

        manual_label = QLabel("Manual Tools")
        manual_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        manual_layout.addWidget(manual_label)

        manual_buttons_layout = QHBoxLayout()
        self.polygon_button = QPushButton("Polygon")
        self.polygon_button.setCheckable(True)
        self.rectangle_button = QPushButton("Rectangle")
        self.rectangle_button.setCheckable(True)
        manual_buttons_layout.addWidget(self.polygon_button)
        manual_buttons_layout.addWidget(self.rectangle_button)
        manual_layout.addLayout(manual_buttons_layout)

        self.tool_group.addButton(self.polygon_button)
        self.tool_group.addButton(self.rectangle_button)
        self.polygon_button.clicked.connect(self.toggle_tool)
        self.rectangle_button.clicked.connect(self.toggle_tool)

        # Create a widget for automated tools
        automated_tools_widget = QWidget()
        automated_layout = QVBoxLayout(automated_tools_widget)
        automated_layout.setSpacing(5)

        automated_label = QLabel("Automated Tools")
        automated_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        automated_layout.addWidget(automated_label)

        automated_buttons_layout = QHBoxLayout()
        self.sam_magic_wand_button = QPushButton("Magic Wand")
        self.sam_magic_wand_button.setCheckable(True)
        automated_buttons_layout.addWidget(self.sam_magic_wand_button)
        automated_layout.addLayout(automated_buttons_layout)

        self.tool_group.addButton(self.sam_magic_wand_button)
        self.sam_magic_wand_button.clicked.connect(self.toggle_tool)

        # Add the grouped tools to the sidebar layout
        self.sidebar_layout.addWidget(manual_tools_widget)
        self.sidebar_layout.addWidget(automated_tools_widget)

        # Set a fixed size for all buttons to make them smaller
        for button in [
            self.polygon_button,
            self.rectangle_button,
            self.load_sam2_button,
            self.sam_magic_wand_button,
        ]:
            button.setFixedSize(100, 30)

    def setup_annotation_list(self):
        """Set up the annotation list widget."""
        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.annotation_list.itemSelectionChanged.connect(
            self.update_highlighted_annotations
        )

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # Project Menu
        project_menu = menu_bar.addMenu("&Project")

        new_project_action = QAction("&New Project", self)
        new_project_action.setShortcut(QKeySequence.StandardKey.New)
        new_project_action.triggered.connect(self.new_project)
        project_menu.addAction(new_project_action)

        open_project_action = QAction("&Open Project", self)
        open_project_action.setShortcut(QKeySequence.StandardKey.Open)
        open_project_action.triggered.connect(self.open_project)
        project_menu.addAction(open_project_action)

        save_project_action = QAction("&Save Project", self)
        save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        save_project_action.triggered.connect(self.save_project)
        project_menu.addAction(save_project_action)

        save_project_as_action = QAction("Save Project &As...", self)
        save_project_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_project_as_action.triggered.connect(self.save_project_as)
        project_menu.addAction(save_project_as_action)

        close_project_action = QAction("&Close Project", self)
        close_project_action.setShortcut(QKeySequence("Ctrl+W"))
        close_project_action.triggered.connect(self.close_project)
        project_menu.addAction(close_project_action)

        project_details_action = QAction("Project &Details", self)
        project_details_action.setShortcut(QKeySequence("Ctrl+I"))
        project_details_action.triggered.connect(self.show_project_details)
        project_menu.addAction(project_details_action)

        search_projects_action = QAction("&Search Projects", self)
        search_projects_action.setShortcut(QKeySequence("Ctrl+F"))
        search_projects_action.triggered.connect(self.show_project_search)
        project_menu.addAction(search_projects_action)

        # Settings Menu
        settings_menu = menu_bar.addMenu("&Settings")

        font_size_menu = settings_menu.addMenu("&Font Size")
        for size in ["Small", "Medium", "Large", "XL", "XXL"]:
            action = QAction(size, self)
            action.triggered.connect(lambda checked, s=size: self.change_font_size(s))
            font_size_menu.addAction(action)

        toggle_dark_mode_action = QAction("Toggle &Dark Mode", self)
        toggle_dark_mode_action.setShortcut(QKeySequence("Ctrl+D"))
        toggle_dark_mode_action.triggered.connect(self.toggle_dark_mode)
        settings_menu.addAction(toggle_dark_mode_action)

        # Tools Menu
        tools_menu = menu_bar.addMenu("&Tools")

        annotation_stats_action = QAction("Annotation Statistics", self)
        annotation_stats_action.triggered.connect(self.show_annotation_statistics)
        annotation_stats_action.setShortcut(QKeySequence("Ctrl+Alt+S"))
        tools_menu.addAction(annotation_stats_action)

        coco_json_combiner_action = QAction("COCO JSON Combiner", self)
        coco_json_combiner_action.triggered.connect(self.show_coco_json_combiner)
        tools_menu.addAction(coco_json_combiner_action)

        dataset_splitter_action = QAction("Dataset Splitter", self)
        dataset_splitter_action.triggered.connect(self.open_dataset_splitter)
        tools_menu.addAction(dataset_splitter_action)

        dino_merge_action = QAction("Merge COCO for Training", self)
        dino_merge_action.triggered.connect(self.show_dino_merge_dialog)
        tools_menu.addAction(dino_merge_action)

        stack_to_slices_action = QAction("Stack to Slices", self)
        stack_to_slices_action.triggered.connect(self.show_stack_to_slices)
        tools_menu.addAction(stack_to_slices_action)

        image_patcher_action = QAction("Image Patcher", self)
        image_patcher_action.triggered.connect(self.show_image_patcher)
        tools_menu.addAction(image_patcher_action)

        image_augmenter_action = QAction("Image Augmenter", self)
        image_augmenter_action.triggered.connect(self.show_image_augmenter)
        tools_menu.addAction(image_augmenter_action)

        slice_registration_action = QAction("Slice Registration", self)
        slice_registration_action.triggered.connect(self.show_slice_registration)
        tools_menu.addAction(slice_registration_action)

        stack_interpolator_action = QAction("Stack Interpolator", self)
        stack_interpolator_action.triggered.connect(self.show_stack_interpolator)
        tools_menu.addAction(stack_interpolator_action)

        dicom_converter_action = QAction("DICOM Converter", self)
        dicom_converter_action.triggered.connect(self.show_dicom_converter)
        tools_menu.addAction(dicom_converter_action)

        tools_menu.addSeparator()

        unload_models_action = QAction("Unload AI Models (Free GPU Memory)", self)
        unload_models_action.triggered.connect(self.unload_ai_models)
        tools_menu.addAction(unload_models_action)

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")

        help_action = QAction("&Show Help", self)
        help_action.setShortcut(QKeySequence.StandardKey.HelpContents)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

    def change_font_size(self, size):
        self.current_font_size = size
        self.apply_theme_and_font()

    def unload_ai_models(self):
        """Drop cached SAM/DINO model objects to free GPU/CPU memory.

        Useful on constrained GPUs (e.g. 8 GB) where SAM 2 base + DINO
        base together exhaust VRAM. After unload, the next inference
        call will re-load the model from disk (~1-3 s).
        """
        self.sam_utils.unload()
        self.dino_utils.unload()
        # Reset the dropdowns to a neutral state so the user knows they
        # need to re-pick the model.
        self.sam_model_selector.setCurrentIndex(0)
        if hasattr(self, "dino_model_selector"):
            self.dino_model_selector.setCurrentIndex(0)
            self.dino_model_loaded = False
            self.lbl_dino_status.setText("No DINO model loaded")
            self.btn_detect_single.setEnabled(False)
            self.btn_detect_batch.setEnabled(False)
        QMessageBox.information(
            self,
            "Models Unloaded",
            "SAM and DINO models have been unloaded from memory.\n\n"
            "Note: PyTorch keeps a per-process CUDA context that survives "
            "this unload (typically a few hundred MB visible in Task Manager / "
            "nvidia-smi). To fully reclaim GPU memory, restart the app.\n\n"
            "Re-select a SAM/DINO model to use AI tools again.",
        )

    def setup_sidebar(self):
        self.sidebar = QWidget()
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.layout.addWidget(self.sidebar, 1)

        # Helper function to create section headers
        def create_section_header(text):
            label = QLabel(text)
            label.setProperty("class", "section-header")
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            return label

        # Import functionality
        self.import_button = QPushButton("Import Annotations with Images")
        self.import_button.clicked.connect(self.import_annotations)
        self.sidebar_layout.addWidget(self.import_button)

        self.import_format_selector = QComboBox()
        self.import_format_selector.addItem("COCO JSON")
        self.import_format_selector.addItem("YOLO (v4 and earlier)")  # Modified name
        self.import_format_selector.addItem("YOLO (v5+)")  # New format

        self.sidebar_layout.addWidget(self.import_format_selector)

        # Add spacing
        self.sidebar_layout.addSpacing(20)

        self.add_images_button = QPushButton("Add New Images")
        self.add_images_button.clicked.connect(self.add_images)
        self.sidebar_layout.addWidget(self.add_images_button)

        self.add_class_button = QPushButton("Add Classes")
        self.add_class_button.clicked.connect(lambda: self.add_class())
        self.sidebar_layout.addWidget(self.add_class_button)

        # Class list (without the "Classes" header)
        self.class_list = QListWidget()
        self.class_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self.show_class_context_menu)
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.sidebar_layout.addWidget(self.class_list)

        # Annotation section
        self.sidebar_layout.addWidget(create_section_header("Annotation"))
        annotation_widget = QWidget()
        annotation_layout = QVBoxLayout(annotation_widget)

        # Manual tools subsection
        manual_widget = QWidget()
        manual_layout = QVBoxLayout(manual_widget)

        button_layout_top = QHBoxLayout()
        self.polygon_button = QPushButton("Polygon")
        self.polygon_button.setCheckable(True)
        self.rectangle_button = QPushButton("Rectangle")
        self.rectangle_button.setCheckable(True)
        button_layout_top.addWidget(self.polygon_button)
        button_layout_top.addWidget(self.rectangle_button)

        button_layout_bottom = QHBoxLayout()
        self.paint_brush_button = QPushButton("Paint Brush")
        self.paint_brush_button.setCheckable(True)
        self.eraser_button = QPushButton("Eraser")
        self.eraser_button.setCheckable(True)
        button_layout_bottom.addWidget(self.paint_brush_button)
        button_layout_bottom.addWidget(self.eraser_button)

        manual_layout.addLayout(button_layout_top)
        manual_layout.addLayout(button_layout_bottom)

        annotation_layout.addWidget(manual_widget)

        # SAM-Assisted tools subsection
        sam_widget = QWidget()
        sam_layout = QVBoxLayout(sam_widget)

        # --- Replace the old SAM-Assisted button block with this: ---
        sam_buttons_layout = QHBoxLayout()

        self.sam_box_button = QPushButton("SAM-box")
        self.sam_box_button.setCheckable(True)
        self.sam_box_button.clicked.connect(self.toggle_sam_box)

        self.sam_points_button = QPushButton("SAM-points")
        self.sam_points_button.setCheckable(True)
        self.sam_points_button.clicked.connect(self.toggle_sam_points)

        sam_buttons_layout.addWidget(self.sam_box_button)
        sam_buttons_layout.addWidget(self.sam_points_button)
        sam_layout.addLayout(sam_buttons_layout)
        # ------------------------------------------------------------

        # Add SAM model selector
        self.sam_model_selector = QComboBox()
        self.sam_model_selector.addItem("Pick a SAM Model")
        self.sam_model_selector.addItems(list(self.sam_utils.sam_models.keys()))
        self.sam_model_selector.currentTextChanged.connect(self.change_sam_model)
        sam_layout.addWidget(self.sam_model_selector)

        annotation_layout.addWidget(sam_widget)

        # --- LLM-Assisted Detection (DINO) subsection ---
        dino_widget = QWidget()
        dino_layout = QVBoxLayout(dino_widget)

        self.dino_model_selector = QComboBox()
        self.dino_model_selector.addItem("Pick a DINO Model")
        self.dino_model_selector.addItem("grounding-dino-base")
        self.dino_model_selector.addItem("grounding-dino-tiny")
        self.dino_model_selector.addItem("Custom / fine-tuned (browse)")
        self.dino_model_selector.currentTextChanged.connect(self._on_dino_model_changed)
        dino_layout.addWidget(self.dino_model_selector)

        # Custom model browse row (hidden by default)
        self.dino_browse_row = QWidget()
        dino_browse_layout = QHBoxLayout(self.dino_browse_row)
        dino_browse_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_dino_custom = QLabel("No path set")
        self.lbl_dino_custom.setWordWrap(True)
        self.lbl_dino_custom.setStyleSheet("font-size:10px;color:#555;")
        btn_dino_browse = QPushButton("Browse")
        btn_dino_browse.setFixedWidth(60)
        btn_dino_browse.clicked.connect(self.browse_dino_model)
        dino_browse_layout.addWidget(self.lbl_dino_custom, 1)
        dino_browse_layout.addWidget(btn_dino_browse)
        self.dino_browse_row.setVisible(False)
        dino_layout.addWidget(self.dino_browse_row)

        self.lbl_dino_status = QLabel("No DINO model loaded")
        self.lbl_dino_status.setWordWrap(True)
        # No hardcoded background — let the active stylesheet (light or
        # dark) provide it via QLabel rules. Hardcoded #f5f5f5 used to
        # punch a bright rectangle into the dark sidebar.
        self.lbl_dino_status.setStyleSheet(
            "font-size:11px;padding:4px;border-radius:3px;"
            "border:1px solid palette(mid);")
        dino_layout.addWidget(self.lbl_dino_status)

        # Threshold table
        self.dino_class_table = ClassThresholdTable()
        self.dino_class_table.itemSelectionChanged.connect(self.on_dino_class_row_changed)
        dino_layout.addWidget(self.dino_class_table)

        # Phrase editor
        self.dino_phrase_panel = PhraseEditorPanel()
        dino_layout.addWidget(self.dino_phrase_panel)

        # Detect buttons
        det_btn_layout = QHBoxLayout()
        self.btn_detect_single = QPushButton("Detect Current Image")
        self.btn_detect_single.clicked.connect(self.run_dino_detection_single)
        self.btn_detect_single.setEnabled(False)
        det_btn_layout.addWidget(self.btn_detect_single)

        self.btn_detect_batch = QPushButton("Detect All Images")
        self.btn_detect_batch.clicked.connect(self.run_dino_detection_batch)
        self.btn_detect_batch.setEnabled(False)
        det_btn_layout.addWidget(self.btn_detect_batch)
        dino_layout.addLayout(det_btn_layout)

        # Batch mode
        self.dino_batch_mode = QComboBox()
        self.dino_batch_mode.addItem("Review before accepting")
        self.dino_batch_mode.addItem("Auto-accept all detections")
        dino_layout.addWidget(self.dino_batch_mode)

        annotation_layout.addWidget(dino_widget)
        # --- END DINO section ---

        # Add tool group
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(False)
        self.tool_group.addButton(self.polygon_button)
        self.tool_group.addButton(self.rectangle_button)
        self.tool_group.addButton(self.paint_brush_button)
        self.tool_group.addButton(self.eraser_button)
        self.tool_group.addButton(self.sam_box_button)
        self.tool_group.addButton(self.sam_points_button)

        self.polygon_button.clicked.connect(self.toggle_tool)
        self.rectangle_button.clicked.connect(self.toggle_tool)
        self.paint_brush_button.clicked.connect(self.toggle_tool)
        self.eraser_button.clicked.connect(self.toggle_tool)
        self.sam_magic_wand_button.clicked.connect(self.toggle_tool)

        # Annotations list subsection
        annotation_layout.addWidget(QLabel("Annotations"))
        self.annotation_list = QListWidget()
        self.annotation_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.annotation_list.itemSelectionChanged.connect(
            self.update_highlighted_annotations
        )
        annotation_layout.addWidget(self.annotation_list)

        # Create a horizontal layout for the sort buttons
        sort_button_layout = QHBoxLayout()

        self.sort_by_class_button = QPushButton("Sort by Class")
        self.sort_by_class_button.clicked.connect(self.sort_annotations_by_class)
        sort_button_layout.addWidget(self.sort_by_class_button)

        self.sort_by_area_button = QPushButton("Sort by Area")
        self.sort_by_area_button.clicked.connect(self.sort_annotations_by_area)
        sort_button_layout.addWidget(self.sort_by_area_button)

        # Add the sort button layout to the annotation layout
        annotation_layout.addLayout(sort_button_layout)

        # Delete and Merge annotation buttons
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_selected_annotations)
        self.merge_button = QPushButton("Merge")
        self.merge_button.clicked.connect(self.merge_annotations)
        self.change_class_button = QPushButton("Change Class")
        self.change_class_button.clicked.connect(self.change_annotation_class)

        # Create a horizontal layout for the other buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.merge_button)
        button_layout.addWidget(self.change_class_button)

        # Add the button layout to the annotation layout
        annotation_layout.addLayout(button_layout)

        # Add export format selector
        self.export_format_selector = QComboBox()
        self.export_format_selector.addItem("COCO JSON")
        self.export_format_selector.addItem("YOLO (v4 and earlier)")  # Modified name
        self.export_format_selector.addItem("YOLO (v5+)")  # New format
        self.export_format_selector.addItem("Labeled Images")
        self.export_format_selector.addItem("Semantic Labels")
        self.export_format_selector.addItem("Pascal VOC (BBox)")
        self.export_format_selector.addItem("Pascal VOC (BBox + Segmentation)")

        annotation_layout.addWidget(QLabel("Export Format:"))
        annotation_layout.addWidget(self.export_format_selector)

        self.export_button = QPushButton("Export Annotations")
        self.export_button.clicked.connect(self.export_annotations)
        annotation_layout.addWidget(self.export_button)

        # Add the annotation widget to the sidebar
        self.sidebar_layout.addWidget(annotation_widget)

    def toggle_sam_box(self):
        if self.sam_box_button.isChecked():
            self.sam_points_button.setChecked(False)
            self.image_label.current_tool = "sam_box"
            self.image_label.sam_box_active = True
            self.image_label.sam_points_active = False
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.image_label.current_tool = None
            self.image_label.sam_box_active = False
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
        self.update_ui_for_current_tool()

    def toggle_sam_points(self):
        if self.sam_points_button.isChecked():
            self.sam_box_button.setChecked(False)
            self.image_label.current_tool = "sam_points"
            self.image_label.sam_points_active = True
            self.image_label.sam_box_active = False
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.image_label.sam_positive_points = []
            self.image_label.sam_negative_points = []
        else:
            self.sam_inference_timer.stop()
            self.image_label.current_tool = None
            self.image_label.sam_points_active = False
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.image_label.sam_positive_points = []
            self.image_label.sam_negative_points = []
        self.update_ui_for_current_tool()

    def sort_annotations_by_class(self):
        current_name = self.current_slice or self.image_file_name
        if current_name not in self.all_annotations:
            QMessageBox.information(
                self,
                "No Annotations",
                "There are no annotations to sort for this image.",
            )
            return

        annotations = self.all_annotations[current_name]
        sorted_annotations = []
        for class_name in sorted(annotations.keys()):
            if not class_name.startswith("Temp-"):  # Skip temporary classes
                class_annotations = sorted(
                    annotations[class_name], key=lambda x: x.get("number", 0)
                )
                sorted_annotations.extend(class_annotations)

        self.update_annotation_list_with_sorted(sorted_annotations)

    def sort_annotations_by_area(self):
        current_name = self.current_slice or self.image_file_name
        if current_name not in self.all_annotations:
            QMessageBox.information(
                self,
                "No Annotations",
                "There are no annotations to sort for this image.",
            )
            return

        annotations = self.all_annotations[current_name]
        sorted_annotations = []
        for class_name in annotations.keys():
            if not class_name.startswith("Temp-"):  # Skip temporary classes
                class_annotations = sorted(
                    annotations[class_name],
                    key=lambda x: calculate_area(x),
                    reverse=True,
                )
                sorted_annotations.extend(class_annotations)

        self.update_annotation_list_with_sorted(sorted_annotations)

    def update_annotation_list_with_sorted(self, sorted_annotations):
        self.annotation_list.clear()
        for annotation in sorted_annotations:
            class_name = annotation["category_name"]
            if not class_name.startswith("Temp-"):  # Only add non-temporary annotations
                number = annotation.get("number", 0)
                area = calculate_area(annotation)
                item_text = f"{class_name} - {number:<3} Area: {area:.2f}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, annotation)
                color = self.image_label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
                item.setForeground(color)
                self.annotation_list.addItem(item)

        self.image_label.update()

    def change_sam_model(self, model_name):
        try:
            self.sam_utils.change_sam_model(model_name)
        except Exception as e:
            QMessageBox.critical(
                self,
                "SAM Model Error",
                f"Failed to load SAM model '{model_name}':\n\n{str(e)}\n\n"
                "Check that the model weights are downloadable and that torch "
                "is correctly installed for your platform / GPU."
            )
            self.sam_model_selector.setCurrentIndex(0)
            return

        self.current_sam_model = self.sam_utils.current_sam_model

        if model_name != "Pick a SAM Model":
            # Enable the SAM Magic Wand button
            self.sam_magic_wand_button.setEnabled(True)

            # Activate the SAM Magic Wand tool
            self.sam_magic_wand_button.setChecked(True)
            self.activate_sam_magic_wand()

            print(f"Changed SAM model to: {model_name}")
        else:
            # Disable and deactivate the SAM Magic Wand button
            self.sam_magic_wand_button.setEnabled(False)
            self.sam_magic_wand_button.setChecked(False)
            self.deactivate_sam_magic_wand()
            print("SAM model unset")

    # --- DINO / LLM-Assisted Detection Methods ---

    def _resolve_dino_model_path(self, model_name: str) -> str | None:
        """Return the canonical local path for a preset DINO model, or None if unknown."""
        from .dino_utils import GDINO_MODEL_PATHS
        # GDINO_MODEL_PATHS now returns absolute paths from models_base_dir().
        return GDINO_MODEL_PATHS.get(model_name)

    def _on_dino_model_changed(self, text):
        """Selection → ready state. Downloads happen lazily on first Detect."""
        self.dino_browse_row.setVisible(text == "Custom / fine-tuned (browse)")

        if text == "Pick a DINO Model":
            self.dino_model_loaded = False
            self.lbl_dino_status.setText("No DINO model loaded")
            self.btn_detect_single.setEnabled(False)
            self.btn_detect_batch.setEnabled(False)
            return

        if text == "Custom / fine-tuned (browse)":
            if self.dino_custom_model_path and os.path.exists(self.dino_custom_model_path):
                self.dino_model_loaded = True
                self.lbl_dino_status.setText(
                    f"Ready: {os.path.basename(self.dino_custom_model_path)}"
                )
                self.btn_detect_single.setEnabled(True)
                self.btn_detect_batch.setEnabled(True)
            else:
                self.dino_model_loaded = False
                self.lbl_dino_status.setText("Browse for a custom model folder")
                self.btn_detect_single.setEnabled(False)
                self.btn_detect_batch.setEnabled(False)
            return

        # Standard preset (grounding-dino-base/tiny)
        self.dino_model_loaded = True
        self.btn_detect_single.setEnabled(True)
        self.btn_detect_batch.setEnabled(True)
        model_path = self._resolve_dino_model_path(text)
        if model_path and os.path.exists(model_path):
            self.lbl_dino_status.setText(f"Ready: {text}")
        else:
            self.lbl_dino_status.setText(f"{text} — will download on first detection")

    def _ensure_dino_model_downloaded(self, model_name: str) -> bool:
        """If the preset model isn't on disk yet, download it. Returns success."""
        if model_name in ("Pick a DINO Model", "Custom / fine-tuned (browse)"):
            return True  # Custom path is validated elsewhere; no download for it.
        model_path = self._resolve_dino_model_path(model_name)
        if model_path and os.path.exists(model_path):
            return True

        # huggingface_hub is the only way to fetch the weights. Surface the
        # actionable install hint if it's missing rather than the generic
        # "Could not download" message.
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                f"Cannot download {model_name}: the huggingface_hub package "
                "is not installed.\n\nRun:\n    pip install huggingface_hub",
            )
            return False

        self.lbl_dino_status.setText(f"Downloading {model_name}...")
        QApplication.processEvents()
        try:
            downloaded = self.dino_utils.download_model(model_name)
        except Exception as e:
            QMessageBox.critical(self, "Download Failed", f"{model_name}:\n{e}")
            return False
        if not downloaded:
            QMessageBox.critical(
                self, "Download Failed",
                f"Could not download {model_name} from Hugging Face Hub.",
            )
            return False
        return True

    def browse_dino_model(self):
        path = QFileDialog.getExistingDirectory(self, "Select DINO Model Folder")
        if path:
            self.dino_custom_model_path = path
            self.lbl_dino_custom.setText(os.path.basename(path))
            # Refresh ready state now that a path is set.
            self._on_dino_model_changed(self.dino_model_selector.currentText())

    def on_dino_class_row_changed(self):
        name = self.dino_class_table.selected_class_name()
        self.dino_phrase_panel.set_active_class(name)

    def _build_dino_class_configs(self) -> list[dict]:
        """Build class_configs from threshold table + phrase panel."""
        configs = []
        for cfg in self.dino_class_table.get_class_configs():
            phrases = self.dino_phrase_panel.get_phrases_for(cfg["name"])
            configs.append({
                "name": cfg["name"],
                "phrases": phrases,
                "box_thr": cfg["box_thr"],
                "txt_thr": cfg["txt_thr"],
                "nms_thr": cfg["nms_thr"],
            })
        return configs

    def run_dino_detection_single(self):
        if not self.dino_model_loaded:
            QMessageBox.warning(self, "No DINO Model",
                                "Please pick a DINO model first.")
            return
        if not self.sam_utils.current_sam_model:
            QMessageBox.warning(
                self, "No SAM Model",
                "DINO produces bounding boxes; SAM is needed to convert them "
                "into segmentation masks. Please pick a SAM model first.",
            )
            return
        if not self.current_image or self.current_image.isNull():
            QMessageBox.warning(self, "No Image",
                                "Please load an image first.")
            return

        model_name = self.dino_model_selector.currentText()
        class_configs = self._build_dino_class_configs()
        if not class_configs:
            QMessageBox.warning(self, "No Classes",
                                "Please add at least one class with phrases.")
            return

        self.btn_detect_single.setEnabled(False)
        self.btn_detect_batch.setEnabled(False)

        if not self._ensure_dino_model_downloaded(model_name):
            self.btn_detect_single.setEnabled(True)
            self.btn_detect_batch.setEnabled(True)
            return

        self.lbl_dino_status.setText("Detecting...")
        QApplication.processEvents()

        print(f"[DINO] detect_single: model={model_name!r} class_configs={class_configs}")
        try:
            results = self.dino_utils.detect(
                self.current_image, class_configs,
                model_name=model_name,
                custom_model_path=self.dino_custom_model_path,
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "DINO Error", str(e))
            self.btn_detect_single.setEnabled(True)
            self.btn_detect_batch.setEnabled(True)
            self.lbl_dino_status.setText("Detection failed.")
            return

        self.btn_detect_single.setEnabled(True)
        self.btn_detect_batch.setEnabled(True)

        if results is None:
            print("[DINO] detect_single: results=None (model resolution failure)")
            self.lbl_dino_status.setText("No detections.")
            return

        print(f"[DINO] detect_single: got {len(results)} result(s)")
        if results:
            for i, r in enumerate(results[:3]):
                print(f"[DINO]   result[{i}] class={r['class_name']!r} score={r['score']:.3f} bbox={r['bbox']}")

        if not results:
            self.lbl_dino_status.setText("No detections found.")
            return

        self.lbl_dino_status.setText(f"{len(results)} detection(s). Running SAM...")
        QApplication.processEvents()

        # Batch SAM segmentation. Wrap in try/except for the same reason
        # as the DINO call above — sam_utils raises on model load
        # failure / CUDA OOM / re-entry now, instead of returning None.
        bboxes = [r["bbox"] for r in results]
        print(f"[SAM] batch call: {len(bboxes)} bbox(es), first 3 = {bboxes[:3]}")
        try:
            sam_results = self.sam_utils.apply_sam_predictions_batch(
                self.current_image, bboxes
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "SAM Error", str(e))
            self.lbl_dino_status.setText("SAM segmentation failed.")
            return

        if sam_results is None:
            print("[SAM] batch returned None (no SAM model loaded)")
            QMessageBox.warning(self, "SAM Error",
                                "Failed to segment detections with SAM.")
            self.lbl_dino_status.setText("SAM segmentation failed.")
            return

        n_errors = sum(1 for s in sam_results if "error" in s)
        n_ok = sum(1 for s in sam_results if "segmentation" in s)
        print(f"[SAM] batch returned {len(sam_results)} result(s): {n_ok} ok, {n_errors} error(s)")

        # Honor the batch-mode dropdown for the single-image case too:
        # "Auto-accept" means commit straight to annotations without
        # showing the temp-review overlay. The dropdown name is "batch"
        # historically but it controls both paths.
        image_name = self.current_slice or self.image_file_name
        auto_accept = (
            self.dino_batch_mode.currentText() == "Auto-accept all detections"
        )
        if auto_accept:
            self._commit_dino_results(image_name, results, sam_results)
            n_committed = sum(1 for s in sam_results if "error" not in s)
            self.image_label.temp_annotations = []
            self.image_label.update()
            self.update_annotation_list()
            # Refresh slice list so the freshly-annotated slice picks
            # up the highlight color; review-mode's accept_dino_results
            # already does this, the auto-accept path didn't.
            self.update_slice_list_colors()
            self.auto_save()
            self.lbl_dino_status.setText(
                f"Loaded: {model_name}  |  {n_committed} mask(s) auto-accepted"
            )
            print(f"[DINO] auto-accept: committed {n_committed} mask(s) to {image_name}")
            return

        # Review mode — build temp annotations and let user accept/reject
        temp_annotations = []
        for r, s in zip(results, sam_results):
            if "error" in s:
                print(f"[SAM]   failed for {r['class_name']}: {s['error']}")
                continue
            temp_annotations.append({
                "segmentation": s["segmentation"],
                "category_name": r["class_name"],
                "score": r["score"],
                "source": "dino",
                "temp": True,
            })

        self.image_label.temp_annotations = temp_annotations
        # Defer setFocus until after the click event chain settles —
        # synchronous setFocus often loses to whatever widget is still
        # processing the original click.
        QTimer.singleShot(0, self.image_label.setFocus)
        self.image_label.update()
        self.lbl_dino_status.setText(
            f"Loaded: {model_name}  |  {len(temp_annotations)} mask(s) ready"
        )
        print(f"[DINO] detection complete: {len(results)} boxes, {len(temp_annotations)} masks attached to canvas")

    def run_dino_detection_batch(self):
        if not self.dino_model_loaded:
            QMessageBox.warning(self, "No DINO Model",
                                "Please pick a DINO model first.")
            return
        if not self.sam_utils.current_sam_model:
            QMessageBox.warning(
                self, "No SAM Model",
                "DINO produces bounding boxes; SAM is needed to convert them "
                "into segmentation masks. Please pick a SAM model first.",
            )
            return
        if not self.all_images:
            QMessageBox.warning(self, "No Images",
                                "Please load images first.")
            return

        model_name = self.dino_model_selector.currentText()
        class_configs = self._build_dino_class_configs()
        if not class_configs:
            QMessageBox.warning(self, "No Classes",
                                "Please add at least one class with phrases.")
            return

        if not self._ensure_dino_model_downloaded(model_name):
            return

        auto_accept = self.dino_batch_mode.currentText() == "Auto-accept all detections"

        # Build a flat list of (display_name, qimage) work items covering
        # both regular images (loaded from disk) and multi-dim image
        # slices (already QImages in memory). Slices live in
        # self.image_slices[base_name], indexed by their slice_name
        # (e.g. "stack_T1_Z1_C1"). The earlier implementation only
        # iterated self.all_images and skipped multi-slice entries with
        # a console warning, leaving slice-based projects unable to use
        # Detect All.
        work_items = self._collect_dino_batch_work_items()
        if not work_items:
            QMessageBox.information(
                self, "Detect All Images",
                "No images or slices available to process."
            )
            return
        total = len(work_items)

        progress = QProgressDialog("Running LLM Detection...", "Cancel", 0, total, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        for idx, (image_name, qimage) in enumerate(work_items):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            QApplication.processEvents()

            try:
                results = self.dino_utils.detect(
                    qimage, class_configs,
                    model_name=model_name,
                    custom_model_path=self.dino_custom_model_path,
                )
            except Exception as e:
                print(f"  DINO failed for {image_name}: {e}")
                continue

            if not results:
                continue

            bboxes = [r["bbox"] for r in results]
            try:
                sam_results = self.sam_utils.apply_sam_predictions_batch(qimage, bboxes)
            except Exception as e:
                print(f"  SAM failed for {image_name}: {e}")
                continue
            if sam_results is None:
                continue

            if auto_accept:
                self._commit_dino_results(image_name, results, sam_results)
            else:
                # Store for later review
                self._store_dino_batch_results(image_name, results, sam_results)

        progress.setValue(total)
        progress.close()

        if auto_accept:
            QMessageBox.information(
                self, "Batch Detection Complete",
                "Detections have been saved to annotations."
            )
            self.update_annotation_list()
            # Multi-dim stacks commonly auto-accept across dozens of
            # slices; the slice list must show which ones gained
            # annotations or the user can't tell what happened.
            self.update_slice_list_colors()
            self.auto_save()
        else:
            self._show_dino_batch_review()

    def _collect_dino_batch_work_items(self):
        """Return a flat ``[(name, QImage), …]`` list for batch DINO.

        Regular images are loaded from disk via PIL → QImage. Multi-dim
        images contribute one entry per slice from ``self.image_slices``;
        slices that haven't been materialised yet (the parent image was
        never opened in this session) are skipped with a console log.
        """
        from PIL import Image as PILImage
        items = []
        for img_info in self.all_images:
            file_name = img_info["file_name"]
            if img_info.get("is_multi_slice", False):
                base_name = os.path.splitext(file_name)[0]
                slices = self.image_slices.get(base_name, [])
                if not slices:
                    print(f"  Skipping multi-slice image '{file_name}': "
                          "no slices loaded (open the image first to "
                          "materialise its slices).")
                    continue
                for slice_name, qimage in slices:
                    items.append((slice_name, qimage))
            else:
                image_path = self.image_paths.get(file_name)
                if not image_path or not os.path.exists(image_path):
                    print(f"  Skipping '{file_name}': missing image path.")
                    continue
                try:
                    pil_img = PILImage.open(image_path).convert("RGB")
                    qimage = QImage(
                        pil_img.tobytes(),
                        pil_img.width,
                        pil_img.height,
                        pil_img.width * 3,
                        QImage.Format.Format_RGB888,
                    )
                    items.append((file_name, qimage))
                except Exception as e:
                    print(f"  Skipping '{file_name}': failed to load ({e}).")
        print(f"[DINO] batch work items: {len(items)} total")
        return items

    def _commit_dino_results(self, image_name, dino_results, sam_results):
        """Commit DINO+SAM results to annotations for a single image.

        If image_name is the currently-displayed image, route through
        image_label.annotations so the canvas reflects the change and the
        next save_current_annotations() doesn't overwrite the additions.
        Otherwise write directly to the project-level cache.
        """
        current_image = self.current_slice or self.image_file_name
        is_current = image_name == current_image

        if is_current:
            target = self.image_label.annotations
        else:
            if image_name not in self.all_annotations:
                self.all_annotations[image_name] = {}
            target = self.all_annotations[image_name]

        for r, s in zip(dino_results, sam_results):
            if "error" in s:
                continue
            class_name = r["class_name"]
            # DINO only returns labels that came from class_configs (which the
            # parent built from the class table), so this should never trigger.
            # Skip with a warning rather than auto-creating a class mid-batch
            # (which would fan out auto_save() per new class).
            if class_name not in self.class_mapping:
                print(f"  Skipping DINO result for unknown class '{class_name}'")
                continue
            existing = target.get(class_name, [])
            number = max((a.get("number", 0) for a in existing), default=0) + 1
            ann = {
                "segmentation": s["segmentation"],
                "category_id": self.class_mapping[class_name],
                "category_name": class_name,
                "score": r["score"],
                "source": "dino",
                "number": number,
            }
            target.setdefault(class_name, []).append(ann)

        if is_current:
            # Sync image_label.annotations -> all_annotations[current] for save.
            self.save_current_annotations()
            self.image_label.update()

    def _store_dino_batch_results(self, image_name, dino_results, sam_results):
        """Store results for batch review mode."""
        valid = []
        for r, s in zip(dino_results, sam_results):
            if "error" not in s:
                valid.append({
                    "segmentation": s["segmentation"],
                    "category_name": r["class_name"],
                    "score": r["score"],
                    "source": "dino",
                    "temp": True,
                })
        self.dino_batch_results[image_name] = valid

    def _show_dino_batch_review(self):
        """Navigate to first image with batch results for review.

        If the next entry refers to an image/slice that's no longer in
        the project (e.g. the source was removed between detection and
        review), pop the orphan and try the next entry so the user
        doesn't get stuck with un-reviewable results.
        """
        if not self.dino_batch_results:
            QMessageBox.information(self, "Batch Detection",
                                    "No detections found in any image.")
            return
        # Drain orphans up front. Navigate to the entry: it may be a
        # regular image (key in image_list) or a slice (key in some
        # image_slices[base_name]). _navigate_to_image_or_slice handles
        # both. After the switch, switch_image / switch_slice's tail
        # call to _refresh_dino_temp_for_current copies
        # dino_batch_results[first] into image_label.temp_annotations
        # and defers setFocus on the canvas — nothing to repeat here.
        while self.dino_batch_results:
            first = next(iter(self.dino_batch_results))
            if self._navigate_to_image_or_slice(first):
                return
            print(f"[DINO] dropping orphan batch result for {first!r} "
                  "(no matching image or slice in project)")
            self.dino_batch_results.pop(first, None)
        # Drained all entries without a single navigable target.
        QMessageBox.warning(
            self, "Batch Detection",
            "Detections were produced but none of them map to an image "
            "or slice still in the project. Results discarded.",
        )

    def _navigate_to_image_or_slice(self, name: str) -> bool:
        """Switch the UI to a regular image or a slice by name.

        Returns True if a match was found and the switch was issued.
        Used by batch-review navigation, which mixes regular image
        names and slice names in ``dino_batch_results``.
        """
        # Regular image — match in image_list directly
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item and item.text() == name:
                self.image_list.setCurrentRow(i)
                self.switch_image(item)
                return True
        # Slice — find which multi-dim image contains it, switch to
        # that parent image first, then activate the specific slice
        # via slice_list.
        for base_name, slices in self.image_slices.items():
            if not any(s_name == name for s_name, _ in slices):
                continue
            # Find the parent file in image_list. The file_name in the
            # list includes the extension (e.g. "stack.tif") while
            # base_name is the stem ("stack"), so match by stripping
            # the extension and comparing for equality.
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if not item:
                    continue
                file_name = item.text()
                if os.path.splitext(file_name)[0] == base_name:
                    self.image_list.setCurrentRow(i)
                    self.switch_image(item)
                    # switch_image populates slice_list. Now find the slice.
                    for s_i in range(self.slice_list.count()):
                        s_item = self.slice_list.item(s_i)
                        if s_item and s_item.text() == name:
                            self.slice_list.setCurrentRow(s_i)
                            self.switch_slice(s_item)
                            return True
                    break
            return False
        return False

    def _refresh_dino_temp_for_current(self):
        """Sync ``image_label.temp_annotations`` to whatever the
        currently-displayed image/slice has stored in
        ``dino_batch_results``. Called from switch_slice / switch_image.

        Why this exists: ``temp_annotations`` is a single field on
        ``ImageLabel``, not a per-image cache. Without this sync, masks
        from the previously-viewed image bleed onto every slice the
        user navigates to. During a batch review the user expects each
        image to show its own pending detections; outside batch review,
        switching simply discards the pending overlay.
        """
        new_image = self.current_slice or self.image_file_name
        pending = self.dino_batch_results.get(new_image, []) if new_image else []
        if pending:
            # Re-stamp the "temp" flag in case it was stripped by a
            # previous accept path; this list also feeds the paintEvent
            # which expects dicts with "segmentation" + "category_name".
            self.image_label.temp_annotations = list(pending)
            self.lbl_dino_status.setText(
                f"Review: {new_image}  ({len(pending)} detection(s))"
            )
            QTimer.singleShot(0, self.image_label.setFocus)
        else:
            if self.image_label.temp_annotations:
                print("[DINO] temp annotations cleared on switch "
                      f"(no pending batch results for {new_image!r})")
            self.image_label.temp_annotations = []
        self.image_label.update()

    def accept_dino_results(self):
        """Accept current temp_annotations (called from keyPressEvent)."""
        if not self.image_label.temp_annotations:
            return
        image_name = self.current_slice or self.image_file_name

        for ann in self.image_label.temp_annotations:
            class_name = ann["category_name"]
            # DINO only returns labels from class_configs (built from the
            # class table), so unknown classes should never reach this point.
            # Skip with a warning rather than auto-creating mid-accept.
            if class_name not in self.class_mapping:
                print(f"  Skipping DINO result for unknown class '{class_name}'")
                continue
            new_ann = {
                "segmentation": ann["segmentation"],
                "category_id": self.class_mapping[class_name],
                "category_name": class_name,
                "score": ann.get("score", 0.0),
                "source": "dino",
            }
            # Append to the live image_label dict; save_current_annotations()
            # below syncs it into self.all_annotations. add_annotation_to_list
            # assigns the per-class "number" used for display.
            self.image_label.annotations.setdefault(class_name, []).append(new_ann)
            self.add_annotation_to_list(new_ann)

        self.image_label.temp_annotations = []
        # Clear batch results if reviewing
        self.dino_batch_results.pop(image_name, None)
        if self.dino_batch_results:
            self._show_dino_batch_review()
        self.save_current_annotations()
        self.update_slice_list_colors()
        self.image_label.update()
        self.lbl_dino_status.setText("Results accepted.")
        print("DINO results accepted.")

    def reject_dino_results(self):
        """Discard current temp_annotations."""
        self.image_label.temp_annotations = []
        image_name = self.current_slice or self.image_file_name
        self.dino_batch_results.pop(image_name, None)
        if self.dino_batch_results:
            self._show_dino_batch_review()
        self.image_label.update()
        self.lbl_dino_status.setText("Results discarded.")
        print("DINO results discarded.")

    # --- END DINO Methods ---

    def setup_font_size_selector(self):
        font_size_label = QLabel("Font Size:")
        self.font_size_selector = QComboBox()
        self.font_size_selector.addItems(["Small", "Medium", "Large"])
        self.font_size_selector.setCurrentText("Medium")
        self.font_size_selector.currentTextChanged.connect(self.on_font_size_changed)

        self.sidebar_layout.addWidget(font_size_label)
        self.sidebar_layout.addWidget(self.font_size_selector)

    def on_font_size_changed(self, size):
        self.current_font_size = size
        self.apply_theme_and_font()

    def apply_theme_and_font(self):
        font_size = self.font_sizes[self.current_font_size]
        if self.dark_mode:
            style = soft_dark_stylesheet
        else:
            style = default_stylesheet

        # Combine the theme stylesheet with font size
        combined_style = f"{style}\nQWidget {{ font-size: {font_size}pt; }}"
        self.setStyleSheet(combined_style)

        # Apply font size to all widgets
        for widget in self.findChildren(QWidget):
            font = widget.font()
            font.setPointSize(font_size)
            widget.setFont(font)

        self.image_label.setFont(QFont("Arial", font_size))
        self.update()

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme_and_font()

        # Update slice list colors
        self.update_slice_list_colors()

        # Update other UI elements if necessary
        self.update_class_list()
        self.update_annotation_list()

        # Force a repaint of the main window
        self.repaint()

    def apply_stylesheet(self):
        if self.dark_mode:
            self.setStyleSheet(soft_dark_stylesheet)
        else:
            self.setStyleSheet(default_stylesheet)

    def update_ui_colors(self):
        # Update colors for elements that need to retain their functionality
        self.update_annotation_list_colors()
        self.update_slice_list_colors()
        self.image_label.update()

    def setup_image_area(self):
        """Set up the main image area."""
        self.image_widget = QWidget()
        self.image_layout = QVBoxLayout(self.image_widget)
        self.layout.addWidget(self.image_widget, 3)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Use the already initialized image_label
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        self.image_layout.addWidget(self.scroll_area)

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(500)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(50)
        self.zoom_slider.valueChanged.connect(self.zoom_image)
        self.image_layout.addWidget(self.zoom_slider)
        self.image_info_label = QLabel()
        self.image_layout.addWidget(self.image_info_label)

    def setup_image_list(self):
        """Set up the image list area."""
        self.image_list_widget = QWidget()
        self.image_list_layout = QVBoxLayout(self.image_list_widget)
        self.layout.addWidget(self.image_list_widget, 1)

        self.image_list_label = QLabel("Images:")
        self.image_list_layout.addWidget(self.image_list_label)

        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.switch_image)
        self.image_list.currentRowChanged.connect(
            lambda row: self.switch_image(self.image_list.currentItem())
        )
        self.image_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image_list.customContextMenuRequested.connect(self.show_image_context_menu)
        self.image_list_layout.addWidget(self.image_list)

        self.clear_all_button = QPushButton("Clear All Images and Annotations")
        self.clear_all_button.clicked.connect(self.clear_all)
        self.image_list_layout.addWidget(self.clear_all_button)

    ##########    ### Tools  ########## I love useful image processing tools :)
    def open_dataset_splitter(self):
        self.dataset_splitter = DatasetSplitterTool(self)
        self.dataset_splitter.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.dataset_splitter.show_centered(self)

    def show_annotation_statistics(self):
        if not self.all_annotations:
            QMessageBox.warning(
                self, "No Annotations", "There are no annotations to analyze."
            )
            return
        try:
            self.annotation_stats_dialog = show_annotation_statistics(
                self, self.all_annotations
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while showing annotation statistics: {str(e)}",
            )

    def show_coco_json_combiner(self):
        self.coco_json_combiner_dialog = show_coco_json_combiner(self)

    def show_dino_merge_dialog(self):
        show_dino_merge_dialog(self)

    def show_stack_to_slices(self):
        self.stack_to_slices_dialog = show_stack_to_slices(self)

    def show_image_patcher(self):
        self.image_patcher_dialog = show_image_patcher(self)

    def show_image_augmenter(self):
        self.image_augmenter_dialog = show_image_augmenter(self)

    def show_slice_registration(self):
        self.slice_registration_dialog = SliceRegistrationTool(self)
        self.slice_registration_dialog.show_centered(self)

    def show_stack_interpolator(self):
        self.stack_interpolator_dialog = StackInterpolator(self)
        self.stack_interpolator_dialog.show_centered(self)

    def show_dicom_converter(self):
        self.dicom_converter_dialog = DicomConverter(self)
        self.dicom_converter_dialog.show_centered(self)

    ###################################################################

    # update the show_help method:
    def show_help(self):
        self.help_window = HelpWindow(
            dark_mode=self.dark_mode, font_size=self.font_sizes[self.current_font_size]
        )
        self.help_window.show_centered(self)

    def add_images(self):
        if not self.image_label.check_unsaved_changes():
            return
        file_names, _ = QFileDialog.getOpenFileNames(
            self, "Add Images", "", "Image Files (*.png *.jpg *.bmp *.tif *.tiff *.czi)"
        )
        if file_names:
            self.add_images_to_list(file_names)

    def clear_all(self, new_project=False, show_messages=True):
        if not new_project and show_messages:
            reply = self.show_question(
                "Clear All",
                "Are you sure you want to clear all images and annotations? This action cannot be undone.",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Clear images
        self.image_list.clear()
        self.image_paths.clear()
        self.all_images.clear()
        self.current_image = None
        self.image_file_name = ""

        # Clear the image display
        self.image_label.clear()
        self.image_label.setPixmap(QPixmap())  # Set an empty pixmap
        self.image_label.original_pixmap = None
        self.image_label.scaled_pixmap = None

        # Clear annotations
        self.all_annotations.clear()
        self.annotation_list.clear()
        self.image_label.annotations.clear()
        self.image_label.highlighted_annotations.clear()

        # Clear current class
        self.current_class = None

        # Reset class-related data
        self.class_list.clear()
        self.image_label.class_colors.clear()
        self.class_mapping.clear()

        # Reset DINO state
        self.dino_class_table.clear_classes()
        self.dino_phrase_panel.clear()
        self.dino_model_loaded = False
        self.dino_custom_model_path = None
        self.dino_model_selector.setCurrentIndex(0)
        self.lbl_dino_status.setText("No DINO model loaded")
        self.btn_detect_single.setEnabled(False)
        self.btn_detect_batch.setEnabled(False)

        # Clear slices
        self.image_slices.clear()
        self.slices = []
        self.slice_list.clear()
        self.current_slice = None
        self.current_stack = None

        # Reset zoom
        self.image_label.zoom_factor = 1.0
        self.zoom_slider.setValue(100)

        # Reset tools
        self.image_label.current_tool = None
        self.polygon_button.setChecked(False)
        self.rectangle_button.setChecked(False)
        self.sam_magic_wand_button.setChecked(False)
        self.sam_magic_wand_button.setEnabled(False)  # Disable the SAM-Assisted button
        self.image_label.sam_magic_wand_active = False  # Deactivate SAM magic wand

        # Reset SAM-related attributes
        self.image_label.sam_bbox = None
        self.image_label.drawing_sam_bbox = False
        self.image_label.temp_sam_prediction = None

        self.image_label.setCursor(Qt.CursorShape.ArrowCursor)  # Reset cursor to default
        self.sam_model_selector.setCurrentIndex(0)  # Reset to "Pick a SAM Model"
        self.current_sam_model = None  # Reset the current SAM model

        # Reset project-related attributes
        if not new_project:
            if hasattr(self, "current_project_file"):
                del self.current_project_file
            if hasattr(self, "current_project_dir"):
                del self.current_project_dir

        # Update UI
        self.image_label.update()
        self.update_image_info()

        # Force a repaint of the main window
        self.repaint()
        self.update_window_title()

    def show_warning(self, title, message):
        QMessageBox.warning(self, title, message)

    def show_info(self, title, message):
        QMessageBox.information(self, title, message)

    def update_image_info(self, additional_info=None):
        if self.current_image:
            width = self.current_image.width()
            height = self.current_image.height()
            info = f"Image: {width}x{height}"
            if additional_info:
                info += f", {additional_info}"
            self.image_info_label.setText(info)
        else:
            self.image_info_label.setText("No image loaded")

    def show_question(self, title, message):
        return QMessageBox.question(
            self, title, message, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No
        )

    def show_image_context_menu(self, position):
        menu = QMenu()
        current_item = self.image_list.itemAt(position)
        if current_item:
            file_name = current_item.text()
            delete_action = menu.addAction("Remove Image")

            if not self.is_multi_dimensional(file_name):
                predict_action = menu.addAction("Predict using YOLO")

            if self.is_multi_dimensional(file_name):
                redefine_dimensions_action = menu.addAction("Redefine Dimensions")

            action = menu.exec(self.image_list.mapToGlobal(position))

            if action == delete_action:
                self.remove_image()
            elif not self.is_multi_dimensional(file_name) and action == predict_action:
                self.predict_single_image(file_name)
            elif (
                self.is_multi_dimensional(file_name)
                and action == redefine_dimensions_action
            ):
                self.redefine_dimensions(file_name)

    def is_multi_dimensional(self, file_name):
        return file_name.lower().endswith((".tif", ".tiff", ".czi"))

    def predict_single_image(self, file_name):
        if self.is_multi_dimensional(file_name):
            return  # Do nothing for multi-dimensional images

        if not self.yolo_trainer or not self.yolo_trainer.model:
            QMessageBox.warning(
                self,
                "No Model",
                "Please load a YOLO model first from the YOLO > Prediction Settings > Load Model menu.",
            )
            return

        # Deactivate SAM tool before prediction
        self.deactivate_sam_magic_wand()

        image_path = self.image_paths[file_name]
        try:
            results = self.yolo_trainer.predict(image_path)
            self.process_yolo_results(results, file_name)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Prediction Error",
                f"An error occurred during prediction: {str(e)}\n\n"
                "This might be due to a mismatch between the model and the YAML file classes. "
                "Please check that the YAML file corresponds to the loaded model.",
            )

    def redefine_dimensions(self, file_name):
        file_path = self.image_paths.get(file_name)
        if not file_path or not file_path.lower().endswith((".tif", ".tiff", ".czi")):
            return  # Exit the method if it's not a TIFF or CZI file

        reply = QMessageBox.warning(
            self,
            "Redefine Dimensions",
            "Redefining dimensions will cause all associated annotations to be lost. "
            "Do you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Remove existing annotations for this file
            base_name = os.path.splitext(file_name)[0]

            print(f"Removing annotations for image: {base_name}")
            # print(f"Current annotations: {list(self.all_annotations.keys())}")

            # Create a list of keys to remove, using a more specific matching condition
            keys_to_remove = [
                key
                for key in self.all_annotations.keys()
                if key == base_name
                or (
                    key.startswith(f"{base_name}_")
                    and not key.startswith(f"{base_name}_8bit")
                )
            ]

            print(f"Keys to remove: {keys_to_remove}")

            # Remove the annotations
            for key in keys_to_remove:
                del self.all_annotations[key]

            # print(f"Annotations after removal: {list(self.all_annotations.keys())}")

            # Remove existing slices
            if base_name in self.image_slices:
                del self.image_slices[base_name]

            # Clear current image if it's the one being redefined
            if self.image_file_name == file_name:
                self.current_image = None
                self.image_label.clear()

            # Reload the image with new dimension dialog
            if file_path.lower().endswith((".tif", ".tiff")):
                self.load_tiff(file_path, force_dimension_dialog=True)
            elif file_path.lower().endswith(".czi"):
                self.load_czi(file_path, force_dimension_dialog=True)

            # Update UI
            self.update_slice_list()
            self.update_annotation_list()
            self.image_label.update()

            # print(f"Final annotations: {list(self.all_annotations.keys())}")

            QMessageBox.information(
                self,
                "Dimensions Redefined",
                "The dimensions have been redefined and the image reloaded. "
                "All previous annotations for this image have been removed.",
            )

    def remove_image(self):
        current_item = self.image_list.currentItem()
        if current_item:
            file_name = current_item.text()

            # Remove from all data structures
            self.image_list.takeItem(self.image_list.row(current_item))
            self.image_paths.pop(file_name, None)
            self.all_images = [
                img for img in self.all_images if img["file_name"] != file_name
            ]

            # Remove annotations
            self.all_annotations.pop(file_name, None)

            # Handle multi-dimensional images
            base_name = os.path.splitext(file_name)[0]
            if base_name in self.image_slices:
                # Remove slices
                for slice_name, _ in self.image_slices[base_name]:
                    self.all_annotations.pop(slice_name, None)
                del self.image_slices[base_name]

                # Clear slice list
                self.slice_list.clear()

            # Clear current image and slice if it was the removed image
            if self.image_file_name == file_name:
                self.current_image = None
                self.image_file_name = ""
                self.current_slice = None
                self.image_label.clear()
                self.annotation_list.clear()

            # Switch to another image if available
            if self.image_list.count() > 0:
                next_item = self.image_list.item(0)
                self.image_list.setCurrentItem(next_item)
                self.switch_image(next_item)
            else:
                # No images left
                self.current_image = None
                self.image_file_name = ""
                self.current_slice = None
                self.image_label.clear()
                self.annotation_list.clear()
                self.slice_list.clear()

            # Update UI
            self.update_ui()
            self.auto_save()  # Auto-save after removing an image

    def load_annotations(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Annotations", "", "JSON Files (*.json)"
        )
        if file_name:
            with open(file_name, "r") as f:
                self.loaded_json = json.load(f)

            # Load categories
            self.class_list.clear()
            self.image_label.class_colors.clear()
            self.class_mapping.clear()
            for category in self.loaded_json["categories"]:
                class_name = category["name"]
                self.class_mapping[class_name] = category["id"]

                # Assign a color if not already assigned
                if class_name not in self.image_label.class_colors:
                    color = QColor(
                        Qt.GlobalColor(len(self.image_label.class_colors) % 16 + 7)
                    )
                    self.image_label.class_colors[class_name] = color

                # Add item to class list with color indicator
                item = QListWidgetItem(class_name)
                self.update_class_item_color(
                    item, self.image_label.class_colors[class_name]
                )
                self.class_list.addItem(item)

            # Create a mapping of image IDs to file names
            image_id_to_filename = {
                img["id"]: img["file_name"] for img in self.loaded_json["images"]
            }

            # Load image information
            json_images = {img["file_name"]: img for img in self.loaded_json["images"]}

            # Update existing images and add new ones from JSON
            updated_all_images = []
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                file_name = item.text()
                if file_name in json_images:
                    updated_image = self.all_images[i].copy()
                    updated_image.update(json_images[file_name])
                    updated_all_images.append(updated_image)
                    del json_images[file_name]
                else:
                    updated_all_images.append(self.all_images[i])

            # Add remaining images from JSON
            for img in json_images.values():
                updated_all_images.append(img)
                self.image_list.addItem(img["file_name"])

            self.all_images = updated_all_images

            # Load annotations
            self.all_annotations.clear()
            for annotation in self.loaded_json["annotations"]:
                image_id = annotation["image_id"]
                file_name = image_id_to_filename.get(image_id)
                if file_name:
                    if file_name not in self.all_annotations:
                        self.all_annotations[file_name] = {}

                    category = next(
                        (
                            cat
                            for cat in self.loaded_json["categories"]
                            if cat["id"] == annotation["category_id"]
                        ),
                        None,
                    )
                    if category:
                        category_name = category["name"]
                        if category_name not in self.all_annotations[file_name]:
                            self.all_annotations[file_name][category_name] = []

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

                        # Add number field if it's missing
                        if "number" not in ann:
                            ann["number"] = (
                                len(self.all_annotations[file_name][category_name]) + 1
                            )

                        self.all_annotations[file_name][category_name].append(ann)

            # Check for missing images
            missing_images = [
                img["file_name"]
                for img in self.loaded_json["images"]
                if img["file_name"] not in self.image_paths
            ]
            if missing_images:
                self.show_warning(
                    "Missing Images",
                    "The following images are missing:\n" + "\n".join(missing_images),
                )

            # Reload the current image if it exists, otherwise load the first image
            if self.image_file_name and self.image_file_name in self.all_annotations:
                self.switch_image(
                    self.image_list.findItems(self.image_file_name, Qt.MatchFlag.MatchExactly)[0]
                )
            elif self.all_images:
                self.switch_image(self.image_list.item(0))

            self.image_label.highlighted_annotations = []  # Clear existing highlights
            self.update_annotation_list()  # This will repopulate the annotation list
            self.image_label.update()  # Force a redraw of the image label

    def clear_highlighted_annotation(self):
        self.image_label.highlighted_annotation = None
        self.image_label.update()

    def update_highlighted_annotations(self):
        selected_items = self.annotation_list.selectedItems()
        self.image_label.highlighted_annotations = [
            item.data(Qt.ItemDataRole.UserRole) for item in selected_items
        ]
        self.image_label.update()  # Force a redraw of the image label

        # Enable/disable merge and change class buttons based on selection
        self.merge_button.setEnabled(len(selected_items) >= 2)
        self.change_class_button.setEnabled(len(selected_items) > 0)

    def renumber_annotations(self):
        current_name = self.current_slice or self.image_file_name
        if current_name in self.all_annotations:
            for class_name, annotations in self.all_annotations[current_name].items():
                for i, ann in enumerate(annotations, start=1):
                    ann["number"] = i
        self.update_annotation_list()

    def delete_selected_annotations(self):
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "No Selection", "Please select an annotation to delete."
            )
            return

        reply = QMessageBox.question(
            self,
            "Delete Annotations",
            f"Are you sure you want to delete {len(selected_items)} annotation(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Create a list of annotations to remove
            annotations_to_remove = []
            for item in selected_items:
                annotation = item.data(Qt.ItemDataRole.UserRole)
                annotations_to_remove.append((annotation["category_name"], annotation))

            # Remove annotations from image_label.annotations
            for category_name, annotation in annotations_to_remove:
                if category_name in self.image_label.annotations:
                    if annotation in self.image_label.annotations[category_name]:
                        self.image_label.annotations[category_name].remove(annotation)

            # Update all_annotations
            current_name = self.current_slice or self.image_file_name
            self.all_annotations[current_name] = self.image_label.annotations

            # Sort and update the annotation list based on the current sorting method
            if self.current_sort_method == "area":
                self.sort_annotations_by_area()
            else:
                self.sort_annotations_by_class()

            self.image_label.highlighted_annotations.clear()
            self.image_label.update()

            # Update slice list colors
            self.update_slice_list_colors()

            QMessageBox.information(
                self,
                "Annotations Deleted",
                f"{len(selected_items)} annotation(s) have been deleted.",
            )
            self.auto_save()  # Auto-save after deleting annotations

    def merge_annotations(self):
        if self.image_label.editing_polygon is not None:
            QMessageBox.warning(
                self,
                "Edit Mode Active",
                "Please exit the annotation edit mode before merging annotations.",
            )
            return

        selected_items = self.annotation_list.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(
                self,
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
                self,
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

            connected = set([0])  # Start with the first polygon
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
                    return (
                        False  # If no new connections found, they're not all connected
                    )

                connected.update(newly_connected)
                to_check -= newly_connected

            return True  # All polygons are connected

        if not are_all_polygons_connected(polygons):
            QMessageBox.warning(
                self,
                "Disconnected Polygons",
                "Not all selected annotations are connected. Please select only connected annotations to merge.",
            )
            return

        try:
            merged_polygon = unary_union(polygons)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Merge Error",
                f"Unable to merge the selected annotations due to an error: {str(e)}",
            )
            return

        new_annotation = {
            "segmentation": [],
            "category_id": self.class_mapping[class_name],
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

        # Ask user about keeping original annotations
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Merge Annotations")
        msg_box.setText("Do you want to keep the original annotations?")
        msg_box.setIcon(QMessageBox.Icon.Question)

        keep_button = msg_box.addButton("Keep", QMessageBox.ButtonRole.YesRole)
        delete_button = msg_box.addButton("Delete", QMessageBox.ButtonRole.NoRole)
        cancel_button = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)

        msg_box.setDefaultButton(cancel_button)
        msg_box.setEscapeButton(cancel_button)

        msg_box.exec()

        if msg_box.clickedButton() == cancel_button:
            return

        if msg_box.clickedButton() == delete_button:
            for annotation in original_annotations:
                if annotation in self.image_label.annotations[class_name]:
                    self.image_label.annotations[class_name].remove(annotation)

        self.image_label.annotations.setdefault(class_name, []).append(new_annotation)

        current_name = self.current_slice or self.image_file_name
        self.all_annotations[current_name] = self.image_label.annotations

        self.renumber_annotations()
        self.update_annotation_list()
        self.save_current_annotations()
        self.update_slice_list_colors()
        self.image_label.update()

        QMessageBox.information(
            self, "Merge Complete", "Annotations have been merged successfully."
        )
        self.auto_save()  # Auto-save after merging annotations

    def delete_selected_image(self):
        current_item = self.image_list.currentItem()
        if current_item:
            file_name = current_item.text()
            reply = QMessageBox.question(
                self,
                "Delete Image",
                f"Are you sure you want to delete the image '{file_name}'?\n\n"
                "This will remove the image and all its associated annotations.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Remove from all data structures
                self.image_list.takeItem(self.image_list.row(current_item))
                self.image_paths.pop(file_name, None)
                self.all_images = [
                    img for img in self.all_images if img["file_name"] != file_name
                ]

                # Remove annotations
                self.all_annotations.pop(file_name, None)

                # Handle multi-dimensional images
                base_name = os.path.splitext(file_name)[0]
                if base_name in self.image_slices:
                    # Remove slices
                    for slice_name, _ in self.image_slices[base_name]:
                        self.all_annotations.pop(slice_name, None)
                    del self.image_slices[base_name]

                    # Clear slice list
                    self.slice_list.clear()

                # Clear current image and slice if it was the removed image
                if self.image_file_name == file_name:
                    self.current_image = None
                    self.image_file_name = ""
                    self.current_slice = None
                    self.image_label.clear()
                    self.annotation_list.clear()

                # Switch to another image if available
                if self.image_list.count() > 0:
                    next_item = self.image_list.item(0)
                    self.image_list.setCurrentItem(next_item)
                    self.switch_image(next_item)
                else:
                    # No images left
                    self.current_image = None
                    self.image_file_name = ""
                    self.current_slice = None
                    self.image_label.clear()
                    self.annotation_list.clear()
                    self.slice_list.clear()

                # Update UI
                self.update_ui()

                QMessageBox.information(
                    self, "Image Deleted", f"The image '{file_name}' has been deleted."
                )

    def display_image(self):
        if self.current_image:
            if isinstance(self.current_image, QImage):
                pixmap = QPixmap.fromImage(self.current_image)
            elif isinstance(self.current_image, QPixmap):
                pixmap = self.current_image
            else:
                print(f"Unexpected image type: {type(self.current_image)}")
                return

            if not pixmap.isNull():
                self.image_label.setPixmap(pixmap)
                self.image_label.adjustSize()
            else:
                print("Error: Null pixmap")
        else:
            self.image_label.clear()
            print("No current image to display")

    def update_ui(self):
        self.update_image_list()
        self.update_slice_list()
        self.update_class_list()
        self.update_annotation_list()
        self.image_label.update()
        self.update_image_info()

    def add_class(self, class_name=None, color=None):
        if not self.image_label.check_unsaved_changes():
            return

        if class_name is None:
            while True:
                class_name, ok = QInputDialog.getText(
                    self, "Add Class", "Enter class name:"
                )
                if not ok:
                    print("Class addition cancelled")
                    return
                if not class_name.strip():
                    QMessageBox.warning(
                        self,
                        "Invalid Input",
                        "Please enter a class name or press Cancel.",
                    )
                    continue
                if class_name in self.class_mapping:
                    QMessageBox.warning(
                        self,
                        "Duplicate Class",
                        f"The class '{class_name}' already exists. Please choose a different name.",
                    )
                    continue
                break
        else:
            # For programmatic addition (e.g., from YOLO predictions)
            if class_name in self.class_mapping:
                print(f"Class '{class_name}' already exists. Skipping addition.")
                return

        if not isinstance(class_name, str):
            print(
                f"Warning: class_name is not a string. Converting {class_name} to string."
            )
            class_name = str(class_name)

        if color is None:
            color = QColor(Qt.GlobalColor(len(self.image_label.class_colors) % 16 + 7))
        elif isinstance(color, str):
            color = QColor(color)

        print(f"Adding class: {class_name}, color: {color.name()}")

        self.image_label.class_colors[class_name] = color
        self.class_mapping[class_name] = len(self.class_mapping) + 1

        try:
            item = QListWidgetItem(class_name)

            # Create a color indicator
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))

            # Set visibility state
            item.setData(Qt.ItemDataRole.UserRole, True)

            # Set checkbox
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)

            self.class_list.addItem(item)

            self.class_list.setCurrentItem(item)
            self.current_class = class_name
            print(f"Class added successfully: {class_name}")

            # Sync DINO phrase/threshold state. Select the newly added
            # row so the phrase editor below the table reveals itself —
            # it hides by default and only becomes visible when a row is
            # selected (set_active_class). Skip the row-select during
            # project load: classes are added in a loop and we don't want
            # N row-selection signals firing during bulk restoration; the
            # caller will select an appropriate row after load completes.
            row_added = self.dino_class_table.add_class(class_name)
            self.dino_phrase_panel.on_class_added(class_name)
            if row_added and not self.is_loading_project:
                self.dino_class_table.selectRow(self.dino_class_table.rowCount() - 1)

            if not self.is_loading_project:
                self.auto_save()
        except Exception as e:
            print(f"Error adding class: {e}")
            traceback.print_exc()

    def update_class_item_color(self, item, color):
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        item.setIcon(QIcon(pixmap))

    def update_class_list(self):
        self.class_list.clear()
        for class_name, color in self.image_label.class_colors.items():
            item = QListWidgetItem(class_name)

            # Create a color indicator
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))

            # Store the visibility state
            item.setData(
                Qt.ItemDataRole.UserRole, self.image_label.class_visibility.get(class_name, True)
            )

            # Set checkbox
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if item.data(Qt.ItemDataRole.UserRole) else Qt.CheckState.Unchecked)

            self.class_list.addItem(item)

        # Re-select the current class if it exists
        if self.current_class:
            items = self.class_list.findItems(self.current_class, Qt.MatchFlag.MatchExactly)
            if items:
                self.class_list.setCurrentItem(items[0])
        elif self.class_list.count() > 0:
            # If no class is selected, select the first one
            self.class_list.setCurrentItem(self.class_list.item(0))

        print(f"Updated class list with {self.class_list.count()} items")

    def update_class_selection(self):
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.text() == self.current_class:
                item.setSelected(True)
            else:
                item.setSelected(False)

    def toggle_class_visibility(self, item):
        class_name = item.text()
        is_visible = item.checkState() == Qt.CheckState.Checked
        self.image_label.set_class_visibility(class_name, is_visible)
        item.setData(Qt.ItemDataRole.UserRole, is_visible)
        self.image_label.update()

    def change_annotation_class(self):
        selected_items = self.annotation_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select one or more annotations to change class.",
            )
            return

        class_dialog = QDialog(self)
        class_dialog.setWindowTitle("Change Class")
        layout = QVBoxLayout(class_dialog)

        class_combo = QComboBox()
        for class_name in self.class_mapping.keys():
            class_combo.addItem(class_name)
        layout.addWidget(class_combo)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(class_dialog.accept)
        button_box.rejected.connect(class_dialog.reject)
        layout.addWidget(button_box)

        if class_dialog.exec() == QDialog.DialogCode.Accepted:
            new_class = class_combo.currentText()
            current_name = self.current_slice or self.image_file_name

            # Get the current maximum number for the new class
            max_number = max(
                [
                    ann.get("number", 0)
                    for ann in self.image_label.annotations.get(new_class, [])
                ]
                + [0]
            )

            for item in selected_items:
                annotation = item.data(Qt.ItemDataRole.UserRole)
                old_class = annotation["category_name"]

                # Remove from old class
                self.image_label.annotations[old_class].remove(annotation)
                if not self.image_label.annotations[old_class]:
                    del self.image_label.annotations[old_class]

                # Add to new class with updated number
                annotation["category_name"] = new_class
                annotation["category_id"] = self.class_mapping[new_class]
                max_number += 1
                annotation["number"] = max_number
                if new_class not in self.image_label.annotations:
                    self.image_label.annotations[new_class] = []
                self.image_label.annotations[new_class].append(annotation)

            # Update all_annotations
            self.all_annotations[current_name] = self.image_label.annotations

            # Renumber all annotations for consistency
            self.renumber_annotations()

            self.update_annotation_list()
            self.image_label.update()
            self.save_current_annotations()
            self.update_slice_list_colors()
            self.auto_save()

            QMessageBox.information(
                self,
                "Class Changed",
                f"Selected annotations have been changed to class '{new_class}'.",
            )

    def toggle_tool(self):
        if not self.image_label.check_unsaved_changes():
            return

        sender = self.sender()
        if sender is None:
            sender = self.sam_magic_wand_button

        if not self.current_class:
            QMessageBox.warning(
                self,
                "No Class Selected",
                "Please select a class before using annotation tools.",
            )
            sender.setChecked(False)
            return

        if self.current_class and self.current_class.startswith("Temp-"):
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Cannot use annotation tools with temporary classes.",
            )
            sender.setChecked(False)
            return

        other_buttons = [btn for btn in self.tool_group.buttons() if btn != sender]

        # Deactivate SAM if we're switching to a different tool
        if (
            sender != self.sam_magic_wand_button
            and self.image_label.sam_magic_wand_active
        ):
            self.deactivate_sam_magic_wand()

        if sender.isChecked():
            # Uncheck all other buttons
            for btn in other_buttons:
                btn.setChecked(False)

            # Set the current tool based on the checked button
            if sender == self.polygon_button:
                self.image_label.current_tool = "polygon"
            elif sender == self.rectangle_button:
                self.image_label.current_tool = "rectangle"
            elif sender == self.sam_magic_wand_button:
                self.image_label.current_tool = "sam_magic_wand"
                self.activate_sam_magic_wand()
            elif sender == self.paint_brush_button:
                self.image_label.current_tool = "paint_brush"
                self.image_label.setFocus()  # Set focus on the image label
            elif sender == self.eraser_button:
                self.image_label.current_tool = "eraser"
                self.image_label.setFocus()  # Set focus on the image label
        else:
            self.image_label.current_tool = None
            if sender == self.sam_magic_wand_button:
                self.deactivate_sam_magic_wand()

        # Update UI based on the current tool
        self.update_ui_for_current_tool()

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if self.image_label.current_tool == "paint_brush":
                self.paint_brush_size = max(1, self.paint_brush_size + delta // 120)
                print(f"Paint brush size: {self.paint_brush_size}")
            elif self.image_label.current_tool == "eraser":
                self.eraser_size = max(1, self.eraser_size + delta // 120)
                print(f"Eraser size: {self.eraser_size}")
        else:
            super().wheelEvent(event)

    def update_ui_for_current_tool(self):
        # Disable finish_polygon_button if it still exists in your code
        if hasattr(self, "finish_polygon_button"):
            self.finish_polygon_button.setEnabled(
                self.image_label.current_tool in ["polygon", "rectangle"]
            )

        # Update button states
        self.polygon_button.setChecked(self.image_label.current_tool == "polygon")
        self.rectangle_button.setChecked(self.image_label.current_tool == "rectangle")
        self.sam_magic_wand_button.setChecked(
            self.image_label.current_tool == "sam_magic_wand"
        )

        # Enable/disable SAM button based on model availability
        self.sam_magic_wand_button.setEnabled(self.current_sam_model is not None)

        # Disable all tools if no class is selected
        tools_enabled = (
            self.current_class is not None
            and not self.current_class.startswith("Temp-")
        )
        for button in self.tool_group.buttons():
            button.setEnabled(tools_enabled)

        # Update cursor based on the current tool
        if (
            self.image_label.current_tool == "sam_magic_wand"
            and self.sam_magic_wand_button.isEnabled()
        ):
            self.image_label.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

    def on_class_selected(self, current=None, previous=None):
        if not self.image_label.check_unsaved_changes():
            return

        if current is None:
            current = self.class_list.currentItem()

        if current:
            self.current_class = current.text()
            print(f"Class selected: {self.current_class}")

            if self.current_class.startswith("Temp-"):
                self.disable_annotation_tools()
            else:
                self.enable_annotation_tools()
        else:
            self.current_class = None
            self.disable_annotation_tools()

    def disable_annotation_tools(self):
        for button in self.tool_group.buttons():
            button.setChecked(False)
            button.setEnabled(False)
        self.image_label.current_tool = None

    def enable_annotation_tools(self):
        for button in self.tool_group.buttons():
            button.setEnabled(True)

    def show_class_context_menu(self, position):
        menu = QMenu()
        rename_action = menu.addAction("Rename Class")
        change_color_action = menu.addAction("Change Color")
        delete_action = menu.addAction("Delete Class")

        item = self.class_list.itemAt(position)
        if item:
            action = menu.exec(self.class_list.mapToGlobal(position))

            if action == rename_action:
                self.rename_class(item)
            elif action == change_color_action:
                self.change_class_color(item)
            elif action == delete_action:
                self.delete_class(item)
        else:
            QMessageBox.warning(
                self, "No Selection", "Please select a class to perform actions."
            )

    def change_class_color(self, item):
        class_name = item.text()
        current_color = self.image_label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
        color = QColorDialog.getColor(
            current_color, self, f"Select Color for {class_name}"
        )

        if color.isValid():
            self.image_label.class_colors[class_name] = color

            # Update the color indicator
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))

            self.update_annotation_list_colors(class_name, color)
            self.image_label.update()
            self.auto_save()  # Auto-save after changing class color

    def rename_class(self, item):
        old_name = item.text()
        new_name, ok = QInputDialog.getText(
            self, "Rename Class", "Enter new class name:", text=old_name
        )
        if ok and new_name and new_name != old_name:
            # Update class mapping
            if old_name in self.class_mapping:
                old_id = self.class_mapping[old_name]
                self.class_mapping[new_name] = old_id
                del self.class_mapping[old_name]
            else:
                print(f"Warning: Class '{old_name}' not found in class_mapping")
                return

            # Update class colors
            if old_name in self.image_label.class_colors:
                self.image_label.class_colors[new_name] = (
                    self.image_label.class_colors.pop(old_name)
                )
            else:
                print(f"Warning: Class '{old_name}' not found in class_colors")
                return

            # Update annotations for all images and slices
            for image_name, image_annotations in self.all_annotations.items():
                if old_name in image_annotations:
                    image_annotations[new_name] = image_annotations.pop(old_name)
                    for annotation in image_annotations[new_name]:
                        annotation["category_name"] = new_name

            # Update current image annotations
            if old_name in self.image_label.annotations:
                self.image_label.annotations[new_name] = (
                    self.image_label.annotations.pop(old_name)
                )
                for annotation in self.image_label.annotations[new_name]:
                    annotation["category_name"] = new_name

            # Update current class if it's the renamed one
            if self.current_class == old_name:
                self.current_class = new_name

            # Update annotation list for all images and slices
            self.update_all_annotation_lists()

            # Update class list
            item.setText(new_name)

            # Update the image label
            self.image_label.update()
            self.auto_save()  # Auto-save after renaming a class

            print(f"Class renamed from '{old_name}' to '{new_name}'")

    def delete_class(self, item=None):
        if item is None:
            item = self.class_list.currentItem()

        if item is None:
            QMessageBox.warning(
                self, "No Selection", "Please select a class to delete."
            )
            return

        class_name = item.text()

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Delete Class",
            f"Are you sure you want to delete the class '{class_name}'?\n\n"
            "This will remove all annotations associated with this class.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Proceed with deletion
            # Remove class color
            self.image_label.class_colors.pop(class_name, None)

            # Remove class from mapping
            self.class_mapping.pop(class_name, None)

            # Remove annotations for this class from all images
            for image_annotations in self.all_annotations.values():
                image_annotations.pop(class_name, None)

            # Remove annotations for this class from current image
            self.image_label.annotations.pop(class_name, None)

            # Sync DINO state
            self.dino_class_table.remove_class(class_name)
            self.dino_phrase_panel.on_class_removed(class_name)

            # Update annotation list
            self.update_annotation_list()

            # Remove class from list
            row = self.class_list.row(item)
            self.class_list.takeItem(row)

            # Update current_class
            if self.current_class == class_name:
                self.current_class = None
                if self.class_list.count() > 0:
                    self.class_list.setCurrentRow(0)
                    self.on_class_selected(self.class_list.item(0))
                else:
                    self.disable_annotation_tools()

            self.image_label.update()

            # Inform the user
            QMessageBox.information(
                self, "Class Deleted", f"The class '{class_name}' has been deleted."
            )
            self.auto_save()  # Auto-save after deleting a class
        else:
            # User cancelled the operation
            QMessageBox.information(
                self, "Deletion Cancelled", "The class deletion was cancelled."
            )

    def finish_polygon(self):
        if (
            self.image_label.current_tool == "polygon"
            and len(self.image_label.current_annotation) > 2
        ):
            if self.current_class is None:
                QMessageBox.warning(
                    self,
                    "No Class Selected",
                    "Please select a class before finishing the annotation.",
                )
                return

            # Create a polygon from the current annotation
            polygon = Polygon(self.image_label.current_annotation)

            # Define the image boundary as a rectangle
            image_boundary = Polygon(
                [
                    (0, 0),
                    (self.current_image.width(), 0),
                    (self.current_image.width(), self.current_image.height()),
                    (0, self.current_image.height()),
                ]
            )

            # Intersect the polygon with the image boundary
            clipped_polygon = polygon.intersection(image_boundary)

            if clipped_polygon.is_empty:
                QMessageBox.warning(
                    self,
                    "Invalid Annotation",
                    "The annotation is completely outside the image boundaries.",
                )
                self.image_label.clear_current_annotation()
                self.image_label.update()
                return

            # Convert the clipped polygon to a segmentation format
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
                    self, "Invalid Annotation", "The annotation could not be processed."
                )
                return

            new_annotation = {
                "segmentation": segmentation,
                "category_id": self.class_mapping[self.current_class],
                "category_name": self.current_class,
            }
            self.image_label.annotations.setdefault(self.current_class, []).append(
                new_annotation
            )
            self.add_annotation_to_list(new_annotation)
            self.image_label.clear_current_annotation()
            self.image_label.drawing_polygon = False  # Reset the drawing_polygon flag
            self.image_label.reset_annotation_state()
            self.image_label.update()

            # Save the current annotations
            self.save_current_annotations()

            # Update the slice list colors
            self.update_slice_list_colors()
            self.auto_save()  # Auto-save after adding a polygon annotation

    def highlight_annotation(self, item):
        self.image_label.highlighted_annotation = item.data(Qt.ItemDataRole.UserRole)
        self.image_label.update()

    def delete_annotation(self):
        current_item = self.annotation_list.currentItem()
        if current_item:
            annotation = current_item.data(Qt.ItemDataRole.UserRole)
            category_name = annotation["category_name"]
            self.image_label.annotations[category_name].remove(annotation)
            self.annotation_list.takeItem(self.annotation_list.row(current_item))
            self.image_label.highlighted_annotation = None
            self.image_label.update()

    def add_annotation_to_list(self, annotation):
        class_name = annotation["category_name"]
        color = self.image_label.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
        annotations = self.image_label.annotations.get(class_name, [])
        number = max([ann.get("number", 0) for ann in annotations] + [0]) + 1
        annotation["number"] = number
        area = calculate_area(annotation)
        item_text = f"{class_name} - {number:<3} Area: {area:.2f}"

        item = QListWidgetItem(item_text)
        item.setData(Qt.ItemDataRole.UserRole, annotation)
        item.setForeground(color)
        self.annotation_list.addItem(item)

        # Clear the current selection
        self.annotation_list.clearSelection()
        self.image_label.highlighted_annotations.clear()
        self.image_label.update()

    def zoom_in(self):
        new_zoom = min(self.image_label.zoom_factor + 0.1, 5.0)
        self.set_zoom(new_zoom)

    def zoom_out(self):
        new_zoom = max(self.image_label.zoom_factor - 0.1, 0.1)
        self.set_zoom(new_zoom)

    def set_zoom(self, zoom_factor):
        self.image_label.set_zoom(zoom_factor)
        self.zoom_slider.setValue(int(zoom_factor * 100))
        self.image_label.update()

    def zoom_image(self):
        zoom_factor = self.zoom_slider.value() / 100
        self.set_zoom(zoom_factor)

    def disable_tools(self):
        self.polygon_button.setEnabled(False)
        self.rectangle_button.setEnabled(False)
        # self.finish_polygon_button.setEnabled(False)

    def enable_tools(self):
        self.polygon_button.setEnabled(True)
        self.rectangle_button.setEnabled(True)

    def finish_rectangle(self):
        if self.image_label.current_rectangle:
            x1, y1, x2, y2 = self.image_label.current_rectangle

            # Create a rectangle polygon from the annotation
            rectangle = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            # Define the image boundary as a rectangle
            image_boundary = Polygon(
                [
                    (0, 0),
                    (self.current_image.width(), 0),
                    (self.current_image.width(), self.current_image.height()),
                    (0, self.current_image.height()),
                ]
            )

            # Intersect the rectangle with the image boundary
            clipped_rectangle = rectangle.intersection(image_boundary)

            if clipped_rectangle.is_empty:
                QMessageBox.warning(
                    self,
                    "Invalid Annotation",
                    "The annotation is completely outside the image boundaries.",
                )
                self.image_label.current_rectangle = None
                self.image_label.update()
                return

            # Convert the clipped rectangle to a segmentation format
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
                    self, "Invalid Annotation", "The annotation could not be processed."
                )
                return

            new_annotation = {
                "segmentation": segmentation,
                "category_id": self.class_mapping[self.current_class],
                "category_name": self.current_class,
            }
            self.image_label.annotations.setdefault(self.current_class, []).append(
                new_annotation
            )
            self.add_annotation_to_list(new_annotation)
            self.image_label.start_point = None
            self.image_label.end_point = None
            self.image_label.current_rectangle = None
            self.image_label.update()

            # Save the current annotations
            self.save_current_annotations()

            # Update the slice list colors
            self.update_slice_list_colors()
            self.auto_save()

    def enter_edit_mode(self, annotation):
        self.editing_mode = True
        self.disable_tools()

        QMessageBox.information(
            self,
            "Edit Mode",
            "You are now in edit mode. Click and drag points to move them, Shift+Click to delete points, or click on edges to add new points.",
        )

    def exit_edit_mode(self):
        self.editing_mode = False
        self.enable_tools()

        self.image_label.editing_polygon = None
        self.image_label.editing_point_index = None
        self.image_label.hover_point_index = None
        self.update_annotation_list()
        self.image_label.update()

    def highlight_annotation_in_list(self, annotation):
        for i in range(self.annotation_list.count()):
            item = self.annotation_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == annotation:
                self.annotation_list.setCurrentItem(item)
                break

    def select_annotation_in_list(self, annotation):
        for i in range(self.annotation_list.count()):
            item = self.annotation_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == annotation:
                self.annotation_list.setCurrentItem(item)
                break

    ################################################################

    def setup_yolo_menu(self):
        yolo_menu = self.menuBar().addMenu("&YOLO (beta)")

        # Training submenu
        training_submenu = yolo_menu.addMenu("Training")

        load_pretrained_action = QAction("Load Pre-trained Model", self)
        load_pretrained_action.triggered.connect(self.load_yolo_model)
        training_submenu.addAction(load_pretrained_action)

        prepare_data_action = QAction("Prepare YOLO Dataset", self)
        prepare_data_action.triggered.connect(self.prepare_yolo_dataset)
        training_submenu.addAction(prepare_data_action)

        load_yaml_action = QAction("Load Dataset YAML", self)
        load_yaml_action.triggered.connect(self.load_yolo_yaml)
        training_submenu.addAction(load_yaml_action)

        train_action = QAction("Train Model", self)
        train_action.triggered.connect(self.show_train_dialog)
        training_submenu.addAction(train_action)

        save_model_action = QAction("Save Model", self)
        save_model_action.triggered.connect(self.save_yolo_model)
        training_submenu.addAction(save_model_action)

        # Prediction Settings submenu
        prediction_submenu = yolo_menu.addMenu("Prediction Settings")

        load_model_action = QAction("Load Model", self)
        load_model_action.triggered.connect(self.load_prediction_model)
        prediction_submenu.addAction(load_model_action)

        set_threshold_action = QAction("Set Confidence Threshold", self)
        set_threshold_action.triggered.connect(self.set_confidence_threshold)
        prediction_submenu.addAction(set_threshold_action)

    def load_yolo_model(self):
        if not hasattr(self, "current_project_dir"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer:
            self.initialize_yolo_trainer()

        if self.yolo_trainer.load_model():
            QMessageBox.information(
                self, "Model Loaded", "YOLO model loaded successfully."
            )
        else:
            QMessageBox.warning(self, "Load Cancelled", "Model loading was cancelled.")

    def prepare_yolo_dataset(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer:
            self.initialize_yolo_trainer()

        try:
            yaml_path = self.yolo_trainer.prepare_dataset()
            QMessageBox.information(
                self,
                "Dataset Prepared",
                f"YOLO dataset prepared successfully. YAML file: {yaml_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while preparing the dataset: {str(e)}",
            )

    def load_yolo_yaml(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer:
            self.initialize_yolo_trainer()

        try:
            if self.yolo_trainer.load_yaml():
                QMessageBox.information(
                    self, "YAML Loaded", "Dataset YAML loaded successfully."
                )
            else:
                QMessageBox.warning(
                    self, "Load Cancelled", "YAML loading was cancelled."
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while loading the YAML file: {str(e)}",
            )

    def save_yolo_model(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer or not self.yolo_trainer.model:
            QMessageBox.warning(
                self, "No Model", "Please train or load a YOLO model first."
            )
            return

        try:
            if self.yolo_trainer.save_model():
                QMessageBox.information(
                    self, "Model Saved", "YOLO model saved successfully."
                )
            else:
                QMessageBox.warning(
                    self, "Save Cancelled", "Model saving was cancelled."
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while saving the model: {str(e)}"
            )

    def load_prediction_model(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer:
            self.initialize_yolo_trainer()

        dialog = LoadPredictionModelDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            model_path = dialog.model_path
            yaml_path = dialog.yaml_path
            if model_path and yaml_path:
                try:
                    result, message = self.yolo_trainer.load_prediction_model(
                        model_path, yaml_path
                    )
                    if result:
                        QMessageBox.information(
                            self,
                            "Model Loaded",
                            "YOLO model and YAML file loaded successfully for prediction.",
                        )
                        if message:
                            QMessageBox.warning(self, "Class Mismatch Warning", message)
                    else:
                        QMessageBox.critical(
                            self,
                            "Error Loading Model",
                            f"Could not load the model or YAML file: {message}",
                        )
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
            else:
                QMessageBox.warning(
                    self,
                    "Files Required",
                    "Both model and YAML files are required for prediction.",
                )

    def show_train_dialog(self):
        if not self.yolo_trainer:
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return
        if not self.yolo_trainer.model:
            QMessageBox.warning(
                self, "No Model", "Please load a pre-trained model first."
            )
            return
        if not self.yolo_trainer.yaml_path:
            QMessageBox.warning(
                self, "No Dataset", "Please prepare or load a dataset YAML first."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Train YOLO Model")
        layout = QVBoxLayout()

        epochs_label = QLabel("Number of Epochs:")
        epochs_input = QLineEdit("100")
        layout.addWidget(epochs_label)
        layout.addWidget(epochs_input)

        imgsz_label = QLabel("Image Size:")
        imgsz_input = QLineEdit("640")
        layout.addWidget(imgsz_label)
        layout.addWidget(imgsz_input)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            epochs = int(epochs_input.text())
            imgsz = int(imgsz_input.text())
            self.start_training(epochs, imgsz)

    def initialize_yolo_trainer(self):
        if hasattr(self, "current_project_dir"):
            self.yolo_trainer = YOLOTrainer(self.current_project_dir, self)
        else:
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )

    def start_training(self, epochs, imgsz):
        if not hasattr(self, "training_dialog"):
            self.training_dialog = TrainingInfoDialog(self)
        self.training_dialog.show()

        self.yolo_trainer.progress_signal.connect(self.training_dialog.update_info)
        self.yolo_trainer.set_progress_callback(self.training_dialog.update_info)
        self.training_dialog.stop_signal.connect(self.yolo_trainer.stop_training_signal)

        self.training_thread = TrainingThread(self.yolo_trainer, epochs, imgsz)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.start()

    def training_finished(self, results):
        self.training_dialog.stop_button.setEnabled(True)
        self.training_dialog.stop_button.setText("Stop Training")
        self.yolo_trainer.progress_signal.disconnect(self.training_dialog.update_info)
        self.training_dialog.stop_signal.disconnect(
            self.yolo_trainer.stop_training_signal
        )

        if isinstance(results, str):
            QMessageBox.critical(
                self, "Training Error", f"An error occurred during training: {results}"
            )
        else:
            QMessageBox.information(
                self, "Training Complete", "YOLO model training completed successfully."
            )

    def set_confidence_threshold(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        if not self.yolo_trainer:
            self.initialize_yolo_trainer()

        current_threshold = self.yolo_trainer.conf_threshold
        new_threshold, ok = QInputDialog.getDouble(
            self,
            "Set Confidence Threshold",
            "Enter confidence threshold (0-1):",
            current_threshold,
            0,
            1,
            2,
        )
        if ok:
            self.yolo_trainer.set_conf_threshold(new_threshold)
            QMessageBox.information(
                self,
                "Threshold Updated",
                f"Confidence threshold set to {new_threshold}",
            )

    def show_predict_dialog(self):
        if not self.yolo_trainer or not self.yolo_trainer.model:
            QMessageBox.warning(self, "No Model", "Please load a YOLO model first.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Predict with YOLO Model")
        layout = QVBoxLayout()

        image_list = QListWidget()
        for image_name in self.image_paths.keys():
            image_list.addItem(image_name)
        layout.addWidget(QLabel("Select images for prediction:"))
        layout.addWidget(image_list)

        conf_label = QLabel("Confidence Threshold:")
        conf_input = QDoubleSpinBox()
        conf_input.setRange(0, 1)
        conf_input.setSingleStep(0.01)
        conf_input.setValue(self.yolo_trainer.conf_threshold)
        layout.addWidget(conf_label)
        layout.addWidget(conf_input)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        predict_button = QPushButton("Predict")
        button_box.addButton(predict_button, QDialogButtonBox.ButtonRole.AcceptRole)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_images = [item.text() for item in image_list.selectedItems()]
            conf = conf_input.value()
            self.yolo_trainer.set_conf_threshold(conf)
            self.run_predictions(selected_images)

    def run_predictions(self, selected_images):
        for image_name in selected_images:
            image_path = self.image_paths[image_name]
            results = self.yolo_trainer.predict(image_path)
            self.process_yolo_results(results, image_name)

    def process_yolo_results(self, results, image_name):
        image_path = self.image_paths[image_name]
        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.warning(self, "Error", f"Failed to load image: {image_name}")
            return
        original_height, original_width = image.shape[:2]

        temp_annotations = {}

        try:
            results, input_size, original_size = (
                results  # Unpack the results, input size, and original size
            )
            input_height, input_width = input_size
            orig_height, orig_width = original_size

            scale_x = original_width / orig_width
            scale_y = original_height / orig_height

            for result in results:
                boxes = result.boxes
                masks = result.masks

                if masks is None:
                    print(f"No masks found for {image_name}")
                    continue

                for mask, box in zip(masks, boxes):
                    try:
                        class_id = int(box.cls)
                        class_name = self.yolo_trainer.class_names[class_id]
                        score = float(box.conf)

                        mask_array = mask.data.cpu().numpy()[0]
                        # Resize mask to original image size
                        mask_array = cv2.resize(mask_array, (orig_width, orig_height))
                        contours, _ = cv2.findContours(
                            (mask_array > 0.5).astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )

                        if contours:
                            epsilon = 0.005 * cv2.arcLength(contours[0], True)
                            approx = cv2.approxPolyDP(contours[0], epsilon, True)
                            polygon = approx.flatten().tolist()

                            # Scale the polygon coordinates
                            scaled_polygon = []
                            for i in range(0, len(polygon), 2):
                                x = polygon[i] * scale_x
                                y = polygon[i + 1] * scale_y
                                scaled_polygon.extend([x, y])

                            temp_class_name = f"Temp-{class_name}"
                            if temp_class_name not in temp_annotations:
                                temp_annotations[temp_class_name] = []

                            temp_annotation = {
                                "segmentation": scaled_polygon,
                                "category_name": temp_class_name,
                                "score": score,
                                "temp": True,
                            }
                            temp_annotations[temp_class_name].append(temp_annotation)
                    except IndexError:
                        QMessageBox.warning(
                            self,
                            "Class Mismatch",
                            "There is a mismatch between the model and the YAML file classes. "
                            "Please check that the YAML file corresponds to the loaded model.",
                        )
                        return

        except Exception as e:
            QMessageBox.warning(
                self,
                "Prediction Error",
                f"An error occurred during prediction: {str(e)}\n\n"
                "This might be due to a mismatch between the model and the YAML file classes. "
                "Please check that the YAML file corresponds to the loaded model.",
            )
            return

        self.add_temp_classes(temp_annotations)
        self.update_class_list()
        self.image_label.update()

        if temp_annotations:
            total_predictions = sum(len(anns) for anns in temp_annotations.values())
            QMessageBox.information(
                self,
                "Review Predictions",
                f"Found {total_predictions} predictions for {len(temp_annotations)} classes.\n"
                "Use class visibility checkboxes to review.\n"
                "Press Enter to accept or Esc to reject visible predictions.",
            )
        else:
            QMessageBox.information(
                self, "No Predictions", "No predictions were found for this image."
            )

        # Deactivate SAM tool
        self.deactivate_sam_magic_wand()

    def add_temp_classes(self, temp_annotations):
        for temp_class_name, annotations in temp_annotations.items():
            if temp_class_name not in self.image_label.class_colors:
                color = QColor(
                    Qt.GlobalColor(len(self.image_label.class_colors) % 16 + 7)
                )
                self.image_label.class_colors[temp_class_name] = color
            self.image_label.annotations[temp_class_name] = annotations

        self.update_class_list()

    def verify_current_class(self):
        if self.current_class is None or self.current_class not in self.class_mapping:
            if self.class_list.count() > 0:
                self.class_list.setCurrentRow(0)
                self.on_class_selected(self.class_list.item(0))
            else:
                self.current_class = None
                self.disable_annotation_tools()

    def accept_visible_temp_classes(self):
        visible_temp_classes = [
            item.text()
            for item in self.class_list.findItems("Temp-*", Qt.MatchFlag.MatchWildcard)
            if item.checkState() == Qt.CheckState.Checked
        ]

        for temp_class_name in visible_temp_classes:
            permanent_class_name = temp_class_name[5:]  # Remove "Temp-" prefix
            if permanent_class_name not in self.image_label.annotations:
                self.add_class(
                    permanent_class_name, self.image_label.class_colors[temp_class_name]
                )

            # Get the current maximum number for this class
            current_max = max(
                [
                    ann.get("number", 0)
                    for ann in self.image_label.annotations.get(
                        permanent_class_name, []
                    )
                ]
                + [0]
            )

            for annotation in self.image_label.annotations[temp_class_name]:
                current_max += 1
                annotation["category_name"] = permanent_class_name
                annotation["number"] = current_max
                self.image_label.annotations.setdefault(
                    permanent_class_name, []
                ).append(annotation)

            del self.image_label.annotations[temp_class_name]
            del self.image_label.class_colors[temp_class_name]

        self.update_class_list()
        current_name = self.current_slice or self.image_file_name
        self.all_annotations[current_name] = self.image_label.annotations
        self.update_annotation_list()
        self.image_label.update()
        self.save_current_annotations()

        # Select the first primary class
        self.select_first_primary_class()
        self.verify_current_class()

        QMessageBox.information(
            self,
            "Annotations Accepted",
            "Temporary annotations have been accepted and added to the permanent classes.",
        )

    def select_first_primary_class(self):
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if not item.text().startswith("Temp-"):
                self.class_list.setCurrentItem(item)
                self.on_class_selected(item)
                break

    def reject_visible_temp_classes(self):
        visible_temp_classes = [
            item.text()
            for item in self.class_list.findItems("Temp-*", Qt.MatchFlag.MatchWildcard)
            if item.checkState() == Qt.CheckState.Checked
        ]

        for temp_class_name in visible_temp_classes:
            if temp_class_name in self.image_label.annotations:
                del self.image_label.annotations[temp_class_name]
            if temp_class_name in self.image_label.class_colors:
                del self.image_label.class_colors[temp_class_name]

        self.update_class_list()
        self.image_label.update()

    def is_class_visible(self, class_name):
        items = self.class_list.findItems(class_name, Qt.MatchFlag.MatchExactly)
        if items:
            return items[0].checkState() == Qt.CheckState.Checked
        return False

    def check_temp_annotations(self):
        temp_classes = [
            class_name
            for class_name in self.image_label.annotations.keys()
            if class_name.startswith("Temp-")
        ]
        if temp_classes:
            reply = QMessageBox.question(
                self,
                "Temporary Annotations",
                "There are temporary annotations that will be discarded. Do you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                for temp_class in temp_classes:
                    del self.image_label.annotations[temp_class]
                    del self.image_label.class_colors[temp_class]
                self.update_class_list()
                self.update_annotation_list()
                return True
            return False
        return True

    def remove_all_temp_annotations(self):
        for image_name in list(self.all_annotations.keys()):
            for class_name in list(self.all_annotations[image_name].keys()):
                if class_name.startswith("Temp-"):
                    del self.all_annotations[image_name][class_name]
            if not self.all_annotations[image_name]:
                del self.all_annotations[image_name]

        for class_name in list(self.image_label.class_colors.keys()):
            if class_name.startswith("Temp-"):
                del self.image_label.class_colors[class_name]

        self.update_class_list()
        self.update_annotation_list()
        self.image_label.update()
