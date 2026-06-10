import copy
import json
import os
import traceback
import warnings

import shapely
from PyQt6.QtCore import Qt, QTimer
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

from .controllers import io_controller
from .controllers.dino_controller import DINOController, _DINOReviewEventFilter
from .controllers.image_controller import ImageController
from .controllers.project_controller import ProjectController
from .controllers.sam_controller import SAMController
from .controllers.yolo_controller import YOLOController
from .core import image_utils
from .ui import theme
from .dialogs.annotation_statistics import show_annotation_statistics
from .dialogs.coco_json_combiner import show_coco_json_combiner
from .dialogs.dino_phrase_editor import ClassThresholdTable, PhraseEditorPanel
from .inference.dino_utils import DINOUtils, GDINO_MODEL_PATHS
from .dialogs.dataset_splitter import DatasetSplitterTool
from .dialogs.dicom_converter import DicomConverter
from .dialogs.dino_merge_dialog import show_dino_merge_dialog
from .dialogs.help_window import HelpWindow
from .dialogs.image_augmenter import show_image_augmenter
from .widgets.image_label import ImageLabel
from .dialogs.image_patcher import show_image_patcher
from .inference.sam_utils import SAMUtils
from .dialogs.slice_registration import SliceRegistrationTool
from .dialogs.snake_game import SnakeGame
from .dialogs.stack_interpolator import StackInterpolator
from .dialogs.stack_to_slices import show_stack_to_slices
from .utils import calculate_area, calculate_bbox

warnings.filterwarnings("ignore", category=UserWarning)


class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.is_loading_project = False
        self.backup_project_path = None

        self.project_controller = ProjectController(self)
        self.image_controller = ImageController(self)

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

        self.sam_controller = SAMController(self)
        self.dino_controller = DINOController(self)
        self.yolo_controller = YOLOController(self)

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
        return self.project_controller.update_window_title()

    def new_project(self):
        return self.project_controller.new_project()

    def show_project_search(self):
        from .dialogs.project_search import show_project_search

        show_project_search(self)

    def open_project(self):
        return self.project_controller.open_project()

    def backup_project_before_open(self, project_file):
        return self.project_controller.backup_project_before_open(project_file)

    def restore_project_from_backup(self):
        return self.project_controller.restore_project_from_backup()

    def open_specific_project(self, project_file):
        return self.project_controller.open_specific_project(project_file)

    def load_project_data(self, project_data):
        return self.project_controller.load_project_data(project_data)

    def handle_missing_images(self, missing_images):
        return self.project_controller.handle_missing_images(missing_images)

    def remove_missing_images(self, missing_images):
        return self.project_controller.remove_missing_images(missing_images)

    def prompt_load_missing_images(self, missing_images):
        return self.project_controller.prompt_load_missing_images(missing_images)

    def load_missing_images(self, missing_images):
        return self.project_controller.load_missing_images(missing_images)

    def update_image_list(self):
        return self.image_controller.update_image_list()

    def select_class(self, index):
        if 0 <= index < self.class_list.count():
            item = self.class_list.item(index)
            self.class_list.setCurrentItem(item)
            self.current_class = item.text()
            print(f"Selected class: {self.current_class}")
        else:
            print("Invalid class index")

    def close_project(self):
        return self.project_controller.close_project()

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
        return self.project_controller.check_missing_images()

    def convert_to_serializable(self, obj):
        return image_utils.convert_to_serializable(obj)

    def save_project(self, show_message=True):
        return self.project_controller.save_project(show_message=show_message)

    def save_project_as(self):
        return self.project_controller.save_project_as()

    def auto_save(self):
        return self.project_controller.auto_save()

    def show_project_details(self):
        if not hasattr(self, "current_project_file"):
            QMessageBox.warning(
                self, "No Project", "Please open or create a project first."
            )
            return

        from .dialogs.annotation_statistics import AnnotationStatisticsDialog
        from .dialogs.project_details import ProjectDetailsDialog

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
        return self.image_controller.load_multi_slice_image(image_path, dimensions, shape)

    def activate_sam_magic_wand(self):
        return self.sam_controller.activate_sam_magic_wand()

    def deactivate_sam_magic_wand(self):
        return self.sam_controller.deactivate_sam_magic_wand()

    def toggle_sam_assisted(self):
        return self.sam_controller.toggle_sam_assisted()

    def toggle_sam_magic_wand(self):
        return self.sam_controller.toggle_sam_magic_wand()

    def schedule_sam_prediction(self):
        return self.sam_controller.schedule_sam_prediction()

    def apply_sam_prediction(self):
        return self.sam_controller.apply_sam_prediction()

    def accept_sam_prediction(self):
        return self.sam_controller.accept_sam_prediction()

    def setup_slice_list(self):
        return self.image_controller.setup_slice_list()

    def open_images(self):
        return self.image_controller.open_images()

    def convert_to_8bit_rgb(self, image_array):
        return image_utils.convert_to_8bit_rgb(image_array)

    def add_images_to_list(self, file_names):
        return self.image_controller.add_images_to_list(file_names)

    def update_all_images(self, new_image_info):
        return self.image_controller.update_all_images(new_image_info)

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
        return self.image_controller.switch_slice(item)

    def switch_image(self, item):
        return self.image_controller.switch_image(item)

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
        return self.image_controller.activate_current_slice()

    def load_image(self, image_path):
        return self.image_controller.load_image(image_path)

    def load_tiff(self, image_path, dimensions=None, shape=None, force_dimension_dialog=False):
        return self.image_controller.load_tiff(image_path, dimensions, shape, force_dimension_dialog)

    def load_czi(self, image_path, dimensions=None, shape=None, force_dimension_dialog=False):
        return self.image_controller.load_czi(image_path, dimensions, shape, force_dimension_dialog)

    def load_regular_image(self, image_path):
        return self.image_controller.load_regular_image(image_path)

    def process_multidimensional_image(
        self, image_array, image_path, dimensions=None,
        force_dimension_dialog=False, axes_hint=None,
    ):
        return self.image_controller.process_multidimensional_image(
            image_array, image_path, dimensions, force_dimension_dialog, axes_hint=axes_hint
        )

    def create_slices(self, image_array, dimensions, image_path):
        return self.image_controller.create_slices(image_array, dimensions, image_path)

    def add_slice_to_list(self, slice_name):
        return self.image_controller.add_slice_to_list(slice_name)

    def normalize_array(self, array):
        return image_utils.normalize_array(array)

    def adjust_contrast(self, image, low_percentile=1, high_percentile=99):
        return image_utils.adjust_contrast(image, low_percentile, high_percentile)

    def activate_slice(self, slice_name):
        return self.image_controller.activate_slice(slice_name)

    def array_to_qimage(self, array):
        return image_utils.array_to_qimage(array)

    def update_slice_list(self):
        return self.image_controller.update_slice_list()

    def clear_slice_list(self):
        return self.image_controller.clear_slice_list()

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
        return self.dino_controller.has_visible_temp_classes()

    def launch_snake_game(self):
        # print("Launching Snake game")
        if not hasattr(self, "snake_game") or not self.snake_game.isVisible():
            self.snake_game = SnakeGame()
        self.snake_game.show()
        self.snake_game.setFocus()

    def import_annotations(self):
        return io_controller.import_annotations(self)

    def export_annotations(self):
        return io_controller.export_annotations(self)

    def save_slices(self, directory):
        return io_controller.save_slices(self, directory)

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
        theme.change_font_size(self, size)

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
        return self.sam_controller.toggle_sam_box()

    def toggle_sam_points(self):
        return self.sam_controller.toggle_sam_points()

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
        return self.sam_controller.change_sam_model(model_name)

    # --- DINO / LLM-Assisted Detection Methods ---

    def _resolve_dino_model_path(self, model_name: str) -> str | None:
        """Return the canonical local path for a preset DINO model, or None if unknown."""
        return self.dino_controller._resolve_dino_model_path(model_name)

    def _on_dino_model_changed(self, text):
        return self.dino_controller._on_dino_model_changed(text)

    def _ensure_dino_model_downloaded(self, model_name):
        return self.dino_controller._ensure_dino_model_downloaded(model_name)

    def browse_dino_model(self):
        return self.dino_controller.browse_dino_model()

    def on_dino_class_row_changed(self):
        return self.dino_controller.on_dino_class_row_changed()

    def _build_dino_class_configs(self):
        return self.dino_controller._build_dino_class_configs()

    def run_dino_detection_single(self):
        return self.dino_controller.run_dino_detection_single()

    def run_dino_detection_batch(self):
        return self.dino_controller.run_dino_detection_batch()


    def _collect_dino_batch_work_items(self):
        return self.dino_controller._collect_dino_batch_work_items()

    def _commit_dino_results(self, image_name, dino_results, sam_results):
        return self.dino_controller._commit_dino_results(image_name, dino_results, sam_results)

    def _store_dino_batch_results(self, image_name, dino_results, sam_results):
        return self.dino_controller._store_dino_batch_results(image_name, dino_results, sam_results)

    def _show_dino_batch_review(self):
        return self.dino_controller._show_dino_batch_review()

    def _navigate_to_image_or_slice(self, name):
        return self.dino_controller._navigate_to_image_or_slice(name)

    def _refresh_dino_temp_for_current(self):
        return self.dino_controller._refresh_dino_temp_for_current()

    def accept_dino_results(self):
        return self.dino_controller.accept_dino_results()

    def reject_dino_results(self):
        return self.dino_controller.reject_dino_results()

    def setup_font_size_selector(self):
        theme.setup_font_size_selector(self)

    def on_font_size_changed(self, size):
        theme.on_font_size_changed(self, size)

    def apply_theme_and_font(self):
        theme.apply_theme_and_font(self)

    def toggle_dark_mode(self):
        theme.toggle_dark_mode(self)

    def apply_stylesheet(self):
        theme.apply_stylesheet(self)

    def update_ui_colors(self):
        theme.update_ui_colors(self)

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
        return self.image_controller.is_multi_dimensional(file_name)

    def predict_single_image(self, file_name):
        return self.yolo_controller.predict_single_image(file_name)

    def redefine_dimensions(self, file_name):
        return self.image_controller.redefine_dimensions(file_name)

    def remove_image(self):
        return self.image_controller.remove_image()

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
        return self.image_controller.delete_selected_image()

    def display_image(self):
        return self.image_controller.display_image()

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
        return self.yolo_controller.setup_yolo_menu()

    def load_yolo_model(self):
        return self.yolo_controller.load_yolo_model()

    def prepare_yolo_dataset(self):
        return self.yolo_controller.prepare_yolo_dataset()

    def load_yolo_yaml(self):
        return self.yolo_controller.load_yolo_yaml()

    def save_yolo_model(self):
        return self.yolo_controller.save_yolo_model()

    def load_prediction_model(self):
        return self.yolo_controller.load_prediction_model()

    def show_train_dialog(self):
        return self.yolo_controller.show_train_dialog()

    def initialize_yolo_trainer(self):
        return self.yolo_controller.initialize_yolo_trainer()

    def start_training(self, epochs, imgsz):
        return self.yolo_controller.start_training(epochs, imgsz)

    def training_finished(self, results):
        return self.yolo_controller.training_finished(results)

    def set_confidence_threshold(self):
        return self.yolo_controller.set_confidence_threshold()

    def show_predict_dialog(self):
        return self.yolo_controller.show_predict_dialog()

    def run_predictions(self, selected_images):
        return self.yolo_controller.run_predictions(selected_images)

    def process_yolo_results(self, results, image_name):
        return self.yolo_controller.process_yolo_results(results, image_name)

    def add_temp_classes(self, temp_annotations):
        return self.dino_controller.add_temp_classes(temp_annotations)

    def verify_current_class(self):
        return self.dino_controller.verify_current_class()

    def accept_visible_temp_classes(self):
        return self.dino_controller.accept_visible_temp_classes()

    def select_first_primary_class(self):
        return self.dino_controller.select_first_primary_class()

    def reject_visible_temp_classes(self):
        return self.dino_controller.reject_visible_temp_classes()

    def is_class_visible(self, class_name):
        items = self.class_list.findItems(class_name, Qt.MatchFlag.MatchExactly)
        if items:
            return items[0].checkState() == Qt.CheckState.Checked
        return False

    def check_temp_annotations(self):
        return self.dino_controller.check_temp_annotations()

    def remove_all_temp_annotations(self):
        return self.dino_controller.remove_all_temp_annotations()
