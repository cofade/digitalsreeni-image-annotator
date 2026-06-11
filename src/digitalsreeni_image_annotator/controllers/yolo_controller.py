"""YOLO training / prediction coordination controller.

Extracted from `ImageAnnotator`. Owns:

- The YOLO menu (Training submenu + Prediction Settings submenu)
- Pre-trained model loading and dataset preparation
- Training: dialog wiring, the `TrainingThread` worker, progress
  callback chain, finish handler
- Prediction: model loading via `LoadPredictionModelDialog`, the
  confidence-threshold dialog, single-image and multi-image prediction
- Result post-processing (`process_yolo_results`) that converts YOLO
  output into temp annotations for the user to review

State (`yolo_trainer`, `training_thread`, `training_dialog`) stays on
the main window — the menu actions and signal connections are
addressed from elsewhere as `main_window.X`, and `training_dialog` is
referenced via `hasattr(self, "training_dialog")` to lazily initialize.
"""

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..dialogs.yolo_trainer import (
    LoadPredictionModelDialog,
    TrainingInfoDialog,
    YOLOTrainer,
)


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


class YOLOController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window

    def setup_yolo_menu(self):
        yolo_menu = self.mw.menuBar().addMenu("&YOLO (beta)")

        training_submenu = yolo_menu.addMenu("Training")

        load_pretrained_action = QAction("Load Pre-trained Model", self.mw)
        load_pretrained_action.triggered.connect(self.load_yolo_model)
        training_submenu.addAction(load_pretrained_action)

        prepare_data_action = QAction("Prepare YOLO Dataset", self.mw)
        prepare_data_action.triggered.connect(self.prepare_yolo_dataset)
        training_submenu.addAction(prepare_data_action)

        load_yaml_action = QAction("Load Dataset YAML", self.mw)
        load_yaml_action.triggered.connect(self.load_yolo_yaml)
        training_submenu.addAction(load_yaml_action)

        train_action = QAction("Train Model", self.mw)
        train_action.triggered.connect(self.show_train_dialog)
        training_submenu.addAction(train_action)

        save_model_action = QAction("Save Model", self.mw)
        save_model_action.triggered.connect(self.save_yolo_model)
        training_submenu.addAction(save_model_action)

        prediction_submenu = yolo_menu.addMenu("Prediction Settings")

        load_model_action = QAction("Load Model", self.mw)
        load_model_action.triggered.connect(self.load_prediction_model)
        prediction_submenu.addAction(load_model_action)

        set_threshold_action = QAction("Set Confidence Threshold", self.mw)
        set_threshold_action.triggered.connect(self.set_confidence_threshold)
        prediction_submenu.addAction(set_threshold_action)

    def initialize_yolo_trainer(self):
        if hasattr(self.mw, "current_project_dir"):
            self.mw.yolo_trainer = YOLOTrainer(self.mw.current_project_dir, self.mw)
        else:
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )

    def load_yolo_model(self):
        if not hasattr(self.mw, "current_project_dir"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer:
            self.initialize_yolo_trainer()

        if self.mw.yolo_trainer.load_model():
            QMessageBox.information(
                self.mw, "Model Loaded", "YOLO model loaded successfully."
            )
        else:
            QMessageBox.warning(
                self.mw, "Load Cancelled", "Model loading was cancelled."
            )

    def prepare_yolo_dataset(self):
        if not hasattr(self.mw, "current_project_file"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer:
            self.initialize_yolo_trainer()

        try:
            yaml_path = self.mw.yolo_trainer.prepare_dataset()
            QMessageBox.information(
                self.mw,
                "Dataset Prepared",
                f"YOLO dataset prepared successfully. YAML file: {yaml_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self.mw,
                "Error",
                f"An error occurred while preparing the dataset: {str(e)}",
            )

    def load_yolo_yaml(self):
        if not hasattr(self.mw, "current_project_file"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer:
            self.initialize_yolo_trainer()

        try:
            if self.mw.yolo_trainer.load_yaml():
                QMessageBox.information(
                    self.mw, "YAML Loaded", "Dataset YAML loaded successfully."
                )
            else:
                QMessageBox.warning(
                    self.mw, "Load Cancelled", "YAML loading was cancelled."
                )
        except Exception as e:
            QMessageBox.critical(
                self.mw,
                "Error",
                f"An error occurred while loading the YAML file: {str(e)}",
            )

    def save_yolo_model(self):
        if not hasattr(self.mw, "current_project_file"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer or not self.mw.yolo_trainer.model:
            QMessageBox.warning(
                self.mw, "No Model", "Please train or load a YOLO model first."
            )
            return

        try:
            if self.mw.yolo_trainer.save_model():
                QMessageBox.information(
                    self.mw, "Model Saved", "YOLO model saved successfully."
                )
            else:
                QMessageBox.warning(
                    self.mw, "Save Cancelled", "Model saving was cancelled."
                )
        except Exception as e:
            QMessageBox.critical(
                self.mw, "Error", f"An error occurred while saving the model: {str(e)}"
            )

    def load_prediction_model(self):
        if not hasattr(self.mw, "current_project_file"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer:
            self.initialize_yolo_trainer()

        dialog = LoadPredictionModelDialog(self.mw)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            model_path = dialog.model_path
            yaml_path = dialog.yaml_path
            if model_path and yaml_path:
                try:
                    result, message = self.mw.yolo_trainer.load_prediction_model(
                        model_path, yaml_path
                    )
                    if result:
                        QMessageBox.information(
                            self.mw,
                            "Model Loaded",
                            "YOLO model and YAML file loaded successfully for prediction.",
                        )
                        if message:
                            QMessageBox.warning(
                                self.mw, "Class Mismatch Warning", message
                            )
                    else:
                        QMessageBox.critical(
                            self.mw,
                            "Error Loading Model",
                            f"Could not load the model or YAML file: {message}",
                        )
                except Exception as e:
                    QMessageBox.critical(
                        self.mw, "Error", f"An error occurred: {str(e)}"
                    )
            else:
                QMessageBox.warning(
                    self.mw,
                    "Files Required",
                    "Both model and YAML files are required for prediction.",
                )

    def show_train_dialog(self):
        if not self.mw.yolo_trainer:
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return
        if not self.mw.yolo_trainer.model:
            QMessageBox.warning(
                self.mw, "No Model", "Please load a pre-trained model first."
            )
            return
        if not self.mw.yolo_trainer.yaml_path:
            QMessageBox.warning(
                self.mw, "No Dataset", "Please prepare or load a dataset YAML first."
            )
            return

        dialog = QDialog(self.mw)
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

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            epochs = int(epochs_input.text())
            imgsz = int(imgsz_input.text())
            self.start_training(epochs, imgsz)

    def start_training(self, epochs, imgsz):
        if not hasattr(self.mw, "training_dialog"):
            self.mw.training_dialog = TrainingInfoDialog(self.mw)
        self.mw.training_dialog.show()

        self.mw.yolo_trainer.progress_signal.connect(
            self.mw.training_dialog.update_info
        )
        self.mw.yolo_trainer.set_progress_callback(self.mw.training_dialog.update_info)
        self.mw.training_dialog.stop_signal.connect(
            self.mw.yolo_trainer.stop_training_signal
        )

        self.mw.training_thread = TrainingThread(self.mw.yolo_trainer, epochs, imgsz)
        self.mw.training_thread.finished.connect(self.training_finished)
        self.mw.training_thread.start()

    def training_finished(self, results):
        self.mw.training_dialog.stop_button.setEnabled(True)
        self.mw.training_dialog.stop_button.setText("Stop Training")
        self.mw.yolo_trainer.progress_signal.disconnect(
            self.mw.training_dialog.update_info
        )
        self.mw.training_dialog.stop_signal.disconnect(
            self.mw.yolo_trainer.stop_training_signal
        )

        if isinstance(results, str):
            QMessageBox.critical(
                self.mw,
                "Training Error",
                f"An error occurred during training: {results}",
            )
        else:
            QMessageBox.information(
                self.mw,
                "Training Complete",
                "YOLO model training completed successfully.",
            )

    def set_confidence_threshold(self):
        if not hasattr(self.mw, "current_project_file"):
            QMessageBox.warning(
                self.mw, "No Project", "Please open or create a project first."
            )
            return

        if not self.mw.yolo_trainer:
            self.initialize_yolo_trainer()

        current_threshold = self.mw.yolo_trainer.conf_threshold
        new_threshold, ok = QInputDialog.getDouble(
            self.mw,
            "Set Confidence Threshold",
            "Enter confidence threshold (0-1):",
            current_threshold,
            0,
            1,
            2,
        )
        if ok:
            self.mw.yolo_trainer.set_conf_threshold(new_threshold)
            QMessageBox.information(
                self.mw,
                "Threshold Updated",
                f"Confidence threshold set to {new_threshold}",
            )

    def show_predict_dialog(self):
        if not self.mw.yolo_trainer or not self.mw.yolo_trainer.model:
            QMessageBox.warning(self.mw, "No Model", "Please load a YOLO model first.")
            return

        dialog = QDialog(self.mw)
        dialog.setWindowTitle("Predict with YOLO Model")
        layout = QVBoxLayout()

        image_list = QListWidget()
        for image_name in self.mw.image_paths.keys():
            image_list.addItem(image_name)
        layout.addWidget(QLabel("Select images for prediction:"))
        layout.addWidget(image_list)

        conf_label = QLabel("Confidence Threshold:")
        conf_input = QDoubleSpinBox()
        conf_input.setRange(0, 1)
        conf_input.setSingleStep(0.01)
        conf_input.setValue(self.mw.yolo_trainer.conf_threshold)
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
            self.mw.yolo_trainer.set_conf_threshold(conf)
            self.run_predictions(selected_images)

    def run_predictions(self, selected_images):
        for image_name in selected_images:
            image_path = self.mw.image_paths[image_name]
            results = self.mw.yolo_trainer.predict(image_path)
            self.process_yolo_results(results, image_name)

    def predict_single_image(self, file_name):
        if self.mw.is_multi_dimensional(file_name):
            return

        if not self.mw.yolo_trainer or not self.mw.yolo_trainer.model:
            QMessageBox.warning(
                self.mw,
                "No Model",
                "Please load a YOLO model first from the YOLO > Prediction Settings > Load Model menu.",
            )
            return

        self.mw.deactivate_sam_tools()

        image_path = self.mw.image_paths[file_name]
        try:
            results = self.mw.yolo_trainer.predict(image_path)
            self.process_yolo_results(results, file_name)
        except Exception as e:
            QMessageBox.warning(
                self.mw,
                "Prediction Error",
                f"An error occurred during prediction: {str(e)}\n\n"
                "This might be due to a mismatch between the model and the YAML file classes. "
                "Please check that the YAML file corresponds to the loaded model.",
            )

    def process_yolo_results(self, results, image_name):
        image_path = self.mw.image_paths[image_name]
        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.warning(self.mw, "Error", f"Failed to load image: {image_name}")
            return
        original_height, original_width = image.shape[:2]

        temp_annotations = {}

        try:
            results, input_size, original_size = results
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
                        class_name = self.mw.yolo_trainer.class_names[class_id]
                        score = float(box.conf)

                        mask_array = mask.data.cpu().numpy()[0]
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
                            self.mw,
                            "Class Mismatch",
                            "There is a mismatch between the model and the YAML file classes. "
                            "Please check that the YAML file corresponds to the loaded model.",
                        )
                        return

        except Exception as e:
            QMessageBox.warning(
                self.mw,
                "Prediction Error",
                f"An error occurred during prediction: {str(e)}\n\n"
                "This might be due to a mismatch between the model and the YAML file classes. "
                "Please check that the YAML file corresponds to the loaded model.",
            )
            return

        self.mw.add_temp_classes(temp_annotations)
        self.mw.update_class_list()
        self.mw.image_label.update()

        if temp_annotations:
            total_predictions = sum(len(anns) for anns in temp_annotations.values())
            QMessageBox.information(
                self.mw,
                "Review Predictions",
                f"Found {total_predictions} predictions for {len(temp_annotations)} classes.\n"
                "Use class visibility checkboxes to review.\n"
                "Press Enter to accept or Esc to reject visible predictions.",
            )
        else:
            QMessageBox.information(
                self.mw,
                "No Predictions",
                "No predictions were found for this image.",
            )

        self.mw.deactivate_sam_tools()
