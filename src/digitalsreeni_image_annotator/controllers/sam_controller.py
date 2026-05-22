"""SAM (Segment Anything) coordination controller.

Extracted from `ImageAnnotator`. Owns the SAM tool lifecycle (magic
wand, box, points), the debounce timer state machine, ADR-013's
in-flight re-entrancy guard, and the model picker dropdown plumbing.

State (`sam_utils`, `sam_inference_timer`, `_sam_inference_in_flight`,
`current_sam_model`) stays on the main window in this phase for the
same reason ProjectController / ImageController state stays there:
external callers (image_label.py, clear_all, the sidebar button
enabling logic) read these attributes directly via `main_window.X`. A
future phase may migrate ownership.

ADR-013 invariants preserved verbatim:
- `_sam_inference_in_flight` flag set BEFORE calling
  `sam_utils.apply_sam_*`, cleared in `finally`.
- `InferenceBusyError` (raised by `sam_utils._run_sync` when the worker
  thread is already running) is swallowed silently — the next user
  click restarts the debounce.
- `change_sam_model` blocks via `_run_sync` event-loop pump; UI stays
  responsive.
"""

import traceback

from PyQt6.QtCore import Qt, QObject
from PyQt6.QtWidgets import QMessageBox

from ..inference.sam_utils import InferenceBusyError


class SAMController(QObject):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window

    def activate_sam_magic_wand(self):
        for button in self.mw.tool_group.buttons():
            if button != self.mw.sam_magic_wand_button:
                button.setChecked(False)

        self.mw.image_label.current_tool = "sam_magic_wand"
        self.mw.image_label.sam_magic_wand_active = True
        self.mw.image_label.setCursor(Qt.CursorShape.CrossCursor)

        self.mw.update_ui_for_current_tool()

        if self.mw.current_class is None and self.mw.class_list.count() > 0:
            self.mw.class_list.setCurrentRow(0)
            self.mw.current_class = self.mw.class_list.currentItem().text()
        elif self.mw.class_list.count() == 0:
            QMessageBox.warning(
                self.mw,
                "No Class Selected",
                "Please add a class before using annotation tools.",
            )
            self.mw.sam_magic_wand_button.setChecked(False)
            self.deactivate_sam_magic_wand()

    def deactivate_sam_magic_wand(self):
        self.mw.image_label.current_tool = None
        self.mw.image_label.sam_magic_wand_active = False
        self.mw.sam_magic_wand_button.setChecked(False)
        self.mw.sam_magic_wand_button.setEnabled(False)
        self.mw.image_label.setCursor(Qt.CursorShape.ArrowCursor)

        self.mw.image_label.sam_bbox = None
        self.mw.image_label.drawing_sam_bbox = False
        self.mw.image_label.temp_sam_prediction = None

        self.mw.update_ui_for_current_tool()

    def toggle_sam_assisted(self):
        if not self.mw.current_sam_model:
            QMessageBox.warning(
                self.mw,
                "No SAM Model Selected",
                "Please pick a SAM model before using the SAM-Assisted tool.",
            )
            self.mw.sam_magic_wand_button.setChecked(False)
            return

        if self.mw.sam_magic_wand_button.isChecked():
            self.activate_sam_magic_wand()
        else:
            self.deactivate_sam_magic_wand()

        self.mw.image_label.clear_temp_sam_prediction()

    def toggle_sam_magic_wand(self):
        if self.mw.sam_magic_wand_button.isChecked():
            if self.mw.current_class is None:
                QMessageBox.warning(
                    self.mw,
                    "No Class Selected",
                    "Please select a class before using SAM2 Magic Wand.",
                )
                self.mw.sam_magic_wand_button.setChecked(False)
                return
            self.mw.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.mw.image_label.sam_magic_wand_active = True
        else:
            self.mw.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.mw.image_label.sam_magic_wand_active = False
            self.mw.image_label.sam_bbox = None

        self.mw.image_label.clear_temp_sam_prediction()

    def schedule_sam_prediction(self):
        """Restart the debounce timer; inference fires 1s after last click."""
        self.mw.sam_inference_timer.stop()
        self.mw.sam_inference_timer.start(1000)

    def cancel_sam_prediction(self):
        """Cancel a pending SAM points prediction. Triggered by Escape
        in ImageLabel while sam_points mode is active."""
        self.mw.sam_inference_timer.stop()

    def apply_sam_prediction(self):
        # Re-entry guard (ADR-013): the event-loop pump inside _run_sync
        # can deliver this timer fire before the first call returns.
        # Bail and rely on the user clicking again (which restarts the
        # debounce) to issue a fresh inference with the up-to-date
        # point set.
        if self.mw._sam_inference_in_flight:
            return
        self.mw._sam_inference_in_flight = True
        try:
            try:
                if self.mw.image_label.current_tool == "sam_box":
                    if self.mw.image_label.sam_bbox is None:
                        print("SAM bbox is None")
                        return
                    x1, y1, x2, y2 = self.mw.image_label.sam_bbox
                    bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    prediction = self.mw.sam_utils.apply_sam_prediction(
                        self.mw.current_image, bbox
                    )
                    self.mw.image_label.sam_bbox = None
                elif self.mw.image_label.current_tool == "sam_points":
                    pos_points = self.mw.image_label.sam_positive_points
                    neg_points = self.mw.image_label.sam_negative_points
                    print(
                        f"[SAM-POINTS] Predicting with {len(pos_points)} positive points: {pos_points} "
                        f"and {len(neg_points)} negative points: {neg_points}"
                    )
                    if not pos_points:
                        print("No positive points for SAM-points")
                        return
                    prediction = self.mw.sam_utils.apply_sam_points(
                        self.mw.current_image,
                        pos_points,
                        neg_points,
                    )
                else:
                    return
            except InferenceBusyError:
                # Re-entry safety net from sam_utils itself. The
                # call-site flag above should catch this first, but if
                # a different caller drives inference concurrently we
                # skip — the user keeps interacting; their next click
                # will restart the debounce.
                return
            except Exception as exc:
                traceback.print_exc()
                QMessageBox.critical(
                    self.mw,
                    "SAM Error",
                    f"SAM inference failed:\n\n{exc}\n\n"
                    "See the log for details.",
                )
                return

            if prediction:
                temp_annotation = {
                    "segmentation": prediction["segmentation"],
                    "category_id": self.mw.class_mapping[self.mw.current_class],
                    "category_name": self.mw.current_class,
                    "score": prediction["score"],
                }
                self.mw.image_label.temp_sam_prediction = temp_annotation
                self.mw.image_label.update()
            elif prediction is None:
                QMessageBox.information(
                    self.mw,
                    "SAM",
                    "No mask matches the given constraints. "
                    "Try adjusting the box or point positions."
                )
            else:
                print("Failed to generate prediction")

            if self.mw.image_label.current_tool == "sam_box":
                self.mw.image_label.sam_bbox = None
                self.mw.image_label.update()
        finally:
            self.mw._sam_inference_in_flight = False

    def accept_sam_prediction(self):
        if self.mw.image_label.temp_sam_prediction:
            new_annotation = self.mw.image_label.temp_sam_prediction
            self.mw.image_label.annotations.setdefault(
                new_annotation["category_name"], []
            ).append(new_annotation)
            self.mw.add_annotation_to_list(new_annotation)
            self.mw.save_current_annotations()
            self.mw.update_slice_list_colors()
            self.mw.image_label.temp_sam_prediction = None
            self.mw.image_label.sam_positive_points = []
            self.mw.image_label.sam_negative_points = []
            self.mw.image_label.update()
            print("SAM prediction accepted, points cleared, and added to annotations.")

    def toggle_sam_box(self):
        if self.mw.sam_box_button.isChecked():
            self.mw.sam_points_button.setChecked(False)
            self.mw.image_label.current_tool = "sam_box"
            self.mw.image_label.sam_box_active = True
            self.mw.image_label.sam_points_active = False
            self.mw.image_label.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.mw.image_label.current_tool = None
            self.mw.image_label.sam_box_active = False
            self.mw.image_label.setCursor(Qt.CursorShape.ArrowCursor)
        self.mw.update_ui_for_current_tool()

    def toggle_sam_points(self):
        if self.mw.sam_points_button.isChecked():
            self.mw.sam_box_button.setChecked(False)
            self.mw.image_label.current_tool = "sam_points"
            self.mw.image_label.sam_points_active = True
            self.mw.image_label.sam_box_active = False
            self.mw.image_label.setCursor(Qt.CursorShape.CrossCursor)
            self.mw.image_label.sam_positive_points = []
            self.mw.image_label.sam_negative_points = []
        else:
            self.mw.sam_inference_timer.stop()
            self.mw.image_label.current_tool = None
            self.mw.image_label.sam_points_active = False
            self.mw.image_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.mw.image_label.sam_positive_points = []
            self.mw.image_label.sam_negative_points = []
        self.mw.update_ui_for_current_tool()

    def change_sam_model(self, model_name):
        try:
            self.mw.sam_utils.change_sam_model(model_name)
        except Exception as e:
            QMessageBox.critical(
                self.mw,
                "SAM Model Error",
                f"Failed to load SAM model '{model_name}':\n\n{str(e)}\n\n"
                "Check that the model weights are downloadable and that torch "
                "is correctly installed for your platform / GPU."
            )
            self.mw.sam_model_selector.setCurrentIndex(0)
            return

        self.mw.current_sam_model = self.mw.sam_utils.current_sam_model

        if model_name != "Pick a SAM Model":
            self.mw.sam_magic_wand_button.setEnabled(True)

            self.mw.sam_magic_wand_button.setChecked(True)
            self.activate_sam_magic_wand()

            print(f"Changed SAM model to: {model_name}")
        else:
            self.mw.sam_magic_wand_button.setEnabled(False)
            self.mw.sam_magic_wand_button.setChecked(False)
            self.deactivate_sam_magic_wand()
            print("SAM model unset")
