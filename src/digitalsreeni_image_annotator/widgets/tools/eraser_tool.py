"""EraserTool — circular strokes mask out existing polygons; Enter commits."""

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

from .base import ToolHandler


class EraserTool(ToolHandler):
    """Mutates ImageLabel's `temp_eraser_mask` and `is_erasing`. The
    commit path (OpenCV polygon clipping) is moved byte-for-byte
    from the pre-Phase-7 ImageLabel.commit_eraser_changes — do not
    refactor here."""

    def on_mouse_press(self, event, img_pt) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        self._start(img_pt)
        return True

    def on_mouse_move(self, event, img_pt) -> bool:
        if event.buttons() != Qt.MouseButton.LeftButton:
            return False
        if not self.label.is_erasing:
            return False
        self._continue(img_pt)
        return True

    def on_mouse_release(self, event, img_pt) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if not self.label.is_erasing:
            return False
        self.label.is_erasing = False
        # Don't commit the eraser changes yet; Enter or image-switch
        # finalises.
        return True

    def on_enter(self) -> bool:
        if self.label.temp_eraser_mask is None:
            return False
        self.commit()
        return True

    def on_escape(self) -> bool:
        if self.label.temp_eraser_mask is None:
            return False
        self.discard()
        return True

    def paint_overlay(self, painter) -> None:
        mask = self.label.temp_eraser_mask
        if mask is None:
            return
        painter.save()
        painter.translate(self.label.offset_x, self.label.offset_y)
        painter.scale(self.label.zoom_factor, self.label.zoom_factor)

        mask_copy = mask.copy()
        mask_image = QImage(
            mask_copy.data,
            mask_copy.shape[1],
            mask_copy.shape[0],
            mask_copy.shape[1],
            QImage.Format.Format_Grayscale8,
        )
        mask_pixmap = QPixmap.fromImage(mask_image)
        painter.setOpacity(0.5)
        painter.drawPixmap(0, 0, mask_pixmap)
        painter.setOpacity(1.0)
        painter.restore()

    def has_unsaved_state(self) -> bool:
        return self.label.temp_eraser_mask is not None

    def commit(self) -> None:
        if self.label.temp_eraser_mask is None:
            return
        eraser_mask = self.label.temp_eraser_mask.astype(bool)
        current_name = self.label._ctx.current_image_key()

        for class_name, annotations in self.label.annotations.items():
            updated_annotations = []
            max_number = max([ann.get("number", 0) for ann in annotations] + [0])
            for annotation in annotations:
                if "segmentation" in annotation:
                    points = (
                        np.array(annotation["segmentation"])
                        .reshape(-1, 2)
                        .astype(int)
                    )
                    mask = np.zeros_like(self.label.temp_eraser_mask)
                    cv2.fillPoly(mask, [points], 255)
                    mask = mask.astype(bool)
                    mask[eraser_mask] = False
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    for i, contour in enumerate(contours):
                        if cv2.contourArea(contour) > 10:  # Minimum area threshold
                            new_segmentation = contour.flatten().tolist()
                            new_annotation = annotation.copy()
                            new_annotation["segmentation"] = new_segmentation
                            if i == 0:
                                new_annotation["number"] = annotation.get(
                                    "number", max_number + 1
                                )
                            else:
                                max_number += 1
                                new_annotation["number"] = max_number
                            updated_annotations.append(new_annotation)
                else:
                    updated_annotations.append(annotation)
            self.label.annotations[class_name] = updated_annotations

        self.label.temp_eraser_mask = None
        # AnnotationController.replace_annotations writes into
        # all_annotations and triggers save + slice-color refresh.
        self.label.annotationsReplaced.emit(current_name, self.label.annotations)
        self.label.update()

    def discard(self) -> None:
        self.label.temp_eraser_mask = None
        self.label.update()

    # --- internals ---

    def _start(self, pos):
        if self.label.temp_eraser_mask is None:
            self.label.temp_eraser_mask = np.zeros(
                (
                    self.label.original_pixmap.height(),
                    self.label.original_pixmap.width(),
                ),
                dtype=np.uint8,
            )
        self.label.is_erasing = True
        self._continue(pos)

    def _continue(self, pos):
        if not self.label.is_erasing:
            return
        eraser_size = self.label._ctx.eraser_size()
        cv2.circle(
            self.label.temp_eraser_mask,
            (int(pos[0]), int(pos[1])),
            eraser_size,
            255,
            -1,
        )
        self.label.update()
