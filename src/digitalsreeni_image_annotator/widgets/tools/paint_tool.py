"""PaintBrushTool — circular brush strokes into a temp mask; Enter commits."""

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

from .base import ToolHandler


class PaintBrushTool(ToolHandler):
    """Mutates ImageLabel's `temp_paint_mask` and `is_painting` so
    other code paths (notably `check_unsaved_changes` callers and
    paint-mask rendering) see the same state they did pre-Phase-7."""

    def on_mouse_press(self, event, img_pt) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        self._start(img_pt)
        return True

    def on_mouse_move(self, event, img_pt) -> bool:
        if event.buttons() != Qt.MouseButton.LeftButton:
            return False
        if not self.label.is_painting:
            return False
        self._continue(img_pt)
        return True

    def on_mouse_release(self, event, img_pt) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if not self.label.is_painting:
            return False
        self.label.is_painting = False
        # Don't commit the annotation yet; Enter / image-switch dialog
        # finalises.
        return True

    def on_enter(self) -> bool:
        if self.label.temp_paint_mask is None:
            return False
        self.commit()
        return True

    def on_escape(self) -> bool:
        if self.label.temp_paint_mask is None:
            return False
        self.discard()
        return True

    def paint_overlay(self, painter) -> None:
        mask = self.label.temp_paint_mask
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
        return self.label.temp_paint_mask is not None

    def commit(self) -> None:
        if self.label.temp_paint_mask is None or not self.label._ctx.current_class():
            return
        class_name = self.label._ctx.current_class()
        contours, _ = cv2.findContours(
            self.label.temp_paint_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Minimum area threshold
                segmentation = contour.flatten().tolist()
                new_annotation = {
                    "segmentation": segmentation,
                    "category_id": self.label._ctx.class_id(class_name),
                    "category_name": class_name,
                }
                self.label.annotations.setdefault(class_name, []).append(new_annotation)
                self.label.annotationCommitted.emit(new_annotation)
        self.label.temp_paint_mask = None
        self.label.annotationsBatchSaved.emit()
        self.label.update()

    def discard(self) -> None:
        self.label.temp_paint_mask = None
        self.label.update()

    # --- internals ---

    def _start(self, pos):
        if self.label.temp_paint_mask is None:
            # Fresh stroke: capture the pre-paint state for undo before any
            # mask exists. The commit pushes it on annotationsBatchSaved
            # (ADR-026).
            self.label.editBaselineRequested.emit()
            self.label.temp_paint_mask = np.zeros(
                (
                    self.label.original_pixmap.height(),
                    self.label.original_pixmap.width(),
                ),
                dtype=np.uint8,
            )
        self.label.is_painting = True
        self._continue(pos)

    def _continue(self, pos):
        if not self.label.is_painting:
            return
        brush_size = self.label._ctx.paint_brush_size()
        cv2.circle(
            self.label.temp_paint_mask,
            (int(pos[0]), int(pos[1])),
            brush_size,
            255,
            -1,
        )
        self.label.update()
