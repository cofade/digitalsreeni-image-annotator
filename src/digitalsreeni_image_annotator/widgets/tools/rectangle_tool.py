"""RectangleTool — drag a bbox; release commits via finishRectangleRequested."""

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPen

from .base import ToolHandler


class RectangleTool(ToolHandler):
    """Mutates ImageLabel state fields (`start_point`, `end_point`,
    `current_rectangle`, `drawing_rectangle`) directly. Those fields
    stay on the widget because AnnotationController.finish_rectangle
    reads `mw.image_label.current_rectangle`. Moving them onto the
    tool would require a parallel controller refactor; out of scope
    for Phase 7."""

    def on_mouse_press(self, event, img_pt) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        self.label.start_point = img_pt
        self.label.end_point = img_pt
        self.label.drawing_rectangle = True
        self.label.current_rectangle = None
        return True

    def on_mouse_move(self, event, img_pt) -> bool:
        if not self.label.drawing_rectangle:
            return False
        self.label.end_point = img_pt
        self.label.current_rectangle = self._rect_from_points()
        return True

    def on_mouse_release(self, event, img_pt) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if not self.label.drawing_rectangle:
            return False
        self.label.drawing_rectangle = False
        if self.label.current_rectangle:
            self.label.finishRectangleRequested.emit()
        return True

    def _rect_from_points(self):
        s = self.label.start_point
        e = self.label.end_point
        if not s or not e:
            return None
        x1, y1 = s
        x2, y2 = e
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def paint_overlay(self, painter) -> None:
        if not (self.label.drawing_rectangle and self.label.current_rectangle):
            return
        painter.save()
        painter.translate(self.label.offset_x, self.label.offset_y)
        painter.scale(self.label.zoom_factor, self.label.zoom_factor)

        x1, y1, x2, y2 = self.label.current_rectangle
        color = self.label.class_colors.get(
            self.label._ctx.current_class(), QColor(Qt.GlobalColor.red)
        )
        painter.setPen(
            QPen(color, 2 / self.label.zoom_factor, Qt.PenStyle.SolidLine)
        )
        painter.drawRect(
            QRectF(float(x1), float(y1), float(x2 - x1), float(y2 - y1))
        )
        painter.restore()

    def has_unsaved_state(self) -> bool:
        # A rectangle commits automatically on mouse release (emits
        # finishRectangleRequested) — there's no "draft" rectangle the
        # user might want to save later. The only way to have lingering
        # state is mid-drag; in that case discard() clears it on tool
        # switch / image switch via check_unsaved_changes.
        return self.label.drawing_rectangle

    def discard(self) -> None:
        self.label.start_point = None
        self.label.end_point = None
        self.label.current_rectangle = None
        self.label.drawing_rectangle = False

    def commit(self) -> None:
        # Mid-drag rectangle isn't finishable (no mouse-release signal
        # was emitted yet). Treat "Yes save" as discard for consistency
        # with the dialog's intent — the user clicked Yes meaning "keep
        # what I drew" but there's nothing complete to keep.
        self.discard()
