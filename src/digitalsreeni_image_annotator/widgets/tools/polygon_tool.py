"""PolygonTool — click to add vertices, double-click / Enter to finish."""

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QPen, QPolygonF

from .base import ToolHandler


class PolygonTool(ToolHandler):
    """Mutates ImageLabel state fields (`current_annotation`,
    `temp_point`, `drawing_polygon`) directly — those fields are
    still read by AnnotationController.finish_polygon and stay on
    the widget for now."""

    def on_mouse_press(self, event, img_pt) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if not self.label.drawing_polygon:
            self.label.drawing_polygon = True
            self.label.current_annotation = []
        self.label.current_annotation.append(img_pt)
        return True

    def on_mouse_move(self, event, img_pt) -> bool:
        if not self.label.current_annotation:
            return False
        self.label.temp_point = img_pt
        return True

    def on_double_click(self, event, img_pt) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if self.label.drawing_polygon and len(self.label.current_annotation) > 2:
            self.label.drawing_polygon = False
            self.label.finishPolygonRequested.emit()
            return True
        return False

    def on_enter(self) -> bool:
        if self.label.drawing_polygon and len(self.label.current_annotation) > 2:
            self.label.drawing_polygon = False
            self.label.finishPolygonRequested.emit()
            return True
        return False

    def on_escape(self) -> bool:
        if self.label.current_annotation:
            self.discard()
            return True
        return False

    def paint_overlay(self, painter) -> None:
        if not self.label.current_annotation:
            return
        painter.save()
        painter.translate(self.label.offset_x, self.label.offset_y)
        painter.scale(self.label.zoom_factor, self.label.zoom_factor)

        zf = self.label.zoom_factor
        painter.setPen(QPen(Qt.GlobalColor.red, 2 / zf, Qt.PenStyle.SolidLine))
        points = [QPointF(float(x), float(y)) for x, y in self.label.current_annotation]
        if len(points) > 1:
            painter.drawPolyline(QPolygonF(points))
        for point in points:
            painter.drawEllipse(point, 5 / zf, 5 / zf)
        if self.label.temp_point:
            painter.drawLine(
                points[-1],
                QPointF(float(self.label.temp_point[0]), float(self.label.temp_point[1])),
            )
        painter.restore()

    def has_unsaved_state(self) -> bool:
        return self.label.drawing_polygon and len(self.label.current_annotation) > 0

    def commit(self) -> None:
        if self.has_unsaved_state() and len(self.label.current_annotation) > 2:
            self.label.drawing_polygon = False
            self.label.finishPolygonRequested.emit()

    def discard(self) -> None:
        self.label.current_annotation = []
        self.label.temp_point = None
        self.label.drawing_polygon = False
