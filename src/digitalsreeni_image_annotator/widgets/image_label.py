"""
ImageLabel module for the Image Annotator application.

This module contains the ImageLabel class, which is responsible for
displaying the image and handling annotation interactions.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import os
import warnings

from PIL import Image
from PyQt6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PyQt6.QtWidgets import QLabel, QMessageBox

from .tools import EraserTool, PaintBrushTool, PolygonTool, RectangleTool
from ..core.constants import DEFAULT_FILL_OPACITY
from ..utils import calculate_area

warnings.filterwarnings("ignore", category=UserWarning)


class ImageLabel(QLabel):
    """
    A custom QLabel for displaying images and handling annotations.
    """

    # Annotation lifecycle
    annotationCommitted = pyqtSignal(dict)              # paint / accept-temp per-annotation add
    annotationsBatchSaved = pyqtSignal()                # batch finalizer: save + slice-color refresh
    annotationsReplaced = pyqtSignal(str, dict)         # eraser path: (image_key, per-class dict)
    annotationListUpdateRequested = pyqtSignal()        # editing-mode exit refresh
    annotationSelected = pyqtSignal(object)             # double-click selection
    canvasSelectionChanged = pyqtSignal(object, str)    # (list[annotation], mode); mode: replace|add|toggle
    deleteSelectionRequested = pyqtSignal()
    finishPolygonRequested = pyqtSignal()
    finishRectangleRequested = pyqtSignal()

    # Class
    classRequested = pyqtSignal(str)                    # accept-temp path needs a new class

    # SAM
    samPredictionRequested = pyqtSignal()               # debounced (mouse press)
    samPredictionApplyRequested = pyqtSignal()          # post-debounce (mouse release)
    samPredictionAccepted = pyqtSignal()                # Enter on temp prediction
    samPointsCleared = pyqtSignal()                     # Escape during sam_points: stop timer

    # Tool / UI state
    enableToolsRequested = pyqtSignal()
    disableToolsRequested = pyqtSignal()
    resetToolButtonsRequested = pyqtSignal()
    toolSizeChanged = pyqtSignal(str, int)              # ("paint" | "eraser", new_size)

    # Navigation / info
    zoomInRequested = pyqtSignal()
    zoomOutRequested = pyqtSignal()
    imageInfoChanged = pyqtSignal()

    # Selection highlight: a semi-transparent selection-blue, drawn as the
    # dashed bounding-box marquee (handles use the opaque variant). Class-
    # colour-independent, so it never vanishes the way the old red-on-red
    # highlight did.
    _SELECTION_COLOR = QColor(0, 120, 215, 220)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annotations = {}
        self.current_annotation = []
        self.temp_point = None
        self.current_tool = None
        self.zoom_factor = 1.0
        # Low-vision UI zoom: ui_font_pt / 10 (legacy default), set by
        # theme.apply_theme_and_font. Multiplies overlay sizes (label
        # fonts, marker radii, pen widths) — orthogonal to zoom_factor,
        # which keeps them constant-size on screen across image zoom.
        self.ui_scale = 1.0
        self.class_colors = {}
        self.class_visibility = {}
        self.start_point = None
        self.end_point = None
        self.highlighted_annotations = []
        # Idle-mode mask selection (issue #75): drag a rubber band to box-select.
        # selection_rect is (x0, y0, x1, y1) in image coords while a drag is live.
        self.selection_origin = None
        self.selecting = False
        self.selection_rect = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.pan_start_pos = None
        self._ctx = None
        self.offset_x = 0
        self.offset_y = 0
        self.drawing_polygon = False
        self.editing_polygon = None
        self.editing_point_index = None
        self.hover_point_index = None
        self.fill_opacity = DEFAULT_FILL_OPACITY
        self.drawing_rectangle = False
        self.current_rectangle = None
        self.bit_depth = None
        self.image_path = None
        self.dark_mode = False

        self.temp_paint_mask = None
        self.is_painting = False
        self.temp_eraser_mask = None
        self.is_erasing = False
        self.cursor_pos = None

        # SAM
        self.sam_bbox = None
        self.drawing_sam_bbox = False
        self.temp_sam_prediction = None
        self.temp_annotations = []

        # --- New for SAM modes ---
        self.sam_box_active = False
        self.sam_points_active = False
        self.sam_positive_points = []
        self.sam_negative_points = []

        # Per-tool handlers (Phase 7). Each owns its event-handling
        # behaviour; state fields used by controllers (current_rectangle,
        # current_annotation, temp_paint_mask, …) stay on the widget.
        self._tools = {
            "polygon":     PolygonTool(self),
            "rectangle":   RectangleTool(self),
            "paint_brush": PaintBrushTool(self),
            "eraser":      EraserTool(self),
        }

    def set_context(self, ctx):
        self._ctx = ctx

    def set_ui_scale(self, scale):
        self.ui_scale = scale
        self.update()

    def _pen_w(self, base):
        """Overlay pen width: ui-scaled, zoom-compensated (constant on screen)."""
        return base * self.ui_scale / self.zoom_factor

    def _overlay_font(self, base=12):
        """Overlay label font: ui-scaled, zoom-compensated (constant on screen)."""
        return QFont("Arial", max(1, int(base * self.ui_scale / self.zoom_factor)))

    @property
    def active_tool_handler(self):
        return self._tools.get(self.current_tool)

    def set_active_tool(self, tool_name):
        """Called by ImageAnnotator when the user switches tools. Gives
        the previous handler a chance to clean up (default no-op
        preserves the existing 'drop temp state silently' behaviour;
        explicit commit/discard goes through Enter/Escape or the
        check_unsaved_changes dialog)."""
        prev = self.active_tool_handler
        new = self._tools.get(tool_name)
        if prev is not None and prev is not new:
            prev.deactivate()
        self.current_tool = tool_name

    def set_dark_mode(self, is_dark):
        self.dark_mode = is_dark
        self.update()

    def setPixmap(self, pixmap):
        """Set the pixmap and update the scaled version."""
        if isinstance(pixmap, QImage):
            pixmap = QPixmap.fromImage(pixmap)
        self.original_pixmap = pixmap
        self.update_scaled_pixmap()

    def detect_bit_depth(self):
        """Detect and store the actual image bit depth using PIL."""
        if self.image_path and os.path.exists(self.image_path):
            with Image.open(self.image_path) as img:
                if img.mode == "1":
                    self.bit_depth = 1
                elif img.mode == "L":
                    self.bit_depth = 8
                elif img.mode == "I;16":
                    self.bit_depth = 16
                elif img.mode in ["RGB", "HSV"]:
                    self.bit_depth = 24
                elif img.mode in ["RGBA", "CMYK"]:
                    self.bit_depth = 32
                else:
                    self.bit_depth = img.bits

                self.imageInfoChanged.emit()

    def update_scaled_pixmap(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            scaled_size = self.original_pixmap.size() * self.zoom_factor
            self.scaled_pixmap = self.original_pixmap.scaled(
                scaled_size.width(),
                scaled_size.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            super().setPixmap(self.scaled_pixmap)
            self.setMinimumSize(self.scaled_pixmap.size())
            self.update_offset()
        else:
            self.scaled_pixmap = None
            super().setPixmap(QPixmap())
            self.setMinimumSize(QSize(0, 0))

    def update_offset(self):
        """Update the offset for centered image display."""
        if self.scaled_pixmap:
            self.offset_x = int((self.width() - self.scaled_pixmap.width()) / 2)
            self.offset_y = int((self.height() - self.scaled_pixmap.height()) / 2)

    def reset_annotation_state(self):
        """Reset the annotation state."""
        self.temp_point = None
        self.start_point = None
        self.end_point = None
        # Drop any in-progress rubber-band so a stale rect can't render on
        # the next image/slice (switch_image/switch_slice call through here).
        self.selection_origin = None
        self.selecting = False
        self.selection_rect = None

    def clear_current_annotation(self):
        """Clear the current annotation."""
        self.current_annotation = []

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update_offset()

    # Paint, eraser, polygon, and rectangle behaviour lives in
    # widgets/tools/*; this widget dispatches events to the active
    # handler (see set_active_tool / active_tool_handler).

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.scaled_pixmap:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # Draw the image
            painter.drawPixmap(
                int(self.offset_x), int(self.offset_y), self.scaled_pixmap
            )
            # Draw committed annotations
            self.draw_annotations(painter)
            # Polygon edit mode is modal; runs orthogonal to tool selection
            if self.editing_polygon:
                self.draw_editing_polygon(painter)
            # Idle-mode rubber-band selection rectangle (issue #75)
            if self.selection_rect is not None:
                self.draw_selection_rect(painter)
            # SAM overlays (cross-cutting; not part of the tool handlers)
            if self.sam_box_active and self.sam_bbox:
                self.draw_sam_bbox(painter)
            if self.sam_points_active:
                painter.save()
                painter.translate(self.offset_x, self.offset_y)
                painter.scale(self.zoom_factor, self.zoom_factor)
                # Radii intentionally NOT zoom-compensated — the dots
                # grow with image zoom (pre-existing behaviour).
                dot_r = 4 * self.ui_scale
                for pt in self.sam_positive_points:
                    painter.setPen(QPen(Qt.GlobalColor.green, self._pen_w(6), Qt.PenStyle.SolidLine))
                    painter.setBrush(QBrush(Qt.GlobalColor.green))
                    painter.drawEllipse(QPointF(pt[0], pt[1]), dot_r, dot_r)
                for pt in self.sam_negative_points:
                    painter.setPen(QPen(Qt.GlobalColor.red, self._pen_w(6), Qt.PenStyle.SolidLine))
                    painter.setBrush(QBrush(Qt.GlobalColor.red))
                    painter.drawEllipse(QPointF(pt[0], pt[1]), dot_r, dot_r)
                painter.restore()
            # In-progress overlays from every tool that has state to
            # render (paint mask, eraser mask, polygon-in-progress,
            # rectangle preview). Pre-Phase-7 these drew whenever
            # their state field was populated regardless of the active
            # tool; iterating all handlers preserves that — switching
            # tools mid-stroke does not hide an unsaved mark.
            for handler in self._tools.values():
                handler.paint_overlay(painter)
            self.draw_tool_size_indicator(painter)
            if self.temp_annotations:
                self.draw_temp_annotations(painter)
            painter.end()

    def draw_temp_annotations(self, painter):
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        for annotation in self.temp_annotations:
            color = QColor(255, 165, 0, 128)  # Semi-transparent orange
            painter.setPen(QPen(color, self._pen_w(2), Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(color))

            # Prefer segmentation polygon over bbox when both are present
            # (DINO+SAM temp annotations carry both — the polygon is the mask).
            points = None
            if "segmentation" in annotation:
                points = [
                    QPointF(float(x), float(y))
                    for x, y in zip(
                        annotation["segmentation"][0::2],
                        annotation["segmentation"][1::2],
                    )
                ]
                painter.drawPolygon(QPolygonF(points))
            elif "bbox" in annotation:
                x, y, w, h = annotation["bbox"]
                painter.drawRect(QRectF(x, y, w, h))

            # Draw label and score
            painter.setFont(self._overlay_font())
            label = f"{annotation['category_name']} {annotation['score']:.2f}"
            if points is not None:
                centroid = self.calculate_centroid(points)
                if centroid:
                    painter.drawText(centroid, label)
            elif "bbox" in annotation:
                x, y, _, _ = annotation["bbox"]
                painter.drawText(QPointF(x, y - 5), label)

        painter.restore()

    def accept_temp_annotations(self):
        for annotation in self.temp_annotations:
            class_name = annotation["category_name"]

            # Check if the class exists, if not, add it
            if class_name not in self._ctx.class_mapping():
                self.classRequested.emit(class_name)

            if class_name not in self.annotations:
                self.annotations[class_name] = []

            del annotation["temp"]
            del annotation[
                "score"
            ]  # Remove the score as it's not needed in the final annotation
            self.annotations[class_name].append(annotation)
            self.annotationCommitted.emit(annotation)

        self.temp_annotations.clear()
        self.annotationsBatchSaved.emit()
        self.update()

    def discard_temp_annotations(self):
        self.temp_annotations.clear()
        self.update()

    def draw_tool_size_indicator(self, painter):
        if self.current_tool in ["paint_brush", "eraser"] and hasattr(
            self, "cursor_pos"
        ):
            painter.save()
            painter.translate(self.offset_x, self.offset_y)
            painter.scale(self.zoom_factor, self.zoom_factor)

            if self.current_tool == "paint_brush":
                size = self._ctx.paint_brush_size()
                color = QColor(255, 0, 0, 128)  # Semi-transparent red
            else:  # eraser
                size = self._ctx.eraser_size()
                color = QColor(0, 0, 255, 128)  # Semi-transparent blue

            # Draw filled circle with lower opacity
            painter.setOpacity(0.3)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(
                QPointF(self.cursor_pos[0], self.cursor_pos[1]), size, size
            )

            # Draw circle outline with full opacity
            painter.setOpacity(1.0)
            painter.setPen(QPen(color.darker(150), self._pen_w(1), Qt.PenStyle.SolidLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(
                QPointF(self.cursor_pos[0], self.cursor_pos[1]), size, size
            )

            # Draw size text
            # Reset the transform to ensure text is drawn at screen coordinates
            painter.resetTransform()
            font = QFont()
            # Screen-space text (transform was reset above): scale with
            # the UI font setting only, not with image zoom.
            font.setPointSize(max(1, int(10 * self.ui_scale)))
            painter.setFont(font)
            painter.setPen(QPen(Qt.GlobalColor.black))  # Use black color for better visibility

            # Convert cursor position back to screen coordinates
            screen_x = self.cursor_pos[0] * self.zoom_factor + self.offset_x
            screen_y = self.cursor_pos[1] * self.zoom_factor + self.offset_y

            # Position text above the circle
            text_rect = QRectF(
                screen_x + (size * self.zoom_factor),
                screen_y - (size * self.zoom_factor),
                100,
                20,
            )

            text = f"Size: {size}"
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)

            painter.restore()

    def draw_sam_bbox(self, painter):
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)
        painter.setPen(QPen(Qt.GlobalColor.red, self._pen_w(2), Qt.PenStyle.SolidLine))
        x1, y1, x2, y2 = self.sam_bbox
        painter.drawRect(QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
        painter.restore()

    def draw_selection_rect(self, painter):
        """Draw the idle-mode rubber-band selection rectangle (issue #75).

        A single dashed selection-blue rect with a faint fill — same restrained
        style as the selection outline (not red, which clashes with class colours)."""
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)
        x0, y0, x1, y1 = self.selection_rect
        rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        fill = QColor(self._SELECTION_COLOR)
        fill.setAlphaF(0.10)
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(self._SELECTION_COLOR, self._pen_w(1), Qt.PenStyle.DashLine))
        painter.drawRect(rect)
        painter.restore()

    def clear_temp_sam_prediction(self):
        self.temp_sam_prediction = None
        self.update()

    def check_unsaved_changes(self):
        dirty = [t for t in self._tools.values() if t.has_unsaved_state()]
        if not dirty:
            return True
        reply = QMessageBox.question(
            self._ctx.dialog_parent(),
            "Unsaved Changes",
            "You have unsaved changes. Do you want to save them?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Yes:
            for t in dirty:
                t.commit()
            return True
        if reply == QMessageBox.StandardButton.No:
            for t in dirty:
                t.discard()
            return True
        return False  # Cancel

    def clear(self):
        super().clear()
        self.annotations.clear()
        self.current_annotation.clear()
        self.temp_point = None
        self.current_tool = None
        self.start_point = None
        self.end_point = None
        self.highlighted_annotations.clear()
        self.selection_origin = None
        self.selecting = False
        self.selection_rect = None
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.editing_polygon = None
        self.editing_point_index = None
        self.hover_point_index = None
        self.current_rectangle = None
        self.sam_bbox = None
        self.temp_sam_prediction = None
        self.update()

    def set_class_visibility(self, class_name, is_visible):
        self.class_visibility[class_name] = is_visible

    def draw_annotations(self, painter):
        """Draw all annotations on the image."""
        if not self.original_pixmap:
            return

        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        for class_name, class_annotations in self.annotations.items():
            if not self._ctx.is_class_visible(class_name):
                continue

            color = self.class_colors.get(class_name, QColor(Qt.GlobalColor.white))
            for annotation in class_annotations:
                # Selection no longer recolours the mask (it used to turn red,
                # which was invisible on a red-class mask). The mask always
                # keeps its class colour; selection is drawn as a
                # class-colour-independent overlay in a final pass below.
                border_color = color
                fill_color = QColor(color)
                fill_color.setAlphaF(self.fill_opacity)

                text_color = Qt.GlobalColor.white if self.dark_mode else Qt.GlobalColor.black
                painter.setPen(QPen(border_color, self._pen_w(2), Qt.PenStyle.SolidLine))
                painter.setBrush(QBrush(fill_color))

                if "segmentation" in annotation:
                    segmentation = annotation["segmentation"]
                    if isinstance(segmentation, list) and len(segmentation) > 0:
                        if isinstance(segmentation[0], list):  # Multiple polygons
                            for polygon in segmentation:
                                points = [
                                    QPointF(float(x), float(y))
                                    for x, y in zip(polygon[0::2], polygon[1::2])
                                ]
                                if points:
                                    painter.drawPolygon(QPolygonF(points))
                        else:  # Single polygon
                            points = [
                                QPointF(float(x), float(y))
                                for x, y in zip(segmentation[0::2], segmentation[1::2])
                            ]
                            if points:
                                painter.drawPolygon(QPolygonF(points))

                        # Draw centroid and label
                        if points:
                            centroid = self.calculate_centroid(points)
                            if centroid:
                                painter.setFont(self._overlay_font())
                                painter.setPen(
                                    QPen(text_color, self._pen_w(2), Qt.PenStyle.SolidLine)
                                )
                                painter.drawText(
                                    centroid,
                                    f"{class_name} {annotation.get('number', '')}",
                                )

                elif "bbox" in annotation:
                    x, y, width, height = annotation["bbox"]
                    painter.drawRect(QRectF(x, y, width, height))
                    painter.setPen(QPen(text_color, self._pen_w(2), Qt.PenStyle.SolidLine))
                    painter.drawText(
                        QPointF(x, y), f"{class_name} {annotation.get('number', '')}"
                    )

        # Polygon-in-progress is rendered by PolygonTool.paint_overlay
        # (paintEvent calls active_tool_handler.paint_overlay).

        # Draw temporary SAM prediction
        if self.temp_sam_prediction:
            temp_color = QColor(255, 165, 0, 128)  # Semi-transparent orange
            painter.setPen(QPen(temp_color, self._pen_w(2), Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(temp_color))

            segmentation = self.temp_sam_prediction["segmentation"]
            points = [
                QPointF(float(x), float(y))
                for x, y in zip(segmentation[0::2], segmentation[1::2])
            ]
            if points:
                painter.drawPolygon(QPolygonF(points))
                centroid = self.calculate_centroid(points)
                if centroid:
                    painter.setFont(self._overlay_font())
                    painter.drawText(
                        centroid, f"SAM: {self.temp_sam_prediction['score']:.2f}"
                    )

        # Selection overlay — drawn LAST so it sits on top of every mask's
        # fill, and in a class-colour-independent style so it's recognisable
        # regardless of the selected mask's colour (issue #75 follow-up).
        for annotation in self.highlighted_annotations:
            self._draw_selection_overlay(painter, annotation)

        painter.restore()

    def _draw_selection_overlay(self, painter, annotation):
        """Mark a selected annotation the way the sibling open-garden-planner
        app does: a dashed selection-blue bounding box plus bright square
        handles at the 4 corners and 4 edge midpoints. Class-colour-independent
        and clearly visible regardless of the mask's own colour."""
        if not self._is_class_pickable(annotation.get("category_name")):
            return  # don't draw selection chrome over a hidden mask
        bb = self._annotation_bbox(annotation)
        if bb is None:
            return
        x0, y0, x1, y1 = bb
        rect = QRectF(x0, y0, x1 - x0, y1 - y0)

        # Dashed bounding-box marquee.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(self._SELECTION_COLOR, self._pen_w(1.5), Qt.PenStyle.DashLine))
        painter.drawRect(rect)

        # Handle squares — opaque blue with a white casing so they read on any
        # background; fixed on-screen size (zoom-compensated). Visual selection
        # markers; resizing via handles is a separate feature (upstream #40).
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        handles = [
            (x0, y0), (cx, y0), (x1, y0),
            (x0, cy), (x1, cy),
            (x0, y1), (cx, y1), (x1, y1),
        ]
        half = 4 * self.ui_scale / self.zoom_factor
        painter.setPen(QPen(Qt.GlobalColor.white, self._pen_w(1), Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(QColor(0, 120, 215)))
        for hx, hy in handles:
            painter.drawRect(QRectF(hx - half, hy - half, 2 * half, 2 * half))

    def draw_editing_polygon(self, painter):
        """Draw the polygon being edited."""
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)

        points = [
            QPointF(float(x), float(y))
            for x, y in zip(
                self.editing_polygon["segmentation"][0::2],
                self.editing_polygon["segmentation"][1::2],
            )
        ]
        color = self.class_colors.get(
            self.editing_polygon["category_name"], QColor(Qt.GlobalColor.white)
        )
        fill_color = QColor(color)
        fill_color.setAlphaF(self.fill_opacity)

        painter.setPen(QPen(color, self._pen_w(2), Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(fill_color))
        painter.drawPolygon(QPolygonF(points))  # Changed QPolygon to QPolygonF - Sreeni

        for i, point in enumerate(points):
            if i == self.hover_point_index:
                painter.setBrush(QColor(255, 0, 0))
            else:
                painter.setBrush(QColor(0, 255, 0))
            r = 5 * self.ui_scale / self.zoom_factor
            painter.drawEllipse(point, r, r)

        painter.restore()

    def calculate_centroid(self, points):
        """Calculate the centroid of a polygon."""
        if not points:
            return None
        x_coords = [point.x() for point in points]
        y_coords = [point.y() for point in points]
        centroid_x = sum(x_coords) / len(points)
        centroid_y = sum(y_coords) / len(points)
        return QPointF(centroid_x, centroid_y)

    def set_zoom(self, zoom_factor):
        """Set the zoom factor and update the display."""
        self.zoom_factor = zoom_factor
        self.update_scaled_pixmap()
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if not self.original_pixmap or not self.scaled_pixmap:
                event.accept()
                return

            cursor_widget_pos = event.position()
            # Image-space coords of the pixel under the cursor BEFORE zoom.
            img_x = (cursor_widget_pos.x() - self.offset_x) / self.zoom_factor
            img_y = (cursor_widget_pos.y() - self.offset_y) / self.zoom_factor

            scroll_area = self._ctx.scroll_area()
            scrollbar_h = scroll_area.horizontalScrollBar()
            scrollbar_v = scroll_area.verticalScrollBar()
            old_scroll_h = scrollbar_h.value()
            old_scroll_v = scrollbar_v.value()

            delta = event.angleDelta().y()
            if delta > 0:
                self.zoomInRequested.emit()
            else:
                self.zoomOutRequested.emit()

            # Compute the post-zoom offset analytically from the
            # viewport size and the new scaled-pixmap size. Reading
            # self.offset_x here is unreliable on zoom-OUT: setMinimumSize
            # in update_scaled_pixmap only relaxes the minimum, so the
            # widget hasn't shrunk yet when update_offset ran. self.width()
            # is stale → offset_x is wrong → cursor drifts. The viewport
            # width is always current.
            viewport = scroll_area.viewport()
            new_scaled_w = self.scaled_pixmap.width()
            new_scaled_h = self.scaled_pixmap.height()
            new_offset_x = max(0, (viewport.width() - new_scaled_w) / 2)
            new_offset_y = max(0, (viewport.height() - new_scaled_h) / 2)

            new_widget_x = img_x * self.zoom_factor + new_offset_x
            new_widget_y = img_y * self.zoom_factor + new_offset_y
            scrollbar_h.setValue(int(round(new_widget_x - cursor_widget_pos.x() + old_scroll_h)))
            scrollbar_v.setValue(int(round(new_widget_y - cursor_widget_pos.y() + old_scroll_v)))

            event.accept()
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if not self.original_pixmap:
            return
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.button() == Qt.MouseButton.LeftButton:
            # Track pan in global (screen) coords so the reference frame
            # doesn't shift when the scrollbar moves the widget under the
            # cursor — previously caused effective half-speed pan.
            self.pan_start_pos = event.globalPosition()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        pos = self.get_image_coordinates(event.position())

        # SAM points has priority over the rest (it accepts both
        # mouse buttons and short-circuits the tool dispatch).
        if self.current_tool == "sam_points" and self.sam_points_active:
            if event.button() == Qt.MouseButton.LeftButton:
                self.sam_positive_points.append(pos)
                self.update()
                self.samPredictionRequested.emit()
                return
            elif event.button() == Qt.MouseButton.RightButton:
                self.sam_negative_points.append(pos)
                self.update()
                self.samPredictionRequested.emit()
                return

        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool == "sam_box" and self.sam_box_active:
                self.sam_bbox = [pos[0], pos[1], pos[0], pos[1]]
                self.drawing_sam_bbox = True
            elif self.editing_polygon:
                self.handle_editing_click(pos, event)
            elif self._is_select_mode():
                # Idle-mode mask selection (issue #75): remember the press as
                # the potential rubber-band origin; a click vs. drag is
                # decided on move/release.
                self.selection_origin = pos
                self.selecting = False
                self.selection_rect = None
            else:
                handler = self.active_tool_handler
                if handler is not None:
                    handler.on_mouse_press(event, pos)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.original_pixmap:
            return
        self.cursor_pos = self.get_image_coordinates(event.position())
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.buttons() == Qt.MouseButton.LeftButton:
            if self.pan_start_pos:
                cur = event.globalPosition()
                delta = cur - self.pan_start_pos
                scroll_area = self._ctx.scroll_area()
                scrollbar_h = scroll_area.horizontalScrollBar()
                scrollbar_v = scroll_area.verticalScrollBar()
                scrollbar_h.setValue(scrollbar_h.value() - int(delta.x()))
                scrollbar_v.setValue(scrollbar_v.value() - int(delta.y()))
                self.pan_start_pos = cur
            event.accept()
            return

        pos = self.cursor_pos
        if (
            self.current_tool == "sam_box"
            and self.sam_box_active
            and self.drawing_sam_bbox
            and self.sam_bbox is not None
        ):
            self.sam_bbox[2] = pos[0]
            self.sam_bbox[3] = pos[1]
        elif self.editing_polygon:
            self.handle_editing_move(pos)
        elif self._is_select_mode() and self.selection_origin is not None and (
            event.buttons() & Qt.MouseButton.LeftButton
        ):
            self._update_selection_drag(pos)
        else:
            handler = self.active_tool_handler
            if handler is not None:
                handler.on_mouse_move(event, pos)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.original_pixmap:
            return
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.button() == Qt.MouseButton.LeftButton:
            self.pan_start_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            pos = self.get_image_coordinates(event.position())
            if event.button() == Qt.MouseButton.LeftButton:
                if (
                    self.sam_box_active
                    and self.drawing_sam_bbox
                    and self.sam_bbox is not None
                ):
                    self.sam_bbox[2] = pos[0]
                    self.sam_bbox[3] = pos[1]
                    self.drawing_sam_bbox = False
                    self.samPredictionApplyRequested.emit()
                elif self.editing_polygon:
                    self.editing_point_index = None
                elif self._is_select_mode() and self.selection_origin is not None:
                    self._finish_selection(pos, event)
                else:
                    handler = self.active_tool_handler
                    if handler is not None:
                        handler.on_mouse_release(event, pos)
            self.update()

    def mouseDoubleClickEvent(self, event):
        if not self.pixmap():
            return
        pos = self.get_image_coordinates(event.position())
        if event.button() == Qt.MouseButton.LeftButton:
            # Polygon handler can consume the double-click to finish
            # the polygon. If it doesn't (no in-progress polygon), fall
            # through to polygon-edit mode.
            handler = self.active_tool_handler
            consumed = False
            if handler is not None:
                consumed = handler.on_double_click(event, pos)
            if not consumed:
                self.clear_current_annotation()
                annotation = self.start_polygon_edit(pos)
                if annotation:
                    self.annotationSelected.emit(annotation)
        self.update()

    def get_image_coordinates(self, pos):
        if not self.scaled_pixmap:
            return (0, 0)
        x = (pos.x() - self.offset_x) / self.zoom_factor
        y = (pos.y() - self.offset_y) / self.zoom_factor
        return (int(x), int(y))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # DINO temp_annotations are accepted via the application-wide
            # DINOReviewEventFilter (see ADR-015) so Enter works regardless
            # of focus. The branch below only catches non-DINO temp state
            # (legacy YOLO model-prediction review path).
            if self.temp_annotations:
                self.accept_temp_annotations()
            elif self.temp_sam_prediction:
                self.samPredictionAccepted.emit()
            elif self.editing_polygon:
                self.editing_polygon = None
                self.editing_point_index = None
                self.hover_point_index = None
                self.enableToolsRequested.emit()
                self.annotationListUpdateRequested.emit()
            else:
                handler = self.active_tool_handler
                if handler is not None:
                    handler.on_enter()
        elif event.key() == Qt.Key.Key_Escape:
            if self.sam_points_active:
                self.samPointsCleared.emit()
                self.sam_positive_points = []
                self.sam_negative_points = []
                self.clear_temp_sam_prediction()
                self.update()
            # DINO temp_annotations are rejected via the application-wide
            # DINOReviewEventFilter (see ADR-015). Branch below catches
            # non-DINO temp state only.
            elif self.temp_annotations:
                self.discard_temp_annotations()
            elif self.sam_box_active:
                self.sam_bbox = None
                self.clear_temp_sam_prediction()
            elif self.editing_polygon:
                self.editing_polygon = None
                self.editing_point_index = None
                self.hover_point_index = None
                self.enableToolsRequested.emit()
            elif self._is_select_mode() and (
                self.selecting or self.selection_origin is not None
            ):
                # Cancel an in-progress rubber band (selection unchanged).
                self.selection_origin = None
                self.selecting = False
                self.selection_rect = None
            else:
                handler = self.active_tool_handler
                if handler is not None:
                    handler.on_escape()
        elif event.key() == Qt.Key.Key_Delete:
            if self.editing_polygon:
                self.deleteSelectionRequested.emit()
                self.editing_polygon = None
                self.editing_point_index = None
                self.hover_point_index = None
                self.enableToolsRequested.emit()
                self.update()
            elif (
                self._is_select_mode()
                and self._ctx is not None
                and self._ctx.has_annotation_selection()
            ):
                # Idle-mode canvas selection: delete the selected masks.
                # Gate on the annotation-list selection (the controller's
                # source of truth) rather than the red highlight, which a
                # list rebuild (e.g. a sort) can leave stale — otherwise a
                # canvas Delete would pop a spurious "nothing selected"
                # warning. See ADR-022.
                self.deleteSelectionRequested.emit()
        elif event.key() == Qt.Key.Key_Minus:
            if self.current_tool == "paint_brush":
                new_size = max(1, self._ctx.paint_brush_size() - 1)
                self.toolSizeChanged.emit("paint", new_size)
                print(f"Paint brush size: {new_size}")
            elif self.current_tool == "eraser":
                new_size = max(1, self._ctx.eraser_size() - 1)
                self.toolSizeChanged.emit("eraser", new_size)
                print(f"Eraser size: {new_size}")
        elif event.key() in (Qt.Key.Key_Equal, Qt.Key.Key_Plus):
            if self.current_tool == "paint_brush":
                new_size = self._ctx.paint_brush_size() + 1
                self.toolSizeChanged.emit("paint", new_size)
                print(f"Paint brush size: {new_size}")
            elif self.current_tool == "eraser":
                new_size = self._ctx.eraser_size() + 1
                self.toolSizeChanged.emit("eraser", new_size)
                print(f"Eraser size: {new_size}")
        self.update()

    # --- Idle-mode mask selection (issue #75) ---

    def _is_select_mode(self):
        """True when the canvas is idle (no drawing/SAM tool, not editing,
        no temp review) — the only state where bare clicks/drags select
        existing masks instead of drawing."""
        return (
            self.current_tool is None
            and not self.editing_polygon
            and not self.sam_box_active
            and not self.sam_points_active
            and not self.temp_annotations
            and not self.temp_sam_prediction
        )

    @staticmethod
    def _annotation_contains(annotation, pos):
        """Hit-test a single annotation (segmentation polygon or bbox)."""
        if "segmentation" in annotation:
            seg = annotation["segmentation"]
            points = [QPoint(int(x), int(y)) for x, y in zip(seg[0::2], seg[1::2])]
            return len(points) >= 3 and ImageLabel.point_in_polygon(pos, points)
        if "bbox" in annotation:
            x, y, w, h = annotation["bbox"]
            return x <= pos[0] <= x + w and y <= pos[1] <= y + h
        return False

    @staticmethod
    def _annotation_bbox(annotation):
        """Axis-aligned bounds (x0, y0, x1, y1) of an annotation, or None."""
        if "segmentation" in annotation:
            seg = annotation["segmentation"]
            xs, ys = seg[0::2], seg[1::2]
            if not xs or not ys:
                return None
            return (min(xs), min(ys), max(xs), max(ys))
        if "bbox" in annotation:
            x, y, w, h = annotation["bbox"]
            return (x, y, x + w, y + h)
        return None

    def _is_class_pickable(self, class_name):
        # No context (e.g. unit tests) → everything is pickable.
        return self._ctx is None or self._ctx.is_class_visible(class_name)

    def annotation_at(self, pos):
        """Smallest-area annotation containing pos, or None. Covers both
        segmentation and bbox annotations and skips hidden classes. Smallest
        wins so a mask nested inside another stays reachable (cf.
        start_polygon_edit / upstream #33)."""
        best = None
        best_area = None
        for class_name, annotations in self.annotations.items():
            if not self._is_class_pickable(class_name):
                continue
            for annotation in annotations:
                if self._annotation_contains(annotation, pos):
                    area = calculate_area(annotation)
                    if best is None or area < best_area:
                        best = annotation
                        best_area = area
        return best

    def annotations_in_rect(self, rect):
        """All annotations whose bounds intersect the rubber-band rect.
        rect is (x0, y0, x1, y1) in image coords (any corner order)."""
        x0, y0, x1, y1 = rect
        rx0, rx1 = min(x0, x1), max(x0, x1)
        ry0, ry1 = min(y0, y1), max(y0, y1)
        result = []
        for class_name, annotations in self.annotations.items():
            if not self._is_class_pickable(class_name):
                continue
            for annotation in annotations:
                bb = self._annotation_bbox(annotation)
                if bb is None:
                    continue
                ax0, ay0, ax1, ay1 = bb
                if ax0 <= rx1 and ax1 >= rx0 and ay0 <= ry1 and ay1 >= ry0:
                    result.append(annotation)
        return result

    def _update_selection_drag(self, pos):
        """Grow the rubber band once the drag clears the click threshold."""
        if self.selection_origin is None:
            return
        if not self.selecting:
            threshold = 3.0 / max(self.zoom_factor, 1e-6)
            if self.distance(pos, self.selection_origin) < threshold:
                return
            self.selecting = True
        ox, oy = self.selection_origin
        self.selection_rect = (ox, oy, pos[0], pos[1])

    def _finish_selection(self, pos, event):
        """Resolve a press→release in select mode into a selection change.
        Shift makes it additive (drag) / toggling (click)."""
        additive = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if self.selecting and self.selection_rect is not None:
            anns = self.annotations_in_rect(self.selection_rect)
            self.canvasSelectionChanged.emit(anns, "add" if additive else "replace")
        else:
            ann = self.annotation_at(pos)
            if additive:
                if ann is not None:  # Shift+click on empty space keeps the selection
                    self.canvasSelectionChanged.emit([ann], "toggle")
            else:
                self.canvasSelectionChanged.emit(
                    [ann] if ann is not None else [], "replace"
                )
        self.selection_origin = None
        self.selecting = False
        self.selection_rect = None

    def start_polygon_edit(self, pos):
        # Among all polygons containing the click, edit the smallest by
        # area so an annotation fully nested inside another is reachable
        # (upstream issue #33) instead of always grabbing the first/outer
        # match.
        best = None
        best_area = None
        for class_name, annotations in self.annotations.items():
            for annotation in annotations:
                if "segmentation" in annotation:
                    points = [
                        QPoint(int(x), int(y))
                        for x, y in zip(
                            annotation["segmentation"][0::2],
                            annotation["segmentation"][1::2],
                        )
                    ]
                    if self.point_in_polygon(pos, points):
                        area = calculate_area(annotation)
                        if best is None or area < best_area:
                            best = annotation
                            best_area = area
        if best is not None:
            self.editing_polygon = best
            self.current_tool = None
            self.disableToolsRequested.emit()
            self.resetToolButtonsRequested.emit()
            return best
        return None

    def handle_editing_click(self, pos, event):
        """Handle clicks during polygon editing."""
        points = [
            QPoint(int(x), int(y))
            for x, y in zip(
                self.editing_polygon["segmentation"][0::2],
                self.editing_polygon["segmentation"][1::2],
            )
        ]
        for i, point in enumerate(points):
            if self.distance(pos, point) < 10 * self.ui_scale / self.zoom_factor:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    # Delete point
                    del self.editing_polygon["segmentation"][i * 2 : i * 2 + 2]
                else:
                    # Start moving point
                    self.editing_point_index = i
                return
        # Add new point
        for i in range(len(points)):
            if self.point_on_line(pos, points[i], points[(i + 1) % len(points)]):
                self.editing_polygon["segmentation"][i * 2 + 2 : i * 2 + 2] = [
                    pos[0],
                    pos[1],
                ]
                self.editing_point_index = i + 1
                return

    def handle_editing_move(self, pos):
        """Handle mouse movement during polygon editing."""
        points = [
            QPoint(int(x), int(y))
            for x, y in zip(
                self.editing_polygon["segmentation"][0::2],
                self.editing_polygon["segmentation"][1::2],
            )
        ]
        self.hover_point_index = None
        for i, point in enumerate(points):
            if self.distance(pos, point) < 10 * self.ui_scale / self.zoom_factor:
                self.hover_point_index = i
                break
        if self.editing_point_index is not None:
            self.editing_polygon["segmentation"][self.editing_point_index * 2] = pos[0]
            self.editing_polygon["segmentation"][self.editing_point_index * 2 + 1] = (
                pos[1]
            )

    def exit_editing_mode(self):
        self.editing_polygon = None
        self.editing_point_index = None
        self.hover_point_index = None
        self.update()

    @staticmethod
    def point_in_polygon(point, polygon):
        """Check if a point is inside a polygon."""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0].x(), polygon[0].y()
        for i in range(n + 1):
            p2x, p2y = polygon[i % n].x(), polygon[i % n].y()
            if point[1] > min(p1y, p2y):
                if point[1] <= max(p1y, p2y):
                    if point[0] <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or point[0] <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    @staticmethod
    def point_to_tuple(point):
        """Convert QPoint to tuple."""
        if isinstance(point, QPoint):
            return (point.x(), point.y())
        return point

    @staticmethod
    def distance(p1, p2):
        """Calculate distance between two points."""
        p1 = ImageLabel.point_to_tuple(p1)
        p2 = ImageLabel.point_to_tuple(p2)
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    @staticmethod
    def point_on_line(p, start, end):
        """Check if a point is on a line segment."""
        p = ImageLabel.point_to_tuple(p)
        start = ImageLabel.point_to_tuple(start)
        end = ImageLabel.point_to_tuple(end)
        d1 = ImageLabel.distance(p, start)
        d2 = ImageLabel.distance(p, end)
        line_length = ImageLabel.distance(start, end)
        buffer = 0.1  # Adjust this value for more or less strict "on-line" detection
        return abs(d1 + d2 - line_length) < buffer
