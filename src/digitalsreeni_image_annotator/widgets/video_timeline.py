"""Video timeline scrub bar with annotated-frame markers (issue #48).

A dumb, self-contained widget with **no main-window reference**: it renders a
horizontal scrub slider, a thin marker strip that ticks every annotated frame
plus the current one, and a ``F i/N  •  MM:SS / MM:SS`` position label. It is a
pure VIEW — it never changes any frame itself. User interaction emits
``frameSelected(idx)`` and the orchestrator routes that through the normal
``ImageController.switch_slice`` path; programmatic sync comes back via
``set_current_frame`` (which must NOT re-emit ``frameSelected``, or it would
re-enter ``switch_slice``).

Colours come exclusively from the widget PALETTE (``highlight`` / ``mid`` /
``text``), never colour literals, so the marks stay visible in both the light
and soft-dark themes (No Hardcoded Colors Rule, CLAUDE.md).

A follow-up (issue #51) may extend ``set_annotated_frames`` to carry per-frame
states (e.g. reviewed vs. auto-accepted) — kept deliberately simple here.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# fps guard: containers sometimes report 0/NaN fps; treat as 30 for MM:SS.
_DEFAULT_FPS = 30.0


class _MarkerStrip(QWidget):
    """Thin painted strip that ticks annotated frames + the current frame.

    Holds only its own render state (total / current / annotated set); the
    parent :class:`VideoTimeline` pushes updates via :meth:`configure`.
    """

    _STRIP_HEIGHT = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self._STRIP_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._total = 0
        self._current = 0
        self._annotated = set()

    def configure(self, total, current, annotated):
        self._total = total
        self._current = current
        self._annotated = annotated
        self.update()

    def _x_for(self, idx, width):
        # Guard total <= 1: a single (or zero) frame maps to the left edge
        # rather than dividing by zero.
        if self._total <= 1:
            return 0
        return int(idx / (self._total - 1) * (width - 1))

    def paintEvent(self, event):
        painter = QPainter(self)
        pal = self.palette()
        width = self.width()
        height = self.height()

        # Track baseline (mid) — theme-aware grey.
        painter.setPen(QPen(pal.mid().color(), 1))
        mid_y = height // 2
        painter.drawLine(0, mid_y, width, mid_y)

        if self._total <= 0:
            return

        # Annotated frames: 1px highlight (accent) ticks spanning the strip.
        painter.setPen(QPen(pal.highlight().color(), 1))
        for idx in self._annotated:
            if 0 <= idx < self._total:
                x = self._x_for(idx, width)
                painter.drawLine(x, 0, x, height)

        # Current frame: a distinct, high-contrast 2px tick. `text()` always
        # contrasts the widget background in both themes, so it reads clearly
        # over the accent-coloured annotated marks.
        if 0 <= self._current < self._total:
            painter.setPen(QPen(pal.text().color(), 2))
            x = self._x_for(self._current, width)
            painter.drawLine(x, 0, x, height)


class VideoTimeline(QWidget):
    """Scrub slider + annotated-frame markers + position label (issue #48).

    Public API (relied on by the wiring in :mod:`annotator_window` and by the
    #51 follow-up):

    - ``set_video(total_frames: int, fps: float)``
    - ``set_current_frame(idx: int)`` — programmatic, never emits ``frameSelected``
    - ``set_annotated_frames(indices: set[int])``
    - ``clear()``
    - signal ``frameSelected = pyqtSignal(int)`` — user interaction only
    - attribute ``annotated_frames`` — the stored annotated-index set
    """

    frameSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._total_frames = 0
        self._fps = _DEFAULT_FPS
        self._current_frame = 0
        # Re-entrancy guard: set True around programmatic slider.setValue so the
        # resulting valueChanged never re-emits frameSelected (feedback loop).
        self._updating = False
        self.annotated_frames = set()

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # Strip stacked directly on top of the slider so ticks line up with the
        # groove.
        bar = QVBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(0)

        self.strip = _MarkerStrip()
        bar.addWidget(self.strip)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        # Never swallow the arrow keys used for slice navigation.
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        bar.addWidget(self.slider)

        root.addLayout(bar, 1)

        self.label = QLabel()
        root.addWidget(self.label)

        # valueChanged covers keyboard/track-click/programmatic-user moves;
        # gated on `not _updating` (no feedback loop) and `not isSliderDown`
        # (throttle live scrub — each switch triggers a lazy frame decode, so we
        # emit the drag's final position via sliderReleased instead of every
        # intermediate frame).
        self.slider.valueChanged.connect(self._on_value_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)

        self._update_label()

    # ── configuration ───────────────────────────────────────────────────────

    def set_video(self, total_frames, fps):
        """Configure for a video of ``total_frames`` frames at ``fps``.

        Resets the marker set and the slider range (0..max(0, total-1)); the fps
        guard keeps MM:SS finite when a container reports 0/NaN fps.
        """
        self._total_frames = max(0, int(total_frames))
        self._fps = fps if fps and fps > 0 else _DEFAULT_FPS
        self._current_frame = 0
        self.annotated_frames = set()

        self._updating = True
        self.slider.setMaximum(max(0, self._total_frames - 1))
        self.slider.setValue(0)
        self._updating = False

        self._refresh_strip()
        self._update_label()

    def set_current_frame(self, idx):
        """Programmatically sync the slider/label/strip to frame ``idx``.

        MUST NOT emit ``frameSelected`` — this is called from ``switch_slice``
        and emitting would re-enter it (feedback loop). The ``_updating`` guard
        suppresses the valueChanged emission.
        """
        self._current_frame = int(idx)
        self._updating = True
        self.slider.setValue(self._current_frame)
        self._updating = False
        self._refresh_strip()
        self._update_label()

    def set_annotated_frames(self, indices):
        """Store the annotated-frame index set and repaint the strip.

        Kept simple (a plain set of ints); issue #51 may extend this to carry
        per-frame states.
        """
        self.annotated_frames = set(indices)
        self._refresh_strip()

    def clear(self):
        """Reset to the empty state (no video)."""
        self._total_frames = 0
        self._fps = _DEFAULT_FPS
        self._current_frame = 0
        self.annotated_frames = set()
        self._updating = True
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self._updating = False
        self._refresh_strip()
        self._update_label()

    # ── user-interaction handlers ───────────────────────────────────────────

    def _on_value_changed(self, value):
        # Suppress programmatic syncs (feedback loop) and live-drag intermediate
        # steps (throttle — sliderReleased emits the final drag position).
        if self._updating or self.slider.isSliderDown():
            return
        self._emit(value)

    def _on_slider_released(self):
        self._emit(self.slider.value())

    def _emit(self, value):
        self._current_frame = int(value)
        self._refresh_strip()
        self._update_label()
        self.frameSelected.emit(int(value))

    # ── rendering helpers ───────────────────────────────────────────────────

    def _refresh_strip(self):
        self.strip.configure(
            self._total_frames, self._current_frame, self.annotated_frames
        )

    def _fmt_time(self, idx):
        fps = self._fps if self._fps and self._fps > 0 else _DEFAULT_FPS
        minutes, seconds = divmod(int(idx / fps), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _update_label(self):
        if self._total_frames <= 0:
            self.label.setText("")
            return
        last = self._total_frames - 1
        self.label.setText(
            f"F {self._current_frame + 1}/{self._total_frames}  •  "
            f"{self._fmt_time(self._current_frame)} / {self._fmt_time(last)}"
        )
