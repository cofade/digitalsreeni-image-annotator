"""Global shortcuts and application-wide event filters for ImageAnnotator.

Both pieces were inline init blocks in ImageAnnotator.__init__ before
Phase 8; factored out here for symmetry with the other ui/ builders
and so the orchestrator stays focused on wiring.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication

from ..controllers.dino_controller import _DINOReviewEventFilter


def install_shortcuts(window):
    """Register global keyboard shortcuts. Currently just F2 → Snake
    game. Registered as a QShortcut with ApplicationShortcut context
    so it fires regardless of which widget has focus — putting it in
    keyPressEvent didn't work because QTableWidget (DINO threshold
    table) and other focusable children consume F2 before it bubbles
    up to the main window."""
    window._snake_shortcut = QShortcut(QKeySequence("F2"), window)
    window._snake_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
    window._snake_shortcut.activated.connect(window.launch_snake_game)


def install_event_filters(window):
    """Install application-wide event filters. Currently just the DINO
    review filter — Enter/Escape for DINO temp_annotations need to
    work even when focus is on slice_list / image_list / a button,
    none of which forward the key to ImageLabel.keyPressEvent. See
    ADR-015."""
    window._dino_review_filter = _DINOReviewEventFilter(window)
    QApplication.instance().installEventFilter(window._dino_review_filter)
