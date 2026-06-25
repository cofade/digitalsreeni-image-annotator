"""Global shortcuts and application-wide event filters for ImageAnnotator.

Both pieces were inline init blocks in ImageAnnotator.__init__ before
Phase 8; factored out here for symmetry with the other ui/ builders
and so the orchestrator stays focused on wiring.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication

from ..controllers.dino_controller import DINOReviewEventFilter


def install_shortcuts(window):
    """Register global keyboard shortcuts. Registered as QShortcuts with
    ApplicationShortcut context so they fire regardless of which widget has
    focus — putting them in keyPressEvent didn't work because QTableWidget
    (the annotation list / DINO threshold table) and other focusable children
    consume the keys before they bubble up to the main window."""
    window._snake_shortcut = QShortcut(QKeySequence("F2"), window)
    window._snake_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
    window._snake_shortcut.activated.connect(window.launch_snake_game)

    # Undo / redo of annotation edits (ADR-026). Ctrl+Z, plus Ctrl+Y and
    # Ctrl+Shift+Z as cross-platform redo aliases.
    ac = window.annotation_controller
    window._undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, window)
    window._undo_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
    window._undo_shortcut.activated.connect(ac.undo)

    window._redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, window)
    window._redo_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
    window._redo_shortcut.activated.connect(ac.redo)

    window._redo_shortcut_alt = QShortcut(QKeySequence("Ctrl+Shift+Z"), window)
    window._redo_shortcut_alt.setContext(Qt.ShortcutContext.ApplicationShortcut)
    window._redo_shortcut_alt.activated.connect(ac.redo)


def install_event_filters(window):
    """Install application-wide event filters. Currently just the DINO
    review filter — Enter/Escape for DINO temp_annotations need to
    work even when focus is on slice_list / image_list / a button,
    none of which forward the key to ImageLabel.keyPressEvent. See
    ADR-015."""
    window._dino_review_filter = DINOReviewEventFilter(window)
    QApplication.instance().installEventFilter(window._dino_review_filter)
