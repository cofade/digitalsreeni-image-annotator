"""
Reusable widgets for LLM-assisted DINO detection:
- ClassThresholdTable : per-class thresholds (box / text / nms)
- PhraseEditorPanel   : phrase list per class

Ported from annotation_tool_v4.py and adapted for integration.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


_COL_NAME = 0
_COL_BOX = 1
_COL_TXT = 2
_COL_NMS = 3

DEFAULT_BOX_THR = 0.25
DEFAULT_TXT_THR = 0.25
DEFAULT_NMS_THR = 0.50


class ClassThresholdTable(QTableWidget):
    """
    A QTableWidget where each row = one class.
    Columns: Class Name | Box thr | Text thr | NMS thr
    The spinboxes are embedded directly in the cells.
    """

    def __init__(self, parent=None):
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(
            ["Class", "Box thr", "Txt thr", "NMS thr"])
        self.horizontalHeader().setSectionResizeMode(
            _COL_NAME, QHeaderView.ResizeMode.Stretch)
        # Fixed-width threshold columns — wide enough for "0,99" plus
        # spin arrows plus frame, with margin for macOS Retina font
        # metrics (where 72px clipped the down-arrow on some setups).
        for col in (_COL_BOX, _COL_TXT, _COL_NMS):
            self.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.Fixed)
            self.setColumnWidth(col, 88)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.setMaximumHeight(160)
        # No hardcoded background colors — pick them up from the active
        # stylesheet so the table integrates with both light and dark
        # mode. The earlier "background: #e0e0e0" produced a bright bar
        # across the top of the panel in dark mode.
        self.setStyleSheet(
            "QTableWidget { font-size: 11px; }"
            "QHeaderView::section { font-size: 11px; font-weight: bold; "
            "  padding: 2px; background-color: palette(mid); color: palette(text); }"
        )

    def _make_spin(self, value=0.25):
        sp = QDoubleSpinBox()
        sp.setRange(0.01, 0.99)
        sp.setSingleStep(0.05)
        sp.setDecimals(2)
        sp.setValue(value)
        sp.setFrame(True)
        sp.setStyleSheet("font-size: 11px;")
        return sp

    def add_class(self, name: str) -> bool:
        """Append a new class row with default thresholds."""
        for r in range(self.rowCount()):
            if self.item(r, _COL_NAME).text() == name:
                return False

        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, _COL_NAME, QTableWidgetItem(name))
        self.setCellWidget(row, _COL_BOX, self._make_spin(DEFAULT_BOX_THR))
        self.setCellWidget(row, _COL_TXT, self._make_spin(DEFAULT_TXT_THR))
        self.setCellWidget(row, _COL_NMS, self._make_spin(DEFAULT_NMS_THR))
        self.setRowHeight(row, 26)
        return True

    def remove_class(self, name: str) -> bool:
        """Remove the row matching the given class name."""
        for r in range(self.rowCount()):
            if self.item(r, _COL_NAME).text() == name:
                self.removeRow(r)
                return True
        return False

    def get_class_configs(self) -> list[dict]:
        """Return list of {name, box_thr, txt_thr, nms_thr}."""
        configs = []
        for r in range(self.rowCount()):
            configs.append({
                "name": self.item(r, _COL_NAME).text(),
                "box_thr": self.cellWidget(r, _COL_BOX).value(),
                "txt_thr": self.cellWidget(r, _COL_TXT).value(),
                "nms_thr": self.cellWidget(r, _COL_NMS).value(),
            })
        return configs

    def get_thresholds_dict(self) -> dict[str, dict[str, float]]:
        """Return {class_name: {"box": ..., "txt": ..., "nms": ...}} for project save."""
        return {
            cfg["name"]: {
                "box": cfg["box_thr"],
                "txt": cfg["txt_thr"],
                "nms": cfg["nms_thr"],
            }
            for cfg in self.get_class_configs()
        }

    def set_thresholds(self, name: str, box: float, txt: float, nms: float) -> bool:
        """Push saved threshold values into an existing class row. Returns False if class not found."""
        for r in range(self.rowCount()):
            if self.item(r, _COL_NAME).text() == name:
                self.cellWidget(r, _COL_BOX).setValue(box)
                self.cellWidget(r, _COL_TXT).setValue(txt)
                self.cellWidget(r, _COL_NMS).setValue(nms)
                return True
        return False

    def clear_classes(self):
        """Remove all rows. Used by clear_all / new project."""
        self.setRowCount(0)

    def get_class_names(self) -> list[str]:
        return [self.item(r, _COL_NAME).text()
                for r in range(self.rowCount())]

    def selected_class_name(self) -> str | None:
        """Return the class name of the currently selected row."""
        row = self.currentRow()
        if row < 0:
            return None
        item = self.item(row, _COL_NAME)
        return item.text() if item else None


class PhraseEditorPanel(QWidget):
    """
    Shows the phrase list for whichever class row is currently selected in
    ClassThresholdTable.  The class name itself is always the first phrase
    and is locked (cannot be removed).

    Phrases are stored in a dict keyed by class name:
        self._phrases = {"mitochondria": ["mitochondria", "elongated organelle"], ...}
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phrases: dict[str, list[str]] = {}
        self._active_class: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(3)

        self.lbl_title = QLabel("Phrases for: ---")
        self.lbl_title.setStyleSheet(
            "font-size: 11px; font-weight: bold;")
        layout.addWidget(self.lbl_title)

        hint = QLabel(
            "DINO uses all phrases below for this class.\n"
            "First phrase (class name) cannot be removed.")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; font-style: italic;")
        layout.addWidget(hint)

        self.phrase_list = QListWidget()
        self.phrase_list.setMaximumHeight(90)
        self.phrase_list.setStyleSheet("font-size: 11px;")
        self.phrase_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.phrase_list.customContextMenuRequested.connect(self._show_phrase_context_menu)
        layout.addWidget(self.phrase_list)

        btn_row = QHBoxLayout()
        self.btn_add_phrase = QPushButton("Add Phrase")
        self.btn_add_phrase.setStyleSheet(
            "QPushButton{font-size:11px;padding:3px 6px;}")
        self.btn_add_phrase.clicked.connect(self._add_phrase)

        self.btn_rem_phrase = QPushButton("Remove Selected")
        self.btn_rem_phrase.setStyleSheet(
            "QPushButton{font-size:11px;padding:3px 6px;}")
        self.btn_rem_phrase.clicked.connect(self._remove_phrase)

        btn_row.addWidget(self.btn_add_phrase)
        btn_row.addWidget(self.btn_rem_phrase)
        layout.addLayout(btn_row)

        self.setVisible(False)

    def set_active_class(self, class_name: str | None):
        """Called when the user selects a different class row."""
        self._active_class = class_name
        if class_name is None:
            self.setVisible(False)
            return
        if class_name not in self._phrases:
            self._phrases[class_name] = [class_name]
        self.lbl_title.setText(f"Phrases for:  {class_name}")
        self._refresh_list()
        self.setVisible(True)

    def _refresh_list(self):
        self.phrase_list.clear()
        if self._active_class is None:
            return
        # Defensive .get(): set_phrases() replaces self._phrases
        # wholesale. The danger is cross-project state carryover —
        # Project A leaves a row selected (_active_class != None), then
        # Project B loads via set_phrases(kept) where `kept` doesn't
        # contain that class. KeyError on _phrases[active] was a P1.
        for i, phrase in enumerate(self._phrases.get(self._active_class, [])):
            item = QListWidgetItem(phrase)
            if i == 0:
                item.setForeground(QColor("#2E75B6"))
                item.setToolTip("Class name --- cannot be removed")
            self.phrase_list.addItem(item)

    def _add_phrase(self):
        if self._active_class is None:
            return
        text, ok = QInputDialog.getText(
            self, "Add Phrase",
            f'New detection phrase for "{self._active_class}":')
        if not (ok and text.strip()):
            return
        phrase = text.strip().rstrip(".")
        existing = self._phrases[self._active_class]
        if phrase.lower() in [p.lower() for p in existing]:
            QMessageBox.information(self, "Duplicate",
                                    "That phrase already exists for this class.")
            return
        self._phrases[self._active_class].append(phrase)
        self._refresh_list()

    def _remove_phrase(self):
        if self._active_class is None:
            return
        row = self.phrase_list.currentRow()
        if row <= 0:
            if row == 0:
                QMessageBox.information(
                    self, "Cannot Remove",
                    "The class name phrase cannot be removed.")
            return
        self._phrases[self._active_class].pop(row)
        self._refresh_list()

    def _show_phrase_context_menu(self, position):
        """Right-click menu on a phrase row. Rename is allowed for every
        row including row 0 (the class-name phrase); delete is still
        locked for row 0 — handled in _remove_phrase, not here.
        """
        if self._active_class is None:
            return
        item = self.phrase_list.itemAt(position)
        if item is None:
            return
        row = self.phrase_list.row(item)

        menu = QMenu(self)
        rename_action = QAction("Rename Phrase", self)
        rename_action.triggered.connect(lambda: self._rename_phrase(row))
        menu.addAction(rename_action)
        menu.exec(self.phrase_list.mapToGlobal(position))

    def _rename_phrase(self, row: int):
        """Prompt for a new phrase text and replace the row. Mirrors the
        class-rename flow in annotator_window: validate non-empty, strip,
        reject duplicates within the same class. Row 0 may be renamed —
        it just can't be removed.
        """
        if self._active_class is None or row < 0:
            return
        phrases = self._phrases.get(self._active_class, [])
        if row >= len(phrases):
            return
        current = phrases[row]
        text, ok = QInputDialog.getText(
            self, "Rename Phrase",
            f'New text for "{current}":',
            text=current,
        )
        if not (ok and text.strip()):
            return
        new_phrase = text.strip().rstrip(".")
        if new_phrase == current:
            return
        existing_lower = [p.lower() for i, p in enumerate(phrases) if i != row]
        if new_phrase.lower() in existing_lower:
            QMessageBox.information(self, "Duplicate",
                                    "That phrase already exists for this class.")
            return
        phrases[row] = new_phrase
        self._refresh_list()
        # Restore selection on the renamed row so a follow-up rename
        # works without re-clicking.
        self.phrase_list.setCurrentRow(row)

    def on_class_added(self, class_name: str):
        if class_name not in self._phrases:
            self._phrases[class_name] = [class_name]

    def on_class_removed(self, class_name: str):
        self._phrases.pop(class_name, None)
        if self._active_class == class_name:
            self.set_active_class(None)

    def get_phrases_for(self, class_name: str) -> list[str]:
        # Return the user-edited phrase list as-is. The class-name
        # phrase (row 0) was historically auto-prepended here as a
        # safety net, but that defeated the row-0 rename feature: the
        # user would rename "cell" → "small green blob" and DINO would
        # still receive ["cell", "small green blob"] because the
        # original was re-injected. Trust the editor's state. Fall back
        # to a single-phrase list only when nothing was ever stored.
        return list(self._phrases.get(class_name, [class_name]))

    def get_all_phrases(self) -> dict[str, list[str]]:
        return dict(self._phrases)

    def set_phrases(self, phrases: dict[str, list[str]]):
        self._phrases = {k: list(v) for k, v in phrases.items()}
        if self._active_class:
            self._refresh_list()

    def clear(self):
        """Wipe all stored phrases. Used by clear_all / new project."""
        self._phrases.clear()
        self._active_class = None
        self.phrase_list.clear()
        self.setVisible(False)
