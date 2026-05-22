# soft_dark_stylesheet.py

soft_dark_stylesheet = """
QWidget {
    background-color: #2F2F2F;
    color: #E0E0E0;
    font-family: Arial, sans-serif;
}

QMainWindow {
    background-color: #2A2A2A;
}

QPushButton {
    background-color: #4A4A4A;
    border: 1px solid #5E5E5E;
    padding: 5px 10px;
    border-radius: 3px;
    color: #E0E0E0;
}

QPushButton:hover {
    background-color: #545454;
}

QPushButton:pressed {
    background-color: #404040;
}

QPushButton:checked {
    background-color: #606060;
    border: 2px solid #808080;
    color: #FFFFFF;
}

QListWidget, QTreeWidget {
    background-color: #3A3A3A;
    border: 1px solid #4A4A4A;
    border-radius: 3px;
    color: #E0E0E0;
}

QListWidget::item, QTreeWidget::item {
    color: #E0E0E0;  
}

QListWidget::item:selected, QTreeWidget::item:selected {
    background-color: #4A4A4A;
    color: #FFFFFF;  /* Make selected items a bit brighter */
}

QLabel {
    color: #E0E0E0;
}

QLabel.section-header {
    font-weight: bold;
    font-size: 14px;
    padding: 5px 0;
    color: #FFFFFF;  /* Bright white color for better visibility in dark mode */
}

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #3A3A3A;
    border: 1px solid #4A4A4A;
    color: #E0E0E0;
    padding: 2px;
    border-radius: 3px;
}

QSlider::groove:horizontal {
    background: #4A4A4A;
    height: 8px;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #6A6A6A;
    width: 18px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #7A7A7A;
}

QScrollBar:vertical, QScrollBar:horizontal {
    background-color: #3A3A3A;
    width: 12px;
    height: 12px;
}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background-color: #5A5A5A;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background-color: #6A6A6A;
}

QScrollBar::add-line, QScrollBar::sub-line {
    background: none;
}

QMenuBar {
    background-color: #2F2F2F;
}

QMenuBar::item {
    padding: 5px 10px;
    background-color: transparent;
}

QMenuBar::item:selected {
    background-color: #3A3A3A;
}

QMenu {
    background-color: #2F2F2F;
    border: 1px solid #3A3A3A;
}

QMenu::item {
    padding: 5px 20px 5px 20px;
}

QMenu::item:selected {
    background-color: #3A3A3A;
}

QToolTip {
    background-color: #2F2F2F;
    color: #E0E0E0;
    border: 1px solid #3A3A3A;
}

QStatusBar {
    background-color: #2A2A2A;
    color: #B0B0B0;
}

QListWidget::item {
    color: none;
}

/* Form controls — radio + check were invisible-when-selected on the
   default OS theme under dark mode. Tables and spin boxes used to
   render with a bright header bar in the DINO panel because of
   hardcoded #e0e0e0 in code (now removed). */

QRadioButton {
    color: #E0E0E0;
    spacing: 6px;
}

QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 8px;
    border: 1px solid #6A6A6A;
    background-color: #3A3A3A;
}

QRadioButton::indicator:checked {
    background-color: #4DA3FF;
    border: 2px solid #BCD7FF;
}

QRadioButton::indicator:hover {
    border-color: #8A8A8A;
}

QCheckBox {
    color: #E0E0E0;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #6A6A6A;
    background-color: #3A3A3A;
}

QCheckBox::indicator:checked {
    background-color: #4DA3FF;
    border: 1px solid #BCD7FF;
}

QCheckBox::indicator:hover {
    border-color: #8A8A8A;
}

QGroupBox {
    color: #E0E0E0;
    border: 1px solid #4A4A4A;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}

QTableWidget {
    background-color: #2F2F2F;
    color: #E0E0E0;
    gridline-color: #4A4A4A;
    border: 1px solid #4A4A4A;
}

QTableWidget::item:selected {
    background-color: #4A4A4A;
    color: #FFFFFF;
}

QHeaderView::section {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #4A4A4A;
    padding: 4px;
}

QSpinBox, QDoubleSpinBox {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #4A4A4A;
    border-radius: 3px;
    padding: 2px;
}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #4A4A4A;
    border: 1px solid #5A5A5A;
    width: 16px;
}

QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #5A5A5A;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 4px solid #E0E0E0;
    width: 0px;
    height: 0px;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 4px solid #E0E0E0;
    width: 0px;
    height: 0px;
}

QComboBox {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #4A4A4A;
    border-radius: 3px;
    padding: 3px 6px;
}

QComboBox:hover {
    border-color: #6A6A6A;
}

QComboBox::drop-down {
    border-left: 1px solid #4A4A4A;
    width: 18px;
}

QComboBox QAbstractItemView {
    background-color: #2F2F2F;
    color: #E0E0E0;
    selection-background-color: #4A4A4A;
    selection-color: #FFFFFF;
    border: 1px solid #4A4A4A;
}
"""