"""Theme + font size application, extracted from `ImageAnnotator`.

The functions here take the main window as their first argument; they
read state directly off it (`dark_mode`, `current_font_size`, etc.) and
write to its widgets. Kept as plain functions rather than a controller
class because they are stateless and the call sites are sparse.
"""

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QComboBox, QLabel, QWidget

from .default_stylesheet import default_stylesheet
from .soft_dark_stylesheet import soft_dark_stylesheet


def apply_theme_and_font(mw):
    font_size = mw.font_sizes[mw.current_font_size]
    style = soft_dark_stylesheet if mw.dark_mode else default_stylesheet
    combined_style = f"{style}\nQWidget {{ font-size: {font_size}pt; }}"
    mw.setStyleSheet(combined_style)

    for widget in mw.findChildren(QWidget):
        font = widget.font()
        font.setPointSize(font_size)
        widget.setFont(font)

    mw.image_label.setFont(QFont("Arial", font_size))
    mw.update()


def toggle_dark_mode(mw):
    mw.dark_mode = not mw.dark_mode
    apply_theme_and_font(mw)
    mw.update_slice_list_colors()
    mw.update_class_list()
    mw.update_annotation_list()
    mw.repaint()


def apply_stylesheet(mw):
    mw.setStyleSheet(soft_dark_stylesheet if mw.dark_mode else default_stylesheet)


def update_ui_colors(mw):
    mw.update_annotation_list_colors()
    mw.update_slice_list_colors()
    mw.image_label.update()


def setup_font_size_selector(mw):
    font_size_label = QLabel("Font Size:")
    mw.font_size_selector = QComboBox()
    mw.font_size_selector.addItems(["Small", "Medium", "Large"])
    mw.font_size_selector.setCurrentText("Medium")
    mw.font_size_selector.currentTextChanged.connect(lambda size: on_font_size_changed(mw, size))

    mw.sidebar_layout.addWidget(font_size_label)
    mw.sidebar_layout.addWidget(mw.font_size_selector)


def on_font_size_changed(mw, size):
    mw.current_font_size = size
    apply_theme_and_font(mw)


def change_font_size(mw, size):
    mw.current_font_size = size
    apply_theme_and_font(mw)
