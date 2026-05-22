"""
Main entry point for the Image Annotator application.

This module creates and runs the main application window.

@DigitalSreeni
Dr. Sreenivas Bhattiprolu
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from .annotator_window import ImageAnnotator

# Legacy defensive cleanup from the PyQt5 era: a stale
# QT_QPA_PLATFORM_PLUGIN_PATH could shadow Qt's bundled XCB plugin and
# break startup on Linux. PyQt6 packaging is more robust about this, but
# the pop is cheap and harmless to keep.
if sys.platform.startswith("linux"):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

def main():
    """
    Main function to run the Image Annotator application.
    """
    app = QApplication(sys.argv)
    window = ImageAnnotator()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()