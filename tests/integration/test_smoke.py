"""
Smoke tests: verify the package's public API surface and every internal
module can be imported.

These tests are the safety net for the modular refactoring. They catch
the most common refactor regressions (renames, missing re-exports, broken
intra-package imports) without needing a real image, a SAM model, or a
running Qt event loop.

If a module gets moved into a subpackage, the corresponding line below
must be updated. That is the test's whole point.
"""

import importlib

import pytest


def test_public_api_exports():
    """The five names documented in __init__.py must remain importable."""
    import digitalsreeni_image_annotator as pkg

    assert pkg.__version__ == "0.9.0"
    assert hasattr(pkg, "ImageAnnotator")
    assert hasattr(pkg, "ImageLabel")
    assert hasattr(pkg, "SAMUtils")
    assert hasattr(pkg, "calculate_area")
    assert hasattr(pkg, "calculate_bbox")


INTERNAL_MODULES = [
    # Core
    "digitalsreeni_image_annotator.main",
    "digitalsreeni_image_annotator.annotator_window",
    "digitalsreeni_image_annotator.utils",
    # Widgets
    "digitalsreeni_image_annotator.widgets.image_label",
    # Inference
    "digitalsreeni_image_annotator.inference.sam_utils",
    "digitalsreeni_image_annotator.inference.dino_utils",
    # I/O
    "digitalsreeni_image_annotator.io.export_formats",
    "digitalsreeni_image_annotator.io.import_formats",
    # Core helpers
    "digitalsreeni_image_annotator.core.constants",
    "digitalsreeni_image_annotator.core.annotation_utils",
    # UI
    "digitalsreeni_image_annotator.ui.default_stylesheet",
    "digitalsreeni_image_annotator.ui.soft_dark_stylesheet",
    # Dialogs
    "digitalsreeni_image_annotator.dialogs.annotation_statistics",
    "digitalsreeni_image_annotator.dialogs.coco_json_combiner",
    "digitalsreeni_image_annotator.dialogs.dataset_splitter",
    "digitalsreeni_image_annotator.dialogs.dicom_converter",
    "digitalsreeni_image_annotator.dialogs.dino_merge_dialog",
    "digitalsreeni_image_annotator.dialogs.dino_phrase_editor",
    "digitalsreeni_image_annotator.dialogs.help_window",
    "digitalsreeni_image_annotator.dialogs.image_augmenter",
    "digitalsreeni_image_annotator.dialogs.image_patcher",
    "digitalsreeni_image_annotator.dialogs.project_details",
    "digitalsreeni_image_annotator.dialogs.project_search",
    "digitalsreeni_image_annotator.dialogs.slice_registration",
    "digitalsreeni_image_annotator.dialogs.snake_game",
    "digitalsreeni_image_annotator.dialogs.stack_interpolator",
    "digitalsreeni_image_annotator.dialogs.stack_to_slices",
    "digitalsreeni_image_annotator.dialogs.yolo_trainer",
]


@pytest.mark.parametrize("module_name", INTERNAL_MODULES)
def test_internal_module_imports(module_name):
    """Every internal module must import without raising."""
    importlib.import_module(module_name)
