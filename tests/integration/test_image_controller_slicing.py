"""ImageController multi-dimensional slicing tests (fork issue #36).

Multi-dim slicing (`controllers/image_controller.py`) is the machinery future
work (lazy slice loading, video-frames-as-slices) builds on, and it shipped
with **zero tests** plus one documented near-miss: the TIFF axis-hint default
computation had a ``[-ndim:]`` slice bug that produced 2560 wrong slices on a
5D ``TZCYX`` file (see arc42). ``test_5d_tzcyx_hint_regression`` pins that.

These build one real ImageAnnotator (offscreen), write tiny synthetic TIFFs,
and drive ``image_controller.load_tiff`` directly. The dimension dialog is
monkeypatched on the ``image_controller`` module (where ``DimensionDialog`` is
*defined*, not in ``dialogs/``) so it both captures the controller-computed
defaults and answers with them — no user interaction, no real dialog.
"""

import numpy as np
import pytest
import tifffile

import digitalsreeni_image_annotator.controllers.image_controller as ic
from digitalsreeni_image_annotator.core import image_utils

from PyQt6.QtGui import QImage


@pytest.fixture
def window(qt_application):
    from digitalsreeni_image_annotator.annotator_window import ImageAnnotator

    w = ImageAnnotator()
    yield w
    w.deleteLater()


def make_tiff(tmp_path, name, shape, axes=None):
    """Write a synthetic uint16 TIFF.

    With ``axes`` given (e.g. ``"TZCYX"``) write an ImageJ hyperstack so
    ``TiffFile(...).series[0].axes`` round-trips and ``load_tiff`` derives an
    axes hint. Without ``axes``, plain ``imwrite`` — a stack whose leading dim
    is not 3/4 reads back with an unknown leading axis (``Q``), so no hint
    applies and the ``ndim_defaults`` fallback is exercised.

    ImageJ hyperstacks require axes to be a suffix of ``TZCYXS`` and dtype
    uint8/uint16/float32; the shapes used here comply (round-trip verified in
    the session that wrote this file).
    """
    data = (np.random.rand(*shape) * 65535).astype(np.uint16)
    path = str(tmp_path / name)
    if axes:
        tifffile.imwrite(path, data, imagej=True, metadata={"axes": axes})
    else:
        tifffile.imwrite(path, data)
    return path, data


@pytest.fixture
def fake_dimension_dialog(monkeypatch):
    """Intercept ``DimensionDialog``: capture the controller-computed defaults
    and answer the dialog with them (accept, return those dimensions).

    ``captured`` stays ``{}`` if the dialog is never constructed (the 2D
    shortcut and the stored-dimensions reload path both skip it)."""
    captured = {}

    class FakeDim:
        def __init__(self, shape, file_name, parent=None, default_dimensions=None):
            captured["defaults"] = list(default_dimensions or [])
            self._dims = list(default_dimensions or [])

        def exec(self):
            return True

        def get_dimensions(self):
            return self._dims

    monkeypatch.setattr(ic, "DimensionDialog", FakeDim)
    return captured


def _slice_names(window, base_name):
    """Slice names as stored on the window (value comparison, not identity)."""
    return [name for name, _ in window.image_slices[base_name]]


def _slice_list_texts(window):
    return [
        window.slice_list.item(i).text()
        for i in range(window.slice_list.count())
    ]


def test_2d_tiff_single_slice(tmp_path, window, fake_dimension_dialog):
    """Plain 2D TIFF: one slice named exactly the base name, no dialog — 2D
    takes the ``["H","W"]`` shortcut before any default computation."""
    path, _ = make_tiff(tmp_path, "stack2d.tif", (16, 12))

    window.image_controller.load_tiff(path)

    assert fake_dimension_dialog == {}  # dialog never constructed for 2D
    assert _slice_names(window, "stack2d") == ["stack2d"]
    assert _slice_list_texts(window) == ["stack2d"]


def test_3d_unknown_axes_fallback_defaults(tmp_path, window, fake_dimension_dialog):
    """Plain 3D TIFF whose leading dim is 5 reads back as ``QYX`` (unknown
    axis Q → no hint), so ``ndim_defaults[3] == ["Z","H","W"]`` is used and
    the slices are named ``stack3d_Z1 … stack3d_Z5``.

    (Spec used shape ``(4,16,12)``, but tifffile labels a size-4 leading dim
    ``S`` — a *known* axis, so the hint would apply and defaults would be
    ``["S","H","W"]``; ``(5,16,12)`` → ``QYX`` genuinely exercises the
    fallback the case is meant to cover. Count is therefore 5, not 4.)"""
    path, _ = make_tiff(tmp_path, "stack3d.tif", (5, 16, 12))

    window.image_controller.load_tiff(path)

    assert fake_dimension_dialog["defaults"] == ["Z", "H", "W"]
    expected = [f"stack3d_Z{i}" for i in range(1, 6)]
    assert _slice_names(window, "stack3d") == expected
    assert _slice_list_texts(window) == expected


def test_3d_imagej_zyx_hint_applied(tmp_path, window, fake_dimension_dialog):
    """ImageJ ``ZYX`` hint is applied (len(hint) == ndim): defaults sourced
    from the hint, 4 slices ``stack3d_Z1 … stack3d_Z4``."""
    path, _ = make_tiff(tmp_path, "stack3d.tif", (4, 16, 12), axes="ZYX")

    window.image_controller.load_tiff(path)

    assert fake_dimension_dialog["defaults"] == ["Z", "H", "W"]
    expected = [f"stack3d_Z{i}" for i in range(1, 5)]
    assert _slice_names(window, "stack3d") == expected
    assert _slice_list_texts(window) == expected


def test_5d_tzcyx_hint_regression(tmp_path, window, fake_dimension_dialog):
    """The ``[-ndim:]`` regression test.

    ImageJ ``TZCYX`` (2,3,2,16,12): the axis hint must yield defaults
    ``["T","Z","C","H","W"]`` and exactly ``2*3*2 == 12`` slices (not a
    blow-up), first ``stack5d_T1_Z1_C1``, last ``stack5d_T2_Z3_C2``. It fails
    if the axis-hint default computation regresses to the old slice bug."""
    path, _ = make_tiff(tmp_path, "stack5d.tif", (2, 3, 2, 16, 12), axes="TZCYX")

    window.image_controller.load_tiff(path)

    assert fake_dimension_dialog["defaults"] == ["T", "Z", "C", "H", "W"]

    names = _slice_names(window, "stack5d")
    assert len(names) == 12  # 2*3*2, not the 2560 blow-up
    assert names[0] == "stack5d_T1_Z1_C1"
    assert names[-1] == "stack5d_T2_Z3_C2"

    # Fully enumerate the expected 12 names in ndindex order.
    expected = [
        f"stack5d_T{t+1}_Z{z+1}_C{c+1}"
        for t in range(2)
        for z in range(3)
        for c in range(2)
    ]
    assert names == expected
    assert _slice_list_texts(window) == expected
    assert window.slice_list.count() == 12


def test_stored_dimensions_skip_dialog(tmp_path, window, fake_dimension_dialog):
    """Project-reload path: passing stored ``dimensions`` + ``shape`` reshapes
    and skips the dialog entirely (4 slices, ``captured == {}``).

    The real reload entry point (``load_multi_slice_image``) pre-populates
    ``image_dimensions[base_name]`` / ``image_shapes[base_name]`` *before*
    calling ``load_tiff`` — ``load_tiff``'s stored-dimensions branch relies on
    that (``process_multidimensional_image`` reads ``image_dimensions[base]``
    without setting it in this branch). We mirror that precondition here rather
    than call ``load_tiff`` bare, which would ``KeyError`` — not a production
    bug, since the only caller passing stored dimensions pre-sets these."""
    path, _ = make_tiff(tmp_path, "stack3d.tif", (4, 16, 12), axes="ZYX")
    window.image_dimensions["stack3d"] = ["Z", "H", "W"]
    window.image_shapes["stack3d"] = (4, 16, 12)

    window.image_controller.load_tiff(
        path, dimensions=["Z", "H", "W"], shape=(4, 16, 12)
    )

    assert fake_dimension_dialog == {}  # dialog never constructed
    expected = [f"stack3d_Z{i}" for i in range(1, 5)]
    assert _slice_names(window, "stack3d") == expected


def test_slice_state_after_load(tmp_path, window, fake_dimension_dialog):
    """After loading the 5D file the window's current-slice state is the first
    slice, and ``current_image`` is a QImage of the plane (width 12, height 16)."""
    path, _ = make_tiff(tmp_path, "stack5d.tif", (2, 3, 2, 16, 12), axes="TZCYX")

    window.image_controller.load_tiff(path)

    assert window.current_slice == "stack5d_T1_Z1_C1"
    assert window.image_file_name == window.current_slice
    assert isinstance(window.current_image, QImage)
    assert window.current_image.width() == 12
    assert window.current_image.height() == 16


def test_uint16_normalized_to_uint8():
    """Unit-style (no window): the 16→8-bit normalization contract (ADR-010).

    ``normalize_array`` min-max scales uint16 to uint8 0-255;
    ``convert_to_8bit_rgb`` stacks to 3 channels; ``array_to_qimage`` picks
    ``Format_RGB888`` for a 3-channel array."""
    arr = np.linspace(0, 65535, 16 * 12).reshape(16, 12).astype(np.uint16)

    norm = image_utils.normalize_array(arr)
    assert norm.dtype == np.uint8
    assert int(norm.min()) == 0
    assert int(norm.max()) == 255

    rgb = image_utils.convert_to_8bit_rgb(arr)
    assert rgb.shape == (16, 12, 3)

    qimg = image_utils.array_to_qimage(rgb)
    assert qimg.format() == QImage.Format.Format_RGB888
