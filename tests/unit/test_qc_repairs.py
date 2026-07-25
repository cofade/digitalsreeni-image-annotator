"""QC repairs through the real controller and the real undo stack (#70).

Written after a senior review found that ``fix_findings`` recorded history for
the **current image only** while mutating annotations on any image — so a sweep
across a project was irreversible everywhere except the one image on screen,
and the dialog promised "a single Ctrl+Z".

The rule engine has thorough unit tests, but they exercise pure functions. The
bug lived in the controller, between the engine and ``AnnotationHistory``, and
nothing covered that seam. These tests do, using the real
``AnnotationController`` and the real per-image history rather than a double —
because the mechanism under test *is* the per-image keying.
"""

import pytest
from PyQt6.QtWidgets import QWidget

from src.digitalsreeni_image_annotator.controllers.annotation_controller import (
    AnnotationController,
)
from src.digitalsreeni_image_annotator.controllers.qc_controller import QCController
from src.digitalsreeni_image_annotator.core import annotation_qc as qc


def _bowtie(number=1):
    """A self-intersecting ring — an ERROR-severity, repairable finding."""
    return {
        "segmentation": [0, 0, 40, 40, 40, 0, 0, 40],
        "category_name": "cell",
        "number": number,
    }


class _FakeTable:
    """The handful of QTableWidget calls update_annotation_list makes."""

    def setRowCount(self, _count):
        pass

    def rowCount(self):
        return 0

    def item(self, _row, _col):
        return None

    def insertRow(self, _row):
        pass

    def setItem(self, *_args):
        pass

    def setCellWidget(self, *_args):
        pass

    def clearSelection(self):
        pass

    def selectedIndexes(self):
        return []


class _FakeImageLabel:
    def __init__(self):
        self.annotations = {}
        self.class_colors = {}
        self.highlighted_annotations = []
        self.original_pixmap = None

    def update(self):
        pass


class _Window(QWidget):
    """Enough main window for AnnotationController + QCController.

    Deliberately real controllers on both sides: the interaction between
    per-image history keys and cross-image repairs is the whole subject.
    """

    def __init__(self, all_annotations, current):
        super().__init__()
        self.is_loading_project = False
        self.all_annotations = all_annotations
        self.image_file_name = current
        self.current_slice = None
        self.image_label = _FakeImageLabel()
        self.annotation_list = _FakeTable()
        self.all_images = [
            {"file_name": name} for name in all_annotations
        ]
        self.image_shapes = {name: (200, 200) for name in all_annotations}
        self.image_slices = {}
        self.class_mapping = {"cell": 1}
        self.merge_button = _FakeButton()
        self.change_class_button = _FakeButton()
        self.annotation_controller = AnnotationController(self)
        self.qc_controller = QCController(self)
        self.saved = 0

    # --- the delegating surface the controllers call ---
    def update_slice_list_colors(self):
        pass

    def save_current_annotations(self):
        self.saved += 1

    def load_image_annotations(self):
        pass

    def update_annotation_list(self):
        pass

    def auto_save(self):
        pass


class _FakeButton:
    def setEnabled(self, _value):
        pass


@pytest.fixture
def window(qtbot):
    project = {
        "a.png": {"cell": [_bowtie(1)]},
        "b.png": {"cell": [_bowtie(1)]},
        "c.png": {"cell": [_bowtie(1)]},
    }
    win = _Window(project, current="a.png")
    qtbot.addWidget(win)
    return win


def _findings(window):
    return [
        f
        for f in qc.run_audit(
            window.all_annotations,
            image_sizes=window.qc_controller.collect_image_sizes(),
            class_names=["cell"],
        )
        if f.fixable
    ]


def test_the_audit_finds_the_seeded_problem_on_every_image(window):
    findings = _findings(window)
    assert {f.image for f in findings} == {"a.png", "b.png", "c.png"}
    assert all(f.rule == qc.RULE_SELF_INTERSECTING for f in findings)


def test_repairs_are_applied_across_every_image(window):
    repaired, images = window.qc_controller.fix_findings(_findings(window))

    assert repaired == 3
    assert images == 3
    assert _findings(window) == [], "the geometry should now be valid"


def test_every_touched_image_gets_its_own_undo_entry(window):
    """THE regression. A single keyless record_history() snapshots only the
    current image, so repairs to the other two were permanent."""
    history = window.annotation_controller.history
    window.qc_controller.fix_findings(_findings(window))

    for name in ("a.png", "b.png", "c.png"):
        assert history.can_undo(name), f"{name} has no undo entry"


def test_undo_restores_an_off_screen_image(window):
    """Not just "an entry exists" — the snapshot has to predate the mutation."""
    before = list(window.all_annotations["b.png"]["cell"][0]["segmentation"])
    window.qc_controller.fix_findings(_findings(window))
    assert window.all_annotations["b.png"]["cell"][0]["segmentation"] != before

    window.annotation_controller._restore_snapshot(
        "b.png",
        window.annotation_controller.history.undo(
            "b.png", {"cell": list(window.all_annotations["b.png"]["cell"])}
        ),
    )
    assert window.all_annotations["b.png"]["cell"][0]["segmentation"] == before


def test_one_snapshot_per_image_not_one_per_finding(window):
    """Two problems on one image are one undo step for that image."""
    window.all_annotations = {"a.png": {"cell": [_bowtie(1), _bowtie(2)]}}
    window.all_images = [{"file_name": "a.png"}]
    window.image_shapes = {"a.png": (200, 200)}

    repaired, images = window.qc_controller.fix_findings(_findings(window))

    assert repaired == 2
    assert images == 1


def test_fix_findings_returns_a_pair(window):
    """The dialog unpacks two values to word the confirmation truthfully."""
    result = window.qc_controller.fix_findings(_findings(window))
    assert isinstance(result, tuple) and len(result) == 2


def test_an_empty_finding_list_is_a_no_op(window):
    assert window.qc_controller.fix_findings([]) == (0, 0)


def test_nothing_is_recorded_during_a_project_load(window):
    """record_history no-ops while is_loading_project is set (ADR-005), so the
    repair path must not depend on it having happened."""
    window.is_loading_project = True
    repaired, _images = window.qc_controller.fix_findings(_findings(window))
    assert repaired == 3
    assert not window.annotation_controller.history.can_undo("b.png")


def test_image_sizes_cover_every_image_in_the_project(window):
    sizes = window.qc_controller.collect_image_sizes()
    assert sizes["a.png"] == (200, 200)
    assert set(sizes) >= {"a.png", "b.png", "c.png"}


def test_a_finding_naming_a_vanished_image_is_skipped(window):
    """An image removed between audit and repair must not crash the sweep."""
    findings = _findings(window)
    del window.all_annotations["b.png"]

    repaired, images = window.qc_controller.fix_findings(findings)

    assert repaired == 2
    assert images == 2
