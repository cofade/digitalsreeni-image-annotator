"""Tests for the dataset-splitter percentage auto-balancing (#80).

Constructing DatasetSplitterTool needs a QApplication (offscreen via
conftest). We drive the spin boxes with setValue (which fires valueChanged,
exactly as a user edit does) and assert the three always re-sum to 100 with
the documented absorption order.
"""

import pytest

from digitalsreeni_image_annotator.dialogs.dataset_splitter import (
    DatasetSplitterTool,
)


@pytest.fixture
def splitter(qt_application):
    dlg = DatasetSplitterTool(None)
    yield dlg
    dlg.deleteLater()


def _values(dlg):
    return (
        dlg.train_percent.value(),
        dlg.val_percent.value(),
        dlg.test_percent.value(),
    )


def test_defaults_sum_to_100(splitter):
    assert sum(_values(splitter)) == 100


def test_train_change_val_absorbs(splitter):
    splitter.train_percent.setValue(80)
    train, val, test = _values(splitter)
    assert (train, val, test) == (80, 20, 0)
    assert train + val + test == 100


def test_train_overflow_spills_into_test(splitter):
    # Start 0/40/60, then bump train high enough that val can't absorb it.
    splitter.test_percent.setValue(60)  # -> 0/40/60 after rebalance
    splitter.train_percent.setValue(90)
    train, val, test = _values(splitter)
    assert train == 90
    assert val == 0  # val absorbed first, exhausted
    assert train + val + test == 100


def test_val_change_test_absorbs_first(splitter):
    splitter.train_percent.setValue(50)  # 50/50/0 -> wait, val recomputed
    splitter.test_percent.setValue(20)   # establish some test budget
    # Now nudge val; test should absorb before train.
    before_train = splitter.train_percent.value()
    splitter.val_percent.setValue(splitter.val_percent.value() + 10)
    train, val, test = _values(splitter)
    assert train + val + test == 100
    assert train == before_train  # train untouched while test had room


def test_val_exhausts_test_then_pulls_train(splitter):
    splitter.train_percent.setValue(50)
    splitter.test_percent.setValue(20)
    splitter.val_percent.setValue(60)  # needs more than test can give
    train, val, test = _values(splitter)
    assert val == 60
    assert test == 0
    assert train + val + test == 100


def test_test_change_val_absorbs_first(splitter):
    splitter.train_percent.setValue(50)  # 50/50/0
    splitter.test_percent.setValue(20)   # val absorbs -> 50/30/20
    train, val, test = _values(splitter)
    assert (train, val, test) == (50, 30, 20)


def test_test_exhausts_val_then_pulls_train(splitter):
    splitter.train_percent.setValue(40)  # 40/60/0
    splitter.test_percent.setValue(80)   # val gives all 60, train gives rest
    train, val, test = _values(splitter)
    assert test == 80
    assert val == 0
    assert train + val + test == 100


def test_boundary_clamps_and_no_recursion(splitter):
    splitter.train_percent.setValue(100)
    train, val, test = _values(splitter)
    assert (train, val, test) == (100, 0, 0)
    # All within range, sum still 100 (proves handlers terminated cleanly).
    assert all(0 <= v <= 100 for v in (train, val, test))
