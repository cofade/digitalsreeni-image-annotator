"""Unit tests for DatasetSplitterTool.compute_split_counts.

compute_split_counts is a @staticmethod with no Qt dependency, so these
tests call it directly without constructing the dialog. They lock in the
#81 fix (a 0% subset must get exactly 0 images) and the invariant that the
three counts always sum to the dataset size.
"""

import pytest

from digitalsreeni_image_annotator.dialogs.dataset_splitter import (
    DatasetSplitterTool,
)

compute = DatasetSplitterTool.compute_split_counts


def test_zero_test_yields_zero_test_images_even_n():
    # 10 images, 70/30/0 -> exactly (7, 3, 0); no phantom test image.
    assert compute(10, 70, 30, 0) == (7, 3, 0)


def test_zero_test_yields_zero_test_images_odd_n():
    # The #81 regression: odd N used to leak the flooring leftover into the
    # open-ended test slice. Test must stay 0 and counts must sum to N.
    train, val, test = compute(11, 70, 30, 0)
    assert test == 0
    assert train + val + test == 11


@pytest.mark.parametrize("n", [1, 3, 7, 11, 13, 99, 100, 101, 1000])
@pytest.mark.parametrize(
    "pcts",
    [(70, 30, 0), (80, 10, 10), (33, 33, 34), (50, 50, 0), (0, 50, 50),
     (50, 0, 50), (100, 0, 0), (60, 20, 20)],
)
def test_counts_always_sum_to_n(n, pcts):
    counts = compute(n, *pcts)
    assert sum(counts) == n
    assert all(c >= 0 for c in counts)


@pytest.mark.parametrize(
    "pcts,zero_index",
    [((0, 50, 50), 0), ((50, 0, 50), 1), ((50, 50, 0), 2),
     ((0, 100, 0), 0), ((0, 0, 100), 0)],
)
def test_zero_percent_subset_gets_zero(pcts, zero_index):
    counts = compute(17, *pcts)
    assert counts[zero_index] == 0


def test_no_positive_subset_is_starved():
    # Every subset with a positive percentage should receive at least one
    # image when the dataset is large enough to go around.
    train, val, test = compute(10, 33, 33, 34)
    assert train >= 1 and val >= 1 and test >= 1
    assert train + val + test == 10


def test_empty_dataset():
    assert compute(0, 70, 30, 0) == (0, 0, 0)


def test_single_image_all_train():
    assert compute(1, 100, 0, 0) == (1, 0, 0)
