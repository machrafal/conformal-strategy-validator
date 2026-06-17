"""tests/test_pbo.py - unit tests for csv_validator/pbo.py"""

import numpy as np

from csv_validator.pbo import generate_cpcv_splits, probability_of_backtest_overfitting


def test_generate_cpcv_splits_returns_c_splits():
    splits = generate_cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=5)
    assert len(splits) == 15


def test_train_and_test_indices_never_overlap():
    splits = generate_cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=5)
    for train_idx, test_idx in splits:
        overlap = np.intersect1d(train_idx, test_idx)
        assert len(overlap) == 0
