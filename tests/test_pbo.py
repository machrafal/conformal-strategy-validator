"""tests/test_pbo.py - unit tests for csv_validator/pbo.py"""

import numpy as np
import pytest

from csv_validator.pbo import generate_cpcv_splits, probability_of_backtest_overfitting


def test_generate_cpcv_splits_returns_c_splits():
    splits = generate_cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=5)
    assert len(splits) == 15


def test_train_and_test_indices_never_overlap():
    splits = generate_cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=5)
    for train_idx, test_idx in splits:
        overlap = np.intersect1d(train_idx, test_idx)
        assert len(overlap) == 0


def test_probability_of_backtest_overfitting_returns_between_0_and_1():
    rng = np.random.default_rng(100)
    n_splits, n_strategies = 15, 20
    is_sharpes = rng.normal(size=(n_splits, n_strategies))
    oos_sharpes = rng.normal(size=(n_splits, n_strategies))
    pbo = probability_of_backtest_overfitting(is_sharpes, oos_sharpes)
    assert 0 <= pbo <= 1


def test_is_best_always_oos_best():
    rng = np.random.default_rng(100)
    n_splits, n_strategies = 15, 20
    is_sharpes = rng.normal(size=(n_splits, n_strategies))
    oos_sharpes = is_sharpes.copy()
    pbo = probability_of_backtest_overfitting(is_sharpes, oos_sharpes)
    assert pbo == 0


def test_is_best_always_oos_worst():
    rng = np.random.default_rng(100)
    n_splits, n_strategies = 15, 20
    is_sharpes = rng.normal(size=(n_splits, n_strategies))
    oos_sharpes = -is_sharpes.copy()
    pbo = probability_of_backtest_overfitting(is_sharpes, oos_sharpes)
    assert pbo == 1
