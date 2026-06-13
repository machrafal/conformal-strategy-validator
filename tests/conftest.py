"""tests/conftest.py - unit tests for csv_validator/permutation.py"""

import numpy as np
import pytest

from csv_validator.metrics import sharpe_ratio
from csv_validator.permutation import (
    circular_shift_test,
    sign_flip_test,
    stationary_bootstrap_test,
)


# ---fixtures---
@pytest.fixture
def flat_returns():
    """All zeros - no edge, no volatility."""
    return np.zeros(252)


@pytest.fixture
def positive_returns():
    """Steadily positive returns - clean uptrend."""
    return np.full(252, 0.001)


@pytest.fixture
def real_like_returns():
    """Realistic noisy returns with positive drift."""
    rng = np.random.default_rng(100)
    return rng.normal(loc=0.0005, scale=0.01, size=252)


# --- circular_shift_test ---
def test_circular_shift_pvalue_between_0_and_1(real_like_returns):
    p_value, _, _ = circular_shift_test(real_like_returns)
    assert 0 <= p_value <= 1


def test_circular_shift_null_distribution_shape(real_like_returns):
    _, null_dist, _ = circular_shift_test(real_like_returns)
    assert null_dist.shape == (1000,)


def test_circular_shift_observed_stat_matches_sharpe(real_like_returns):
    _, _, observed_stat = circular_shift_test(real_like_returns)
    assert observed_stat == sharpe_ratio(real_like_returns)


def test_circular_shift_reproducible(real_like_returns):
    p1, _, _ = circular_shift_test(real_like_returns, seed=100)
    p2, _, _ = circular_shift_test(real_like_returns, seed=100)
    assert p1 == p2
