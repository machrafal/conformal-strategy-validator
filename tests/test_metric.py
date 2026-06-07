"""tests/test_metric.py - unit tests for csv_validator/metrics.py"""

import numpy as np
import pytest
from csv_validator.metrics import (
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
    sortino_ratio,
    deflated_sharpe_ratio,
)


# --- fixtures ---
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


# --- sharpe_ratio ---
def test_sharpe_zero_vol_returns_zero(flat_returns):
    assert sharpe_ratio(flat_returns) == 0.0


def test_sharpe_positive_returns_positive(positive_returns):
    assert sharpe_ratio(positive_returns) > 0


def test_sharpe_returns_float(real_like_returns):
    assert isinstance(sharpe_ratio(real_like_returns), float)
