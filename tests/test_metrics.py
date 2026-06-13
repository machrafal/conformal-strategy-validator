"""tests/test_metric.py - unit tests for csv_validator/metrics.py"""

from csv_validator.metrics import (
    sharpe_ratio,
    max_drawdown,
    calmar_ratio,
    sortino_ratio,
    deflated_sharpe_ratio,
)


# --- sharpe_ratio ---
def test_sharpe_zero_vol_returns_zero(flat_returns):
    assert sharpe_ratio(flat_returns) == 0.0


def test_sharpe_positive_returns_positive(positive_returns):
    assert sharpe_ratio(positive_returns) > 0


def test_sharpe_returns_float(real_like_returns):
    assert isinstance(sharpe_ratio(real_like_returns), float)


# --- max_drawdown ---
def test_max_drawdown_zero_vol_returns_zero(flat_returns):
    assert max_drawdown(flat_returns) == 0.0


def test_max_drawdown_positive_returns_zero(positive_returns):
    assert max_drawdown(positive_returns) == 0.0


def test_max_drawdown_returns_float(real_like_returns):
    assert isinstance(max_drawdown(real_like_returns), float)


# --- calmar_ratio ---
def test_calmar_ratio_zero_vol_returns_zero(flat_returns):
    assert calmar_ratio(flat_returns) == 0.0


def test_calmar_ratio_positive_returns_zero(positive_returns):
    assert calmar_ratio(positive_returns) == 0


def test_calmar_ratio_returns_float(real_like_returns):
    assert isinstance(calmar_ratio(real_like_returns), float)


# --- sortino_ratio ---
def test_sortino_ratio_zero_vol_returns_zero(flat_returns):
    assert sortino_ratio(flat_returns) == 0.0


def test_sortino_ratio_positive_returns_zero(positive_returns):
    assert sortino_ratio(positive_returns) == 0


def test_sortino_ratio_returns_float(real_like_returns):
    assert isinstance(sortino_ratio(real_like_returns), float)


# --- deflated_sharpe_ratio ---
def test_deflated_sharpe_ratio_zero_vol_returns_zero(flat_returns):
    assert deflated_sharpe_ratio(flat_returns, n_trials=50) == 0.0


def test_deflated_sharpe_ratio_positive_returns_zero(real_like_returns):
    result = deflated_sharpe_ratio(real_like_returns, n_trials=50)
    assert 0.0 <= result <= 1.0


def test_deflated_sharpe_ratio_returns_float(real_like_returns):
    assert isinstance(deflated_sharpe_ratio(real_like_returns, n_trials=50), float)
