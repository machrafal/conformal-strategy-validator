"""tests/conftest.py - unit tests for csv_validator/permutation.py"""

from csv_validator.metrics import sharpe_ratio
from csv_validator.permutation import (
    circular_shift_test,
    sign_flip_test,
    stationary_bootstrap_test,
)


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


# --- sign_flip_test ---
def test_sign_flip_pvalue_between_0_and_1(real_like_returns):
    p_value, _, _ = sign_flip_test(real_like_returns)
    assert 0 <= p_value <= 1


def test_sign_flip_null_distribution_shape(real_like_returns):
    _, null_dist, _ = sign_flip_test(real_like_returns)
    assert null_dist.shape == (1000,)


def test_sign_flip_reproducible(real_like_returns):
    p1, _, _ = sign_flip_test(real_like_returns, seed=100)
    p2, _, _ = sign_flip_test(real_like_returns, seed=100)
    assert p1 == p2


# --- stationary_bootstrap_test ---
def test_stationary_bootstrap_pvalue_between_0_and_1(real_like_returns):
    p_value, _, _ = stationary_bootstrap_test(real_like_returns)
    assert 0 <= p_value <= 1


def test_stationary_bootstrap_null_distribution_shape(real_like_returns):
    _, null_dist, _ = stationary_bootstrap_test(real_like_returns)
    assert null_dist.shape == (1000,)


def test_stationary_bootstrap_reproducible(real_like_returns):
    p1, _, _ = stationary_bootstrap_test(real_like_returns, seed=100)
    p2, _, _ = stationary_bootstrap_test(real_like_returns, seed=100)
    assert p1 == p2
