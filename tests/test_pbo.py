"""tests/test_pbo.py - unit tests for csv_validator/pbo.py"""

from csv_validator.pbo import probability_of_backtest_overfitting, generate_cpcv_splits


def test_generate_cpcv_splits_returns_c_splits():
    splits = generate_cpcv_splits(n_obs=120, n_groups=6, n_test_groups=2, embargo=5)
    assert len(splits) == 15
