"""tests/test_conformal.py - unit tests for csv_validator/conformal.py"""

import numpy as np
import pytest

from csv_validator.conformal import AdaptiveConformalValidator, SplitConformalValidator


def test_fit_returns_self(real_like_returns):
    validator = SplitConformalValidator()
    result = validator.fit(real_like_returns)
    assert result is validator


def test_predict_interval_lower_less_than_upper(real_like_returns):
    validator = SplitConformalValidator().fit(real_like_returns)
    lower, upper = validator.predict_interval(alpha=0.1)
    assert lower < upper


def test_predict_interval_before_fit_raises():
    validator = SplitConformalValidator()
    with pytest.raises(RuntimeError):
        validator.predict_interval()


def test_split_coverage_between_0_and_1(real_like_returns):
    validator = SplitConformalValidator().fit(real_like_returns)
    cov = validator.coverage(real_like_returns, alpha=0.1)
    assert 0 <= cov <= 1


def test_split_coverage_near_nominal(real_like_returns):
    validator = SplitConformalValidator().fit(real_like_returns)
    cov = validator.coverage(real_like_returns, alpha=0.1)
    assert cov >= 0.85


def test_fit_and_transform_arrays_length(real_like_returns):
    validator = AdaptiveConformalValidator()
    lower, upper = validator.fit_transform(real_like_returns, alpha=0.1, window=50)
    assert len(lower) == len(real_like_returns) and len(upper) == len(real_like_returns)


def test_first_elements_of_lower_all_nans(real_like_returns):
    validator = AdaptiveConformalValidator()
    lower, _ = validator.fit_transform(real_like_returns, alpha=0.1, window=50)
    assert np.isnan(lower[:50]).all()


def test_adaptive_coverage_between_0_and_1(real_like_returns):
    validator = AdaptiveConformalValidator()
    cov = validator.coverage(real_like_returns, alpha=0.1, window=50)
    assert 0 <= cov <= 1


def test_adaptive_coverage_near_nominal(real_like_returns):
    validator = AdaptiveConformalValidator()
    cov = validator.coverage(real_like_returns, alpha=0.1, window=50)
    assert cov >= 0.85
