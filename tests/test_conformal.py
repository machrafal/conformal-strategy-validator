"""tests/test_conformal.py - unit tests for csv_validator/conformal.py"""

import numpy as np
import pytest

from csv_validator.conformal import AdaptiveConformalValidator, SplitConformalValidator


def test_fit_returns_self(real_like_returns):
    validator = SplitConformalValidator()
    result = validator(real_like_returns)
    assert result is validator


def test_predict_interval_lower_less_than_upper(real_like_returns):
    validator = SplitConformalValidator.fit(real_like_returns)
    lower, upper = validator.predict_interval(alpha=0.1)
    assert lower < upper


def test_predict_interval_before_fit_raises():
    validator = SplitConformalValidator
    with pytest.raises(RuntimeError):
        validator.predict_interval()


def split_coverage_between_0_and_1(real_like_returns):
    validator = SplitConformalValidator().fit(real_like_returns)
    cov = validator.coverage(real_like_returns, alpha=0.1)
    assert 0 <= cov <= 1


def test_split_coverage_near_nomial(real_like_returns):
    validator = SplitConformalValidator().fit(real_like_returns)
    cov = validator.coverage(real_like_returns, alpha=0.1)
    assert cov >= 0.85
