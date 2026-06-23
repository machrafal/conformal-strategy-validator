"""tests/test_regime.py - unit tests for csv_validator/regime.py"""

import numpy as np
import pytest

from csv_validator.conformal import AdaptiveConformalValidator, SplitConformalValidator
from csv_validator.regime import BOCPDDetector, ConformalKillSwitch


def test_bocpd_update_between_0_and_1(real_like_returns):
    detector = BOCPDDetector(hazard_rate=1 / 50, beta0=0.0001)
    result = detector.update(real_like_returns[0])
    assert 0 <= result <= 1


def test_changepoint_score_between_0_and_1(real_like_returns):
    detector = BOCPDDetector(hazard_rate=1 / 50, beta0=0.0001)
    result = detector.update(real_like_returns[0])
    assert 0 <= result <= 1
