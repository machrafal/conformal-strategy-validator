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
    for r in real_like_returns:
        detector.update(r)
    cp_score = detector.changepoint_score(recent_window=10)
    assert 0 <= cp_score <= 1


def test_regime_shift():
    rng = np.random.default_rng(100)
    regime1 = rng.normal(loc=0.0, scale=0.001, size=200)
    regime2 = rng.normal(loc=0.05, scale=0.05, size=100)

    detector = BOCPDDetector(hazard_rate=1 / 50, beta0=0.0001)
    validator = AdaptiveConformalValidator()
    kill_switch = ConformalKillSwitch(detector, validator)

    for r in regime1:
        kill_switch.update(r, regime1[-50:])

    signals = [
        kill_switch.update(r, regime2[: i + 1] if i > 0 else regime1[-50:])
        for i, r in enumerate(regime2)
    ]

    assert any(signals)


def test_kill_switch_stable_regime_no_signal(real_like_returns):
    detector = BOCPDDetector(hazard_rate=1 / 50, beta0=0.0001)
    validator = AdaptiveConformalValidator()
    kill_switch = ConformalKillSwitch(detector, validator)

    warmup = real_like_returns[:100]
    for r in warmup:
        kill_switch.update(r, warmup)

    # check stable period - should not fire consistently
    signals = [
        kill_switch.update(r, real_like_returns[i : i + 50])
        for i, r in enumerate(real_like_returns[100:])
    ]
    # not ALL signals should be True in a stable regime
    assert not all(signals)
