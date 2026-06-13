import numpy as np
import pytest


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
