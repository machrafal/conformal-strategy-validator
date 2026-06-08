"""csv_validator/permutation.py — distribution-free permutation tests for strategy validation."""

import numpy as np

from csv_validator.metrics import sharpe_ratio


def circular_shift_test(
    returns: np.ndarray, n_trials: int = 1000, freq: int = 252, seed: int = 100
) -> tuple[float, np.ndarray, float]:
    """
    Circular shift permutation test.

    Rotates the return series by a random amount n_trials times,
    computing Sharpe on each shuffle to build a null distribution.

    Params:
    returns     : 1-D array of period returns
    n_trials    : number of shuffles
    freq        : periods per year - 252 daily, 52 weekly, 12 monthly
    seed        : random seed for reproducibility

    Returns:
    p_value:            : fraction of null Sharpes >= observed Sharpe
    null_distribution   : Sharpe ratios from shuffled series
    observed_stat       : Sharpe ratio of original returns

    """
    rng = np.random.default_rng(seed)
    observed_stat = sharpe_ratio(returns, freq)
    null_distribution = np.empty(n_trials)

    for i in range(n_trials):
        shift = rng.integers(1, len(returns))
        shuffled = np.roll(returns, shift)
        null_distribution[i] = sharpe_ratio(shuffled, freq)

    p_value = float(np.mean(null_distribution >= observed_stat))
    return p_value, null_distribution, observed_stat
