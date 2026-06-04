"""A snippet for risk metrics calculations. WIP."""

import numpy as np


def sharpe_ratio(returns: np.ndarray, freq: int = 252) -> float:
    """
    Annualised Sharpe Ratio (risk-free rate assumed 0).

    Params:
    returns: 1-D array of period returns
    freq:    periods per year - 252 daily, 52 weekly, 12 monthly
    """
    mu = np.mean(returns)
    sd = np.std(returns, ddof=1)
    if sd == 0:
        return 0.0
    return float((mu / sd) * np.sqrt(freq))


def max_drawdown(returns: np.ndarray) -> float:
    """
    Maximum Drawdown.

    Params:
    returns: 1-D array of period returns
    """
    equity_curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return float(-np.min(drawdown))
