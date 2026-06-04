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
    Maximum Drawdown (returned as positive number.)

    Params:
    returns: 1-D array of period returns
    """
    equity_curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    return float(-np.min(drawdown))


def calmar_ratio(returns: np.ndarray, freq: int = 252) -> float:
    """
    Calmar Ratio.

    Params:
    returns: 1-D array of returns.
    freq:    periods per year - 252 daily, 52 weekly, 12 monthly
    """
    annualised_return = (1 + np.mean(returns)) ** freq - 1
    md = max_drawdown(returns)
    if md == 0:
        return 0.0
    return float(annualised_return / md)


def sortino_ratio(returns: np.ndarray, freq: int = 252) -> float:
    """
    Sortino Ratio.

    Params:
    returns: 1-D array of returns.
    freq:    periods per year - 252 daily, 52 weekly, 12 monthly
    """
    annualised_return = (1 + np.mean(returns)) ** freq - 1
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return 0.0
    downside_deviation = np.std(negative_returns, ddof=1) * np.sqrt(freq)
    if downside_deviation == 0:
        return 0.0
    return float(annualised_return / downside_deviation)
