"""A snippet for risk metrics calculations. WIP."""

import numpy as np
from scipy.stats import kurtosis, norm, skew


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


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, freq: int = 252) -> float:
    """
    Deflated Sharpe Ratio.

    Params:
    returns:  1-D array of returns
    n_trials: Number of independent trails
    freq:     periods per year - 252 daily, 52 weekly, 12 monthly
    """
    expected_max_sr = (1 - 0.5772) * norm.ppf(1 - 1 / n_trials) + 0.5772 * norm.ppf(
        1 - 1 / (n_trials * np.e)
    )
    sr = sharpe_ratio(returns, freq)
    skewness = skew(returns)
    excess_kurt = kurtosis(returns, fisher=True)

    under_sqrt = (len(returns) - 1) / (
        1 - skewness * sr + ((excess_kurt - 1) / 4) * sr**2
    )
    if under_sqrt <= 0:
        return 0.0
    sr_adjusted = sr * np.sqrt(under_sqrt)
    return float(norm.cdf(sr_adjusted - expected_max_sr))
