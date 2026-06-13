"""csv_validator/conformal.py - conformal prediction intervals for strategy validation."""

import numpy as np


class SplitConformalValidator:
    def __init__(self):
        self._scores = None
        self._mu = None

    def fit(self, calibration_returns: np.ndarray) -> "SplitConformalValidator":
        self._mu = np.mean(calibration_returns)
        self._scores = np.abs(calibration_returns - self._mu)
        return self  # return self so you can chain validator.fit(cal).predict_interval(0.1)

    def predict_interval(self, alpha: float = 0.1) -> tuple[float, float]:
        """
        Prediction interval at significance level alpha.

        Params:
        alpha: significance level (default 0.1 gives 90% coverage)

        Returns:
        (lower, upper) bounds of the prediction interval.
        """
        if self._scores is None:
            raise RuntimeError("Call fit() before predict_interval().")
        q = np.quantile(self._scores, 1 - alpha)
        return float(self._mu - q), float(self._mu + q)

    def coverage(self, test_returns: np.ndarray, alpha: float = 0.1) -> float:
        lower, upper = self.predict_interval(alpha)
        coverage = (test_returns >= lower) & (test_returns <= upper)
        return float(np.mean(coverage))


class AdaptiveConformalValidator:
    def fit_transform(
        self, returns: np.ndarray, alpha: float = 0.1, window: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(len(returns), np.nan)
        upper = np.full(len(returns), np.nan)
        for t in range(window, len(returns)):
            window_returns = returns[t - window : t]
            mu = np.mean(window_returns)
            scores = np.abs(window_returns - mu)
            q = np.quantile(scores, 1 - alpha)
            lower[t] = mu - q
            upper[t] = mu + q
        return lower, upper

    def coverage(
        self, returns: np.ndarray, alpha: float = 0.1, window: int = 50
    ) -> float:
        lower, upper = self.fit_transform(returns, alpha, window)
        valid = ~np.isnan(lower)
        covered = (returns[valid] >= lower[valid]) & (returns[valid] <= upper[valid])
        return float(np.mean(covered))
