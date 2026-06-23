"""csv_validator/regime.py - Bayesian changepoint detection and conformal kill switch."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

from csv_validator.conformal import AdaptiveConformalValidator


class BOCPDDetector:
    def __init__(
        self, hazard_rate=1 / 250, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=0.01
    ):
        self.H = hazard_rate
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        # log probabilites of each run length - starts with run length 0
        self._log_probs = np.array([0.0])  # log(1.0) = 0
        # sufficient statistics for each run length
        self._n = np.array([0])  # number of observations in run
        self._mu = np.array([mu0])  # posterior mean
        self._kappa = np.array([kappa0])
        self._alpha = np.array([alpha0])
        self._beta = np.array([beta0])

    def changepoint_score(self, recent_window: int = 10) -> float:
        """
        Probability that a changepoint occured within the last 'recent_window' steps.
        """
        probs = np.exp(self._log_probs)
        return float(np.sum(probs[:recent_window]))

    def _log_predictive(self, x: float) -> np.ndarray:
        df = 2 * self._alpha
        loc = self._mu
        scale = np.sqrt(self._beta * (self._kappa + 1) / (self._alpha * self._kappa))
        return student_t.logpdf(x, df=df, loc=loc, scale=scale)

    def _update_params(self, x: float):
        # for each existing run, update the NIG posterior
        kappa_new = self._kappa + 1
        mu_new = (self._kappa * self._mu + x) / kappa_new
        alpha_new = self._alpha + 0.5
        beta_new = self._beta + (self._kappa * (x - self._mu) ** 2) / (2 * kappa_new)

        # prepend fresh prior for run length 0 (new regime)
        self._n = np.concatenate([[0], self._n + 1])
        self._mu = np.concatenate([[self.mu0], mu_new])
        self._kappa = np.concatenate([[self.kappa0], kappa_new])
        self._alpha = np.concatenate([[self.alpha0], alpha_new])
        self._beta = np.concatenate([[self.beta0], beta_new])

    def update(self, x: float) -> float:
        # 1. Compute log predictive probability for each current run length
        log_pred = self._log_predictive(x)

        # 2. Compute log joint: log P(r_{t-1}) + log P(x_t | r)
        log_joint = self._log_probs + log_pred

        # 3. changepoint prob: sum of all runs * hazard, reset to run length 0
        log_cp = logsumexp(log_joint) + np.log(self.H)

        # 4. growth probs: each run grows by 1, multiplied by (1-H)
        log_growth = log_joint + np.log(1 - self.H)

        # 5. new log probs: prepend changepoint prob, append growth probs
        self._log_probs = np.concatenate([[log_cp], log_growth])

        # 6. Normalise
        self._log_probs -= logsumexp(self._log_probs)

        # 7. Update sufficient statistics for each run length
        self._update_params(x)

        # 8. Return changepoint probability
        return float(np.exp(self._log_probs[0]))


class ConformalKillSwitch:
    def __init__(
        self,
        bocpd: BOCPDDetector,
        validator: AdaptiveConformalValidator,
        cp_threshold: float = 0.5,
        coverage_threshold: float = 0.85,
    ):
        self.bocpd = bocpd
        self.validator = validator
        self.cp_threshold = cp_threshold
        self.coverage_threshold = coverage_threshold

    def update(self, x: float, recent_returns: np.ndarray) -> bool:
        """
        Returns TRUE kill signal when BOTH:
        - BOCPD changepoint score > cp_threshold
        - Conformal coverage of recent_returns < coverage_threshold
        """
        self.bocpd.update(x)
        cp_score = self.bocpd.changepoint_score()
        coverage = self.validator.coverage(recent_returns)
        return bool(cp_score > self.cp_threshold and coverage < self.coverage_threshold)
