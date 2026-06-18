import numpy as np
from scipy.special import logsumexp


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
