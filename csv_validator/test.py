def update(self, x: float) -> float:
    # 1. compute log predictive probability for each current run length
    log_pred = self._log_predictive(x)

    # 2. compute log joint: log P(r_{t-1}) + log P(x_t | r)
    log_joint = self._log_probs + log_pred

    # 3. changepoint prob: sum of all runs * hazard, reset to run length 0
    log_cp = logsumexp(log_joint) + np.log(self.H)

    # 4. growth probs: each run grows by 1, multiplied by (1-H)
    log_growth = log_joint + np.log(1 - self.H)

    # 5. new log probs: prepend changepoint prob, append growth probs
    self._log_probs = np.concatenate([[log_cp], log_growth])

    # 6. normalise
    self._log_probs -= logsumexp(self._log_probs)

    # 7. update sufficient statistics for each run length
    self._update_params(x)

    # 8. return changepoint probability
    return float(np.exp(self._log_probs[0]))
