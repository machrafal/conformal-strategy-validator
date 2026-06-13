import numpy as np

from csv_validator.metrics import sharpe_ratio


class SplitConformalValidator:
    def __init__(self):
        self._scores = None
        self._mu = None

    def fit(self, calibration_returns: np.ndarray) -> "SplitConformalValidator":
        self._mu = np.mean(calibration_returns)
        self._scores = np.abs(calibration_returns - self._mu)
        return self # return self so you can chain validator.fit(cal).predict_interval(0.1)
    
    def predict_interval(self, alpha:float=0.1)->tuple[float,float]:
        if self._scores is None:
            raise RuntimeError("Call fit() before predict_interval().")
        q = np.quantile(self._scores,1-alpha)
        return float(self._mu - q), float(self._mu+q)

        
    def coverage(self, test_returns:np.ndarray,alpha:float=0.1)->float:
        # for each test return, check if it falls inside predict_intreval(alpha)
        # return fraction that are covered
        # ...
        

        
