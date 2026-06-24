from dataclasses import dataclass

import numpy as np

from csv_validator.conformal import AdaptiveConformalValidator, SplitConformalValidator
from csv_validator.metrics import (
    calmar_ratio,
    deflated_sharpe_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from csv_validator.pbo import generate_cpcv_splits, probability_of_backtest_overfitting
from csv_validator.permutation import (
    circular_shift_test,
    sign_flip_test,
    stationary_bootstrap_test,
)
from csv_validator.regime import BOCPDDetector


@dataclass
class ValidationReport:
    # metrics
    sharpe: float = float("nan")
    max_drawdown: float = float("nan")
    calmar: float = float("nan")
    sortino: float = float("nan")
    dsr: float = float("nan")

    # permuation tests
    circular_shift_pvalue: float = float("nan")
    sign_flip_pvalue: float = float("nan")
    bootstrap_pvalue: float = float("nan")

    # conformal
    split_coverage: float = float("nan")
    adaptive_coverage: float = float("nan")

    # PBO
    pbo: float = float("nan")

    # regime
    final_cp_score: float = float("nan")
    kill_signal: bool = False

    def summary(self) -> str:
        """Print a human-readable summary."""
        lines = [
            "=== Validation Report ===",
            f"Sharpe:                {self.sharpe:.3f}",
            f"Max Drawdown:          {self.max_drawdown:.3f}",
            f"Calmar:                {self.calmar:.3f}",
            f"Sortino:               {self.sortino:.3f}",
            f"Deflated Sharpe:       {self.dsr:.3f}",
            "",
            f"Circular shift p:      {self.circular_shift_pvalue:.3f}",
            f"Sign flip p:           {self.sign_flip_pvalue:.3f}",
            f"Bootstrap p:           {self.bootstrap_pvalue:.3f}",
            "",
            f"Split coverage:        {self.split_coverage:.3f}",
            f"Adaptive coverage:     {self.adaptive_coverage:.3f}",
            "",
            f"PBO:                   {self.pbo:.3f}",
            "",
            f"Changepoint score:     {self.final_cp_score:.3f}",
            f"Kill signal:           {self.kill_signal}",
        ]
        return "\n".join(lines)


def validate(
    returns: np.ndarray,
    n_trials: int = 50,
    freq: int = 252,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 5,
) -> ValidationReport:
    """
    Run the full validation pipeline on a return series.

    Params:
    returns         : 1-D array of period returns
    n_trials        : number of strategies tested (for DSR)
    freq            : periods per year
    n_groups        : CPCV groups
    n_test_groups   : CPCV test groups
    embargo         : CPCV embargo period

    Returns:
    ValidationReport with all results populated
    """
    report = ValidationReport()

    # 1. Metrics
    report.sharpe = sharpe_ratio(returns, freq)
    report.max_drawdown = max_drawdown(returns)
    report.calmar = calmar_ratio(returns, freq)
    report.sortino = sortino_ratio(returns, freq)
    report.dsr = deflated_sharpe_ratio(returns, n_trials, freq)

    # 2. Permutation tests
    p_circ, _, _ = circular_shift_test(returns, freq=freq)
    report.circular_shift_pvalue = p_circ

    p_sign, _, _ = sign_flip_test(returns, n_trials=n_trials, freq=freq)
    report.sign_flip_pvalue = p_sign

    p_stat_bootstrap, _, _ = stationary_bootstrap_test(
        returns, block_size=10, n_trials=n_trials, freq=freq
    )
    report.bootstrap_pvalue = p_stat_bootstrap

    # 3. Conformal
    split_val = SplitConformalValidator().fit(returns[: len(returns) // 2])
    report.split_coverage = split_val.coverage(returns[len(returns) // 2 :])

    adapt_val = AdaptiveConformalValidator()
    report.adaptive_coverage = adapt_val.coverage(returns)
