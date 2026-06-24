from dataclasses import dataclass, field

import numpy as np


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
