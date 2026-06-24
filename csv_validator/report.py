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
