# conformal-strategy-validator

A production-grade strategy validation toolkit for algorithmic trading. Applies distribution-free statistical methods to detect overfitting and quantify out-of-sample reliability — without making assumptions about return distributions.

## Motivation

Most strategy validation tools rely on metrics like the Sharpe ratio and standard backtesting. These have two well-documented failure modes:

1. **Selection bias** — testing many strategy variants and picking the best inflates performance metrics even when no genuine edge exists
2. **Distribution assumptions** — confidence intervals and significance tests based on normality break down on financial returns, which exhibit fat tails, skewness, and regime changes

This library addresses both problems using methods that are either assumption-free or make only weak assumptions that financial returns actually satisfy.

## Methods

### Performance Metrics (`metrics.py`)
Standard risk-adjusted return metrics plus the **Deflated Sharpe Ratio** (Bailey & López de Prado, 2012), which corrects for the number of strategy variants tested:

$$DSR = \Phi\left(\hat{SR}_{adj} - E[\max SR]\right)$$

Where $\Phi$ is the normal CDF and $E[\max SR]$ is the expected maximum Sharpe from random search across $N$ trials.

### Permutation Tests (`permutation.py`)
Three distribution-free tests that answer: *could this result be luck?*

| Test | What it shuffles | What it tests |
|------|-----------------|---------------|
| Circular shift | Rotates series by random offset | Signal/return alignment |
| Sign flip | Randomly negates each return | Directional edge |
| Stationary bootstrap | Resamples geometric-length blocks | General edge, preserving autocorrelation |

All three return a p-value, null distribution, and observed statistic with a consistent interface.

### Conformal Prediction (`conformal.py`)
Distribution-free prediction intervals with a **finite-sample coverage guarantee**:

$$P(r_{t+1} \in \hat{C}_{1-\alpha}) \geq 1 - \alpha$$

This holds for any return distribution — no normality required. Two variants:
- **Split conformal**: single calibration/test split, simple and fast
- **Adaptive conformal**: rolling window recalibration, adapts to distribution shift

### Probability of Backtest Overfitting (`pbo.py`)
Implements the CPCV framework (López de Prado, 2018) to quantify the probability that a selected strategy underperforms the median strategy out-of-sample:

$$PBO = \frac{1}{S} \sum_{s=1}^{S} \mathbf{1}[\lambda_s < 0]$$

Where $\lambda_s = \text{logit}(\omega_s)$ and $\omega_s$ is the relative OOS rank of the IS-best strategy in split $s$. Uses purging and embargo to prevent information leakage across train/test boundaries.

### Bayesian Changepoint Detection (`regime.py`)
Online regime detection using **BOCPD** (Adams & MacKay, 2007) with a Normal-Inverse-Gamma conjugate prior. Tracks the posterior distribution over run lengths and detects structural breaks in the return distribution as they happen.

Wrapped in a **dual-trigger kill switch** that fires only when both:
- Changepoint probability exceeds threshold
- Conformal coverage drops below nominal

This dual-trigger design reduces false positives compared to either signal alone.

## Installation

```bash
git clone https://github.com/machrafal/conformal-strategy-validator
cd conformal-strategy-validator
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from csv_validator.report import validate

# your strategy's daily returns
returns = np.array([...])

report = validate(returns, n_trials=50)
print(report.summary())
```

Output:
```
=== Validation Report ===
Sharpe:                1.351
Max Drawdown:          0.168
Calmar:                1.425
Sortino:               2.685
Deflated Sharpe:       1.000

Circular shift p:      0.514
Sign flip p:           0.040
Bootstrap p:           0.560

Split coverage:        0.869
Adaptive coverage:     0.879

PBO:                   0.533

Changepoint score:     0.019
Kill signal:           False
```

## Interpreting results

| Metric | Good signal | Weak signal |
|--------|------------|-------------|
| DSR | > 0.95 | < 0.95 |
| Permutation p-values | < 0.05 | > 0.10 |
| Conformal coverage | ≥ nominal (1-α) | < nominal |
| PBO | < 0.10 | > 0.40 |
| Changepoint score | < 0.20 | > 0.50 |

## Using individual modules

```python
from csv_validator.metrics import sharpe_ratio, deflated_sharpe_ratio
from csv_validator.permutation import circular_shift_test
from csv_validator.conformal import SplitConformalValidator, AdaptiveConformalValidator
from csv_validator.pbo import generate_cpcv_splits, probability_of_backtest_overfitting
from csv_validator.regime import BOCPDDetector, ConformalKillSwitch

# Permutation test
p_value, null_dist, observed = circular_shift_test(returns, n_trials=1000)

# Conformal intervals
validator = SplitConformalValidator().fit(calibration_returns)
lower, upper = validator.predict_interval(alpha=0.1)
coverage = validator.coverage(test_returns)

# Regime detection
detector = BOCPDDetector(hazard_rate=1/250, beta0=0.0001)
for r in returns:
    cp_prob = detector.update(r)
score = detector.changepoint_score(recent_window=10)
```

## Running the full pipeline example

```bash
python examples/full_pipeline.py
```

## Tests

```bash
pytest -v
```

```
46 passed, 99% coverage
```

## References

- Adams, R.P. & MacKay, D.J.C. (2007). *Bayesian Online Changepoint Detection*
- Angelopoulos, A. & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*
- Bailey, D. & López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier*
- Bailey, D. et al. (2015). *The Probability of Backtest Overfitting*
- Gibbs, I. & Candès, E. (2021). *Adaptive Conformal Inference Under Distribution Shift*
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 12
- Politis, D. & Romano, J. (1994). *The Stationary Bootstrap*
