# %% [markdown]
# # Conformal Strategy Validator - Full Pipeline Demo
#
# This example runs a complete validation on two synthetic strategies:
# - Strategy A: genuine edge (positive drift, consistent)
# - Strategy B: no edge (pure noise, similar Sharpe by luck)
#
# The validator should distinguish between them.

# %% Imports
import numpy as np
from csv_validator.report import validate

# %% Generate strategies
rng = np.random.default_rng(100)

# Strategy A: real edge - consistent positive drift
strategy_a = rng.normal(loc=0.0008, scale=0.01, size=504)

# Strategy B: no edge - noise, similar mean by luck
strategy_b = rng.normal(loc=0.0001, scale=0.015, size=504)

# %% Run validation
print("Validating Strategy A...")
report_a = validate(strategy_a, n_trials=50)
print(report_a.summary())

print("\nValidating Strategy B...")
report_b = validate(strategy_b, n_trials=50)
print(report_b.summary())

# %% Compare key metrics
print("\n=== Comparison ===")
print(f"{'Metric':<25} {'Strategy A':>12} {'Strategy B':>12}")
print("-" * 50)
metrics = [
    ("Sharpe", report_a.sharpe, report_b.sharpe),
    ("Max Drawdown", report_a.max_drawdown, report_b.max_drawdown),
    ("DSR", report_a.dsr, report_b.dsr),
    (
        "Circular p-value",
        report_a.circular_shift_pvalue,
        report_b.circular_shift_pvalue,
    ),
    ("PBO", report_a.pbo, report_b.pbo),
    ("CP Score", report_a.final_cp_score, report_b.final_cp_score),
]
for name, a, b in metrics:
    print(f"{name:<25} {a:>12.3f} {b:>12.3f}")
