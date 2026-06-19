import matplotlib.pyplot as plt
import numpy as np

from csv_validator.regime import BOCPDDetector

rng = np.random.default_rng(100)

# Regime 1: low vol, mean ~ 0
regime1 = rng.normal(loc=0.001, scale=0.005, size=100)

# Regime 2: high vol, negative mean - clear shift
regime2 = rng.normal(loc=-0.005, scale=0.02, size=100)

returns = np.concatenate([regime1, regime2])

detector = BOCPDDetector(hazard_rate=1 / 50, beta0=0.0001)
cp_probs = [detector.update(r) for r in returns]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(returns)
ax1.axvline(100, color="red", linestyle="--", label="True changepoint")
ax1.set_ylabel("Returns")
ax1.legend()
ax2.plot(cp_probs)
ax2.axvline(100, color="red", linestyle="--")
ax2.axhline(0.5, color="orange", linestyle=":", label="Threshold 0.5")
ax2.set_ylabel("P(changepoint)")
ax2.legend()
plt.tight_layout()
