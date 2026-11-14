"""Multiple-testing-aware performance statistics.

Bailey and Lopez de Prado, "The Deflated Sharpe Ratio" (2014):
- PSR(SR*) = P(true Sharpe > SR*) given the observed per-period Sharpe, sample length T, and the
  skew and kurtosis of the returns (non-normal returns widen the error of a Sharpe estimate).
- When N variants were tried, the best observed Sharpe is biased upward. Under the null (no edge),
  the expected maximum of N Sharpe estimates with cross-trial variance V is
      SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e))),  g = Euler-Mascheroni.
- DSR = PSR(SR0): the probability the selected variant's Sharpe beats what luck alone would give.

Sharpe here is per trade (mean R / std R), not annualized; the formulas hold for any period.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1))


def _moments(r: np.ndarray) -> tuple[float, float]:
    """Skewness and (non-excess) kurtosis."""
    z = (r - r.mean()) / r.std(ddof=0)
    return float((z ** 3).mean()), float((z ** 4).mean())


def psr(r: np.ndarray, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe ratio: P(true per-period Sharpe > sr_star)."""
    r = np.asarray(r, dtype=float)
    sr = sharpe(r)
    if not math.isfinite(sr):
        return float("nan")
    skew, kurt = _moments(r)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return _N.cdf((sr - sr_star) * math.sqrt(r.size - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, trial_var: float) -> float:
    """Expected best Sharpe among n_trials unskilled variants (SR0 in the module docstring)."""
    if n_trials < 2 or trial_var <= 0:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(trial_var) * ((1 - g) * _N.inv_cdf(1 - 1 / n_trials)
                                   + g * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(r: np.ndarray, trial_sharpes: list[float]) -> dict:
    """DSR of the chosen variant `r`, deflated for every variant tried (`trial_sharpes`, NaNs dropped)."""
    t = np.array([x for x in trial_sharpes if math.isfinite(x)])
    sr0 = expected_max_sharpe(t.size, float(t.var(ddof=1)) if t.size > 1 else 0.0)
    return {"sharpe": round(sharpe(r), 4), "trials": int(t.size), "sr0": round(sr0, 4),
            "psr_vs_0": round(psr(r, 0.0), 4), "dsr": round(psr(r, sr0), 4)}
