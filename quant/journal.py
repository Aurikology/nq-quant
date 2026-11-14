"""Score a discretionary trader's replayed trades with the same statistics as the mechanical replay.

Input: an FX Replay analytics export (one row per trade; R per trade in `avgRiskReward`, best
excursion in R in `maxRiskReward`). The export has no session column, but every session keeps its
own running balance, so trades are linked into sessions by balance chains: a trade belongs to the
session whose last balance plus this trade's realized P&L equals the trade's new balance.

Sessions are excluded only by what was traded (`exclude_final` lists the final balances of sessions
that ran a different strategy), never by how they did. Breakevens are counted as trades: a
win rate that drops them overstates the hit rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stats import psr, sharpe

BE = 0.05  # |R| at or under this is a breakeven


def load_fxreplay(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["status"] == "closed"].reset_index(drop=True)
    d["r"] = pd.to_numeric(d["avgRiskReward"], errors="coerce")
    d["mfe_r"] = pd.to_numeric(d["maxRiskReward"], errors="coerce")
    d["session"] = link_sessions(d)
    return d


def link_sessions(d: pd.DataFrame, tol: float = 0.05) -> np.ndarray:
    """Session id per row by balance chains (see the module docstring)."""
    chains: list[list[float]] = []  # [initial balance, current balance]
    out = np.empty(len(d), dtype=int)
    for i, (init, pnl, bal) in enumerate(zip(d["initialBalance"], d["rPnL"], d["currentRealizedBalance"])):
        k = next((k for k, c in enumerate(chains) if c[0] == init and abs(c[1] + pnl - bal) < tol), None)
        if k is None:
            chains.append([init, init])
            k = len(chains) - 1
        chains[k][1] = bal
        out[i] = k
    return out


def sessions(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("session")
    return pd.DataFrame({
        "trades": g.size(), "first": g["dateStart"].first().str[:10], "last": g["dateStart"].last().str[:10],
        "final_balance": g["currentRealizedBalance"].last().round(2),
        "mean_r": g["r"].mean().round(3),
        "wins": g["r"].apply(lambda x: int((x > BE).sum())),
        "losses": g["r"].apply(lambda x: int((x < -BE).sum())),
        "breakevens": g["r"].apply(lambda x: int((x.abs() <= BE).sum())),
    })


def ci(values: np.ndarray, groups: np.ndarray | None = None, boot: int = 5000, seed: int = 11) -> list[float]:
    """95% bootstrap CI on the mean; with `groups`, whole sessions are resampled (block bootstrap)."""
    rng = np.random.default_rng(seed)
    if groups is None:
        means = rng.choice(values, size=(boot, values.size), replace=True).mean(axis=1)
    else:
        keys = np.unique(groups)
        sums = np.array([values[groups == k].sum() for k in keys])
        counts = np.array([(groups == k).sum() for k in keys])
        pick = rng.integers(0, keys.size, size=(boot, keys.size))
        means = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def score(d: pd.DataFrame, exclude_final: tuple[float, ...] = ()) -> dict:
    s = sessions(d)
    drop = s.index[s["final_balance"].isin([round(x, 2) for x in exclude_final])]
    k = d[~d["session"].isin(drop) & d["r"].notna()]
    r, grp = k["r"].to_numpy(), k["session"].to_numpy()
    wins, losses = int((r > BE).sum()), int((r < -BE).sum())
    top = np.sort(r)[::-1]
    return {
        "trades": int(r.size), "sessions": int(np.unique(grp).size), "excluded_sessions": int(drop.size),
        "from": str(k["dateStart"].min())[:10], "to": str(k["dateStart"].max())[:10],
        "wins": wins, "losses": losses, "breakevens": int(r.size - wins - losses),
        "win_rate_all": round(wins / r.size, 3), "win_rate_ex_be": round(wins / max(1, wins + losses), 3),
        "mean_r": round(float(r.mean()), 3), "median_r": round(float(np.median(r)), 3),
        "total_r": round(float(r.sum()), 2),
        "ci95_trades": ci(r), "ci95_sessions": ci(r, grp),
        "mean_r_without_top3": round(float(top[3:].mean()), 3) if r.size > 3 else None,
        "mean_mfe_r": round(float(k["mfe_r"].mean()), 3),
        "worst_losing_streak": _streak(r),
        "sharpe_per_trade": round(sharpe(r), 4),
        "psr_vs_0": round(psr(r, 0.0), 4),  # P(true per-trade Sharpe > 0), skew/kurtosis adjusted
    }


def _streak(r: np.ndarray) -> int:
    best = cur = 0
    for x in r:
        cur = cur + 1 if x < -BE else 0
        best = max(best, cur)
    return best
