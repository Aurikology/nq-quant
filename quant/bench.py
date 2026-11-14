"""Reproducible timings for the README's speed claims: `python -m quant bench`.

Each benchmark times the old algorithm and its replacement on the same bars and asserts they
return the same answer before reporting a speedup, so a number here is never for a wrong result.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .liquidity import FVG, PoolIndex, _first, _ohlc, find_fvgs, find_pools


def _find_fvgs_scan(bars: pd.DataFrame, cut: int, min_gap: float = 1.0, max_age: int = 96) -> list[FVG]:
    """The original tracker: every state change found by scanning the bars after the gap."""
    o, h, l, c = _ohlc(bars, cut)
    out = []
    for side, idxs in (("bull", np.flatnonzero(l[2:] - h[:-2] >= min_gap) + 2),
                       ("bear", np.flatnonzero(l[:-2] - h[2:] >= min_gap) + 2)):
        for i in idxs:
            lo, hi = (float(h[i - 2]), float(l[i])) if side == "bull" else (float(h[i]), float(l[i - 2]))
            s, ce, end = i + 1, (lo + hi) / 2, min(len(c), i + 1 + max_age)
            if side == "bull":
                mit, ceb, thru = _first(l[s:] <= hi, s), _first(l[s:] <= ce, s), _first(l[s:] <= lo, s)
                inv = _first((c[s:end] < lo) & (c[s:end] < o[s:end]), s)
                kill = _first(c[inv + 1:] > hi, inv + 1) if inv is not None else None
            else:
                mit, ceb, thru = _first(h[s:] >= lo, s), _first(h[s:] >= ce, s), _first(h[s:] >= hi, s)
                inv = _first((c[s:end] > hi) & (c[s:end] > o[s:end]), s)
                kill = _first(c[inv + 1:] < lo, inv + 1) if inv is not None else None
            state = ("inverted" if inv is not None else "filled" if thru is not None else
                     "ce" if ceb is not None else "mitigated" if mit is not None else "open")
            out.append(FVG(side, lo, hi, int(i), state, mit, ceb, thru, inv, kill))
    out.sort(key=lambda g: (g.formed, g.side))
    return out


def _time(fn, repeat: int = 3):
    best, out = float("inf"), None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def run(bars: pd.DataFrame, cuts: int = 40) -> list[dict]:
    rows = []
    n = len(bars)
    t_old, a = _time(lambda: _find_fvgs_scan(bars, n, 1.0, 10 ** 6), 1)
    t_new, b = _time(lambda: find_fvgs(bars, n, 1.0, 10 ** 6))
    assert a == b, "find_fvgs disagrees with the scan"
    rows.append({"what": f"FVG states, all gaps on {n:,} bars", "old_s": t_old, "new_s": t_new})

    picks = np.linspace(n // 4, n, cuts).astype(int)
    t_old, a = _time(lambda: [find_pools(bars, int(c)) for c in picks], 1)
    t_new, b = _time(lambda: (lambda ix: [ix.pools_at(int(c)) for c in picks])(PoolIndex(bars)))
    assert a == b, "PoolIndex disagrees with find_pools"
    rows.append({"what": f"liquidity pools at {cuts} cuts on {n:,} bars (index build included)",
                 "old_s": t_old, "new_s": t_new})
    for r in rows:
        r["speedup"] = round(r["old_s"] / r["new_s"], 1)
        r["old_s"], r["new_s"] = round(r["old_s"], 3), round(r["new_s"], 3)
    return rows
