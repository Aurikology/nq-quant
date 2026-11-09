"""Judas-swing replay: Claude's paper model (decision 2026-10-26), with exit variants side by side.

Research tool, not the live gate. Every setup is found bar by bar with no lookahead, and the same
entries are run through several exit models so they can be compared on identical fills.

The setup (`dol-and-liquidity.md` § the 8:30 Judas Swing; `killzones-and-sessions.md`):
- Direction: daily and weekly `htf_bias` must agree at the session start, else the session is passed.
- Sweep: inside the window, a bar trades beyond an unswept pool on the side AGAINST the bias (BSL
  for a short). `sweep_kinds` "significant" = session/PD/PW/EQ pools; "any" adds swing pivots.
- Shift: within `mss_bars` of the sweep, a body close beyond the last confirmed swing pivot before
  the sweep (a swing low for a short), with a bias-side FVG formed between the sweep and 2 bars
  after the shift. Box = the latest such gap, entry = its CE, stop = far edge + 2 ticks.
- Order: a limit at the CE, live from the bar after the box is known until filled, invalidated (a
  close beyond the far edge), or the session ends. Touch fill. Stops over `max_stop` are passed.
- Shorts are coded once; a long is the same code on mirrored bars (high/low swapped, negated).

Exit models, all in R of the full 2-contract position, flattened at the session's last close:
  ext      the whole position to the nearest significant pool >= 2R (the live rule today)
  pool2r   the whole position to the nearest pool of any kind >= 2R
  pool1    the whole position to the nearest pool of any kind >= 1R
  p1_be    half at the nearest pool >= 1R, stop to entry, the rest to `ext`
  p2r_be   half at 2R, stop to entry, the rest to `ext`
  p1_p2    half at the nearest pool >= 1R, stop to entry, the rest at the next pool beyond it
  be1r     the whole position to `ext`, stop to entry once 1R is reached
  p2r_trail  half at 2R, stop to entry, the rest to `ext` under a stop trailed behind confirmed swings
  p2r_free   half at 2R, stop to entry, the rest has no target and only the trailed stop
  p1_trail   half at the nearest pool >= 1R, stop to entry, the rest to `ext` under the trailed stop
  trail_1r   the whole position to `ext`, stop to entry at 1R, then trailed
Trailing: a swing confirms `TRAIL_PIVOT` bars late, the stop sits 2 ticks past it, is live from the
next bar and never widens. `Params.stop_slip` adds points to every stop fill (0 is the old behaviour).
Direction, `Params.htf`: "agree" = daily and weekly must agree; "sweep" = when they agree that side
only, when they disagree the sweep picks the side (decision 2026-11-05); "both" = either side always.
Ambiguous bars resolve against the trade: the stop is checked before any target, a target on the
fill bar does not count, and a stop moved to entry is live from the next bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA, TICK_SIZE
from .liquidity import find_fvgs, find_pools, htf_bias, swing_pivots
from .market import cache_path, read_csv

WINDOWS = {  # session: (start minute, end minute) ET; asia ends at midnight
    "asia": (20 * 60, 24 * 60),
    "ny_am": (8 * 60 + 30, 11 * 60),  # killzone from 08:30
    "ny_pm": (13 * 60 + 30, 16 * 60),
}
MODELS = ("ext", "pool2r", "pool1", "p1_be", "p2r_be", "p1_p2", "be1r",
          "p2r_trail", "p2r_free", "p1_trail", "trail_1r")
TRAIL_PIVOT = 2  # bars each side of a swing the trailing stop follows (default, not in the knowledge base)
POINT_VALUE = 20.0  # NQ
COMMISSION_RT = 5.0  # per contract round trip (default, not in the knowledge base)


@dataclass
class Params:
    sweep_kinds: str = "significant"  # "significant" | "any"
    mss_bars: int = 20
    max_stop: float = 25.0  # NQ points: 2 NQ at the $1,000 paper cap
    min_gap: float = 1.0
    contracts: int = 2
    max_trades: int = 2
    sessions: tuple = ("asia", "ny_am", "ny_pm")
    mss_pivot: int = 2  # pivot length of the swing the shift must close beyond
    stop_mode: str = "gap"  # "gap": 2 ticks past the box's far edge; "swing": 2 ticks past the sweep extreme
    entry_mode: str = "ce"  # "ce" | "edge" (the box's near edge)
    htf: str = "agree"  # "agree" | "sweep" | "both" (see the module docstring); "daily": daily alone (research only)
    stop_slip: float = 0.0  # points added to every stop fill (2026-11-05 live fill slipped 7.6)


@dataclass
class Trade:
    date: str
    session: str
    side: str
    sweep_kind: str
    sweep_time: str
    fill_time: str
    entry: float
    stop: float
    risk_pts: float
    gap_pts: float
    obstacles: int  # open bias-opposing gaps between entry and the ext target
    targets: dict
    r: dict = field(default_factory=dict)
    mfe_r: float = 0.0  # best excursion before the trade's ext-model exit
    mae_r: float = 0.0
    htf_agree: bool = True
    exit_time: str = ""  # the ext model's exit, used to keep one position at a time across sides


def _mirror(b: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"open": -b["open"], "high": -b["low"], "low": -b["high"], "close": -b["close"]},
                        index=b.index)


def _windows(b: pd.DataFrame, p: Params):
    """(session, first index, end index exclusive) for every complete-or-live window in the bars."""
    minute = b.index.hour * 60 + b.index.minute
    dates = pd.Index(b.index.date)
    out = []
    for d in sorted(set(dates)):
        wd = pd.Timestamp(d).weekday()
        for s in p.sessions:
            a, z = WINDOWS[s]
            if (s == "asia" and wd in (4, 5)) or (s != "asia" and wd >= 5):
                continue  # asia Sun-Thu evenings, NY Mon-Fri
            sel = np.flatnonzero((dates == d) & (minute >= a) & (minute < z))
            if sel.size > 5:
                out.append((s, int(sel[0]), int(sel[-1]) + 1))
    return out


def _pool_targets(pools, entry: float, risk: float):
    """Unswept SSL below the entry, nearest first, with their R distance."""
    below = sorted((pl for pl in pools if pl.side == "SSL" and pl.swept is None and pl.price < entry),
                   key=lambda pl: -pl.price)
    return [(pl, (entry - pl.price) / risk) for pl in below]


def _targets(pools, entry: float, risk: float) -> dict | None:
    cands = _pool_targets(pools, entry, risk)
    sig = [(pl, r) for pl, r in cands if pl.significant]
    ext = next((pl for pl, r in sig if r >= 2), None)
    if ext is None:
        return None
    first2 = next((pl for pl, r in cands if r >= 2), None)
    first1 = next((pl for pl, r in cands if r >= 1), None)
    second = None
    if first1 is not None:
        second = next((pl for pl, r in cands if pl.price < first1.price - TICK_SIZE), None)
    t = {"ext": (ext.price, ext.kind), "pool2r": (first2.price, first2.kind)}
    if first1 is not None:
        t["pool1"] = (first1.price, first1.kind)
        t["p2"] = (second.price, second.kind) if second is not None else (ext.price, ext.kind)
    return t


def structure(b: pd.DataFrame, pivot: int = 2) -> np.ndarray:
    """Market structure per bar, no lookahead: +1 after a close above the last confirmed swing high,
    -1 after a close below the last confirmed swing low, else unchanged (0 until the first break).
    A swing confirms `pivot` bars late. Same rule as the Structure row in pine/slash-ict.pine."""
    h, l, c = (b[k].to_numpy() for k in ("high", "low", "close"))
    state = np.zeros(len(b), dtype=int)
    last_h, last_l, cur = np.nan, np.nan, 0
    for j in range(len(b)):
        k = j - pivot
        if k - pivot >= 0:
            if h[k] == h[k - pivot:j + 1].max():
                last_h = h[k]
            if l[k] == l[k - pivot:j + 1].min():
                last_l = l[k]
        if not np.isnan(last_h) and c[j] > last_h:
            cur = 1
        elif not np.isnan(last_l) and c[j] < last_l:
            cur = -1
        state[j] = cur
    return state


def _exit(h, l, c, fill: int, end: int, entry: float, stop: float, legs: list, be_at: float | None,
          trail: int = 0, slip: float = 0.0):
    """R for a short. legs = [(fraction, target price or None)]; be_at = price that arms a stop at
    entry (a partial fill also arms it). trail > 0 follows confirmed swings once armed. Returns
    (R, last bar index)."""
    risk = stop - entry
    open_legs = list(legs)
    r, cur_stop, armed, last = 0.0, stop, False, end - 1
    for j in range(fill, end):
        if h[j] >= cur_stop:  # stop first: ambiguity goes against the trade
            r += sum(f for f, _ in open_legs) * (entry - cur_stop - slip) / risk
            return r, j
        if j == fill:
            continue  # a target on the fill bar does not count
        hit = [(f, t) for f, t in open_legs if t is not None and l[j] <= t]
        for f, t in hit:
            r += f * (entry - t) / risk
            open_legs.remove((f, t))
            armed = True  # a partial moves the stop to entry
        if not open_legs:
            return r, j
        if be_at is not None and l[j] <= be_at:
            armed = True
        if armed:
            cur_stop = min(cur_stop, entry)
            k = j - trail
            if trail and k - trail >= fill and h[k] == h[k - trail:j + 1].max():
                cur_stop = min(cur_stop, h[k] + 2 * TICK_SIZE)
    r += sum(f for f, _ in open_legs) * (entry - c[last]) / risk
    return r, last


def _session(b: pd.DataFrame, s: str, ws: int, we: int, p: Params, side: str) -> tuple[list[Trade], dict]:
    """Short-side logic on (possibly mirrored) bars b."""
    o, h, l, c = (b[k].to_numpy() for k in ("open", "high", "low", "close"))
    log = {"session": s, "date": str(b.index[ws].date()), "side": side, "sweeps": 0, "shifts": 0,
           "orders": 0, "passed": []}
    trades: list[Trade] = []
    pools0 = find_pools(b, ws)
    kinds = (lambda pl: pl.significant) if p.sweep_kinds == "significant" else (lambda pl: True)
    bsl = [pl for pl in pools0 if pl.side == "BSL" and pl.swept is None and kinds(pl)]
    i = ws
    while i < we and len(trades) < p.max_trades:
        hit = [pl for pl in bsl if h[i] > pl.price]
        if not hit:
            i += 1
            continue
        swept = max(hit, key=lambda pl: pl.price)
        bsl = [pl for pl in bsl if pl not in hit]
        log["sweeps"] += 1
        _, lows = swing_pivots(b, i, p.mss_pivot)
        if lows.size == 0:
            i += 1
            continue
        ref = l[lows[-1]]
        mss = next((j for j in range(i, min(we, i + p.mss_bars)) if c[j] < ref and c[j] < o[j]), None)
        if mss is None:
            i += 1
            continue
        log["shifts"] += 1
        cut = min(we, mss + 3)
        gaps = [g for g in find_fvgs(b.iloc[:cut], cut, p.min_gap)
                if g.side == "bear" and i <= g.formed <= mss + 2]
        if not gaps:
            log["passed"].append("shift without a bearish gap")
            i = mss + 1
            continue
        g = gaps[-1]
        live = max(g.formed, mss) + 1
        entry = g.ce if p.entry_mode == "ce" else g.low
        far = g.high if p.stop_mode == "gap" else max(g.high, float(h[i:mss + 1].max()))
        stop = far + 2 * TICK_SIZE
        risk = stop - entry
        if risk > p.max_stop:
            log["passed"].append(f"stop {risk:.2f} over {p.max_stop}")
            i = live
            continue
        pools = find_pools(b, live)
        tg = _targets(pools, entry, risk)
        if tg is None:
            log["passed"].append("no significant pool >= 2R")
            i = live
            continue
        log["orders"] += 1
        fill = None
        for j in range(live, we):
            if c[j - 1] > g.high and j - 1 >= live:  # invalidated before the fill
                break
            if h[j] >= entry:
                fill = j
                break
        if fill is None:
            log["passed"].append("not filled / invalidated")
            i = live
            continue
        ext_p = tg["ext"][0]
        obstacles = sum(1 for f in find_fvgs(b.iloc[:fill], fill, p.min_gap, 10 ** 6)
                        if f.side == "bull" and f.state in ("open", "mitigated") and f.high > ext_p and f.low < entry)
        t = Trade(str(b.index[ws].date()), s, side, swept.kind, str(b.index[i]), str(b.index[fill]),
                  entry, stop, risk, g.high - g.low, obstacles,
                  {k: v for k, v in tg.items()})
        two_r = entry - 2 * risk
        p1 = tg.get("pool1", (None,))[0]
        p2 = tg.get("p2", (None,))[0]
        plans = {
            "ext": ([(1.0, ext_p)], None),
            "pool2r": ([(1.0, tg["pool2r"][0])], None),
            "pool1": ([(1.0, p1)], None),
            "p1_be": ([(0.5, p1), (0.5, ext_p)], None),
            "p2r_be": ([(0.5, two_r), (0.5, ext_p)], None),
            "p1_p2": ([(0.5, p1), (0.5, p2)], None),
            "be1r": ([(1.0, ext_p)], entry - risk),
            "p2r_trail": ([(0.5, two_r), (0.5, ext_p)], None, TRAIL_PIVOT),
            "p2r_free": ([(0.5, two_r), (0.5, None)], None, TRAIL_PIVOT),
            "p1_trail": ([(0.5, p1), (0.5, ext_p)], None, TRAIL_PIVOT),
            "trail_1r": ([(1.0, ext_p)], entry - risk, TRAIL_PIVOT),
        }
        last = fill
        for m, plan in plans.items():
            legs, be = plan[0], plan[1]
            r, j = _exit(h, l, c, fill, we, entry, stop, legs, be,
                         plan[2] if len(plan) > 2 else 0, p.stop_slip)
            cost = p.contracts * COMMISSION_RT / (risk * POINT_VALUE * p.contracts)
            t.r[m] = round(r - cost, 3)
            if m == "ext":
                last = j
        seg = slice(fill + 1, last + 1)
        if last > fill:
            t.mfe_r = round((entry - l[seg].min()) / risk, 2)
            t.mae_r = round((h[seg].max() - entry) / risk, 2)
        t.htf_agree = True
        t.exit_time = str(b.index[last])
        trades.append(t)
        i = last + 1
    return trades, log


def run(b: pd.DataFrame, p: Params = Params()) -> tuple[list[Trade], list[dict]]:
    mb = _mirror(b)
    trades, logs = [], []
    for s, ws, we in _windows(b, p):
        lo = max(0, int(np.searchsorted(b.index, b.index[ws] - pd.Timedelta(days=16))))
        d, w, _ = htf_bias(b.iloc[lo:we], ws - lo)
        agree = d == w and d != 0
        if p.htf == "both" or (p.htf == "sweep" and not agree):
            sides = ["short", "long"]
        elif d == 0 or (p.htf in ("agree", "sweep") and not agree):
            logs.append({"session": s, "date": str(b.index[ws].date()), "side": None, "passed": ["HTF disagree"]})
            continue
        else:
            sides = ["short" if d < 0 else "long"]
        found = []
        for side in sides:
            src = b if side == "short" else mb
            t, log = _session(src.iloc[lo:we], s, ws - lo, we - lo, p, side)
            if side == "long":
                for x in t:  # back to real prices
                    x.entry, x.stop = -x.entry, -x.stop
                    x.targets = {k: (-v[0], _flip(v[1])) for k, v in x.targets.items()}
                    x.sweep_kind = _flip(x.sweep_kind)
            for x in t:
                x.htf_agree = agree
            found += t
            logs.append(log)
        # One position at a time and the session's trade cap, across both sides, in fill order.
        found.sort(key=lambda x: x.fill_time)
        kept, busy_until = [], ""
        for x in found:
            if len(kept) < p.max_trades and x.fill_time > busy_until:
                kept.append(x)
                busy_until = x.exit_time
        trades += kept
    return trades, logs


def _flip(kind: str) -> str:
    """A pool kind found on mirrored bars, named for the real chart."""
    swap = {"_H": "_L", "_L": "_H", "EQH": "EQL", "EQL": "EQH", "PDH": "PDL", "PDL": "PDH", "PWH": "PWL", "PWL": "PWH"}
    for a, z in swap.items():
        if kind.endswith(a):
            return kind[: -len(a)] + z
    return kind


def summary(trades: list[Trade]) -> dict:
    out = {}
    for m in MODELS:
        r = np.array([t.r[m] for t in trades])
        if r.size == 0:
            continue
        wins = r[r > 0.05]
        out[m] = {"n": int(r.size), "total_r": round(float(r.sum()), 2), "mean_r": round(float(r.mean()), 3),
                  "win_rate": round(float(wins.size / r.size), 3),
                  "worst_streak": _streak(r)}
    return out


def _streak(r: np.ndarray) -> int:
    best = cur = 0
    for x in r:
        cur = cur + 1 if x < -0.05 else 0
        best = max(best, cur)
    return best


def load(interval: str = "1m") -> pd.DataFrame:
    return read_csv(cache_path("NQ=F", interval))
