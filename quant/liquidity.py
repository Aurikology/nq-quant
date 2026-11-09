"""The detector spec, made mechanical: FVG states, liquidity pools, the DOL, and the run grade.

One definition shared with pine/ (both implement the same spec). Every rule cites the knowledge
base; a number the knowledge base leaves open is a parameter, labelled "default, not in the
knowledge base". Instrument NQ/MNQ, tick 0.25 (`config.TICK_SIZE`). Times are America/Toronto
(the New York offsets); the trading day is CME's, 18:00 to 17:00 ET (`signals.trading_day`).

No lookahead, anywhere: every public function takes `bars` plus `cut`, an exclusive bar index, and
never reads a bar at or after `cut`. A swing pivot is confirmed `pivot_len` bars after its bar.

1. FVG and IFVG (`docs/knowledge/ifvg.md`)
   - Three candles, § 2.1 The three-candle imbalance. Bullish (BISI) on candle i: low[i] > high[i-2],
     gap [high[i-2], low[i]]. Bearish (SIBI): high[i] < low[i-2], gap [high[i], low[i-2]].
     `min_gap` = 1.0 point (default, not in the knowledge base; the same as `strategy.Params`).
   - CE is the midpoint, § 2.2 Consequent encroachment (CE).
   - State, evaluated bar by bar from i+1 (bullish shown; bearish is the mirror on highs):
     "open" untouched; "mitigated" a wick traded into the gap (low <= gap high) short of the CE;
     "ce" a wick reached the CE; "filled" a wick traded through the far edge (low <= gap low) with
     no inverting body close, which § 4.1 Candle body closure versus wick piercing calls a deep
     mitigation, not an inversion; "inverted" a bearish body closed below the gap low (close < low
     and close < open), so the BISI is now a bearish IFVG (side "short"); a SIBI inverts to a
     bullish IFVG ("long"). The wick thresholds for "mitigated" and "filled" are the spec's
     (default, not in the knowledge base).
   - Inversion is looked for only within `max_age` = 96 bars of formation, exactly as
     `signals.find_ifvgs` does (default, not in the knowledge base); the other states keep updating.
   - IFVG invalidation, § 7.2 Stop placement and gap invalidation (the distal boundary): a close
     back beyond the IFVG's far edge kills it (a long IFVG dies on a close below its low).

2. Liquidity pools (`docs/knowledge/dol-and-liquidity.md` § The taxonomy and hierarchy of liquidity
   pools). Each pool has a price, a side ("BSL" highs above, "SSL" lows below), a kind, the bar it
   formed on, and `swept`: the first bar after formation whose wick trades beyond it (high > price
   for BSL, low < price for SSL), None while unswept.
   - PDH/PDL, PWH/PWL: § Previous day, week and month extremes. The week starts Sunday 18:00 ET,
     `killzones-and-sessions.md` § CME Globex mechanics for NQ and MNQ.
   - ASIA_H/L 20:00-00:00, LONDON_H/L 02:00-05:00, NYAM_H/L 09:30-11:00 ET: § Session highs and
     lows. NY AM uses the repo's `ny_am` window (data/config.json); the knowledge base's killzone
     opens 08:30 (`killzones-and-sessions.md` § New York AM killzone: 08:30 to 11:00 ET). A session
     becomes a pool only once its window has closed (default, not in the knowledge base).
   - SWING_H/L, § Old highs and lows: a fractal pivot higher (lower) than the `pivot_len` = 2 bars
     on each side, a 5-bar fractal (default, not in the knowledge base), confirmed `pivot_len` bars
     late.
   - EQH/EQL, § Equal highs and lows (absolute and relative): two or more swing pivots whose prices
     are within `eq_tol` = 2.0 points of each other (default, not in the knowledge base). The pool
     price is the cluster's highest high (lowest low); it forms on the cluster's last pivot. A pivot
     joins a cluster only if no bar between them traded beyond both (the level held between the
     touches; default, not in the knowledge base).
   - Significant kinds: everything except SWING_H/L.

3. DOL pick (`dol-and-liquidity.md` § The single-session DOL (intraday NQ);
   `internal-to-external.md` § External range liquidity (ERL)).
   - Candidates: unswept pools on the bias side, BSL strictly above the last close for a long bias,
     SSL strictly below it for a short one.
   - Recency: intraday kinds (session, EQ, SWING) must have formed within the last
     `lookback_sessions` = 3 CME trading days, counting the current one ("typically formed 1 to 3
     sessions earlier"). PDH/PDL/PWH/PWL are always eligible.
   - The nearest significant candidate wins (the knowledge base's 40-point equal high over the
     150-point daily swing high). Within 1 tick, EQ beats session beats PD beats PW: EQ first per
     § Equal highs and lows (absolute and relative); the rest of the order is the spec's (default,
     not in the knowledge base).
   - Only when no significant candidate exists does the nearest SWING pool win (§ Equal highs and
     lows: equal highs override minor isolated swing highs).
   - The next tier is the nearest candidate of the same class farther out by more than a tick, for
     the "recalibrate" step.
   - Bias, when not given: premium/discount against the previous day's equilibrium,
     `internal-to-external.md` § Premium, discount, and the equilibrium threshold, the same rule as
     `strategy.build_plan` (below the equilibrium is long).

4. LRLR / HRLR (`internal-to-external.md` § Low-resistance versus high-resistance liquidity runs).
   - HTF bias proxy (default, not in the knowledge base, which names monthly and weekly): daily bias
     +1 if the last completed trading day closed above the midpoint of the day before's range, -1
     if below, 0 if on it; weekly the same on weeks. Aligned = both equal the run's direction.
   - Obstacles: opposing FVGs still "open" or "mitigated" whose gap lies between the start price and
     the DOL (SIBIs on a long run, BISIs on a short one), on the bars given (default, not in the
     knowledge base).
   - "LRLR" if aligned and obstacles <= `max_obstacles` = 0 (default, not in the knowledge base);
     otherwise "HRLR". Counter to the HTF is HRLR whatever the path looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TICK_SIZE
from .signals import levels_at, trading_day

SIGNIFICANT_PRIORITY = {"EQH": 0, "EQL": 0, "ASIA_H": 1, "ASIA_L": 1, "LONDON_H": 1, "LONDON_L": 1,
                        "NYAM_H": 1, "NYAM_L": 1, "PDH": 2, "PDL": 2, "PWH": 3, "PWL": 3}
ALWAYS_ELIGIBLE = {"PDH", "PDL", "PWH", "PWL"}
# (kind prefix, start minute, end minute) in ET; Asia runs the evening before the trading day's date
SESSIONS = (("ASIA", 20 * 60, 24 * 60), ("LONDON", 2 * 60, 5 * 60), ("NYAM", 9 * 60 + 30, 11 * 60))


def _ohlc(bars: pd.DataFrame, cut: int):
    return tuple(bars[k].to_numpy()[:cut] for k in ("open", "high", "low", "close"))


def _first(mask: np.ndarray, offset: int) -> int | None:
    idx = np.flatnonzero(mask)
    return int(offset + idx[0]) if idx.size else None


# --- 1. FVG and IFVG -------------------------------------------------------------------------

@dataclass(frozen=True)
class FVG:
    side: str  # "bull" (BISI) or "bear" (SIBI)
    low: float
    high: float
    formed: int  # bar index of the third candle
    state: str  # "open" | "mitigated" | "ce" | "filled" | "inverted"
    mitigated: int | None = None  # first bar of each transition, None if not (yet) reached
    ce_hit: int | None = None
    filled: int | None = None
    inverted: int | None = None
    invalidated: int | None = None  # IFVG killed by a close back through its far edge

    @property
    def ce(self) -> float:
        """§ 2.2: the midpoint."""
        return (self.low + self.high) / 2

    @property
    def ifvg_side(self) -> str | None:
        """A flipped BISI is a short IFVG, a flipped SIBI a long one."""
        if self.inverted is None:
            return None
        return "short" if self.side == "bull" else "long"

    @property
    def live_ifvg(self) -> bool:
        return self.inverted is not None and self.invalidated is None


def find_fvgs(bars: pd.DataFrame, cut: int, min_gap: float = 1.0, max_age: int = 96) -> list[FVG]:
    """Every FVG formed before `cut`, with its state as of the close of bar cut-1."""
    o, h, l, c = _ohlc(bars, cut)
    out: list[FVG] = []
    for i in np.flatnonzero(l[2:] - h[:-2] >= min_gap) + 2:  # BISI
        out.append(_track("bull", float(h[i - 2]), float(l[i]), int(i), o, h, l, c, max_age))
    for i in np.flatnonzero(l[:-2] - h[2:] >= min_gap) + 2:  # SIBI
        out.append(_track("bear", float(h[i]), float(l[i - 2]), int(i), o, h, l, c, max_age))
    out.sort(key=lambda g: (g.formed, g.side))
    return out


def _track(side: str, lo: float, hi: float, i: int, o, h, l, c, max_age: int) -> FVG:
    n, s, ce = len(c), i + 1, (lo + hi) / 2
    end = min(n, i + 1 + max_age)
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
    return FVG(side, lo, hi, i, state, mit, ceb, thru, inv, kill)


# --- 2. Liquidity pools ----------------------------------------------------------------------

@dataclass(frozen=True)
class Pool:
    kind: str
    side: str  # "BSL" or "SSL"
    price: float
    formed: int  # bar index
    swept: int | None  # first bar after `formed` that traded beyond it, before cut

    @property
    def significant(self) -> bool:
        return self.kind in SIGNIFICANT_PRIORITY


def _pool(kind: str, side: str, price: float, formed: int, h, l) -> Pool:
    s = formed + 1
    swept = _first(h[s:] > price, s) if side == "BSL" else _first(l[s:] < price, s)
    return Pool(kind, side, float(price), int(formed), swept)


def _bar_secs(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 300
    return int(pd.Series(index).diff().dropna().mode().iloc[0].total_seconds())


def swing_pivots(bars: pd.DataFrame, cut: int, pivot_len: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Bar indices of confirmed swing highs and lows: strictly beyond `pivot_len` bars each side.

    A pivot at j needs bars j+1..j+pivot_len, so it exists only when j + pivot_len < cut.
    """
    _, h, l, _ = _ohlc(bars, cut)
    k, w = pivot_len, 2 * pivot_len + 1
    if len(h) < w:
        return np.array([], dtype=int), np.array([], dtype=int)
    wh = np.lib.stride_tricks.sliding_window_view(h, w)
    wl = np.lib.stride_tricks.sliding_window_view(l, w)
    oh, ol = np.delete(wh, k, axis=1), np.delete(wl, k, axis=1)
    highs = np.flatnonzero(wh[:, k] > oh.max(axis=1)) + k
    lows = np.flatnonzero(wl[:, k] < ol.min(axis=1)) + k
    return highs, lows


def _clusters(piv: np.ndarray, px: np.ndarray, eq_tol: float, high: bool) -> list[list[int]]:
    """Chronological clustering of pivots into equal highs (lows); see the module docstring.

    A cluster dies once a bar after its last pivot trades beyond its level, unless that bar is the
    approach to a new pivot within `eq_tol` that the cluster absorbs (relative equal highs).
    """
    sign = 1.0 if high else -1.0
    v = sign * px  # work on highs; lows are mirrored
    clusters: list[list[int]] = []
    alive: list[list[int]] = []
    for j in piv:
        p, best, still = v[j], None, []
        for cl in alive:
            lvl, low = v[cl].max(), v[cl].min()
            between = v[cl[-1] + 1:j]
            peak = between.max() if between.size else -np.inf
            fits = max(lvl, p) - min(low, p) <= eq_tol and peak <= max(lvl, p)
            if peak > lvl and not fits:
                continue  # its level was traded through: no longer a pool
            still.append(cl)
            if fits:
                best = cl  # the most recent qualifying cluster
        alive = still
        if best is not None:
            best.append(int(j))
        else:
            alive.append([int(j)])
            clusters.append(alive[-1])
    return [cl for cl in clusters if len(cl) >= 2]


def recent_start(bars: pd.DataFrame, cut: int, lookback_sessions: int = 3) -> int:
    """First bar index of the recency window: the last `lookback_sessions` trading days before cut."""
    tday = trading_day(bars.index[:cut])
    days = np.unique(tday)
    return int(np.flatnonzero(tday >= days[-lookback_sessions:][0])[0]) if cut > 0 else 0


def find_pools(bars: pd.DataFrame, cut: int, pivot_len: int = 2, eq_tol: float = 2.0,
               lookback_sessions: int = 3) -> list[Pool]:
    """PD and PW extremes, plus the session, EQ and swing pools formed in the recency window."""
    if cut <= 0:
        return []
    idx = bars.index[:cut]
    _, h, l, _ = _ohlc(bars, cut)
    tday = trading_day(idx)
    days = np.unique(tday)
    recent_from = days[-lookback_sessions:][0]  # first trading day of the recency window
    out: list[Pool] = []

    def extremes(mask: np.ndarray, hk: str, lk: str):
        sel = np.flatnonzero(mask)
        if sel.size:
            out.append(_pool(hk, "BSL", h[sel].max(), int(sel[np.argmax(h[sel])]), h, l))
            out.append(_pool(lk, "SSL", l[sel].min(), int(sel[np.argmin(l[sel])]), h, l))

    if days.size >= 2:
        extremes(tday == days[-2], "PDH", "PDL")
    di = pd.DatetimeIndex(tday)
    week = (di - pd.to_timedelta(di.weekday, unit="D")).to_numpy()  # Monday of the trading week
    weeks = np.unique(week)
    if weeks.size >= 2:
        extremes(week == weeks[-2], "PWH", "PWL")

    # Sessions: only windows that had closed by the close of bar cut-1.
    closed_at = idx[-1] + pd.Timedelta(seconds=_bar_secs(idx))
    minute = idx.hour * 60 + idx.minute
    for d in days[-lookback_sessions:]:
        day0 = pd.Timestamp(d).tz_localize(idx.tz) if idx.tz is not None else pd.Timestamp(d)
        on_day = tday == d
        for name, start, end in SESSIONS:
            if name == "ASIA":  # 20:00 to 00:00 of the evening before the trading day's date
                in_s, ends = on_day & (minute >= start), day0
            else:
                in_s, ends = on_day & (minute >= start) & (minute < end), day0 + pd.Timedelta(minutes=end)
            if ends <= closed_at:
                extremes(in_s, f"{name}_H", f"{name}_L")

    ph, pl = swing_pivots(bars, cut, pivot_len)
    for kind, piv, px, side, high in (("EQH", ph, h, "BSL", True), ("EQL", pl, l, "SSL", False)):
        for cl in _clusters(piv, px, eq_tol, high):
            prices = px[cl]
            price = prices.max() if high else prices.min()
            if tday[cl[-1]] >= recent_from:
                out.append(_pool(kind, side, price, cl[-1], h, l))
    for j in ph:
        if tday[j] >= recent_from:
            out.append(_pool("SWING_H", "BSL", h[j], int(j), h, l))
    for j in pl:
        if tday[j] >= recent_from:
            out.append(_pool("SWING_L", "SSL", l[j], int(j), h, l))
    return out


# --- 3. DOL ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class DOL:
    bias: str | None  # "long" | "short"; None when it could not be set
    bias_source: str  # "given" or the premium/discount reason
    price: float  # last close before cut
    pool: Pool | None
    distance: float | None  # points from the last close
    next_tier: Pool | None
    reason: str  # "ok" or why there is no DOL


def default_bias(bars: pd.DataFrame, cut: int) -> tuple[str | None, str]:
    """Premium/discount against the previous day's equilibrium, as `strategy.build_plan`."""
    lv = levels_at(bars, trading_day(bars.index[:cut]), cut)
    if lv is None:
        return None, "no prior trading day before the cut"
    eq = (lv.pdh + lv.pdl) / 2
    side = "long" if lv.price < eq else "short"
    where = "discount" if side == "long" else "premium"
    return side, f"price {lv.price:.2f} in {where} of the previous day (EQ {eq:.2f})"


def pick_dol(bars: pd.DataFrame, cut: int, bias: str | None = None, pools: list[Pool] | None = None,
             **pool_kw) -> DOL:
    """The draw on liquidity from the close of bar cut-1, per the module docstring § 3."""
    if bias is None:
        bias, source = default_bias(bars, cut)
        if bias is None:
            return DOL(None, source, float("nan"), None, None, None, source)
    else:
        source = "given"
    price = float(bars["close"].to_numpy()[cut - 1])
    pools = find_pools(bars, cut, **pool_kw) if pools is None else pools
    if bias == "long":
        cands = [p for p in pools if p.side == "BSL" and p.swept is None and p.price > price]
    else:
        cands = [p for p in pools if p.side == "SSL" and p.swept is None and p.price < price]
    sig = sorted((p for p in cands if p.significant), key=lambda p: abs(p.price - price))
    swing = sorted((p for p in cands if not p.significant), key=lambda p: abs(p.price - price))
    group = sig or swing
    if not group:
        side = "BSL above" if bias == "long" else "SSL below"
        return DOL(bias, source, price, None, None, None, f"no unswept {side} the last close {price:.2f}")
    near = abs(group[0].price - price)
    tied = [p for p in group if abs(p.price - price) <= near + TICK_SIZE + 1e-9]
    pick = min(tied, key=lambda p: (SIGNIFICANT_PRIORITY.get(p.kind, 9), abs(p.price - price)))
    dist = abs(pick.price - price)
    nxt = next((p for p in group if abs(p.price - price) > dist + TICK_SIZE + 1e-9), None)
    reason = "ok" if sig else "ok (no significant pool; nearest swing)"
    return DOL(bias, source, price, pick, dist, nxt, reason)


# --- 4. LRLR / HRLR --------------------------------------------------------------------------

@dataclass(frozen=True)
class RunGrade:
    grade: str  # "LRLR" | "HRLR"
    direction: str
    start: float
    target: float
    daily_bias: int  # +1, -1, 0 (0 also when there is not enough history)
    weekly_bias: int
    aligned: bool
    obstacles: list[FVG] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _htf_bias(keys: np.ndarray, h, l, c) -> tuple[int, str]:
    """Last completed period's close against the period before's midpoint. The last key is live."""
    uniq = np.unique(keys)
    if uniq.size < 3:
        return 0, "not enough completed periods"
    prev, last = (np.flatnonzero(keys == k) for k in uniq[-3:-1])
    mid = (h[prev].max() + l[prev].min()) / 2
    close = c[last[-1]]
    return int(np.sign(close - mid)), f"close {close:.2f} vs prior midpoint {mid:.2f}"


def htf_bias(bars: pd.DataFrame, cut: int) -> tuple[int, int, list[str]]:
    _, h, l, c = _ohlc(bars, cut)
    tday = trading_day(bars.index[:cut])
    di = pd.DatetimeIndex(tday)
    week = (di - pd.to_timedelta(di.weekday, unit="D")).to_numpy()
    d, dwhy = _htf_bias(tday, h, l, c)
    w, wwhy = _htf_bias(week, h, l, c)
    return d, w, [f"daily {d:+d} ({dwhy})", f"weekly {w:+d} ({wwhy})"]


def grade_run(bars: pd.DataFrame, cut: int, target: float, direction: str, start: float | None = None,
              fvgs: list[FVG] | None = None, max_obstacles: int = 0, min_gap: float = 1.0,
              max_age: int = 96) -> RunGrade:
    """LRLR or HRLR for a run from `start` (default the last close) to `target`."""
    start = float(bars["close"].to_numpy()[cut - 1]) if start is None else float(start)
    fvgs = find_fvgs(bars, cut, min_gap, max_age) if fvgs is None else fvgs
    want = 1 if direction == "long" else -1
    d, w, reasons = htf_bias(bars, cut)
    aligned = d == want and w == want
    lo, hi = min(start, target), max(start, target)
    opposing = "bear" if direction == "long" else "bull"
    obstacles = [g for g in fvgs if g.side == opposing and g.state in ("open", "mitigated")
                 and g.high > lo and g.low < hi]
    reasons = [("aligned" if aligned else "not aligned") + f" with a {direction} run"] + reasons
    if not aligned:
        reasons.append("counter to (or unconfirmed by) the HTF: HRLR whatever the path")
    reasons.append(f"{len(obstacles)} opposing unmitigated FVG(s) in the path (max {max_obstacles})"
                   + "".join(f"; {'SIBI' if g.side == 'bear' else 'BISI'} {g.low:.2f}-{g.high:.2f} {g.state}"
                             for g in obstacles))
    grade = "LRLR" if aligned and len(obstacles) <= max_obstacles else "HRLR"
    return RunGrade(grade, direction, start, float(target), d, w, aligned, obstacles, reasons)
