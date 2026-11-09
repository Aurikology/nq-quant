"""The model's chart vocabulary, made mechanical: FVGs, their inversion, and the liquidity levels.

Every definition cites the knowledge base, because a backtest of a rule nobody wrote down is a
backtest of the coder's opinion. Where the knowledge base leaves a choice open, the choice is a
parameter and the default is stated.

- FVG, three candles: `docs/knowledge/ifvg.md` § 2.1 The three-candle imbalance.
  Bullish (BISI): low[i] > high[i-2], gap [high[i-2], low[i]].
  Bearish (SIBI): high[i] < low[i-2], gap [high[i], low[i-2]].
- Inversion: § 4.1 Candle body closure versus wick piercing. A bearish FVG becomes a bullish IFVG
  only when a bullish candle closes above the gap high; a wick through it is not an inversion.
- Invalidation of an IFVG: § 7.2, the distal boundary. A body close back through the far edge
  kills it.
- Liquidity levels: `docs/knowledge/dol-and-liquidity.md` § Session highs and lows and
  § Previous day, week and month extremes. The trading day is CME's, 18:00 to 17:00 ET.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IFVG:
    side: str  # "long" for a bullish IFVG (flipped SIBI), "short" for a bearish one (flipped BISI)
    low: float
    high: float
    formed: int  # bar index of the third FVG candle
    inverted: int  # bar index of the body close that inverted it

    @property
    def ce(self) -> float:
        """Consequent encroachment, § 2.2: the midpoint."""
        return (self.low + self.high) / 2


def find_ifvgs(bars: pd.DataFrame, min_gap: float = 1.0, max_age: int = 96) -> list[IFVG]:
    """Every FVG of at least `min_gap` points that a body close inverted within `max_age` bars.

    `max_age` bounds the search, not the model: an FVG that took more than eight hours of 5m bars
    to fail is a different setup from the one in ifvg.md § 3.2.
    """
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    n = len(bars)
    out: list[IFVG] = []

    bull_fvg = np.flatnonzero(l[2:] - h[:-2] >= min_gap) + 2  # BISI, inverts bearish
    bear_fvg = np.flatnonzero(l[:-2] - h[2:] >= min_gap) + 2  # SIBI, inverts bullish

    for i in bear_fvg:
        lo, hi = h[i], l[i - 2]
        end = min(n, i + 1 + max_age)
        hit = np.flatnonzero((c[i + 1:end] > hi) & (c[i + 1:end] > o[i + 1:end]))
        if hit.size:
            out.append(IFVG("long", float(lo), float(hi), int(i), int(i + 1 + hit[0])))
    for i in bull_fvg:
        lo, hi = h[i - 2], l[i]
        end = min(n, i + 1 + max_age)
        hit = np.flatnonzero((c[i + 1:end] < lo) & (c[i + 1:end] < o[i + 1:end]))
        if hit.size:
            out.append(IFVG("short", float(lo), float(hi), int(i), int(i + 1 + hit[0])))

    out.sort(key=lambda g: g.inverted)
    return out


def trading_day(index: pd.DatetimeIndex) -> np.ndarray:
    """CME trading day of each bar: 18:00 ET belongs to the next date."""
    return (index + pd.Timedelta(hours=6)).normalize().tz_localize(None).to_numpy()


@dataclass(frozen=True)
class Levels:
    pdh: float  # previous trading day high
    pdl: float
    onh: float  # current trading day so far (the overnight range, for the NY sessions)
    onl: float
    price: float  # last close before the cutoff


def levels_at(bars: pd.DataFrame, tday: np.ndarray, cut: int) -> Levels | None:
    """External liquidity visible from bar index `cut` (exclusive). None without a full prior day."""
    if cut <= 0:
        return None
    today = tday[cut - 1]
    prior = np.flatnonzero(tday[:cut] < today)
    if prior.size == 0:
        return None
    prev_day = tday[prior[-1]]
    prev = np.flatnonzero(tday[:cut] == prev_day)
    cur = np.flatnonzero(tday[:cut] == today)
    h, l = bars["high"].to_numpy(), bars["low"].to_numpy()
    return Levels(
        pdh=float(h[prev].max()), pdl=float(l[prev].min()),
        onh=float(h[cur].max()), onl=float(l[cur].min()),
        price=float(bars["close"].iat[cut - 1]),
    )
