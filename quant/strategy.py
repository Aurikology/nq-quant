"""Aurik's three sentences as a function that files a plan.

> Find your DOL, draw on liquidity.
> Trade from internal to external.
> The box is where you wait for price to trace into before you enter.

`build_plan` runs at the plan deadline (session start minus `planDeadlineMin`) and sees only bars
that had closed by then. It returns a plan in the exact `core/types.js` Plan schema, which the
backtester sends through `core/plan.js validatePlan` before it is allowed to trade: the automated
plan has to pass the same gate as his hand-filed one.

1. Bias, premium/discount (`docs/knowledge/internal-to-external.md` § Premium, discount, and the
   equilibrium threshold). Below the previous day's equilibrium is discount: look for longs. Above
   is premium: look for shorts. `bias="momentum"` inverts this, as the control the sweep tests.
2. Box: the nearest live IFVG on the bias side, inside the previous and current trading days, that
   price has to retrace into (`ifvg.md` § 3.2 stage 3: retracements into the gap are the entries).
   With `first_touch=True` a box that price has already traded back into since it inverted is spent.
3. Target and DOL: the nearest external level beyond the box, overnight extreme or previous-day
   extreme, that is still unswept and offers at least `min_rr` from the entry. None qualifies: no
   plan, and a session with no plan and no trade is a good session (CLAUDE.md § SESSIONS).
4. Entry `ce` (the midpoint, § 7.1) or `proximal` (the near edge); stop `buffer_ticks` beyond the
   distal edge (§ 7.2, distal boundary stop). The stop is the plan's invalidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TICK_SIZE, Window
from .signals import IFVG, Levels, levels_at


@dataclass(frozen=True)
class Params:
    entry: str = "ce"  # "ce" | "proximal"
    min_rr: float = 2.0
    buffer_ticks: int = 2
    min_gap: float = 1.0  # points; FVGs smaller than this are noise on 5m NQ
    first_touch: bool = True
    bias: str = "discount"  # "discount" | "momentum"
    instrument: str = "NQ"
    risk_dollars: float | None = None  # None: the firm's per-trade cap

    def label(self) -> str:
        return (f"{self.bias}/{self.entry}/rr{self.min_rr:g}/gap{self.min_gap:g}/buf{self.buffer_ticks}/"
                f"{'first' if self.first_touch else 'any'}")


@dataclass
class Setup:
    plan: dict | None
    reason: str  # "ok" or why there is no plan
    entry: float = float("nan")
    stop: float = float("nan")
    target: float = float("nan")
    box: IFVG | None = None
    levels: Levels | None = None
    extras: dict = field(default_factory=dict)


def build_plan(bars: pd.DataFrame, tday: np.ndarray, bar_secs: int, ifvgs: list[IFVG],
               window: Window, rules: dict, params: Params, timeframe: str) -> Setup:
    # Only bars that had closed by the deadline: open time + bar length <= deadline.
    last_open = pd.Timestamp(window.plan_deadline) - pd.Timedelta(seconds=bar_secs)
    cut = int(np.searchsorted(bars.index.as_unit("ns").asi8, last_open.value, side="right"))
    lv = levels_at(bars, tday, cut)
    if lv is None:
        return Setup(None, "no prior trading day in the data")

    eq = (lv.pdh + lv.pdl) / 2
    discount = lv.price < eq
    long_bias = discount if params.bias == "discount" else not discount
    side = "long" if long_bias else "short"

    lows, highs, closes = (bars[k].to_numpy() for k in ("low", "high", "close"))
    first_bar = int(np.flatnonzero(tday[:cut] >= _prev_day(tday, cut))[0])

    best: IFVG | None = None
    for g in ifvgs:
        if g.inverted >= cut:
            break
        if g.side != side or g.formed - 2 < first_bar:
            continue
        after = slice(g.inverted + 1, cut)
        if side == "long":
            if g.high >= lv.price or (closes[after] < g.low).any():
                continue  # price is inside or below it, or a body closed back through it
            if params.first_touch and (lows[after] <= g.high).any():
                continue
            if best is None or g.high > best.high:
                best = g
        else:
            if g.low <= lv.price or (closes[after] > g.high).any():
                continue
            if params.first_touch and (highs[after] >= g.low).any():
                continue
            if best is None or g.low < best.low:
                best = g
    if best is None:
        return Setup(None, f"no live {side} IFVG to wait in", levels=lv)

    buf = params.buffer_ticks * TICK_SIZE
    if side == "long":
        entry = best.ce if params.entry == "ce" else best.high
        stop = best.low - buf
        # The overnight high is unswept by definition; the PDH only if the overnight did not take it.
        cands = [("session_high", lv.onh)] + ([("pdh", lv.pdh)] if lv.pdh > lv.onh else [])
        cands = sorted(((k, v) for k, v in cands if v > best.high), key=lambda kv: kv[1])
    else:
        entry = best.ce if params.entry == "ce" else best.low
        stop = best.high + buf
        cands = [("session_low", lv.onl)] + ([("pdl", lv.pdl)] if lv.pdl < lv.onl else [])
        cands = sorted(((k, v) for k, v in cands if v < best.low), key=lambda kv: -kv[1])

    risk_pts = abs(entry - stop)
    target = next(((k, v) for k, v in cands if abs(v - entry) / risk_pts >= params.min_rr), None)
    if target is None:
        return Setup(None, f"no external target {params.min_rr:g}R away", box=best, levels=lv)

    kind, level = target
    pct_cap = rules["accountSize"] * rules["maxRiskPerTradePct"] / 100
    max_dollars = min(params.risk_dollars or pct_cap, pct_cap)
    liquidity = {"type": "external", "kind": kind, "level": level, "label": _label(kind)}
    plan = {
        "schema": 1,
        "date": window.date.isoformat(),
        "session": window.session,
        "filedAt": window.plan_deadline.isoformat(),
        "bias": side,
        "narrative": (
            f"Automated: price {lv.price:.2f} is in {'discount' if discount else 'premium'} of the "
            f"previous day ({lv.pdl:.2f} to {lv.pdh:.2f}, EQ {eq:.2f}). Waiting in the {timeframe} "
            f"IFVG {best.low:.2f} to {best.high:.2f} for a draw to the {_label(kind)} at {level:.2f}."
        ),
        "dol": liquidity,
        "dealingRange": {"high": max(lv.pdh, lv.onh), "low": min(lv.pdl, lv.onl), "timeframe": "1d"},
        "box": {"high": best.high, "low": best.low, "timeframe": timeframe, "kind": "ifvg"},
        "invalidation": stop,
        "target": liquidity,
        "risk": {"maxDollars": max_dollars, "maxContracts": rules["maxContracts"][params.instrument],
                 "instrument": params.instrument},
        "entryOnlyInsideBox": True,
        "maxTrades": 1,
        # The scanner cannot read the economic calendar, so it never claims it did.
        "checklist": {"dolIdentified": True, "internalToExternal": True, "boxMarked": True, "newsChecked": False},
    }
    return Setup(plan, "ok", entry=entry, stop=stop, target=level, box=best, levels=lv,
                 extras={"cut": cut, "rr": abs(level - entry) / risk_pts})


def _prev_day(tday: np.ndarray, cut: int):
    today = tday[cut - 1]
    return tday[np.flatnonzero(tday[:cut] < today)[-1]]


def _label(kind: str) -> str:
    return {"session_high": "overnight high", "session_low": "overnight low",
            "pdh": "previous day high", "pdl": "previous day low"}[kind]
