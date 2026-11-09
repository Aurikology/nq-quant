"""Bar-by-bar replay of the plan → box → entry loop under the firm's rules.

Fill model, conservative on purpose, because an optimistic fill is how a backtest lies:
- The entry is a resting limit at the plan's entry price, live only inside the session window
  (OUTSIDE_SESSION). With `fill="through"` price must trade one tick past it, not just touch it.
- A bar that opens beyond the stop before the fill cancels the order: the box was invalidated.
- If a bar touches both the stop and the target, the stop wins. The bar that fills the entry can
  stop it out but cannot take profit, because the bar's order of high and low is unknown.
- Stops and time exits are market orders and pay `slip_ticks`. Targets are limits and do not.
- Flat at the session end, at the last bar's close.
- Commission is per contract, round turn, taken from every trade's P&L.

Firm rules applied before each entry, with the same numbers `core/risk.js` uses (`risk.py` is
checked against it): daily loss limit, per-trade cap, contract caps, the revenge window, max trades
per session, and the trailing drawdown. A drawdown breach is recorded with its date and, unless
`stop_at_breach`, the replay continues, so the strategy's edge is measured on the whole sample
rather than on the part before the account died.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import REPO, TICK_SIZE, enabled_sessions, session_window
from .market import infer_interval
from .risk import TrailingDrawdown, max_contracts, remaining_daily_loss, trade_metrics
from .signals import find_ifvgs, trading_day
from .strategy import Params, Setup, build_plan

COMMISSION = {"NQ": 4.50, "MNQ": 1.50}  # round turn per contract; an assumption, set your own


@dataclass
class Costs:
    commission: float | None = None  # None: COMMISSION[instrument]
    slip_ticks: int = 1
    fill: str = "through"  # "through" | "touch"


@dataclass
class Result:
    params: Params
    trades: list[dict] = field(default_factory=list)
    sessions: list[dict] = field(default_factory=list)  # one row per session window considered
    breach: dict | None = None
    dates: list[str] = field(default_factory=list)


def validate_plans(plans: list[dict], rules: dict) -> list[dict]:
    """Every generated plan through core/plan.js validatePlan, in one Node process."""
    if not plans:
        return []
    out = subprocess.run(
        ["node", str(REPO / "quant" / "bridge.mjs")],
        input=json.dumps({"op": "validatePlans", "plans": plans, "rules": rules}),
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return json.loads(out.stdout)


class Market:
    """Bars plus everything derived from them once, shared by every run of a sweep."""

    def __init__(self, bars: pd.DataFrame, min_gap: float = 1.0):
        self.bars = bars
        self.timeframe = infer_interval(bars)
        self.bar_secs = int(pd.Series(bars.index).diff().dropna().mode().iloc[0].total_seconds())
        self.tday = trading_day(bars.index)
        self.o, self.h, self.l, self.c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
        self.t = bars.index
        self._ifvgs: dict[float, list] = {}

    def ifvgs(self, min_gap: float):
        if min_gap not in self._ifvgs:
            self._ifvgs[min_gap] = find_ifvgs(self.bars, min_gap=min_gap)
        return self._ifvgs[min_gap]

    def span(self, start, end) -> tuple[int, int]:
        idx = self.t.asi8  # UTC nanoseconds; comparing ints sidesteps tz-aware object arrays
        return (int(np.searchsorted(idx, pd.Timestamp(start).value, "left")),
                int(np.searchsorted(idx, pd.Timestamp(end).value, "left")))


def session_dates(m: Market) -> list[date]:
    first, last = m.t[0].date(), m.t[-1].date()
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def run(m: Market, rules: dict, config: dict, params: Params, costs: Costs = Costs(),
        dates: list[date] | None = None, validate: bool = True, exec_m: Market | None = None,
        stop_at_breach: bool = False) -> Result:
    """`m` finds the boxes; `exec_m` (finer bars, e.g. 1m under 5m) fills and exits them.

    On a single timeframe the fill bar's own stop-out is unknowable and is scored as a loss, which
    on 5m NQ is most of them. Executing on 1m bars shrinks that unknown to one minute. Only sessions
    that `exec_m` covers are traded.
    """
    res = Result(params)
    em = exec_m or m
    commission = COMMISSION[params.instrument] if costs.commission is None else costs.commission
    ifvgs = m.ifvgs(params.min_gap)
    sessions = enabled_sessions(config, rules)
    expected = {s: _expected_bars(config, s, em.bar_secs) for s in sessions}

    # Pass 1: file every plan. Plans depend only on bars before their deadline, never on trades.
    windows, setups = [], []
    for d in dates or session_dates(m):
        for s in sessions:
            w = session_window(config, s, d)
            a, b = em.span(w.start, w.end)
            if b - a < 0.8 * expected[s]:
                continue  # weekend, holiday, or a hole in the data: not a session
            windows.append((w, a, b))
            setups.append(build_plan(m.bars, m.tday, m.bar_secs, ifvgs, w, rules, params, m.timeframe))
    windows_setups = sorted(zip(windows, setups), key=lambda ws: ws[0][0].start)

    planned = [st.plan for _, st in windows_setups if st.plan]
    verdicts = iter(validate_plans(planned, rules) if validate else [{"ok": True, "errors": []}] * len(planned))
    for _, st in windows_setups:
        if st.plan:
            v = next(verdicts)
            if not v["ok"]:
                st.reason = "rejected by core/plan.js: " + "; ".join(v["errors"])
                st.plan = None

    # Pass 2: trade them in time order, with the account state the gate would see.
    dd = TrailingDrawdown(rules)
    day_pnl: dict[str, float] = {}
    last_loss_exit = None
    current_day = None
    for (w, a, b), st in windows_setups:
        d = w.date.isoformat()
        if current_day is not None and d != current_day:
            dd.close_day()
        current_day = d
        if d not in res.dates:
            res.dates.append(d)
        row = {"date": d, "session": w.session, "reason": st.reason, "traded": False}
        res.sessions.append(row)
        if st.plan is None:
            continue
        if stop_at_breach and dd.remaining <= 0:
            row["reason"] = "TRAILING_DD_REACHED"
            continue
        remaining = remaining_daily_loss(day_pnl.get(d, 0.0), rules)
        if remaining <= 0:
            row["reason"] = "DAILY_LOSS_REACHED"
            continue
        contracts = max_contracts(st.plan, rules, remaining)
        if contracts <= 0:
            row["reason"] = "stop too wide for the risk budget"
            continue

        cut, _ = em.span(st.plan["filedAt"], st.plan["filedAt"])
        if _invalidated(em, st, cut, a):
            row["reason"] = "box invalidated before the session opened"
            continue
        trade = _simulate(em, st, a, b, contracts, costs, commission, params.instrument)
        if trade is None:
            row["reason"] = "box never traded"
            continue
        if last_loss_exit is not None and (trade["_entry_ts"] - last_loss_exit) <= pd.Timedelta(minutes=rules["revengeWindowMin"]):
            row["reason"] = "REVENGE_TRADE window"
            continue

        trade.update({"date": d, "session": w.session, "id": f"bt_{d.replace('-', '')}_{w.session}",
                      "box": st.plan["box"], "target": st.target, "plannedRisk": st.plan["risk"]["maxDollars"]})
        row.update(traded=True, reason="traded")
        day_pnl[d] = day_pnl.get(d, 0.0) + trade["net"]
        dd.add(trade["net"])
        if trade["net"] < 0:
            last_loss_exit = trade["_exit_ts"]
        if dd.remaining <= 0 and res.breach is None:
            res.breach = {"date": d, "equity": round(dd.equity, 2), "floor": dd.floor}
        trade.pop("_entry_ts"), trade.pop("_exit_ts")
        res.trades.append(trade)
    return res


def _expected_bars(config: dict, session: str, bar_secs: int) -> int:
    w = session_window(config, session, date(2026, 1, 6))  # any weekday away from a DST switch
    return int((w.end - w.start).total_seconds() // bar_secs)


def _invalidated(m: Market, st: Setup, cut: int, a: int) -> bool:
    """Between the deadline and the open, a body close through the far edge kills the box."""
    seg = m.c[cut:a]
    return bool((seg < st.box.low).any() if st.plan["bias"] == "long" else (seg > st.box.high).any())


def _simulate(m: Market, st: Setup, a: int, b: int, contracts: int, costs: Costs,
              commission: float, instrument: str) -> dict | None:
    long = st.plan["bias"] == "long"
    e, stop, tgt = st.entry, st.stop, st.target
    need = TICK_SIZE if costs.fill == "through" else 0.0
    slip = costs.slip_ticks * TICK_SIZE
    o, h, l, c = m.o, m.h, m.l, m.c

    k = a
    fill = None
    while k < b:
        if (o[k] <= stop) if long else (o[k] >= stop):
            return None  # opened through the stop before any fill
        if (l[k] <= e - need) if long else (h[k] >= e + need):
            fill = min(o[k], e) if long else max(o[k], e)
            break
        k += 1
    if fill is None:
        return None

    entry_k = k
    exit_px, reason = None, None
    while k < b:
        hit_stop = (l[k] <= stop) if long else (h[k] >= stop)
        hit_tgt = k > entry_k and ((h[k] >= tgt) if long else (l[k] <= tgt))
        if hit_stop:
            gap = (o[k] < stop) if long else (o[k] > stop)
            base = o[k] if gap and k > entry_k else stop
            exit_px, reason = (base - slip) if long else (base + slip), "stop"
            break
        if hit_tgt:
            exit_px, reason = (max(o[k], tgt) if long else min(o[k], tgt)), "target"
            break
        k += 1
    if exit_px is None:
        k = b - 1
        exit_px, reason = (c[k] - slip) if long else (c[k] + slip), "session_end"

    side = "long" if long else "short"
    gross = trade_metrics(side, contracts, fill, stop, exit_px, instrument)
    fees = commission * contracts
    net = gross["pnl"] - fees
    return {
        "instrument": instrument, "side": side, "contracts": contracts,
        "entryPrice": fill, "entryTime": m.t[entry_k].isoformat(),
        "stopPrice": stop, "exitPrice": exit_px, "exitTime": m.t[k].isoformat(), "exitReason": reason,
        "gross": gross["pnl"], "fees": round(fees, 2), "net": round(net, 2),
        "riskDollars": gross["riskDollars"], "r": gross["r"],
        "rNet": round(net / gross["riskDollars"], 3) if gross["riskDollars"] > 0 else 0.0,
        "_entry_ts": m.t[entry_k], "_exit_ts": m.t[k],
    }


def random_control(m: Market, res: Result, rules: dict, config: dict, n: int = 500,
                   seed: int = 7, costs: Costs = Costs()) -> dict:
    """Is the box worth waiting for? A permutation test.

    For every real trade, enter instead at the open of a random bar of the same session, same side,
    same stop and target distances in points, same size and costs. Repeat `n` times. The p-value is
    the share of random runs whose mean net R is at least the strategy's.
    """
    if not res.trades:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    slip = costs.slip_ticks * TICK_SIZE
    commission = COMMISSION[res.params.instrument] if costs.commission is None else costs.commission
    pv = 20.0 if res.params.instrument == "NQ" else 2.0
    specs = []
    for t in res.trades:
        w = session_window(config, t["session"], date.fromisoformat(t["date"]))
        a, b = m.span(w.start, w.end)
        long = t["side"] == "long"
        specs.append((a, b, long, abs(t["entryPrice"] - t["stopPrice"]), abs(t["target"] - t["entryPrice"]), t["contracts"]))

    means = np.empty(n)
    for i in range(n):
        rs = []
        for a, b, long, risk_pts, tgt_pts, contracts in specs:
            k = int(rng.integers(a, b - 1))
            e = m.o[k] + (slip if long else -slip)
            stop = e - risk_pts if long else e + risk_pts
            tgt = e + tgt_pts if long else e - tgt_pts
            exit_px = None
            for j in range(k, b):
                if (m.l[j] <= stop) if long else (m.h[j] >= stop):
                    exit_px = (stop - slip) if long else (stop + slip)
                    break
                if j > k and ((m.h[j] >= tgt) if long else (m.l[j] <= tgt)):
                    exit_px = tgt
                    break
            if exit_px is None:
                exit_px = (m.c[b - 1] - slip) if long else (m.c[b - 1] + slip)
            pts = (exit_px - e) if long else (e - exit_px)
            risk = risk_pts * pv * contracts
            rs.append((pts * pv * contracts - commission * contracts) / risk)
        means[i] = np.mean(rs)
    real = float(np.mean([t["rNet"] for t in res.trades]))
    return {
        "n": n, "strategyMeanR": round(real, 3),
        "randomMeanR": round(float(means.mean()), 3),
        "random95": [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)],
        "pValue": round(float((means >= real).mean()), 4),
    }
