"""A line-for-line port of the money functions in `core/risk.js`.

The backtester needs these numbers once per simulated bar, so it cannot shell out to Node for each
one. The cost of a second implementation is the risk that it drifts, so `tests/test_parity.py`
runs both on the same cases through `bridge.mjs` and fails on any difference. Change
`core/risk.js` first, then this file, then watch the parity test go green.
"""

from __future__ import annotations

import math

from .config import POINT_VALUE


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def point_value(instrument: str) -> float:
    if instrument not in POINT_VALUE:
        raise TypeError(f'point_value: "{instrument}" is not NQ or MNQ')
    return POINT_VALUE[instrument]


def stop_points(plan: dict) -> float:
    """core/plan.js stopPoints: box midpoint to invalidation, the stop if he enters mid-box."""
    mid = (plan["box"]["high"] + plan["box"]["low"]) / 2
    return abs(mid - plan["invalidation"])


def contracts_for_stop(risk_dollars: float, stop_pts: float, instrument: str, rules: dict) -> int:
    if not _num(risk_dollars) or risk_dollars <= 0:
        return 0
    if not _num(stop_pts) or stop_pts <= 0:
        return 0
    per_contract = stop_pts * point_value(instrument)
    if per_contract <= 0:
        return 0
    raw = math.floor(risk_dollars / per_contract)
    cap = (rules.get("maxContracts") or {}).get(instrument)
    capped = min(raw, cap) if _num(cap) else raw
    return max(0, capped)


def max_contracts(plan: dict, rules: dict, remaining_daily_loss: float | None = None) -> int:
    left = remaining_daily_loss if _num(remaining_daily_loss) else rules["dailyLossLimit"]
    pct_cap = rules["accountSize"] * rules["maxRiskPerTradePct"] / 100
    budget = min(plan["risk"]["maxDollars"], pct_cap, left)
    n = contracts_for_stop(budget, stop_points(plan), plan["risk"]["instrument"], rules)
    plan_cap = plan["risk"]["maxContracts"] if _num(plan["risk"].get("maxContracts")) else math.inf
    return int(max(0, min(n, plan_cap)))


def remaining_daily_loss(day_pnl: float, rules: dict) -> float:
    """A green day does not buy extra room: only a negative day eats into the limit."""
    return max(0.0, rules["dailyLossLimit"] + min(0.0, day_pnl))


class TrailingDrawdown:
    """core/risk.js trailingDrawdown, made incremental so the simulator can feed it trade by trade.

    trailing_eod moves the peak only when an ET date closes; trailing_intraday after every trade;
    static never. The floor rises with the peak and stops at the starting balance.
    """

    def __init__(self, rules: dict):
        self.rules = rules
        self.equity = float(rules["accountSize"])
        self.peak = float(rules["accountSize"])

    def add(self, pnl: float) -> None:
        self.equity += pnl
        if self.rules["drawdownType"] == "trailing_intraday" and self.equity > self.peak:
            self.peak = self.equity

    def close_day(self) -> None:
        if self.rules["drawdownType"] == "trailing_eod" and self.equity > self.peak:
            self.peak = self.equity

    @property
    def floor(self) -> float:
        acct, dd = self.rules["accountSize"], self.rules["maxDrawdown"]
        return max(min(self.peak - dd, acct), acct - dd)

    @property
    def remaining(self) -> float:
        return self.equity - self.floor

    def snapshot(self) -> dict:
        return {"equity": self.equity, "peak": self.peak, "floor": self.floor, "remaining": self.remaining}


def trailing_drawdown(trades: list[dict], rules: dict) -> dict:
    """Batch form, same signature and output as core/risk.js, for the parity test."""
    closed = sorted((t for t in trades if t.get("outcome") != "open"), key=_closed_at)
    dd = TrailingDrawdown(rules)
    for i, t in enumerate(closed):
        dd.add(t["pnl"] if _num(t.get("pnl")) else 0.0)
        last_of_date = i == len(closed) - 1 or closed[i + 1]["date"] != t["date"]
        if last_of_date:
            dd.close_day()
    return dd.snapshot()


def _closed_at(t: dict) -> str:
    return t.get("exitTime") or t.get("entryTime") or ""


def trade_metrics(side: str, contracts: int, entry_price: float, stop_price: float,
                  exit_price: float | None, instrument: str) -> dict:
    pv = point_value(instrument)
    size = contracts if _num(contracts) and contracts > 0 else 0
    direction = -1 if side == "short" else 1
    stop_pts = abs(entry_price - stop_price) if _num(entry_price) and _num(stop_price) else 0.0
    risk_dollars = stop_pts * pv * size
    is_open = not _num(exit_price)
    pnl = 0.0 if is_open or not _num(entry_price) else (exit_price - entry_price) * direction * pv * size
    r = pnl / risk_dollars if risk_dollars > 0 else 0.0
    outcome = "open" if is_open else "win" if pnl > 0 else "loss" if pnl < 0 else "breakeven"
    return {
        "pnl": _round2(pnl),
        "riskDollars": _round2(risk_dollars),
        "r": _round2(r),
        "outcome": outcome,
        "stopPoints": stop_pts,
        "pointValue": pv,
    }


def _round2(x: float) -> float:
    # JS Math.round(x * 100) / 100 rounds half up toward +infinity; Python's round() is banker's.
    return math.floor(x * 100 + 0.5) / 100


def inside_box(price: float, box: dict) -> bool:
    if not _num(price) or not box or not _num(box.get("low")) or not _num(box.get("high")):
        return False
    return box["low"] <= price <= box["high"]
