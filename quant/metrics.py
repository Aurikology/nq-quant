"""The scorecard of a backtest. Numbers first; every one is net of commission and slippage."""

from __future__ import annotations

import numpy as np

from .backtest import Result


def summarize(res: Result, rules: dict, boot: int = 2000, seed: int = 11) -> dict:
    t = res.trades
    sessions = res.sessions
    funnel: dict[str, int] = {}
    for s in sessions:
        key = s["reason"].split(":")[0]
        funnel[key] = funnel.get(key, 0) + 1

    base = {
        "params": res.params.label(),
        "days": len(res.dates),
        "sessions": len(sessions),
        "funnel": dict(sorted(funnel.items(), key=lambda kv: -kv[1])),
        "trades": len(t),
    }
    if not t:
        return base | {"note": "no trades"}

    net = np.array([x["net"] for x in t])
    r = np.array([x["rNet"] for x in t])
    wins, losses = net[net > 0], net[net < 0]
    equity = rules["accountSize"] + np.cumsum(net)
    peak = np.maximum.accumulate(np.concatenate([[rules["accountSize"]], equity]))[1:]
    max_dd = float((peak - equity).max())

    daily: dict[str, float] = {d: 0.0 for d in res.dates}
    for x in t:
        daily[x["date"]] += x["net"]
    dvals = np.array(list(daily.values()))
    sharpe = float(dvals.mean() / dvals.std(ddof=1) * np.sqrt(252)) if dvals.std(ddof=1) > 0 else 0.0

    streak = worst = 0
    for v in net:
        streak = streak + 1 if v < 0 else 0
        worst = max(worst, streak)

    total = float(net.sum())
    best_day = max(daily.values())
    rng = np.random.default_rng(seed)
    boots = rng.choice(r, size=(boot, len(r)), replace=True).mean(axis=1)

    exits: dict[str, int] = {}
    for x in t:
        exits[x["exitReason"]] = exits.get(x["exitReason"], 0) + 1

    return base | {
        "winRate": round(len(wins) / len(t), 3),
        "avgR": round(float(r.mean()), 3),
        "avgR95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
        "expectancy": round(total / len(t), 2),
        "netPnl": round(total, 2),
        "fees": round(sum(x["fees"] for x in t), 2),
        "profitFactor": round(float(wins.sum() / -losses.sum()), 2) if losses.size else None,
        "maxDrawdown": round(max_dd, 2),
        "sharpeDaily": round(sharpe, 2),
        "worstLosingStreak": worst,
        "exits": exits,
        # Payout consistency: the best day as a share of total profit, against rules.consistencyRulePct.
        "consistencyPct": round(100 * best_day / total, 1) if total > 0 else None,
        "consistencyLimit": rules.get("consistencyRulePct"),
        "trailingDdBreach": res.breach,
        "realizedRiskOverBudget": round(float(np.mean([x["riskDollars"] / x["plannedRisk"] for x in t])), 3),
    }
