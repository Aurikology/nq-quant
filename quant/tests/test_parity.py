"""quant/risk.py must give the same number as core/risk.js on every case, or the backtest is
testing a different account from the one the app gates. Cases are generated, then both sides run.
"""

import json
import random
import subprocess

import pytest

from quant import risk
from quant.config import REPO, load_rules


def js(calls):
    out = subprocess.run(["node", str(REPO / "quant" / "bridge.mjs")],
                         input=json.dumps({"op": "call", "calls": calls}),
                         capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


RULES = load_rules()
rng = random.Random(42)


def _plan():
    low = round(rng.uniform(20000, 30000) / 0.25) * 0.25
    high = low + rng.choice([0.25, 1, 2.5, 4, 7.75, 15, 30])
    side = rng.choice(["long", "short"])
    inv = low - rng.choice([0.5, 1, 3, 10]) if side == "long" else high + rng.choice([0.5, 1, 3, 10])
    return {"box": {"low": low, "high": high}, "invalidation": inv,
            "risk": {"maxDollars": rng.choice([50, 250, 500, 900]), "maxContracts": rng.choice([1, 3, 5, 50]),
                     "instrument": rng.choice(["NQ", "MNQ"])}}


def test_max_contracts_matches_core():
    cases = [(_plan(), rng.choice([None, 0, 120.0, 499.99, 1000])) for _ in range(300)]
    calls = [{"fn": "maxContracts", "args": {"plan": p, "rules": RULES,
                                              **({} if left is None else {"remainingDailyLoss": left})}}
             for p, left in cases]
    assert [risk.max_contracts(p, RULES, left) for p, left in cases] == js(calls)


def test_trade_metrics_matches_core():
    cases = []
    for _ in range(300):
        e = round(rng.uniform(20000, 30000) / 0.25) * 0.25
        cases.append({"side": rng.choice(["long", "short"]), "contracts": rng.choice([0, 1, 2, 5, 13]),
                      "entryPrice": e, "stopPrice": e + rng.choice([-12.25, -3, 2.5, 8]),
                      "exitPrice": rng.choice([None, e + rng.uniform(-40, 40)]),
                      "instrument": rng.choice(["NQ", "MNQ"])})
    got = [risk.trade_metrics(c["side"], c["contracts"], c["entryPrice"], c["stopPrice"], c["exitPrice"], c["instrument"])
           for c in cases]
    want = js([{"fn": "tradeMetrics", "args": c} for c in cases])
    for g, w in zip(got, want):
        assert g["outcome"] == w["outcome"]
        for k in ("pnl", "riskDollars", "r", "stopPoints", "pointValue"):
            assert g[k] == pytest.approx(w[k], abs=1e-9), k


@pytest.mark.parametrize("dd_type", ["trailing_eod", "trailing_intraday", "static"])
def test_trailing_drawdown_matches_core(dd_type):
    rules = RULES | {"drawdownType": dd_type}
    for _ in range(40):
        trades, day = [], 1
        for i in range(rng.randint(0, 25)):
            day += rng.choice([0, 0, 1])
            trades.append({"date": f"2026-09-{day:02d}", "outcome": rng.choice(["win", "loss", "open"]),
                           "pnl": round(rng.uniform(-900, 1200), 2),
                           "exitTime": f"2026-09-{day:02d}T10:{i:02d}:00-04:00"})
        want = js([{"fn": "trailingDrawdown", "args": {"trades": trades, "rules": rules}}])[0]
        got = risk.trailing_drawdown(trades, rules)
        for k in ("equity", "peak", "floor", "remaining"):
            assert got[k] == pytest.approx(want[k]), (dd_type, k)


def test_remaining_daily_loss_matches_core():
    for pnl in (-1500, -1000, -999.5, -1, 0, 250, 4000):
        trades = [{"date": "2026-09-15", "outcome": "loss" if pnl < 0 else "win", "pnl": pnl}]
        want = js([{"fn": "remainingDailyLoss", "args": {"trades": trades, "rules": RULES, "date": "2026-09-15"}}])[0]
        assert risk.remaining_daily_loss(pnl, RULES) == pytest.approx(want)
