"""The definitions in signals.py, the fill rules in backtest.py, and the two properties a backtest
cannot be trusted without: it never reads a bar after the plan deadline, and every plan it trades
passes core/plan.js.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant.backtest import Costs, Market, _simulate, run
from quant.config import load_config, load_rules, session_window
from quant.signals import IFVG, find_ifvgs, trading_day
from quant.strategy import Params, Setup, build_plan

ET = "America/Toronto"


def bars(rows, start="2026-09-15 09:30", freq="1min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz=ET)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


# --- signals ---------------------------------------------------------------------------------

def test_bearish_fvg_inverts_on_a_body_close_above_not_a_wick():
    b = bars([
        (112, 113, 110, 111),   # c1, low 110
        (111, 111, 104, 104.5),  # displacement down
        (104.5, 105, 102, 103),  # c3, high 105 -> SIBI [105, 110]
        (103, 112, 103, 108),    # wick above 110, closes inside: not an inversion
        (108, 113, 107, 111.5),  # bullish body closes above 110: inversion
    ])
    gaps = find_ifvgs(b, min_gap=1.0)
    assert gaps == [IFVG("long", 105.0, 110.0, formed=2, inverted=4)]


def test_bullish_fvg_inverts_short_and_respects_min_gap():
    b = bars([
        (100, 101, 99, 100.5),   # c1 high 101
        (100.5, 108, 100.5, 107.5),
        (107.5, 109, 103, 108),  # c3 low 103 -> BISI [101, 103]
        (108, 108, 99, 100),     # bearish body closes below 101
    ])
    assert find_ifvgs(b, min_gap=1.0) == [IFVG("short", 101.0, 103.0, formed=2, inverted=3)]
    assert find_ifvgs(b, min_gap=2.5) == []


# --- fills -----------------------------------------------------------------------------------

def setup_long(entry=100.0, stop=98.0, target=108.0):
    plan = {"bias": "long"}
    return Setup(plan, "ok", entry=entry, stop=stop, target=target, box=IFVG("long", 98.5, 101.5, 0, 0))


def sim(rows, st=None, fill="through"):
    m = Market(bars(rows + [rows[-1]]))  # one spare bar past the session so the bar size is inferable
    return _simulate(m, st or setup_long(), 0, len(rows), 2, Costs(commission=0, slip_ticks=1, fill=fill), 0.0, "NQ")


def test_limit_fills_then_target():
    t = sim([(105, 106, 99.5, 100), (100, 110, 99, 109)])
    assert (t["entryPrice"], t["exitPrice"], t["exitReason"]) == (100.0, 108.0, "target")
    assert t["gross"] == pytest.approx((108 - 100) * 20 * 2)


def test_fill_bar_can_stop_but_cannot_take_profit():
    t = sim([(105, 112, 97, 99)])  # fills, stops, and touches the target, all in one bar
    assert (t["exitPrice"], t["exitReason"]) == (97.75, "stop")


def test_stop_wins_when_one_bar_touches_both():
    t = sim([(105, 106, 99.5, 100), (100, 109, 97.5, 104)])
    assert t["exitReason"] == "stop"


def test_touch_is_not_a_fill_unless_asked():
    rows = [(105, 106, 100, 101), (101, 103, 100, 102)]
    assert sim(rows) is None
    assert sim(rows, fill="touch")["entryPrice"] == 100.0


def test_open_through_the_stop_cancels_the_order():
    assert sim([(103, 104, 101, 102), (97, 99, 96, 98)]) is None


def test_flat_at_session_end_pays_slippage():
    t = sim([(105, 106, 99.5, 100), (100, 104, 99.5, 103)])
    assert (t["exitPrice"], t["exitReason"]) == (102.75, "session_end")


# --- the two properties ----------------------------------------------------------------------

def random_walk(days=12, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-09-07 18:00", "2026-09-07 18:00", tz=ET)
    idx = pd.date_range(idx[0], periods=days * 288, freq="5min")
    idx = idx[~((idx.dayofweek == 5) | ((idx.dayofweek == 4) & (idx.hour >= 17)) | ((idx.dayofweek == 6) & (idx.hour < 18)))]
    steps = rng.standard_t(3, len(idx)) * 6
    close = 29000 + np.cumsum(steps)
    open_ = np.concatenate([[29000], close[:-1]])
    wick = np.abs(rng.normal(0, 4, (2, len(idx))))
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    q = lambda x: np.round(x / 0.25) * 0.25  # noqa: E731
    return pd.DataFrame({"open": q(open_), "high": q(high), "low": q(low), "close": q(close), "volume": 1.0}, index=idx)


def test_plan_never_reads_a_bar_after_the_deadline():
    full = random_walk()
    rules, config = load_rules(), load_config()
    checked = 0
    for d in pd.date_range("2026-09-09", "2026-09-17").date:
        for s in ("ny_am", "ny_pm", "asia"):
            w = session_window(config, s, d)
            cut = full[full.index < pd.Timestamp(w.plan_deadline) - pd.Timedelta(minutes=5) + pd.Timedelta(seconds=1)]
            a = build_plan(full, trading_day(full.index), 300, find_ifvgs(full), w, rules, Params(), "5m")
            b = build_plan(cut, trading_day(cut.index), 300, find_ifvgs(cut), w, rules, Params(), "5m")
            assert (a.plan, a.reason) == (b.plan, b.reason)
            checked += a.plan is not None
    assert checked > 0, "the fixture produced no plans, so the property was never exercised"


def test_every_automated_plan_passes_core_plan_js():
    rules, config = load_rules(), load_config()
    res = run(Market(random_walk()), rules, config, Params(), Costs())
    reasons = [s["reason"] for s in res.sessions]
    assert not [r for r in reasons if r.startswith("rejected")], reasons
    assert any(r not in ("no prior trading day in the data",) and not r.startswith("no live") for r in reasons)
