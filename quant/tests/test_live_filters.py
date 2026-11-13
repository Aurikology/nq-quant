"""The live entry filters in quant/judas.py (`Params.live()`), one rule per test, on short-side bars."""

import numpy as np
import pandas as pd

from quant.judas import LIVE_FILTERS, Params, _entry_filters, run
from quant.tests.test_liquidity import random_walk


def frame(rows):
    idx = pd.date_range("2026-11-05 09:30", periods=len(rows), freq="min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def hl(b):
    return b["high"].to_numpy(), b["low"].to_numpy()


QUIET = frame([[100, 101, 99, 100]] * 10)  # 2-point bars, no gaps
DOWN = np.full(10, -1)  # structure already short


def check(b, entry, risk, p, st=DOWN, live=None):
    h, l = hl(b)
    return _entry_filters(b, h, l, live or len(b), entry, risk, p, st)


def test_baseline_params_skip_nothing():
    assert check(QUIET, 100, 0.5, Params()) is None


def test_min_stop_skips_a_stop_under_the_floor():
    assert "under 3.0" in check(QUIET, 100, 2.5, Params(min_stop=3.0))
    assert check(QUIET, 100, 3.0, Params(min_stop=3.0)) is None


def test_stop_small_next_to_recent_bars_is_skipped():
    wide = frame([[100, 120, 80, 100]] * 10)  # 40-point bars, like the 2026-11-06 long
    assert "small next to 40.00-pt bars" in check(wide, 100, 8.7, Params(stop_vs_range=0.5))
    assert check(wide, 100, 20.0, Params(stop_vs_range=0.5)) is None


def test_structure_must_already_point_the_trades_way():
    p = Params(structure_agree=True)
    assert check(QUIET, 100, 5, p, st=np.full(10, 1)) == "against structure"
    assert check(QUIET, 100, 5, p, st=np.zeros(10, dtype=int)) == "against structure"
    assert check(QUIET, 100, 5, p, st=DOWN) is None


def test_structure_is_read_on_the_bar_before_the_order_only():
    st = np.r_[np.full(9, -1), 1]  # flips on bar 9, the order bar itself
    assert check(QUIET, 100, 5, Params(structure_agree=True), st=st, live=9) is None


def test_open_opposing_gap_between_entry_and_2r_is_skipped():
    # A 12-point bullish gap [80, 92] forms below a short entry at 100 (2R = 80 at 10 risk).
    b = frame([[79, 80, 78, 79.5], [79.5, 95, 79.5, 94], [94, 96, 92, 95]] + [[95, 96, 94, 95]] * 5)
    assert "opposing gap" in check(b, 100, 10, Params(max_opposing_gap=10.0))
    assert check(b, 100, 10, Params(max_opposing_gap=15.0)) is None  # smaller than the limit
    assert check(b, 100, 3, Params(max_opposing_gap=10.0)) is None  # 2R = 94 stops above the gap


def test_live_preset_turns_every_filter_on():
    p, base = Params.live(), Params()
    for name in LIVE_FILTERS:
        assert getattr(p, name) != getattr(base, name), name
    assert p.htf == "both"
    assert Params.live(min_stop=5.0).min_stop == 5.0


def test_every_live_trade_passes_the_filters_and_reasons_are_logged():
    b = random_walk(n=6000, seed=3, start="2026-09-14 00:00")
    trades, logs = run(b, Params.live())
    for t in trades:
        assert t.risk_pts >= 3.0
    reasons = [r for lg in logs for r in lg.get("passed", [])]
    assert all(isinstance(r, str) for r in reasons)
