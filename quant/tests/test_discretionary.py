"""The two discretionary rules in quant/judas.py: higher-timeframe gap entry and low-resistance target."""

import pandas as pd

from quant.judas import Params, _discretionary, htf_bars, in_htf_gap


def minute_bars(rows, start="2026-11-05 09:30"):
    idx = pd.date_range(start, periods=len(rows), freq="min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def five_min(blocks):
    """Each block (o, h, l, c) becomes five 1m bars that resample back to exactly that 5m bar."""
    rows = []
    for o, h, l, c in blocks:
        rows += [[o, o, o, o], [o, h, o, o], [o, o, l, o], [o, o, o, c], [c, c, c, c]]
    return minute_bars(rows)


# 5m: a SIBI (bearish gap) [110, 115] forms on the third bar, then price drifts below it.
DROP = [(120, 121, 115, 116), (116, 116, 108, 109), (109, 110, 104, 105), (105, 106, 103, 104)]


def test_htf_bars_drop_the_unfinished_bar():
    b = five_min(DROP)
    assert len(htf_bars(b, len(b), 5)) == 4
    assert len(htf_bars(b, len(b) - 1, 5)) == 3  # the last 5m bar is not closed yet


def test_entry_inside_a_respected_bearish_htf_gap_passes():
    b = five_min(DROP)
    assert in_htf_gap(b, len(b), 112.0, 5)
    assert not in_htf_gap(b, len(b), 100.0, 5)


def test_htf_gap_is_invisible_until_its_third_bar_closes():
    b = five_min(DROP)
    assert not in_htf_gap(b, 14, 112.0, 5)  # bars 0..13: the third 5m bar is not complete


def test_discretionary_reasons():
    b = five_min(DROP)
    assert _discretionary(b, len(b), 100.0, 90.0, Params(htf_gap_min=5)) == "entry not in a 5m gap"
    assert _discretionary(b, len(b), 112.0, 90.0, Params(htf_gap_min=5)) is None


def test_open_bullish_gap_below_entry_makes_the_target_high_resistance():
    # 1m BISI [101, 105] below a short entry at 110, target 95.
    b = minute_bars([[100, 101, 99, 100.5], [100.5, 108, 100.5, 107.5], [107.5, 109, 105, 108], [108, 109, 107, 108]])
    assert _discretionary(b, len(b), 110.0, 95.0, Params(lr_target=True)) == "high-resistance target"
    assert _discretionary(b, len(b), 110.0, 106.0, Params(lr_target=True)) is None  # target above the gap
