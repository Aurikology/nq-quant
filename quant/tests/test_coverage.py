"""quant/coverage.py: one-to-one time matching of hand trades to detector setups."""

import pandas as pd

from quant.coverage import TZ, coverage, match


def frame(rows, cols):
    d = pd.DataFrame(rows, columns=cols)
    d["t"] = pd.to_datetime(d["t"]).dt.tz_localize(TZ)
    return d


HAND = frame([("2024-01-02 09:40", "long", 2.0, 0), ("2024-01-02 09:42", "long", -1.0, 0),
              ("2024-01-03 10:00", "short", 0.0, 0)], ["t", "side", "r", "session"])
SETUPS = frame([("2024-01-02 09:45", "long", 1.0), ("2024-01-03 10:50", "short", -1.0),
                ("2024-01-03 10:05", "long", 3.0)], ["t", "side", "r"])


def test_match_is_one_to_one_same_side_and_within_the_window():
    m = match(HAND, SETUPS, 15)
    assert m.tolist() == [0, -1, -1]  # the second long finds the setup taken; the short has no short within 15


def test_wider_window_finds_more_and_counts_precision_on_replayed_days():
    rep = coverage(HAND, SETUPS, windows=(15, 60))
    w15, w60 = rep["by_window"]
    assert w15["recall"] == round(1 / 3, 3) and w60["recall"] == round(2 / 3, 3)
    assert rep["setups_in_replayed_spans"] == 3
    assert w60["precision"] == round(2 / 3, 3)
    assert w15["hand_r_matched"] == 2.0
