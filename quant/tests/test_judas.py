"""Structure state and the trailing and partial exits in quant/judas.py (decision 2026-11-05)."""

import numpy as np
import pandas as pd

from quant.judas import _exit, structure


def bars(rows):
    idx = pd.date_range("2026-11-05 09:30", periods=len(rows), freq="min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def arrays(rows):
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2]  # high, low, close


def test_structure_flips_on_a_close_through_the_last_confirmed_swing():
    b = bars([
        [101, 102, 100, 100.5], [100.5, 101, 98, 99], [99, 100, 95, 96], [96, 99, 96, 98],
        [98, 103, 97, 102],  # confirms the swing low 95 at bar 2
        [102, 106, 101, 105], [105, 107, 104, 106], [106, 106, 102, 103],
        [103, 104, 100, 101],  # confirms the swing high 107 at bar 6
        [101, 101, 94, 94.5],  # closes under 95
        [94, 100, 94, 99],
        [99, 108, 99, 107.5],  # closes above 107
    ])
    s = structure(b)
    assert list(s[:9]) == [0] * 9  # no break yet
    assert s[9] == -1 and s[10] == -1
    assert s[11] == 1


def test_structure_has_no_lookahead():
    rows = [[100 + i, 101 + i, 99 + i, 100 + i] for i in range(12)]
    rows += [[110, 111, 90, 91], [91, 92, 85, 86]]
    full = structure(bars(rows))
    cut = structure(bars(rows[:10]))
    assert list(full[:10]) == list(cut)


def test_a_stop_hit_with_no_partial_is_minus_one_r():
    h, l, c = arrays([[100, 99, 99.5], [111, 100, 110]])
    r, j = _exit(h, l, c, 0, 2, 100.0, 110.0, [(1.0, 80.0)], None)
    assert r == -1.0 and j == 1


def test_slip_makes_a_stop_worse():
    h, l, c = arrays([[100, 99, 99.5], [111, 100, 110]])
    r, _ = _exit(h, l, c, 0, 2, 100.0, 110.0, [(1.0, 80.0)], None, slip=5.0)
    assert r == -1.5


def test_a_partial_moves_the_stop_to_entry_and_the_runner_scratches():
    # short at 100, stop 110; TP1 at 80 on bar 1, then back up through entry on bar 2
    h, l, c = arrays([[100, 99, 99], [99, 79, 80], [101, 80, 100], [100, 90, 95]])
    r, j = _exit(h, l, c, 0, 4, 100.0, 110.0, [(0.5, 80.0), (0.5, 40.0)], None)
    assert r == 1.0 and j == 2  # half of +2R, runner at entry


def test_the_trailed_stop_follows_a_confirmed_swing_down_and_locks_profit():
    h, l, c = arrays([
        [100, 99, 99],    # fill
        [99, 79, 80],     # TP1, armed
        [80, 70, 72],
        [75, 68, 70],     # swing high candidate needs bars either side
        [85, 69, 84],     # swing high 85
        [80, 66, 67],
        [78, 60, 62],     # confirms 85: stop 85.5
        [86, 60, 85],     # stopped at 85.5
    ])
    r, j = _exit(h, l, c, 0, 8, 100.0, 110.0, [(0.5, 80.0), (0.5, 40.0)], None, trail=2)
    assert j == 7
    assert abs(r - (0.5 * 2 + 0.5 * (100 - 85.5) / 10)) < 1e-9


def test_the_trailed_stop_never_widens():
    h, l, c = arrays([
        [100, 99, 99], [99, 79, 80], [80, 70, 72], [75, 68, 70], [85, 69, 84], [80, 66, 67],
        [78, 60, 62], [84, 60, 83],  # below 85.5: not stopped
        [90, 80, 81], [88, 75, 76], [85, 72, 73],  # a higher high at 90 must not lift the stop past 85.5
    ])
    r, j = _exit(h, l, c, 0, 11, 100.0, 110.0, [(0.5, 80.0), (0.5, 40.0)], None, trail=2)
    assert j == 8  # 90 trades through 85.5
    assert abs(r - (0.5 * 2 + 0.5 * (100 - 85.5) / 10)) < 1e-9


def test_with_no_runner_target_the_trail_rides_to_the_session_close():
    h, l, c = arrays([[100, 99, 99], [99, 79, 80], [80, 70, 72], [72, 60, 61]])
    r, j = _exit(h, l, c, 0, 4, 100.0, 110.0, [(0.5, 80.0), (0.5, None)], None, trail=2)
    assert j == 3
    assert abs(r - (1.0 + 0.5 * (100 - 61) / 10)) < 1e-9
