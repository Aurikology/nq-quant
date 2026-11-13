"""quant/journal.py: balance-chain session linking and the scoring of a discretionary trade export."""

import pandas as pd

from quant.journal import link_sessions, load_fxreplay, score, sessions


def export(rows, tmp_path):
    """rows: (start, initialBalance, rPnL, balance after, R)."""
    df = pd.DataFrame([{
        "id": i, "dateStart": s, "status": "closed", "initialBalance": ib, "rPnL": p,
        "currentRealizedBalance": bal, "avgRiskReward": r, "maxRiskReward": max(r, 0.5),
    } for i, (s, ib, p, bal, r) in enumerate(rows)])
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    return path


# Two sessions on the same account size, interleaved in time (as the export sorts by date).
ROWS = [
    ("2023/01/03 09:31", 100000, 1000, 101000, 2.0),   # A
    ("2023/01/03 09:40", 100000, -500, 99500, -1.0),   # B
    ("2023/01/04 09:31", 100000, 0, 101000, 0.0),      # A, breakeven
    ("2023/01/04 09:45", 100000, 1500, 101000, 3.0),   # B: 99,500 + 1,500
    ("2023/01/05 09:31", 100000, -500, 100500, -1.0),  # A
]


def test_interleaved_sessions_are_split_by_balance_chain(tmp_path):
    d = load_fxreplay(export(ROWS, tmp_path))
    assert list(d["session"]) == [0, 1, 0, 1, 0]
    s = sessions(d)
    assert list(s["final_balance"]) == [100500.0, 101000.0]


def test_breakevens_count_as_trades_in_the_honest_win_rate(tmp_path):
    rep = score(load_fxreplay(export(ROWS, tmp_path)))
    assert (rep["wins"], rep["losses"], rep["breakevens"]) == (2, 2, 1)
    assert rep["win_rate_all"] == 0.4
    assert rep["win_rate_ex_be"] == 0.5
    assert rep["mean_r"] == 0.6


def test_a_session_is_excluded_by_its_final_balance_only(tmp_path):
    rep = score(load_fxreplay(export(ROWS, tmp_path)), exclude_final=(101000.0,))
    assert rep["trades"] == 3 and rep["sessions"] == 1 and rep["excluded_sessions"] == 1


def test_session_bootstrap_ci_brackets_the_mean(tmp_path):
    rep = score(load_fxreplay(export(ROWS * 1, tmp_path)))
    lo, hi = rep["ci95_sessions"]
    assert lo <= rep["mean_r"] <= hi
