"""quant/metalabel.py: features, purged walk-forward folds, and that the model finds planted signal."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from quant.metalabel import FEATURES, frame, splits, walk_forward  # noqa: E402


def fake_trades(n=900, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    out = []
    t0 = pd.Timestamp("2019-01-02 09:45", tz="America/New_York")
    for i in range(n):
        fill = t0 + pd.Timedelta(days=int(i * 7 * 365 / n), minutes=int(rng.integers(0, 60)))
        risk = float(rng.uniform(3, 30))
        long = bool(rng.integers(0, 2))
        good = (risk > 15) if signal else bool(rng.integers(0, 2))  # planted: wide stops win
        r = float(rng.uniform(0.5, 3)) if (good and rng.random() < 0.8) else -1.0
        entry = 20000.0
        out.append({"date": str(fill.date()), "session": "ny_am", "side": "long" if long else "short",
                    "sweep_kind": "PDL", "sweep_time": str(fill - pd.Timedelta(minutes=5)),
                    "fill_time": str(fill), "exit_time": str(fill + pd.Timedelta(minutes=40)),
                    "entry": entry, "stop": entry - risk, "risk_pts": risk, "gap_pts": 4.0, "obstacles": 0,
                    "targets": {"ext": [entry + 3 * risk, "PDH"]}, "r": {"p2r_trail": r}, "htf_agree": True})
    return out


def test_frame_has_every_feature_and_no_outcome_leaks():
    d = frame(fake_trades(50))
    assert set(FEATURES) <= set(d.columns)
    assert not {"mfe_r", "mae_r"} & set(FEATURES)
    assert d["fill"].is_monotonic_increasing


def test_folds_train_only_on_the_past_and_purge_overlap():
    d = frame(fake_trades(400))
    d.loc[d.index[100], "exit"] = pd.Timestamp("2030-01-01", tz="America/New_York")  # a trade that never exits
    for year, tr, te in splits(d):
        start = pd.Timestamp(year=year, month=1, day=1, tz="America/New_York")
        assert (d.loc[tr, "fill"] < start).all()
        assert (d.loc[tr, "exit"] < start - pd.Timedelta(days=1)).all()
        assert (d.loc[te, "fill"].dt.year == year).all()
        assert 100 not in tr


def test_planted_signal_is_found_out_of_sample():
    rep = walk_forward(frame(fake_trades(900, signal=True)))
    for name, m in rep["models"].items():
        assert m["oos_taken"]["mean_r"] > rep["oos_all"]["mean_r"] + 0.3, name
        assert m["oos_taken"]["lift_vs_all_ci95"][0] > 0, name


def test_no_signal_gives_no_reliable_lift():
    rep = walk_forward(frame(fake_trades(900, seed=3, signal=False)))
    for m in rep["models"].values():
        lo, hi = m["oos_taken"]["lift_vs_all_ci95"]
        assert lo < 0.15
