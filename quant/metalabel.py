"""Meta-labeling the mechanical setups (Lopez de Prado, Advances in Financial Machine Learning, ch. 3).

The primary model is the unfiltered Judas replay: it decides side, entry and stop. A secondary
classifier decides whether to take each setup, from features known when the order is placed. The
question it answers: is there signal in the setups that the fixed filters miss?

Validation is walk-forward by calendar year. For test year Y the model trains only on trades that
filled before Y, minus a purge: any training trade whose exit falls within `embargo` of the test
window's start is dropped, so no training label overlaps the test period. Thresholds are set from
training data only (take a setup when its predicted win probability beats the training base rate).
Every variant tried is counted in the deflated Sharpe ratio of the one reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .stats import deflated_sharpe, sharpe

MODEL = "p2r_trail"
FEATURES = ("long", "s_asia", "s_ny_am", "s_ny_pm", "k_pd_pw", "k_session", "k_eq", "log_risk",
            "gap_over_risk", "obstacles", "target_r", "htf_agree", "sweep_to_fill_min", "hour")


def frame(trades: list[dict]) -> pd.DataFrame:
    """One row per setup: features known at the order, the label, and the R it made."""
    rows = []
    for t in trades:
        fill = pd.Timestamp(t["fill_time"])
        ext = t["targets"]["ext"][0]
        kind = t["sweep_kind"]
        rows.append({
            "fill": fill, "exit": pd.Timestamp(t["exit_time"]), "date": t["date"], "session": t["session"],
            "long": float(t["side"] == "long"),
            "s_asia": float(t["session"] == "asia"), "s_ny_am": float(t["session"] == "ny_am"),
            "s_ny_pm": float(t["session"] == "ny_pm"),
            "k_pd_pw": float(kind[:2] in ("PD", "PW")),
            "k_session": float(kind.split("_")[0] in ("ASIA", "LONDON", "NYAM")),
            "k_eq": float(kind.startswith("EQ")),
            "log_risk": float(np.log(t["risk_pts"])),
            "gap_over_risk": t["gap_pts"] / t["risk_pts"],
            "obstacles": float(t["obstacles"]),
            "target_r": abs(t["entry"] - ext) / t["risk_pts"],
            "htf_agree": float(t["htf_agree"]),
            "sweep_to_fill_min": (fill - pd.Timestamp(t["sweep_time"])).total_seconds() / 60,
            "hour": fill.hour + fill.minute / 60,
            "r": t["r"][MODEL],
        })
    d = pd.DataFrame(rows)
    for col in ("fill", "exit"):  # mixed DST offsets: normalize through UTC
        d[col] = pd.to_datetime(d[col], utc=True).dt.tz_convert("America/New_York")
    d = d.sort_values("fill").reset_index(drop=True)
    d["y"] = (d["r"] > 0).astype(int)
    return d


def splits(d: pd.DataFrame, min_train_years: int = 2, embargo: pd.Timedelta = pd.Timedelta(days=1)):
    """(test year, train index, test index) walk-forward folds with purging and an embargo."""
    years = sorted(d["fill"].dt.year.unique())
    for y in years[min_train_years:]:
        start = pd.Timestamp(year=int(y), month=1, day=1, tz=d["fill"].dt.tz)
        end = pd.Timestamp(year=int(y) + 1, month=1, day=1, tz=d["fill"].dt.tz)
        test = d.index[(d["fill"] >= start) & (d["fill"] < end)]
        train = d.index[(d["fill"] < start) & (d["exit"] < start - embargo)]
        if len(test) and d.loc[train, "y"].nunique() == 2:
            yield int(y), train, test


def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "boosted_trees": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                                l2_regularization=1.0, random_state=7),
    }


def walk_forward(d: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score
    X = d[list(FEATURES)].to_numpy(dtype=float)
    out = {"setups": int(len(d)), "folds": [], "models": {}}
    take_by_model: dict[str, np.ndarray] = {}
    for name, make in _models().items():
        take = np.zeros(len(d), dtype=bool)
        tested = np.zeros(len(d), dtype=bool)
        folds = []
        for year, tr, te in splits(d):
            m = make().fit(X[tr], d.loc[tr, "y"])
            p_tr, p_te = m.predict_proba(X[tr])[:, 1], m.predict_proba(X[te])[:, 1]
            base = d.loc[tr, "y"].mean()
            take[te] = p_te > base
            tested[te] = True
            yt = d.loc[te, "y"]
            folds.append({"year": year, "train": int(len(tr)), "test": int(len(te)),
                          "auc": round(float(roc_auc_score(yt, p_te)), 3) if yt.nunique() == 2 else None,
                          "taken": int(take[te].sum()),
                          "mean_r_all": round(float(d.loc[te, "r"].mean()), 3),
                          "mean_r_taken": round(float(d.loc[te[take[te]], "r"].mean()), 3) if take[te].any() else None})
        take_by_model[name] = take & tested
        out["models"][name] = {"folds": folds}
        out["tested"] = int(tested.sum())
    tested_mask = np.zeros(len(d), dtype=bool)
    for _, _, te in splits(d):
        tested_mask[te] = True
    r_all = d.loc[tested_mask, "r"].to_numpy()
    trial = [sharpe(r_all)] + [sharpe(d.loc[t, "r"].to_numpy()) for t in take_by_model.values()]
    out["oos_all"] = _summ(d[tested_mask])
    for name, take in take_by_model.items():
        s = _summ(d[take])
        s["deflated"] = deflated_sharpe(d.loc[take, "r"].to_numpy(), trial)
        s["lift_vs_all_ci95"] = _lift_ci(d, tested_mask, take)
        out["models"][name]["oos_taken"] = s
    return out


def _summ(x: pd.DataFrame) -> dict:
    r = x["r"].to_numpy()
    if r.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(11)
    keys = (x["date"] + x["session"]).to_numpy()
    uk = np.unique(keys)
    sums = np.array([r[keys == k].sum() for k in uk])
    cnts = np.array([(keys == k).sum() for k in uk])
    pick = rng.integers(0, uk.size, size=(5000, uk.size))
    means = sums[pick].sum(1) / cnts[pick].sum(1)
    return {"n": int(r.size), "mean_r": round(float(r.mean()), 3), "total_r": round(float(r.sum()), 2),
            "win_rate": round(float((r > 0).mean()), 3),
            "ci95_sessions": [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]}


def _lift_ci(d: pd.DataFrame, tested: np.ndarray, take: np.ndarray, boot: int = 5000) -> list[float]:
    """95% CI of (mean R of taken setups) - (mean R of all tested setups), resampling sessions."""
    x = d[tested]
    tk = take[tested]
    keys = (x["date"] + x["session"]).to_numpy()
    uk = np.unique(keys)
    r = x["r"].to_numpy()
    s_all = np.array([r[keys == k].sum() for k in uk]); n_all = np.array([(keys == k).sum() for k in uk])
    s_tk = np.array([r[(keys == k) & tk].sum() for k in uk]); n_tk = np.array([((keys == k) & tk).sum() for k in uk])
    pick = np.random.default_rng(13).integers(0, uk.size, size=(boot, uk.size))
    with np.errstate(invalid="ignore", divide="ignore"):
        lift = s_tk[pick].sum(1) / n_tk[pick].sum(1) - s_all[pick].sum(1) / n_all[pick].sum(1)
    lift = lift[np.isfinite(lift)]
    return [round(float(np.percentile(lift, 2.5)), 3), round(float(np.percentile(lift, 97.5)), 3)]


def from_report(path: Path) -> dict:
    trades = json.loads(Path(path).read_text(encoding="utf-8"))["baseline_trades"]
    d = frame(trades)
    rep = walk_forward(d)
    rep["source"] = str(path)
    return rep
