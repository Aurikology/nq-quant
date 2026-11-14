"""How much of a discretionary trader's activity does the mechanical detector reproduce?

Inputs: the trader's replayed trades (`journal.load_fxreplay`, times in America/New_York, as the
replay tool's analytics are set) and the detector's unfiltered setups (`baseline_trades` in a judas
report). A hand trade and a setup match when they are on the same side and the setup's fill is
within `window` minutes of the hand entry.

- recall: the share of hand trades the detector also produced;
- precision: of the detector's setups inside the date spans the trader replayed, the share he took;
- the R of matched vs unmatched hand trades, which says whether the overlap is where his edge is.

Matching is greedy in time order and one-to-one, so one setup never explains two hand trades.
Only dates inside both data sets count: the detector runs on bars that start 2019-01-01.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "America/New_York"


def hand_frame(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "t": pd.to_datetime(d["dateStart"], format="%Y/%m/%d %H:%M:%S").dt.tz_localize(TZ, ambiguous="NaT",
                                                                                     nonexistent="NaT"),
        "side": np.where(d["side"].str.lower() == "buy", "long", "short"),
        "r": d["r"].to_numpy(), "session": d["session"].to_numpy(),
    }).dropna(subset=["t", "r"])
    return out.sort_values("t").reset_index(drop=True)


def setup_frame(trades: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame({
        "t": pd.to_datetime([t["fill_time"] for t in trades], utc=True).tz_convert(TZ),
        "side": [t["side"] for t in trades], "r": [t["r"]["p2r_trail"] for t in trades],
    })
    return out.sort_values("t").reset_index(drop=True)


def match(hand: pd.DataFrame, setups: pd.DataFrame, window: int) -> np.ndarray:
    """For each hand trade, the index of its matched setup or -1."""
    used = np.zeros(len(setups), dtype=bool)
    out = np.full(len(hand), -1)
    st = setups["t"].to_numpy()
    w = np.timedelta64(window, "m")
    for i, (t, side) in enumerate(zip(hand["t"].to_numpy(), hand["side"])):
        a, z = np.searchsorted(st, t - w), np.searchsorted(st, t + w, side="right")
        best, gap = -1, None
        for j in range(a, z):
            if used[j] or setups.at[j, "side"] != side:
                continue
            g = abs(st[j] - t)
            if gap is None or g < gap:
                best, gap = j, g
        if best >= 0:
            used[best] = True
            out[i] = best
    return out


def coverage(hand: pd.DataFrame, setups: pd.DataFrame, windows=(15, 30, 60),
             data_from: pd.Timestamp | None = None) -> dict:
    """`data_from`: first bar of the detector's data (hand trades before it cannot be matched).
    Defaults to midnight of the first setup's day."""
    lo = data_from if data_from is not None else setups["t"].min().normalize()
    hand = hand[hand["t"] >= lo].reset_index(drop=True)
    days = set()
    for _, g in hand.groupby("session"):  # the trader's replayed spans, by session
        days |= set(pd.date_range(g["t"].min().normalize(), g["t"].max().normalize(), freq="D").date)
    in_span = setups["t"].dt.date.isin(days)
    sp = setups[in_span].reset_index(drop=True)
    rep = {"hand_trades": int(len(hand)), "setups_in_replayed_spans": int(len(sp)), "by_window": []}
    for w in windows:
        m = match(hand, sp, w)
        hit = m >= 0
        rep["by_window"].append({
            "window_min": w,
            "recall": round(float(hit.mean()), 3) if len(hand) else None,
            "precision": round(float(hit.sum() / len(sp)), 3) if len(sp) else None,
            "hand_r_matched": round(float(hand.loc[hit, "r"].mean()), 3) if hit.any() else None,
            "hand_r_unmatched": round(float(hand.loc[~hit, "r"].mean()), 3) if (~hit).any() else None,
            "setup_r_matched": round(float(sp.loc[m[hit], "r"].mean()), 3) if hit.any() else None,
            "setup_r_all": round(float(sp["r"].mean()), 3) if len(sp) else None,
        })
    return rep
