"""The detector spec in quant/liquidity.py: FVG states, pools and sweeps, the DOL pick, the run
grade, and the property none of them is worth anything without: no bar at or after the cut is read.
"""

import numpy as np
import pandas as pd
import pytest

from quant.liquidity import FVG, Pool, find_fvgs, find_pools, grade_run, pick_dol, swing_pivots
from quant.market import cache_path, read_csv
from quant.signals import find_ifvgs

ET = "America/Toronto"


def bars(rows, start="2026-09-15 09:30", freq="5min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz=ET)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


def from_highs(highs, start="2026-09-15 09:30"):
    """Bars shaped only by their highs (lows mirror below), for pivot fixtures."""
    return bars([(x - 0.5, x, x - 1, x - 0.5) for x in highs], start)


def from_lows(lows, start="2026-09-15 09:30"):
    return bars([(x + 0.5, x + 1, x, x + 0.5) for x in lows], start)


def random_walk(n=1500, seed=7, start="2026-09-13 18:00"):
    rng = np.random.default_rng(seed)
    c = 20000 + np.cumsum(rng.normal(0, 4, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + rng.exponential(2, n)
    l = np.minimum(o, c) - rng.exponential(2, n)
    return bars(np.round(np.c_[o, h, l, c] * 4) / 4, start)


# --- 1. FVG states ---------------------------------------------------------------------------

BISI = [
    (100, 101, 99, 100.5),      # c1, high 101
    (100.5, 108, 100.5, 107.5),  # displacement up
    (107.5, 109, 105, 108),      # c3, low 105 -> BISI [101, 105], CE 103
]


def state_after(*rows, **kw):
    b = bars(BISI + list(rows))
    (g,) = find_fvgs(b, len(b), **kw)
    assert (g.side, g.low, g.high, g.formed, g.ce) == ("bull", 101.0, 105.0, 2, 103.0)
    return g


def test_fvg_open():
    assert state_after((108, 110, 106, 109)).state == "open"


def test_fvg_mitigated_by_a_wick_into_the_gap_short_of_ce():
    g = state_after((108, 110, 106, 109), (108, 108.5, 104, 107))
    assert (g.state, g.mitigated, g.ce_hit) == ("mitigated", 4, None)


def test_fvg_ce():
    g = state_after((108, 108, 103, 106))
    assert (g.state, g.mitigated, g.ce_hit, g.filled) == ("ce", 3, 3, None)


def test_fvg_filled_by_a_wick_through_with_a_close_back_inside():
    g = state_after((108, 108, 100, 104))
    assert (g.state, g.filled, g.inverted) == ("filled", 3, None)


def test_wick_through_is_not_an_inversion_nor_is_a_bullish_close_below():
    # Closes below the gap low, but the candle is bullish (close > open): § 4.1 wants a bearish body.
    g = state_after((99, 104, 98, 100))
    assert (g.state, g.inverted) == ("filled", None)


def test_fvg_inverted_by_a_bearish_body_close_below_becomes_a_short_ifvg():
    g = state_after((108, 108, 99, 100))
    assert (g.state, g.inverted, g.ifvg_side, g.live_ifvg) == ("inverted", 3, "short", True)


def test_ifvg_invalidated_by_a_close_back_above_its_far_edge():
    g = state_after((108, 108, 99, 100), (100, 104.5, 100, 104), (104, 106, 103, 105.5))
    assert (g.inverted, g.invalidated, g.live_ifvg) == (3, 5, False)  # a wick to 104.5 did not kill it


def test_inversion_only_within_max_age_like_find_ifvgs():
    g = state_after((108, 110, 106, 109), (108, 108, 99, 100), max_age=1)
    assert (g.state, g.inverted) == ("filled", None)


def test_bearish_fvg_mirror_inverts_long():
    b = bars([
        (112, 113, 110, 111),    # c1, low 110
        (111, 111, 104, 104.5),
        (104.5, 105, 102, 103),  # c3, high 105 -> SIBI [105, 110]
        (103, 112, 103, 108),    # wick above 110, close inside: filled, not inverted
        (108, 113, 107, 111.5),  # bullish body close above 110: inverted
    ])
    (g,) = find_fvgs(b, 4)
    assert (g.side, g.low, g.high, g.state, g.filled) == ("bear", 105.0, 110.0, "filled", 3)
    (g,) = [g for g in find_fvgs(b, 5) if g.side == "bear"]  # bar 4 also forms a BISI
    assert (g.state, g.inverted, g.ifvg_side) == ("inverted", 4, "long")


def test_inversions_match_find_ifvgs():
    for b in [random_walk(seed=s) for s in (1, 2, 3)]:
        ours = sorted((g.ifvg_side, g.low, g.high, g.formed, g.inverted)
                      for g in find_fvgs(b, len(b)) if g.inverted is not None)
        ref = sorted((g.side, g.low, g.high, g.formed, g.inverted) for g in find_ifvgs(b))
        assert ours == ref and ref


@pytest.mark.skipif(not cache_path("NQ=F", "5m").exists(), reason="no cached NQ 5m bars")
def test_inversions_match_find_ifvgs_on_cached_nq():
    b = read_csv(cache_path("NQ=F", "5m"))
    ours = sorted((g.ifvg_side, g.low, g.high, g.formed, g.inverted)
                  for g in find_fvgs(b, len(b)) if g.inverted is not None)
    assert ours == sorted((g.side, g.low, g.high, g.formed, g.inverted) for g in find_ifvgs(b))


# --- 2. Pools --------------------------------------------------------------------------------

def kinds(pools, kind):
    return [p for p in pools if p.kind == kind]


def test_equal_highs_cluster_within_tolerance():
    b = from_highs([195, 196, 200, 197, 196, 197, 198, 201.5, 198, 197, 196])
    assert swing_pivots(b, len(b))[0].tolist() == [2, 7]
    (eq,) = kinds(find_pools(b, len(b), eq_tol=2.0), "EQH")
    assert (eq.price, eq.formed, eq.side, eq.swept) == (201.5, 7, "BSL", None)
    assert kinds(find_pools(b, len(b), eq_tol=1.0), "EQH") == []


def test_equal_highs_need_the_level_to_hold_between_touches():
    # 200 and 201 are within tolerance, but 202.5 traded through 200 in between: the pool is
    # 202.5 with 201 (1.5 apart), not 200 with 201.
    b = from_highs([195, 196, 200, 197, 196, 202.5, 197, 196, 201, 198, 197])
    (eq,) = kinds(find_pools(b, len(b), eq_tol=2.0), "EQH")
    assert (eq.price, eq.formed) == (202.5, 8)


def test_equal_lows_and_their_sweep():
    lows = [105, 104, 100, 103, 104, 103, 102, 100.75, 102, 103, 104]
    b = from_lows(lows + [99.5])  # the last bar wicks below 100: swept
    (eq,) = kinds(find_pools(b, len(b)), "EQL")
    assert (eq.price, eq.formed, eq.side, eq.swept) == (100.0, 7, "SSL", 11)


def test_pivot_confirms_only_pivot_len_bars_late():
    b = from_highs([195, 196, 200, 197, 196])
    assert swing_pivots(b, 4)[0].tolist() == []  # bar 4 not yet closed
    assert swing_pivots(b, 5)[0].tolist() == [2]


def day_bars():
    """Flat 5m bars from Sun 2026-09-13 18:00 through Tue 09-15 12:00 ET, then shaped levels."""
    idx = pd.date_range("2026-09-13 18:00", "2026-09-15 12:00", freq="5min", tz=ET, inclusive="left")
    b = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0}, index=idx)
    def at(t):
        return b.index.get_loc(pd.Timestamp(t, tz=ET))
    b.iloc[at("2026-09-14 10:00"), 1] = 110.0  # PDH, trading day Mon 09-14
    b.iloc[at("2026-09-14 14:00"), 2] = 90.0   # PDL
    b.iloc[at("2026-09-14 21:00"), 1] = 105.0  # Asia high of Tue 09-15
    b.iloc[at("2026-09-14 22:00"), 2] = 96.0   # Asia low
    b.iloc[at("2026-09-15 03:00"), 1] = 104.0  # London high
    b.iloc[at("2026-09-15 04:00"), 2] = 97.0   # London low
    b.iloc[at("2026-09-15 06:00"), 2] = 95.0   # pre-market low sweeps the Asia low
    b.iloc[at("2026-09-15 09:45"), 1] = 104.0  # NY AM touches London high exactly: not a sweep
    return b, at


def test_session_and_previous_day_pools_and_sweeps():
    b, at = day_bars()
    cut = at("2026-09-15 10:30")
    pools = {p.kind: p for p in find_pools(b, cut)}  # the later day's session overwrites the earlier
    assert (pools["PDH"].price, pools["PDH"].swept) == (110.0, None)
    assert (pools["PDL"].price, pools["PDL"].swept) == (90.0, None)
    assert (pools["ASIA_H"].price, pools["ASIA_H"].formed) == (105.0, at("2026-09-14 21:00"))
    assert pools["ASIA_L"].swept == at("2026-09-15 06:00")
    assert (pools["LONDON_H"].price, pools["LONDON_H"].swept) == (104.0, None)  # equal is not beyond
    assert pools["NYAM_H"].formed < at("2026-09-15 00:00")  # today's NY AM is still open at 10:30
    assert "PWH" not in pools  # no previous week in the data
    later = {p.kind: p for p in find_pools(b, at("2026-09-15 11:00"))}
    assert (later["NYAM_H"].price, later["NYAM_H"].formed) == (104.0, at("2026-09-15 09:45"))


# --- 3. DOL ----------------------------------------------------------------------------------

def P(kind, price, side="BSL", formed=0, swept=None):
    return Pool(kind, side, price, formed, swept)


FLAT = bars([(100, 100.5, 99.5, 100)] * 3)  # last close 100


def test_dol_nearest_significant_beats_a_farther_one():
    # The knowledge-base example: a clean equal high 40 points away over a 150-point daily high.
    d = pick_dol(FLAT, 3, "long", pools=[P("PDH", 250), P("EQH", 140)])
    assert (d.pool.kind, d.distance, d.next_tier.kind) == ("EQH", 40.0, "PDH")
    d = pick_dol(FLAT, 3, "long", pools=[P("EQH", 250), P("ASIA_H", 140)])
    assert d.pool.kind == "ASIA_H"  # nearest wins over priority outside a tick


def test_dol_eq_wins_a_one_tick_tie_but_not_two():
    d = pick_dol(FLAT, 3, "long", pools=[P("ASIA_H", 140), P("EQH", 140.25), P("PDH", 140)])
    assert d.pool.kind == "EQH"
    d = pick_dol(FLAT, 3, "long", pools=[P("ASIA_H", 140), P("EQH", 140.5)])
    assert (d.pool.kind, d.next_tier.kind) == ("ASIA_H", "EQH")
    d = pick_dol(FLAT, 3, "short", pools=[P("PWL", 60, "SSL"), P("PDL", 60, "SSL")])
    assert d.pool.kind == "PDL"  # PD beats PW


def test_dol_swing_only_when_no_significant_pool():
    d = pick_dol(FLAT, 3, "long", pools=[P("SWING_H", 105), P("PDH", 200)])
    assert d.pool.kind == "PDH"
    d = pick_dol(FLAT, 3, "long", pools=[P("SWING_H", 105), P("SWING_H", 120), P("PDH", 200, swept=1)])
    assert (d.pool.kind, d.next_tier.price) == ("SWING_H", 120.0) and "swing" in d.reason


def test_dol_ignores_swept_wrong_side_and_reports_none():
    pools = [P("PDH", 150, swept=2), P("EQL", 90, "SSL"), P("ASIA_H", 95)]
    d = pick_dol(FLAT, 3, "long", pools=pools)
    assert d.pool is None and "no unswept BSL" in d.reason


def test_dol_default_bias_is_premium_discount_of_the_previous_day():
    b, at = day_bars()
    d = pick_dol(b, at("2026-09-15 10:30"))  # close 100 vs PD EQ (110 + 90) / 2 = 100: premium
    assert d.bias == "short" and "premium" in d.bias_source


def test_dol_recency_drops_old_intraday_pools():
    b, at = day_bars()
    pools = find_pools(b, at("2026-09-15 10:30"), lookback_sessions=1)
    assert {p.kind for p in pools} >= {"PDH", "PDL", "ASIA_H"}  # today's sessions and PD stay
    assert all(p.kind in ("PDH", "PDL") or b.index[p.formed] >= pd.Timestamp("2026-09-14 18:00", tz=ET)
               for p in pools)


# --- 4. LRLR / HRLR --------------------------------------------------------------------------

def uptrend():
    """Hourly bars rising 0.1 a bar for three weeks: daily and weekly bias +1, no FVGs."""
    idx = pd.date_range("2026-08-30 18:00", "2026-09-16 12:00", freq="1h", tz=ET)
    p = 100 + 0.1 * np.arange(len(idx))
    return pd.DataFrame({"open": p, "high": p + 0.3, "low": p - 0.3, "close": p + 0.1, "volume": 1.0}, index=idx)


def test_lrlr_aligned_and_clean():
    b = uptrend()
    assert find_fvgs(b, len(b)) == []
    g = grade_run(b, len(b), target=b["close"].iat[-1] + 50, direction="long")
    assert (g.grade, g.daily_bias, g.weekly_bias, g.aligned, g.obstacles) == ("LRLR", 1, 1, True, [])


def test_hrlr_counter_trend_whatever_the_path():
    b = uptrend()
    g = grade_run(b, len(b), target=b["close"].iat[-1] - 50, direction="short")
    assert (g.grade, g.aligned, g.obstacles) == ("HRLR", False, [])
    assert any("counter" in r for r in g.reasons)


def test_hrlr_aligned_but_an_obstacle_in_the_path():
    b = uptrend()
    px = b["close"].iat[-1]
    fvgs = [FVG("bear", px + 10, px + 12, 5, "open"),          # in the path: obstacle
            FVG("bear", px + 20, px + 22, 5, "filled"),        # filled: no obstacle
            FVG("bull", px + 30, px + 32, 5, "open"),          # same direction: no obstacle
            FVG("bear", px + 80, px + 82, 5, "mitigated")]     # beyond the target
    g = grade_run(b, len(b), target=px + 50, direction="long", fvgs=fvgs)
    assert (g.grade, g.aligned, [o.low for o in g.obstacles]) == ("HRLR", True, [px + 10])
    assert grade_run(b, len(b), px + 50, "long", fvgs=fvgs, max_obstacles=1).grade == "LRLR"


# --- no lookahead ----------------------------------------------------------------------------

def everything(b, cut):
    fvgs = find_fvgs(b, cut)
    pools = find_pools(b, cut)
    dol = pick_dol(b, cut)
    grade = grade_run(b, cut, dol.pool.price, dol.bias) if dol.pool else None
    return fvgs, pools, swing_pivots(b, cut)[0].tolist(), swing_pivots(b, cut)[1].tolist(), dol, grade


@pytest.mark.parametrize("cut", [400, 1000, 1499])
def test_no_lookahead_bars_after_cut_never_change_anything(cut):
    b = random_walk()
    base = everything(b, cut)
    assert base[0] and base[1]
    mutated = b.copy()
    rng = np.random.default_rng(99)
    mutated.iloc[cut:, :4] += rng.normal(0, 200, (len(b) - cut, 4))
    assert everything(mutated, cut) == base
    assert everything(b.iloc[:cut], cut) == base
