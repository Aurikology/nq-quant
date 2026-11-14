"""PoolIndex (one pass, O(log n) sweep lookups) returns exactly what find_pools returns at every cut."""

import numpy as np
import pytest

from quant.liquidity import PoolIndex, RangeMax, find_pools
from quant.tests.test_liquidity import random_walk


@pytest.mark.parametrize("seed", range(12))
def test_first_above_matches_a_scan(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=int(rng.integers(1, 300)))
    t = RangeMax(v)
    for _ in range(200):
        start, x = int(rng.integers(0, len(v) + 2)), float(rng.normal())
        hits = np.flatnonzero(v[start:] > x)
        assert t.first_above(start, x) == (start + int(hits[0]) if hits.size else None)


@pytest.mark.parametrize("seed", range(6))
def test_pool_index_equals_find_pools_at_every_sampled_cut(seed):
    b = random_walk(n=3000, seed=seed, start="2026-09-13 18:00")
    ix = PoolIndex(b)
    rng = np.random.default_rng(seed)
    for cut in sorted(set(rng.integers(1, len(b) + 1, size=60).tolist()) | {1, 2, len(b)}):
        assert ix.pools_at(cut) == find_pools(b, cut), cut


def test_pool_index_on_one_minute_bars_over_several_days():
    b = random_walk(n=9000, seed=21, start="2026-09-13 18:00").asfreq("min").dropna()
    ix = PoolIndex(b)
    for cut in (500, 2000, 4321, 7000, len(b)):
        assert ix.pools_at(cut) == find_pools(b, cut), cut


# --- find_fvgs: O(log n) state lookups against the original per-gap scans ---------------------

from quant.liquidity import FVG, _first, _ohlc, find_fvgs  # noqa: E402


def _track_scan(side, lo, hi, i, o, h, l, c, max_age):
    """The original O(bars) tracker, kept as the reference."""
    n, s, ce = len(c), i + 1, (lo + hi) / 2
    end = min(n, i + 1 + max_age)
    if side == "bull":
        mit, ceb, thru = _first(l[s:] <= hi, s), _first(l[s:] <= ce, s), _first(l[s:] <= lo, s)
        inv = _first((c[s:end] < lo) & (c[s:end] < o[s:end]), s)
        kill = _first(c[inv + 1:] > hi, inv + 1) if inv is not None else None
    else:
        mit, ceb, thru = _first(h[s:] >= lo, s), _first(h[s:] >= ce, s), _first(h[s:] >= hi, s)
        inv = _first((c[s:end] > hi) & (c[s:end] > o[s:end]), s)
        kill = _first(c[inv + 1:] < lo, inv + 1) if inv is not None else None
    state = ("inverted" if inv is not None else "filled" if thru is not None else
             "ce" if ceb is not None else "mitigated" if mit is not None else "open")
    return FVG(side, lo, hi, i, state, mit, ceb, thru, inv, kill)


def _find_fvgs_scan(bars, cut, min_gap=1.0, max_age=96):
    o, h, l, c = _ohlc(bars, cut)
    out = [_track_scan("bull", float(h[i - 2]), float(l[i]), int(i), o, h, l, c, max_age)
           for i in np.flatnonzero(l[2:] - h[:-2] >= min_gap) + 2]
    out += [_track_scan("bear", float(h[i]), float(l[i - 2]), int(i), o, h, l, c, max_age)
            for i in np.flatnonzero(l[:-2] - h[2:] >= min_gap) + 2]
    out.sort(key=lambda g: (g.formed, g.side))
    return out


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("max_age", [5, 96, 10 ** 6])
def test_find_fvgs_equals_the_scan_reference(seed, max_age):
    b = random_walk(n=2500, seed=seed)
    for cut in (3, 400, 1777, len(b)):
        for gap in (0.25, 1.0, 3.0):
            assert find_fvgs(b, cut, gap, max_age) == _find_fvgs_scan(b, cut, gap, max_age)
