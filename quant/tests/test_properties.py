"""Property-based tests (Hypothesis): generated price series, shrunk to a minimal case on failure.

The project's central claim is that nothing it computes about bar `cut` depends on bars after it.
These properties check that over thousands of adversarial series, plus algebraic identities of the
range structures every detector is built on.
"""

import numpy as np
import pandas as pd
import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

from quant.judas import structure  # noqa: E402
from quant.liquidity import PoolIndex, RangeMax, find_fvgs, find_pools, swing_pivots  # noqa: E402

PRICE = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)


@st.composite
def series(draw, min_n=8, max_n=160):
    """OHLC bars on a quarter-point grid from a drawn path of closes and drawn wick lengths."""
    n = draw(st.integers(min_n, max_n))
    steps = draw(st.lists(st.integers(-24, 24), min_size=n, max_size=n))
    wicks = draw(st.lists(st.tuples(st.integers(0, 12), st.integers(0, 12)), min_size=n, max_size=n))
    c = 20000 + np.cumsum(steps) * 0.25
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + np.array([w[0] for w in wicks]) * 0.25
    lo = np.minimum(o, c) - np.array([w[1] for w in wicks]) * 0.25
    idx = pd.date_range("2026-09-14 08:00", periods=n, freq="5min", tz="America/Toronto")
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1.0}, index=idx)


def _future_scrambled(b, cut, seed):
    m = b.copy()
    rng = np.random.default_rng(seed)
    shift = rng.normal(0, 300, (len(b) - cut, 1))
    m.iloc[cut:, :4] = m.iloc[cut:, :4].to_numpy() + shift
    return m


@settings(max_examples=150, deadline=None)
@given(series(), st.data())
def test_fvgs_pivots_and_pools_ignore_the_future(b, data):
    cut = data.draw(st.integers(3, len(b)))
    m = _future_scrambled(b, cut, data.draw(st.integers(0, 10 ** 6)))
    for gap in (0.25, 1.0):
        assert find_fvgs(m, cut, gap) == find_fvgs(b, cut, gap)
    assert [a.tolist() for a in swing_pivots(m, cut)] == [a.tolist() for a in swing_pivots(b, cut)]
    assert find_pools(m, cut) == find_pools(b, cut)


@settings(max_examples=100, deadline=None)
@given(series(min_n=20, max_n=220), st.data())
def test_pool_index_equals_find_pools_and_ignores_the_future(b, data):
    cut = data.draw(st.integers(1, len(b)))
    assert PoolIndex(b).pools_at(cut) == find_pools(b, cut)
    m = _future_scrambled(b, cut, data.draw(st.integers(0, 10 ** 6)))
    assert PoolIndex(m).pools_at(cut) == PoolIndex(b).pools_at(cut)


@settings(max_examples=150, deadline=None)
@given(series(), st.data())
def test_structure_state_at_j_reads_bars_up_to_j_only(b, data):
    cut = data.draw(st.integers(1, len(b)))
    m = _future_scrambled(b, cut, data.draw(st.integers(0, 10 ** 6)))
    assert structure(m)[:cut].tolist() == structure(b)[:cut].tolist()


@settings(max_examples=300, deadline=None)
@given(st.lists(PRICE, min_size=1, max_size=300), st.data())
def test_range_max_queries_agree_with_numpy(v, data):
    a = np.array(v)
    t = RangeMax(a)
    i = data.draw(st.integers(0, len(a)))
    j = data.draw(st.integers(0, len(a)))
    x = data.draw(PRICE)
    assert t.query(i, j) == (a[i:j].max() if j > i else -np.inf)
    hits = np.flatnonzero(a[i:] > x)
    assert t.first_above(i, x) == (i + int(hits[0]) if hits.size else None)
    ge = np.flatnonzero(a[i:j] >= x) if j > i else np.array([])
    assert t.first_ge(i, x, j) == (i + int(ge[0]) if ge.size else None)
