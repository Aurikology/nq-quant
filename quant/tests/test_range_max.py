"""RangeMax (sparse table) and the faster `_clusters` that uses it give the same answers as a scan."""

import numpy as np
import pytest

from quant.liquidity import RangeMax, _clusters, swing_pivots
from quant.tests.test_liquidity import random_walk


def _clusters_scan(piv, px, eq_tol, high):
    """The original O(pivots x clusters x bars) version, kept as the reference."""
    v = (1.0 if high else -1.0) * px
    clusters, alive = [], []
    for j in piv:
        p, best, still = v[j], None, []
        for cl in alive:
            lvl, low = v[cl].max(), v[cl].min()
            between = v[cl[-1] + 1:j]
            peak = between.max() if between.size else -np.inf
            fits = max(lvl, p) - min(low, p) <= eq_tol and peak <= max(lvl, p)
            if peak > lvl and not fits:
                continue
            still.append(cl)
            if fits:
                best = cl
        alive = still
        if best is not None:
            best.append(int(j))
        else:
            alive.append([int(j)])
            clusters.append(alive[-1])
    return [cl for cl in clusters if len(cl) >= 2]


@pytest.mark.parametrize("seed", range(20))
def test_range_max_matches_a_scan_on_every_range(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=int(rng.integers(1, 200)))
    t = RangeMax(v)
    for _ in range(300):
        a, b = sorted(rng.integers(0, len(v) + 1, size=2))
        want = v[a:b].max() if b > a else -np.inf
        assert t.query(a, b) == want


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("high", [True, False])
def test_fast_clusters_equal_the_scan_reference(seed, high):
    b = random_walk(n=1500, seed=seed)
    ph, pl = swing_pivots(b, len(b))
    piv, px = (ph, b["high"].to_numpy()) if high else (pl, b["low"].to_numpy())
    for tol in (0.5, 2.0, 5.0):
        assert _clusters(piv, px, tol, high) == _clusters_scan(piv, px, tol, high)
