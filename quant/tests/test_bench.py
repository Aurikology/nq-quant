"""quant/bench.py runs and its equality assertions hold on a random walk."""

from quant import bench
from quant.tests.test_liquidity import random_walk


def test_bench_runs_and_reports_a_speedup_per_row():
    rows = bench.run(random_walk(n=1500, seed=5), cuts=5)
    assert len(rows) == 2 and all(r["speedup"] > 0 for r in rows)
