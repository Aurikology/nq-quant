"""Chunked replay (bounded memory) gives the same trades as the whole-series replay."""

from dataclasses import asdict

import pytest

from quant import judas
from quant.market import cache_path


@pytest.mark.skipif(not cache_path("NQ=F", "1m").exists(), reason="needs the 1m cache")
@pytest.mark.parametrize("params", [judas.Params(htf="both"), judas.Params.live()])
def test_chunked_replay_equals_whole_series(params):
    b = judas.load("1m")
    whole, _ = judas.run(b, params, chunk_days=None)
    judas._INDEX_CACHE.clear()
    chunked, _ = judas.run(b, params, chunk_days=12)  # many chunks over the two-month cache
    assert [asdict(t) for t in chunked] == [asdict(t) for t in whole]
