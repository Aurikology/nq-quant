"""quant/stats.py: PSR, the expected maximum Sharpe under the null, and the deflated Sharpe ratio."""

import numpy as np
import pytest

from quant.stats import deflated_sharpe, expected_max_sharpe, psr, sharpe


def test_psr_is_one_half_at_the_observed_sharpe():
    r = np.random.default_rng(1).normal(0.2, 1, 500)
    assert psr(r, sharpe(r)) == pytest.approx(0.5)


def test_psr_grows_with_sample_length():
    rng = np.random.default_rng(2)
    short, long = rng.normal(0.1, 1, 50), rng.normal(0.1, 1, 5000)
    long = (long - long.mean()) / long.std(ddof=1) * short.std(ddof=1) + short.mean()  # same Sharpe
    assert psr(long) > psr(short)


def test_expected_max_sharpe_rises_with_trials_and_is_zero_for_one():
    assert expected_max_sharpe(1, 0.04) == 0.0
    vals = [expected_max_sharpe(n, 0.04) for n in (2, 10, 100, 1000)]
    assert vals == sorted(vals) and vals[0] > 0


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(3)
    n, sd = 50, 0.2
    sim = rng.normal(0, sd, (20000, n)).max(axis=1).mean()
    assert expected_max_sharpe(n, sd ** 2) == pytest.approx(sim, rel=0.03)


def test_deflation_lowers_the_probability_when_many_variants_were_tried():
    rng = np.random.default_rng(4)
    best = rng.normal(0.15, 1, 300)
    few = deflated_sharpe(best, [0.0, 0.05])
    many = deflated_sharpe(best, list(rng.normal(0, 0.1, 200)))
    assert many["dsr"] < few["psr_vs_0"]
    assert many["trials"] == 200 and many["sr0"] > 0
