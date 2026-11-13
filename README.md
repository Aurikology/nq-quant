# nq-quant

A backtester for Nasdaq-100 futures (NQ) on intraday bars. It tests whether a rule-based price-pattern detector beats random entries. Written in Python with NumPy and pandas, with a small JavaScript cross-check.

The pattern is a mechanical rule set for "inverted fair value gaps": three-bar price imbalances that price later closes through. The interesting part is not the pattern. It is the machinery that tries hard **not** to fool itself about whether the pattern works.

## What it does

| Stage | Module | Method |
|---|---|---|
| Ingest | `quant/market.py` | Pulls 1m/5m bars from Yahoo's chart endpoint or any OHLC CSV. The cache only grows, so the sample outlasts the vendor's 7- and 60-day limits |
| Feature detection | `quant/liquidity.py`, `quant/judas.py`, `quant/signals.py` | Finds imbalances, liquidity pools and session sweeps in the series without look-ahead. A test asserts that no plan reads a bar after its deadline |
| Strategy | `quant/strategy.py` | Turns the detected structure into an entry/stop/target plan before each session opens |
| Replay | `quant/backtest.py` | Replays bar by bar with a deliberately pessimistic fill model: the stop wins when a bar touches both levels, stops pay slippage, and the entry bar cannot take profit |
| Risk | `quant/risk.py` | Applies daily-loss, drawdown and position-size limits. Tested for parity with an independent JavaScript implementation (`core/risk.js`) via `quant/bridge.mjs` |
| Statistics | `quant/metrics.py` | Net P&L, profit factor, daily Sharpe, max drawdown, a **bootstrap 95% CI** on mean R, and a **permutation test** against random entries |
| Model selection | `python -m quant sweep` | Grid search, with parameters chosen **in-sample** and reported **out-of-sample** (70/30 split by date) |
| Live-rule ablation | `python -m quant judas` | Replays a discretionary trader's written entry rules as five switchable filters, then turns each off in turn to measure what it does. CIs by trade and by **session block bootstrap** (trades in one session share a tape) |
| Discretionary benchmark | `python -m quant journal` | Scores a trader's 312 hand-replayed trades (FX Replay export) with the same bootstrap and session-block CIs. Sessions are recovered from interleaved rows by **balance-chain linking**, since the export has no session column |
| Performance | `quant/liquidity.py` `RangeMax` | Equal-high/low clustering asks "did any bar between two pivots trade past this level?" thousands of times per call. A **sparse table** answers it in O(1) after an O(n log n) build, and pools are memoized across runs. Tested equal to the original scan on randomized series |

## Latest result (5m structure, 1m fills, 2026-09-15 to 2026-10-13)

| Metric | Value |
|---|---|
| Sessions replayed | 62 (13,651 bars) |
| Trades | 11 |
| Win rate | 9.1% |
| Mean R per trade | +4.71, **95% bootstrap CI [−1.19, +16.44]** |
| Profit factor | 2.94 |
| Permutation test vs. 500 random-entry runs | p = 0.036 |
| Max drawdown | $3,521 (breached the 2% trailing limit on 2026-09-22) |

**Reading it honestly:** the permutation test says the entry timing beats chance at the 5% level. The confidence interval still includes zero, though, and one large winner carries the result. Eleven trades is not enough evidence. The next step is a longer sample from exported multi-year data (`--csv`), not more parameter tuning.

## Live-rule ablation (1m bars, 2026-09-14 to 2026-11-13)

The Judas-swing replay (`quant/judas.py`) with the exit a trader uses live: half off at 2R, stop to entry, the rest under a stop trailed behind confirmed swings. "Live" adds the five entry filters a discretionary trader wrote down between 2026-10-27 and 2026-11-06: minimum 3-point stop, stop at least half the median range of the last 5 bars, market structure already pointing the trade's way, no open opposing gap of 10+ points before 2R, and cancel the order if 2R prints before the fill.

| Run (59,385 bars) | Trades | Total R | Mean R | 95% CI, session bootstrap |
|---|---|---|---|---|
| No filters | 37 | −13.08 | −0.35 | [−0.89, +0.39] |
| All live filters | 3 | −3.12 | −1.04 | too few trades |
| Live, without cancel-before-fill | 11 | −2.25 | −0.20 | [−1.04, +0.68] |
| Live, without any one of the other four | 3 | −3.12 | −1.04 | unchanged |

**Reading it honestly:** written down as code, the live rules remove 92% of setups, and no version shows an edge. The cancel-before-fill rule is the only one that changes this sample; the other four overlap. The trader's discretionary paper results come from judgement this code does not capture, such as higher-timeframe gaps and target quality, and this table measures that gap rather than assuming it away. The target is a sample of **200+ trades** from multi-year exported 1m data (`--csv`) before any of these numbers mean much.

Speed (same machine, identical output): two weeks of 1m bars went from 1.6 s to 0.3 s per replay with the sparse table, and the seven-run ablation from 25.3 s to 7.2 s with pool memoization. The full two-month ablation runs in about 2 minutes.

## Discretionary benchmark: 312 hand-replayed trades (NQ, 2019 to 2026)

The same strategy traded by hand on bar replay (FX Replay, `CME_MINI:NQ1`), 14 sessions covering stretches of 2019 to 2026. One session was dropped before scoring because it practised a different strategy (top/bottom-tick entries); none was dropped for its results. Only these summary statistics are published, not the trade rows.

| Metric | Value |
|---|---|
| Trades / sessions | 312 / 14 |
| Wins / losses / breakevens | 106 / 27 / 179 |
| Win rate, all trades | 34.0% (79.7% if breakevens are left out, as the replay tool reports it) |
| Mean R per trade | **+0.42**, 95% CI **[+0.32, +0.53]** by trade, **[+0.35, +0.58]** by session block |
| Mean R without the 3 largest winners | +0.39 |
| Worst losing streak | 2 |

**Reading it honestly:** unlike the coded rules, the hand-traded version has a confidence interval clear of zero, and it does not rest on a few outliers. Two caveats keep this a benchmark, not proof: bar replay is still traded by a person who may remember how a period went, and the trader chose which stretches to replay. More than half the trades are breakevens, so most of the edge is in cutting risk early, not in the hit rate. The gap between +0.42R by hand and no edge in code is the measurable target for the next rules to encode (higher-timeframe gaps, target quality).

## Run it

```bash
pip install -r quant/requirements.txt
python -m quant fetch --interval 5m      # and --interval 1m for fills
python -m quant backtest                 # writes data/backtests/latest.json
python -m quant sweep                    # in-sample pick, out-of-sample report
python -m quant judas                    # live-rule ablation on 1m bars; writes judas-latest.json
python -m quant journal --csv <export> --exclude-final <balance>   # score hand-replayed trades
python -m pytest quant/tests             # 104 tests; the parity tests need Node 24+
```

Pulled out of a larger private trading-journal project, where the same rule files drive a desktop app.
