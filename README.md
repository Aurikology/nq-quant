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

## Run it

```bash
pip install -r quant/requirements.txt
python -m quant fetch --interval 5m      # and --interval 1m for fills
python -m quant backtest                 # writes data/backtests/latest.json
python -m quant sweep                    # in-sample pick, out-of-sample report
python -m pytest quant/tests             # 52 tests; the parity tests need Node 24+
```

Pulled out of a larger private trading-journal project, where the same rule files drive a desktop app.
