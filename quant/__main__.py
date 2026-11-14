"""python -m quant <command>

  fetch      merge the newest Yahoo NQ=F bars into data/market/ (5m: 60 days per request)
  backtest   replay the model under data/rules.json; writes data/backtests/latest.json
  sweep      parameter grid, chosen in-sample, reported out-of-sample; writes sweep-latest.json
  scan       the automated box for the next session, validated by core/plan.js; files nothing
  levels     FVGs, IFVGs, liquidity pools, the DOL and its LRLR/HRLR grade at the latest closed bar
  metalabel  walk-forward meta-labeling of the replay's setups (purged, embargoed); writes metalabel-latest.json
  journal    score an FX Replay trade export with the same statistics; writes journal-latest.json
  judas      Judas-swing replay: baseline vs the live rule set, one filter ablated at a time; writes judas-latest.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .backtest import Costs, Market, random_control, run, session_dates, validate_plans
from .config import DATA, TICK_SIZE, enabled_sessions, load_config, load_rules, session_window
from .market import cache_path, read_csv, update_cache
from .liquidity import FVG, Pool, find_fvgs, find_pools, grade_run, pick_dol, recent_start
from .metrics import summarize
from .strategy import Params, build_plan

OUT = DATA / "backtests"


def _params(a) -> Params:
    return Params(entry=a.entry, min_rr=a.min_rr, buffer_ticks=a.buffer_ticks, min_gap=a.min_gap,
                  first_touch=not a.any_touch, bias=a.bias, instrument=a.instrument)


def _costs(a) -> Costs:
    return Costs(commission=a.commission, slip_ticks=a.slip_ticks, fill=a.fill)


def _bars(a):
    path = Path(a.csv) if a.csv else cache_path("NQ=F", a.interval)
    if not path.exists():
        sys.exit(f"no bars at {path}; run `python -m quant fetch --interval {a.interval}` or pass --csv")
    return read_csv(path), path


def _exec(a, m: Market) -> Market | None:
    """The fill timeframe: --exec-csv, else the 1m Yahoo cache, else the structure bars."""
    if a.exec == "same":
        return None
    path = Path(a.exec_csv) if a.exec_csv else (None if a.csv else cache_path("NQ=F", "1m"))
    if path is None:
        return None
    if not path.exists():
        sys.exit(f"no 1m bars at {path}; run `python -m quant fetch --interval 1m` or pass --exec same")
    em = Market(read_csv(path))
    return None if em.bar_secs >= m.bar_secs else em


def cmd_metalabel(a):
    from . import metalabel
    rep = metalabel.from_report(Path(a.report))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metalabel-latest.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(f"{rep['setups']} setups, {rep['tested']} tested out of sample; all tested: {rep['oos_all']}")
    for name, m in rep["models"].items():
        print()
        print(f"{name}: taken {m['oos_taken']}")
        for f in m["folds"]:
            print(f"  {f}")


def cmd_journal(a):
    from . import journal
    path = Path(a.csv)
    if not path.exists():
        sys.exit(f"no export at {path}; FX Replay -> Testing -> Analytics -> download")
    d = journal.load_fxreplay(path)
    rep = journal.score(d, tuple(a.exclude_final))
    rep["by_session"] = journal.sessions(d).reset_index().to_dict(orient="records")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "journal-latest.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    for k, v in rep.items():
        if k != "by_session":
            print(f"{k:22} {v}")
    print(journal.sessions(d).to_string())


def cmd_judas(a):
    import time
    from . import judas
    path = Path(a.csv) if a.csv else cache_path("NQ=F", "1m")
    if not path.exists():
        sys.exit(f"no bars at {path}; run `python -m quant fetch --interval 1m` or pass --csv")
    t0 = time.perf_counter()
    rep = judas.ablation(path=str(path), days=a.days, workers=a.workers)
    rep["workers"] = a.workers
    rep["seconds"] = round(time.perf_counter() - t0, 1)
    rep["ranAt"] = datetime.now(ZoneInfo("America/Toronto")).isoformat(timespec="seconds")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "judas-latest.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(f"{rep['bars']:,} bars {rep['from'][:10]} to {rep['to'][:10]}, exit model {rep['model']}, "
          f"{len(rep['rows'])} runs in {rep['seconds']} s")
    print(f"{'run':34} {'n':>3} {'totR':>7} {'meanR':>7} {'win':>5}  95% CI (trades)   95% CI (sessions)")
    for r in rep["rows"]:
        ci = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]" if c else "   n/a"
        mean = f"{r['mean_r']:+.3f}" if r["mean_r"] is not None else "   n/a"
        win = f"{r['win_rate']:.2f}" if r["win_rate"] is not None else "  n/a"
        print(f"{r['name']:34} {r['trades']:>3} {r['total_r']:>+7.2f} {mean:>7} {win:>5}  "
              f"{ci(r['ci95_trades']):17} {ci(r['ci95_sessions'])}")
    print(f"wrote {OUT / 'judas-latest.json'}")


def cmd_fetch(a):
    path, before, after = update_cache("NQ=F", a.interval)
    bars = read_csv(path)
    print(f"{path.relative_to(DATA.parent)}: {before} -> {after} bars, {bars.index[0]} to {bars.index[-1]}")


def cmd_backtest(a):
    rules, config = load_rules(), load_config()
    bars, path = _bars(a)
    m = Market(bars)
    em = _exec(a, m)
    res = run(m, rules, config, _params(a), _costs(a), exec_m=em, stop_at_breach=a.stop_at_breach)
    summary = summarize(res, rules)
    control = random_control(em or m, res, rules, config, n=a.control, costs=_costs(a)) if a.control else None
    report = {
        "schema": 1,
        "ranAt": datetime.now(ZoneInfo(config.get("timezone", "America/Toronto"))).isoformat(timespec="seconds"),
        "data": {"source": str(path.relative_to(DATA.parent)) if path.is_relative_to(DATA.parent) else str(path),
                 "timeframe": m.timeframe, "execTimeframe": (em or m).timeframe, "bars": len(bars),
                 "from": bars.index[0].isoformat(), "to": bars.index[-1].isoformat(),
                 # The sessions actually replayed: with 1m fills, only the days the 1m cache covers.
                 "sessionsFrom": res.dates[0] if res.dates else None, "sessionsTo": res.dates[-1] if res.dates else None},
        "rules": rules, "params": asdict(res.params), "costs": asdict(_costs(a)),
        "summary": summary, "control": control,
        "trades": res.trades, "sessions": res.sessions,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(report)


def _print_summary(rep):
    s, d = rep["summary"], rep["data"]
    print(f"{d['timeframe']} NQ boxes, {d['execTimeframe']} fills, sessions {d['sessionsFrom']} to {d['sessionsTo']}, {s['params']}")
    print(f"sessions {s['sessions']} over {s['days']} days -> trades {s['trades']}")
    for k, v in s["funnel"].items():
        print(f"  {v:4d}  {k}")
    if s["trades"]:
        print(f"win {s['winRate']:.0%}  avgR {s['avgR']:+.2f} (95% {s['avgR95'][0]:+.2f}..{s['avgR95'][1]:+.2f})  "
              f"net ${s['netPnl']:,.0f}  PF {s['profitFactor']}  maxDD ${s['maxDrawdown']:,.0f}  "
              f"Sharpe {s['sharpeDaily']}  fees ${s['fees']:,.0f}")
        print(f"exits {s['exits']}  worst streak {s['worstLosingStreak']}  "
              f"consistency {s['consistencyPct']}% (limit {s['consistencyLimit']}%)  breach {s['trailingDdBreach']}")
    c = rep.get("control")
    if c and c.get("n"):
        print(f"random-entry control x{c['n']}: mean R {c['randomMeanR']:+.3f} "
              f"(95% {c['random95'][0]:+.3f}..{c['random95'][1]:+.3f}), strategy {c['strategyMeanR']:+.3f}, p={c['pValue']}")
    print("wrote data/backtests/latest.json")


def _lo(summary: dict) -> float:
    return summary["avgR95"][0] if summary.get("avgR95") else -9.0


def _rank(row: dict) -> float:
    return _lo(row["is"])


def cmd_sweep(a):
    rules, config = load_rules(), load_config()
    bars, _ = _bars(a)
    m = Market(bars)
    em = _exec(a, m)
    days = [d for d in session_dates(em or m) if d.weekday() < 5 or d.weekday() == 6]
    split = days[int(len(days) * a.split)]
    is_days, oos_days = [d for d in days if d < split], [d for d in days if d >= split]

    base = _params(a)
    grid = itertools.product(["discount", "momentum"], ["ce", "proximal"], [1.5, 2.0, 3.0], [True, False], [1.0, 4.0])
    rows = []
    for bias, entry, rr, first, gap in grid:
        p = replace(base, bias=bias, entry=entry, min_rr=rr, first_touch=first, min_gap=gap)
        ins = summarize(run(m, rules, config, p, _costs(a), dates=is_days, exec_m=em), rules)
        rows.append({"params": p.label(), "p": p, "is": ins})
    eligible = [r for r in rows if r["is"]["trades"] >= a.min_trades]
    if not eligible:
        sys.exit(f"no combination reached {a.min_trades} in-sample trades; lower --min-trades or fetch more data")
    # Rank by the bootstrap lower bound of mean R, not the mean: on a few dozen trades the mean is
    # set by one +60R outlier, and picking on it selects lottery tickets.
    eligible.sort(key=_rank, reverse=True)
    pick = eligible[0]
    for r in rows:
        r["oos"] = summarize(run(m, rules, config, r["p"], _costs(a), dates=oos_days, exec_m=em), rules)

    print(f"in-sample {is_days[0]} to {is_days[-1]}, out-of-sample {oos_days[0]} to {oos_days[-1]}")
    print(f"{'params':46s} {'IS n':>5s} {'IS avgR':>8s} {'IS lo95':>8s} {'OOS n':>6s} {'OOS avgR':>9s}")
    for r in sorted(rows, key=_rank, reverse=True):
        mark = "  <- picked in-sample" if r is pick else ""
        print(f"{r['params']:46s} {r['is']['trades']:5d} {r['is'].get('avgR', 0):+8.2f} {_lo(r['is']):+8.2f} "
              f"{r['oos']['trades']:6d} {r['oos'].get('avgR', 0):+9.2f}{mark}")
    out = {"split": split.isoformat(), "minTrades": a.min_trades, "picked": pick["params"],
           "rows": [{"params": r["params"], "inSample": r["is"], "outOfSample": r["oos"]} for r in rows]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep-latest.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote data/backtests/sweep-latest.json")


def cmd_scan(a):
    rules, config = load_rules(), load_config()
    if not a.no_fetch:
        update_cache("NQ=F", a.interval)
    bars, _ = _bars(a)
    m = Market(bars)
    tz = ZoneInfo(config.get("timezone", "America/Toronto"))
    now = datetime.fromisoformat(a.at).replace(tzinfo=tz) if a.at else datetime.now(tz)
    if a.at:
        m = Market(bars[bars.index < now])  # replay: the scanner as it would have run at --at
    sessions = [a.session] if a.session else enabled_sessions(config, rules)
    upcoming = sorted(
        (w for d in (now.date(), now.date() + timedelta(days=1)) for s in sessions
         if (w := session_window(config, s, d)).end > now),
        key=lambda w: w.start)
    if not upcoming:
        sys.exit("no upcoming session")
    w = upcoming[0]
    as_of = replace(w, plan_deadline=min(w.plan_deadline, now))
    st = build_plan(m.bars, m.tday, m.bar_secs, m.ifvgs(a.min_gap), as_of, rules, _params(a), m.timeframe)
    print(f"{w.session} {w.date} opens {w.start:%H:%M} ET; bars through {m.t[-1]:%Y-%m-%d %H:%M} ET")
    if st.plan is None:
        print(f"no box: {st.reason}. A session with no trade is a good session.")
        return
    verdict = validate_plans([st.plan], rules)[0]
    print(json.dumps(st.plan, indent=2))
    print(f"entry {st.entry:.2f}  stop {st.stop:.2f}  target {st.target:.2f}  ({st.extras['rr']:.1f}R)")
    print("core/plan.js: " + ("valid" if verdict["ok"] else "; ".join(verdict["errors"])))
    print("This is a draft. Hard rule 1 still applies: he files the plan with /pre-session; nothing here did.")


def _pool_json(p: Pool | None, t, price: float) -> dict | None:
    if p is None:
        return None
    return {"kind": p.kind, "side": p.side, "price": p.price, "formed": t[p.formed].isoformat(),
            "swept": t[p.swept].isoformat() if p.swept is not None else None,
            "distance": round(p.price - price, 2)}


def _fvg_json(g: FVG, t, price: float) -> dict:
    return {"type": "BISI" if g.side == "bull" else "SIBI", "low": g.low, "high": g.high, "ce": g.ce,
            "state": g.state, "formed": t[g.formed].isoformat(), "ifvgSide": g.ifvg_side,
            "inverted": t[g.inverted].isoformat() if g.inverted is not None else None,
            "distance": round(_gap_dist(g, price), 2)}


def _gap_dist(g: FVG, price: float) -> float:
    """Signed points from price to the gap: 0 inside, + above, - below."""
    return g.low - price if g.low > price else g.high - price if g.high < price else 0.0


def cmd_levels(a):
    """Read-only: prints what the detectors see at the latest closed bar. Files nothing."""
    bars, _ = _bars(a)
    cut, t = len(bars), bars.index
    price = float(bars["close"].iat[-1])
    fvgs = find_fvgs(bars, cut, a.min_gap)
    pools = find_pools(bars, cut)
    recent = recent_start(bars, cut)
    near = lambda gs: sorted(gs, key=lambda g: abs(_gap_dist(g, price)))[:a.near]
    open_fvgs = near([g for g in fvgs if g.state in ("open", "mitigated", "ce") and g.formed >= recent])
    live = near([g for g in fvgs if g.live_ifvg and g.formed >= recent])
    dol = pick_dol(bars, cut, a.bias, pools=pools)
    grade = grade_run(bars, cut, dol.pool.price, dol.bias, fvgs=fvgs) if dol.pool else None
    out = {
        "bar": t[-1].isoformat(), "interval": a.interval, "close": price,
        "fvgs": [_fvg_json(g, t, price) for g in open_fvgs],
        "ifvgs": [_fvg_json(g, t, price) for g in live],
        "pools": [_pool_json(p, t, price) for p in sorted(pools, key=lambda p: -p.price)],
        "dol": {"bias": dol.bias, "biasSource": dol.bias_source, "reason": dol.reason,
                "pool": _pool_json(dol.pool, t, price), "distance": dol.distance,
                "nextTier": _pool_json(dol.next_tier, t, price)},
        "run": None if grade is None else {
            "grade": grade.grade, "direction": grade.direction, "start": grade.start, "target": grade.target,
            "dailyBias": grade.daily_bias, "weeklyBias": grade.weekly_bias, "aligned": grade.aligned,
            "obstacles": [_fvg_json(g, t, price) for g in grade.obstacles], "reasons": grade.reasons},
    }
    if a.json:
        print(json.dumps(out, indent=2))
        return
    print(f"NQ {a.interval}, last closed bar {t[-1]:%Y-%m-%d %H:%M} ET, close {price:.2f}")
    print(f"open FVGs near price (nearest {a.near}, since {t[recent]:%Y-%m-%d %H:%M}):")
    for g in out["fvgs"]:
        print(f"  {g['type']} {g['low']:.2f}-{g['high']:.2f} CE {g['ce']:.2f} {g['state']:9s} {g['distance']:+8.2f}  {g['formed'][:16]}")
    print("live IFVGs near price:")
    for g in out["ifvgs"]:
        print(f"  {g['ifvgSide']:5s} {g['low']:.2f}-{g['high']:.2f} CE {g['ce']:.2f} {g['distance']:+8.2f}  inverted {g['inverted'][:16]}")
    print("pools (significant, plus unswept swings), highest first:")
    for p, pj in zip(sorted(pools, key=lambda p: -p.price), out["pools"]):
        if p.significant or p.swept is None:
            sw = f"swept {pj['swept'][:16]}" if pj["swept"] else "UNSWEPT"
            print(f"  {p.kind:9s} {p.side} {p.price:10.2f} {pj['distance']:+8.2f}  formed {pj['formed'][:16]}  {sw}")
    print(f"bias {dol.bias} ({dol.bias_source})")
    if dol.pool is None:
        print(f"DOL: none, {dol.reason}")
        return
    nt = dol.next_tier
    print(f"DOL: {dol.pool.kind} {dol.pool.price:.2f}, {dol.distance:.2f} pts ({dol.distance / TICK_SIZE:.0f} ticks); "
          f"next tier {f'{nt.kind} {nt.price:.2f}' if nt else 'none'}  [{dol.reason}]")
    print(f"run: {grade.grade}")
    for r in grade.reasons:
        print(f"  {r}")
    print("Read-only: nothing filed, nothing placed.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m quant", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--csv", help="bars CSV (time,open,high,low,close[,volume]); default the Yahoo cache")
        p.add_argument("--interval", default="5m")
        p.add_argument("--instrument", default="NQ", choices=["NQ", "MNQ"])
        p.add_argument("--entry", default="ce", choices=["ce", "proximal"])
        p.add_argument("--min-rr", type=float, default=2.0)
        p.add_argument("--buffer-ticks", type=int, default=2)
        p.add_argument("--min-gap", type=float, default=1.0)
        p.add_argument("--any-touch", action="store_true", help="allow boxes already retested")
        p.add_argument("--bias", default="discount", choices=["discount", "momentum"])
        p.add_argument("--commission", type=float, help="round turn per contract (default NQ 4.50, MNQ 1.50)")
        p.add_argument("--slip-ticks", type=int, default=1)
        p.add_argument("--fill", default="through", choices=["through", "touch"])
        p.add_argument("--exec", default="1m", choices=["1m", "same"], help="fill on 1m bars (default) or on the box bars")
        p.add_argument("--exec-csv", help="fill-timeframe bars CSV, finer than --csv")

    f = sub.add_parser("fetch")
    f.add_argument("--interval", default="5m")
    b = sub.add_parser("backtest"); common(b)
    b.add_argument("--control", type=int, default=500, help="random-entry permutation runs (0 to skip)")
    b.add_argument("--stop-at-breach", action="store_true", help="stop trading once the trailing drawdown is hit")
    s = sub.add_parser("sweep"); common(s)
    s.add_argument("--split", type=float, default=0.7)
    s.add_argument("--min-trades", type=int, default=8)
    sc = sub.add_parser("scan"); common(sc)
    sc.add_argument("--session", choices=["asia", "ny_am", "ny_pm"])
    sc.add_argument("--no-fetch", action="store_true")
    sc.add_argument("--at", help='replay at an ET time, e.g. "2026-10-08 19:45"')

    lv = sub.add_parser("levels")
    lv.add_argument("--csv", help="bars CSV; default the Yahoo cache")
    lv.add_argument("--interval", default="5m")
    lv.add_argument("--bias", choices=["long", "short"], help="default: premium/discount of the previous day")
    lv.add_argument("--min-gap", type=float, default=1.0)
    lv.add_argument("--near", type=int, default=6, help="how many FVGs and IFVGs to list")
    lv.add_argument("--json", action="store_true", help="machine-readable output to stdout")

    jd = sub.add_parser("judas")
    jd.add_argument("--csv", help="1m bars CSV; default the Yahoo 1m cache")
    jd.add_argument("--days", type=int, help="only the last N calendar days")
    jd.add_argument("--workers", type=int, default=1, help="run the ablation configurations in parallel")

    ml = sub.add_parser("metalabel")
    ml.add_argument("--report", default=str(OUT / "judas-latest.json"), help="a judas report with baseline_trades")

    jn = sub.add_parser("journal")
    jn.add_argument("--csv", default=str(DATA / "fxreplay" / "trades.csv"), help="FX Replay analytics export")
    jn.add_argument("--exclude-final", type=float, nargs="*", default=[],
                    help="final balances of sessions that ran a different strategy")

    a = ap.parse_args(argv)
    {"fetch": cmd_fetch, "backtest": cmd_backtest, "sweep": cmd_sweep, "scan": cmd_scan,
     "levels": cmd_levels, "judas": cmd_judas, "journal": cmd_journal, "metalabel": cmd_metalabel}[a.cmd](a)


if __name__ == "__main__":
    main()
