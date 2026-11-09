/**
 * The money logic. Priority 1 of `CLAUDE.md`: capital survives. Every number an entry call
 * depends on is computed here and nowhere else.
 *
 * Pure — no I/O, no clock of its own. `now` is always passed in.
 *
 * @typedef {import("./types.js").Config} Config
 * @typedef {import("./types.js").EntryVerdict} EntryVerdict
 * @typedef {import("./types.js").Event} Event
 * @typedef {import("./types.js").Instrument} Instrument
 * @typedef {import("./types.js").Plan} Plan
 * @typedef {import("./types.js").Rules} Rules
 * @typedef {import("./types.js").Side} Side
 * @typedef {import("./types.js").Trade} Trade
 * @typedef {import("./types.js").ViolationCode} ViolationCode
 */

import { POINT_VALUE } from "./types.js";
import { stopPoints } from "./plan.js";
import { windowFor } from "./clock.js";
import { boxTouched } from "./events.js";

export { stopPoints };

/**
 * A candidate entry, as the app or a skill offers it to the gate. Every field is optional:
 * the gate checks what it is given and skips the rest.
 * @typedef {Object} Candidate
 * @property {number} [contracts]
 * @property {Side} [side]
 * @property {number} [entryPrice]
 * @property {number} [stopPrice]
 * @property {string} [entryTime]   ISO datetime
 */

/**
 * @param {unknown} v
 * @returns {boolean}
 */
function isNum(v) {
  return typeof v === "number" && Number.isFinite(v);
}

/**
 * A trade is closed once it has an outcome that is not "open".
 * @param {Trade} t
 * @returns {boolean}
 */
function isClosed(t) {
  return Boolean(t) && t.outcome !== "open";
}

/**
 * Sort key for a closed trade: when it left the market, falling back to when it entered.
 * @param {Trade} t
 * @returns {number}
 */
function closedAt(t) {
  const exit = t.exitTime ? Date.parse(t.exitTime) : NaN;
  if (Number.isFinite(exit)) return exit;
  const entry = t.entryTime ? Date.parse(t.entryTime) : NaN;
  return Number.isFinite(entry) ? entry : 0;
}

/**
 * Dollars per one full point, one contract.
 * @param {Instrument} instrument
 * @returns {number}
 */
export function pointValue(instrument) {
  const v = /** @type {Record<string, number>} */ (POINT_VALUE)[instrument];
  if (!isNum(v)) throw new TypeError(`pointValue: "${String(instrument)}" is not NQ or MNQ`);
  return v;
}

/**
 * Size from a dollar risk and a stop distance. Floors, caps at the firm's contract limit,
 * never goes below zero. A stop of zero points sizes to zero rather than to infinity.
 *
 * @param {Object} args
 * @param {number} args.riskDollars
 * @param {number} args.stopPoints
 * @param {Instrument} args.instrument
 * @param {Rules} args.rules
 * @returns {number}
 */
export function contractsForStop({ riskDollars, stopPoints: pts, instrument, rules }) {
  if (!isNum(riskDollars) || riskDollars <= 0) return 0;
  if (!isNum(pts) || pts <= 0) return 0;
  const perContract = pts * pointValue(instrument);
  if (perContract <= 0) return 0;
  const raw = Math.floor(riskDollars / perContract);
  const cap = rules?.maxContracts?.[instrument];
  const capped = isNum(cap) ? Math.min(raw, /** @type {number} */ (cap)) : raw;
  return Math.max(0, capped);
}

/**
 * The largest size this plan may be taken in right now: the tightest of the plan's own dollar
 * risk, the firm's per-trade percentage cap and what is left of today's loss limit, divided by
 * the stop, then capped at the firm's contract limit and the plan's own contract limit.
 *
 * Including `plan.risk.maxContracts` in the cap is a deliberate tightening: the plan is a
 * promise he made before the session and the gate should not hand back a number larger than it.
 *
 * @param {Object} args
 * @param {Plan} args.plan
 * @param {Rules} args.rules
 * @param {number} [args.remainingDailyLoss]  defaults to the full daily loss limit
 * @returns {number}
 */
export function maxContracts({ plan, rules, remainingDailyLoss: remaining }) {
  if (!plan || !plan.risk || !rules) return 0;
  const left = isNum(remaining) ? /** @type {number} */ (remaining) : rules.dailyLossLimit;
  const pctCap = (rules.accountSize * rules.maxRiskPerTradePct) / 100;
  const budget = Math.min(plan.risk.maxDollars, pctCap, left);
  const n = contractsForStop({
    riskDollars: budget,
    stopPoints: stopPoints(plan),
    instrument: plan.risk.instrument,
    rules,
  });
  const planCap = isNum(plan.risk.maxContracts) ? plan.risk.maxContracts : Infinity;
  return Math.max(0, Math.min(n, planCap));
}

/**
 * Net P&L of the closed trades on one ET date.
 * @param {Trade[]} trades
 * @param {string} date   "YYYY-MM-DD" in ET
 * @returns {number}
 */
export function dailyPnl(trades, date) {
  let sum = 0;
  for (const t of trades || []) {
    if (t.date !== date || !isClosed(t)) continue;
    if (isNum(t.pnl)) sum += t.pnl;
  }
  return sum;
}

/**
 * Dollars still available against the daily loss limit. A green day does not buy extra room:
 * the limit is the limit, so only a negative day eats into it.
 * @param {Trade[]} trades
 * @param {Rules} rules
 * @param {string} date
 * @returns {number}
 */
export function remainingDailyLoss(trades, rules, date) {
  const pnl = dailyPnl(trades, date);
  return Math.max(0, rules.dailyLossLimit + Math.min(0, pnl));
}

/**
 * @typedef {Object} Drawdown
 * @property {number} equity    accountSize + cumulative closed P&L
 * @property {number} peak      highest equity the drawdown type lets count
 * @property {number} floor     the number the account dies at
 * @property {number} remaining equity - floor
 */

/**
 * Trailing drawdown, walked chronologically over the closed trades.
 *
 * `trailing_intraday` — the peak moves after every trade.
 * `trailing_eod`      — the peak moves only at the close of an ET date, using the equity at
 *                       that date's last trade.
 * `static`            — the peak never moves off the starting balance.
 *
 * The floor rises with the peak but stops rising once it reaches the starting balance, which is
 * the Apex/Topstep shape; it can never sit below `accountSize - maxDrawdown`.
 *
 * @param {Trade[]} trades
 * @param {Rules} rules
 * @returns {Drawdown}
 */
export function trailingDrawdown(trades, rules) {
  const accountSize = rules.accountSize;
  const closed = (trades || []).filter(isClosed).sort((a, b) => closedAt(a) - closedAt(b));

  let equity = accountSize;
  let peak = accountSize;
  const type = rules.drawdownType;

  if (type === "static") {
    for (const t of closed) equity += isNum(t.pnl) ? t.pnl : 0;
  } else if (type === "trailing_intraday") {
    for (const t of closed) {
      equity += isNum(t.pnl) ? t.pnl : 0;
      if (equity > peak) peak = equity;
    }
  } else {
    // trailing_eod: hold the peak until an ET date is finished, then mark it once.
    for (let i = 0; i < closed.length; i++) {
      equity += isNum(closed[i].pnl) ? closed[i].pnl : 0;
      const lastOfDate = i === closed.length - 1 || closed[i + 1].date !== closed[i].date;
      if (lastOfDate && equity > peak) peak = equity;
    }
  }

  const floorRaw = Math.min(peak - rules.maxDrawdown, accountSize);
  const floor = Math.max(floorRaw, accountSize - rules.maxDrawdown);
  return { equity, peak, floor, remaining: equity - floor };
}

/**
 * Is this candidate a revenge trade? True when it enters inside `rules.revengeWindowMin`
 * minutes of the exit of the most recent losing trade. The boundary minute counts.
 *
 * @param {Object} args
 * @param {Candidate} args.candidate
 * @param {Trade[]} args.trades
 * @param {Rules} args.rules
 * @returns {boolean}
 */
export function isRevenge({ candidate, trades, rules }) {
  if (!candidate || !candidate.entryTime) return false;
  const entry = Date.parse(candidate.entryTime);
  if (!Number.isFinite(entry)) return false;

  const losses = (trades || [])
    .filter((t) => isClosed(t) && (t.outcome === "loss" || (isNum(t.pnl) && t.pnl < 0)))
    .sort((a, b) => closedAt(a) - closedAt(b));
  const last = losses[losses.length - 1];
  if (!last) return false;

  const exit = closedAt(last);
  if (!exit) return false;
  const gapMin = (entry - exit) / 60000;
  return gapMin >= 0 && gapMin <= rules.revengeWindowMin;
}

/**
 * Is a price inside the box? Exported so the app never re-implements the comparison in the
 * renderer — `Trade.insideBox` and the gate's `ENTRY_OUTSIDE_BOX` are the same test.
 * @param {number} price
 * @param {{ low: number, high: number }} box
 * @returns {boolean}
 */
export function insideBox(price, box) {
  if (!isNum(price) || !box || !isNum(box.low) || !isNum(box.high)) return false;
  return price >= box.low && price <= box.high;
}

/**
 * @typedef {Object} TradeMetrics
 * @property {number} pnl           dollars, (exit - entry) signed by side, times size
 * @property {number} riskDollars   dollars at risk between entry and stop, times size
 * @property {number} r             pnl / riskDollars, 0 when there was nothing at risk
 * @property {"win" | "loss" | "breakeven" | "open"} outcome
 * @property {number} stopPoints    points between entry and stop
 * @property {number} pointValue    dollars per point per contract
 */

/**
 * The four numbers a logged fill is made of. Pure, and the only place they are computed: the
 * Trade screen sends prices and gets these back, so the renderer never does arithmetic on money.
 *
 * A trade with no exit price is `open`: its P&L is 0 and its R is 0, because an unrealised
 * number is not evidence of anything.
 *
 * @param {Object} args
 * @param {Side} args.side
 * @param {number} args.contracts
 * @param {number} args.entryPrice
 * @param {number} args.stopPrice
 * @param {number | null} [args.exitPrice]
 * @param {Instrument} args.instrument
 * @returns {TradeMetrics}
 */
export function tradeMetrics({ side, contracts, entryPrice, stopPrice, exitPrice = null, instrument }) {
  const pv = pointValue(instrument);
  const size = isNum(contracts) && contracts > 0 ? contracts : 0;
  const dir = side === "short" ? -1 : 1;

  const stopPts = isNum(entryPrice) && isNum(stopPrice) ? Math.abs(entryPrice - stopPrice) : 0;
  const riskDollars = stopPts * pv * size;

  const open = !isNum(exitPrice);
  const pnl = open || !isNum(entryPrice)
    ? 0
    : (/** @type {number} */ (exitPrice) - /** @type {number} */ (entryPrice)) * dir * pv * size;

  const r = riskDollars > 0 ? pnl / riskDollars : 0;
  /** @type {"win" | "loss" | "breakeven" | "open"} */
  const outcome = open ? "open" : pnl > 0 ? "win" : pnl < 0 ? "loss" : "breakeven";

  return {
    pnl: Math.round(pnl * 100) / 100,
    riskDollars: Math.round(riskDollars * 100) / 100,
    r: Math.round(r * 100) / 100,
    outcome,
    stopPoints: stopPts,
    pointValue: pv,
  };
}

/**
 * The entry gate. Hard blocks stop the call; flags are logged and counted but do not block.
 * Everything it needs is handed in, so the same call can be replayed for any instant.
 *
 * @param {Object} args
 * @param {Plan | null | undefined} args.plan
 * @param {Rules} args.rules
 * @param {Config} args.config
 * @param {Trade[]} [args.trades]
 * @param {Event[]} [args.events]
 * @param {Date} [args.now]
 * @param {Candidate} [args.candidate]
 * @returns {EntryVerdict}
 */
export function evaluateEntry({ plan, rules, config, trades = [], events = [], now = new Date(), candidate = {} }) {
  /** @type {ViolationCode[]} */
  const blocks = [];
  /** @type {ViolationCode[]} */
  const flags = [];

  // Hard rule 1: no plan file, no blessing. Nothing else is worth computing.
  if (!plan) return { allowed: false, blocks: ["NO_PLAN"], flags, maxContracts: 0 };

  const dayTrades = (trades || []).filter((t) => t.date === plan.date);
  const remaining = remainingDailyLoss(dayTrades, rules, plan.date);
  const dd = trailingDrawdown(trades || [], rules);
  const cap = maxContracts({ plan, rules, remainingDailyLoss: remaining });

  // OUTSIDE_SESSION — a disabled session is outside it too.
  let inWindow = false;
  try {
    const w = windowFor(config, plan.session, plan.date);
    const t = now.getTime();
    inWindow = t >= w.start.getTime() && t < w.end.getTime();
  } catch {
    inWindow = false;
  }
  if (!inWindow) blocks.push("OUTSIDE_SESSION");

  if (remaining <= 0) blocks.push("DAILY_LOSS_REACHED");
  if (dd.remaining <= 0) blocks.push("TRAILING_DD_REACHED");

  if (isNum(candidate.contracts) && /** @type {number} */ (candidate.contracts) > cap) {
    blocks.push("OVER_MAX_CONTRACTS");
  }

  const sessionTrades = dayTrades.filter((t) => t.session === plan.session);
  const tradeCap = Math.min(plan.maxTrades, rules.maxTradesPerSession);
  if (sessionTrades.length >= tradeCap) blocks.push("OVER_MAX_TRADES");

  // Hard rule 2: the box must be touched since the plan was filed.
  if (plan.entryOnlyInsideBox && !boxTouched({ plan, events })) blocks.push("BOX_NOT_TOUCHED");

  if (plan.entryOnlyInsideBox && isNum(candidate.entryPrice) &&
      !insideBox(/** @type {number} */ (candidate.entryPrice), plan.box)) {
    blocks.push("ENTRY_OUTSIDE_BOX");
  }

  if (isRevenge({ candidate, trades: trades || [], rules })) flags.push("REVENGE_TRADE");
  if (candidate.side && candidate.side !== plan.bias) flags.push("AGAINST_BIAS");
  if (isNum(candidate.stopPrice)) {
    const stop = /** @type {number} */ (candidate.stopPrice);
    const side = candidate.side ?? plan.bias;
    if ((side === "long" && stop < plan.invalidation) || (side === "short" && stop > plan.invalidation)) {
      flags.push("STOP_BEYOND_INVALIDATION");
    }
  }

  return { allowed: blocks.length === 0, blocks, flags, maxContracts: cap };
}
