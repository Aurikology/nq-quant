/**
 * Event parsing and the box-touch rule. Hard rule 2 of `CLAUDE.md` lives in `boxTouched`:
 * no entry call until the event log shows price inside the box since the plan was filed.
 * "It's close" is not inside.
 *
 * Pure — the receiver does the I/O, this file only reads what it wrote.
 *
 * @typedef {import("./types.js").AlertKind} AlertKind
 * @typedef {import("./types.js").Event} Event
 * @typedef {import("./types.js").Plan} Plan
 * @typedef {import("./types.js").SessionId} SessionId
 */

import { ALERTS, SESSIONS } from "./types.js";

const ALERT_SET = new Set(/** @type {readonly string[]} */ (ALERTS));
const SESSION_SET = new Set(/** @type {readonly string[]} */ (SESSIONS));

/**
 * @param {unknown} v
 * @returns {boolean}
 */
function isNum(v) {
  return typeof v === "number" && Number.isFinite(v);
}

/**
 * When an event happened. The sender's `ts` is authoritative; `receivedAt` is the fallback for a
 * manual entry with no sender clock.
 * @param {Event} e
 * @returns {number}  epoch ms, or NaN
 */
export function eventAt(e) {
  const ts = e && e.ts ? Date.parse(e.ts) : NaN;
  if (Number.isFinite(ts)) return ts;
  return e && e.receivedAt ? Date.parse(e.receivedAt) : NaN;
}

/**
 * Validate one raw event against the data contract. Throws with the offending field named;
 * it never repairs a bad payload, because a wrong price is worse than a missing one.
 *
 * @param {any} raw
 * @returns {Event}
 */
export function parseEvent(raw) {
  if (!raw || typeof raw !== "object") throw new TypeError("parseEvent: event must be an object");
  if (raw.schema !== undefined && raw.schema !== 1) throw new TypeError("parseEvent: schema must be 1");
  if (typeof raw.symbol !== "string" || raw.symbol.trim() === "") {
    throw new TypeError("parseEvent: symbol must be a non-empty string");
  }
  if (!isNum(raw.price) || raw.price <= 0) throw new TypeError("parseEvent: price must be a positive number");
  if (typeof raw.alert !== "string" || !ALERT_SET.has(raw.alert)) {
    throw new TypeError(`parseEvent: alert "${String(raw.alert)}" is not one of ${ALERTS.join(", ")}`);
  }
  if (raw.source !== "tradingview" && raw.source !== "manual") {
    throw new TypeError('parseEvent: source must be "tradingview" or "manual"');
  }

  const receivedAt = typeof raw.receivedAt === "string" ? raw.receivedAt : raw.ts;
  if (typeof receivedAt !== "string" || Number.isNaN(Date.parse(receivedAt))) {
    throw new TypeError("parseEvent: receivedAt must be an ISO datetime");
  }
  const ts = typeof raw.ts === "string" && raw.ts !== "" ? raw.ts : receivedAt;
  if (Number.isNaN(Date.parse(ts))) throw new TypeError("parseEvent: ts must be an ISO datetime");

  if (raw.planDate != null && typeof raw.planDate !== "string") {
    throw new TypeError("parseEvent: planDate must be a string or null");
  }
  if (raw.session != null && !SESSION_SET.has(raw.session)) {
    throw new TypeError(`parseEvent: session "${String(raw.session)}" is not a session`);
  }

  return {
    schema: 1,
    receivedAt,
    source: raw.source,
    symbol: raw.symbol,
    price: raw.price,
    alert: /** @type {AlertKind} */ (raw.alert),
    ts,
    planDate: raw.planDate ?? null,
    session: /** @type {SessionId | null} */ (raw.session ?? null),
  };
}

/**
 * Has price traded into the box since the plan was filed?
 *
 * Two ways to satisfy it: a `BOX_ENTER` alert from TradingView, or any price the log can vouch
 * for — a `PRICE` alert or a manual entry — that sits inside [box.low, box.high]. Other alert
 * kinds (`DOL_HIT`, `INVALIDATED`, `BOX_EXIT`) carry a price but are not a touch on their own.
 *
 * @param {Object} args
 * @param {Plan} args.plan
 * @param {Event[]} [args.events]
 * @returns {boolean}
 */
export function boxTouched({ plan, events = [] }) {
  if (!plan || !plan.box) return false;
  const filed = Date.parse(plan.filedAt);
  for (const e of events || []) {
    if (!e) continue;
    const at = eventAt(e);
    if (Number.isFinite(filed) && Number.isFinite(at) && at < filed) continue;
    if (e.alert === "BOX_ENTER") return true;
    if ((e.alert === "PRICE" || e.source === "manual") && isNum(e.price) &&
        e.price >= plan.box.low && e.price <= plan.box.high) {
      return true;
    }
  }
  return false;
}

/**
 * @typedef {Object} LastPrice
 * @property {number} price
 * @property {string} at      the event's own timestamp
 * @property {number} ageSec  seconds between `at` and `now`, never negative
 */

/**
 * The freshest price in the log. Age is what the Live screen shows to say whether the feed
 * is alive; a stale price is not a reason to enter.
 *
 * @param {Event[]} events
 * @param {Date} [now]
 * @returns {LastPrice | null}
 */
export function lastPrice(events, now = new Date()) {
  /** @type {Event | null} */
  let best = null;
  let bestAt = -Infinity;
  for (const e of events || []) {
    if (!e || !isNum(e.price)) continue;
    const at = eventAt(e);
    if (!Number.isFinite(at) || at < bestAt) continue;
    best = e;
    bestAt = at;
  }
  if (!best) return null;
  return {
    price: best.price,
    at: best.ts || best.receivedAt,
    ageSec: Math.max(0, Math.round((now.getTime() - bestAt) / 1000)),
  };
}

/**
 * Stamp an event with the plan it belongs to. Returns a copy; the log line is never mutated.
 * @param {Event} event
 * @param {{ plan: Plan | null | undefined }} ctx
 * @returns {Event}
 */
export function tagEvent(event, { plan }) {
  return {
    ...event,
    planDate: plan ? plan.date : (event.planDate ?? null),
    session: plan ? plan.session : (event.session ?? null),
  };
}
