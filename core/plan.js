/**
 * Plan validation. Hard rule 3 lives here: internal to external, or no trade.
 * Pure — no I/O, no clock.
 *
 * @typedef {import("./types.js").Plan} Plan
 * @typedef {import("./types.js").Rules} Rules
 * @typedef {import("./types.js").SessionId} SessionId
 */

import { SESSIONS } from "./types.js";

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const LIQUIDITY_KINDS = new Set([
  "session_high", "session_low", "pdh", "pdl", "pwh", "pwl", "eqh", "eql",
  "old_high", "old_low", "fvg", "ifvg", "ob", "other",
]);
const BOX_KINDS = new Set(["ifvg", "fvg", "ob", "breaker"]);
const INSTRUMENTS = new Set(["NQ", "MNQ"]);

/**
 * @param {unknown} v
 * @returns {boolean}
 */
function isNum(v) {
  return typeof v === "number" && Number.isFinite(v);
}

/**
 * @param {unknown} v
 * @returns {boolean}
 */
function isNonEmptyString(v) {
  return typeof v === "string" && v.trim().length > 0;
}

/**
 * @param {any} l
 * @param {string} field
 * @param {string[]} errors
 * @returns {boolean}  true when the shape is good enough to keep checking against it
 */
function checkLiquidity(l, field, errors) {
  if (!l || typeof l !== "object") {
    errors.push(`${field}: missing`);
    return false;
  }
  let ok = true;
  if (l.type !== "internal" && l.type !== "external") {
    errors.push(`${field}.type: must be "internal" or "external"`);
    ok = false;
  }
  if (!LIQUIDITY_KINDS.has(l.kind)) {
    errors.push(`${field}.kind: "${String(l.kind)}" is not a known liquidity kind`);
    ok = false;
  }
  if (!isNum(l.level)) {
    errors.push(`${field}.level: must be a number`);
    ok = false;
  }
  return ok;
}

/**
 * Validate a plan against the data contract, the firm rules and what is left of today's loss cap.
 *
 * @param {Plan} plan
 * @param {{ rules: Rules, remainingDailyLoss?: number }} ctx
 * @returns {{ ok: boolean, errors: string[] }}
 */
export function validatePlan(plan, { rules, remainingDailyLoss } = /** @type {any} */ ({})) {
  /** @type {string[]} */
  const errors = [];
  if (!plan || typeof plan !== "object") {
    return { ok: false, errors: ["plan: missing"] };
  }
  if (!rules || typeof rules !== "object") {
    return { ok: false, errors: ["rules: missing"] };
  }
  const remaining = isNum(remainingDailyLoss) ? /** @type {number} */ (remainingDailyLoss) : rules.dailyLossLimit;

  // --- required fields, per types.js ---
  if (plan.schema !== 1) errors.push("schema: must be 1");
  if (!isNonEmptyString(plan.date) || !DATE_RE.test(plan.date)) errors.push("date: must be YYYY-MM-DD");
  if (!SESSIONS.includes(/** @type {any} */ (plan.session))) {
    errors.push(`session: "${String(plan.session)}" is not a session`);
  } else if (Array.isArray(rules.allowedSessions) && !rules.allowedSessions.includes(plan.session)) {
    errors.push(`session: "${plan.session}" is not in rules.allowedSessions`);
  }
  if (!isNonEmptyString(plan.filedAt) || Number.isNaN(Date.parse(plan.filedAt))) {
    errors.push("filedAt: must be an ISO datetime with offset");
  }
  if (plan.bias !== "long" && plan.bias !== "short") errors.push('bias: must be "long" or "short"');
  if (!isNonEmptyString(plan.narrative)) errors.push("narrative: missing");

  checkLiquidity(plan.dol, "dol", errors);
  const targetOk = checkLiquidity(plan.target, "target", errors);
  if (targetOk && plan.target.type !== "external") {
    errors.push('target.type: must be "external" — internal to external, or no trade');
  }

  const dr = plan.dealingRange;
  let drOk = true;
  if (!dr || typeof dr !== "object" || !isNum(dr.high) || !isNum(dr.low) || !isNonEmptyString(dr.timeframe)) {
    errors.push("dealingRange: needs numeric high, low and a timeframe");
    drOk = false;
  } else if (dr.low >= dr.high) {
    errors.push("dealingRange: low must be below high");
    drOk = false;
  }

  const box = plan.box;
  let boxOk = true;
  if (!box || typeof box !== "object" || !isNum(box.high) || !isNum(box.low)) {
    errors.push("box: needs numeric high and low");
    boxOk = false;
  } else {
    if (!isNonEmptyString(box.timeframe)) errors.push("box.timeframe: missing");
    if (!BOX_KINDS.has(box.kind)) errors.push(`box.kind: "${String(box.kind)}" is not ifvg/fvg/ob/breaker`);
    if (box.low >= box.high) {
      errors.push("box: low must be below high");
      boxOk = false;
    }
  }

  if (!isNum(plan.invalidation)) errors.push("invalidation: must be a number");
  if (typeof plan.entryOnlyInsideBox !== "boolean") errors.push("entryOnlyInsideBox: must be a boolean");
  if (!isNum(plan.maxTrades) || plan.maxTrades < 0) errors.push("maxTrades: must be a number");

  const risk = plan.risk;
  let riskOk = true;
  if (!risk || typeof risk !== "object" || !isNum(risk.maxDollars) || !isNum(risk.maxContracts)) {
    errors.push("risk: needs numeric maxDollars and maxContracts");
    riskOk = false;
  } else if (!INSTRUMENTS.has(risk.instrument)) {
    errors.push(`risk.instrument: "${String(risk.instrument)}" is not NQ or MNQ`);
  }

  const cl = plan.checklist;
  if (!cl || typeof cl !== "object") {
    errors.push("checklist: missing");
  } else {
    for (const k of ["dolIdentified", "internalToExternal", "boxMarked", "newsChecked"]) {
      if (typeof (/** @type {any} */ (cl)[k]) !== "boolean") errors.push(`checklist.${k}: must be a boolean`);
    }
  }

  // --- geometry: the box, the invalidation and the target must agree with the bias ---
  if (boxOk && isNum(plan.invalidation) && targetOk && (plan.bias === "long" || plan.bias === "short")) {
    if (plan.bias === "long") {
      if (plan.invalidation >= box.low) {
        errors.push("invalidation: for a long it must sit below the box low");
      }
      if (plan.target.level <= box.high) {
        errors.push("target.level: for a long it must sit above the box high");
      }
    } else {
      if (plan.invalidation <= box.high) {
        errors.push("invalidation: for a short it must sit above the box high");
      }
      if (plan.target.level >= box.low) {
        errors.push("target.level: for a short it must sit below the box low");
      }
    }
  }

  if (boxOk && drOk && (box.low < dr.low || box.high > dr.high)) {
    errors.push("box: must sit inside the dealing range");
  }

  // --- risk: the floor comes first ---
  if (riskOk) {
    if (isNum(remaining) && risk.maxDollars > remaining) {
      errors.push(`risk.maxDollars: ${risk.maxDollars} is more than the $${remaining} left of today's loss limit`);
    }
    const cap = (rules.accountSize * rules.maxRiskPerTradePct) / 100;
    if (isNum(cap) && risk.maxDollars > cap) {
      errors.push(`risk.maxDollars: ${risk.maxDollars} is over the per-trade cap of $${cap}`);
    }
  }

  return { ok: errors.length === 0, errors };
}

/**
 * Where the plan file lives, relative to the data root.
 * @param {Pick<Plan, "date" | "session">} plan
 * @returns {string}
 */
export function planPath(plan) {
  return `plans/${plan.date}-${plan.session}.json`;
}

/**
 * Distance in points from the box midpoint to the invalidation — the stop, if he enters mid-box.
 * @param {Pick<Plan, "box" | "invalidation">} plan
 * @returns {number}
 */
export function stopPoints(plan) {
  const mid = (plan.box.high + plan.box.low) / 2;
  return Math.abs(mid - plan.invalidation);
}
