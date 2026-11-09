// The only door from quant/ (Python) into core/ (JavaScript). One process per call, JSON on
// stdin, JSON on stdout. It exists so the backtester cannot disagree with the app: every plan
// the strategy generates is validated by core/plan.js, and the Python risk port is checked
// against core/risk.js by quant/tests/test_parity.py.
//
//   {"op": "validatePlans", "plans": [...], "rules": {...}}  -> [{ok, errors}, ...]
//   {"op": "call", "calls": [{"fn": "maxContracts", "args": {...}}, ...]}  -> [result, ...]

import { validatePlan } from "../core/plan.js";
import * as risk from "../core/risk.js";

const CALLABLE = new Set([
  "contractsForStop", "maxContracts", "remainingDailyLoss", "trailingDrawdown", "tradeMetrics",
  "insideBox", "isRevenge",
]);

/** @param {any} req */
function handle(req) {
  if (req.op === "validatePlans") {
    return req.plans.map((/** @type {any} */ p) => validatePlan(p, { rules: req.rules }));
  }
  if (req.op === "call") {
    return req.calls.map((/** @type {{ fn: string, args: any }} */ c) => {
      if (!CALLABLE.has(c.fn)) throw new Error(`bridge: ${c.fn} is not callable`);
      const fn = /** @type {any} */ (risk)[c.fn];
      // remainingDailyLoss and trailingDrawdown take positional args in core/risk.js.
      if (c.fn === "remainingDailyLoss") return fn(c.args.trades, c.args.rules, c.args.date);
      if (c.fn === "trailingDrawdown") return fn(c.args.trades, c.args.rules);
      if (c.fn === "insideBox") return fn(c.args.price, c.args.box);
      if (c.fn === "isRevenge") return fn({ ...c.args });
      return fn(c.args);
    });
  }
  throw new Error(`bridge: unknown op ${String(req.op)}`);
}

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { input += chunk; });
process.stdin.on("end", () => {
  try {
    process.stdout.write(JSON.stringify(handle(JSON.parse(input))));
  } catch (err) {
    process.stderr.write(String(err instanceof Error ? err.stack : err));
    process.exitCode = 1;
  }
});
