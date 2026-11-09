/**
 * Slash data contract. Every lane builds against this file. Change it only with a decision entry.
 * Plain ESM + JSDoc; `npm run typecheck` runs tsc over it.
 */

/** Dollars per one full point of movement, one contract. */
export const POINT_VALUE = Object.freeze({ NQ: 20, MNQ: 2 });

/** Minimum price increment. */
export const TICK_SIZE = 0.25;

/** @typedef {"asia" | "ny_am" | "ny_pm"} SessionId */
export const SESSIONS = /** @type {const} */ (["asia", "ny_am", "ny_pm"]);

/** @typedef {"NQ" | "MNQ"} Instrument */

/** @typedef {"long" | "short"} Side */

/**
 * Hard blocks stop an entry call. Flags are logged and counted but do not block.
 * @typedef {"NO_PLAN" | "OUTSIDE_SESSION" | "BOX_NOT_TOUCHED" | "ENTRY_OUTSIDE_BOX" |
 *   "OVER_MAX_CONTRACTS" | "DAILY_LOSS_REACHED" | "TRAILING_DD_REACHED" | "OVER_MAX_TRADES" |
 *   "REVENGE_TRADE" | "AGAINST_BIAS" | "STOP_BEYOND_INVALIDATION"} ViolationCode
 */
export const HARD_VIOLATIONS = /** @type {const} */ ([
  "NO_PLAN", "OUTSIDE_SESSION", "BOX_NOT_TOUCHED", "ENTRY_OUTSIDE_BOX", "OVER_MAX_CONTRACTS",
  "DAILY_LOSS_REACHED", "TRAILING_DD_REACHED", "OVER_MAX_TRADES",
]);
export const FLAG_VIOLATIONS = /** @type {const} */ ([
  "REVENGE_TRADE", "AGAINST_BIAS", "STOP_BEYOND_INVALIDATION",
]);

/** @typedef {"PRICE" | "BOX_ENTER" | "BOX_EXIT" | "DOL_HIT" | "INVALIDATED" | "SETUP"} AlertKind */
export const ALERTS = /** @type {const} */ (["PRICE", "BOX_ENTER", "BOX_EXIT", "DOL_HIT", "INVALIDATED", "SETUP"]);

/**
 * @typedef {"UNPLANNED" | "PLANNED" | "PLANNED_LATE" | "ACTIVE" | "REVIEWED" | "MISSED" | "SKIPPED"} SessionState
 */

/**
 * A liquidity level. `internal` = inside the dealing range (FVG, IFVG, OB); `external` = the
 * range's own highs/lows or older highs/lows beyond it.
 * @typedef {Object} Liquidity
 * @property {"internal" | "external"} type
 * @property {"session_high" | "session_low" | "pdh" | "pdl" | "pwh" | "pwl" | "eqh" | "eql" |
 *   "old_high" | "old_low" | "fvg" | "ifvg" | "ob" | "other"} kind
 * @property {number} level
 * @property {string} [label]
 */

/**
 * The entry zone. Price must trade into [low, high] before an entry is allowed.
 * @typedef {Object} Box
 * @property {number} high
 * @property {number} low
 * @property {"1m" | "3m" | "5m" | "15m" | "1h"} timeframe
 * @property {"ifvg" | "fvg" | "ob" | "breaker"} kind
 */

/**
 * Session plan, file `data/plans/YYYY-MM-DD-<session>.json`.
 * @typedef {Object} Plan
 * @property {1} schema
 * @property {string} date          ISO date in ET, e.g. "2026-09-10"
 * @property {SessionId} session
 * @property {string} filedAt       ISO datetime with offset
 * @property {Side} bias
 * @property {string} narrative
 * @property {Liquidity} dol
 * @property {{ high: number, low: number, timeframe: string }} dealingRange
 * @property {Box} box
 * @property {number} invalidation
 * @property {Liquidity} target
 * @property {{ maxDollars: number, maxContracts: number, instrument: Instrument }} risk
 * @property {boolean} entryOnlyInsideBox
 * @property {number} maxTrades
 * @property {{ dolIdentified: boolean, internalToExternal: boolean, boxMarked: boolean, newsChecked: boolean }} checklist
 * @property {boolean} [skip]       Pre-declared skip; does not break streaks.
 * @property {string} [skipReason]
 */

/**
 * One trade, one line of `data/trades.jsonl`.
 * @typedef {Object} Trade
 * @property {1} schema
 * @property {string} id            e.g. "t_20260910_2214_01"
 * @property {string} date
 * @property {SessionId} session
 * @property {string | null} planFile   relative path from data/, or null when NO_PLAN
 * @property {Instrument} instrument
 * @property {Side} side
 * @property {number} contracts
 * @property {number} entryPrice
 * @property {string} entryTime
 * @property {number} stopPrice
 * @property {number | null} exitPrice
 * @property {string | null} exitTime
 * @property {number} pnl           Dollars, net of nothing (fees out of scope v1)
 * @property {number} riskDollars
 * @property {number} r
 * @property {"win" | "loss" | "breakeven" | "open"} outcome
 * @property {boolean} insideBox
 * @property {boolean} boxTouchedBeforeEntry
 * @property {ViolationCode[]} violations
 * @property {"A" | "B" | "C" | "D" | "F"} grade
 * @property {string} notes
 * @property {"manual" | "import"} source
 */

/**
 * One event, one line of `data/events.jsonl`. The secret is stripped before write.
 * @typedef {Object} Event
 * @property {1} schema
 * @property {string} receivedAt
 * @property {"tradingview" | "manual"} source
 * @property {string} symbol
 * @property {number} price
 * @property {AlertKind} alert
 * @property {string} ts            Sender timestamp (TradingView {{timenow}}, UTC)
 * @property {string | null} planDate
 * @property {SessionId | null} session
 */

/**
 * One state transition, one line of `data/sessions.jsonl`.
 * @typedef {Object} SessionTransition
 * @property {1} schema
 * @property {string} date
 * @property {SessionId} session
 * @property {SessionState} from
 * @property {SessionState} to
 * @property {string} at
 * @property {string} [reason]
 */

/**
 * Post-session review, file `data/reviews/YYYY-MM-DD-<session>.json`.
 * @typedef {Object} Review
 * @property {1} schema
 * @property {string} date
 * @property {SessionId} session
 * @property {string} filedAt
 * @property {boolean} waitedForBox
 * @property {boolean} tradedInternalToExternal
 * @property {string} whatHappened
 * @property {string} whatSlashWouldSay
 * @property {"A" | "B" | "C" | "D" | "F"} grade
 * @property {ViolationCode[]} violations
 */

/**
 * `data/rules.json`. Prop-firm rules; generic until he picks a firm.
 * @typedef {Object} Rules
 * @property {1} schema
 * @property {string} firm
 * @property {number} accountSize
 * @property {number} dailyLossLimit
 * @property {number} maxDrawdown
 * @property {"trailing_intraday" | "trailing_eod" | "static"} drawdownType
 * @property {number} consistencyRulePct
 * @property {number} minTradingDays
 * @property {number} payoutCadenceDays
 * @property {Record<Instrument, number>} maxContracts
 * @property {number} maxRiskPerTradePct
 * @property {number} revengeWindowMin
 * @property {number} maxTradesPerSession
 * @property {SessionId[]} allowedSessions
 */

/**
 * @typedef {Object} SessionWindow
 * @property {string} start   "HH:MM" ET
 * @property {string} end     "HH:MM" ET; "00:00" means midnight, end of the same trading night
 * @property {string} [killzoneStart]
 * @property {boolean} enabled
 */

/**
 * `data/config.json` (ignored by git; copy from config.example.json).
 * @typedef {Object} Config
 * @property {1} schema
 * @property {string} timezone
 * @property {Instrument} instrument
 * @property {Record<SessionId, SessionWindow>} sessions
 * @property {number} planDeadlineMin
 * @property {string} reviewDeadline   "HH:MM" ET next day
 * @property {{ port: number, secret: string, publicUrl: string }} receiver
 * @property {string} dataRoot
 */

/**
 * Weekly scorecard, file `data/scorecards/YYYY-Www.json`.
 * @typedef {Object} Scorecard
 * @property {1} schema
 * @property {string} week              ISO week, e.g. "2026-W37"
 * @property {number} sessionsEnabled
 * @property {number} sessionsPlanned
 * @property {number} sessionsReviewed
 * @property {number} sessionsMissed
 * @property {number} trades
 * @property {number} wins
 * @property {number} losses
 * @property {number} netPnl
 * @property {number} avgR
 * @property {number} expectancy
 * @property {Partial<Record<ViolationCode, number>>} violations
 * @property {number} boxAdherencePct
 * @property {number} consistencyPct    largest winning day / total profit, percent
 * @property {number} remainingTrailingDD
 * @property {number} drawdownFloor
 * @property {number} processStreak
 * @property {number} planStreak
 */

/**
 * Result of the entry gate.
 * @typedef {Object} EntryVerdict
 * @property {boolean} allowed
 * @property {ViolationCode[]} blocks
 * @property {ViolationCode[]} flags
 * @property {number} maxContracts
 */
