/**
 * Eastern Time session clock. No libraries: every conversion goes through
 * `Intl.DateTimeFormat` with an explicit `timeZone`, so the machine timezone never matters
 * and DST is whatever the IANA database says it is.
 *
 * @typedef {import("./types.js").Config} Config
 * @typedef {import("./types.js").SessionId} SessionId
 * @typedef {import("./types.js").SessionWindow} SessionWindow
 */

import { SESSIONS } from "./types.js";

/** Fallback when a config is not supplied. */
export const DEFAULT_TZ = "America/Toronto";

/**
 * @typedef {Object} ETNow
 * @property {string} date     "YYYY-MM-DD" in ET
 * @property {string} time     "HH:MM" in ET
 * @property {number} minutes  Minutes since ET midnight
 * @property {string} iso      ISO datetime with the ET UTC offset, e.g. "2026-09-10T23:59:00-04:00"
 */

/**
 * @typedef {Object} Window
 * @property {SessionId} session
 * @property {string} date            ET date the window belongs to (the evening's date for asia)
 * @property {Date} start
 * @property {Date} end
 * @property {Date | null} killzoneStart
 * @property {Date} planDeadline
 * @property {string} startET         "HH:MM"
 * @property {string} endET           "HH:MM"
 */

/** @type {Map<string, Intl.DateTimeFormat>} */
const FORMATTERS = new Map();

/**
 * @param {string} tz
 * @returns {Intl.DateTimeFormat}
 */
function formatter(tz) {
  let f = FORMATTERS.get(tz);
  if (!f) {
    f = new Intl.DateTimeFormat("en-US", {
      timeZone: tz,
      hourCycle: "h23",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    FORMATTERS.set(tz, f);
  }
  return f;
}

/**
 * Wall-clock parts of an instant in a timezone.
 * @param {Date} date
 * @param {string} tz
 * @returns {{ year: number, month: number, day: number, hour: number, minute: number, second: number }}
 */
export function tzParts(date, tz = DEFAULT_TZ) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) {
    throw new TypeError("tzParts: invalid Date");
  }
  /** @type {Record<string, string>} */
  const out = {};
  for (const p of formatter(tz).formatToParts(date)) {
    if (p.type !== "literal") out[p.type] = p.value;
  }
  return {
    year: Number(out["year"]),
    month: Number(out["month"]),
    day: Number(out["day"]),
    hour: Number(out["hour"]),
    minute: Number(out["minute"]),
    second: Number(out["second"]),
  };
}

/**
 * Offset of `tz` at that instant, in milliseconds east of UTC (EDT = -4h, EST = -5h).
 * @param {Date} date
 * @param {string} tz
 * @returns {number}
 */
export function tzOffsetMs(date, tz = DEFAULT_TZ) {
  const p = tzParts(date, tz);
  return Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second) - Math.floor(date.getTime() / 1000) * 1000;
}

/**
 * "-04:00" / "-05:00" for the offset in force at that instant.
 * @param {Date} date
 * @param {string} tz
 * @returns {string}
 */
export function tzOffsetString(date, tz = DEFAULT_TZ) {
  const total = Math.round(tzOffsetMs(date, tz) / 60000);
  const sign = total < 0 ? "-" : "+";
  const abs = Math.abs(total);
  return `${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`;
}

/**
 * @param {number} n
 * @returns {string}
 */
function pad(n) {
  return String(n).padStart(2, "0");
}

/**
 * Turn an ET wall clock (date + "HH:MM") into the absolute instant.
 * Two fixed-point passes settle the offset, which is enough for whole-minute zone offsets.
 * @param {string} etDate  "YYYY-MM-DD"
 * @param {string} hhmm    "HH:MM"
 * @param {string} tz
 * @returns {Date}
 */
export function zonedToUtc(etDate, hhmm, tz = DEFAULT_TZ) {
  const [y, m, d] = etDate.split("-").map(Number);
  const [hh, mm] = hhmm.split(":").map(Number);
  if ([y, m, d, hh, mm].some((n) => !Number.isFinite(n))) {
    throw new TypeError(`zonedToUtc: bad input ${etDate} ${hhmm}`);
  }
  const wall = Date.UTC(y, m - 1, d, hh, mm, 0);
  let ts = wall - tzOffsetMs(new Date(wall), tz);
  ts = wall - tzOffsetMs(new Date(ts), tz);
  return new Date(ts);
}

/**
 * Calendar arithmetic on an ET date string. Never touches a clock, so DST cannot shift it.
 * @param {string} etDate
 * @param {number} days
 * @returns {string}
 */
export function addDaysET(etDate, days) {
  const [y, m, d] = etDate.split("-").map(Number);
  const t = new Date(Date.UTC(y, m - 1, d));
  t.setUTCDate(t.getUTCDate() + days);
  return `${t.getUTCFullYear()}-${pad(t.getUTCMonth() + 1)}-${pad(t.getUTCDate())}`;
}

/**
 * Day of week for an ET date string: 0 = Sunday.
 * @param {string} etDate
 * @returns {number}
 */
export function dayOfWeekET(etDate) {
  const [y, m, d] = etDate.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).getUTCDay();
}

/**
 * @param {string} etDate
 * @returns {boolean}
 */
export function isWeekendET(etDate) {
  const dow = dayOfWeekET(etDate);
  return dow === 0 || dow === 6;
}

/**
 * "HH:MM" to minutes since midnight.
 * @param {string} hhmm
 * @returns {number}
 */
export function toMinutes(hhmm) {
  const [hh, mm] = hhmm.split(":").map(Number);
  return hh * 60 + mm;
}

/**
 * Current ET wall clock. The ET calendar date is the trading date: the asia window
 * (20:00–00:00) lives entirely inside its evening's date, so 23:59 on 2026-09-10 is 2026-09-10.
 * @param {Date} [date]
 * @param {string} [tz]
 * @returns {ETNow}
 */
export function nowET(date = new Date(), tz = DEFAULT_TZ) {
  const p = tzParts(date, tz);
  const dateStr = `${p.year}-${pad(p.month)}-${pad(p.day)}`;
  const time = `${pad(p.hour)}:${pad(p.minute)}`;
  return {
    date: dateStr,
    time,
    minutes: p.hour * 60 + p.minute,
    iso: `${dateStr}T${time}:${pad(p.second)}${tzOffsetString(date, tz)}`,
  };
}

/**
 * @param {Config} config
 * @returns {string}
 */
function tzOf(config) {
  return (config && config.timezone) || DEFAULT_TZ;
}

/**
 * Absolute start/end of every enabled session on one ET date, ordered by start.
 * `end: "00:00"` means midnight closing that date, i.e. 00:00 on the following day.
 * @param {Config} config
 * @param {string} [etDate]  defaults to today in ET
 * @returns {Window[]}
 */
export function sessionWindows(config, etDate = nowET(new Date(), tzOf(config)).date) {
  const tz = tzOf(config);
  /** @type {Window[]} */
  const out = [];
  for (const session of SESSIONS) {
    const w = /** @type {SessionWindow | undefined} */ (config.sessions?.[session]);
    if (!w || !w.enabled) continue;
    const start = zonedToUtc(etDate, w.start, tz);
    const endDate = toMinutes(w.end) <= toMinutes(w.start) ? addDaysET(etDate, 1) : etDate;
    const end = zonedToUtc(endDate, w.end, tz);
    out.push({
      session,
      date: etDate,
      start,
      end,
      killzoneStart: w.killzoneStart ? zonedToUtc(etDate, w.killzoneStart, tz) : null,
      planDeadline: new Date(start.getTime() - (config.planDeadlineMin ?? 0) * 60000),
      startET: w.start,
      endET: w.end,
    });
  }
  out.sort((a, b) => a.start.getTime() - b.start.getTime());
  return out;
}

/**
 * The one window for a session on a date. Throws when the session is unknown or disabled.
 * @param {Config} config
 * @param {SessionId} session
 * @param {string} etDate
 * @returns {Window}
 */
export function windowFor(config, session, etDate) {
  const found = sessionWindows(config, etDate).find((w) => w.session === session);
  if (!found) throw new Error(`windowFor: session "${session}" is not enabled on ${etDate}`);
  return found;
}

/**
 * Nearest session to `date`: the one containing it, else the closest by edge distance.
 * Looks at the previous, current and next ET date so a window that straddles midnight is seen.
 * @param {Config} config
 * @param {Date} [date]
 * @returns {{ session: SessionId, state: "before" | "in" | "after", date: string, window: Window }}
 */
export function currentSession(config, date = new Date()) {
  const tz = tzOf(config);
  const today = nowET(date, tz).date;
  /** @type {Window[]} */
  const all = [];
  for (const d of [addDaysET(today, -1), today, addDaysET(today, 1)]) {
    all.push(...sessionWindows(config, d));
  }
  if (all.length === 0) throw new Error("currentSession: no enabled sessions in config");
  const t = date.getTime();
  const inside = all.find((w) => t >= w.start.getTime() && t < w.end.getTime());
  if (inside) return { session: inside.session, state: "in", date: inside.date, window: inside };

  let best = all[0];
  let bestDist = Infinity;
  for (const w of all) {
    const dist = t < w.start.getTime() ? w.start.getTime() - t : t - w.end.getTime();
    if (dist < bestDist) {
      bestDist = dist;
      best = w;
    }
  }
  return {
    session: best.session,
    state: t < best.start.getTime() ? "before" : "after",
    date: best.date,
    window: best,
  };
}

/**
 * Session id when `date` is inside a window, else null.
 * @param {Config} config
 * @param {Date} [date]
 * @returns {SessionId | null}
 */
export function activeSession(config, date = new Date()) {
  const cur = currentSession(config, date);
  return cur.state === "in" ? cur.session : null;
}

/**
 * Plan is due `config.planDeadlineMin` before the window opens.
 * @param {Config} config
 * @param {SessionId} session
 * @param {string} etDate
 * @returns {Date}
 */
export function planDeadline(config, session, etDate) {
  return windowFor(config, session, etDate).planDeadline;
}

/**
 * Review is due the next ET day at `config.reviewDeadline`.
 * @param {Config} config
 * @param {SessionId} session
 * @param {string} etDate
 * @returns {Date}
 */
export function reviewDeadline(config, session, etDate) {
  windowFor(config, session, etDate); // validates the session is enabled
  return zonedToUtc(addDaysET(etDate, 1), config.reviewDeadline || "06:00", tzOf(config));
}

/**
 * "3h 45m" / "12m" / "0m". Negative input is clamped to zero; callers word the direction.
 * @param {number} ms
 * @returns {string}
 */
export function formatDuration(ms) {
  const total = Math.max(0, Math.floor(ms / 60000));
  const h = Math.floor(total / 60);
  const m = total % 60;
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}
