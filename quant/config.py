"""Reads the same files the app reads. quant/ never keeps its own copy of a rule.

`data/rules.json` is the firm floor, `data/config.json` (or the committed example when the real one
has not been generated yet) holds the session windows. Session windows are resolved to tz-aware
timestamps exactly the way `core/clock.js` does it: a window whose end is not after its start (asia,
20:00 to 00:00) ends on the next ET date, and the window belongs to the evening's date.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"

POINT_VALUE = {"NQ": 20.0, "MNQ": 2.0}  # mirrors core/types.js POINT_VALUE
TICK_SIZE = 0.25  # mirrors core/types.js TICK_SIZE
SESSIONS = ("asia", "ny_am", "ny_pm")


def load_rules(data_root: Path = DATA) -> dict:
    return json.loads((data_root / "rules.json").read_text(encoding="utf-8"))


def load_config(data_root: Path = DATA) -> dict:
    real = data_root / "config.json"
    path = real if real.exists() else data_root / "config.example.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


@dataclass(frozen=True)
class Window:
    session: str
    date: date  # the ET date the window belongs to (the evening's date for asia)
    start: datetime
    end: datetime
    plan_deadline: datetime  # start minus config.planDeadlineMin: the box must be marked by then


def session_window(config: dict, session: str, d: date) -> Window:
    tz = ZoneInfo(config.get("timezone", "America/Toronto"))
    s = config["sessions"][session]
    start_t, end_t = _hhmm(s["start"]), _hhmm(s["end"])
    end_d = d + timedelta(days=1) if end_t <= start_t else d
    start = datetime.combine(d, start_t, tz)
    end = datetime.combine(end_d, end_t, tz)
    deadline = start - timedelta(minutes=int(config.get("planDeadlineMin", 15)))
    return Window(session, d, start, end, deadline)


def enabled_sessions(config: dict, rules: dict) -> list[str]:
    allowed = set(rules.get("allowedSessions", SESSIONS))
    return [s for s in SESSIONS if s in allowed and config["sessions"].get(s, {}).get("enabled", True)]
