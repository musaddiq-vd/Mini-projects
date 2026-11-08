"""Reminders: in N minutes or at a clock time. Saved in state.json, so they survive a restart;
a reminder that came due while Jarvis was closed is told as soon as it starts again."""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timedelta

from jarvis.tools import tool

MAX_MINUTES = 7 * 24 * 60
STATE_KEY = "reminders"
RUNG_KEY = "reminders_rung"  # reminders that rang and wait for Done / Snooze (newest last)
DONE_KEY = "reminders_done"  # finished tasks, for the evening recap

_timers: dict[int, threading.Timer] = {}
_mem: dict[int, dict] = {}  # used when there is no state store (tests, text mode)
_lock = threading.Lock()


def parse_clock(text: str, now: datetime | None = None) -> datetime | None:
    """'18:30', '6:30 pm', '6pm', '6.30 PM', 'shaam 7' -> the next time that clock shows it."""
    now = now or datetime.now()
    t = (text or "").strip().lower()
    pm_hint = any(w in t for w in ("pm", "evening", "shaam", "sham", "raat", "night"))
    am_hint = any(w in t for w in ("am", "morning", "subah", "savere"))
    m = re.search(r"(\d{1,2})(?:[:.](\d{2}))?", t)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    if minute > 59 or hour > 23:
        return None
    if pm_hint and hour < 12:
        hour += 12
    elif am_hint and hour == 12:
        hour = 0
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due <= now:
        # "7" said at 3 pm without am/pm means 7 pm today, otherwise tomorrow
        if not (pm_hint or am_hint) and hour < 12 and due + timedelta(hours=12) > now:
            due += timedelta(hours=12)
        else:
            due += timedelta(days=1)
    return due


# -- storage ---------------------------------------------------------------------
def _load(ctx) -> dict[int, dict]:
    if ctx.state is None:
        return dict(_mem)
    return {int(r["id"]): r for r in ctx.state.get(STATE_KEY, [])}


def _save(ctx, items: dict[int, dict]) -> None:
    if ctx.state is None:
        _mem.clear()
        _mem.update(items)
    else:
        ctx.state.set(STATE_KEY, sorted(items.values(), key=lambda r: r["due"]))


def _fmt(due: float) -> str:
    d = datetime.fromtimestamp(due)
    day = "" if d.date() == datetime.now().date() else (
        " tomorrow" if d.date() == (datetime.now() + timedelta(days=1)).date() else d.strftime(" on %a %d %b"))
    return d.strftime("%I:%M %p").lstrip("0") + day


REPEATS = ("daily", "weekdays")


def next_repeat(due: float, repeat: str, now: float | None = None) -> float:
    """The next time a repeating reminder should ring after `due` (always in the future)."""
    now = now or time.time()
    d = datetime.fromtimestamp(due)
    while True:
        d += timedelta(days=1)
        if repeat == "weekdays" and d.weekday() >= 5:
            continue
        if d.timestamp() > now:
            return d.timestamp()


def _emit(ctx, kind: str, **data) -> None:
    if ctx.events is not None:
        try:
            ctx.events.emit(kind, **data)
        except Exception:
            pass


def _rung(ctx) -> list[dict]:
    if ctx.state is None:
        return list(_mem_rung)
    return list(ctx.state.get(RUNG_KEY, []) or [])


def _set_rung(ctx, items: list[dict]) -> None:
    items = [r for r in items if time.time() - r.get("at", 0) < 2 * 86400][-20:]  # old ones fade out
    if ctx.state is None:
        _mem_rung[:] = items
    else:
        ctx.state.set(RUNG_KEY, items)


_mem_rung: list[dict] = []


def _schedule(ctx, rid: int, due: float, message: str, missed_due: float | None = None) -> None:
    def fire():
        with _lock:
            _timers.pop(rid, None)
            items = _load(ctx)
            item = items.pop(rid, None)
            repeat = (item or {}).get("repeat", "")
            if item and repeat in REPEATS:  # repeating: book the next one
                item["due"] = next_repeat(item["due"], repeat)
                items[rid] = item
            _save(ctx, items)
        late = f" (it was due at {_fmt(missed_due)} while I was off)" if missed_due else ""
        with _lock:
            _set_rung(ctx, [r for r in _rung(ctx) if r["id"] != rid]
                      + [{"id": rid, "message": message, "at": time.time(), "repeat": repeat}])
        _emit(ctx, "reminder_due", id=rid, message=message, due=missed_due or due, late=bool(missed_due),
              repeat=repeat)
        ctx.notify(f"Reminder: {message}{late}")
        if item and repeat in REPEATS:
            _schedule(ctx, rid, item["due"], message)

    timer = threading.Timer(max(0.0, due - time.time()), fire)
    timer.daemon = True
    with _lock:
        _timers[rid] = timer
    timer.start()


def restore_reminders(ctx, missed_delay: float = 6.0) -> int:
    """Re-arm saved reminders after a restart. Returns how many were restored."""
    items = _load(ctx)
    now = time.time()
    for rid, r in items.items():
        if rid in _timers:
            continue
        if r["due"] <= now:  # came due while Jarvis was closed: tell it right after start
            # (a repeating one is told once; fire() then books its next time)
            _schedule(ctx, rid, now + missed_delay, r["message"], missed_due=r["due"])
        else:
            _schedule(ctx, rid, r["due"], r["message"])
    return len(items)


# -- tools -------------------------------------------------------------------------
@tool(
    "set_reminder",
    "Remind the user about something, either after a number of minutes or at a clock time "
    "('at' like '18:30', '6:30 pm', '7 am'). Reminders are kept even if Jarvis is restarted.",
    {"message": {"type": "string"},
     "minutes": {"type": "number", "description": "remind after this many minutes"},
     "at": {"type": "string", "description": "clock time instead of minutes, e.g. '18:30' or '6:30 pm'"},
     "repeat": {"type": "string", "enum": ["once", "daily", "weekdays"],
                "description": "'daily' for every day ('roz', 'har din'), 'weekdays' for Mon-Fri; needs 'at'"}},
    required=["message"],
)
def set_reminder(ctx, message: str, minutes: float | None = None, at: str | None = None, repeat: str = ""):
    repeat = (repeat or "").lower().strip()
    if repeat in {"once", "none", "no"}:
        repeat = ""
    if repeat and repeat not in REPEATS:
        return {"error": "repeat must be 'daily' or 'weekdays'."}
    if repeat and not at:
        return {"error": "A repeating reminder needs a clock time ('at')."}
    if at:
        when = parse_clock(at)
        if when is None:
            return {"error": f"Couldn't understand the time '{at}'. Try like '18:30' or '6:30 pm'."}
        due = when.timestamp()
    elif minutes is not None:
        minutes = float(minutes)
        if not 0 < minutes <= MAX_MINUTES:
            return {"error": "Minutes must be more than 0 and at most a week."}
        due = time.time() + minutes * 60
    else:
        return {"error": "Say when: 'minutes' or 'at' (a clock time)."}
    with _lock:
        items = _load(ctx)
        rid = max(items, default=0) + 1
        while rid in _timers:
            rid += 1
        if repeat == "weekdays" and datetime.fromtimestamp(due).weekday() >= 5:
            due = next_repeat(due, repeat)
        items[rid] = {"id": rid, "due": due, "message": message, "repeat": repeat}
        _save(ctx, items)
    _schedule(ctx, rid, due, message)
    _emit(ctx, "reminder_set", id=rid, message=message, due=due, repeat=repeat)
    every = {"daily": " and every day after that", "weekdays": " and every weekday after that"}.get(repeat, "")
    return f"Reminder #{rid} set for {_fmt(due)}{every}."


@tool("list_reminders", "List reminders that are still waiting.")
def list_reminders(ctx):
    items = [{"id": r["id"], "at": _fmt(r["due"]), "message": r["message"], "repeat": r.get("repeat") or "once"}
             for r in sorted(_load(ctx).values(), key=lambda r: r["due"])]
    return {"reminders": items} if items else "No reminders."


@tool(
    "cancel_reminder",
    "Cancel a waiting reminder by its number.",
    {"id": {"type": "integer"}},
    required=["id"],
)
def cancel_reminder(ctx, id: int):  # noqa: A002 - matches the tool argument name
    rid = int(id)
    with _lock:
        items = _load(ctx)
        found = items.pop(rid, None)
        _save(ctx, items)
        timer = _timers.pop(rid, None)
    if timer:
        timer.cancel()
    _emit(ctx, "reminder_closed", id=rid)
    return f"Cancelled reminder #{rid}." if found else f"No reminder #{rid}."


def _pick_rung(ctx, rid: int | None) -> dict | None:
    items = _rung(ctx)
    if rid:
        return next((r for r in items if r["id"] == int(rid)), None)
    return items[-1] if items else None


@tool(
    "reminder_done",
    "The user finished the task of a reminder that just rang ('ho gaya', 'done', 'kar liya'). Without an id it "
    "means the latest one.",
    {"id": {"type": "integer"}},
)
def reminder_done(ctx, id: int = 0):  # noqa: A002
    with _lock:
        item = _pick_rung(ctx, id)
        if item is None:
            return {"error": "No reminder is waiting to be marked done."}
        _set_rung(ctx, [r for r in _rung(ctx) if r["id"] != item["id"]])
        if ctx.state is not None:
            done = list(ctx.state.get(DONE_KEY, []) or [])
            done.append({"message": item["message"], "at": time.time()})
            ctx.state.set(DONE_KEY, done[-100:])
    _emit(ctx, "reminder_closed", id=item["id"])
    return {"result": f"Marked done: {item['message']}. Congratulate the user in a few words."}


@tool(
    "snooze_reminder",
    "Ring a reminder that just went off again later ('snooze', '10 minute baad yaad dilana'). Default 10 minutes; "
    "without an id it means the latest one.",
    {"minutes": {"type": "number"}, "id": {"type": "integer"}},
)
def snooze_reminder(ctx, minutes: float = 10, id: int = 0):  # noqa: A002
    try:
        minutes = float(minutes or 10)
    except (TypeError, ValueError):
        return {"error": "Minutes must be a number."}
    if not 1 <= minutes <= 24 * 60:
        return {"error": "Snooze for 1 minute up to a day."}
    with _lock:
        item = _pick_rung(ctx, id)
        if item is None:
            return {"error": "No reminder has rung recently to snooze. Set a new one instead."}
        _set_rung(ctx, [r for r in _rung(ctx) if r["id"] != item["id"]])
    _emit(ctx, "reminder_closed", id=item["id"])
    out = set_reminder(ctx, item["message"], minutes=minutes)
    return {"result": f"Snoozed: '{item['message']}' - " + out.replace("Reminder #", "now reminder #")}
