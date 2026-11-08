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


def _schedule(ctx, rid: int, due: float, message: str, missed_due: float | None = None) -> None:
    def fire():
        with _lock:
            _timers.pop(rid, None)
            items = _load(ctx)
            items.pop(rid, None)
            _save(ctx, items)
        late = f" (it was due at {_fmt(missed_due)} while I was off)" if missed_due else ""
        ctx.notify(f"Reminder: {message}{late}")

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
     "at": {"type": "string", "description": "clock time instead of minutes, e.g. '18:30' or '6:30 pm'"}},
    required=["message"],
)
def set_reminder(ctx, message: str, minutes: float | None = None, at: str | None = None):
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
        items[rid] = {"id": rid, "due": due, "message": message}
        _save(ctx, items)
    _schedule(ctx, rid, due, message)
    return f"Reminder #{rid} set for {_fmt(due)}."


@tool("list_reminders", "List reminders that are still waiting.")
def list_reminders(ctx):
    items = [{"id": r["id"], "at": _fmt(r["due"]), "message": r["message"]}
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
    return f"Cancelled reminder #{rid}." if found else f"No reminder #{rid}."
