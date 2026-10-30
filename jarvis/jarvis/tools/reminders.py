"""Simple reminders that fire while Jarvis is running."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

from jarvis.tools import tool

_active: dict[int, tuple[float, str, threading.Timer]] = {}
_next_id = 1
_lock = threading.Lock()


@tool(
    "set_reminder",
    "Remind the user about something after a number of minutes (works while Jarvis is running).",
    {"minutes": {"type": "number"}, "message": {"type": "string"}},
    required=["minutes", "message"],
)
def set_reminder(ctx, minutes: float, message: str):
    global _next_id
    minutes = float(minutes)
    if not 0 < minutes <= 24 * 60:
        return {"error": "Minutes must be between 0 and 1440."}
    with _lock:
        rid = _next_id
        _next_id += 1

    def fire():
        with _lock:
            _active.pop(rid, None)
        ctx.notify(f"Reminder: {message}")

    timer = threading.Timer(minutes * 60, fire)
    timer.daemon = True
    with _lock:
        _active[rid] = (time.time() + minutes * 60, message, timer)
    timer.start()
    at = (datetime.now() + timedelta(minutes=minutes)).strftime("%I:%M %p")
    return f"Reminder #{rid} set for {at}."


@tool("list_reminders", "List reminders that are still waiting.")
def list_reminders(ctx):
    with _lock:
        items = [
            {"id": rid, "at": datetime.fromtimestamp(t).strftime("%I:%M %p"), "message": m}
            for rid, (t, m, _) in sorted(_active.items())
        ]
    return {"reminders": items} if items else "No reminders."


@tool(
    "cancel_reminder",
    "Cancel a waiting reminder by its number.",
    {"id": {"type": "integer"}},
    required=["id"],
)
def cancel_reminder(ctx, id: int):  # noqa: A002 - matches the tool argument name
    with _lock:
        entry = _active.pop(int(id), None)
    if not entry:
        return f"No reminder #{id}."
    entry[2].cancel()
    return f"Cancelled reminder #{id}."
