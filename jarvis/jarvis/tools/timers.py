"""Countdown timers and focus mode. The window draws each running timer as a ring on the orb.

Events sent (via ctx.events):  timer {id, label, ends, total}   timer_end {id, label, done: bool}
"""

from __future__ import annotations

import threading
import time

from jarvis.tools import tool

MAX_SECONDS = 24 * 60 * 60
_timers: dict[int, dict] = {}  # id -> {label, ends, total, timer, focus}
_lock = threading.Lock()
_next_id = 1


def _emit(ctx, kind: str, **data) -> None:
    if ctx.events is not None:
        ctx.events.emit(kind, **data)


def _fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h}h {m}m"
    if m and s:
        return f"{m} min {s} sec"
    return f"{m} min" if m else f"{s} sec"


def focus_active(ctx) -> bool:
    """True while a focus session runs (check-ins and alerts stay quiet)."""
    until = ctx.state.get("focus_until", 0) if ctx.state is not None else 0
    return time.time() < (until or 0)


def start_timer(ctx, seconds: float, label: str, focus: bool = False) -> int:
    global _next_id
    with _lock:
        tid = _next_id
        _next_id += 1

    def done():
        with _lock:
            info = _timers.pop(tid, None)
        if not info:
            return
        _emit(ctx, "timer_end", id=tid, label=label, done=True)
        if focus:
            if ctx.state is not None:
                ctx.state.set("focus_until", 0)
            ctx.notify(f"Focus session over ({label}). Great work! Suggest a 5 minute break.")
        else:
            ctx.notify(f"Timer done: {label}" if label else "Timer done!")

    timer = threading.Timer(seconds, done)
    timer.daemon = True
    ends = time.time() + seconds
    with _lock:
        _timers[tid] = {"label": label, "ends": ends, "total": seconds, "timer": timer, "focus": focus}
    timer.start()
    _emit(ctx, "timer", id=tid, label=label, ends=ends, total=seconds, focus=focus)
    return tid


@tool(
    "set_timer",
    "Start a countdown timer ('5 minute ka timer', 'timer for 30 seconds'). It shows as a ring on the "
    "orb and rings when done. Give minutes and/or seconds, and a short label like 'tea' or 'eggs'.",
    {"minutes": {"type": "number"}, "seconds": {"type": "number"},
     "label": {"type": "string", "description": "what the timer is for"}},
)
def set_timer(ctx, minutes: float = 0, seconds: float = 0, label: str = ""):
    total = float(minutes or 0) * 60 + float(seconds or 0)
    if not 0 < total <= MAX_SECONDS:
        return {"error": "A timer needs a length between 1 second and 24 hours."}
    tid = start_timer(ctx, total, label.strip())
    return f"Timer #{tid} started for {_fmt(total)}" + (f" ({label})." if label else ".")


@tool("list_timers", "List running timers and how much time is left.")
def list_timers(ctx):
    now = time.time()
    with _lock:
        items = [{"id": tid, "label": t["label"] or ("focus" if t["focus"] else "timer"),
                  "left": _fmt(max(0, t["ends"] - now))} for tid, t in sorted(_timers.items())]
    return {"timers": items} if items else "No timers running."


@tool(
    "cancel_timer",
    "Stop a running timer (or focus session) by number; without a number, stop the most recent one.",
    {"id": {"type": "integer"}},
)
def cancel_timer(ctx, id: int | None = None):  # noqa: A002 - tool argument name
    with _lock:
        if not _timers:
            return "No timers running."
        tid = int(id) if id is not None else max(_timers)
        info = _timers.pop(tid, None)
    if not info:
        return {"error": f"No timer #{tid}."}
    info["timer"].cancel()
    if info["focus"] and ctx.state is not None:
        ctx.state.set("focus_until", 0)
    _emit(ctx, "timer_end", id=tid, label=info["label"], done=False)
    return f"Stopped timer #{tid}."


@tool(
    "focus_mode",
    "Start a focus (pomodoro) session: a countdown, and Jarvis stays quiet - no check-ins or alerts - "
    "until it ends. 'focus mode 25 minutes', 'padhai ke liye focus mode'. Default 25 minutes.",
    {"minutes": {"type": "number"}, "task": {"type": "string", "description": "what the user works on"}},
)
def focus_mode(ctx, minutes: float = 25, task: str = ""):
    minutes = float(minutes or 25)
    if not 1 <= minutes <= 240:
        return {"error": "Focus sessions can be 1 to 240 minutes."}
    if ctx.state is not None:
        ctx.state.set("focus_until", time.time() + minutes * 60)
    label = f"Focus: {task}" if task else "Focus"
    tid = start_timer(ctx, minutes * 60, label, focus=True)
    return (f"Focus session #{tid} started for {_fmt(minutes * 60)}. I'll stay quiet until it ends. "
            "Wish the user luck in a few words.")
