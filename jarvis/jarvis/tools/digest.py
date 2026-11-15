"""Daily digest: the whole day on one card - weather, meetings, what to do first, reminders, habits and
today's spending. Every part is optional; a part that fails (no internet, no calendar) is simply left out."""

from __future__ import annotations

import logging
from datetime import datetime

from jarvis.tools import tool

log = logging.getLogger("jarvis.digest")


def _city(ctx) -> str:
    if getattr(ctx.settings, "city", ""):
        return ctx.settings.city
    for key in ("city", "home_city"):
        fact = ctx.memory.get(key)
        if fact:
            return fact.value
    return ""


def _part(name, fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - one missing part never spoils the digest
        log.info("digest: %s unavailable", name, exc_info=True)
        return None


def greeting(hour: int) -> str:
    return "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"


@tool(
    "daily_digest",
    "The user's day on one card: weather, today's meetings, top to-dos, reminders, habits and today's spending "
    "('aaj ka plan', 'my day', 'daily digest', 'subah ki briefing'). Summarise it in two or three spoken sentences.",
    {"weather": {"type": "boolean", "description": "include the weather (needs internet), default true"}},
    slow=True,
)
def daily_digest(ctx, weather: bool = True):
    from jarvis.tools.habits import habit_status
    from jarvis.tools.money import expense_summary
    from jarvis.tools.notes import what_first
    from jarvis.tools.reminders import list_reminders

    now = datetime.now()
    name = getattr(ctx.settings, "user_name", "") or ""
    out: dict = {"digest": now.strftime("%A %d %B"), "greeting": f"{greeting(now.hour)}{', ' + name if name else ''}"}

    city = _city(ctx)
    if weather and city:
        from jarvis.tools.web import get_weather

        w = _part("weather", lambda: get_weather(ctx, city))
        if isinstance(w, dict) and "error" not in w:
            out["weather"] = {k: w.get(k) for k in ("place", "now", "temperature_c", "today_max_c", "today_min_c",
                                                    "rain_chance_percent")}

    from jarvis.tools import calendar_sync

    if calendar_sync.links(ctx.settings):
        cal = _part("calendar", lambda: calendar_sync.calendar_events(ctx, "today"))
        if isinstance(cal, dict) and "error" not in cal:
            out["meetings"] = [{"title": e.get("title") or "Event", "time": e.get("when", "")}
                               for e in cal.get("events", [])][:5]

    todo = _part("todos", lambda: what_first(ctx))
    if isinstance(todo, dict):
        out["todos"] = {"first": [t["text"] for t in todo.get("do_first", [])], "open": todo.get("open", 0),
                        "overdue": todo.get("overdue", 0), "due_today": todo.get("due_today", 0)}

    rem = _part("reminders", lambda: list_reminders(ctx))
    if isinstance(rem, dict):
        rows = rem.get("reminders", [])
        out["reminders"] = [f"{r['at']}: {r['message']}" for r in rows][:4]
        out["reminders_total"] = len(rows)

    hab = _part("habits", lambda: habit_status(ctx))
    if isinstance(hab, dict):
        out["habits"] = [{"habit": h["habit"], "today": h["today"], "streak": h["streak"]}
                         for h in hab.get("habits", [])][:5]

    from jarvis.tools.contacts import upcoming_birthdays

    bdays = _part("birthdays", lambda: upcoming_birthdays(ctx, 7))
    if isinstance(bdays, dict):
        out["birthdays"] = [f"{b['name']}: " + ("today! 🎂" if b["in_days"] == 0 else
                                                 "tomorrow" if b["in_days"] == 1 else b["date"])
                            for b in bdays["birthdays"]]

    spent = _part("money", lambda: expense_summary(ctx, "today"))
    if isinstance(spent, dict):
        out["spent_today"] = spent.get("total")
    month = _part("budget", lambda: expense_summary(ctx, "month"))
    if isinstance(month, dict) and month.get("budget"):
        out["budget"] = month["budget"]

    parts = [k for k in ("weather", "meetings", "todos", "reminders", "habits", "birthdays", "spent_today") if out.get(k)]
    if not parts:
        out["result"] = ("A clear day: no meetings, to-dos, reminders or habits yet. Suggest adding a to-do or a "
                         "habit to get started.")
    return out
