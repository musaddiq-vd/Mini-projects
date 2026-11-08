"""Habit tracker: daily (or weekday) habits with a per-day target, streaks and a weekly view.

"Make gym a daily habit", "I drank 2 glasses of water", "studied today", "how are my habits going?"
Stored in ~/.jarvis/habits.json.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

from jarvis.dates import parse_day
from jarvis.datafile import store
from jarvis.tools import tool

MAX_HABITS = 30


def habits():
    return store("habits", {"habits": [], "next_id": 1})


def _key(name: str) -> str:
    return " ".join(name.lower().split())


def _find(data: dict, name: str) -> dict | None:
    key = _key(name)
    exact = next((h for h in data["habits"] if _key(h["name"]) == key), None)
    if exact:
        return exact
    close = [h for h in data["habits"] if key and (key in _key(h["name"]) or _key(h["name"]) in key)]
    return close[0] if len(close) == 1 else None


def counts_on(h: dict, day: date) -> bool:
    """Does this habit count on that day (weekday habits skip weekends)?"""
    return h.get("days", "daily") == "daily" or day.weekday() < 5


def done_on(h: dict, day: date) -> bool:
    return h["log"].get(day.isoformat(), 0) >= h.get("target", 1)


def streak(h: dict, today: date | None = None) -> int:
    """Days in a row the target was met (today counts if already met; a not-yet-met today doesn't break it)."""
    day = today or date.today()
    if not done_on(h, day):
        day -= timedelta(days=1)
    n = 0
    created = date.fromisoformat(h.get("created_day", "2000-01-01"))
    while day >= created:
        if counts_on(h, day):
            if not done_on(h, day):
                break
            n += 1
        day -= timedelta(days=1)
    return n


def week(h: dict, today: date | None = None) -> list[tuple[str, int, bool | None]]:
    """Last 7 days: (weekday letter, count, met? / None when it doesn't count that day)."""
    today = today or date.today()
    created = date.fromisoformat(h.get("created_day", "2000-01-01"))
    out = []
    for back in range(6, -1, -1):
        d = today - timedelta(days=back)
        counted = counts_on(h, d) and d >= created  # days before the habit existed don't count as misses
        out.append((d.strftime("%a")[0], h["log"].get(d.isoformat(), 0), done_on(h, d) if counted else None))
    return out


@tool(
    "add_habit",
    "Start tracking a habit. target = how many times a day (e.g. 8 for glasses of water, 1 for gym). "
    "days = 'daily' or 'weekdays'. icon = one emoji.",
    {"name": {"type": "string"}, "target": {"type": "integer"},
     "days": {"type": "string", "enum": ["daily", "weekdays"]}, "icon": {"type": "string"}},
    required=["name"],
)
def add_habit(ctx, name: str, target: int = 1, days: str = "daily", icon: str = "✅"):
    name = " ".join(name.split())[:40]
    if not name:
        return {"error": "The habit needs a name."}
    target = 1 if target is None else int(target)
    if not 1 <= target <= 100:
        return {"error": "The daily target must be between 1 and 100."}
    days = days if days in ("daily", "weekdays") else "daily"
    icon = (icon or "✅").strip()[:2] or "✅"
    st = habits()

    def change(data):
        if _find(data, name) and _key(_find(data, name)["name"]) == _key(name):
            return "exists"
        if len(data["habits"]) >= MAX_HABITS:
            return "full"
        hid = data["next_id"]
        data["next_id"] = hid + 1
        data["habits"].append({"id": hid, "name": name, "target": target, "days": days, "icon": icon,
                               "created": time.time(), "created_day": date.today().isoformat(), "log": {}})
        return hid

    result = st.update(change)
    if result == "exists":
        return {"error": f"'{name}' is already a habit."}
    if result == "full":
        return {"error": f"You can track up to {MAX_HABITS} habits."}
    ctx.undo.push(f"habit {name}", lambda: st.update(lambda d: d.__setitem__(
        "habits", [h for h in d["habits"] if h["id"] != result])))
    per = f"{target} times " if target > 1 else ""
    return f"Now tracking {icon} {name}: {per}{'every day' if days == 'daily' else 'on weekdays'}."


@tool(
    "check_habit",
    "Mark a habit as done (or add to its count): 'gym ho gaya', '2 glass paani piya'. count defaults to 1; "
    "when = 'today' (default) or 'yesterday'/'kal'. Use a negative count to correct a mistake.",
    {"name": {"type": "string"}, "count": {"type": "integer"}, "when": {"type": "string"}},
    required=["name"],
)
def check_habit(ctx, name: str, count: int = 1, when: str = ""):
    day = parse_day(when, past=True)
    if day is None or day > date.today():
        return {"error": "Habits can be checked for today or earlier days."}
    count = int(count or 1)
    st = habits()

    def change(data):
        h = _find(data, name)
        if h is None:
            return None
        key = day.isoformat()
        if key < h.get("created_day", key):
            h["created_day"] = key  # filling in an earlier day: the habit started then
        before = h["log"].get(key, 0)
        h["log"][key] = max(0, min(before + count, 1000))
        return dict(h), before

    found = st.update(change)
    if found is None:
        names = ", ".join(h["name"] for h in habits().load()["habits"]) or "none yet"
        return {"error": f"No habit called '{name}'. Your habits: {names}."}
    h, before = found
    ctx.undo.push(f"check {h['name']}", lambda: st.update(
        lambda d: next(x for x in d["habits"] if x["id"] == h["id"])["log"].__setitem__(day.isoformat(), before)))
    now = h["log"][day.isoformat()]
    target = h.get("target", 1)
    s = streak(h)
    status = "done for the day 🎉" if now >= target else f"{now}/{target}"
    if ctx.events is not None and now >= target > before:
        ctx.events.emit("habit_done", name=h["name"], streak=s)
    return f"{h['icon']} {h['name']}: {status}. Streak: {s} day{'s' if s != 1 else ''}."


@tool("habit_status", "How the user's habits are going: today's progress, streaks and the last 7 days.")
def habit_status(ctx):
    data = habits().load()
    if not data["habits"]:
        return "No habits yet. Example: 'make gym a daily habit' or 'track 8 glasses of water a day'."
    today = date.today()
    rows = []
    for h in data["habits"]:
        week_ = week(h, today)
        rows.append({"habit": f"{h['icon']} {h['name']}", "today": f"{h['log'].get(today.isoformat(), 0)}/{h['target']}",
                     "streak": streak(h, today),
                     "last_7_days": "".join("✓" if met else ("·" if met is None else "✗") for _, _, met in week_)})
    return {"habits": rows}


@tool("delete_habit", "Stop tracking a habit.", {"name": {"type": "string"}}, required=["name"])
def delete_habit(ctx, name: str):
    st = habits()

    def change(data):
        h = _find(data, name)
        if h:
            data["habits"] = [x for x in data["habits"] if x["id"] != h["id"]]
        return h

    h = st.update(change)
    if not h:
        return {"error": f"No habit called '{name}'."}
    ctx.undo.push(f"delete habit {h['name']}", lambda: st.update(lambda d: d["habits"].append(h)))
    return f"Stopped tracking {h['name']}."
