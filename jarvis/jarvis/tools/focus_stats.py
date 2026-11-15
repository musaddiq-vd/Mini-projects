"""Focus stats: how much the user focused today, this week and overall, their focus streak, best day and
what they focused on most."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from jarvis.tools import tool


def _hm(minutes: int) -> str:
    h, m = divmod(int(minutes), 60)
    return f"{h} h {m} min" if h and m else f"{h} h" if h else f"{m} min"


def summarize(log: list[dict], today: date | None = None) -> dict:
    today = today or date.today()
    by_day: dict[str, int] = defaultdict(int)
    by_task: dict[str, int] = defaultdict(int)
    for e in log or []:
        try:
            by_day[e["date"]] += int(e.get("minutes", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if e.get("task"):
            by_task[e["task"].strip().lower()] += int(e.get("minutes", 0))
    week_start = today - timedelta(days=today.weekday())
    week = sum(m for d, m in by_day.items() if week_start.isoformat() <= d <= today.isoformat())
    last_week_start = week_start - timedelta(days=7)
    last_week = sum(m for d, m in by_day.items() if last_week_start.isoformat() <= d < week_start.isoformat())
    streak, day = 0, today if by_day.get(today.isoformat()) else today - timedelta(days=1)
    while by_day.get(day.isoformat()):
        streak += 1
        day -= timedelta(days=1)
    best = max(by_day.items(), key=lambda kv: kv[1]) if by_day else None
    last7 = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
    return {
        "today": by_day.get(today.isoformat(), 0), "week": week, "last_week": last_week,
        "total": sum(by_day.values()), "sessions": len(log or []), "streak_days": streak,
        "best_day": best, "top_tasks": sorted(by_task.items(), key=lambda kv: -kv[1])[:3],
        "last_7_days": [(d.strftime("%a"), by_day.get(d.isoformat(), 0)) for d in last7],
    }


@tool(
    "focus_stats",
    "How much the user focused (focus / pomodoro sessions): today, this week vs last week, streak, best day and "
    "top subjects ('aaj kitna padha?', 'my focus this week'). Be encouraging.",
)
def focus_stats(ctx):
    log = ctx.state.get("focus_log", []) if ctx.state is not None else []
    if not log:
        return "No focus sessions yet. Suggest 'focus mode 25 minutes' to start one."
    s = summarize(log)
    out = {"today": _hm(s["today"]) if s["today"] else "nothing yet", "this_week": _hm(s["week"]),
           "last_week": _hm(s["last_week"]), "streak_days": s["streak_days"], "sessions": s["sessions"],
           "total": _hm(s["total"]),
           "last_7_days": " · ".join(f"{d} {_hm(m) if m else '-'}" for d, m in s["last_7_days"])}
    if s["best_day"]:
        d, m = s["best_day"]
        out["best_day"] = f"{date.fromisoformat(d).strftime('%A %d %B')}: {_hm(m)}"
    if s["top_tasks"]:
        out["top_subjects"] = [f"{t.title()} {_hm(m)}" for t, m in s["top_tasks"]]
    if s["last_week"]:
        change = round((s["week"] - s["last_week"]) * 100 / s["last_week"])
        out["vs_last_week"] = f"{'+' if change >= 0 else ''}{change}%"
    return out
