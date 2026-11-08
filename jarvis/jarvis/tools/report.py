"""The weekly report: spending, habits, goals, tasks, focus and the coming week, as a card in the chat.

Weeks run Monday to Sunday. Every Sunday evening Jarvis offers it by itself (background.py); the user can
also ask any time ("this week's report", "how was last week?").
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

from jarvis.tools import tool


def week_bounds(which: str = "this", today: date | None = None) -> tuple[date, date]:
    today = today or date.today()
    monday = today - timedelta(days=today.weekday())
    if which == "last":
        monday -= timedelta(days=7)
    return monday, monday + timedelta(days=6)


def _ts_in(ts: float, start: date, end: date) -> bool:
    return start <= datetime.fromtimestamp(ts).date() <= end


def _hm(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60}h {m % 60}m" if m >= 60 else f"{m}m"


def build(ctx, which: str = "this", today: date | None = None) -> dict:
    from jarvis.tools.goals import goals
    from jarvis.tools.habits import counts_on, done_on, habits
    from jarvis.tools.money import expenses

    today = today or date.today()
    start, end = week_bounds(which, today)
    upto = min(end, today)  # this week: only the days so far
    prev_start, prev_end = start - timedelta(days=7), start - timedelta(days=7) + (upto - start)
    st = ctx.state

    # money
    items = expenses().load()["items"]
    spent = [e for e in items if start.isoformat() <= e["date"] <= upto.isoformat()]
    before = [e for e in items if prev_start.isoformat() <= e["date"] <= prev_end.isoformat()]
    total, prev_total = sum(e["amount"] for e in spent), sum(e["amount"] for e in before)
    cats: dict[str, float] = {}
    for e in spent:
        cats[e["category"]] = cats.get(e["category"], 0) + e["amount"]
    change = round(100 * (total - prev_total) / prev_total) if prev_total else None

    # habits
    scheduled = met = 0
    best = ("", 0)
    days = [start + timedelta(days=i) for i in range((upto - start).days + 1)]
    for h in habits().load()["habits"]:
        n_met = 0
        for d in days:
            if counts_on(h, d) and d.isoformat() >= h.get("created_day", "0000"):
                scheduled += 1
                if done_on(h, d):
                    met += 1
                    n_met += 1
        if n_met > best[1]:
            best = (h["name"], n_met)
    habit_pct = round(100 * met / scheduled) if scheduled else None

    # goals, tasks, reminders, focus
    steps_done, active = 0, []
    for g in goals().load()["goals"]:
        for s in g["steps"]:
            if s.get("done_at") and _ts_in(s["done_at"], start, upto):
                steps_done += 1
        if not g.get("done"):
            done = sum(1 for s in g["steps"] if s.get("done"))
            active.append({"title": g["title"], "percent": round(100 * done / len(g["steps"])) if g["steps"] else 0})
    notes = st.get("notes", []) if st is not None else []
    tasks = sum(1 for n in notes or [] if n.get("done_at") and _ts_in(n["done_at"], start, upto))
    rem_done = sum(1 for r in (st.get("reminders_done", []) if st is not None else []) or []
                   if _ts_in(r.get("at", 0), start, upto))
    focus = sum(f.get("minutes", 0) for f in (st.get("focus_log", []) if st is not None else []) or []
                if start.isoformat() <= f.get("date", "") <= upto.isoformat())

    # the coming week (calendar)
    meetings_next = None
    try:
        from jarvis.tools import calendar_sync as cal

        if cal.links(ctx.settings):
            nxt_start = datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp()
            if which == "this" and today < end:
                nxt_start = time.time()
            rows = cal.between(ctx.settings, nxt_start, nxt_start + 7 * 86400)
            meetings_next = sum(1 for e in rows if not e["all_day"])
    except Exception:
        pass

    wins = []
    if habit_pct is not None and habit_pct >= 80:
        wins.append(f"habits {habit_pct}% done")
    if best[1] >= 5:
        wins.append(f"{best[0]} on {best[1]} days")
    if change is not None and change <= -10:
        wins.append(f"spent {abs(change)}% less than last week")
    if steps_done:
        wins.append(f"{steps_done} goal step{'s' if steps_done != 1 else ''} done")
    if focus >= 120:
        wins.append(f"{_hm(focus)} of focus")

    label = f"{start:%d %b} – {end:%d %b}" if start.month != end.month else f"{start:%d}–{end:%d %b}"
    return {
        "week": label, "which": which, "currency": getattr(ctx.settings, "currency", "₹"),
        "spent": round(total, 2), "spent_change_percent": change,
        "top_categories": sorted(cats.items(), key=lambda kv: -kv[1])[:3],
        "habits_percent": habit_pct, "best_habit": best[0] if best[1] else "",
        "goal_steps_done": steps_done, "goals": active[:3], "tasks_done": tasks + rem_done,
        "focus": _hm(focus) if focus else "", "meetings_next_week": meetings_next, "wins": wins,
    }


@tool(
    "weekly_report",
    "A report of the user's week (Monday-Sunday): money spent vs last week, habits %, goal steps, tasks, focus "
    "time and next week's meetings. A card is shown in the chat. which: 'this' (default) or 'last'. "
    "Use for 'is hafte ka report', 'weekly summary', 'last week kaisa raha'.",
    {"which": {"type": "string", "enum": ["this", "last"]}},
)
def weekly_report(ctx, which: str = "this"):
    which = which if which in ("this", "last") else "this"
    data = build(ctx, which)
    data["result"] = ("A card with these numbers is on screen. Say a warm 3-4 sentence summary: celebrate the "
                      "wins, mention money briefly, and give ONE small, specific suggestion for next week.")
    return data
