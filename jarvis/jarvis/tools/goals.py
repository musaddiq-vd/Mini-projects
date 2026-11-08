"""Goal coach: a big goal broken into small dated steps, with progress and a daily follow-up.

"Mujhe 3 mahine mein Python seekhna hai" -> the model plans steps and calls create_goal; due dates are spread
evenly until the deadline. Stored in ~/.jarvis/goals.json.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

from jarvis.dates import parse_day
from jarvis.datafile import store
from jarvis.tools import tool

MAX_ACTIVE = 10
MAX_STEPS = 20


def goals():
    return store("goals", {"goals": [], "next_id": 1})


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def _find(data: dict, ref: str, active_only: bool = True) -> dict | None:
    pool = [g for g in data["goals"] if not active_only or not g.get("done")]
    key = _key(str(ref))
    if key.isdigit():
        return next((g for g in pool if g["id"] == int(key)), None)
    exact = next((g for g in pool if _key(g["title"]) == key), None)
    if exact:
        return exact
    close = [g for g in pool if key and (key in _key(g["title"]) or _key(g["title"]) in key)]
    if len(close) == 1:
        return close[0]
    return pool[0] if len(pool) == 1 and not key else None


def spread(n: int, start: date, end: date) -> list[str]:
    """n due dates spread evenly from start (exclusive) to end (inclusive)."""
    span = max(1, (end - start).days)
    return [(start + timedelta(days=max(1, round(span * (i + 1) / n)))).isoformat() for i in range(n)]


def progress(g: dict) -> dict:
    steps = g["steps"]
    done = sum(1 for s in steps if s.get("done"))
    today = date.today().isoformat()
    upcoming = [s for s in steps if not s.get("done")]
    nxt = upcoming[0] if upcoming else None
    overdue = [s for s in upcoming if s["due"] < today]
    days_left = (date.fromisoformat(g["deadline"]) - date.today()).days
    return {"done": done, "total": len(steps), "percent": round(100 * done / len(steps)) if steps else 0,
            "next": nxt, "overdue": len(overdue), "days_left": days_left}


@tool(
    "create_goal",
    "Create a goal broken into small, concrete steps (you plan the steps: 3-12 short actions in order). "
    "deadline like '3 months', '31 dec', 'in 6 weeks'. Use for 'mujhe ... karna hai', 'help me reach ...'.",
    {"title": {"type": "string"}, "deadline": {"type": "string"},
     "why": {"type": "string", "description": "the user's reason, if they said one"},
     "steps": {"type": "array", "items": {"type": "string"}}},
    required=["title", "steps"],
)
def create_goal(ctx, title: str, steps: list, deadline: str = "", why: str = ""):
    title = " ".join(str(title).split())[:80]
    steps = [" ".join(str(s).split())[:120] for s in (steps or []) if str(s).strip()]
    if not title:
        return {"error": "The goal needs a title."}
    if not 1 <= len(steps) <= MAX_STEPS:
        return {"error": f"Give between 1 and {MAX_STEPS} steps."}
    end = parse_day(deadline, past=False) if deadline else date.today() + timedelta(days=30)
    if end is None:
        return {"error": f"Couldn't understand the deadline '{deadline}'. Try 'in 3 months' or '31 Dec'."}
    if end <= date.today():
        return {"error": "The deadline must be in the future."}
    st = goals()

    def change(data):
        active = [g for g in data["goals"] if not g.get("done")]
        if len(active) >= MAX_ACTIVE:
            return "full"
        if _find(data, title) and _key(_find(data, title)["title"]) == _key(title):
            return "exists"
        gid = data["next_id"]
        data["next_id"] = gid + 1
        dues = spread(len(steps), date.today(), end)
        data["goals"].append({"id": gid, "title": title, "why": why.strip()[:200], "created": time.time(),
                              "deadline": end.isoformat(), "done": False,
                              "steps": [{"text": t, "due": d, "done": False} for t, d in zip(steps, dues)]})
        return gid

    gid = st.update(change)
    if gid == "full":
        return {"error": f"You already have {MAX_ACTIVE} active goals - finish or delete one first."}
    if gid == "exists":
        return {"error": f"You already have a goal called '{title}'."}
    ctx.undo.push(f"goal {title}", lambda: st.update(lambda d: d.__setitem__(
        "goals", [g for g in d["goals"] if g["id"] != gid])))
    return (f"Goal #{gid} '{title}' created with {len(steps)} steps until {end:%d %b %Y}. "
            f"First step (by {date.fromisoformat(spread(len(steps), date.today(), end)[0]):%d %b}): {steps[0]}. "
            "Encourage the user in one sentence.")


@tool(
    "goal_status",
    "Progress on the user's goals (or one goal): percent done, next step and when it's due, overdue steps, "
    "days left.",
    {"goal": {"type": "string", "description": "title or number; empty = all active goals"}},
)
def goal_status(ctx, goal: str = ""):
    data = goals().load()
    pool = [g for g in data["goals"] if not g.get("done")]
    if goal:
        g = _find(data, goal, active_only=False)
        pool = [g] if g else []
    if not pool:
        return "No active goals. Tell me something you'd like to achieve and I'll break it into steps."
    out = []
    for g in pool:
        p = progress(g)
        out.append({"id": g["id"], "goal": g["title"], "progress": f"{p['done']}/{p['total']} steps ({p['percent']}%)",
                    "next_step": f"{p['next']['text']} (by {p['next']['due']})" if p["next"] else "all done",
                    "overdue_steps": p["overdue"], "days_left": p["days_left"], "why": g.get("why", "")})
    return {"goals": out}


@tool(
    "complete_goal_step",
    "Tick off a step of a goal (by its number, 1 = first, or words from it). Without a step, the next one.",
    {"goal": {"type": "string"}, "step": {"type": "string"}},
    required=["goal"],
)
def complete_goal_step(ctx, goal: str, step: str = ""):
    st = goals()

    def change(data):
        g = _find(data, goal)
        if g is None:
            return None
        todo = [s for s in g["steps"] if not s.get("done")]
        if not todo:
            return g, None
        ref = _key(step)
        if ref.isdigit() and 1 <= int(ref) <= len(g["steps"]):
            target = g["steps"][int(ref) - 1]
        elif ref:
            target = next((s for s in todo if ref in _key(s["text"])), None)
        else:
            target = todo[0]
        if target is None:
            return g, "missing"
        target["done"] = True
        target["done_at"] = time.time()
        if all(s.get("done") for s in g["steps"]):
            g["done"] = True
            g["done_at"] = time.time()
        return dict(g), dict(target)

    res = st.update(change)
    if res is None:
        return {"error": f"No active goal matching '{goal}'."}
    g, target = res
    if target is None:
        return f"All steps of '{g['title']}' are already done."
    if target == "missing":
        return {"error": f"Couldn't find that step in '{g['title']}'. Use its number from goal_status."}
    p = progress(g)
    if ctx.events is not None:
        ctx.events.emit("goal_step", goal=g["title"], step=target["text"], finished=bool(g.get("done")))
    if g.get("done"):
        return f"Goal complete: '{g['title']}'! All {p['total']} steps done. Celebrate big!"
    nxt = f" Next: {p['next']['text']} (by {p['next']['due']})." if p["next"] else ""
    return f"Ticked off '{target['text']}'. '{g['title']}' is {p['percent']}% done.{nxt}"


@tool(
    "add_goal_step",
    "Add a step to an existing goal (optionally with a due date).",
    {"goal": {"type": "string"}, "text": {"type": "string"}, "due": {"type": "string"}},
    required=["goal", "text"],
)
def add_goal_step(ctx, goal: str, text: str, due: str = ""):
    when = parse_day(due, past=False) if due else None
    if due and when is None:
        return {"error": f"Couldn't understand the date '{due}'."}

    def change(data):
        g = _find(data, goal)
        if g is None:
            return None
        if len(g["steps"]) >= MAX_STEPS:
            return "full"
        g["steps"].append({"text": " ".join(text.split())[:120], "due": (when or date.fromisoformat(g["deadline"])).isoformat(),
                           "done": False})
        g["steps"].sort(key=lambda s: (s.get("done", False), s["due"]))
        return g["title"]

    title = goals().update(change)
    if title is None:
        return {"error": f"No active goal matching '{goal}'."}
    if title == "full":
        return {"error": f"A goal can have up to {MAX_STEPS} steps."}
    return f"Added a step to '{title}'."


@tool("delete_goal", "Delete a goal.", {"goal": {"type": "string"}}, required=["goal"])
def delete_goal(ctx, goal: str):
    st = goals()

    def change(data):
        g = _find(data, goal, active_only=False)
        if g:
            data["goals"] = [x for x in data["goals"] if x["id"] != g["id"]]
        return g

    g = st.update(change)
    if not g:
        return {"error": f"No goal matching '{goal}'."}
    ctx.undo.push(f"delete goal {g['title']}", lambda: st.update(lambda d: d["goals"].append(g)))
    return f"Deleted the goal '{g['title']}'."


def checkin_instruction(today: date | None = None) -> str | None:
    """The daily follow-up message for the model, or None when nothing is due."""
    today = today or date.today()
    iso = today.isoformat()
    for g in goals().load()["goals"]:
        if g.get("done"):
            continue
        due = [s for s in g["steps"] if not s.get("done") and s["due"] <= iso]
        if due:
            step = due[0]
            late = " (it's a bit overdue - be kind about it)" if step["due"] < iso else ""
            return (f"(system: goal check-in) Ask the user, in one or two warm sentences, how their goal "
                    f"'{g['title']}' is going: today's step is '{step['text']}'{late}. If they did it, call "
                    "complete_goal_step. Offer a tiny tip to get started.")
    return None
