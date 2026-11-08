"""Notes and a to-do list, kept in state.json. "Note karo...", "add milk to my list", "what's on my list?"."""

from __future__ import annotations

import threading
import time
from datetime import date, datetime

from jarvis.tools import tool

PRIORITIES = ("high", "normal", "low")

KEY = "notes"
_mem: list[dict] = []  # used when there is no state store
_lock = threading.Lock()


def _load(ctx) -> list[dict]:
    return list(ctx.state.get(KEY, [])) if ctx.state is not None else list(_mem)


def _save(ctx, items: list[dict]) -> None:
    if ctx.state is not None:
        ctx.state.set(KEY, items)
    else:
        _mem[:] = items


def _find(items: list[dict], ref) -> dict | None:
    """By number, or by words in the text ('milk')."""
    text = str(ref).strip().lower().lstrip("#")
    if text.isdigit():
        return next((n for n in items if n["id"] == int(text)), None)
    matches = [n for n in items if text and text in n["text"].lower()]
    open_ = [n for n in matches if not n.get("done")]
    return (open_ or matches or [None])[0]


def parse_due(text: str, today: date | None = None) -> str | None:
    """'kal' / 'friday' / '12 nov' / 'kal tak' -> '2026-11-09'. '' -> ''. None when not understood."""
    from jarvis.dates import parse_day

    t = (text or "").strip().lower()
    for tail in (" tak", " by", " se pehle", " before"):
        t = t.removesuffix(tail)
    t = t.removeprefix("by ").removeprefix("before ").strip()
    if not t or t in {"none", "no", "remove", "hatao", "koi nahi"}:
        return ""
    day = parse_day(t, today, past=False)
    return day.isoformat() if day else None


def due_label(iso: str, today: date | None = None) -> str:
    if not iso:
        return ""
    today = today or date.today()
    d = date.fromisoformat(iso)
    diff = (d - today).days
    if diff < 0:
        return f"overdue {-diff} day{'s' if diff < -1 else ''}"
    if diff == 0:
        return "today"
    if diff == 1:
        return "tomorrow"
    if diff < 7:
        return d.strftime("%a")
    return d.strftime("%d %b")


def rank(items: list[dict], today: date | None = None) -> list[tuple[dict, str]]:
    """Open to-dos, most urgent first, each with a short reason."""
    today = today or date.today()
    out = []
    for n in items:
        if not n.get("todo") or n.get("done"):
            continue
        due = n.get("due") or ""
        days = (date.fromisoformat(due) - today).days if due else None
        pr = n.get("priority") or "normal"
        if days is not None and days < 0:
            score, why = 0, f"overdue by {-days} day{'s' if days < -1 else ''}"
        elif days == 0:
            score, why = (1 if pr != "low" else 2), "due today"
        elif pr == "high":
            score, why = 2, "high priority" + (f", due {due_label(due, today)}" if due else "")
        elif days is not None and days <= 2:
            score, why = 3, f"due {due_label(due, today)}"
        elif pr == "low":
            score, why = 6, "low priority"
        else:
            score, why = (4 if due else 5), (f"due {due_label(due, today)}" if due else "no deadline")
        out.append(((score, days if days is not None else 999, n.get("created", 0)), n, why))
    out.sort(key=lambda x: x[0])
    return [(n, why) for _, n, why in out]


def _push_undo(ctx, label: str, before: list[dict]) -> None:
    ctx.undo.push(label, lambda: _save(ctx, before))


@tool(
    "add_note",
    "Save a note or a to-do item for the user ('note karo ...', 'add eggs to my list', "
    "'mujhe kal bank jaana hai, list mein daal do'). todo=true for things to do, false for plain notes. "
    "due: a deadline if they said one ('kal tak', 'friday', '12 nov'); priority high for 'urgent / zaroori'.",
    {"text": {"type": "string"}, "todo": {"type": "boolean", "description": "a task to tick off (default true)"},
     "due": {"type": "string", "description": "deadline: today, kal, friday, 12 nov ..."},
     "priority": {"type": "string", "enum": list(PRIORITIES)}},
    required=["text"],
)
def add_note(ctx, text: str, todo: bool = True, due: str = "", priority: str = "normal"):
    text = text.strip()
    if not text:
        return {"error": "The note is empty."}
    due_iso = parse_due(due)
    if due_iso is None:
        return {"error": f"Couldn't understand the deadline '{due}'. Try 'kal', 'friday' or '12 nov'."}
    priority = priority if priority in PRIORITIES else "normal"
    with _lock:
        items = _load(ctx)
        before = [dict(n) for n in items]
        nid = max((n["id"] for n in items), default=0) + 1
        item = {"id": nid, "text": text, "todo": bool(todo), "done": False, "created": time.time()}
        if todo and due_iso:
            item["due"] = due_iso
        if todo and priority != "normal":
            item["priority"] = priority
        items.append(item)
        _save(ctx, items)
    _push_undo(ctx, f"add note {nid}", before)
    extra = (f", due {due_label(due_iso)}" if todo and due_iso else "") + (
        " (high priority)" if todo and priority == "high" else "")
    return f"Added #{nid} to your {'to-do list' if todo else 'notes'}{extra}."


@tool(
    "list_notes",
    "Read the user's to-do list and notes. show: 'open' (default: unfinished to-dos and notes), "
    "'todo', 'notes', 'done' or 'all'.",
    {"show": {"type": "string", "enum": ["open", "todo", "notes", "done", "all"]}},
)
def list_notes(ctx, show: str = "open"):
    items = _load(ctx)
    pick = {
        "open": lambda n: not n.get("done"),
        "todo": lambda n: n.get("todo") and not n.get("done"),
        "notes": lambda n: not n.get("todo"),
        "done": lambda n: n.get("done"),
        "all": lambda n: True,
    }.get(show, lambda n: not n.get("done"))
    order = {n["id"]: i for i, (n, _) in enumerate(rank(items))}  # urgent to-dos first
    chosen = sorted((n for n in items if pick(n)), key=lambda n: (order.get(n["id"], 10_000), n["id"]))
    rows = []
    for n in chosen:
        row = {"id": n["id"], "text": n["text"], "type": "todo" if n.get("todo") else "note",
               "done": n.get("done", False), "added": datetime.fromtimestamp(n.get("created", 0)).strftime("%d %b")}
        if n.get("due"):
            row["due"] = due_label(n["due"])
        if n.get("priority"):
            row["priority"] = n["priority"]
        rows.append(row)
    if not rows:
        return "Nothing here yet." if show != "done" else "Nothing ticked off yet."
    return {"items": rows, "count": len(rows)}


@tool(
    "complete_note",
    "Tick off a to-do item as done, by its number or a word from it ('milk ho gaya', 'mark 2 done').",
    {"item": {"type": "string", "description": "number or words of the item"}},
    required=["item"],
)
def complete_note(ctx, item: str):
    with _lock:
        items = _load(ctx)
        before = [dict(n) for n in items]
        note = _find(items, item)
        if not note:
            return {"error": f"Couldn't find '{item}' on the list."}
        note["done"] = True
        note["done_at"] = time.time()
        _save(ctx, items)
    _push_undo(ctx, f"complete note {note['id']}", before)
    left = sum(1 for n in items if n.get("todo") and not n.get("done"))
    return f"Ticked off: {note['text']}. {left} left on the list." + (" All done - celebrate!" if not left else "")


@tool(
    "delete_note",
    "Delete a note or to-do item by number or words; item='done' clears all finished items.",
    {"item": {"type": "string"}},
    required=["item"],
)
def delete_note(ctx, item: str):
    with _lock:
        items = _load(ctx)
        before = [dict(n) for n in items]
        if str(item).strip().lower() in {"done", "completed", "finished"}:
            kept = [n for n in items if not n.get("done")]
            removed = len(items) - len(kept)
            _save(ctx, kept)
            msg = f"Cleared {removed} finished item(s)."
        else:
            note = _find(items, item)
            if not note:
                return {"error": f"Couldn't find '{item}'."}
            _save(ctx, [n for n in items if n is not note])
            msg = f"Deleted: {note['text']}."
    _push_undo(ctx, "delete note", before)
    return msg


@tool(
    "update_todo",
    "Change a to-do's deadline or priority ('isko urgent kar do', 'bank wala friday tak', 'deadline hatao').",
    {"item": {"type": "string", "description": "number or words of the item"},
     "due": {"type": "string", "description": "new deadline, or 'none' to remove it"},
     "priority": {"type": "string", "enum": list(PRIORITIES)}},
    required=["item"],
)
def update_todo(ctx, item: str, due: str = "", priority: str = ""):
    with _lock:
        items = _load(ctx)
        before = [dict(n) for n in items]
        note = _find(items, item)
        if not note or not note.get("todo"):
            return {"error": f"Couldn't find the to-do '{item}'."}
        changed = []
        if due:
            iso = parse_due(due)
            if iso is None:
                return {"error": f"Couldn't understand the deadline '{due}'."}
            if iso:
                note["due"] = iso
                changed.append(f"due {due_label(iso)}")
            else:
                note.pop("due", None)
                changed.append("no deadline")
        if priority:
            if priority not in PRIORITIES:
                return {"error": "priority must be high, normal or low."}
            if priority == "normal":
                note.pop("priority", None)
            else:
                note["priority"] = priority
            changed.append(f"{priority} priority")
        if not changed:
            return {"error": "Say what to change: a deadline or a priority."}
        _save(ctx, items)
    _push_undo(ctx, f"update to-do {note['id']}", before)
    return f"{note['text']}: {', '.join(changed)}."


@tool(
    "what_first",
    "What should the user do first? Ranks the open to-dos (overdue, due today, high priority, due soon) and "
    "gives the top ones with reasons. Use for 'kya pehle karun', 'what should I do now', 'plan my day'.",
)
def what_first(ctx):
    ranked = rank(_load(ctx))
    if not ranked:
        return "The to-do list is empty - nothing pending."
    today = date.today().isoformat()
    top = [{"id": n["id"], "text": n["text"], "why": why} for n, why in ranked[:3]]
    overdue = sum(1 for n, _ in ranked if n.get("due") and n["due"] < today)
    due_today = sum(1 for n, _ in ranked if n.get("due") == today)
    return {"do_first": top, "open": len(ranked), "overdue": overdue, "due_today": due_today,
            "tip": "Suggest starting with the first one, in one or two sentences, and offer a focus timer."}


def due_summary(items: list[dict], today: date | None = None) -> tuple[list[dict], list[dict]]:
    """(overdue, due today) open to-dos."""
    today_s = (today or date.today()).isoformat()
    open_ = [n for n in items if n.get("todo") and not n.get("done") and n.get("due")]
    return [n for n in open_ if n["due"] < today_s], [n for n in open_ if n["due"] == today_s]
