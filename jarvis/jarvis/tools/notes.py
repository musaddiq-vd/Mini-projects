"""Notes and a to-do list, kept in state.json. "Note karo...", "add milk to my list", "what's on my list?"."""

from __future__ import annotations

import threading
import time
from datetime import datetime

from jarvis.tools import tool

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


def _push_undo(ctx, label: str, before: list[dict]) -> None:
    ctx.undo.push(label, lambda: _save(ctx, before))


@tool(
    "add_note",
    "Save a note or a to-do item for the user ('note karo ...', 'add eggs to my list', "
    "'mujhe kal bank jaana hai, list mein daal do'). todo=true for things to do, false for plain notes.",
    {"text": {"type": "string"}, "todo": {"type": "boolean", "description": "a task to tick off (default true)"}},
    required=["text"],
)
def add_note(ctx, text: str, todo: bool = True):
    text = text.strip()
    if not text:
        return {"error": "The note is empty."}
    with _lock:
        items = _load(ctx)
        before = [dict(n) for n in items]
        nid = max((n["id"] for n in items), default=0) + 1
        items.append({"id": nid, "text": text, "todo": bool(todo), "done": False, "created": time.time()})
        _save(ctx, items)
    _push_undo(ctx, f"add note {nid}", before)
    return f"Added #{nid} to your {'to-do list' if todo else 'notes'}."


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
    rows = [{"id": n["id"], "text": n["text"], "type": "todo" if n.get("todo") else "note",
             "done": n.get("done", False),
             "added": datetime.fromtimestamp(n.get("created", 0)).strftime("%d %b")} for n in items if pick(n)]
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
