"""One search for everything: files, chats, notes and to-dos, memory, contacts, reminders, meeting notes and the
knowledge base. "Where did I write about the Goa trip last week?" -> search_everything(query='goa trip',
when='last week').

Results come back grouped by kind and ranked (word matches first, then newest), each with a date and a short
snippet. The window shows them as a results card; files open with a click.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

from jarvis.tools import tool

KINDS = ("files", "chats", "notes", "memory", "contacts", "reminders", "meetings", "knowledge")
TITLES = {"files": "Files", "chats": "Chats", "notes": "Notes & to-dos", "memory": "Things I remember",
          "contacts": "Contacts", "reminders": "Reminders", "meetings": "Meeting notes",
          "knowledge": "Knowledge base"}
PER_KIND = 5
STOP = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "is", "was", "what", "where", "did", "we", "i", "me",
        "my", "about", "find", "search", "show", "kya", "ka", "ki", "ke", "tha", "thi", "hai", "maine", "mera",
        "meri", "wala", "wali", "jisme", "kahan", "kaha", "everything", "anything", "for", "with", "that", "it"}
SNIPPET = 140


def _words(query: str) -> list[str]:
    return [w for w in re.findall(r"\w+", (query or "").lower()) if w not in STOP and len(w) > 1]


def _score(text: str, words: list[str]) -> int:
    low = text.lower()
    return sum(low.count(w) for w in words) + 3 * sum(1 for w in set(words) if w in low)


def snippet(text: str, words: list[str], size: int = SNIPPET) -> str:
    text = " ".join(str(text).split())
    low = text.lower()
    at = min((i for i in (low.find(w) for w in words) if i >= 0), default=0)
    start = max(0, at - size // 3)
    piece = text[start:start + size]
    return ("…" if start else "") + piece + ("…" if start + size < len(text) else "")


def _in_range(ts: float | None, rng: tuple[date, date] | None) -> bool:
    if rng is None:
        return True
    if not ts:
        return False
    return rng[0] <= datetime.fromtimestamp(ts).date() <= rng[1]


def _fmt(ts: float | None) -> str:
    if not ts:
        return ""
    d = datetime.fromtimestamp(ts)
    if d.date() == date.today():
        return "today " + d.strftime("%I:%M %p").lstrip("0")
    return d.strftime("%d %b %Y")


def _pick(rows: list[dict], words: list[str]) -> list[dict]:
    """Keep rows that match (all rows when there are no words), best first, then newest."""
    if words:
        rows = [r for r in rows if r["score"] > 0]
    rows.sort(key=lambda r: (-r["score"], -(r.get("ts") or 0)))
    for r in rows:
        r["date"] = _fmt(r.get("ts"))
    return rows


# -- sources -------------------------------------------------------------------------------------------
def _files(ctx, words, rng) -> list[dict]:
    from jarvis.tools.filesearch import match_files, search_roots

    if not words and rng is None:
        return []
    roots = search_roots(ctx)
    if not roots:
        return []
    out = []
    for path, mtime, size, score, how in match_files(roots, words, None, rng, look_inside=bool(words))[:PER_KIND * 2]:
        text = how.removeprefix("inside: ") if how.startswith("inside: ") else str(path.parent)
        out.append({"title": path.name, "snippet": text[:SNIPPET], "ts": mtime, "path": str(path),
                    "score": score or 1})
    return out


def _chats(ctx, words, rng) -> list[dict]:
    from jarvis.config import home_dir
    from jarvis.history import ChatHistory

    out = []
    for item in ChatHistory(home_dir() / "history.jsonl").recent(5000):
        if not _in_range(item.get("t"), rng):
            continue
        text = item["text"]
        who = "You" if item["role"] == "user" else ctx.settings.assistant_name
        out.append({"title": f"{who}: {snippet(text, words, 60)}", "snippet": snippet(text, words),
                    "ts": item.get("t"), "score": _score(text, words)})
    return out


def _notes(ctx, words, rng) -> list[dict]:
    items = list(ctx.state.get("notes", [])) if ctx.state is not None else []
    if ctx.state is None:
        from jarvis.tools import notes

        items = list(notes._mem)
    out = []
    for n in items:
        if not _in_range(n.get("created"), rng):
            continue
        kind = "to-do" if n.get("todo") else "note"
        status = " · done" if n.get("done") else (f" · due {n['due']}" if n.get("due") else "")
        out.append({"title": f"#{n.get('id')} {n.get('text', '')[:70]}", "snippet": f"{kind}{status}",
                    "ts": n.get("created"), "score": _score(n.get("text", ""), words)})
    return out


def _memory(ctx, words, rng) -> list[dict]:
    from jarvis.tools.contacts import PREFIX

    out = []
    for f in ctx.memory.all():
        if f.key.startswith(tuple(PREFIX.values())) or not _in_range(f.updated, rng):
            continue
        text = f"{f.key.replace('_', ' ')}: {f.value}"
        out.append({"title": f.key.replace("_", " "), "snippet": snippet(f.value, words), "ts": f.updated,
                    "score": _score(text, words)})
    return out


def _contacts(ctx, words, rng) -> list[dict]:
    from jarvis.tools.contacts import all_contacts

    if rng is not None:  # contacts have no useful date
        return []
    out = []
    for c in all_contacts(ctx.memory).values():
        details = " · ".join(str(c[k]) for k in ("relation", "phone", "email", "birthday") if c.get(k))
        out.append({"title": c["name"], "snippet": details, "ts": None,
                    "score": _score(f"{c['name']} {details}", words)})
    return out


def _reminders(ctx, words, rng) -> list[dict]:
    items = list(ctx.state.get("reminders", [])) if ctx.state is not None else []
    out = []
    for r in items:
        if not _in_range(r.get("due"), rng):
            continue
        repeat = f" · {r['repeat']}" if r.get("repeat") else ""
        due = datetime.fromtimestamp(r["due"]).strftime("%d %b %I:%M %p") if r.get("due") else ""
        out.append({"title": r.get("message", ""), "snippet": f"due {due}{repeat}", "ts": r.get("due"),
                    "score": _score(r.get("message", ""), words)})
    return out


def _meetings(ctx, words, rng) -> list[dict]:
    """Meeting notes, if Jarvis keeps any: text files in ~/.jarvis/meetings or a meetings.json list."""
    from jarvis.config import home_dir

    home = home_dir()
    out = []
    folder = home / "meetings"
    if folder.is_dir():
        for f in sorted(folder.glob("*"))[-300:]:
            if f.suffix.lower() not in {".md", ".txt", ".json"} or not f.is_file():
                continue
            mtime = f.stat().st_mtime
            if not _in_range(mtime, rng):
                continue
            text = f.read_text(encoding="utf-8", errors="replace")[:50_000]
            title = next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), f.stem)
            out.append({"title": title[:80], "snippet": snippet(text, words), "ts": mtime, "path": str(f),
                        "score": _score(text, words)})
    store = home / "meetings.json"
    if store.is_file():
        try:
            data = json.loads(store.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        items = data.get("meetings", data.get("items", [])) if isinstance(data, dict) else data
        for m in items if isinstance(items, list) else []:
            if not isinstance(m, dict):
                continue
            ts = m.get("start") or m.get("time") or m.get("created") or m.get("t")
            ts = ts if isinstance(ts, (int, float)) else None
            if not _in_range(ts, rng):
                continue
            text = " ".join(str(m.get(k, "")) for k in ("title", "summary", "notes", "transcript", "actions"))
            out.append({"title": str(m.get("title") or "Meeting")[:80], "snippet": snippet(text, words), "ts": ts,
                        "score": _score(text, words)})
    return out


def _knowledge(ctx, words, rng, query: str) -> list[dict]:
    from jarvis import kb

    base = kb.get(ctx)
    if not words or not base.sources():
        return []
    out = []
    for h in base.search(query, k=PER_KIND * 2):
        ts = None
        if h["path"] != kb.NOTES_SOURCE:
            try:
                ts = Path(h["path"]).stat().st_mtime
            except OSError:
                continue
        if rng is not None and not _in_range(ts, rng):
            continue
        out.append({"title": h["file"] + (f" · {h['where']}" if h["where"] else ""),
                    "snippet": snippet(h["text"], words),
                    "ts": ts, "path": None if h["path"] == kb.NOTES_SOURCE else h["path"],
                    "score": round(h["score"] * 10, 2)})
    return out


SOURCES: dict[str, Callable] = {"files": _files, "chats": _chats, "notes": _notes, "memory": _memory,
                                "contacts": _contacts, "reminders": _reminders, "meetings": _meetings}
FILE_KINDS = {"files", "knowledge", "meetings"}


def _parse_kinds(kinds) -> list[str]:
    if not kinds:
        return list(KINDS)
    if isinstance(kinds, str):
        kinds = re.split(r"[,\s]+", kinds)
    alias = {"file": "files", "documents": "files", "docs": "files", "chat": "chats", "history": "chats",
             "conversations": "chats", "note": "notes", "todos": "notes", "to-dos": "notes", "todo": "notes",
             "memories": "memory", "facts": "memory", "contact": "contacts", "people": "contacts",
             "reminder": "reminders", "meeting": "meetings", "kb": "knowledge", "knowledge base": "knowledge"}
    picked = [alias.get(str(k).lower().strip(), str(k).lower().strip()) for k in kinds]
    return [k for k in KINDS if k in picked] or list(KINDS)


@tool(
    "search_everything",
    "Search everywhere at once: files (names and text inside), past chats, notes and to-dos, things you "
    "remember, contacts, reminders, meeting notes and the knowledge base. Use for 'search for X', 'where did I "
    "see/write X', 'X ke baare mein sab dhundo'. when: everyday time words like 'last week', 'kal', 'this month', "
    "'march'. kinds: optional list to narrow it (files, chats, notes, memory, contacts, reminders, meetings, "
    "knowledge). The window shows the results as a card; summarise the best few briefly.",
    {"query": {"type": "string", "description": "key words"},
     "kinds": {"type": "array", "items": {"type": "string", "enum": list(KINDS)}},
     "when": {"type": "string", "description": "e.g. today, kal, last week, this month, last 3 days, march"}},
    slow=True,
)
def search_everything(ctx, query: str = "", kinds=None, when: str = ""):
    from jarvis import permissions
    from jarvis.dates import when_range

    words = _words(query)
    rng = when_range(when) if when else None
    if when and rng is None:
        return {"error": f"Couldn't understand '{when}' as a time. Try 'kal', 'last week' or 'this month'."}
    if not words and rng is None:
        return {"error": "Tell me what to look for: some words, or a time like 'last week'."}
    wanted = _parse_kinds(kinds)
    skipped = []
    files_level = permissions.level(ctx.settings, "files")
    if files_level == permissions.OFF:
        skipped = [k for k in wanted if k in FILE_KINDS]
    elif files_level == permissions.ASK and any(k in FILE_KINDS for k in wanted):
        if not ctx.confirm.confirm(f"Allow Jarvis to search your files for '{query or when}'?"):
            skipped = [k for k in wanted if k in FILE_KINDS]
    groups, started = [], time.monotonic()
    for kind in wanted:
        if kind in skipped:
            continue
        try:
            rows = _knowledge(ctx, words, rng, query) if kind == "knowledge" else SOURCES[kind](ctx, words, rng)
        except Exception as exc:  # one broken source never spoils the whole search
            groups.append({"kind": kind, "title": TITLES[kind], "count": 0, "items": [],
                           "error": f"{type(exc).__name__}: {str(exc)[:80]}"})
            continue
        rows = _pick(rows, words)
        if rows:
            items = [{k: v for k, v in r.items() if k in {"title", "snippet", "date", "path"} and v}
                     for r in rows[:PER_KIND]]
            groups.append({"kind": kind, "title": TITLES[kind], "count": len(rows), "items": items,
                           "best": rows[0]["score"]})
    found = [g for g in groups if g.get("items")]
    found.sort(key=lambda g: -g.get("best", 0))
    for g in found:
        g.pop("best", None)
    total = sum(g["count"] for g in found)
    out = {"search": query, "query": query, "when": when, "found": total, "groups": found,
           "took_s": round(time.monotonic() - started, 1)}
    if rng is not None:
        out["range"] = f"{rng[0]:%d %b %Y} – {rng[1]:%d %b %Y}" if rng[0] != rng[1] else f"{rng[0]:%d %b %Y}"
    if skipped:
        out["not_searched"] = ("files (the user turned file access off or said no; they can allow it in "
                               "Settings > Privacy > Permissions)")
    out["result"] = (f"Found {total} result{'s' if total != 1 else ''}; the card shows them. Summarise the best "
                     "few in a sentence or two." if total else "Nothing found. Say so and offer other words or "
                     "a wider time.")
    return out
