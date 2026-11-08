"""Smart file search: "yesterday's PDF that had an invoice" -> find_files(query='invoice', file_type='pdf',
modified='yesterday').

Looks at file names and, for documents, the text inside. Only inside the allowed folders, never in secret
folders, skips system/app folders, and has a time budget so it can't hang.
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime
from pathlib import Path

from jarvis.dates import parse_day, period_range
from jarvis.tools import tool

TYPES = {
    "pdf": {".pdf"}, "word": {".docx", ".doc", ".odt", ".rtf"}, "excel": {".xlsx", ".xls", ".xlsm", ".csv", ".ods"},
    "powerpoint": {".pptx", ".ppt", ".odp"},
    "image": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".heic"},
    "video": {".mp4", ".mkv", ".mov", ".avi", ".webm"}, "audio": {".mp3", ".wav", ".m4a", ".flac", ".ogg"},
    "text": {".txt", ".md", ".log"}, "code": {".py", ".js", ".ts", ".java", ".c", ".cpp", ".html", ".css", ".json"},
    "archive": {".zip", ".rar", ".7z"},
}
READABLE = {".pdf", ".docx", ".pptx", ".xlsx", ".xlsm", ".txt", ".md", ".csv"}
SKIP_DIRS = {"appdata", "node_modules", "__pycache__", ".git", "venv", ".venv", "site-packages", "$recycle.bin",
             "windows", "program files", "program files (x86)", "programdata", ".cache", "cache", "temp", "tmp"}
WALK_BUDGET_S = 6.0
CONTENT_BUDGET_S = 6.0
MAX_ENTRIES = 80_000
STOP = {"the", "a", "an", "file", "files", "wala", "wali", "jisme", "jismein", "tha", "thi", "mera", "meri", "my",
        "with", "of", "about", "document", "doc", "kal", "aaj", "that", "which", "had", "in", "ka", "ki", "ke"}
_cache: dict[str, tuple[float, list[tuple[Path, float, int]]]] = {}


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[\w]+", text.lower()) if w not in STOP and len(w) > 1]


def search_roots(ctx) -> list[Path]:
    """Usual places first, all inside the allowed folders."""
    home = Path.home()
    preferred = [home / n for n in ("Desktop", "Downloads", "Documents", "Pictures", "OneDrive", "Videos", "Music")]
    roots = []
    for root in ctx.settings.roots:
        inside = [p for p in preferred if p.is_dir() and (p == root or root in p.parents)]
        roots.extend(inside or ([root] if root.is_dir() else []))
    return list(dict.fromkeys(roots))


def _walk(roots: list[Path]) -> list[tuple[Path, float, int]]:
    """(path, modified, size) for files under the roots, within a time budget. Cached for 3 minutes."""
    from jarvis.guardrails import sensitive_path

    key = "|".join(map(str, roots))
    cached = _cache.get(key)
    if cached and time.time() - cached[0] < 180:
        return cached[1]
    started, out = time.monotonic(), []
    for root in roots:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith((".", "$")) and d.lower() not in SKIP_DIRS
                       and not sensitive_path(Path(dirpath) / d)]
            for f in files:
                p = Path(dirpath) / f
                try:
                    st = p.stat()
                except OSError:
                    continue
                out.append((p, st.st_mtime, st.st_size))
            if len(out) >= MAX_ENTRIES or time.monotonic() - started > WALK_BUDGET_S:
                break
    _cache[key] = (time.time(), out)
    return out


def _date_range(modified: str) -> tuple[date, date] | None:
    m = (modified or "").strip().lower()
    if not m or m in {"any", "anytime", "kabhi bhi"}:
        return None
    if m in {"this week", "week", "is hafte", "last 7 days"}:
        start, end, _ = period_range("week")
        return start, end
    if m in {"this month", "month", "is mahine"}:
        start, end, _ = period_range("month")
        return start, end
    if m in {"last month", "pichhle mahine"}:
        start, end, _ = period_range("last_month")
        return start, end
    day = parse_day(m, past=True)
    return (day, day) if day else None


def _snippet(text: str, words: list[str]) -> str:
    low = text.lower()
    for w in words:
        i = low.find(w)
        if i >= 0:
            s = text[max(0, i - 60): i + 80].replace("\n", " ")
            return " ".join(s.split())
    return ""


@tool(
    "find_files",
    "Find files on the user's computer by words in the name OR inside the document, type and when it was "
    "changed: 'kal wala PDF jisme invoice tha' -> query='invoice', file_type='pdf', modified='kal'. "
    "To answer a question about a found file use ask_document with its path; open_file only when the user "
    "wants to see it on screen.",
    {"query": {"type": "string", "description": "key words (not filler words)"},
     "file_type": {"type": "string", "enum": ["any", *TYPES]},
     "modified": {"type": "string", "description": "today, kal/yesterday, this week, this month, last month, a date"},
     "look_inside": {"type": "boolean", "description": "also search the text inside documents (default true)"}},
    slow=True,
)
def find_files(ctx, query: str = "", file_type: str = "any", modified: str = "", look_inside: bool = True):
    words = _words(query)
    exts = TYPES.get(file_type)
    rng = _date_range(modified)
    if modified and rng is None:
        return {"error": f"Couldn't understand '{modified}' as a date."}
    if not words and not exts and not rng:
        return {"error": "Tell me something to look for: a word, a type or when it was saved."}
    roots = search_roots(ctx)
    if not roots:
        return {"error": "There are no folders I'm allowed to search."}
    files = _walk(roots)
    cands = []
    for path, mtime, size in files:
        if exts and path.suffix.lower() not in exts:
            continue
        if rng:
            d = datetime.fromtimestamp(mtime).date()
            if not rng[0] <= d <= rng[1]:
                continue
        name_words = path.stem.lower()
        name_hits = sum(1 for w in words if w in name_words)
        cands.append([path, mtime, size, name_hits * 3, "name" if name_hits else ""])
    # look inside readable documents that didn't match by name (newest first, within a budget)
    if words and look_inside:
        from jarvis.tools.documents import extract_text

        started = time.monotonic()
        for c in sorted((c for c in cands if not c[3] and c[0].suffix.lower() in READABLE and c[2] < 25_000_000),
                        key=lambda c: -c[1])[:200]:
            if time.monotonic() - started > CONTENT_BUDGET_S:
                break
            try:
                text = extract_text(c[0], max_pages=5)[:60_000]
            except Exception:
                continue
            hits = sum(1 for w in words if w in text.lower())
            if hits:
                c[3] = hits * 2
                c[4] = "inside: " + _snippet(text, words)
    if words:
        cands = [c for c in cands if c[3] > 0]
    cands.sort(key=lambda c: (-c[3], -c[1]))
    if not cands:
        where = ", ".join(r.name for r in roots)
        return f"No matching files found in {where}."
    rows = [{"name": c[0].name, "path": str(c[0]), "folder": str(c[0].parent),
             "modified": datetime.fromtimestamp(c[1]).strftime("%d %b %Y %I:%M %p"),
             "size_kb": round(c[2] / 1024), "matched": c[4] or "type/date"} for c in cands[:10]]
    return {"found": len(cands), "files": rows,
            "tip": "Questions about a file -> ask_document(path=...). Showing it on screen -> open_file only if asked."}
