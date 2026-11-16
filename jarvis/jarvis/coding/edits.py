"""Proposed code changes: find/replace edits -> a unified diff; applying them with a backup and undo.

The model never writes files directly. It proposes small edits ({"find": exact old text, "replace": new text});
each one must match the file exactly once. Jarvis shows the diff, and only `apply_fix` (after the user says
yes) writes the file, keeping a backup so the change can be undone.
"""

from __future__ import annotations

import difflib
import hashlib
import itertools
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

MAX_EDITS = 12
MAX_PENDING = 20
MAX_DIFF = 12_000
_ids = itertools.count(1)
_NUMBERED = re.compile(r"^\s*\d+ \| ?")


class EditError(Exception):
    pass


@dataclass
class Fix:
    id: str
    path: Path
    rel: str
    old_hash: str
    new_text: str
    diff: str
    title: str
    created: float


_lock = threading.Lock()
_pending: dict[str, Fix] = {}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def apply_edits(text: str, edits: list[dict]) -> str:
    """Apply find/replace edits in order. Each `find` must appear exactly once in the current text."""
    if not edits:
        raise EditError("No change was proposed.")
    newline = "\r\n" if "\r\n" in text else "\n"
    out = text.replace("\r\n", "\n")
    for edit in edits[:MAX_EDITS]:
        find = str(edit.get("find", "")).replace("\r\n", "\n")
        replace = str(edit.get("replace", "")).replace("\r\n", "\n")
        if not find.strip():
            raise EditError("A proposed edit had nothing to find.")
        count = out.count(find)
        if count == 0 and _NUMBERED.match(find):  # the model copied the line numbers it was shown
            find = "\n".join(_NUMBERED.sub("", line, count=1) for line in find.split("\n"))
            replace = "\n".join(_NUMBERED.sub("", line, count=1) for line in replace.split("\n"))
            count = out.count(find)
        if count != 1:
            where = "doesn't appear" if count == 0 else f"appears {count} times"
            raise EditError(f"A proposed edit {where} in the file, so I won't guess where it goes.")
        out = out.replace(find, replace, 1)
    return out.replace("\n", newline) if newline != "\n" else out


def unified(old: str, new: str, rel: str) -> str:
    diff = "".join(difflib.unified_diff(old.replace("\r\n", "\n").splitlines(keepends=True),
                                        new.replace("\r\n", "\n").splitlines(keepends=True),
                                        f"a/{rel}", f"b/{rel}", n=3))
    if len(diff) > MAX_DIFF:
        diff = diff[:MAX_DIFF] + "\n... (diff cut)\n"
    return diff


def diff_stats(diff: str) -> tuple[int, int]:
    lines = diff.splitlines()
    added = sum(1 for x in lines if x.startswith("+") and not x.startswith("+++"))
    removed = sum(1 for x in lines if x.startswith("-") and not x.startswith("---"))
    return added, removed


def propose(path: Path, rel: str, old: str, edits: list[dict], title: str) -> Fix:
    new = apply_edits(old, edits)
    if new == old:
        raise EditError("The proposed change doesn't change anything.")
    fix = Fix(f"fix{next(_ids)}", path, rel, digest(old), new, unified(old, new, rel), title[:200], time.time())
    with _lock:
        _pending[fix.id] = fix
        while len(_pending) > MAX_PENDING:
            _pending.pop(next(iter(_pending)))
    return fix


def get(fix_id: str = "") -> Fix | None:
    """A pending fix by id, or the newest one."""
    with _lock:
        if fix_id:
            return _pending.get(fix_id.strip())
        return next(reversed(_pending.values()), None) if _pending else None


def discard(fix_id: str) -> None:
    with _lock:
        _pending.pop(fix_id, None)


def backup_dir() -> Path:
    from jarvis.config import home_dir

    folder = home_dir() / "code_backups"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def write_with_backup(path: Path, new_text: str) -> Path:
    """Write `new_text` to `path`, first copying the current file to Jarvis's backup folder."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = backup_dir() / f"{stamp}-{next(_ids)}-{path.name}"
    shutil.copy2(path, backup)
    data = new_text.encode("utf-8")
    tmp = path.with_name(path.name + ".jarvis-tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return backup


def restore(backup: Path, path: Path) -> None:
    shutil.copy2(backup, path)
