"""Audit log of every tool Jarvis runs (~/.jarvis/audit.jsonl).

One line per tool call: when, which tool, where the request came from (the PC or the phone), the arguments
and how it ended (ok, error, blocked or cancelled). Secrets are masked before anything is written, long
values are cut short, and the file is capped in size. It never leaves the computer and is not part of
backups; Settings > Privacy shows it and deletes it.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

MAX_BYTES = 1_000_000  # the oldest half is trimmed away above this
MAX_VALUE = 200  # characters kept of each argument
MAX_RESULT = 160

_lock = threading.Lock()


def audit_path() -> Path:
    from jarvis.config import home_dir

    return home_dir() / "audit.jsonl"


def _short(value):
    if isinstance(value, dict):
        return {str(k): _short(v) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return [_short(v) for v in list(value)[:20]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = str(value)
    return text if len(text) <= MAX_VALUE else text[:MAX_VALUE] + "…"


def outcome(result: dict) -> str:
    """ok / error / blocked / cancelled, from a tool's (already guarded) result."""
    from jarvis.guardrails import BLOCK_PREFIX

    error = result.get("error") if isinstance(result, dict) else None
    if error:
        return "blocked" if str(error).startswith(BLOCK_PREFIX) else "error"
    text = str(result.get("result", "")) if isinstance(result, dict) else ""
    return "cancelled" if text.startswith("Cancelled") else "ok"


def record(tool: str, args: dict, result: dict, source: str = "pc", seconds: float = 0.0) -> None:
    """Add one line. Never raises: the audit log must not break a tool call."""
    try:
        from jarvis.guardrails import redact_deep

        clean_args = _short(redact_deep(args or {})[0])  # mask first: a cut could split a secret in two
        how = outcome(result)
        entry = {"t": round(time.time(), 3), "tool": tool, "source": source or "pc", "args": clean_args,
                 "outcome": how, "ms": round(seconds * 1000)}
        if how != "ok":
            detail = result.get("error") or result.get("result") or ""
            entry["detail"] = str(redact_deep(str(detail))[0])[:MAX_RESULT]
        line = json.dumps(entry, ensure_ascii=False, default=str)
        path = audit_path()
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if path.stat().st_size > MAX_BYTES:
                _trim(path)
    except Exception:  # noqa: BLE001 - a full disk or odd value must never stop Jarvis
        pass


def _trim(path: Path) -> None:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    keep, size = [], 0
    for line in reversed(lines):
        size += len(line.encode("utf-8")) + 1
        if size > MAX_BYTES // 2:
            break
        keep.append(line)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(reversed(keep)) + ("\n" if keep else ""), encoding="utf-8")
    tmp.replace(path)


def recent(n: int = 50) -> list[dict]:
    """The last n entries, oldest first."""
    with _lock:
        try:
            lines = audit_path().read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
    out = []
    for line in lines[-n:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def clear() -> None:
    with _lock:
        try:
            audit_path().unlink(missing_ok=True)
        except OSError:
            pass
