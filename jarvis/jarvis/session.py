"""Crash recovery: a small "running" marker (~/.jarvis/session.json).

It is written when Jarvis starts and removed when it quits cleanly. If it is still there on the next start,
the last run ended unexpectedly (a crash, a power cut, the process killed), so Jarvis says so and catches up.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from jarvis.config import home_dir

log = logging.getLogger(__name__)

FILE = "session.json"
CRASH_NOTICE = "⚠ Jarvis closed unexpectedly last time - your reminders and lists are safe."


def marker_path() -> Path:
    return home_dir() / FILE


def begin(path: Path | None = None) -> dict | None:
    """Mark this run as started. Returns what the previous run left behind if it never quit cleanly, else None."""
    path = path or marker_path()
    previous = None
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(previous, dict):
                previous = {}
        except (OSError, ValueError):
            previous = {}  # a half-written marker still means the last run didn't finish
        if previous and _alive(previous.get("pid"), previous.get("started")):
            previous = None  # that run is still closing (a restart), it didn't crash
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("couldn't write the session marker: %s", exc)
    return previous


def end(path: Path | None = None) -> None:
    """A clean exit: remove the marker - only our own, so a run quitting after a restart keeps the new one's."""
    path = path or marker_path()
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("pid") not in (None, os.getpid()):
            return
    except (OSError, ValueError, AttributeError):
        pass
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("couldn't remove the session marker: %s", exc)


def _alive(pid, started) -> bool:
    """Is the process that wrote the marker still running? (Checks its start time too: pids get reused.)"""
    if not isinstance(pid, int) or pid == os.getpid():
        return False
    try:
        import psutil

        proc = psutil.Process(pid)
        return not isinstance(started, (int, float)) or proc.create_time() <= started + 5
    except Exception:  # no such process, or no access to it
        return False


def recover(app, add_system) -> int:
    """After an unexpected exit: show the notice and re-arm reminders at once (missed ones ring now), so they don't
    wait for the voice session to connect. Returns how many reminders were restored. The chat itself is already
    reloaded from history.jsonl when the window opens."""
    add_system(CRASH_NOTICE)
    from jarvis.tools.reminders import restore_reminders

    try:
        return restore_reminders(app.ctx)  # safe to call again later: armed reminders are skipped
    except Exception as exc:
        log.warning("couldn't restore reminders after a crash: %s", exc)
        return 0
