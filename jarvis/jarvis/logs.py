"""Logging to ~/.jarvis/logs/jarvis.log (rotated), plus crash capture for every thread.

When something goes wrong, the log file is the first place to look.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger("jarvis")
_on_crash: list[Callable[[str], None]] = []


def log_dir() -> Path:
    from jarvis.config import home_dir

    path = home_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(level: int = logging.INFO, console: bool = False) -> Path | None:
    """Idempotent. Returns the log file path (None if it can't be written)."""
    if getattr(log, "_jarvis_ready", False):
        return getattr(log, "_jarvis_file", None)
    log.setLevel(level)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)s %(name)s: %(message)s")
    path = None
    try:
        path = log_dir() / "jarvis.log"
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(fmt)
        handler.addFilter(RedactFilter())
        log.addHandler(handler)
    except OSError:
        path = None
    if console and sys.stderr:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        stream.setLevel(logging.WARNING)
        stream.addFilter(RedactFilter())
        log.addHandler(stream)
    _install_hooks()
    log._jarvis_ready = True
    log._jarvis_file = path
    log.info("---- Jarvis starting ----")
    return path


class RedactFilter(logging.Filter):
    """Masks secrets (cards, OTPs, keys...) in every log line, including error details."""

    def filter(self, record: logging.LogRecord) -> bool:
        from jarvis.guardrails import redact

        try:
            message = record.getMessage()
        except Exception:
            return True
        clean = redact(message)[0]
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)[0]
        record.msg, record.args = clean, None
        return True


def on_crash(callback: Callable[[str], None]) -> None:
    """Called with a one-line summary whenever an uncaught error happens (any thread)."""
    _on_crash.append(callback)


def _report(exc_type, exc, tb, where: str) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        return
    log.error("Uncaught error in %s", where, exc_info=(exc_type, exc, tb))
    summary = f"{exc_type.__name__}: {exc}"[:200]
    for cb in list(_on_crash):
        try:
            cb(summary)
        except Exception:
            pass


def _install_hooks() -> None:
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        _report(exc_type, exc, tb, "main thread")
        if previous and previous is not sys.__excepthook__:
            previous(exc_type, exc, tb)

    sys.excepthook = hook
    threading.excepthook = lambda args: _report(
        args.exc_type, args.exc_value, args.exc_traceback, f"thread {args.thread.name if args.thread else '?'}")
