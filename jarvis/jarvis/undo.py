"""One shared undo stack. Tools push a closure that reverses what they did."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class UndoEntry:
    description: str
    revert: Callable[[], None]


class UndoStack:
    def __init__(self, limit: int = 50):
        self._items: list[UndoEntry] = []
        self._limit = limit
        self._lock = threading.Lock()

    def push(self, description: str, revert: Callable[[], None]) -> None:
        with self._lock:
            self._items.append(UndoEntry(description, revert))
            if len(self._items) > self._limit:
                self._items.pop(0)

    def undo(self) -> str:
        with self._lock:
            if not self._items:
                return "Nothing to undo."
            entry = self._items.pop()
        try:
            entry.revert()
        except Exception as exc:  # report, never crash the session
            return f"Could not undo '{entry.description}': {exc}"
        return f"Undone: {entry.description}"

    def __len__(self) -> int:
        return len(self._items)
