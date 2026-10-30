"""Tiny publish/subscribe bus between the assistant core and the front end.

Event kinds and their data:
    state     state="connecting|listening|speaking|thinking|sleeping|offline"
    user      text=...              (what the user said or typed, one turn)
    assistant text=...              (what Jarvis said, one turn)
    tool      name=..., args=..., result=...
    level     source="mic|speaker", value=0..1
    log       text=...              (status lines, warnings)
    confirm   question=..., reply=callable(bool)
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable

Handler = Callable[[str, dict], None]


class Events:
    def __init__(self) -> None:
        self._subs: list[Handler] = []
        self._lock = threading.Lock()

    def subscribe(self, handler: Handler) -> None:
        with self._lock:
            self._subs.append(handler)

    def emit(self, kind: str, **data) -> None:
        with self._lock:
            subs = list(self._subs)
        for handler in subs:
            try:
                handler(kind, data)
            except Exception:
                traceback.print_exc()


def console_printer(assistant_name: str) -> Handler:
    """Prints events for the terminal modes."""
    me = assistant_name.lower()

    def handle(kind: str, data: dict) -> None:
        if kind == "user":
            print(f"you   > {data['text']}")
        elif kind == "assistant":
            print(f"{me:5} > {data['text']}")
        elif kind == "tool":
            print(f"   [tool] {data['name']} {data.get('args', {})}")
        elif kind == "log":
            print(data["text"])
        elif kind == "state" and data["state"] == "sleeping":
            print("[sleeping]")

    return handle
