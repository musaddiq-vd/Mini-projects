"""Confirmation gate for irreversible actions (shutdown, restart, ...).

The model can't approve its own action: the answer comes from a person,
through whatever ``ask`` function the front end plugs in.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

AskFn = Callable[[str], bool]


def console_ask(question: str) -> bool:
    try:
        answer = input(f"\n[confirm] {question} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes", "haan", "ha"}


class ConfirmGate:
    def __init__(self, ask: AskFn = console_ask, timeout: float = 60.0):
        self._ask = ask
        self._timeout = timeout
        self._lock = threading.Lock()  # one question at a time

    def confirm(self, question: str) -> bool:
        result: list[bool] = []

        def worker() -> None:
            try:
                result.append(bool(self._ask(question)))
            except Exception:
                result.append(False)

        with self._lock:
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            t.join(self._timeout)
        return bool(result and result[0])
