"""A small JSON file store for personal trackers (expenses, habits, goals...).

- atomic writes (a crash never leaves half a file)
- a corrupt file is moved aside as *.broken-<time>.json and a fresh one starts, so nothing crashes
- one lock per file, so tool threads and the window can use it together
"""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


class DataFile:
    def __init__(self, path: Path, default: dict):
        self.path = Path(path)
        self.default = default
        self.lock = _lock_for(self.path)

    def load(self) -> dict:
        with self.lock:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("not an object")
            except FileNotFoundError:
                return copy.deepcopy(self.default)
            except (OSError, ValueError):
                self._quarantine()
                return copy.deepcopy(self.default)
            for key, value in self.default.items():
                data.setdefault(key, copy.deepcopy(value))
            return data

    def save(self, data: dict) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)

    def update(self, fn) -> Any:
        """Load, let fn change the data (and return something), save. All under the lock."""
        with self.lock:
            data = self.load()
            result = fn(data)
            self.save(data)
            return result

    def exists(self) -> bool:
        return self.path.exists()

    def delete(self) -> None:
        with self.lock:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass

    def _quarantine(self) -> None:
        try:
            self.path.replace(self.path.with_name(f"{self.path.stem}.broken-{int(time.time())}.json"))
        except OSError:
            pass


def store(name: str, default: dict) -> DataFile:
    from jarvis.config import home_dir

    return DataFile(home_dir() / f"{name}.json", default)
