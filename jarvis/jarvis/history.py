"""Conversation history on disk (~/.jarvis/history.jsonl), so the chat survives a restart."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

MAX_LINES = 1000  # older lines are trimmed away
MAX_BYTES = 1_500_000


class ChatHistory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def add(self, role: str, text: str) -> None:
        """role: "user" or "assistant"."""
        text = (text or "").strip()
        if not text or text.startswith("(system"):
            return
        line = json.dumps({"t": time.time(), "role": role, "text": text}, ensure_ascii=False)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                if self.path.stat().st_size > MAX_BYTES:
                    self._trim()
            except OSError:
                pass  # history is a convenience; never break the conversation over it

    def _trim(self) -> None:
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()[-MAX_LINES // 2:]
        size = sum(len(line.encode("utf-8")) + 1 for line in lines)
        while lines and size > MAX_BYTES // 2:  # long messages: trim by size too
            size -= len(lines.pop(0).encode("utf-8")) + 1
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def recent(self, n: int = 30) -> list[dict]:
        with self._lock:
            try:
                lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                return []
        out = []
        for line in lines[-n:]:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and item.get("text"):
                out.append(item)
        return out

    def clear(self) -> None:
        with self._lock:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
