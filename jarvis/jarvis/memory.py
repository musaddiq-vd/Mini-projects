"""Long-term memory: facts about the user, stored locally as JSON.

Nothing is deleted to make room. The system prompt gets a small "core"
(most recent facts that fit a character budget) plus a list of the other keys,
so the model knows what it can look up with the ``recall_memory`` tool.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CATEGORIES = ("identity", "preference", "project", "relationship", "note")


@dataclass
class Fact:
    key: str
    value: str
    category: str = "note"
    updated: float = 0.0


def _normalize_key(key: str) -> str:
    key = re.sub(r"[^\w]+", "_", key.strip().lower(), flags=re.UNICODE)
    return key.strip("_")[:64]


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^\w]+", text.lower(), flags=re.UNICODE) if len(w) > 1}


class MemoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._facts: dict[str, Fact] = {}
        self._load()

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for item in raw.get("facts", []):
            try:
                fact = Fact(**item)
            except TypeError:
                continue
            self._facts[fact.key] = fact

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"facts": [asdict(f) for f in self._facts.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # -- public API ----------------------------------------------------------
    def remember(self, key: str, value: str, category: str = "note") -> Fact:
        key = _normalize_key(key)
        if not key or not value.strip():
            raise ValueError("key and value are required")
        if category not in CATEGORIES:
            category = "note"
        with self._lock:
            fact = Fact(key=key, value=value.strip(), category=category, updated=time.time())
            self._facts[key] = fact
            self._save()
        return fact

    def forget(self, key: str) -> bool:
        key = _normalize_key(key)
        with self._lock:
            if key not in self._facts:
                return False
            del self._facts[key]
            self._save()
        return True

    def get(self, key: str) -> Fact | None:
        return self._facts.get(_normalize_key(key))

    def all(self) -> list[Fact]:
        return sorted(self._facts.values(), key=lambda f: f.updated, reverse=True)

    def recall(self, query: str, limit: int = 5) -> list[Fact]:
        """Simple local keyword search over keys and values."""
        q = _words(query.replace("_", " "))
        if not q:
            return self.all()[:limit]
        scored = []
        for fact in self._facts.values():
            key_words = _words(fact.key.replace("_", " "))
            value_words = _words(fact.value)
            score = 2 * len(q & key_words) + len(q & value_words)
            # partial match helps with names and plurals
            if score == 0 and any(w in fact.key or w in fact.value.lower() for w in q):
                score = 0.5
            if score:
                scored.append((score, fact.updated, fact))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [f for _, _, f in scored[:limit]]

    def prompt_block(self, budget: int = 1200) -> str:
        """Text for the system prompt: identity first, then recent facts, then an index."""
        facts = sorted(self._facts.values(), key=lambda f: (f.category != "identity", -f.updated))
        lines, used, left_out = [], 0, []
        for fact in facts:
            line = f"- {fact.key}: {fact.value}"
            if used + len(line) <= budget:
                lines.append(line)
                used += len(line)
            else:
                left_out.append(fact.key)
        if not lines and not left_out:
            return "No saved memories yet."
        text = "Known facts about the user:\n" + "\n".join(lines)
        if left_out:
            text += (
                "\nMore saved facts (use recall_memory to read them): "
                + ", ".join(left_out[:60])
            )
        return text
