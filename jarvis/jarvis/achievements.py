"""Achievements: small badges you unlock just by using Jarvis. Saved in state.json.

Listens to the event bus (works in every mode) and emits
    achievement  id, icon, title, desc     when one is unlocked.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

THANKS = ("thank", "thanks", "shukriya", "dhanyavad", "dhanyawad", "शुक्रिया", "धन्यवाद", "شکریہ")
JOKES = ("joke", "mazak", "mazaak", "chutkula", "मज़ाक", "चुटकुला")


@dataclass(frozen=True)
class Badge:
    id: str
    icon: str
    title: str
    desc: str
    counter: str = ""  # count that must reach `goal`
    goal: int = 1


BADGES: list[Badge] = [
    Badge("hello", "👋", "First Words", "Talk to Jarvis for the first time", "messages", 1),
    Badge("chatterbox", "💬", "Chatterbox", "100 messages to Jarvis", "messages", 100),
    Badge("first_action", "⚡", "Getting Things Done", "Jarvis did its first task for you", "actions", 1),
    Badge("power_user", "🦾", "Power User", "100 tasks done by Jarvis", "actions", 100),
    Badge("explorer", "🧭", "Explorer", "Used 15 different abilities", "kinds", 15),
    Badge("early_bird", "🌅", "Early Bird", "Talk to Jarvis before 7 am", "early", 1),
    Badge("night_owl", "🦉", "Night Owl", "Talk to Jarvis after midnight", "night", 1),
    Badge("timer_master", "⏱", "Timer Master", "10 timers finished", "timers", 10),
    Badge("focused", "🎯", "Deep Focus", "Finish a focus session", "focus", 1),
    Badge("list_maker", "📝", "List Maker", "Add 10 notes or to-dos", "notes", 10),
    Badge("done_dusted", "✅", "Done & Dusted", "Tick off 10 to-dos", "ticks", 10),
    Badge("routine", "🔁", "Creature of Habit", "Save your first routine", "routines", 1),
    Badge("shutterbug", "📸", "Shutterbug", "Take 5 screenshots", "shots", 5),
    Badge("polite", "💖", "So Polite", "Say thank you 10 times", "thanks", 10),
    Badge("comedian", "😂", "Comedy Club", "Ask for 5 jokes", "jokes", 5),
    Badge("streak3", "🔥", "On a Roll", "Use Jarvis 3 days in a row", "streak", 3),
    Badge("streak7", "🏅", "Week Warrior", "Use Jarvis 7 days in a row", "streak", 7),
    Badge("safe", "🛡", "Safety First", "Jarvis protected your private info", "guarded", 1),
    Badge("scholar", "🎓", "Perfect Score", "Get every answer right in a practice task", "perfect", 1),
]
BY_ID = {b.id: b for b in BADGES}


class Achievements:
    def __init__(self, state, events, now=datetime.now):
        self.state = state
        self.events = events
        self.now = now
        self._lock = threading.Lock()
        events.subscribe(self._on_event)

    # -- data ---------------------------------------------------------------------------------
    def _data(self) -> dict:
        d = self.state.get("achievements") or {}
        d.setdefault("unlocked", {})
        d.setdefault("counts", {})
        d.setdefault("kinds", [])
        d.setdefault("days", [])
        return d

    def unlocked(self) -> dict[str, float]:
        return dict(self._data()["unlocked"])

    def progress(self) -> list[tuple[Badge, int, float | None]]:
        """(badge, current count, unlocked-at) for every badge, for the gallery."""
        d = self._data()
        out = []
        for b in BADGES:
            count = len(d["kinds"]) if b.counter == "kinds" else d["counts"].get(b.counter, 0)
            out.append((b, min(count, b.goal), d["unlocked"].get(b.id)))
        return out

    # -- counting --------------------------------------------------------------------------------
    def bump(self, counter: str, by: int = 1, set_to: int | None = None) -> list[Badge]:
        """Raise a counter and return badges that just unlocked."""
        with self._lock:
            d = self._data()
            if set_to is not None:
                d["counts"][counter] = max(d["counts"].get(counter, 0), set_to)
            else:
                d["counts"][counter] = d["counts"].get(counter, 0) + by
            new = self._check(d)
            self.state.set("achievements", d)
        self._announce(new)
        return new

    def _check(self, d: dict) -> list[Badge]:
        new = []
        for b in BADGES:
            if b.id in d["unlocked"]:
                continue
            value = len(d["kinds"]) if b.counter == "kinds" else d["counts"].get(b.counter, 0)
            if value >= b.goal:
                d["unlocked"][b.id] = time.time()
                new.append(b)
        return new

    def _announce(self, badges: list[Badge]) -> None:
        for b in badges:
            self.events.emit("achievement", id=b.id, icon=b.icon, title=b.title, desc=b.desc)

    def _touch_day(self) -> None:
        """Count consecutive days of use for the streak badges."""
        today = self.now().date()
        with self._lock:
            d = self._data()
            iso = today.isoformat()
            if iso in d["days"]:
                return
            d["days"] = (d["days"] + [iso])[-30:]
            streak, day = 0, today
            while day.isoformat() in d["days"]:
                streak += 1
                day -= timedelta(days=1)
            d["counts"]["streak"] = max(d["counts"].get("streak", 0), streak)
            d["counts"]["streak_now"] = streak
            new = self._check(d)
            self.state.set("achievements", d)
        self._announce(new)

    def streak(self) -> int:
        d = self._data()
        days = set(d["days"])
        day, n = date.today(), 0
        if day.isoformat() not in days:
            day -= timedelta(days=1)
        while day.isoformat() in days:
            n += 1
            day -= timedelta(days=1)
        return n

    # -- events ------------------------------------------------------------------------------------
    def _on_event(self, kind: str, data: dict) -> None:
        if kind == "user":
            text = (data.get("text") or "").lower()
            if text.startswith("(system"):
                return
            self._touch_day()
            self.bump("messages")
            hour = self.now().hour
            if hour < 7 and hour >= 4:
                self.bump("early")
            if hour < 4:
                self.bump("night")
            if any(w in text for w in THANKS):
                self.bump("thanks")
            if any(w in text for w in JOKES):
                self.bump("jokes")
        elif kind == "tool":
            if "error" in (data.get("result") or {}):
                return
            name = data.get("name", "")
            with self._lock:
                d = self._data()
                if name not in d["kinds"]:
                    d["kinds"].append(name)
                    self.state.set("achievements", d)
            self.bump("actions")
            counter = {"add_note": "notes", "complete_note": "ticks", "save_routine": "routines",
                       "take_screenshot": "shots"}.get(name)
            if counter:
                self.bump(counter)
        elif kind == "timer_end" and data.get("done"):
            self.bump("focus" if str(data.get("label", "")).startswith("Focus") else "timers")
        elif kind == "guardrail":
            self.bump("guarded")
        elif kind == "practice_progress":
            if data.get("total") and data.get("score", 0) >= data["total"]:
                self.bump("perfect")
