"""Practice progress and preferences in ~/.jarvis/english.json.

Remembers: your language and the one you learn, level, focus module, whether you were in practice
mode, today's plan (so it stays the same all day), results per task, skill accuracy, XP, streak and
the tutor's latest corrections. Next time Jarvis opens, you continue right where you left off.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import date, timedelta
from pathlib import Path

from jarvis.english.curriculum import LEVELS, MODULES, SKILLS, Task, day_plan

XP_PER_CORRECT = 10
XP_TASK_BONUS = 15


def _default() -> dict:
    return {
        "profile": {"native": "Hindi", "target": "English", "level": "beginner", "focus": "conversation"},
        "mode": "agent",
        "xp": 0,
        "streak": 0,
        "best_streak": 0,
        "last_active": None,
        "active_days": [],  # ISO dates with at least one finished task
        "plans": {},  # ISO date -> {"key": ..., "day": n, "tasks": [...]}
        "results": {},  # ISO date -> {task_id: {"score", "total", "at"}}
        "skills": {s: [0, 0] for s in SKILLS},  # skill -> [correct, total]
        "current_task": None,
        "corrections": [],
    }


class EnglishStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        data = _default()
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                data.update(saved)
                data["profile"] = {**_default()["profile"], **saved.get("profile", {})}
                data["skills"] = {**_default()["skills"], **saved.get("skills", {})}
        except (OSError, json.JSONDecodeError):
            pass
        self.data = data

    def save(self) -> None:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.path)
            except OSError:
                pass

    # -- preferences ----------------------------------------------------------------
    @property
    def profile(self) -> dict:
        return self.data["profile"]

    def set_profile(self, **changes) -> None:
        with self._lock:
            for key, value in changes.items():
                if key == "level" and value not in LEVELS:
                    continue
                if key == "focus" and value not in MODULES:
                    continue
                if value:
                    self.data["profile"][key] = value
            self.save()

    @property
    def mode(self) -> str:
        return self.data.get("mode", "agent")

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self.data["mode"] = "practice" if mode == "practice" else "agent"
            self.save()

    # -- days and plans -----------------------------------------------------------------
    def day_number(self, today: date | None = None) -> int:
        today = today or date.today()
        days = self.data["active_days"]
        return len(days) + (0 if today.isoformat() in days else 1)

    def _plan_key(self) -> str:
        p = self.profile
        return f"{p['target']}|{p['level']}|{p['focus']}"

    def plan(self, today: date | None = None, generator=None) -> list[Task]:
        """Today's tasks; built once per day (and again if level, focus or language change)."""
        today = today or date.today()
        iso = today.isoformat()
        with self._lock:
            cached = self.data["plans"].get(iso)
            if cached and cached.get("key") == self._plan_key():
                return [Task.from_dict(t) for t in cached["tasks"]]
            day = self.day_number(today)
            tasks = day_plan(day, self.profile["level"], self.profile["focus"])
            if generator is not None and self.profile["target"] != "English":
                tasks = generator(tasks, self.profile)
            self.data["plans"] = {k: v for k, v in self.data["plans"].items() if k >= (today - timedelta(days=7)).isoformat()}
            self.data["plans"][iso] = {"key": self._plan_key(), "day": day, "tasks": [t.to_dict() for t in tasks]}
            self.save()
            return tasks

    def task(self, task_id: str, today: date | None = None) -> Task | None:
        return next((t for t in self.plan(today) if t.id == task_id), None)

    def set_current(self, task_id: str | None) -> None:
        with self._lock:
            self.data["current_task"] = task_id
            self.save()

    # -- results ----------------------------------------------------------------------
    def result(self, task_id: str, today: date | None = None) -> dict | None:
        return self.data["results"].get((today or date.today()).isoformat(), {}).get(task_id)

    def done_today(self, today: date | None = None) -> set[str]:
        return set(self.data["results"].get((today or date.today()).isoformat(), {}))

    def record(self, task_id: str, skill: str, score: float, total: float, today: date | None = None) -> dict:
        """Save a finished task. Returns {'xp_gained', 'streak', 'first_time'}."""
        today = today or date.today()
        iso = today.isoformat()
        total = max(1.0, float(total))
        score = max(0.0, min(float(score), total))
        with self._lock:
            day_results = self.data["results"].setdefault(iso, {})
            first_time = task_id not in day_results
            previous = day_results.get(task_id, {}).get("score", 0)
            day_results[task_id] = {"score": score, "total": total, "at": time.time()}
            gained = 0
            if first_time:
                gained = int(score * XP_PER_CORRECT if total > 1 else score * 50) + XP_TASK_BONUS
                stat = self.data["skills"].setdefault(skill if skill in SKILLS else "speaking", [0, 0])
                stat[0] += score
                stat[1] += total
            elif score > previous:  # a better retry earns the difference
                gained = int((score - previous) * XP_PER_CORRECT)
            self.data["xp"] += gained
            self._touch_streak(today)
            self.save()
            return {"xp_gained": gained, "streak": self.data["streak"], "first_time": first_time}

    def _touch_streak(self, today: date) -> None:
        iso = today.isoformat()
        last = self.data.get("last_active")
        if last == iso:
            return
        if last == (today - timedelta(days=1)).isoformat():
            self.data["streak"] += 1
        else:
            self.data["streak"] = 1
        self.data["best_streak"] = max(self.data.get("best_streak", 0), self.data["streak"])
        self.data["last_active"] = iso
        if iso not in self.data["active_days"]:
            self.data["active_days"].append(iso)

    def streak(self, today: date | None = None) -> int:
        """Current streak (0 if the last practice was before yesterday)."""
        today = today or date.today()
        last = self.data.get("last_active")
        if last in (today.isoformat(), (today - timedelta(days=1)).isoformat()):
            return self.data["streak"]
        return 0

    # -- corrections ------------------------------------------------------------------
    def add_correction(self, said: str, better: str, tip: str = "") -> None:
        with self._lock:
            self.data["corrections"] = (self.data["corrections"] + [
                {"said": said, "better": better, "tip": tip, "at": time.time()}])[-50:]
            self.save()

    # -- summary ------------------------------------------------------------------------
    def summary(self, today: date | None = None) -> dict:
        tasks = self.plan(today)
        done = self.done_today(today)
        skills = {s: (round(100 * c / t) if t else None) for s, (c, t) in self.data["skills"].items()}
        return {
            "day": self.day_number(today),
            "done": len([t for t in tasks if t.id in done]),
            "total": len(tasks),
            "xp": self.data["xp"],
            "level_name": self.profile["level"],
            "streak": self.streak(today),
            "best_streak": self.data.get("best_streak", 0),
            "practice_days": len(self.data["active_days"]),
            "skills": skills,
        }
