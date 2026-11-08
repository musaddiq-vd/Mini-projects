"""Modules, levels and the day-by-day plan.

A practice day has five tasks: vocabulary, grammar, reading OR listening (alternating), a speaking task
from the chosen focus module, and a bonus task for that module. "Day N" counts the days you practised,
so skipping a day never skips content. The same day always gives the same tasks.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

from jarvis.english import bank

MODULES: dict[str, tuple[str, str, str]] = {  # id -> (icon, title, skill)
    "conversation": ("💬", "Daily conversation", "speaking"),
    "routine": ("🌅", "Daily routine", "speaking"),
    "interview": ("💼", "Interview practice", "speaking"),
    "pronunciation": ("🗣", "Pronunciation", "speaking"),
    "grammar": ("✍", "Grammar & sentences", "grammar"),
    "vocab": ("📚", "Vocabulary", "vocab"),
    "reading": ("📖", "Reading", "reading"),
    "listening": ("🎧", "Listening", "listening"),
}
SKILLS = ("listening", "reading", "speaking", "grammar", "vocab")
LEVELS = {"beginner": 1, "intermediate": 2, "advanced": 3}
SPEAKING_POOLS = {"conversation": bank.CONVERSATION, "routine": bank.ROUTINE, "interview": bank.INTERVIEW,
                  "pronunciation": bank.PRONUNCIATION}

LANGUAGES = ["English", "Hindi", "Urdu", "Bengali", "Marathi", "Gujarati", "Punjabi", "Tamil", "Telugu", "Kannada",
             "Malayalam", "Arabic", "Spanish", "French", "German", "Portuguese", "Italian", "Russian", "Turkish",
             "Japanese", "Chinese", "Korean", "Indonesian"]


@dataclass
class Task:
    id: str
    kind: str  # mcq | reading | listening | speaking
    skill: str
    module: str
    title: str
    icon: str
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(**d)


def _mcq(item) -> dict:
    _, q, options, answer, why = item
    return {"q": q, "options": list(options), "answer": answer, "why": why}


def _pick(pool: list, level: int, n: int, rng: random.Random, day: int) -> list:
    """Items at the learner's level first (then easier ones), rotating with the day so days differ."""
    same = [x for x in pool if x[0] == level]
    easier = [x for x in pool if x[0] < level]
    ordered = same + easier if same else pool
    k = len(ordered)
    start = (day - 1) * n % k if k else 0
    picked = [ordered[(start + i) % k] for i in range(min(n, k))]
    rng.shuffle(picked)
    return picked


def day_plan(day: int, level: str = "beginner", focus: str = "conversation") -> list[Task]:
    lv = LEVELS.get(level, 1)
    focus = focus if focus in MODULES else "conversation"
    rng = random.Random(f"{day}|{lv}|{focus}")
    tasks: list[Task] = []

    tasks.append(Task(f"d{day}-vocab", "mcq", "vocab", "vocab", "Word power - 5 quick questions", "📚",
                      {"questions": [_mcq(x) for x in _pick(bank.VOCAB, lv, 5, rng, day)]}))
    tasks.append(Task(f"d{day}-grammar", "mcq", "grammar", "grammar", "Grammar & sentence structure", "✍",
                      {"questions": [_mcq(x) for x in _pick(bank.GRAMMAR, lv, 5, rng, day)]}))

    if day % 2:
        lvl, title, passage, qs = _pick(bank.READING, lv, 1, rng, day)[0]
        tasks.append(Task(f"d{day}-reading", "reading", "reading", "reading", f"Read: {title}", "📖",
                          {"title": title, "passage": passage, "questions": [_mcq((lvl, *q)) for q in qs]}))
    else:
        lvl, title, script, qs = _pick(bank.LISTENING, lv, 1, rng, day)[0]
        tasks.append(Task(f"d{day}-listening", "listening", "listening", "listening", f"Listen: {title}", "🎧",
                          {"title": title, "script": script, "questions": [_mcq((lvl, *q)) for q in qs]}))

    speak_module = focus if focus in SPEAKING_POOLS else ("conversation", "routine", "interview", "pronunciation")[day % 4]
    prompt = _pick(SPEAKING_POOLS[speak_module], lv, 1, rng, day)[0][1]
    icon, mtitle, _ = MODULES[speak_module]
    tasks.append(Task(f"d{day}-speak", "speaking", "speaking", speak_module, f"Speak: {mtitle}", icon,
                      {"prompt": prompt}))

    # bonus task in the focus module
    if focus in SPEAKING_POOLS:
        other = _pick(SPEAKING_POOLS[focus], lv, 2, random.Random(f"bonus{day}{focus}"), day + 7)
        extra = next((p for _, p in other if p != prompt), other[0][1])
        tasks.append(Task(f"d{day}-bonus", "speaking", "speaking", focus, f"Bonus: {MODULES[focus][1]}", "⭐",
                          {"prompt": extra}))
    elif focus in ("grammar", "vocab"):
        pool = bank.GRAMMAR if focus == "grammar" else bank.VOCAB
        tasks.append(Task(f"d{day}-bonus", "mcq", focus, focus, f"Bonus: {MODULES[focus][1]}", "⭐",
                          {"questions": [_mcq(x) for x in _pick(pool, lv, 5, rng, day + 3)]}))
    else:  # reading / listening focus: the other one too
        if focus == "reading":
            lvl, title, passage, qs = _pick(bank.READING, lv, 1, rng, day + 1)[0]
            tasks.append(Task(f"d{day}-bonus", "reading", "reading", "reading", f"Bonus read: {title}", "⭐",
                              {"title": title, "passage": passage, "questions": [_mcq((lvl, *q)) for q in qs]}))
        else:
            lvl, title, script, qs = _pick(bank.LISTENING, lv, 1, rng, day + 1)[0]
            tasks.append(Task(f"d{day}-bonus", "listening", "listening", "listening", f"Bonus listen: {title}", "⭐",
                              {"title": title, "script": script, "questions": [_mcq((lvl, *q)) for q in qs]}))
    return tasks
