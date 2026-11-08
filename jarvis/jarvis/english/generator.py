"""Exercises for languages other than English, written by the text model and checked before use.

English uses the built-in bank. For e.g. Spanish or Hindi as the target language, each quiz/reading/
listening task is re-created in that language (explanations in the learner's own language). If the
model fails or returns something broken, the English task is kept and marked, so practice never stops.
"""

from __future__ import annotations

import json
import re

from jarvis.english.curriculum import Task

LEVEL_WORDS = {"beginner": "beginner (A1-A2)", "intermediate": "intermediate (B1-B2)", "advanced": "advanced (C1)"}


def _valid_mcq(q: dict) -> bool:
    return (isinstance(q, dict) and isinstance(q.get("q"), str) and isinstance(q.get("options"), list)
            and len(q["options"]) == 4 and all(isinstance(o, str) and o for o in q["options"])
            and isinstance(q.get("answer"), int) and 0 <= q["answer"] < 4 and isinstance(q.get("why", ""), str))


def _parse_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    return json.loads(text[start:] if start >= 0 else text)


def _ask(llm, task: Task, profile: dict) -> dict | None:
    target, native, level = profile["target"], profile["native"], LEVEL_WORDS.get(profile["level"], "beginner")
    if task.kind == "mcq":
        topic = "vocabulary (word meanings, synonyms, everyday words)" if task.skill == "vocab" else \
            "grammar and sentence structure (verb forms, word order, articles, tenses)"
        shape = '{"questions": [{"q": "...", "options": ["a","b","c","d"], "answer": 0, "why": "..."}]}'
        want = f"5 multiple-choice questions to practise {target} {topic}"
    elif task.kind in ("reading", "listening"):
        key = "passage" if task.kind == "reading" else "script"
        style = "a short everyday text (70-110 words)" if task.kind == "reading" else \
            "a short spoken dialogue or announcement (50-90 words) that will be read aloud"
        shape = f'{{"title": "...", "{key}": "...", "questions": [{{"q": "...", "options": ["a","b","c","d"], "answer": 0, "why": "..."}}]}}'
        want = f"{style} in {target}, plus 3 multiple-choice comprehension questions"
    else:
        return None
    prompt = (f"Create {want} for a {level} learner whose own language is {native}. Questions and options in "
              f"{target}; 'why' is a one-sentence explanation in {native}. 'answer' is the index (0-3) of the "
              f"correct option. Reply with JSON only, exactly this shape: {shape}")
    data = _parse_json(llm.ask(prompt))
    qs = data.get("questions") if isinstance(data, dict) else None
    if not qs or not all(_valid_mcq(q) for q in qs):
        return None
    if task.kind == "mcq":
        return {"questions": qs[:5]}
    key = "passage" if task.kind == "reading" else "script"
    if not isinstance(data.get(key), str) or len(data[key]) < 20:
        return None
    return {"title": str(data.get("title") or task.payload.get("title", "")), key: data[key], "questions": qs[:3]}


def make_generator(llm, log=None):
    """Returns generator(tasks, profile) for EnglishStore.plan()."""

    def generate(tasks: list[Task], profile: dict) -> list[Task]:
        out = []
        for task in tasks:
            payload = None
            if task.kind != "speaking":
                try:
                    payload = _ask(llm, task, profile)
                except Exception as exc:  # network, quota, bad JSON
                    if log:
                        log(f"[practice] couldn't create {profile['target']} {task.kind}: {exc}")
            if payload:
                task.payload = payload
                if task.kind in ("reading", "listening"):
                    task.title = f"{'Read' if task.kind == 'reading' else 'Listen'}: {payload.get('title', '')}"
            elif task.kind != "speaking":
                task.payload["fallback"] = True  # shown as English content
            out.append(task)
        return out

    return generate
