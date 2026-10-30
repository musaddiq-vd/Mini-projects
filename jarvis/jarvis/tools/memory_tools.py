"""Let the model save, look up and forget facts about the user."""

from __future__ import annotations

from dataclasses import asdict

from jarvis.memory import CATEGORIES
from jarvis.tools import tool


@tool(
    "remember",
    "Save a lasting fact about the user (name, city, preferences, projects, people). "
    "Use short English keys like 'city' or 'sister_name'.",
    {
        "key": {"type": "string"},
        "value": {"type": "string"},
        "category": {"type": "string", "enum": list(CATEGORIES)},
    },
    required=["key", "value"],
)
def remember(ctx, key: str, value: str, category: str = "note"):
    previous = ctx.memory.get(key)
    fact = ctx.memory.remember(key, value, category)

    def revert():
        if previous:
            ctx.memory.remember(previous.key, previous.value, previous.category)
        else:
            ctx.memory.forget(fact.key)

    ctx.undo.push(f"remember {fact.key}", revert)
    return f"Saved {fact.key}."


@tool(
    "recall_memory",
    "Search saved facts about the user. Use before saying you don't know something personal.",
    {"query": {"type": "string"}},
    required=["query"],
)
def recall_memory(ctx, query: str):
    facts = ctx.memory.recall(query)
    if not facts:
        return "Nothing saved about that."
    return {"facts": [{k: v for k, v in asdict(f).items() if k != "updated"} for f in facts]}


@tool(
    "forget_memory",
    "Delete a saved fact about the user by its key.",
    {"key": {"type": "string"}},
    required=["key"],
)
def forget_memory(ctx, key: str):
    return f"Forgot {key}." if ctx.memory.forget(key) else f"No saved fact called '{key}'."
