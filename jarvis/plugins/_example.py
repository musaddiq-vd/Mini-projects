"""Example plugin. Copy this file to plugins/my_tool.py (no leading underscore)
and restart Jarvis - it will pick the tool up automatically.

Files starting with "_" are skipped, so this example is not loaded.
"""

import random

from jarvis.tools import tool


@tool(
    "roll_dice",
    "Roll one or more six-sided dice.",
    {"count": {"type": "integer", "description": "How many dice (1-10)"}},
)
def roll_dice(ctx, count: int = 1):
    count = max(1, min(int(count), 10))
    rolls = [random.randint(1, 6) for _ in range(count)]
    return {"rolls": rolls, "total": sum(rolls)}
