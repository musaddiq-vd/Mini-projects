"""Control Jarvis itself (not the computer)."""

from __future__ import annotations

import threading

from jarvis.tools import tool


@tool(
    "assistant_sleep",
    "Put YOURSELF (the assistant) to sleep: stop listening until the wake word or the Wake button. "
    "Use for 'sleep', 'so jao', 'go to sleep', 'stop listening', 'chup ho jao', 'bye for now' "
    "when said to you. This never touches the computer's power.",
)
def assistant_sleep(ctx):
    if ctx.assistant is None:
        return {"error": "Sleeping is only available in voice mode."}
    threading.Timer(2.5, lambda: ctx.assistant.sleep("voice")).start()  # let the goodbye be heard first
    how = ctx.assistant.how_to_wake() if hasattr(ctx.assistant, "how_to_wake") else "press Wake"
    return f"Going to sleep. Tell the user in a few words; to wake you they can {how}."
