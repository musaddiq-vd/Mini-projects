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
    threading.Timer(2.5, ctx.assistant.sleep).start()  # let the goodbye be heard first
    wake = (ctx.settings.wake_word or "").replace("_", " ")
    how = f'say "{wake}"' if wake and getattr(ctx.assistant, "wake", None) else "press Wake or type a message"
    return f"Going to sleep. Tell the user in a few words; to wake you they can {how}."
