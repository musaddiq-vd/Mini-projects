"""Watched news topics and start-with-Windows."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from jarvis.tools import tool

OS = platform.system()
LAUNCHER = Path(__file__).resolve().parents[2] / "jarvis_launcher.pyw"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


@tool(
    "watch_topic",
    "Keep an eye on a news topic (e.g. 'Bitcoin', 'India cricket') and tell the user about new headlines; "
    "or stop watching / list topics.",
    {
        "action": {"type": "string", "enum": ["add", "remove", "list"]},
        "topic": {"type": "string"},
    },
    required=["action"],
)
def watch_topic(ctx, action: str, topic: str = ""):
    topics: list[str] = list(ctx.state.get("watch_topics", []))
    topic = topic.strip()
    if action == "list":
        return {"topics": topics} if topics else "Not watching any topics."
    if not topic:
        return {"error": "Which topic?"}
    if action == "add":
        if topic.lower() not in (t.lower() for t in topics):
            topics.append(topic)
            ctx.state.set("watch_topics", topics)
        return f"Watching '{topic}'. I'll check a few times a day and tell you about new headlines."
    kept = [t for t in topics if t.lower() != topic.lower()]
    ctx.state.set("watch_topics", kept)
    return f"Stopped watching '{topic}'." if len(kept) < len(topics) else f"Wasn't watching '{topic}'."


def _pythonw() -> str:
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return str(w if w.exists() else exe)


def autostart_enabled() -> bool:
    if OS != "Windows":
        return False
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            winreg.QueryValueEx(key, "Jarvis")
            return True
    except OSError:
        return False


def set_autostart(enabled: bool) -> None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        try:
            winreg.DeleteValue(key, "Orion")  # entry from before the rename
        except FileNotFoundError:
            pass
        if enabled:
            winreg.SetValueEx(key, "Jarvis", 0, winreg.REG_SZ, f'"{_pythonw()}" "{LAUNCHER}"')
        else:
            try:
                winreg.DeleteValue(key, "Jarvis")
            except FileNotFoundError:
                pass


@tool(
    "autostart",
    "Make Jarvis start automatically when the computer starts, stop that, or check it.",
    {"action": {"type": "string", "enum": ["on", "off", "status"]}},
    required=["action"],
)
def autostart(ctx, action: str):
    if OS != "Windows":
        return {"error": "Auto-start is only supported on Windows for now."}
    if action == "status":
        return "Auto-start is on." if autostart_enabled() else "Auto-start is off."
    set_autostart(action == "on")
    return f"Auto-start turned {action}."
