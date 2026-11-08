"""Jarvis changes its own window by voice: night/day mode, themes, mini mode, help, achievements."""

from __future__ import annotations

from jarvis.tools import tool

THEME_NAMES = ["Jarvis Blue", "Neon Pink", "Emerald", "Sunset", "Royal Purple", "Mono"]
ACTIONS = ["night_mode", "day_mode", "auto_mode", "theme", "mini_mode", "full_window", "show_help",
           "show_achievements", "clear_chat", "dashboard"]
PAGES = ["week", "habits", "money", "goals"]
SAID = {
    "night_mode": "Night mode is on.", "day_mode": "Day mode is on.",
    "auto_mode": "Auto mode: light by day, dark at night.", "mini_mode": "Mini mode - I'm floating on top now.",
    "full_window": "Back to the full window.", "show_help": "Here's everything I can do.",
    "show_achievements": "Here are your achievements.", "clear_chat": "Chat cleared.",
    "dashboard": "Here's your dashboard.",
}


@tool(
    "jarvis_display",
    "Change YOUR OWN window (not the computer's): night_mode / day_mode / auto_mode for your look, theme to pick "
    "a colour theme, mini_mode to float as a small orb, full_window to come back, show_help for the list of "
    "things you can do, show_achievements for the user's badges, clear_chat to clear the chat view, dashboard (page: week, "
    "habits, money or goals) for charts of their week, habits, spending and goals. "
    "Use for 'night mode karo', 'Sunset theme lagao', 'chhote ho jao', 'help dikhao'. For the computer's own "
    "dark mode use dark_mode instead.",
    {"action": {"type": "string", "enum": ACTIONS},
     "theme": {"type": "string", "enum": THEME_NAMES, "description": "only with action=theme"},
     "page": {"type": "string", "enum": PAGES, "description": "only with action=dashboard"}},
    required=["action"],
)
def jarvis_display(ctx, action: str, theme: str = "", page: str = ""):
    action = (action or "").strip()
    if action not in ACTIONS:
        return {"error": f"Unknown action '{action}'."}
    if action == "theme" and theme not in THEME_NAMES:
        return {"error": f"Pick one of: {', '.join(THEME_NAMES)}."}
    if ctx.events is None:
        return {"error": "That only works in the Jarvis window."}
    ctx.events.emit("ui_command", action=action, theme=page if action == "dashboard" else theme)
    return f"{theme} theme is on." if action == "theme" else SAID[action]
