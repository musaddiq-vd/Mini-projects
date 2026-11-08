"""Jarvis changes its own window by voice: night/day mode, themes, mini mode, help, achievements."""

from __future__ import annotations

from jarvis.tools import tool

THEME_NAMES = ["Jarvis Blue", "Neon Pink", "Emerald", "Sunset", "Royal Purple", "Mono"]
ACTIONS = ["night_mode", "day_mode", "auto_mode", "theme", "mini_mode", "full_window", "show_help",
           "show_achievements", "clear_chat", "dashboard", "minimize", "maximize", "move", "glide", "place",
           "bigger", "smaller", "stop", "widget_on", "widget_off", "avatar",
           "wallpaper_theme"]
AVATAR_NAMES = ["ai", "robot", "cat", "anime", "minimal"]
DIRECTIONS = ["up", "down", "left", "right"]
PLACES = ["top_left", "top_right", "bottom_left", "bottom_right", "center", "left_half", "right_half", "top",
          "bottom", "left", "right"]
PAGES = ["week", "habits", "money", "goals"]
SAID = {
    "night_mode": "Night mode is on.", "day_mode": "Day mode is on.",
    "auto_mode": "Auto mode: light by day, dark at night.", "mini_mode": "Mini mode - I'm floating on top now.",
    "full_window": "Back to the full window.", "show_help": "Here's everything I can do.",
    "show_achievements": "Here are your achievements.", "clear_chat": "Chat cleared.",
    "dashboard": "Here's your dashboard.", "minimize": "Minimized - I'm still listening.",
    "maximize": "Full screen.", "move": "Moved.", "glide": "Moving - say stop when it's right.",
    "place": "Done.", "bigger": "A bit bigger.", "smaller": "A bit smaller.", "stop": "Stopped.",
    "widget_on": "The desktop widget is on your wallpaper.", "widget_off": "Desktop widget removed.",
    "avatar": "New look!", "wallpaper_theme": "Colours now match your wallpaper.",
}


@tool(
    "jarvis_display",
    "Change YOUR OWN window (not the computer's): night_mode / day_mode / auto_mode for your look, theme to pick "
    "a colour theme, mini_mode to float as a small orb, full_window to come back, show_help for the list of "
    "things you can do, show_achievements for the user's badges, clear_chat to clear the chat view, dashboard (page: week, "
    "habits, money or goals) for charts of their week, habits, spending and goals. "
    "Use for 'night mode karo', 'Sunset theme lagao', 'chhote ho jao', 'help dikhao'. For the computer's own "
    "dark mode use dark_mode instead. "
    "Your window by voice: minimize ('apni window minimize karo'), maximize, move (direction + amount: "
    "'thoda neeche jao'), glide (keep moving in a direction until the user says stop - then call stop_moving), "
    "place (a corner/side: 'right corner me jao', 'beech me aao', left_half/right_half), bigger/smaller. "
    "widget_on / widget_off: the small Jarvis card on the desktop wallpaper ('desktop widget lagao / hatao'); "
    "place also moves it when given with action widget_on. avatar: change your face (ai, robot, cat, anime, "
    "minimal) - 'cat avatar lagao', 'robot ban jao'. wallpaper_theme: take your colours from the desktop "
    "wallpaper ('wallpaper ke rang lagao').",
    {"action": {"type": "string", "enum": ACTIONS},
     "theme": {"type": "string", "enum": THEME_NAMES, "description": "only with action=theme"},
     "page": {"type": "string", "enum": PAGES, "description": "only with action=dashboard"},
     "direction": {"type": "string", "enum": DIRECTIONS, "description": "move / glide"},
     "amount": {"type": "string", "enum": ["little", "some", "lot"], "description": "move (default little)"},
     "speed": {"type": "string", "enum": ["slow", "normal", "fast"], "description": "glide"},
     "place": {"type": "string", "enum": PLACES, "description": "place"},
     "avatar": {"type": "string", "enum": AVATAR_NAMES, "description": "only with action=avatar"}},
    required=["action"],
)
def jarvis_display(ctx, action: str, theme: str = "", page: str = "", direction: str = "", amount: str = "little",
                   speed: str = "normal", place: str = "", avatar: str = ""):
    action = (action or "").strip()
    if action not in ACTIONS:
        return {"error": f"Unknown action '{action}'."}
    if action == "theme" and theme not in THEME_NAMES:
        return {"error": f"Pick one of: {', '.join(THEME_NAMES)}."}
    if action in ("move", "glide") and direction not in DIRECTIONS:
        return {"error": "Say which way: up, down, left or right."}
    if action == "avatar" and avatar not in AVATAR_NAMES:
        return {"error": f"Pick an avatar: {', '.join(AVATAR_NAMES)}."}
    if action == "place" and place not in PLACES:
        return {"error": f"Pick a place: {', '.join(PLACES)}."}
    if ctx.events is None:
        return {"error": "That only works in the Jarvis window."}
    ctx.events.emit("ui_command", action=action, theme=page if action == "dashboard" else theme,
                    direction=direction, amount=amount, speed=speed, place=place, avatar=avatar)
    return f"{theme} theme is on." if action == "theme" else SAID[action]
