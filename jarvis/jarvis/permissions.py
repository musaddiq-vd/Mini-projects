"""Permissions: which kinds of actions Jarvis may do on its own, which need a "yes", and which are off.

Each area (camera, screen, files, messaging, system, smart home, phone) is "on", "ask" or "off". The user sets
them in Settings > Privacy; they are stored in ``Settings.permissions``. Missing areas are "on", except the
camera, which follows the older ``ask_camera`` setting. ``ToolRegistry._run`` enforces them for every call.

The "phone" area is about where a request comes from: it covers every action asked for from the phone.
"""

from __future__ import annotations

ON, ASK, OFF = "on", "ask", "off"
LEVELS = (ON, ASK, OFF)

# area -> (title, what it covers) for the Settings screen
AREAS: dict[str, tuple[str, str]] = {
    "camera": ("Camera", "take a photo with the webcam"),
    "screen": ("Screen", "look at the screen, take screenshots, help with what's on screen"),
    "files": ("Files", "read, write, move, open and delete files and folders"),
    "messaging": ("Messages & typing", "WhatsApp, e-mail, typing, key presses and the clipboard"),
    "system": ("System", "power, WiFi, start with Windows, brightness and dark mode"),
    "smart_home": ("Smart home", "lights, plugs and other smart devices"),
    "phone": ("Phone", "anything you ask for from the phone (Telegram)"),
    "meetings": ("Meetings", "record meetings (microphone and PC sound) and make notes from them"),
}

TOOL_AREAS: dict[str, str] = {
    "look_at_camera": "camera",
    "look_at_screen": "screen", "take_screenshot": "screen", "fix_on_screen": "screen",
    "latest_screenshot": "screen",
    "watch_screen": "screen", "describe_screen": "screen", "click_on": "screen", "computer_task": "screen",
    "read_file": "files", "write_file": "files", "move_file": "files", "delete_file": "files",
    "create_folder": "files", "list_folder": "files", "open_file": "files", "find_files": "files",
    "read_document": "files", "open_document": "files", "ask_document": "files", "explain_code": "files",
    "write_code": "files", "merge_pdfs": "files", "extract_pdf_pages": "files", "rotate_pdf": "files",
    "compress_pdf": "files", "read_image_text": "files",
    "kb_add": "files", "kb_remove": "files", "kb_reindex": "files", "kb_ask": "files", "kb_sources": "files",
    "kb_status": "files",
    "send_whatsapp": "messaging", "draft_email": "messaging", "type_text": "messaging", "press_keys": "messaging",
    "write_clipboard": "messaging", "read_clipboard": "messaging", "clipboard_history": "messaging",
    "copy_from_history": "messaging", "clear_clipboard_history": "messaging",
    "power": "system", "wifi": "system", "autostart": "system", "brightness": "system", "dark_mode": "system",
    "smart_home": "smart_home", "smart_devices": "smart_home",
    # coding assistant
    "open_project": "files", "debug_code": "files", "apply_fix": "files", "review_code": "files",
    "generate_tests": "files", "run_tests": "files", "refactor_code": "files", "code_search": "files",
    "start_meeting": "meetings", "meeting_notes": "meetings", "ask_meeting": "meetings",
    "list_meetings": "meetings", "save_meeting_actions": "meetings",
    "energy_report": "system", "power_mode": "system", "battery_saver_tips": "system",
    "organize_folder": "files",
}

# Tools that change something. With phone set to "ask", only these need a "yes" from the phone.
CHANGING_TOOLS = {
    "write_file", "move_file", "delete_file", "create_folder", "write_code", "open_file", "open_document",
    "send_whatsapp", "draft_email", "type_text", "press_keys", "write_clipboard",
    "power", "wifi", "autostart", "brightness", "dark_mode", "smart_home",
    "open_website", "open_app", "close_app", "window_control", "save_routine", "run_routine", "delete_routine",
    "remember", "forget_memory", "look_at_camera", "look_at_screen", "take_screenshot", "fix_on_screen",
    "volume", "media_control", "play_youtube", "game_launcher", "make_backup", "scroll_screen",
    "merge_pdfs", "extract_pdf_pages", "rotate_pdf", "compress_pdf", "copy_from_history", "clear_clipboard_history",
    "save_contact", "delete_contact", "add_alert", "cancel_alert",
    "watch_screen", "describe_screen", "click_on", "computer_task",
    "debug_code", "apply_fix", "generate_tests", "run_tests",  # running code counts as a change
    "start_meeting", "stop_meeting", "delete_meeting", "save_meeting_actions", "enroll_voice", "delete_voice_profile",
    "kb_add", "kb_remove", "kb_reindex",
    "run_agents", "create_workflow", "edit_workflow", "delete_workflow", "run_workflow", "browse_and_report",
    "power_mode", "organize_folder", "proactive_settings",
}


def level(settings, area: str) -> str:
    """The user's choice for an area: "on", "ask" or "off"."""
    perms = getattr(settings, "permissions", None)
    chosen = perms.get(area) if isinstance(perms, dict) else None
    if chosen in LEVELS:
        return chosen
    if area == "camera":
        return ASK if getattr(settings, "ask_camera", True) else ON
    return ON


def set_level(settings, area: str, value: str) -> None:
    if area not in AREAS or value not in LEVELS:
        raise ValueError(f"unknown permission {area}={value}")
    perms = getattr(settings, "permissions", None)
    perms = dict(perms) if isinstance(perms, dict) else {}
    perms[area] = value
    settings.permissions = perms
    if area == "camera" and value != OFF:
        settings.ask_camera = value == ASK  # keep the older setting in step


def confirms_itself(name: str, args: dict, settings=None) -> bool:
    """True when the tool already asks the user before it does this (so Jarvis doesn't ask twice)."""
    action = str(args.get("action", "")).lower().strip()
    if name in {"send_whatsapp", "delete_file", "apply_fix", "generate_tests", "delete_meeting",
                "delete_voice_profile", "kb_add"}:
        return True
    if name == "look_at_camera":  # asks by itself when the camera is set to "ask"
        return settings is not None and level(settings, "camera") == ASK
    if name == "power":
        return action != "lock"
    if name in {"power_mode", "organize_folder"}:  # power_mode always asks; organize_folder asks before moving
        return True
    if name == "wifi":
        return action == "disconnect"
    return False


def only_checks(args: dict) -> bool:
    """A "status" call only reports; it changes nothing, so it never needs a "yes"."""
    return str(args.get("action", "")).lower().strip() == "status"


def describe_call(name: str, args: dict) -> str:
    """A short, readable description of a tool call for a confirmation question."""
    shown = []
    for key, value in list(args.items())[:3]:
        text = str(value).replace("\n", " ")
        shown.append(f"{key}: {text[:60] + '…' if len(text) > 60 else text}")
    return f"{name.replace('_', ' ')}" + (f" ({'; '.join(shown)})" if shown else "")


def _guest_block(name: str, settings, source: str) -> str | None:
    """An unknown voice (voice profiles) can't use WhatsApp, contacts or money tools unless the owner allows it."""
    if source == "phone" or getattr(settings, "guest_private_tools", False) or not getattr(settings, "voice_id", True):
        return None
    from jarvis import voices

    if name in voices.GUEST_TOOLS and voices.is_guest():
        return ("Not allowed for guests: the voice speaking doesn't match a saved voice profile, so WhatsApp, contacts "
                "and money are off. Tell them kindly. The owner can say a sentence so I recognise them, or allow "
                "guests in Settings > Privacy > Voice profiles.")
    return None


def check(name: str, args: dict, settings, source: str = "pc") -> tuple[str, str | None]:
    """What to do with this call: ("on", None), ("ask", question) or ("off", error message)."""
    area = TOOL_AREAS.get(name)
    phone = level(settings, "phone") if source == "phone" else ON
    if phone == OFF:
        return OFF, ("Not allowed: the user turned off phone control in Settings > Privacy > Permissions. "
                     "Tell them kindly; they can do it on the PC or allow it in Settings.")
    guest = _guest_block(name, settings, source)
    if guest:
        return OFF, guest
    if area is not None and level(settings, area) == OFF:
        title = AREAS[area][0].lower()
        return OFF, (f"Not allowed: the user turned off '{title}' actions for Jarvis in Settings > Privacy > "
                     "Permissions. Tell them kindly and don't try another way; they can allow it there.")
    asking = (area is not None and level(settings, area) == ASK) or (phone == ASK and name in CHANGING_TOOLS)
    if not asking or only_checks(args) or confirms_itself(name, args, settings):
        return ON, None
    return ASK, f"Allow Jarvis to {describe_call(name, args)}?"
