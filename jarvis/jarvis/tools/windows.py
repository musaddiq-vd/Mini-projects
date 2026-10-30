"""Close apps and browser tabs, switch/minimize/maximize windows, press keys, type text."""

from __future__ import annotations

import difflib
import platform
import re
import time

import psutil

from jarvis.tools import tool

OS = platform.system()

# What people say -> process name
APP_ALIASES = {
    "chrome": "chrome.exe", "google chrome": "chrome.exe", "edge": "msedge.exe",
    "firefox": "firefox.exe", "brave": "brave.exe", "notepad": "notepad.exe",
    "calculator": "calculatorapp.exe", "file explorer": "explorer.exe", "explorer": "explorer.exe",
    "vs code": "code.exe", "vscode": "code.exe", "visual studio code": "code.exe",
    "word": "winword.exe", "excel": "excel.exe", "powerpoint": "powerpnt.exe",
    "spotify": "spotify.exe", "whatsapp": "whatsapp.exe", "telegram": "telegram.exe",
    "discord": "discord.exe", "steam": "steam.exe", "vlc": "vlc.exe", "paint": "mspaint.exe",
    "task manager": "taskmgr.exe", "settings": "systemsettings.exe",
}
PROTECTED = {"explorer.exe"}  # closing all of these would kill the taskbar; only windows are closed


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _match_windows(query: str):
    from jarvis.tools import _win

    q = _norm(query)
    proc = APP_ALIASES.get(q)
    windows = _win.list_windows()
    by_app = [w for w in windows if w.process.lower() == proc] if proc else []
    if not by_app:
        by_app = [w for w in windows if _norm(w.app_name) == q or (len(q) > 2 and q in _norm(w.app_name))]
    by_title = [w for w in windows if q and q in _norm(w.title)]
    if not by_app and not by_title:
        names = {_norm(w.app_name): w for w in windows}
        close = difflib.get_close_matches(q, list(names), n=1, cutoff=0.75)
        if close:
            by_app = [w for w in windows if _norm(w.app_name) == close[0]]
    return by_app, by_title


def _press(*keys: str) -> None:
    import pyautogui

    pyautogui.hotkey(*keys)


@tool(
    "close_app",
    "Close an app window or a browser tab by name, e.g. 'notepad', 'chrome', 'youtube'. "
    "If the name is a website open in a browser, only that tab is closed.",
    {"name": {"type": "string"}},
    required=["name"],
)
def close_app(ctx, name: str):
    if OS != "Windows":
        return _close_other_os(name)
    from jarvis.tools import _win

    by_app, by_title = _match_windows(name)
    if by_app:
        for w in by_app:
            _win.close(w.hwnd)
        return f"Closed {by_app[0].app_name} ({len(by_app)} window{'s' if len(by_app) > 1 else ''})."
    tabs = [w for w in by_title if w.is_browser]
    if tabs:
        # The browser title shows the active tab, so focus that window and close the tab.
        w = tabs[0]
        _win.focus(w.hwnd)
        _press("ctrl", "w")
        return f"Closed the '{w.title.rsplit(' - ', 1)[0]}' tab."
    if by_title:
        _win.close(by_title[0].hwnd)
        return f"Closed '{by_title[0].title}'."
    return {"error": f"No open window matches '{name}'. Use list_windows to see what is open."}


def _close_other_os(name: str):
    q = _norm(name)
    closed = 0
    for p in psutil.process_iter(["name"]):
        if q and q in _norm(p.info["name"] or ""):
            try:
                p.terminate()
                closed += 1
            except psutil.Error:
                pass
    return f"Closed {closed} process(es)." if closed else {"error": f"Nothing called '{name}' is running."}


@tool("list_windows", "List the open app windows (app name and title).")
def list_windows(ctx):
    if OS != "Windows":
        return {"error": "Only supported on Windows for now."}
    from jarvis.tools import _win

    return {"windows": [{"app": w.app_name, "title": w.title[:80]} for w in _win.list_windows()][:40]}


@tool(
    "window_control",
    "Focus (switch to), minimize, maximize or restore a window by app name or title, "
    "or minimize everything ('show desktop').",
    {
        "action": {"type": "string", "enum": ["focus", "minimize", "maximize", "restore", "show_desktop"]},
        "name": {"type": "string", "description": "App name or part of the title (not needed for show_desktop)"},
    },
    required=["action"],
)
def window_control(ctx, action: str, name: str = ""):
    if OS != "Windows":
        return {"error": "Only supported on Windows for now."}
    from jarvis.tools import _win

    if action == "show_desktop":
        _press("win", "d")
        return "Showing the desktop."
    by_app, by_title = _match_windows(name)
    targets = by_app or by_title
    if not targets:
        return {"error": f"No open window matches '{name}'."}
    w = targets[0]
    if action == "focus":
        _win.focus(w.hwnd)
    else:
        _win.show(w.hwnd, {"minimize": _win.SW_MINIMIZE, "maximize": _win.SW_MAXIMIZE,
                           "restore": _win.SW_RESTORE}[action])
    return f"{action.capitalize()}d {w.app_name}." if action != "focus" else f"Switched to {w.app_name}."


ALLOWED_KEYS = {
    "ctrl", "alt", "shift", "win", "enter", "esc", "tab", "space", "backspace", "delete",
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown", "printscreen",
    *[f"f{i}" for i in range(1, 13)], *"abcdefghijklmnopqrstuvwxyz0123456789",
}


@tool(
    "press_keys",
    "Press a keyboard shortcut in the active window, e.g. 'ctrl+t' (new tab), 'ctrl+w' (close tab), "
    "'ctrl+shift+t', 'alt+tab', 'win+d', 'enter', 'f5'.",
    {"keys": {"type": "string"}},
    required=["keys"],
)
def press_keys(ctx, keys: str):
    parts = [k.strip().lower() for k in keys.replace(" ", "").split("+") if k.strip()]
    parts = ["esc" if k == "escape" else "win" if k in {"windows", "super", "cmd"} else k for k in parts]
    bad = [k for k in parts if k not in ALLOWED_KEYS]
    if not parts or bad:
        return {"error": f"Unknown key(s): {', '.join(bad) or keys}"}
    _press(*parts)
    return f"Pressed {'+'.join(parts)}."


@tool(
    "type_text",
    "Type text into the active window (any language). Set press_enter to send it.",
    {"text": {"type": "string"}, "press_enter": {"type": "boolean"}},
    required=["text"],
)
def type_text(ctx, text: str, press_enter: bool = False):
    import pyperclip

    old = None
    try:
        old = pyperclip.paste()
    except Exception:
        pass
    pyperclip.copy(text)  # pasting handles every language, unlike simulated key presses
    time.sleep(0.1)
    _press("ctrl", "v")
    time.sleep(0.2)
    if press_enter:
        _press("enter")
    if old is not None:
        time.sleep(0.3)
        pyperclip.copy(old)
    return "Typed it."
