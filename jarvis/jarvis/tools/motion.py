"""Scrolling the screen by voice: "scroll down", "page down", "keep scrolling slowly" ... "stop".

Scrolls the window the user is looking at (the top window that isn't Jarvis) with arrow / page keys, which
browsers, PDFs, Word and editors all understand. Continuous scrolling runs until stop_moving, the user switches
to another window, or 90 seconds pass.
"""

from __future__ import annotations

import os
import platform
import threading
import time

from jarvis.tools import tool

OS = platform.system()
KEYS = {"up": "up", "down": "down", "left": "left", "right": "right"}
INTERVAL = {"slow": 0.2, "normal": 0.1, "fast": 0.045}
MAX_SCROLL_S = 90
_stop = threading.Event()
_running: dict[str, threading.Thread] = {}
_lock = threading.Lock()


def _press(key: str, times: int = 1) -> None:
    import pyautogui

    for _ in range(times):
        pyautogui.press(key, _pause=False)


def target_window():
    """The window the user is reading: the top-most visible window that isn't Jarvis."""
    from jarvis.tools import _win

    me = os.getpid()
    fg = _win.foreground()
    if fg and fg.pid != me:
        return fg
    for w in _win.list_windows():  # top to bottom
        if w.pid != me:
            return w
    return None


def scrolling() -> bool:
    t = _running.get("scroll")
    return bool(t and t.is_alive())


def stop_scrolling() -> bool:
    was = scrolling()
    _stop.set()
    return was


@tool(
    "scroll_screen",
    "Scroll the page/app the user is looking at (not Jarvis): 'neeche scroll karo', 'thoda upar', 'page down', "
    "'sabse upar jao'. keep_going=true scrolls slowly until the user says stop (then call stop_moving).",
    {"direction": {"type": "string", "enum": ["down", "up", "left", "right", "top", "bottom"]},
     "amount": {"type": "string", "enum": ["little", "page", "lot"], "description": "default little"},
     "keep_going": {"type": "boolean", "description": "keep scrolling until stop ('chalte raho', 'dheere dheere')"},
     "speed": {"type": "string", "enum": ["slow", "normal", "fast"]}},
    required=["direction"],
)
def scroll_screen(ctx, direction: str, amount: str = "little", keep_going: bool = False, speed: str = "normal"):
    if OS != "Windows":
        return {"error": "Scrolling by voice works on Windows for now."}
    from jarvis.tools import _win

    if direction not in ("down", "up", "left", "right", "top", "bottom"):
        return {"error": "direction must be up, down, left, right, top or bottom."}
    w = target_window()
    if w is None:
        return {"error": "There's no window to scroll."}
    stop_scrolling()
    with _lock:
        old = _running.get("scroll")
    if old and old.is_alive():
        old.join(timeout=1)
    _win.focus(w.hwnd)
    time.sleep(0.15)
    if direction in ("top", "bottom"):
        _press("home" if direction == "top" else "end")
        return {"result": f"Went to the {direction} of {w.app_name}."}
    key = KEYS[direction]
    if not keep_going:
        if amount == "page" and direction in ("up", "down"):
            _press("pageup" if direction == "up" else "pagedown")
        elif amount == "lot" and direction in ("up", "down"):
            _press("pageup" if direction == "up" else "pagedown", 3)
        else:
            _press(key, 3 if amount == "little" else 8)
        return {"result": f"Scrolled {w.app_name} {direction}."}

    _stop.clear()
    interval = INTERVAL.get(speed, INTERVAL["normal"])

    def run():
        end = time.monotonic() + MAX_SCROLL_S
        while not _stop.is_set() and time.monotonic() < end:
            fg = _win.foreground()
            if fg is not None and fg.hwnd != w.hwnd:
                if fg.pid != os.getpid():
                    break  # the user went to another app
                _stop.wait(interval)  # Jarvis is in front: pause, never type into its own window
                continue
            try:
                _press(key)
            except Exception:
                break
            _stop.wait(interval)
        if ctx.events is not None:
            ctx.events.emit("motion", what="scroll", state="stopped")

    t = threading.Thread(target=run, daemon=True, name="jarvis-scroll")
    with _lock:
        _running["scroll"] = t
    t.start()
    if ctx.events is not None:
        ctx.events.emit("motion", what="scroll", state="running")
    return {"result": f"Scrolling {w.app_name} {direction}. Say stop when you want it to stop.",
            "tip": "When the user says ruko / bas / stop, call stop_moving right away."}


@tool(
    "stop_moving",
    "Stop scrolling or moving right now ('ruko', 'bas', 'stop', 'thamo') - stops the screen scroll and your own "
    "window gliding.",
)
def stop_moving(ctx):
    was = stop_scrolling()
    if ctx.events is not None:
        ctx.events.emit("ui_command", action="stop")
    return "Stopped." if was else "Stopped (nothing was scrolling)."
