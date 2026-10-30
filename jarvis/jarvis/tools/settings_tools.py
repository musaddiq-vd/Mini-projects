"""Screen brightness, dark mode and WiFi."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess

from jarvis.tools import tool

OS = platform.system()
NO_WINDOW = 0x08000000 if OS == "Windows" else 0  # don't flash a console window


def _run(cmd: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)


def _powershell(script: str) -> str:
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
    return out.stdout.strip()


# -- brightness ---------------------------------------------------------------
def get_brightness() -> int | None:
    if OS == "Windows":
        value = _powershell("(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness")
        return int(value.split()[0]) if value[:1].isdigit() else None
    if OS == "Linux" and shutil.which("brightnessctl"):
        cur, mx = _run(["brightnessctl", "g"]).stdout, _run(["brightnessctl", "m"]).stdout
        return round(int(cur) * 100 / int(mx)) if cur.strip().isdigit() and mx.strip().isdigit() else None
    return None


def set_brightness_level(level: int) -> bool:
    level = max(0, min(100, int(level)))  # an int, so nothing can be injected into the script
    if OS == "Windows":
        _powershell(
            "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
            f".WmiSetBrightness(1,{level})"
        )
        return get_brightness() is not None
    if OS == "Linux" and shutil.which("brightnessctl"):
        return _run(["brightnessctl", "s", f"{level}%"]).returncode == 0
    if OS == "Darwin" and shutil.which("brightness"):
        return _run(["brightness", str(level / 100)]).returncode == 0
    return False


@tool(
    "brightness",
    "Set screen brightness: a level 0-100, or 'up'/'down' by 10. Laptop screens only.",
    {"level": {"type": "string", "description": "0-100, 'up' or 'down'"}},
    required=["level"],
)
def brightness(ctx, level: str):
    current = get_brightness()
    text = str(level).strip().lower()
    if text in {"up", "down"}:
        if current is None:
            return {"error": "Can't read the brightness of this screen (external monitors aren't supported)."}
        target = current + (10 if text == "up" else -10)
    elif re.fullmatch(r"\d{1,3}%?", text):
        target = int(text.rstrip("%"))
    else:
        return {"error": "level must be 0-100, 'up' or 'down'."}
    target = max(0, min(100, target))
    if not set_brightness_level(target):
        return {"error": "Couldn't change the brightness on this screen."}
    if current is not None:  # undo only when we know the real previous value
        ctx.undo.push(f"brightness {target}%", lambda: set_brightness_level(current))
    return f"Brightness set to {target}%."


# -- dark mode ----------------------------------------------------------------
_THEME_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"


def _get_light() -> int | None:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _THEME_KEY) as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0]
    except OSError:
        return None


def _set_light(value: int) -> None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _THEME_KEY, 0, winreg.KEY_SET_VALUE) as key:
        for name in ("AppsUseLightTheme", "SystemUsesLightTheme"):
            winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, value)


@tool(
    "dark_mode",
    "Turn Windows dark mode on or off, or toggle it.",
    {"mode": {"type": "string", "enum": ["on", "off", "toggle"]}},
    required=["mode"],
)
def dark_mode(ctx, mode: str):
    if OS != "Windows":
        return {"error": "Dark mode switching is only supported on Windows for now."}
    before = _get_light()
    if before is None:
        return {"error": "Couldn't read the current theme."}
    light = {"on": 0, "off": 1}.get(mode, 1 - before)
    _set_light(light)
    ctx.undo.push("dark mode change", lambda: _set_light(before))
    return f"Dark mode {'on' if light == 0 else 'off'}."


# -- wifi ---------------------------------------------------------------------
def _current_wifi() -> str | None:
    out = _run(["netsh", "wlan", "show", "interfaces"]).stdout
    match = re.search(r"^\s*Profile\s*:\s*(.+)$", out, re.MULTILINE)
    return match.group(1).strip() if match else None


@tool(
    "wifi",
    "Check WiFi status, disconnect (asks the user first, this also cuts Jarvis's internet) "
    "or connect to the last/named network.",
    {
        "action": {"type": "string", "enum": ["status", "disconnect", "connect"]},
        "network": {"type": "string", "description": "Network name for connect (optional)"},
    },
    required=["action"],
)
def wifi(ctx, action: str, network: str = ""):
    if OS != "Windows":
        return {"error": "WiFi control is only supported on Windows for now."}
    current = _current_wifi()
    if action == "status":
        return f"Connected to {current}." if current else "Not connected to WiFi."
    if action == "disconnect":
        if not current:
            return "Already disconnected."
        if not ctx.confirm.confirm(f"Disconnect from WiFi '{current}'? Jarvis will go offline too."):
            return "Cancelled - the user did not confirm."
        ctx.state.set("last_wifi", current) if ctx.state else None
        _run(["netsh", "wlan", "disconnect"])
        return f"Disconnected from {current}."
    name = network.strip() or (ctx.state.get("last_wifi") if ctx.state else None)
    if not name:
        return {"error": "Which network? Say its name."}
    if not re.fullmatch(r"[\w .\-'()&@#]{1,64}", name):
        return {"error": "That doesn't look like a WiFi network name."}
    result = _run(["netsh", "wlan", "connect", f"name={name}"])
    return f"Connecting to {name}." if result.returncode == 0 else {"error": result.stdout.strip()[:200]}
