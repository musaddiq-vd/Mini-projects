"""Time, system status and power actions."""

from __future__ import annotations

import platform
import subprocess
import threading
from datetime import datetime

import psutil

from jarvis.tools import tool

OS = platform.system()  # "Windows", "Darwin", "Linux"


@tool("get_time", "Get the current local date, time and weekday.")
def get_time(ctx):
    now = datetime.now()
    return now.strftime("%A, %d %B %Y, %I:%M %p")


@tool("system_status", "Report CPU, RAM, disk and battery usage of this computer.")
def system_status(ctx):
    info = {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "ram_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
    }
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if battery:
        info["battery_percent"] = round(battery.percent)
        info["charging"] = bool(battery.power_plugged)
    return info


def _power_command(action: str) -> list[str]:
    if OS == "Windows":
        return {
            "shutdown": ["shutdown", "/s", "/t", "5"],
            "restart": ["shutdown", "/r", "/t", "5"],
            "lock": ["rundll32.exe", "user32.dll,LockWorkStation"],
        }[action]
    if OS == "Darwin":
        return {
            "shutdown": ["osascript", "-e", 'tell app "System Events" to shut down'],
            "restart": ["osascript", "-e", 'tell app "System Events" to restart'],
            "lock": ["pmset", "displaysleepnow"],
            "sleep": ["pmset", "sleepnow"],
        }[action]
    return {
        "shutdown": ["systemctl", "poweroff"],
        "restart": ["systemctl", "reboot"],
        "lock": ["loginctl", "lock-session"],
        "sleep": ["systemctl", "suspend"],
    }[action]


def _windows_sleep() -> None:
    """Real sleep (not hibernate). `rundll32 powrprof.dll,SetSuspendState` passes its
    arguments wrongly and hibernates when hibernation is on - it looks like a shutdown."""
    import ctypes

    ctypes.windll.powrprof.SetSuspendState(False, False, False)


@tool(
    "power",
    "Shut down, restart, lock or sleep THE COMPUTER itself. Only use when the user clearly talks about "
    "the computer/PC/laptop/system. NEVER use it when the user tells you (the assistant) to sleep, "
    "rest, stop listening or be quiet - use assistant_sleep for that. Everything except lock asks the "
    "user to confirm on screen.",
    {"action": {"type": "string", "enum": ["shutdown", "restart", "lock", "sleep"]}},
    required=["action"],
)
def power(ctx, action: str):
    action = action.lower().strip()
    if action not in {"shutdown", "restart", "lock", "sleep"}:
        return {"error": f"Unknown power action '{action}'."}
    if action != "lock":
        question = {"sleep": "Put the COMPUTER to sleep?"}.get(action, f"Really {action} the COMPUTER?")
        if not ctx.confirm.confirm(question):
            return "Cancelled - the user did not confirm."
    if action == "sleep" and OS == "Windows":
        threading.Timer(2.0, _windows_sleep).start()  # let the reply be spoken first
    else:
        subprocess.Popen(_power_command(action))
    return f"Computer {action} started."
