"""Volume and media keys, using only built-in OS features (no extra packages)."""

from __future__ import annotations

import platform
import shutil
import subprocess

from jarvis.tools import tool

OS = platform.system()

# Windows virtual-key codes
VK = {"volume_up": 0xAF, "volume_down": 0xAE, "mute": 0xAD,
      "play_pause": 0xB3, "next": 0xB0, "previous": 0xB1}


def _press_windows(key: str, times: int = 1) -> None:
    import ctypes

    keybd_event = ctypes.windll.user32.keybd_event
    for _ in range(times):
        keybd_event(VK[key], 0, 0, 0)
        keybd_event(VK[key], 0, 2, 0)  # key up


def _run(cmd: list[str]) -> bool:
    if not shutil.which(cmd[0]):
        return False
    return subprocess.run(cmd, capture_output=True).returncode == 0


@tool(
    "volume",
    "Change system volume: up, down or mute (mute toggles).",
    {
        "action": {"type": "string", "enum": ["up", "down", "mute"]},
        "steps": {"type": "integer", "description": "How much, 1-10 (default 3)"},
    },
    required=["action"],
)
def volume(ctx, action: str, steps: int = 3):
    steps = max(1, min(int(steps or 3), 10))
    action = action.lower()
    if action not in {"up", "down", "mute"}:
        return {"error": "action must be up, down or mute"}
    if OS == "Windows":
        _press_windows("mute" if action == "mute" else f"volume_{action}", 1 if action == "mute" else steps)
        ok = True
    elif OS == "Darwin":
        script = {
            "up": f"set volume output volume ((output volume of (get volume settings)) + {steps * 5})",
            "down": f"set volume output volume ((output volume of (get volume settings)) - {steps * 5})",
            "mute": "set volume output muted (not (output muted of (get volume settings)))",
        }[action]
        ok = _run(["osascript", "-e", script])
    else:
        arg = {"up": f"+{steps * 5}%", "down": f"-{steps * 5}%"}.get(action)
        cmd = (["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"] if action == "mute"
               else ["pactl", "set-sink-volume", "@DEFAULT_SINK@", arg])
        ok = _run(cmd)
    return f"Volume {action}." if ok else {"error": "Could not change the volume on this system."}


@tool(
    "media_control",
    "Control the music/video that is playing: play_pause, next or previous.",
    {"action": {"type": "string", "enum": ["play_pause", "next", "previous"]}},
    required=["action"],
)
def media_control(ctx, action: str):
    if action not in {"play_pause", "next", "previous"}:
        return {"error": "Unknown media action."}
    if OS == "Windows":
        _press_windows(action)
        return "Done."
    if OS == "Linux" and _run(["playerctl", {"play_pause": "play-pause"}.get(action, action)]):
        return "Done."
    return {"error": "Media keys are only supported on Windows and Linux (playerctl) for now."}
