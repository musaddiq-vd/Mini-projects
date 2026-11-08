"""How long since the last keyboard/mouse input (Windows). Used only to time health breaks.

Nothing is recorded: it is a single number from the OS, read when needed.
"""

from __future__ import annotations

import ctypes
import platform


class _LastInput(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def seconds_idle() -> float:
    """Seconds since the user last touched the keyboard or mouse (0 when unknown)."""
    if platform.system() != "Windows":
        return 0.0
    info = _LastInput()
    info.cbSize = ctypes.sizeof(_LastInput)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime
    return max(0.0, (millis & 0xFFFFFFFF) / 1000.0)
