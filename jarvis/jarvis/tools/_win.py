"""Windows window helpers using the Win32 API through ctypes (no extra packages).

Files starting with "_" are not scanned for tools.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass

import psutil

user32 = ctypes.windll.user32

WM_CLOSE = 0x0010
SW_MINIMIZE, SW_MAXIMIZE, SW_RESTORE = 6, 3, 9
VK_MENU = 0x12
KEYEVENTF_KEYUP = 2

BROWSERS = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "vivaldi.exe"}
# Window titles of shell/background windows that should never be touched
IGNORED_TITLES = {"Program Manager", "Windows Input Experience", "Settings", ""}


@dataclass
class Window:
    hwnd: int
    title: str
    process: str  # e.g. "chrome.exe"
    pid: int

    @property
    def is_browser(self) -> bool:
        return self.process.lower() in BROWSERS

    @property
    def app_name(self) -> str:
        return self.process.rsplit(".", 1)[0]


def list_windows() -> list[Window]:
    windows: list[Window] = []
    proc_names: dict[int, str] = {}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, 4):  # skip owned popups
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if title in IGNORED_TITLES:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value not in proc_names:
            try:
                proc_names[pid.value] = psutil.Process(pid.value).name()
            except psutil.Error:
                proc_names[pid.value] = "?"
        windows.append(Window(int(hwnd), title, proc_names[pid.value], pid.value))
        return True

    user32.EnumWindows(callback, 0)
    return windows


def foreground() -> Window | None:
    hwnd = user32.GetForegroundWindow()
    for w in list_windows():
        if w.hwnd == hwnd:
            return w
    return None


def focus(hwnd: int) -> bool:
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    # Windows only lets the foreground app change focus; a tap of Alt lifts that lock.
    user32.keybd_event(VK_MENU, 0, 0, 0)
    user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
    ok = bool(user32.SetForegroundWindow(hwnd))
    time.sleep(0.25)
    return ok


def close(hwnd: int) -> None:
    """Ask the window to close, like clicking X (apps can still ask to save)."""
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)


def show(hwnd: int, how: int) -> None:
    user32.ShowWindow(hwnd, how)
