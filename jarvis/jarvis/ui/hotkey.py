"""System-wide hotkey (Windows): Ctrl+Alt+J calls Jarvis from any app. No extra packages."""

from __future__ import annotations

import logging
import platform
import threading
from collections.abc import Callable

log = logging.getLogger("jarvis.hotkey")

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_NOREPEAT = 0x0001, 0x0002, 0x0004, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
HOTKEY_TEXT = "Ctrl+Alt+J"


class GlobalHotkey:
    def __init__(self, callback: Callable[[], None], key: str = "J", mods: int = MOD_CONTROL | MOD_ALT):
        self.callback = callback
        self.key = key
        self.mods = mods
        self.ok = False
        self._thread_id = 0
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """True if the hotkey is active (False on other systems or if another app owns it)."""
        if platform.system() != "Windows":
            return False
        self._thread = threading.Thread(target=self._run, name="jarvis-hotkey", daemon=True)
        self._thread.start()
        self._ready.wait(2)
        return self.ok

    def _run(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()
        self.ok = bool(user32.RegisterHotKey(None, 1, self.mods | MOD_NOREPEAT, ord(self.key.upper())))
        if not self.ok:
            log.warning("could not register %s (another app may use it)", HOTKEY_TEXT)
        self._ready.set()
        if not self.ok:
            return
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    try:
                        self.callback()
                    except Exception:
                        log.exception("hotkey callback failed")
        finally:
            user32.UnregisterHotKey(None, 1)

    def stop(self) -> None:
        if self._thread_id and platform.system() == "Windows":
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
