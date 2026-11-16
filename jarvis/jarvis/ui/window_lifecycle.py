"""Startup and shutdown of the main window: the voice assistant thread, restart, quit and geometry."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PyQt6.QtCore import QByteArray, QSettings, QTimer
from PyQt6.QtWidgets import QApplication

log = logging.getLogger("jarvis.ui")


class LifecycleMixin:
    """Starts and stops the assistant and the window."""

    # -- assistant lifecycle ------------------------------------------------------
    def start_assistant(self) -> None:
        """Build the voice assistant on a background thread (loading Gemini takes ~2 s), so the window opens
        and animates right away. Messages typed meanwhile are kept and sent once it's ready."""
        if not self.app.settings.effective_api_key:
            self._add_system("Add your Gemini API key in Settings to start.")
            return
        if self._booting:
            return
        self._booting = True
        self._add_system(f"Starting... say \"{self.app.settings.assistant_name}\", or type below.")

        def build():
            try:
                from jarvis.live import LiveAssistant

                made = LiveAssistant(self.app)
            except Exception as exc:  # shown in the chat instead of a silent failure
                log.exception("couldn't start the assistant")
                made = exc
            try:
                self.bridge.assistant_ready.emit(made)
            except RuntimeError:
                pass  # the window closed meanwhile

        threading.Thread(target=build, daemon=True, name="jarvis-boot").start()

    _booting = False
    _typed_early: list[str] = []

    def _assistant_ready(self, made) -> None:
        self._booting = False
        if isinstance(made, Exception):
            self._add_system(f"⚠ Couldn't start: {type(made).__name__}: {made}")
            return
        if self._quitting:
            return
        self.assistant = made
        if self.app.practice_on:  # continue where the user left off
            self.set_practice(True)
        self.thread = self.assistant.run_in_thread()
        early, self._typed_early = self._typed_early, []
        for text in early:
            self.assistant.send_text(text)

    def _toggle_mute(self, muted: bool) -> None:
        self.mute_btn.setText("🔇  Unmute" if muted else "🎙  Mute")
        if self.assistant:
            self.assistant.set_muted(muted)

    def _toggle_sleep(self) -> None:
        if not self.assistant:
            return
        if self.assistant.awake:
            self.assistant.sleep()
        else:
            self.assistant.wake_up()

    # -- close --------------------------------------------------------------------
    def restart_app(self) -> None:
        """Start a fresh Jarvis, then quit this one (after a restore)."""
        import sys

        from PyQt6.QtCore import QProcess

        root = Path(__file__).resolve().parents[2]
        launcher = root / "jarvis_launcher.pyw"
        windowless = Path(sys.executable).with_name("pythonw.exe")
        exe = str(windowless) if windowless.exists() else sys.executable
        if getattr(sys, "frozen", False):
            QProcess.startDetached(sys.executable, sys.argv[1:])
        elif launcher.exists():
            QProcess.startDetached(exe, [str(launcher)], str(root))
        else:
            QProcess.startDetached(sys.executable, ["-m", "jarvis"], str(root))
        self._quit()

    def _quit(self) -> None:
        """The app doesn't quit on its own (mini mode hides this window), so quit explicitly."""
        self._quitting = True
        self.reminders.close_all()
        if self.phone is not None:
            self.phone.stop()
        if self.desk is not None:
            self.desk.close()
        if self.mini:
            self.mini.close()
        QTimer.singleShot(0, QApplication.instance().quit)

    _quitting = False

    def save_geometry(self) -> None:
        QSettings("MusaddiqKhan", "Jarvis").setValue("geometry", self.saveGeometry())

    def restore_geometry(self) -> bool:
        data = QSettings("MusaddiqKhan", "Jarvis").value("geometry")
        return bool(isinstance(data, QByteArray) and self.restoreGeometry(data))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._quitting:  # Qt asks every window again while quitting
            event.accept()
            return
        if self.hotkey:
            self.hotkey.stop()
            self.hotkey = None
        try:
            from jarvis.tools.vision import clear_seen_cache

            clear_seen_cache()  # copies of what Jarvis looked at don't outlive the session
        except Exception:
            pass
        self._leave_mini()
        if not self._closing and self.isVisible():
            self.save_geometry()
        if self._closing or not (self.thread and self.thread.is_alive()):
            if self.tray:
                self.tray.hide()
            event.accept()
            self._quit()
            return
        self._closing = True
        event.ignore()
        self._add_system("Saving this conversation...")
        self.assistant.stop()
        self._wait_ticks = 0

        def check():
            self._wait_ticks += 1
            if not self.thread.is_alive() or self._wait_ticks > 60:  # at most ~15 s
                if self.tray:
                    self.tray.hide()
                self._quit()
            else:
                QTimer.singleShot(250, check)

        QTimer.singleShot(250, check)
