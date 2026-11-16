"""Tray icon, keyboard shortcuts and the global hotkey of the main window."""

from __future__ import annotations

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon

from jarvis.logs import log_dir
from jarvis.ui.hotkey import HOTKEY_TEXT, GlobalHotkey


class TrayMixin:
    """Tray menu, shortcuts and the Ctrl+Alt+J hotkey."""

    # -- tray and shortcuts ---------------------------------------------------------
    def _toggle_desktop_widget(self) -> None:
        self.ui_command("widget_off" if self.app.settings.desktop_widget else "widget_on")

    def _build_tray(self) -> None:
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(self)
        menu = QMenu()
        for text, fn in (("Show", self._show_window), ("Mini mode", self.toggle_mini),
                         ("Desktop widget on / off", self._toggle_desktop_widget),
                         ("Wake / Sleep", self._toggle_sleep),
                         ("Mute / Unmute", self.mute_btn.toggle), ("Open logs folder", self._open_logs),
                         ("Quit", self.close)):
            act = QAction(text, menu)
            act.triggered.connect(fn)
            menu.addAction(act)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self._show_window() if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
        self.tray.show()

    def _shortcuts(self) -> None:
        for keys, fn in (("Ctrl+M", self.mute_btn.toggle), ("Ctrl+Space", self._toggle_sleep),
                         ("Ctrl+L", self._clear_chat), ("Ctrl+,", self.open_settings),
                         ("Ctrl+K", self.input.setFocus), ("Ctrl+Shift+M", self.toggle_mini),
                         ("Ctrl+F", self.open_search), ("Esc", self._escape), ("F1", self.toggle_help),
                         ("Ctrl+P", self.toggle_planner), ("Ctrl+D", self.toggle_dashboard)):
            QShortcut(QKeySequence(keys), self, activated=fn)

    def _open_logs(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir())))

    # -- global hotkey --------------------------------------------------------------
    def start_hotkey(self) -> None:
        self.hotkey = GlobalHotkey(self.bridge.hotkey.emit)
        if self.hotkey.start():
            self.input.setPlaceholderText(f"Type a message, or just talk...  ({HOTKEY_TEXT} from anywhere)")
        else:
            self.hotkey = None

    def _on_hotkey(self) -> None:
        """Ctrl+Alt+J from any app: bring Jarvis up, awake and ready to listen."""
        if self.mini and self.mini.isVisible():
            self.mini.raise_()
        else:
            self._show_window()
            self.input.setFocus()
        if self.assistant and not self.assistant.awake:
            self.assistant.wake_up()
        self.orb.flash()
