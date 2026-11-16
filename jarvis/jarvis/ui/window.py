"""Main Jarvis window: animated orb, live captions, conversation, quick actions, controls.

The window is put together from focused mixins (layout, events, chat, display, settings, tray, lifecycle);
this module holds the Bridge, the MainWindow itself and the start-up of the whole app.
"""

from __future__ import annotations

import getpass
import logging
import sys
import threading

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox

from jarvis.app import Jarvis, create_app
from jarvis.config import home_dir, load_settings
from jarvis.history import ChatHistory
from jarvis.logs import on_crash, setup_logging
from jarvis.ui import theme
from jarvis.ui.hotkey import GlobalHotkey
from jarvis.ui.mini import MiniWindow
from jarvis.ui.window_chat import ChatMixin
from jarvis.ui.window_display import DisplayMixin
from jarvis.ui.window_events import EventsMixin
from jarvis.ui.window_layout import LayoutMixin
from jarvis.ui.window_lifecycle import LifecycleMixin
from jarvis.ui.window_settings import SettingsMixin
from jarvis.ui.window_tray import TrayMixin

# Names that used to live in this module, still importable from here.
from jarvis.ui.chat_view import MAX_CHAT_ROWS, THUMB_WIDTH, TOOL_ICONS, ChatView, ImageThumb, pretty_tool, tool_icon  # noqa: F401
from jarvis.ui.hud import StatsBar, StatusDot, TypingDots, app_icon  # noqa: F401
from jarvis.ui.window_chat import DOC_SUFFIXES  # noqa: F401
from jarvis.ui.window_events import (  # noqa: F401
    AWAKE_STATES, CELEBRATE_WORDS, JOKE_WORDS, PRAISE_WORDS, THANKS_WORDS,
)
from jarvis.ui.window_layout import QUICK_ACTIONS  # noqa: F401
from jarvis.ui.window_settings import RESTART_KEYS  # noqa: F401

INSTANCE_KEY = f"jarvis-assistant-{getpass.getuser()}"
log = logging.getLogger("jarvis.ui")


class Bridge(QObject):
    """Moves events from worker threads onto the Qt (GUI) thread."""

    event = pyqtSignal(str, dict)
    confirm = pyqtSignal(str, object)  # question, reply callback
    assistant_ready = pyqtSignal(object)  # the LiveAssistant (or the error), built off the GUI thread
    hotkey = pyqtSignal()


class MainWindow(LayoutMixin, EventsMixin, ChatMixin, DisplayMixin, SettingsMixin, TrayMixin, LifecycleMixin,
                 QMainWindow):
    """The Jarvis main window; each mixin adds one part of its behaviour."""

    def __init__(self, app: Jarvis, bridge: Bridge):
        super().__init__()
        self.app = app
        self.bridge = bridge
        self.assistant = None
        self.thread: threading.Thread | None = None
        self._closing = False
        self._last_state = ""
        self.accent = theme.accent_for(app.settings.theme_name, app.settings.theme_color)
        self.history = app.history or ChatHistory(home_dir() / "history.jsonl")  # saved by the core
        self.mini: MiniWindow | None = None
        self.hotkey: GlobalHotkey | None = None
        bridge.hotkey.connect(self._on_hotkey)

        bridge.event.connect(self._on_event)
        bridge.confirm.connect(self._on_confirm)
        bridge.assistant_ready.connect(self._assistant_ready)
        app.events.subscribe(self._forward)

        self.setMinimumSize(760, 560)  # fits small laptop screens; rows wrap instead of overflowing
        self.setAcceptDrops(True)
        self._build()
        self._build_tray()
        self._shortcuts()
        self._apply_theme()
        self._restore_history()
        if app.settings.desktop_widget:
            QTimer.singleShot(400, self.apply_desktop_widget)
        if app.settings.telegram_token and app.settings.telegram_chat_id:
            QTimer.singleShot(1500, self.apply_phone)
        self.practice_btn.setText(f"🎓  {app.state.get('practice_target', 'English')} Practice")
        if not app.settings.effective_api_key and app.practice_on:
            QTimer.singleShot(0, lambda: self.set_practice(True))


def make_confirm(bridge: Bridge):
    """A ConfirmGate 'ask' function that shows the banner and waits for a click."""

    def ask(question: str) -> bool:
        done = threading.Event()
        answer: list[bool] = []

        def reply(yes: bool) -> None:
            answer.append(yes)
            done.set()

        bridge.confirm.emit(question, reply)
        done.wait(55)
        return bool(answer and answer[0])

    return ask


def _already_running() -> bool:
    """If Jarvis is already open, ask it to come to the front and return True."""
    sock = QLocalSocket()
    sock.connectToServer(INSTANCE_KEY)
    if sock.waitForConnected(300):
        sock.write(b"show")
        sock.waitForBytesWritten(300)
        sock.disconnectFromServer()
        return True
    return False


def _listen_for_second_start(window: "MainWindow") -> QLocalServer:
    QLocalServer.removeServer(INSTANCE_KEY)  # clean up after a crash
    server = QLocalServer(window)
    server.newConnection.connect(lambda: (server.nextPendingConnection(), window._show_window()))
    if not server.listen(INSTANCE_KEY):
        log.warning("single-instance server: %s", server.errorString())
    return server


def run_window() -> int:
    setup_logging()
    qt = QApplication.instance() or QApplication(sys.argv)
    qt.setApplicationName("Jarvis")
    qt.setFont(QFont("Segoe UI", 10))
    qt.setQuitOnLastWindowClosed(False)  # hiding into mini mode must not end the app
    if _already_running():
        log.info("already running - brought the open window to the front")
        return 0
    from jarvis import session

    crashed = session.begin() is not None  # the marker from last time is still there: it never quit cleanly
    bridge = Bridge()
    settings = load_settings()
    app = create_app(settings, ask=make_confirm(bridge))
    window = MainWindow(app, bridge)
    if not window.restore_geometry():
        window.resize(1160, 740)
    window._instance_server = _listen_for_second_start(window)
    on_crash(lambda text: app.events.emit("crash", text=text))
    window.show()
    window.start_hotkey()
    for err in app.registry.errors:
        window._add_system(f"⚠ {err}")
    if crashed:
        session.recover(app, window._add_system)
    if not app.settings.effective_api_key:
        QTimer.singleShot(200, lambda: window.open_settings(first_run=True) or QMessageBox.information(
            window, "Jarvis", "Jarvis needs a Gemini API key. Open Settings any time to add it."))
    else:
        window.start_assistant()
    code = qt.exec()
    session.end()  # a clean quit; after a crash the marker stays for the next start
    return code
