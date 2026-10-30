"""Main Jarvis window: animated orb, conversation, typed input, controls, confirmations."""

from __future__ import annotations

import html
import sys
import threading
from datetime import datetime

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap, QTextBlockFormat, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QTextBrowser, QVBoxLayout, QWidget,
)

from jarvis import __version__
from jarvis.app import Jarvis, create_app
from jarvis.config import load_settings, save_settings
from jarvis.ui.orb import Orb
from jarvis.ui.settings_dialog import SettingsDialog, apply_settings
from jarvis.ui.theme import MUTED, STATE_LABELS, WARN, stylesheet, valid_color

RESTART_KEYS = {"voice", "live_model", "input_device", "output_device", "wake_word",
                "assistant_name", "user_name", "api_key", "allow_interrupt"}


class Bridge(QObject):
    """Moves events from worker threads onto the Qt (GUI) thread."""

    event = pyqtSignal(str, dict)
    confirm = pyqtSignal(str, object)  # question, reply callback


def app_icon(color: str) -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(color)
    p.setBrush(c.darker(200))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(4, 4, 56, 56)
    p.setBrush(c)
    p.drawEllipse(18, 18, 28, 28)
    p.end()
    return QIcon(pix)


class MainWindow(QMainWindow):
    def __init__(self, app: Jarvis, bridge: Bridge):
        super().__init__()
        self.app = app
        self.bridge = bridge
        self.assistant = None
        self.thread: threading.Thread | None = None
        self._closing = False
        self.accent = valid_color(app.settings.theme_color)

        bridge.event.connect(self._on_event)
        bridge.confirm.connect(self._on_confirm)
        app.events.subscribe(lambda kind, data: bridge.event.emit(kind, data))

        self.setMinimumSize(980, 640)
        self._build()
        self._apply_theme()

    # -- layout -------------------------------------------------------------------
    def _build(self) -> None:
        root = QWidget()
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        # left: orb + controls
        left = QFrame(objectName="panel")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(18, 18, 18, 18)
        self.title = QLabel(objectName="title")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(self.title)
        self.clock = QLabel(objectName="muted")
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(self.clock)
        self.orb = Orb(self.accent)
        lv.addWidget(self.orb, 1)
        self.state_label = QLabel("CONNECTING", objectName="state")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(self.state_label)

        controls = QHBoxLayout()
        self.mute_btn = QPushButton("🎙  Mute")
        self.mute_btn.setCheckable(True)
        self.mute_btn.toggled.connect(self._toggle_mute)
        self.sleep_btn = QPushButton("💤  Sleep")
        self.sleep_btn.clicked.connect(self._toggle_sleep)
        settings_btn = QPushButton("⚙  Settings")
        settings_btn.clicked.connect(self.open_settings)
        for b in (self.mute_btn, self.sleep_btn, settings_btn):
            controls.addWidget(b)
        lv.addLayout(controls)
        credit = QLabel(f"Jarvis v{__version__} · by Musaddiq Khan", objectName="muted")
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(credit)
        outer.addWidget(left, 5)

        # right: banner + conversation + input
        right = QFrame(objectName="panel")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(18, 18, 18, 18)
        rv.addWidget(QLabel("CONVERSATION", objectName="section"))

        self.banner = QFrame(objectName="banner")
        bl = QHBoxLayout(self.banner)
        self.banner_text = QLabel()
        self.banner_text.setWordWrap(True)
        self.banner_text.setStyleSheet(f"color:{WARN}; background:transparent; font-weight:600;")
        bl.addWidget(self.banner_text, 1)
        self.banner_yes = QPushButton("Confirm", objectName="danger")
        self.banner_no = QPushButton("Cancel")
        bl.addWidget(self.banner_yes)
        bl.addWidget(self.banner_no)
        self.banner_yes.clicked.connect(lambda: self._answer_confirm(True))
        self.banner_no.clicked.connect(lambda: self._answer_confirm(False))
        self.banner.hide()
        self._pending_reply = None
        rv.addWidget(self.banner)

        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(True)
        rv.addWidget(self.chat, 1)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a message, or just talk...")
        self.input.returnPressed.connect(self._send_typed)
        send = QPushButton("Send", objectName="primary")
        send.clicked.connect(self._send_typed)
        row.addWidget(self.input, 1)
        row.addWidget(send)
        rv.addLayout(row)
        outer.addWidget(right, 6)
        self.setCentralWidget(root)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(1000)
        self._update_clock()

    def _apply_theme(self) -> None:
        self.accent = valid_color(self.app.settings.theme_color)
        QApplication.instance().setStyleSheet(stylesheet(self.accent))
        self.orb.set_accent(self.accent)
        name = self.app.settings.assistant_name
        self.title.setText(name.upper())
        self.setWindowTitle(name)
        self.setWindowIcon(app_icon(self.accent))
        self.state_label.setStyleSheet(f"color:{self.accent};")

    def _update_clock(self) -> None:
        self.clock.setText(datetime.now().strftime("%A · %d %B · %I:%M %p"))

    # -- assistant lifecycle ------------------------------------------------------
    def start_assistant(self) -> None:
        if not self.app.settings.effective_api_key:
            self._add_system("Add your Gemini API key in Settings to start.")
            return
        from jarvis.live import LiveAssistant

        self.assistant = LiveAssistant(self.app)
        self.thread = self.assistant.run_in_thread()
        self._add_system("Starting... say hello, or type below.")

    # -- events ---------------------------------------------------------------------
    def _on_event(self, kind: str, data: dict) -> None:
        if kind == "level":
            self.orb.set_level(data["value"])
        elif kind == "state":
            state = data["state"]
            self.orb.set_state(state)
            self.state_label.setText(STATE_LABELS.get(state, state.upper()))
            self.sleep_btn.setText("☀  Wake" if state == "sleeping" else "💤  Sleep")
        elif kind == "user":
            self._add_bubble("You", data["text"], right=True)
        elif kind == "assistant":
            self._add_bubble(self.app.settings.assistant_name, data["text"])
        elif kind == "tool":
            self._add_tool(data["name"], data.get("args", {}), data.get("result", {}))
        elif kind == "log":
            self._add_system(data["text"])

    def _append(self, inner_html: str, align: str, top: int = 4) -> None:
        """Add one paragraph with its own alignment (QTextBrowser.append would reuse the last one)."""
        cursor = self.chat.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextBlockFormat()
        fmt.setAlignment({"left": Qt.AlignmentFlag.AlignLeft, "right": Qt.AlignmentFlag.AlignRight,
                          "center": Qt.AlignmentFlag.AlignHCenter}[align])
        fmt.setTopMargin(top)
        first = self.chat.document().isEmpty()
        if not first:
            cursor.insertBlock(fmt)
        cursor.insertHtml(inner_html)
        if first:
            cursor.setBlockFormat(fmt)
        self._scroll_down()

    def _scroll_down(self) -> None:
        bar = self.chat.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _add_bubble(self, who: str, text: str, right: bool = False) -> None:
        if not text:
            return
        if text.startswith("(system"):
            return  # app-generated instructions are not shown as user messages
        color = self.accent if not right else "#d8e6ef"
        align = "right" if right else "left"
        safe = html.escape(text).replace("\n", "<br>")
        self._append(
            f'<span style="color:{color}; font-weight:700; font-size:11px;">{html.escape(who.upper())}</span><br>'
            f'<span style="font-size:14px;">{safe}</span>',
            align, top=10,
        )

    def _add_tool(self, name: str, args: dict, result: dict) -> None:
        ok = "error" not in result
        detail = result.get("error") or result.get("result") or ""
        args_text = ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items())
        mark = "✓" if ok else "✗"
        color = MUTED if ok else WARN
        line = f"{mark} {name}({args_text})"
        if detail:
            line += f" → {str(detail)[:120]}"
        self._append(f'<span style="color:{color}; font-size:11px; font-family:Consolas,monospace;">'
                     f"{html.escape(line)}</span>", "left")

    def _add_system(self, text: str) -> None:
        self._append(f'<span style="color:{MUTED}; font-size:11px;">{html.escape(text)}</span>', "center")

    # -- controls ---------------------------------------------------------------
    def _send_typed(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        if not self.assistant:
            self._add_system("Not connected yet - check Settings.")
            return
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

    def open_settings(self, first_run: bool = False) -> bool:
        dialog = SettingsDialog(self.app, self, first_run=first_run)
        if not dialog.exec():
            return False
        changed = apply_settings(self.app.settings, dialog.result_settings)
        save_settings(self.app.settings)
        if "theme_color" in changed or "assistant_name" in changed:
            self._apply_theme()
        if self.assistant is None:
            self.start_assistant()
        elif changed & RESTART_KEYS:
            self.assistant.restart(fresh="voice" in changed)
        if changed:
            self._add_system("Settings saved.")
        return True

    # -- confirmation -------------------------------------------------------------
    def _on_confirm(self, question: str, reply) -> None:
        self._pending_reply = reply
        self.banner_text.setText(f"⚠  {question}")
        self.banner.show()
        self.raise_()
        self.activateWindow()

    def _answer_confirm(self, yes: bool) -> None:
        self.banner.hide()
        reply, self._pending_reply = self._pending_reply, None
        if reply:
            reply(yes)

    # -- close --------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._closing or not (self.thread and self.thread.is_alive()):
            event.accept()
            return
        self._closing = True
        event.ignore()
        self._add_system("Saving this conversation...")
        self.assistant.stop()
        self._wait_ticks = 0

        def check():
            self._wait_ticks += 1
            if not self.thread.is_alive() or self._wait_ticks > 60:  # at most ~15 s
                QApplication.instance().quit()
            else:
                QTimer.singleShot(250, check)

        QTimer.singleShot(250, check)


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


def run_window() -> int:
    qt = QApplication.instance() or QApplication(sys.argv)
    qt.setApplicationName("Jarvis")
    qt.setFont(QFont("Segoe UI", 10))
    bridge = Bridge()
    settings = load_settings()
    app = create_app(settings, ask=make_confirm(bridge))
    window = MainWindow(app, bridge)
    window.resize(1100, 700)
    window.show()
    for err in app.registry.errors:
        window._add_system(f"⚠ {err}")
    if not app.settings.effective_api_key:
        QTimer.singleShot(200, lambda: window.open_settings(first_run=True) or QMessageBox.information(
            window, "Jarvis", "Jarvis needs a Gemini API key. Open Settings any time to add it."))
    else:
        window.start_assistant()
    return qt.exec()

