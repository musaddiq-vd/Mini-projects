"""Main Jarvis window: animated orb, live captions, conversation, quick actions, controls."""

from __future__ import annotations

import getpass
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

import psutil
from PyQt6.QtCore import (
    QByteArray, QEasingCurve, QObject, QPropertyAnimation, QRectF, QSettings, Qt, QTimer, QUrl, pyqtSignal,
)
from PyQt6.QtGui import (
    QAction, QColor, QDesktopServices, QFont, QIcon, QKeySequence, QPainter, QPainterPath, QPixmap, QShortcut,
)
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from PyQt6.QtWidgets import (
    QApplication, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSystemTrayIcon, QVBoxLayout, QWidget,
)

from jarvis import __version__
from jarvis.app import Jarvis, create_app
from jarvis.config import home_dir, load_settings, save_settings
from jarvis.history import ChatHistory
from jarvis.logs import log_dir, on_crash, setup_logging
from jarvis.ui.hotkey import HOTKEY_TEXT, GlobalHotkey
from jarvis.ui.mini import MiniWindow, OrbGroup
from jarvis.ui.orb import Orb
from jarvis.ui.settings_dialog import SettingsDialog, apply_settings
from jarvis.ui.theme import DANGER, MUTED, OK, STATE_LABELS, WARN, stylesheet, valid_color

RESTART_KEYS = {"voice", "live_model", "input_device", "output_device", "wake_word",
                "assistant_name", "user_name", "api_key", "allow_interrupt"}
MAX_CHAT_ROWS = 250
THUMB_WIDTH = 340
INSTANCE_KEY = f"jarvis-assistant-{getpass.getuser()}"
log = logging.getLogger("jarvis.ui")
AWAKE_STATES = {"listening", "speaking", "muted", "thinking"}

QUICK_ACTIONS = [
    ("🖥  Screen", "What's on my screen?"),
    ("🌦  Weather", "What's the weather today?"),
    ("📰  News", "Give me today's top news in two lines."),
    ("📊  System", "How is my computer doing? CPU, RAM and battery."),
    ("⏰  Reminders", "What reminders do I have?"),
    ("😄  Joke", "Tell me a short joke."),
]

TOOL_ICONS = {
    "open": "🚀", "close": "✖", "volume": "🔊", "brightness": "☀", "media": "🎵", "play": "▶",
    "screenshot": "📸", "screen": "👁", "camera": "📷", "weather": "🌦", "search": "🔎", "news": "📰", "remind": "⏰",
    "remember": "🧠", "recall": "🧠", "memory": "🧠", "file": "📁", "folder": "📁", "whatsapp": "💬",
    "message": "💬", "email": "✉", "code": "💻", "type": "⌨", "key": "⌨", "clipboard": "📋",
    "power": "⏻", "wifi": "📶", "time": "🕒", "system": "📊", "undo": "↩", "sleep": "💤", "window": "🪟",
}


def tool_icon(name: str) -> str:
    for key, icon in TOOL_ICONS.items():
        if key in name:
            return icon
    return "⚡"


def pretty_tool(name: str) -> str:
    return name.replace("_", " ").capitalize()


class Bridge(QObject):
    """Moves events from worker threads onto the Qt (GUI) thread."""

    event = pyqtSignal(str, dict)
    confirm = pyqtSignal(str, object)  # question, reply callback
    hotkey = pyqtSignal()


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
    p.drawEllipse(14, 14, 36, 36)
    p.setBrush(c.darker(400))
    p.drawEllipse(22, 26, 6, 10)  # tiny eyes
    p.drawEllipse(36, 26, 6, 10)
    p.end()
    return QIcon(pix)


class StatsBar(QWidget):
    """Tiny HUD meters: CPU, RAM and battery."""

    def __init__(self, accent: str, parent=None):
        super().__init__(parent)
        self.accent = accent
        self.values: list[tuple[str, float, str]] = []
        self.setFixedHeight(34)
        psutil.cpu_percent(interval=None)  # prime the counter
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(2000)
        self.refresh()

    def refresh(self) -> None:
        vals = [("CPU", psutil.cpu_percent(interval=None), ""), ("RAM", psutil.virtual_memory().percent, "")]
        try:
            bat = psutil.sensors_battery()
        except Exception:
            bat = None
        if bat:
            vals.append(("BAT", bat.percent, "⚡" if bat.power_plugged else ""))
        self.values = vals
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self.values) or 1
        gap = 12
        cell = (self.width() - gap * (n - 1)) / n
        p.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        for i, (name, value, extra) in enumerate(self.values):
            x = i * (cell + gap)
            bad = value > 90 if name != "BAT" else value < 20 and not extra
            color = QColor(DANGER if bad else WARN if (name != "BAT" and value > 70) else self.accent)
            p.setPen(QColor(MUTED))
            p.drawText(QRectF(x, 0, cell, 14), Qt.AlignmentFlag.AlignLeft, name)
            p.setPen(color)
            p.drawText(QRectF(x, 0, cell, 14), Qt.AlignmentFlag.AlignRight, f"{extra}{value:.0f}%")
            track = QRectF(x, 19, cell, 5)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 18))
            p.drawRoundedRect(track, 2.5, 2.5)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(x, 19, cell * min(value, 100) / 100, 5), 2.5, 2.5)
        p.end()


class StatusDot(QWidget):
    """Pulsing dot next to the state label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color = QColor(MUTED)
        self.t = 0.0
        self.setFixedSize(14, 14)
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(50)

    def _tick(self) -> None:
        self.t += 0.05
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        import math

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        halo = QColor(self.color)
        halo.setAlpha(int(60 + 60 * math.sin(self.t * 4)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(halo)
        p.drawEllipse(1, 1, 12, 12)
        p.setBrush(self.color)
        p.drawEllipse(4, 4, 6, 6)
        p.end()


class TypingDots(QWidget):
    """"Jarvis is thinking" with three bouncing dots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ""
        self.color = QColor(MUTED)
        self.t = 0.0
        self.setFixedHeight(22)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def show_text(self, text: str, color: str) -> None:
        self.text, self.color = text, QColor(color)
        if not self._timer.isActive():
            self._timer.start(40)
        self.show()

    def hide_now(self) -> None:
        self._timer.stop()
        self.hide()

    def _tick(self) -> None:
        self.t += 0.04
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        import math

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        for k in range(3):
            c = QColor(self.color)
            bounce = max(0.0, math.sin(self.t * 6 - k * 0.7))
            c.setAlpha(int(110 + 145 * bounce))
            p.setBrush(c)
            p.drawEllipse(QRectF(8 + k * 12, 8 - 4 * bounce, 7, 7))
        p.setPen(QColor(MUTED))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRectF(52, 0, self.width() - 52, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text)
        p.end()


class ImageThumb(QLabel):
    """Rounded, clickable thumbnail of an image file; click opens it."""

    def __init__(self, path: str, accent: str, parent=None):
        super().__init__(parent)
        self.path = path
        pix = QPixmap(path)
        if pix.isNull():
            self.setText(f"🖼  {Path(path).name}")
            return
        pix = pix.scaledToWidth(THUMB_WIDTH, Qt.TransformationMode.SmoothTransformation)
        rounded = QPixmap(pix.size())
        rounded.fill(Qt.GlobalColor.transparent)
        p = QPainter(rounded)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(0, 0, pix.width(), pix.height()), 12, 12)
        p.setClipPath(clip)
        p.drawPixmap(0, 0, pix)
        p.setClipping(False)
        p.setPen(QColor(accent))
        p.drawRoundedRect(QRectF(0.5, 0.5, pix.width() - 1, pix.height() - 1), 12, 12)
        p.end()
        self.setPixmap(rounded)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"{Path(path).name}\nClick to open")

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and Path(self.path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.path))


class ChatView(QScrollArea):
    """Conversation as fading-in bubbles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(objectName="chatInner")
        self.col = QVBoxLayout(inner)
        self.col.setContentsMargins(4, 4, 10, 4)
        self.col.setSpacing(8)
        self.col.addStretch(1)
        self.setWidget(inner)
        self._anims: list[QPropertyAnimation] = []
        self._stick = True
        self.verticalScrollBar().rangeChanged.connect(self._on_range)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def _on_scroll(self, value: int) -> None:
        self._stick = value >= self.verticalScrollBar().maximum() - 30

    def _on_range(self, _min: int, maximum: int) -> None:
        if self._stick:
            self.verticalScrollBar().setValue(maximum)

    def _add_row(self, widget: QWidget, align: str) -> None:
        row = QWidget(objectName="chatRow")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        if align == "right":
            h.addStretch(1)
        h.addWidget(widget, 1 if align == "center" else 0)
        if align == "left":
            h.addStretch(1)
        self.col.addWidget(row)
        while self.col.count() > MAX_CHAT_ROWS + 1:
            old = self.col.takeAt(1).widget()
            if old:
                old.deleteLater()
        effect = QGraphicsOpacityEffect(row)
        row.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(320)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: (row.setGraphicsEffect(None), self._anims.remove(anim)))
        self._anims.append(anim)
        anim.start()
        self._stick = True

    def _max_width(self, share: float) -> int:
        return max(240, int(self.viewport().width() * share))

    @staticmethod
    def _fit(label: QLabel, max_width: int) -> None:
        """Short text gets a snug label, long text wraps at max_width (QLabel can't decide alone)."""
        longest = max((label.fontMetrics().horizontalAdvance(line) for line in label.text().splitlines()),
                      default=0)
        label.setMinimumWidth(min(longest + 6, max_width))

    def add_message(self, who: str, text: str, mine: bool, when: datetime | None = None) -> None:
        bubble = QFrame(objectName="bubbleMe" if mine else "bubbleAi")
        v = QVBoxLayout(bubble)
        v.setContentsMargins(12, 8, 12, 10)
        v.setSpacing(2)
        when = when or datetime.now()
        stamp = when.strftime("%I:%M %p").lstrip("0")
        if when.date() != datetime.now().date():
            stamp = when.strftime("%d %b, ") + stamp
        name = QLabel(f"{who.upper()}   ·   {stamp}", objectName="who")
        if mine:
            name.setAlignment(Qt.AlignmentFlag.AlignRight)
        v.addWidget(name)
        msg = QLabel(text, objectName="msg")
        msg.setTextFormat(Qt.TextFormat.PlainText)
        msg.setWordWrap(True)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(msg)
        width = self._max_width(0.78)
        self._fit(msg, width - 26)
        bubble.setMaximumWidth(width)
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self._add_row(bubble, "right" if mine else "left")

    def add_tool(self, name: str, args: dict, result: dict) -> None:
        ok = "error" not in result
        detail = str(result.get("error") or result.get("result") or "")
        if not result.get("error"):
            first = detail.split(". ")[0]
            if first != detail:
                detail = first + "."  # first sentence only; the rest is for the model
        if len(detail) > 90:
            detail = detail[:88].rsplit(" ", 1)[0] + "…"
        args_text = ", ".join(f"{k}={str(v)[:30]}" for k, v in args.items())
        line = f"{tool_icon(name)}  {pretty_tool(name)}" + (f"  ({args_text})" if args_text else "")
        line += ("  ✓" if ok else "  ✗") + (f"  {detail}" if detail else "")
        chip = QLabel(line, objectName="toolOk" if ok else "toolErr")
        chip.setTextFormat(Qt.TextFormat.PlainText)
        chip.setWordWrap(True)
        width = self._max_width(0.9)
        self._fit(chip, width - 22)
        chip.setMaximumWidth(width)
        self._add_row(chip, "left")

    def add_image(self, path: str, caption: str, accent: str) -> None:
        box = QFrame(objectName="bubbleAi")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        v.addWidget(ImageThumb(path, accent))
        cap = QLabel(caption, objectName="muted")
        cap.setTextFormat(Qt.TextFormat.PlainText)
        v.addWidget(cap)
        self._add_row(box, "left")

    def add_system(self, text: str) -> None:
        label = QLabel(text, objectName="muted")
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._add_row(label, "center")

    def toPlainText(self) -> str:  # noqa: N802 - same name as QTextBrowser
        """All shown text, top to bottom."""
        return "\n".join(label.text() for label in self.widget().findChildren(QLabel))

    def clear(self) -> None:
        while self.col.count() > 1:
            w = self.col.takeAt(1).widget()
            if w:
                w.deleteLater()


class MainWindow(QMainWindow):
    def __init__(self, app: Jarvis, bridge: Bridge):
        super().__init__()
        self.app = app
        self.bridge = bridge
        self.assistant = None
        self.thread: threading.Thread | None = None
        self._closing = False
        self._last_state = ""
        self.accent = valid_color(app.settings.theme_color)
        self.history = ChatHistory(home_dir() / "history.jsonl")
        self.mini: MiniWindow | None = None
        self.hotkey: GlobalHotkey | None = None
        bridge.hotkey.connect(self._on_hotkey)

        bridge.event.connect(self._on_event)
        bridge.confirm.connect(self._on_confirm)
        app.events.subscribe(self._forward)

        self.setMinimumSize(1000, 660)
        self.setAcceptDrops(True)
        self._build()
        self._build_tray()
        self._shortcuts()
        self._apply_theme()
        self._restore_history()

    def _forward(self, kind: str, data: dict) -> None:
        """Worker thread -> Qt thread. After the window is gone, events are dropped quietly."""
        try:
            self.bridge.event.emit(kind, data)
        except RuntimeError:
            pass

    # -- layout -------------------------------------------------------------------
    def _build(self) -> None:
        root = QWidget(objectName="root")
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        # left: orb + captions + HUD + controls
        left = QFrame(objectName="panel")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(18, 18, 18, 14)
        self.title = QLabel(objectName="title")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(self.title)
        self.clock = QLabel(objectName="muted")
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(self.clock)
        self.big_orb = Orb(self.accent, face=self.app.settings.orb_face)
        self.orb = OrbGroup(self.big_orb)  # every orb call also reaches the mini orb
        lv.addWidget(self.big_orb, 1)

        state_row = QHBoxLayout()
        state_row.addStretch(1)
        self.dot = StatusDot()
        state_row.addWidget(self.dot)
        self.state_label = QLabel("CONNECTING", objectName="state")
        state_row.addWidget(self.state_label)
        state_row.addStretch(1)
        lv.addLayout(state_row)
        self.hint = QLabel("", objectName="muted")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(self.hint)

        self.caption = QLabel("", objectName="caption")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setFixedHeight(44)
        lv.addWidget(self.caption)

        self.stats = StatsBar(self.accent)
        lv.addWidget(self.stats)

        controls = QHBoxLayout()
        self.mute_btn = QPushButton("🎙  Mute")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setToolTip("Mute the microphone (Ctrl+M)")
        self.mute_btn.toggled.connect(self._toggle_mute)
        self.sleep_btn = QPushButton("💤  Sleep")
        self.sleep_btn.setToolTip("Put Jarvis to sleep / wake it up (Ctrl+Space)")
        self.sleep_btn.clicked.connect(self._toggle_sleep)
        settings_btn = QPushButton("⚙  Settings")
        settings_btn.setToolTip("Settings (Ctrl+,)")
        settings_btn.clicked.connect(self.open_settings)
        mini_btn = QPushButton("🗗  Mini")
        mini_btn.setToolTip("Floating mini orb that stays on top (Ctrl+Shift+M)")
        mini_btn.clicked.connect(self.toggle_mini)
        for b in (self.mute_btn, self.sleep_btn, mini_btn, settings_btn):
            controls.addWidget(b)
        lv.addLayout(controls)
        credit = QLabel(f"Jarvis v{__version__} · by Musaddiq Khan", objectName="muted")
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lv.addWidget(credit)
        outer.addWidget(left, 5)

        # right: header + banner + conversation + quick actions + input
        right = QFrame(objectName="panel")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(18, 18, 18, 18)
        header = QHBoxLayout()
        header.addWidget(QLabel("CONVERSATION", objectName="section"))
        header.addStretch(1)
        clear = QPushButton("🧹 Clear", objectName="small")
        clear.setToolTip("Clear the conversation view (Ctrl+L)")
        clear.clicked.connect(self._clear_chat)
        header.addWidget(clear)
        rv.addLayout(header)

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
        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(lambda: self._answer_confirm(False, timed_out=True))
        rv.addWidget(self.banner)

        self.chat = ChatView()
        rv.addWidget(self.chat, 1)
        self.typing = TypingDots()
        self.typing.hide()
        rv.addWidget(self.typing)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        for label, prompt in QUICK_ACTIONS:
            b = QPushButton(label, objectName="chip")
            b.setToolTip(prompt)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, t=prompt: self._send(t))
            chips.addWidget(b)
        chips.addStretch(1)
        rv.addLayout(chips)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a message, or just talk...  (Ctrl+K)")
        self.input.returnPressed.connect(self._send_typed)
        self.input.installEventFilter(self)
        self._history: list[str] = []
        self._history_pos = 0
        send = QPushButton("Send ➤", objectName="primary")
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

    def _build_tray(self) -> None:
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(self)
        menu = QMenu()
        for text, fn in (("Show", self._show_window), ("Mini mode", self.toggle_mini),
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
                         ("Ctrl+K", self.input.setFocus), ("Ctrl+Shift+M", self.toggle_mini)):
            QShortcut(QKeySequence(keys), self, activated=fn)

    def _open_logs(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir())))

    # -- input history and drag & drop ---------------------------------------------
    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        from PyQt6.QtCore import QEvent

        if obj is self.input and event.type() == QEvent.Type.KeyPress and self._history:
            if event.key() == Qt.Key.Key_Up:
                self._history_pos = max(0, self._history_pos - 1)
            elif event.key() == Qt.Key.Key_Down:
                self._history_pos = min(len(self._history), self._history_pos + 1)
            else:
                return False
            hist = self._history + [""]
            self.input.setText(hist[self._history_pos])
            return True
        return False

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if not paths:
            return
        for path in paths[:3]:
            if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                self.chat.add_image(path, Path(path).name, self.accent)
        names = "; ".join(paths[:3])
        self._send(f"I dropped this file into the window: {names}. Read or look at it and tell me "
                   f"briefly what it is, then ask what I want to do with it.")

    # -- mini mode and hotkey -------------------------------------------------------
    def toggle_mini(self) -> None:
        if self.mini and self.mini.isVisible():
            self._show_window()
            return
        if self.mini is None:
            self.mini = MiniWindow(self.accent, self.app.settings.orb_face, self._show_window, [
                ("Wake / Sleep", self._toggle_sleep), ("Mute / Unmute", self.mute_btn.toggle),
                ("Quit", self.close)])
            self.mini.orb.set_state(self._last_state or "connecting")
            self.orb.orbs.append(self.mini.orb)
        self.mini.orb._timer.start(16)
        self.hide()
        self.mini.show_at_saved_place()

    def _leave_mini(self) -> None:
        if self.mini and self.mini.isVisible():
            self.mini.hide()
            self.mini.orb._timer.stop()  # no drawing while hidden

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

    def _show_window(self) -> None:
        self._leave_mini()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _apply_theme(self) -> None:
        self.accent = valid_color(self.app.settings.theme_color)
        QApplication.instance().setStyleSheet(stylesheet(self.accent))
        self.orb.set_accent(self.accent)
        self.orb.set_face(self.app.settings.orb_face)
        if self.mini:
            self.mini.set_accent(self.accent)
        self.stats.accent = self.accent
        name = self.app.settings.assistant_name
        self.title.setText(name.upper())
        self.setWindowTitle(name)
        icon = app_icon(self.accent)
        self.setWindowIcon(icon)
        if self.tray:
            self.tray.setIcon(icon)
            self.tray.setToolTip(name)
        self._set_state_style(self._last_state or "connecting")

    def _set_state_style(self, state: str) -> None:
        color = {"thinking": WARN, "muted": DANGER, "sleeping": MUTED, "connecting": MUTED,
                 "offline": MUTED, "speaking": OK}.get(state, self.accent)
        self.state_label.setStyleSheet(f"color:{color};")
        self.dot.color = QColor(color)

    def _update_clock(self) -> None:
        self.clock.setText(datetime.now().strftime("%A · %d %B · %I:%M:%S %p"))

    # -- assistant lifecycle ------------------------------------------------------
    def start_assistant(self) -> None:
        if not self.app.settings.effective_api_key:
            self._add_system("Add your Gemini API key in Settings to start.")
            return
        from jarvis.live import LiveAssistant

        self.assistant = LiveAssistant(self.app)
        self.thread = self.assistant.run_in_thread()
        self._add_system(f"Starting... say \"{self.app.settings.assistant_name}\", or type below.")

    # -- events ---------------------------------------------------------------------
    def _on_event(self, kind: str, data: dict) -> None:
        if kind == "level":
            self.orb.set_level(data["value"])
        elif kind == "state":
            self._on_state(data["state"])
        elif kind == "user":
            self._add_bubble("You", data["text"], mine=True)
            self.history.add("user", data["text"])
        elif kind == "assistant":
            self._add_bubble(self.app.settings.assistant_name, data["text"])
            self.history.add("assistant", data["text"])
        elif kind == "caption":
            self._on_caption(data.get("who", ""), data.get("text", ""))
        elif kind == "tool_start":
            name = data["name"]
            self.orb.action(f"{tool_icon(name)}  {pretty_tool(name)}...")
        elif kind == "tool":
            result = data.get("result", {})
            ok = "error" not in result
            self.orb.pulse(ok)
            self.orb.action(f"{'✓' if ok else '✗'}  {pretty_tool(data['name'])}", OK if ok else DANGER)
            self.chat.add_tool(data["name"], data.get("args", {}), result)
            image = result.get("image") if ok else None
            if image and Path(image).exists():
                caption = Path(image).name if "Screenshot" in Path(image).name else "What I saw"
                self.chat.add_image(image, f"🖼  {caption}  ·  click to open", self.accent)
        elif kind == "crash":
            self._add_system(f"⚠ Something went wrong: {data['text']}  (details in the log: tray → Open logs folder)")
        elif kind == "log":
            text = data["text"]
            self._add_system(text)
            if text.startswith("🔔") and self.tray and not self.isActiveWindow():
                self.tray.showMessage(self.app.settings.assistant_name, text[1:].strip())

    def _on_state(self, state: str) -> None:
        prev, self._last_state = self._last_state, state
        self.orb.set_state(state)
        self.state_label.setText(STATE_LABELS.get(state, state.upper()))
        self._set_state_style(state)
        self.sleep_btn.setText("☀  Wake" if state == "sleeping" else "💤  Sleep")
        name = self.app.settings.assistant_name
        if state == "thinking":
            self.typing.show_text(f"{name} is working on it...", WARN)
        elif state == "connecting":
            self.typing.show_text("Connecting...", MUTED)
        else:
            self.typing.hide_now()
        if prev == "sleeping" and state in AWAKE_STATES:
            self.orb.flash()
        if state == "sleeping":
            how = self.assistant.how_to_wake() if self.assistant else "press Wake"
            self.hint.setText(how[0].upper() + how[1:])
            self.caption.setText("")
        else:
            self.hint.setText("")

    def _on_caption(self, who: str, text: str) -> None:
        if who == "user" and text and self._last_state != "thinking":
            self.typing.show_text(f"{self.app.settings.assistant_name} is listening...", self.accent)
        if not text:
            QTimer.singleShot(1800, self._clear_caption)
            self._caption_live = False
            return
        self._caption_live = True
        if len(text) > 140:
            text = "…" + text[-140:]
        color = OK if who == "assistant" else "#d8e6ef"
        icon = "🔊" if who == "assistant" else "🎙"
        self.caption.setStyleSheet(f"color:{color};")
        self.caption.setText(f"{icon}  {text}")
        if self.mini:
            self.mini.set_caption(f"{icon} {text}")

    def _clear_caption(self) -> None:
        if self._caption_live:
            return
        self.caption.setText("")
        if self.mini:
            self.mini.set_caption("")

    _caption_live = False

    def _add_bubble(self, who: str, text: str, mine: bool = False) -> None:
        if not text or text.startswith("(system"):
            return  # app-generated instructions are not shown as user messages
        self.chat.add_message(who, text, mine)

    def _add_system(self, text: str) -> None:
        self.chat.add_system(text)

    def _clear_chat(self) -> None:
        self.chat.clear()
        self.history.clear()

    def _restore_history(self) -> None:
        items = self.history.recent(30)
        if not items:
            return
        name = self.app.settings.assistant_name
        for item in items:
            mine = item["role"] == "user"
            self.chat.add_message("You" if mine else name, item["text"], mine,
                                  when=datetime.fromtimestamp(item.get("t", 0) or 0))
        self.chat.add_system("──────  earlier conversation  ·  Ctrl+L clears it  ──────")

    # -- controls ---------------------------------------------------------------
    def _send_typed(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        if not self._history or self._history[-1] != text:
            self._history = (self._history + [text])[-50:]
        self._history_pos = len(self._history)
        self._send(text)

    def _send(self, text: str) -> None:
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
        dialog = SettingsDialog(self.app, self, first_run=first_run is True)
        if not dialog.exec():
            return False
        changed = apply_settings(self.app.settings, dialog.result_settings)
        save_settings(self.app.settings)
        if changed & {"theme_color", "assistant_name", "orb_face"}:
            self._apply_theme()
        if self.assistant is not None and changed & {"sound_effects", "sound_volume"}:
            self.assistant.sfx.enabled = self.app.settings.sound_effects
            self.assistant.sfx.volume = self.app.settings.sound_volume
            self.assistant.sfx.play("success")
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
        self._banner_timer.start(54_000)  # the tool stops waiting after ~55 s
        self.app.events.emit("confirm_needed", question=question)
        self._show_window()

    def _answer_confirm(self, yes: bool, timed_out: bool = False) -> None:
        self._banner_timer.stop()
        self.banner.hide()
        reply, self._pending_reply = self._pending_reply, None
        if reply:
            reply(yes)
        if timed_out:
            self._add_system("Confirmation timed out - cancelled.")

    # -- close --------------------------------------------------------------------
    def _quit(self) -> None:
        """The app doesn't quit on its own (mini mode hides this window), so quit explicitly."""
        self._quitting = True
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
    if not app.settings.effective_api_key:
        QTimer.singleShot(200, lambda: window.open_settings(first_run=True) or QMessageBox.information(
            window, "Jarvis", "Jarvis needs a Gemini API key. Open Settings any time to add it."))
    else:
        window.start_assistant()
    return qt.exec()
