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
    QStackedWidget,
    QApplication, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSystemTrayIcon, QVBoxLayout, QWidget,
)

from jarvis import __version__
from jarvis.app import Jarvis, create_app
from jarvis.config import home_dir, load_settings, save_settings
from jarvis.history import ChatHistory
from jarvis.logs import log_dir, on_crash, setup_logging
from jarvis.ui.fx import AnimatedBackground, ToastHost, Typewriter, add_glow, grow_in
from jarvis.ui.hotkey import HOTKEY_TEXT, GlobalHotkey
from jarvis.ui.mini import MiniWindow, OrbGroup
from jarvis.ui.orb import Orb
from jarvis.ui.richtext import has_markup, to_html
from jarvis.ui.settings_dialog import SettingsDialog, apply_settings
from jarvis.ui import theme
from jarvis.ui.theme import DANGER, OK, STATE_LABELS, WARN

RESTART_KEYS = {"voice", "speaking_style", "expressive_voice", "live_model", "input_device", "output_device", "wake_word",
                "assistant_name", "user_name", "api_key", "allow_interrupt"}
MAX_CHAT_ROWS = 250
DOC_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".xlsm", ".txt", ".md", ".csv"}
THUMB_WIDTH = 340
INSTANCE_KEY = f"jarvis-assistant-{getpass.getuser()}"
log = logging.getLogger("jarvis.ui")
AWAKE_STATES = {"listening", "speaking", "muted", "thinking"}

QUICK_ACTIONS = [
    ("🖥  Screen", "What's on my screen?"),
    ("🌦  Weather", "What's the weather today?"),
    ("📰  News", "Give me today's top news in two lines."),
    ("📝  My list", "What's on my to-do list, and do I have any reminders or timers?"),
    ("⏱  5 min", "Set a 5 minute timer."),
    ("😄  Joke", "Tell me a short joke."),
]

THANKS_WORDS = ("thank", "thanks", "shukriya", "dhanyavad", "dhanyawad", "love you", "awesome", "great job",
                "well done", "you're the best", "nice work", "शुक्रिया", "धन्यवाद", "شکریہ")
PRAISE_WORDS = ("wow", "awesome", "amazing", "great", "nice", "cool", "perfect", "superb", "mast", "zabardast",
                "badhiya", "kamaal", "shandaar", "बढ़िया", "ज़बरदस्त", "कमाल")
CELEBRATE_WORDS = ("congratulations", "congrats", "mubarak", "well done", "great job", "shabash", "proud of you",
                   "bahut badhiya", "excellent", "बधाई", "मुबारक", "شاباش", "مبارک")
JOKE_WORDS = ("joke", "mazak", "mazaak", "chutkula", "funny", "मज़ाक", "चुटकुला")

TOOL_ICONS = {
    "open": "🚀", "close": "✖", "volume": "🔊", "brightness": "☀", "media": "🎵", "play": "▶",
    "screenshot": "📸", "screen": "👁", "camera": "📷", "weather": "🌦", "search": "🔎", "news": "📰", "remind": "⏰",
    "remember": "🧠", "recall": "🧠", "memory": "🧠", "file": "📁", "folder": "📁", "whatsapp": "💬",
    "message": "💬", "email": "✉", "code": "💻", "type": "⌨", "key": "⌨", "clipboard": "📋",
    "power": "⏻", "wifi": "📶", "timer": "⏱", "focus": "🎯", "note": "📝", "routine": "🔁",
    "history": "🗂", "time": "🕒", "system": "📊", "undo": "↩", "sleep": "💤", "window": "🪟",
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
            color = QColor(DANGER if bad else WARN if (name != "BAT" and value > 70) else theme.readable(self.accent))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(x, 0, cell, 14), Qt.AlignmentFlag.AlignLeft, name)
            p.setPen(color)
            p.drawText(QRectF(x, 0, cell, 14), Qt.AlignmentFlag.AlignRight, f"{extra}{value:.0f}%")
            track = QRectF(x, 19, cell, 5)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.P.track)
            p.drawRoundedRect(track, 2.5, 2.5)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(x, 19, cell * min(value, 100) / 100, 5), 2.5, 2.5)
        p.end()


class StatusDot(QWidget):
    """Pulsing dot next to the state label."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color = QColor(theme.MUTED)
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
        self.color = QColor(theme.MUTED)
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
        p.setPen(QColor(theme.MUTED))
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
        self.animate = True  # off while old history is put back
        self._last_day = None
        self.query = ""
        self.verticalScrollBar().rangeChanged.connect(self._on_range)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.jump = QPushButton("↓  Latest", self.viewport(), objectName="jump")
        self.jump.setCursor(Qt.CursorShape.PointingHandCursor)
        self.jump.clicked.connect(self.scroll_to_end)
        self.jump.hide()
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._chat_menu)

    def scroll_to_end(self) -> None:
        self._stick = True
        bar = self.verticalScrollBar()
        anim = QPropertyAnimation(bar, b"value", self)
        anim.setDuration(350)
        anim.setEndValue(bar.maximum())
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._scroll_anim = anim
        self.jump.hide()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.jump.adjustSize()
        self.jump.move(self.viewport().width() - self.jump.width() - 14,
                       self.viewport().height() - self.jump.height() - 10)

    def _on_scroll(self, value: int) -> None:
        self._stick = value >= self.verticalScrollBar().maximum() - 30
        if self._stick:
            self.jump.hide()

    def _chat_menu(self, pos) -> None:
        menu = QMenu(self)
        target = self.childAt(pos)
        bubble_text = None
        while target is not None and target is not self:
            if getattr(target, "_copy_text", None) is not None:
                bubble_text = target._copy_text
                break
            target = target.parentWidget()
        if bubble_text is not None:
            menu.addAction("Copy message").triggered.connect(lambda: QApplication.clipboard().setText(bubble_text))
        menu.addAction("Copy whole chat").triggered.connect(
            lambda: QApplication.clipboard().setText(self.toPlainText()))
        menu.exec(self.mapToGlobal(pos))

    def _day_separator(self, when: datetime) -> None:
        day = when.date()
        if day == self._last_day:
            return
        self._last_day = day
        today = datetime.now().date()
        text = "TODAY" if day == today else "YESTERDAY" if (today - day).days == 1 else when.strftime("%a, %d %b").upper()
        line = QLabel(f"──────   {text}   ──────", objectName="dayline")
        line.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row = self._add_row(line, "center", animate=False)
        row._kind = "day"

    def filter(self, query: str) -> int:
        """Show only rows containing the query; returns how many messages match."""
        self.query = query.strip().lower()
        matches = 0
        for i in range(1, self.col.count()):
            row = self.col.itemAt(i).widget()
            if row is None:
                continue
            if not self.query:
                row.setVisible(True)
                continue
            if getattr(row, "_kind", "") == "day":
                row.setVisible(False)
                continue
            hit = self.query in getattr(row, "_search", "").lower()
            row.setVisible(hit)
            matches += hit
        return matches

    def _on_range(self, _min: int, maximum: int) -> None:
        if self._stick:
            self.verticalScrollBar().setValue(maximum)

    def _add_row(self, widget: QWidget, align: str, text: str = "", animate: bool = True) -> QWidget:
        row = QWidget(objectName="chatRow")
        row._search = text
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
        if self.query and self.query not in text.lower():
            row.hide()  # a search is open: new rows that don't match stay hidden
        if not self._stick and self.animate:
            self.jump.show()
            self.jump.raise_()
        if not (self.animate and animate):
            return row
        grow_in(row)
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
        self._stick = self._stick or not self.animate
        return row

    def _max_width(self, share: float) -> int:
        return max(240, int(self.viewport().width() * share))

    @staticmethod
    def _fit(label: QLabel, max_width: int, text: str | None = None) -> None:
        """Short text gets a snug label, long text wraps at max_width (QLabel can't decide alone)."""
        text = label.text() if text is None else text
        longest = max((label.fontMetrics().horizontalAdvance(line) for line in text.splitlines()), default=0)
        label.setMinimumWidth(min(longest + 6, max_width))

    def add_message(self, who: str, text: str, mine: bool, when: datetime | None = None,
                    typewriter: bool = False) -> None:
        when = when or datetime.now()
        self._day_separator(when)
        bubble = QFrame(objectName="bubbleMe" if mine else "bubbleAi")
        bubble._copy_text = text
        v = QVBoxLayout(bubble)
        v.setContentsMargins(12, 8, 12, 10)
        v.setSpacing(2)
        stamp = when.strftime("%I:%M %p").lstrip("0")
        if when.date() != datetime.now().date():
            stamp = when.strftime("%d %b, ") + stamp
        name = QLabel(f"{who.upper()}   ·   {stamp}", objectName="who")
        if mine:
            name.setAlignment(Qt.AlignmentFlag.AlignRight)
        v.addWidget(name)
        msg = QLabel(objectName="msg")
        msg.setWordWrap(True)
        msg.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)  # the chat's own menu (Copy) shows
        html_text = to_html(text, theme.readable(theme.ACCENT)) if has_markup(text) else None
        if html_text is not None:
            msg.setTextFormat(Qt.TextFormat.RichText)
            msg.setText(html_text)
            msg.setOpenExternalLinks(True)
            msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        else:
            msg.setTextFormat(Qt.TextFormat.PlainText)
            msg.setText(text)
            msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(msg)
        width = self._max_width(0.78)
        self._fit(msg, width - 26, text)
        bubble.setMaximumWidth(width)
        bubble.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self._add_row(bubble, "right" if mine else "left", text=f"{who} {text}")
        if typewriter and self.animate:
            msg._typewriter = Typewriter(msg, text, final_html=html_text)

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
        self._add_row(chip, "left", text=line)

    def add_image(self, path: str, caption: str, accent: str) -> None:
        box = QFrame(objectName="bubbleAi")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        v.addWidget(ImageThumb(path, accent))
        cap = QLabel(caption, objectName="muted")
        cap.setTextFormat(Qt.TextFormat.PlainText)
        v.addWidget(cap)
        self._add_row(box, "left", text=caption)

    def add_system(self, text: str) -> None:
        label = QLabel(text, objectName="muted")
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._add_row(label, "center", text=text)

    def toPlainText(self) -> str:  # noqa: N802 - same name as QTextBrowser
        """All shown text, top to bottom."""
        def full(label: QLabel) -> str:  # a message still typing itself out counts in full
            writer = getattr(label, "_typewriter", None)
            if writer:
                return writer.text
            parent = label.parentWidget()
            if label.textFormat() == Qt.TextFormat.RichText and getattr(parent, "_copy_text", None) is not None:
                return parent._copy_text
            return label.text()

        return "\n".join(full(label) for label in self.widget().findChildren(QLabel))

    def clear(self) -> None:
        self._last_day = None
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
        self.accent = theme.accent_for(app.settings.theme_name, app.settings.theme_color)
        self.history = app.history or ChatHistory(home_dir() / "history.jsonl")  # saved by the core
        self.mini: MiniWindow | None = None
        self.hotkey: GlobalHotkey | None = None
        bridge.hotkey.connect(self._on_hotkey)

        bridge.event.connect(self._on_event)
        bridge.confirm.connect(self._on_confirm)
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
        self.practice_btn.setText(f"🎓  {app.state.get('practice_target', 'English')} Practice")
        if not app.settings.effective_api_key and app.practice_on:
            QTimer.singleShot(0, lambda: self.set_practice(True))

    def _planner_follow(self, kind: str) -> None:
        """Keep an open planner in sync when Jarvis changes lists by voice."""
        planner = getattr(self, "planner", None)
        if planner is not None and planner.isVisible() and kind in ("tool", "timer", "timer_end"):
            QTimer.singleShot(0, planner.refresh)  # after this event has been handled
        dashboard = getattr(self, "dashboard", None)
        if dashboard is not None and dashboard.isVisible() and kind == "tool":
            QTimer.singleShot(0, dashboard.refresh)

    def _forward(self, kind: str, data: dict) -> None:
        """Worker thread -> Qt thread. After the window is gone, events are dropped quietly."""
        try:
            self.bridge.event.emit(kind, data)
        except RuntimeError:
            pass

    # -- layout -------------------------------------------------------------------
    def _build(self) -> None:
        root = AnimatedBackground(self.accent)
        root.setObjectName("root")
        self.background = root
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
        self.clock.setWordWrap(True)
        lv.addWidget(self.clock)
        pill_row = QHBoxLayout()
        pill_row.addStretch(1)
        self.net_pill = QLabel("")
        self.net_pill.setToolTip("Connection to the AI")
        pill_row.addWidget(self.net_pill)
        pill_row.addStretch(1)
        lv.addLayout(pill_row)
        self._set_net("offline" if not self.app.settings.effective_api_key else "connecting")
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
        from jarvis.ui.today import TodayStrip

        self.today = TodayStrip(self.app, lambda: self.big_orb.timers, lambda q: self._send(q))
        self.today.on_badges = lambda: self.toggle_badges()
        lv.addWidget(self.today)

        from jarvis.ui.flow import flow_widget

        controls_box, controls = flow_widget(6, center=True)  # 2 rows when the window is narrow
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
        lv.addWidget(controls_box)
        self.practice_btn = QPushButton("🎓  English Practice", objectName="primary")
        self.practice_btn.setToolTip("Practice speaking, listening, reading and grammar with your tutor. "
                                     "Jarvis's computer commands are off until you come back.")
        self.practice_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.practice_btn.clicked.connect(lambda: self.set_practice(not self.practicing))
        lv.addWidget(self.practice_btn)
        credit = QLabel(f"Jarvis v{__version__} · by Musaddiq Khan", objectName="muted")
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit.setWordWrap(True)
        lv.addWidget(credit)
        outer.addWidget(left, 5)

        # right: header + banner + conversation + quick actions + input
        right = QFrame(objectName="panel")
        self.right_panel = right
        rv = QVBoxLayout(right)
        rv.setContentsMargins(18, 18, 18, 18)
        header = QHBoxLayout()
        self.header_label = QLabel("CONVERSATION", objectName="section")
        header.addWidget(self.header_label)
        header.addStretch(1)
        dash_btn = QPushButton("📊", objectName="small")
        dash_btn.setToolTip("Dashboard: week, habits, money, goals (Ctrl+D)")
        dash_btn.clicked.connect(lambda: self.toggle_dashboard())
        header.addWidget(dash_btn)
        planner_btn = QPushButton("📋", objectName="small")
        planner_btn.setToolTip("Planner: to-dos, reminders, timers, routines (Ctrl+P)")
        planner_btn.clicked.connect(self.toggle_planner)
        header.addWidget(planner_btn)
        badges_btn = QPushButton("🏆", objectName="small")
        badges_btn.setToolTip("Achievements")
        badges_btn.clicked.connect(self.toggle_badges)
        header.addWidget(badges_btn)
        help_btn = QPushButton("?", objectName="small")
        help_btn.setToolTip("What can I say? (F1)")
        help_btn.clicked.connect(self.toggle_help)
        header.addWidget(help_btn)
        search_btn = QPushButton("🔍", objectName="small")
        search_btn.setToolTip("Search the chat (Ctrl+F)")
        search_btn.clicked.connect(self.open_search)
        header.addWidget(search_btn)
        export_btn = QPushButton("⤓", objectName="small")
        export_btn.setToolTip("Save the conversation to Documents > Jarvis Chats")
        export_btn.clicked.connect(self.export_chat)
        header.addWidget(export_btn)
        self.mode_btn = QPushButton("🌙", objectName="small")
        self.mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode_btn.clicked.connect(self.cycle_theme_mode)
        header.addWidget(self.mode_btn)
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

        self.search_bar = QFrame(objectName="card")
        sb = QHBoxLayout(self.search_bar)
        sb.setContentsMargins(10, 4, 6, 4)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search messages...")
        self.search.textChanged.connect(self._on_search)
        self.search.setStyleSheet("background: transparent; border: none; padding: 4px;")
        sb.addWidget(QLabel("🔍"))
        sb.addWidget(self.search, 1)
        self.search_count = QLabel("", objectName="muted")
        sb.addWidget(self.search_count)
        close_search = QPushButton("✕", objectName="small")
        close_search.clicked.connect(self.close_search)
        sb.addWidget(close_search)
        self.search_bar.hide()
        rv.addWidget(self.search_bar)
        self.chat = ChatView()
        self.practice = None  # the practice screen is built the first time practice mode is turned on
        self.stack = QStackedWidget()
        self.stack.addWidget(self.chat)
        rv.addWidget(self.stack, 1)
        self.typing = TypingDots()
        self.typing.hide()
        rv.addWidget(self.typing)

        from jarvis.ui.flow import flow_widget

        self.chips, chips = flow_widget(6)  # wraps to a second line in a narrow window
        for label, prompt in QUICK_ACTIONS:
            b = QPushButton(label, objectName="chip")
            b.setToolTip(prompt)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, t=prompt: self._send(t))
            chips.addWidget(b)
        rv.addWidget(self.chips)

        self.doc_chip = QPushButton(objectName="chip")
        self.doc_chip.setToolTip("Questions you ask now are answered from this document. Click to close it.")
        self.doc_chip.setCursor(Qt.CursorShape.PointingHandCursor)
        self.doc_chip.clicked.connect(lambda: self.app.registry.call("close_document", {}, self.app.ctx))
        self.doc_chip.hide()
        rv.addWidget(self.doc_chip, 0, Qt.AlignmentFlag.AlignLeft)

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
        self.toasts = ToastHost(root)
        from jarvis.lang import reminder_language
        from jarvis.ui.reminder_popup import ReminderPopups

        self.reminders = ReminderPopups(
            lambda: self.accent,
            on_done=lambda rid: self._reminder_action("reminder_done", {"id": rid}),
            on_snooze=lambda rid, m: self._reminder_action("snooze_reminder", {"id": rid, "minutes": m}),
            on_cancel=lambda rid: self._reminder_action("cancel_reminder", {"id": rid}),
            on_open=self._show_window,
            language=lambda: reminder_language(self.app.settings, self.app.state))
        self._glows = [add_glow(b, self.accent, 20 if b.objectName() != "chip" else 14)
                       for b in root.findChildren(QPushButton)]
        self._glows.append(add_glow(self.input, self.accent, 18))

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._theme_timer = QTimer(self)
        self._theme_timer.timeout.connect(self._check_auto_theme)
        self._theme_timer.start(60_000)
        self._clock_timer.start(1000)
        self._update_clock()

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
        doc = next((p for p in paths if Path(p).suffix.lower() in DOC_SUFFIXES), None)
        if doc and len(paths) == 1:  # one document: open it for questions
            result = self.app.registry.call("open_document", {"path": doc}, self.app.ctx)
            if "error" in result:
                self._add_system(f"⚠ {result['error']}")
                return
            self._send(f"I opened the document '{Path(doc).name}'. Give me a two-sentence summary using "
                       "ask_document, then ask what I want to know about it.")
            return
        for path in paths[:3]:
            if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
                self.chat.add_image(path, Path(path).name, self.accent)
        names = "; ".join(paths[:3])
        self._send(f"I dropped this file into the window: {names}. Read or look at it and tell me "
                   f"briefly what it is, then ask what I want to do with it.")

    # -- practice mode -------------------------------------------------------------------
    practicing = False

    def set_practice(self, on: bool) -> None:
        """English practice mode: the tutor takes over; Jarvis's commands are off until switched back."""
        if on == self.practicing:
            return
        self.practicing = on
        self.app.state.set("practice_mode", on)
        if on and self.practice is None:
            from jarvis.ui.practice import PracticePanel  # first use: only now does practice code load

            self.practice = PracticePanel(self.app, None)
            self.stack.addWidget(self.practice)
        if self.practice is not None:
            self.app.english.set_mode("practice" if on else "agent")
        if self.assistant is not None:
            self.assistant.set_mode("practice" if on else "agent")
            if on and not self.assistant.awake:
                self.assistant.wake_up()  # the user wants to practise now
        if self.practice is not None:
            # announce(): the tutor is heard even if Jarvis dozes off between tasks
            self.practice.say = self.assistant.announce if self.assistant is not None else None
        self.stack.setCurrentWidget(self.practice if on else self.chat)
        self.orb.set_persona("tutor" if on else "")
        self.chips.setVisible(not on)
        self.header_label.setText("LANGUAGE PRACTICE" if on else "CONVERSATION")
        if on:
            self.close_search()
        target = self.app.english.profile.get("target", "English")
        self.app.state.set("practice_target", target)
        if on:
            self.practice_btn.setText("🤖  Back to Jarvis (agent mode)")
            self.practice_btn.setObjectName("modeOn")
            self.input.setPlaceholderText(f"Talk or type in {target} - your tutor will help...")
            self.title.setText(f"{self.app.settings.assistant_name.upper()}  ·  TUTOR")
            self.practice.reload()
            self.toast("🎓", f"{target} practice mode - commands are off until you press Back to Jarvis.",
                       self.accent, 4)
        else:
            self.practice_btn.setText(f"🎓  {target} Practice")
            self.practice_btn.setObjectName("primary")
            self.input.setPlaceholderText("Type a message, or just talk...  (Ctrl+K)")
            self.title.setText(self.app.settings.assistant_name.upper())
            self.toast("🤖", "Back to Jarvis - commands are on again.", self.accent, 3)
        self.practice_btn.style().unpolish(self.practice_btn)
        self.practice_btn.style().polish(self.practice_btn)

    # -- voice control of this window ------------------------------------------------------
    def _shown_window(self):
        return self.mini if self.mini and self.mini.isVisible() else self

    def ui_command(self, action: str, theme_name: str = "", opts: dict | None = None) -> None:
        s = self.app.settings
        opts = opts or {}
        if action in ("widget_on", "widget_off"):
            s.desktop_widget = action == "widget_on"
            corner = opts.get("place", "")
            if corner in ("top_right", "top_left", "bottom_right", "bottom_left"):
                s.widget_corner = corner
            save_settings(s)
            self.apply_desktop_widget()
            return
        if action in ("minimize", "maximize", "move", "glide", "place", "bigger", "smaller", "stop"):
            self._window_motion(action, opts)
            return
        if action in ("night_mode", "day_mode", "auto_mode"):
            s.theme_mode = {"night_mode": "dark", "day_mode": "light", "auto_mode": "auto"}[action]
        elif action == "theme" and theme_name in theme.THEMES:
            s.theme_name = theme_name
            s.theme_color = theme.THEMES[theme_name][0]  # the theme's own accent
        elif action == "mini_mode":
            if not (self.mini and self.mini.isVisible()):
                self.toggle_mini()
            return
        elif action == "full_window":
            self._show_window()
            return
        elif action == "show_help":
            if not (getattr(self, "help", None) and self.help.isVisible()):
                self.toggle_help()
            return
        elif action == "show_achievements":
            if not (getattr(self, "badges", None) and self.badges.isVisible()):
                self.toggle_badges()
            return
        elif action == "dashboard":
            tab = theme_name if theme_name in ("week", "habits", "money", "goals") else None
            if not (getattr(self, "dashboard", None) and self.dashboard.isVisible()) or tab:
                self.toggle_dashboard(tab or "week")
            return
        elif action == "clear_chat":
            self.chat.clear()
            return
        else:
            return
        save_settings(s)
        self._apply_theme()
        self.orb.flash()

    def _window_motion(self, action: str, opts: dict) -> None:
        """Voice control of Jarvis' own window (or the mini orb when it's showing)."""
        if not hasattr(self, "mover"):
            from jarvis.ui.mover import WindowMover

            self.mover = WindowMover(self._shown_window)
        if action == "stop":
            self.mover.stop()
            return
        target = self._shown_window()
        if action == "minimize":
            self.mover.stop()
            if target is self:
                self.showMinimized()
            else:
                self.mini.hide()  # the orb goes away; Ctrl+Alt+J or the tray brings Jarvis back
            return
        if action == "maximize":
            self._show_window()
            self.showMaximized()
            return
        if target is self and self.isMinimized():
            self.showNormal()
        if action == "move":
            self.mover.nudge(opts.get("direction", ""), opts.get("amount") or "little")
        elif action == "glide":
            self.mover.glide(opts.get("direction", ""), opts.get("speed") or "normal")
        elif action == "place":
            self.mover.place(opts.get("place", ""))
        elif action in ("bigger", "smaller"):
            if self.mover.resize(action == "bigger") == "fixed":
                self.toast("🔍", "The mini orb has one size - say 'full window' for the big one.", self.accent, 3)
        if target is self.mini:
            from PyQt6.QtCore import QSettings

            QTimer.singleShot(600, lambda: QSettings("MusaddiqKhan", "Jarvis").setValue("mini_pos", self.mini.pos()))

    # -- day / night ------------------------------------------------------------------
    def cycle_theme_mode(self) -> None:
        s = self.app.settings
        s.theme_mode = {"dark": "light", "light": "auto"}.get(s.theme_mode, "dark")
        save_settings(s)
        self._apply_theme()
        label = {"dark": "Night mode", "light": "Day mode", "auto": "Auto: day by day, night by night"}[s.theme_mode]
        self.toast("🌓", label, self.accent, 2.5)

    def _check_auto_theme(self) -> None:
        """In auto mode, switch when the sun 'sets' or 'rises'."""
        if self.app.settings.theme_mode == "auto" and theme.resolve_mode("auto") != self._applied_mode:
            self._apply_theme()

    _applied_mode = "dark"

    # -- mini mode and hotkey -------------------------------------------------------
    def toggle_mini(self) -> None:
        if self.mini and self.mini.isVisible():
            self._show_window()
            return
        if self.mini is None:
            from jarvis.ui.mini import next_up

            self.mini = MiniWindow(self.accent, self.app.settings.orb_face, self._show_window, [
                ("Wake / Sleep", self._toggle_sleep), ("Mute / Unmute", self.mute_btn.toggle),
                ("Quit", self.close)], send=self._send, upcoming=lambda: next_up(self.app))
            for who, text in self._recent_for_mini():
                self.mini.add_message(who, text)
            self.mini.orb.set_state(self._last_state or "connecting")
            self.orb.orbs.append(self.mini.orb)
        self.mini.orb._timer.start(16)
        self.hide()
        self.mini.show_at_saved_place()

    def apply_desktop_widget(self) -> None:
        """Show, move or hide the desktop widget to match the settings."""
        s = self.app.settings
        if not s.desktop_widget:
            if self.desk is not None:
                self.desk.hide_from_desktop()
            return
        if self.desk is None:
            from jarvis.ui.desktop_widget import DesktopWidget

            self.desk = DesktopWidget(self.app, self.accent, s.orb_face, s.widget_corner)
            self.desk.orb.set_state(self._last_state or "connecting")
            self.orb.orbs.append(self.desk.orb)
        self.desk.set_accent(self.accent)
        self.desk.place(s.widget_corner)
        self.desk.show_on_desktop()

    desk = None

    def _recent_for_mini(self) -> list[tuple[str, str]]:
        history = self.app.history
        if not history:
            return []
        return [(i.get("role", ""), i.get("text", "")) for i in history.recent(6)
                if i.get("role") in ("user", "assistant")]

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

    def _language_notice(self) -> str:
        from jarvis.lang import prompt_rules

        return ("(system: the user just changed the language settings; this replaces earlier language rules) "
                + prompt_rules(self.app.settings).replace("\n", " ") + " Reply only with a short confirmation.")

    def _reminder_action(self, name: str, args: dict) -> None:
        """A button on a reminder pop-up: run the same tool the voice command uses (off the GUI thread)."""
        import threading

        def run():
            try:
                self.app.registry.call(name, args, self.app.ctx)
            except Exception:
                log.exception("reminder action %s failed", name)

        threading.Thread(target=run, daemon=True).start()

    def _show_window(self) -> None:
        self._leave_mini()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _apply_theme(self) -> None:
        s = self.app.settings
        self.accent = theme.accent_for(s.theme_name, s.theme_color)
        self._applied_mode = theme.resolve_mode(s.theme_mode)
        QApplication.instance().setStyleSheet(theme.apply(s.theme_name, s.theme_mode, self.accent))
        if hasattr(self, "mode_btn"):
            self.mode_btn.setText({"dark": "🌙", "light": "☀", "auto": "🌓"}.get(s.theme_mode, "🌙"))
            self.mode_btn.setToolTip(f"Theme: {s.theme_mode} - click to switch (dark → light → auto)")
        self.orb.set_accent(self.accent)
        self.orb.set_face(self.app.settings.orb_face)
        if self.mini:
            self.mini.set_accent(self.accent)
        if self.desk is not None:
            self.desk.set_accent(self.accent)
        self.stats.accent = self.accent
        self.background.set_accent(self.accent)
        for glow in self._glows:
            glow.set_color(self.accent)
        name = self.app.settings.assistant_name
        self.title.setText(name.upper() + ("  ·  TUTOR" if self.practicing else ""))
        self.setWindowTitle(name)
        icon = app_icon(self.accent)
        self.setWindowIcon(icon)
        if self.tray:
            self.tray.setIcon(icon)
            self.tray.setToolTip(name)
        self._set_state_style(self._last_state or "connecting")

    def _set_state_style(self, state: str) -> None:
        color = {"thinking": WARN, "muted": DANGER, "sleeping": theme.MUTED, "connecting": theme.MUTED,
                 "offline": theme.MUTED, "speaking": OK}.get(state, theme.readable(self.accent))
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
        if self.app.practice_on:  # continue where the user left off
            self.set_practice(True)
        self.thread = self.assistant.run_in_thread()
        self._add_system(f"Starting... say \"{self.app.settings.assistant_name}\", or type below.")

    # -- events ---------------------------------------------------------------------
    def _on_event(self, kind: str, data: dict) -> None:
        self._planner_follow(kind)
        if kind == "level":
            self.orb.set_level(data["value"])
        elif kind == "state":
            self._on_state(data["state"])
        elif kind == "user":
            self._add_bubble("You", data["text"], mine=True)
            if self.mini:
                self.mini.add_message("user", data["text"])
            self._react_to_user(data["text"])
        elif kind == "assistant":
            self._add_bubble(self.app.settings.assistant_name, data["text"], typewriter=True)
            if self.mini:
                self.mini.add_message("assistant", data["text"])
            if self._joke_pending:
                self._joke_pending = False
                self._emote("wink")
            elif any(w in data["text"].lower() for w in CELEBRATE_WORDS):
                self._emote("star")
        elif kind == "reminder_set":
            self.reminders.show("set", data["id"], data.get("message", ""), data.get("due", 0),
                                repeat=data.get("repeat", ""))
        elif kind == "reminder_due":
            self.reminders.show("due", data["id"], data.get("message", ""), data.get("due", 0),
                                late=data.get("late", False), repeat=data.get("repeat", ""))
            self.orb.flash()
        elif kind == "language":
            from jarvis.lang import label

            self.toast("🌐", f"Jarvis: {label(data.get('reply', 'auto'))}  ·  Reminders: "
                       f"{label(data.get('reminders', 'same'))}", theme.ACCENT, 4)
        elif kind == "motion" and data.get("state") == "running":
            self.toast("🖱", "Scrolling… say “ruko” or “stop” when it's enough", self.accent, 4)
        elif kind == "reminder_closed":
            self.reminders.close(data["id"])
        elif kind == "document":
            name = data.get("name", "")
            self.doc_chip.setText(f"📄  {name.replace('&', '&&')}  ·  ask anything about it   ✕")
            self.doc_chip.setVisible(bool(name))
        elif kind == "nudge":
            icon = {"eyes": "👀", "water": "💧", "recap": "🌙", "meeting": "📅"}.get(data.get("kind"), "💡")
            text = data.get("text", "")
            self.toast(icon, text[1:].strip() if text[:1] == icon[:1] else text, theme.ACCENT,
                       15 if data.get("kind") == "meeting" else 8)
            if self.tray and not self.isActiveWindow():
                self.tray.showMessage(self.app.settings.assistant_name, data.get("text", ""))
            if data.get("kind") in ("eyes", "water"):
                self.orb.transform("happy", 2.0)
        elif kind == "ui_command":
            self.ui_command(data.get("action", ""), data.get("theme", ""), data)
        elif kind == "achievement":
            from jarvis.ui.fx import Confetti

            Confetti(self.centralWidget())
            self.orb.transform("star", 3.2)
            self.toast(data.get("icon", "🏆"), f"Achievement unlocked: {data.get('title')} - {data.get('desc')}",
                       "#ffd23f", 6)
            self._add_system(f"🏆 Achievement unlocked: {data.get('icon', '')} {data.get('title')}")
        elif kind == "capture":
            if self.app.settings.notify_capture:
                self.flash_capture(data.get("what", "screen"))
        elif kind == "guardrail":
            if data.get("action") == "blocked":
                self.toast("🛡", "Blocked for your safety - Jarvis never sends or saves passwords, OTPs, "
                                 "card or ID numbers.", DANGER, 6)
                self._add_system("🛡 A risky action was blocked to protect your private information.")
            else:
                self.toast("🛡", f"Your {data.get('text', 'private info')} was hidden for your safety.", WARN, 5)
            self.orb.emote("surprised", 1.6)
        elif kind == "practice_answer":
            if data.get("correct"):
                self.orb.transform("happy", 1.3)
            else:
                self.orb.emote("sad", 1.6)
        elif kind.startswith("practice_") and self.practice is None:
            pass  # practice events only matter once the practice screen exists
        elif kind == "practice_mcq":
            self.practice.tutor_mcq(data["question"], data["options"], data["answer"], data.get("why", ""))
        elif kind == "practice_correction":
            self.practice.correction(data.get("said", ""), data["better"], data.get("tip", ""))
            self.orb.point()
        elif kind == "practice_progress":
            self.practice.progress(**data)
            share = data.get("score", 0) / (data.get("total") or 1)
            if share >= 0.9:
                self._emote("star", 3.2)
            elif share >= 0.6:
                self._emote("happy")
            if data.get("xp_gained"):
                self.toast("⭐", f"+{data['xp_gained']} XP  ·  🔥 {data.get('streak', 0)} day streak", OK, 3)
        elif kind == "timer":
            self.orb.set_timer(data["id"], data["ends"], data["total"], data.get("label", ""))
        elif kind == "timer_end":
            self.orb.end_timer(data["id"])
            if data.get("done"):
                self._emote("surprised")
                self.orb.flash()
                self.toast("⏱", f"{data.get('label') or 'Timer'} - time's up!", OK, 8)
                if self.tray and not self.isActiveWindow():
                    self.tray.showMessage(self.app.settings.assistant_name,
                                          f"⏱ {data.get('label') or 'Timer'} - time's up!")
        elif kind == "caption":
            self._on_caption(data.get("who", ""), data.get("text", ""))
        elif kind == "tool_start":
            name = data["name"]
            self.orb.action(f"{tool_icon(name)}  {pretty_tool(name)}...")
        elif kind == "tool":
            result = data.get("result", {})
            ok = "error" not in result
            self.orb.pulse(ok)
            if not ok:
                self._emote("sad")
            elif data["name"] == "complete_note":
                self._emote("wink")
            self.orb.action(f"{'✓' if ok else '✗'}  {pretty_tool(data['name'])}", OK if ok else DANGER)
            card = self._rich_card(data["name"], result) if ok else None
            if card is not None:  # a card says more than the plain action chip
                self.chat._add_row(card, "left", text=f"{data['name']} {result.get('place', '')} {result.get('query', '')}")
            else:
                self.chat.add_tool(data["name"], data.get("args", {}), result)
            image = result.get("image") if ok else None
            if image and Path(image).exists():
                caption = Path(image).name if "Screenshot" in Path(image).name else "What I saw"
                self.chat.add_image(image, f"🖼  {caption}  ·  click to open", self.accent)
        elif kind == "crash":
            self._add_system(f"⚠ Something went wrong: {data['text']}  (details in the log: tray → Open logs folder)")
            self.toast("⚠", "Something went wrong - details are in the log.", DANGER)
        elif kind == "log":
            text = data["text"]
            self._add_system(text)
            if text.startswith("🔔"):
                self.toast("🔔", text[1:].strip(), WARN, 8)
            elif text.startswith("✓ Reconnected"):
                self.toast("📶", "Back online.", OK)
            elif text.startswith("[connection lost]") and not self._offline_toast:
                self._set_net("reconnecting")
                self._offline_toast = True
                self.toast("📡", "Connection lost - reconnecting...", WARN)
            if text.startswith("✓ Reconnected"):
                self._offline_toast = False
            if text.startswith("🔔") and self.tray and not self.isActiveWindow():
                self.tray.showMessage(self.app.settings.assistant_name, text[1:].strip())

    def _rich_card(self, name: str, result: dict):
        from jarvis.ui.cards import links_card, weather_card

        try:
            if name == "get_weather":
                return weather_card(result)
            if name == "web_search":
                return links_card(result)
        except Exception:
            log.exception("couldn't draw a card for %s", name)
        return None

    _joke_pending = False
    _offline_toast = False

    def _emote(self, kind: str, seconds: float = 2.8) -> None:
        """Happy feelings morph the core into the cute face for a moment; others show on the AI core."""
        if kind in ("love", "happy", "wink", "star"):
            self.orb.transform(kind, seconds)
        else:
            self.orb.emote(kind)
        self.app.events.emit("emote", feeling=kind)  # sound effects listen for this

    def _react_to_user(self, text: str) -> None:
        low = text.lower()
        if any(w in low for w in THANKS_WORDS):
            self._emote("love")
        elif any(w in low for w in PRAISE_WORDS):
            self._emote("happy")
        if any(w in low for w in JOKE_WORDS):
            self._joke_pending = True

    def _set_net(self, status: str) -> None:
        """Small pill under the clock: Online / Connecting / Reconnecting / Offline."""
        text, color = {
            "online": ("● Online", OK), "connecting": ("● Connecting…", WARN),
            "reconnecting": ("● Reconnecting…", WARN), "offline": ("● Offline", DANGER),
        }.get(status, ("● Offline", DANGER))
        if status == "online":
            text += f"  ·  {self.app.settings.live_model}"
        self.net_status = status
        self.net_pill.setText(text)
        self.net_pill.setStyleSheet(f"color: {color}; border: 1px solid {color}; border-radius: 9px; "
                                    "padding: 1px 10px; font-size: 10px; font-weight: 600; background: transparent;")

    def _on_state(self, state: str) -> None:
        prev, self._last_state = self._last_state, state
        if state == "connecting":
            self._set_net("reconnecting" if getattr(self, "net_status", "") == "online" else "connecting")
        elif state == "offline":
            self._set_net("offline")
        elif state != "thinking":
            self._set_net("online")
        self.orb.set_state(state)
        self.state_label.setText(STATE_LABELS.get(state, state.upper()))
        self._set_state_style(state)
        self.sleep_btn.setText("☀  Wake" if state == "sleeping" else "💤  Sleep")
        name = self.app.settings.assistant_name
        if state == "thinking":
            self.typing.show_text(f"{name} is working on it...", WARN)
        elif state == "connecting":
            self.typing.show_text("Connecting...", theme.MUTED)
        else:
            self.typing.hide_now()
        if prev == "sleeping" and state in AWAKE_STATES:
            if self.app.settings.power_up_wake:
                self.orb.power_up()
            else:
                self.orb.flash()
        if state == "sleeping":
            how = self.assistant.how_to_wake() if self.assistant else "press Wake"
            mode = self.assistant.wake_mode if self.assistant else ""
            # privacy at a glance: is anything still being heard while asleep?
            where = {"wake_word": "🔒 offline wake word - nothing leaves your PC",
                     "name": "☁ listening for your name online"}.get(mode, "🔇 mic off until you press Wake")
            self.hint.setText(f"{how[0].upper() + how[1:]}  ·  {where}")
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
        color = (theme.readable(OK) if who == "assistant" else theme.TEXT)
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

    _welcome_row = None

    def _add_bubble(self, who: str, text: str, mine: bool = False, typewriter: bool = False) -> None:
        if not text or text.startswith("(system"):
            return  # app-generated instructions are not shown as user messages
        if self._welcome_row is not None:  # the conversation has started: the welcome card has done its job
            try:
                self._welcome_row.hide()
                self._welcome_row.deleteLater()
            except RuntimeError:
                pass
            self._welcome_row = None
        self.chat.add_message(who, text, mine, typewriter=typewriter)

    def toast(self, icon: str, text: str, color: str, seconds: float = 4.5) -> None:
        if self.isVisible() and not self.isMinimized():
            self.toasts.show(icon, text, color, seconds)

    def _add_system(self, text: str) -> None:
        self.chat.add_system(text)

    # -- help --------------------------------------------------------------------------
    def toggle_help(self) -> None:
        if getattr(self, "help", None) is None:
            from jarvis.ui.help import HelpPanel

            self.help = HelpPanel(self.right_panel, self._send)
        self.help.toggle()

    def toggle_dashboard(self, tab: str | None = None) -> None:
        if getattr(self, "dashboard", None) is None:
            from jarvis.ui.dashboard import DashboardPanel

            self.dashboard = DashboardPanel(self.right_panel, self.app, self._send)
        self.dashboard.toggle(tab)

    def toggle_planner(self) -> None:
        if getattr(self, "planner", None) is None:
            from jarvis.ui.planner import PlannerPanel

            self.planner = PlannerPanel(self.right_panel, self.app, lambda: self.big_orb.timers, self._send)
        self.planner.toggle()

    def toggle_badges(self) -> None:
        if getattr(self, "badges", None) is None:
            from jarvis.ui.badges import BadgesPanel

            self.badges = BadgesPanel(self.right_panel, self.app.achievements)
        self.badges.toggle()

    def _escape(self) -> None:
        if getattr(self, "dashboard", None) is not None and self.dashboard.isVisible():
            self.dashboard.hide()
        elif getattr(self, "planner", None) is not None and self.planner.isVisible():
            self.planner.hide()
        elif getattr(self, "badges", None) is not None and self.badges.isVisible():
            self.badges.hide()
        elif getattr(self, "help", None) is not None and self.help.isVisible():
            self.help.close_panel()
        else:
            self.close_search()

    # -- search and export -------------------------------------------------------
    def open_search(self) -> None:
        self.search_bar.show()
        self.search.setFocus()
        self.search.selectAll()

    def close_search(self) -> None:
        if not self.search_bar.isHidden():
            self.search.clear()
            self.search_bar.hide()
            self.chat.filter("")
            self.chat.scroll_to_end()

    def _on_search(self, text: str) -> None:
        n = self.chat.filter(text)
        self.search_count.setText(f"{n} match{'es' if n != 1 else ''}" if text.strip() else "")

    def export_chat(self) -> Path | None:
        items = self.history.recent(5000)
        if not items:
            self.toast("⤓", "Nothing to save yet.", theme.MUTED)
            return None
        folder = Path.home() / "Documents" / "Jarvis Chats"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"Jarvis chat {datetime.now():%Y-%m-%d %H-%M}.md"
        name = self.app.settings.assistant_name
        lines = [f"# Conversation with {name}", ""]
        last_day = None
        for item in items:
            when = datetime.fromtimestamp(item.get("t", 0) or 0)
            if when.date() != last_day:
                last_day = when.date()
                lines += ["", f"## {when:%A, %d %B %Y}", ""]
            who = "You" if item["role"] == "user" else name
            lines.append(f"**{who}** ({when:%I:%M %p}): {item['text']}  ")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.toast("⤓", f"Saved to Documents › Jarvis Chats ({len(items)} messages)", OK)
        return path

    def _clear_chat(self) -> None:
        self.chat.clear()
        self.history.clear()

    def _restore_history(self) -> None:
        items = self.history.recent(30)
        if not items:
            from jarvis.ui.help import welcome_card

            self._welcome_row = self.chat._add_row(
                welcome_card(self.app.settings.assistant_name, self.app.settings.user_name, self._send),
                "center", text="welcome", animate=False)
            return
        name = self.app.settings.assistant_name
        self.chat.animate = False
        for item in items:
            mine = item["role"] == "user"
            self.chat.add_message("You" if mine else name, item["text"], mine,
                                  when=datetime.fromtimestamp(item.get("t", 0) or 0))
        self.chat.add_system("──────  earlier conversation  ·  Ctrl+L clears it  ──────")
        self.chat.animate = True

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
        self.chat.scroll_to_end()
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
        if changed & {"theme_color", "theme_name", "theme_mode", "assistant_name", "orb_face"}:
            self._apply_theme()
        if changed & {"desktop_widget", "widget_corner", "theme_color", "theme_name"}:
            self.apply_desktop_widget()
        if "history_days" in changed and self.app.history:
            self.app.history.prune(self.app.settings.history_days)
        if self.assistant is not None and "power_up_wake" in changed:
            self.assistant.sfx.power_up = self.app.settings.power_up_wake
        if self.assistant is not None and changed & {"sound_effects", "sound_volume"}:
            self.assistant.sfx.enabled = self.app.settings.sound_effects
            self.assistant.sfx.volume = self.app.settings.sound_volume
            self.assistant.sfx.play("success")
        if self.assistant is None:
            self.start_assistant()
        elif changed & RESTART_KEYS:
            self.assistant.restart(fresh="voice" in changed)
        elif changed & {"reply_language", "reminder_language"}:
            self.assistant.send_system(self._language_notice())  # no restart: the chat goes on
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
        self.reminders.close_all()
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

    # -- privacy: you always see when Jarvis looks ----------------------------------------
    def flash_capture(self, what: str) -> None:
        """A camera-shutter flash over the window and a notice, every time the screen or camera is captured."""
        from PyQt6.QtWidgets import QGraphicsOpacityEffect

        flash = QWidget(self.centralWidget())
        flash.setStyleSheet("background: white;")
        flash.setGeometry(self.centralWidget().rect())
        flash.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        effect = QGraphicsOpacityEffect(flash)
        flash.setGraphicsEffect(effect)
        flash.show()
        flash.raise_()
        anim = QPropertyAnimation(effect, b"opacity", flash)
        anim.setDuration(420)
        anim.setStartValue(0.55)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(flash.deleteLater)
        anim.start()
        icon, text = ("📷", "Jarvis took a photo with your camera") if what == "camera" else \
            ("👁", "Jarvis looked at your screen")
        self.toast(icon, text, theme.ACCENT, 3)
        if self.tray and not self.isVisible():
            self.tray.showMessage(self.app.settings.assistant_name, text)

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
