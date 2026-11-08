"""Mini mode: a small floating Jarvis that stays on top of other windows.

- the orb with a live caption (what you / Jarvis are saying)
- what's next: the coming reminder or meeting
- a fold-out chat: the last few messages and a box to type in (💬 button, remembered)

Drag it anywhere, double-click the orb to go back to the full window, right-click for a menu.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from datetime import datetime

from PyQt6.QtCore import QPoint, QPointF, QRectF, QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QFont, QPainter, QRadialGradient
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMenu, QPushButton, QVBoxLayout, QWidget,
)

from jarvis.ui.orb import Orb

ORB_SIZE = (230, 290)
OPEN_SIZE = (300, 560)
MAX_MESSAGES = 4


class OrbGroup:
    """Sends every orb call to all orbs (the big one and the mini one)."""

    def __init__(self, *orbs: Orb):
        self.orbs = list(orbs)

    def __getattr__(self, name):
        def call(*args, **kwargs):
            for orb in self.orbs:
                getattr(orb, name)(*args, **kwargs)
        return call


def next_up(app, now: float | None = None) -> tuple[str, str] | None:
    """('⏰', '6:30 PM · Call mom') for the coming reminder or meeting (whichever is sooner), or None."""
    now = now or time.time()
    best: tuple[float, str, str] | None = None
    for r in (app.state.get("reminders", []) or []):
        if r.get("due", 0) > now and (best is None or r["due"] < best[0]):
            best = (r["due"], "⏰", r.get("message", ""))
    try:
        from jarvis.tools.calendar_sync import next_event

        e = next_event(app.settings, now)
    except Exception:
        e = None
    if e and (best is None or e["start"] < best[0]):
        best = (e["start"], "📅", e["title"])
    if best is None:
        return None
    when, icon, text = best
    mins = (when - now) / 60
    if when <= now:
        label = "now"
    elif mins < 60:
        label = f"in {max(1, int(mins))} min"
    else:
        d = datetime.fromtimestamp(when)
        label = d.strftime("%I:%M %p").lstrip("0") if d.date() == datetime.now().date() else d.strftime("%a %I:%M %p")
    return icon, f"{label} · {text[:40]}"


class MiniWindow(QWidget):
    SIZE = ORB_SIZE

    def __init__(self, accent: str, face: bool, on_restore, actions: list[tuple[str, object]],
                 send: Callable[[str], None] | None = None, upcoming: Callable[[], tuple[str, str] | None] = None):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setObjectName("mini")
        self.on_restore = on_restore
        self.actions = actions
        self.send = send
        self.upcoming = upcoming or (lambda: None)
        self._drag: QPoint | None = None
        self.accent = accent
        self.messages: deque[tuple[str, str]] = deque(maxlen=12)
        self.chat_open = QSettings("MusaddiqKhan", "Jarvis").value("mini_chat", False, type=bool) and send is not None

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 8)
        v.setSpacing(6)
        top = QWidget()
        top.setFixedHeight(ORB_SIZE[1] - 14)
        tv = QVBoxLayout(top)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(0)
        self.orb = Orb(accent, face=face)
        self.orb.setMinimumSize(200, 220)
        self.orb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        tv.addWidget(self.orb, 1)
        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setFixedHeight(40)
        self.caption.setFont(QFont("Segoe UI", 9))
        self.caption.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        tv.addWidget(self.caption)
        v.addWidget(top)
        self.chat_btn = QPushButton("💬", top)
        self.chat_btn.setObjectName("miniChatBtn")
        self.chat_btn.setFixedSize(30, 30)
        self.chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chat_btn.setToolTip("Show / hide the chat")
        self.chat_btn.clicked.connect(self.toggle_chat)
        self.chat_btn.setVisible(send is not None)

        # what's next
        self.next_chip = QLabel(objectName="miniNext")
        self.next_chip.setTextFormat(Qt.TextFormat.PlainText)
        self.next_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.next_chip.setWordWrap(True)
        v.addWidget(self.next_chip)
        # the chat
        self.panel = QFrame(objectName="miniPanel")
        pv = QVBoxLayout(self.panel)
        pv.setContentsMargins(10, 10, 10, 10)
        pv.setSpacing(6)
        self.bubbles = QVBoxLayout()
        self.bubbles.setSpacing(5)
        pv.addLayout(self.bubbles, 1)
        self.empty = QLabel("Say something or type below.", objectName="miniMuted")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pv.addWidget(self.empty)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type to Jarvis…")
        self.input.returnPressed.connect(self._submit)
        row.addWidget(self.input, 1)
        go = QPushButton("➤", objectName="miniSend")
        go.setFixedSize(32, 30)
        go.setCursor(Qt.CursorShape.PointingHandCursor)
        go.clicked.connect(self._submit)
        row.addWidget(go)
        pv.addLayout(row)
        v.addWidget(self.panel, 1)
        self._style()
        self.setToolTip("Drag to move · double-click the orb to open Jarvis · right-click for options")
        self._apply_open()
        self._tick = QTimer(self)
        self._tick.timeout.connect(self.refresh_next)
        self._tick.start(5000)
        self.refresh_next()

    # -- look --------------------------------------------------------------------------------
    def _style(self) -> None:
        from jarvis.ui import theme

        a = self.accent
        self.setStyleSheet(
            "QWidget#mini { background: transparent; }"
            "QLabel { background: transparent; }"
            f"QLabel#miniNext {{ color: {theme.TEXT}; background: rgba(10,15,20,200); border: 1px solid {a};"
            "  border-radius: 11px; padding: 4px 10px; font-size: 11px; }"
            f"QLabel#miniMuted {{ color: {theme.MUTED}; font-size: 11px; }}"
            "QFrame#miniPanel { background: rgba(10,15,20,225); border: 1px solid rgba(255,255,255,30);"
            "  border-radius: 14px; }"
            f"QLabel#bubbleMe {{ color: #06121a; background: {a}; border-radius: 10px; padding: 5px 9px;"
            "  font-size: 11px; }"
            "QLabel#bubbleJarvis { color: #d8e6ef; background: rgba(255,255,255,18); border-radius: 10px;"
            "  padding: 5px 9px; font-size: 11px; }"
            "QLineEdit { background: rgba(255,255,255,14); color: #d8e6ef; border: 1px solid rgba(255,255,255,40);"
            "  border-radius: 9px; padding: 5px 8px; font-size: 11px; }"
            f"QLineEdit:focus {{ border-color: {a}; }}"
            f"QPushButton#miniSend {{ background: {a}; color: #06121a; border: none; border-radius: 9px;"
            "  font-weight: 700; font-size: 15px; padding: 0px; }"
            "QPushButton#miniChatBtn { background: rgba(10,15,20,210); color: #d8e6ef; padding: 0px;"
            "  border: 1px solid rgba(255,255,255,60); border-radius: 15px; font-size: 13px; }"
            f"QPushButton#miniChatBtn:hover {{ border-color: {a}; }}")
        self.caption.setStyleSheet("color:#d8e6ef; background: transparent;")

    def _apply_open(self) -> None:
        self.panel.setVisible(self.chat_open)
        size = OPEN_SIZE if self.chat_open else ORB_SIZE
        self.setFixedSize(*size)
        self.chat_btn.move(self.width() - 6 - 6 - 30 - 6, 4)
        self.chat_btn.setText("✕" if self.chat_open else "💬")
        self.refresh_next()

    def toggle_chat(self) -> None:
        if self.send is None:
            return
        old_bottom = self.geometry().bottom()
        self.chat_open = not self.chat_open
        QSettings("MusaddiqKhan", "Jarvis").setValue("mini_chat", self.chat_open)
        self._apply_open()
        if self.isVisible():  # grow upwards when near the bottom of the screen, so it stays on screen
            area = (self.screen() or QApplication.primaryScreen()).availableGeometry()
            y = self.y()
            if self.geometry().bottom() > area.bottom():
                y = max(area.top(), old_bottom - self.height() + 1)
            self.move(min(max(self.x(), area.left()), area.right() - self.width()), y)
        if self.chat_open:
            self.input.setFocus()

    # -- content -----------------------------------------------------------------------------
    def set_caption(self, text: str) -> None:
        self.caption.setText(text[-90:] if len(text) > 90 else text)

    def set_accent(self, accent: str) -> None:
        self.accent = accent
        self.orb.set_accent(accent)
        self._style()
        self.update()

    def add_message(self, who: str, text: str) -> None:
        text = " ".join((text or "").split())
        if not text or text.startswith("(system"):
            return
        self.messages.append((who, text))
        self._render()

    def _render(self) -> None:
        while self.bubbles.count():
            item = self.bubbles.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)  # gone right away, not on the next event loop turn
                old.deleteLater()
        shown = list(self.messages)[-MAX_MESSAGES:]
        self.empty.setVisible(not shown)
        self.bubbles.addStretch(1)
        for who, text in shown:
            mine = who == "user"
            b = QLabel(text if len(text) <= 220 else text[:217] + "…", objectName="bubbleMe" if mine else "bubbleJarvis")
            b.setTextFormat(Qt.TextFormat.PlainText)
            b.setWordWrap(True)
            b.setMaximumWidth(OPEN_SIZE[0] - 60)
            b.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            if mine:
                row.addStretch(1)
            row.addWidget(b)
            if not mine:
                row.addStretch(1)
            holder = QWidget()
            holder.setLayout(row)
            self.bubbles.addWidget(holder)

    def refresh_next(self) -> None:
        try:
            nxt = self.upcoming()
        except Exception:
            nxt = None
        self.next_chip.setVisible(bool(nxt) and self.chat_open)
        if nxt:
            self.next_chip.setText(f"{nxt[0]}  {nxt[1]}")

    def _submit(self) -> None:
        text = self.input.text().strip()
        self.input.clear()
        if not text or self.send is None:
            return
        self.send(text)

    def show_at_saved_place(self) -> None:
        pos = QSettings("MusaddiqKhan", "Jarvis").value("mini_pos")
        screen = QApplication.primaryScreen().availableGeometry()
        if isinstance(pos, QPoint) and screen.adjusted(-50, -50, 50, 50).contains(pos):
            self.move(pos)
        else:
            self.move(screen.right() - self.width() - 24, screen.bottom() - self.height() - 24)
        if self.geometry().bottom() > screen.bottom():
            self.move(self.x(), max(screen.top(), screen.bottom() - self.height()))
        self.show()
        self.raise_()

    # -- painting: a soft dark disc behind the orb so it reads on any wallpaper ----------
    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.width() / 2, 118)
        grad = QRadialGradient(center, 112)
        grad.setColorAt(0.0, QColor(10, 15, 20, 235))
        grad.setColorAt(0.8, QColor(10, 15, 20, 200))
        grad.setColorAt(1.0, QColor(10, 15, 20, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(grad)
        p.drawEllipse(center, 112, 112)
        if self.caption.text():
            p.setBrush(QColor(10, 15, 20, 210))
            p.drawRoundedRect(QRectF(6, ORB_SIZE[1] - 56, self.width() - 12, 44), 12, 12)
        p.end()

    # -- mouse ------------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag is not None:
            QSettings("MusaddiqKhan", "Jarvis").setValue("mini_pos", self.pos())
        self._drag = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.position().y() < ORB_SIZE[1]:  # only the orb, not the chat
            self.on_restore()

    def closeEvent(self, event) -> None:  # noqa: N802
        if event.spontaneous():  # Alt+F4 by the user: go back to the full window instead
            event.ignore()
            self.on_restore()
        else:
            event.accept()  # the app is quitting: never block it

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        menu = QMenu(self)
        items = [("Open full window", self.on_restore)]
        if self.send is not None:
            items.append(("Hide chat" if self.chat_open else "Show chat", self.toggle_chat))
        for text, fn in [*items, *self.actions]:
            act = QAction(text, menu)
            act.triggered.connect(fn)
            menu.addAction(act)
        menu.exec(event.globalPos())
