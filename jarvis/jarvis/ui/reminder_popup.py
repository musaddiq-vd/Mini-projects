"""Reminder pop-ups at the bottom-right of the screen (above the taskbar).

- "due": a reminder rang. Stays until Done / Snooze / ✕ (or 15 minutes), a swinging bell and a soft pulse.
- "set": a reminder was just saved. A smaller card with Cancel that goes away by itself.

They are their own little windows, so they show even when Jarvis is minimised or in the tray, and they never
steal the keyboard from what you're typing.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from datetime import datetime

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QGuiApplication, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout, QWidget

WIDTH = 360
MAX_WIDTH = 460  # long words (Tamil, German...) may widen a card this far
MARGIN = 16
DUE_SECONDS = 15 * 60
SET_SECONDS = 6
SNOOZE_CHOICES = (5, 10, 30, 60)


def clock(ts: float) -> str:
    d = datetime.fromtimestamp(ts)
    text = d.strftime("%I:%M %p").lstrip("0")
    if d.date() != datetime.now().date():
        text = d.strftime("%a ") + text
    return text


def until(ts: float, now: float | None = None, lang: str = "en") -> str:
    from jarvis.lang import words

    w = words(lang)
    mins = max(0, round((ts - (now or time.time())) / 60))
    if mins < 1:
        return w["under_min"]
    if mins < 60:
        return w["in_min"].format(n=mins)
    h, m = divmod(mins, 60)
    if h < 24:
        return w["in_hm"].format(h=h, m=m) if m else w["in_h"].format(h=h)
    days = h // 24
    text = w["in_days"].format(d=days)
    return text.replace("days", "day") if lang == "en" and days == 1 else text


def on_accent(accent: str) -> str:
    """Text colour that reads well on a button filled with the accent."""
    c = QColor(accent)
    return "#06121a" if 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue() > 140 else "#ffffff"


class Bell(QWidget):
    """A painted bell that swings when it rings, with a pulsing ring behind it."""

    def __init__(self, color: QColor, ringing: bool, parent=None):
        super().__init__(parent)
        self.color = color
        self.ringing = ringing
        self.setFixedSize(52, 52)
        self._t0 = time.monotonic()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(33)

    def angle(self, t: float) -> float:
        if not self.ringing:
            return 0.0
        cycle = t % 6.0  # swing for 1.6 s every 6 s
        if cycle > 1.6:
            return 0.0
        return 22 * math.sin(cycle * 14) * (1 - cycle / 1.6)

    def paintEvent(self, _):  # noqa: N802
        t = time.monotonic() - self._t0
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = self.rect().center()
        if self.ringing:  # soft pulse
            k = (t % 1.6) / 1.6
            ring = QColor(self.color)
            ring.setAlphaF(0.45 * (1 - k))
            p.setPen(QPen(ring, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            r = 15 + 10 * k
            p.drawEllipse(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r))
        bg = QColor(self.color)
        bg.setAlphaF(0.18)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawEllipse(QRectF(c.x() - 17, c.y() - 17, 34, 34))
        p.translate(c.x(), c.y() - 9)
        p.rotate(self.angle(t))
        p.setPen(QPen(self.color, 1.6))
        body = QPainterPath()  # bell body
        body.moveTo(-9, 12)
        body.cubicTo(-9, 0, -8, -1, 0, -1)
        body.cubicTo(8, -1, 9, 0, 9, 12)
        body.lineTo(11, 15)
        body.lineTo(-11, 15)
        body.closeSubpath()
        fill = QColor(self.color)
        fill.setAlphaF(0.85)
        p.setBrush(fill)
        p.drawPath(body)
        p.drawEllipse(QRectF(-1.8, -4.5, 3.6, 3.6))  # top knob
        p.drawEllipse(QRectF(-3, 15, 6, 5))  # clapper
        p.end()


class ReminderPopup(QWidget):
    def __init__(self, kind: str, rid: int, message: str, due: float, accent: str, *,
                 late: bool = False, repeat: str = "", lang: str = "en", on_done: Callable[[int], None] | None = None,
                 on_snooze: Callable[[int, int], None] | None = None, on_cancel: Callable[[int], None] | None = None,
                 on_open: Callable[[], None] | None = None, on_closed: Callable[[ReminderPopup], None] | None = None):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.WindowDoesNotAcceptFocus)
        from jarvis.lang import RTL, words
        from jarvis.ui import theme

        w = words(lang)
        self.lang = lang
        latin = lang in ("en", "hinglish", "es", "fr", "de")  # letter-spacing breaks joined scripts
        self.kind, self.rid, self.message, self.due = kind, rid, message, due
        self.accent = QColor(accent)
        self.on_closed = on_closed
        self.on_open = on_open
        self._closing = False
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setObjectName("reminderPopup")
        self.setFixedWidth(WIDTH)
        text, muted = theme.TEXT, theme.MUTED
        self.setStyleSheet(
            f"QLabel {{ background: transparent; color: {text}; }}"
            f"QLabel#rpHead {{ color: {accent}; font-size: 11px; font-weight: 700;"
            f" letter-spacing: {'1.5px' if latin else '0px'}; }}"
            f"QLabel#rpMuted {{ color: {muted}; font-size: 12px; }}"
            f"QPushButton {{ background: transparent; color: {text}; border: 1px solid {muted}; border-radius: 9px;"
            "  padding: 6px 12px; font-size: 12px; }"
            f"QPushButton:hover {{ border-color: {accent}; color: {accent}; }}"
            f"QPushButton#rpPrimary {{ background: {accent}; color: {on_accent(accent)}; border: none;"
            "  font-weight: 700; }"
            "QPushButton::menu-indicator { image: none; width: 0px; }"
            f"QPushButton#rpPrimary:hover {{ background: {QColor(accent).lighter(115).name()}; }}"
            f"QPushButton#rpClose {{ border: none; color: {muted}; font-size: 14px; padding: 2px 6px; }}"
            f"QPushButton#rpClose:hover {{ color: {text}; }}")

        outer = QHBoxLayout(self)
        outer.setContentsMargins(16, 14, 12, 14)
        outer.setSpacing(12)
        self.bell = Bell(self.accent, ringing=kind == "due")
        outer.addWidget(self.bell, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(4)
        head_row = QHBoxLayout()
        head_row.setSpacing(4)
        if kind == "due":
            head = w["reminder"] + (f" · {w['missed']}" if late else "") + f" · {clock(due)}"
        else:
            head = w["reminder_set"] + (f" · {w['daily']}" if repeat == "daily"
                                        else f" · {w['weekdays']}" if repeat == "weekdays" else "")
        self.head = QLabel(head, objectName="rpHead")
        self.head.setWordWrap(True)
        head_row.addWidget(self.head, 1)
        close = QPushButton("✕", objectName="rpClose")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.setToolTip(w["hide"])
        close.clicked.connect(self.close_animated)
        head_row.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        col.addLayout(head_row)
        self.text = QLabel(message)
        self.text.setTextFormat(Qt.TextFormat.PlainText)
        self.text.setWordWrap(True)
        f = QFont(self.font())
        f.setPointSizeF(12.5 if kind == "due" else 11)
        f.setBold(True)
        self.text.setFont(f)
        self.text.setMaximumHeight(90)
        if lang in RTL:
            self.text.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.text.setCursor(Qt.CursorShape.PointingHandCursor)
        self.text.mousePressEvent = lambda _e: self.on_open and self.on_open()
        col.addWidget(self.text)
        due_text = w["due_while_off"] if late else w["due_now"]
        sub = due_text if kind == "due" else f"{clock(due)} · {until(due, lang=lang)}"
        self.sub = QLabel(sub, objectName="rpMuted")
        col.addWidget(self.sub)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        if kind == "due":
            done = QPushButton("✓  " + w["done"], objectName="rpPrimary")
            done.clicked.connect(lambda: (on_done and on_done(rid), self.close_animated()))
            buttons.addWidget(done)
            snooze = QPushButton("⏱  " + w["snooze"])
            snooze.clicked.connect(lambda: (on_snooze and on_snooze(rid, 10), self.close_animated()))
            buttons.addWidget(snooze)
            more = QPushButton("▾")
            more.setToolTip(w["snooze_for"])
            menu = QMenu(more)
            for m in SNOOZE_CHOICES:
                label = w["minutes"].format(n=m) if m < 60 else w["hour"]
                menu.addAction(label, lambda m=m: (on_snooze and on_snooze(rid, m), self.close_animated()))
            more.setMenu(menu)
            buttons.addWidget(more)
            self.buttons = (done, snooze, more)
        else:
            cancel = QPushButton(w["cancel"])
            cancel.clicked.connect(lambda: (on_cancel and on_cancel(rid), self.close_animated()))
            buttons.addWidget(cancel)
            self.buttons = (cancel,)
        for b in self.buttons:
            b.setCursor(Qt.CursorShape.PointingHandCursor)
        buttons.addStretch(1)
        col.addLayout(buttons)
        outer.addLayout(col, 1)
        need = 16 + 52 + 12 + 12 + sum(b.sizeHint().width() for b in self.buttons) + 8 * len(self.buttons) + 8
        self.setFixedWidth(max(WIDTH, min(MAX_WIDTH, need)))
        self.adjustSize()
        self._life = QTimer(self)
        self._life.setSingleShot(True)
        self._life.timeout.connect(self.close_animated)
        self._life.start(int((DUE_SECONDS if kind == "due" else SET_SECONDS) * 1000))

    # -- look --------------------------------------------------------------------------------
    def paintEvent(self, _):  # noqa: N802
        from jarvis.ui import theme

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        bg = QColor(theme.P.panel2)
        bg.setAlpha(246)
        p.setBrush(bg)
        grad = QLinearGradient(r.topLeft(), r.bottomRight())
        grad.setColorAt(0, self.accent)
        second = QColor(theme.SECOND)
        second.setAlphaF(0.7)
        grad.setColorAt(1, second)
        p.setPen(QPen(grad, 1.6))
        p.drawRoundedRect(r, 16, 16)
        if self.kind == "due":  # an accent strip on the left
            strip = QColor(self.accent)
            strip.setAlphaF(0.9)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(strip)
            p.drawRoundedRect(QRectF(r.left() + 1, r.top() + 16, 3.5, r.height() - 32), 2, 2)
        p.end()

    # -- motion ------------------------------------------------------------------------------
    def show_at(self, target: QPoint) -> None:
        start = QPoint(target.x() + 70, target.y())
        self.setWindowOpacity(0.0)
        self.move(start)
        self.show()
        self.raise_()
        self._slide = QPropertyAnimation(self, b"pos", self)
        self._slide.setDuration(420)
        self._slide.setStartValue(start)
        self._slide.setEndValue(target)
        self._slide.setEasingCurve(QEasingCurve.Type.OutBack)
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(320)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._slide.start()
        self._fade.start()

    def move_to(self, target: QPoint) -> None:
        if not self.isVisible():
            self.move(target)
            return
        self._slide = QPropertyAnimation(self, b"pos", self)
        self._slide.setDuration(260)
        self._slide.setEndValue(target)
        self._slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._slide.start()

    def close_animated(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._life.stop()
        if self.on_closed:
            self.on_closed(self)
        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(220)
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self.close)
        self._fade.start()


class ReminderPopups:
    """Keeps the pop-ups stacked upwards from the bottom-right corner of the screen."""

    def __init__(self, accent: Callable[[], str], *, on_done, on_snooze, on_cancel, on_open, enabled=lambda: True,
                 language: Callable[[], str] = lambda: "en"):
        self.accent = accent
        self.language = language
        self.on_done, self.on_snooze, self.on_cancel, self.on_open = on_done, on_snooze, on_cancel, on_open
        self.enabled = enabled
        self.popups: list[ReminderPopup] = []

    def show(self, kind: str, rid: int, message: str, due: float, late: bool = False, repeat: str = "") \
            -> ReminderPopup | None:
        if not self.enabled():
            return None
        if kind == "due":  # its "set" card (if still up) is replaced by the ringing one
            self.close(rid, kinds=("set", "due"))
        try:
            lang = self.language()
        except Exception:
            lang = "en"
        popup = ReminderPopup(kind, rid, message, due, self.accent(), late=late, repeat=repeat, lang=lang,
                              on_done=self.on_done, on_snooze=self.on_snooze, on_cancel=self.on_cancel,
                              on_open=self.on_open, on_closed=self._closed)
        self.popups.append(popup)
        while len([p for p in self.popups if not p._closing]) > 4:  # never a wall of cards
            next(p for p in self.popups if not p._closing).close_animated()
        popup.show_at(self._slot(popup))
        return popup

    def close(self, rid: int, kinds=("set", "due")) -> None:
        for p in list(self.popups):
            if p.rid == rid and p.kind in kinds:
                p.close_animated()

    def close_all(self) -> None:
        for p in list(self.popups):
            try:
                p.close()
            except RuntimeError:
                pass
        self.popups = []

    def _area(self):
        screen = QGuiApplication.primaryScreen()
        return screen.availableGeometry() if screen else None

    def _slot(self, popup: ReminderPopup) -> QPoint:
        area = self._area()
        if area is None:
            return QPoint(0, 0)
        y = area.bottom() - MARGIN
        for p in self.popups:
            if p is popup:
                break
            if not p._closing:
                y -= p.height() + 10
        return QPoint(area.right() - popup.width() - MARGIN, y - popup.height())

    def _closed(self, popup: ReminderPopup) -> None:
        self.popups = [p for p in self.popups if p is not popup]
        area = self._area()
        if area is None:
            return
        y = area.bottom() - MARGIN
        for p in self.popups:
            if p._closing:
                continue
            try:
                p.move_to(QPoint(area.right() - p.width() - MARGIN, y - p.height()))
            except RuntimeError:
                continue
            y -= p.height() + 10
