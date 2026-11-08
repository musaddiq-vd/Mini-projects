"""The desktop widget: a small Jarvis on your wallpaper.

A glassy card behind all other windows with a live orb, the time and date, and what's next (meeting,
reminder), open to-dos and today's habits. Clicks go straight through it to the desktop, so it never gets
in the way. Pick the corner in Settings > Look, or ask Jarvis to show or hide it.
"""

from __future__ import annotations

import time
from datetime import date, datetime

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from jarvis.ui.orb import Orb

CORNERS = ("top_right", "top_left", "bottom_right", "bottom_left")
SIZE = (420, 190)
MARGIN = 28


def lines_for(app, now: float | None = None) -> list[str]:
    """The 'what's next' lines: meeting / reminder, to-dos, habits (only the ones that have something)."""
    now = now or time.time()
    out = []
    try:
        from jarvis.ui.mini import next_up

        nxt = next_up(app, now)
        if nxt:
            out.append(f"{nxt[0]}  {nxt[1]}")
    except Exception:
        pass
    todos = [n for n in app.state.get("notes", []) or [] if n.get("todo") and not n.get("done")]
    if todos:
        out.append(f"📝  {len(todos)} to-do{'s' if len(todos) != 1 else ''}" + f" · {todos[0]['text'][:28]}")
    try:
        from jarvis.tools.habits import counts_on, done_on, habits

        today = date.today()
        hs = [h for h in habits().load().get("habits", []) if counts_on(h, today)]
        if hs:
            done = sum(1 for h in hs if done_on(h, today))
            out.append(f"✅  {done}/{len(hs)} habits today" + ("  🎉" if done == len(hs) else ""))
    except Exception:
        pass
    if not out:
        out.append("✨  Nothing planned - enjoy your day")
    return out[:3]


class DesktopWidget(QWidget):
    def __init__(self, app, accent: str, face: bool = True, corner: str = "top_right"):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnBottomHint
                         | Qt.WindowType.Tool | Qt.WindowType.WindowTransparentForInput
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.app = app
        self.accent = accent
        self.corner = corner if corner in CORNERS else "top_right"
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedSize(*SIZE)
        h = QHBoxLayout(self)
        h.setContentsMargins(10, 10, 18, 10)
        h.setSpacing(8)
        self.orb = Orb(accent, face=face)
        self.orb.setMinimumSize(0, 0)
        self.orb.setFixedSize(150, 170)
        self.orb.boot = 1.0
        self.orb._timer.setInterval(33)  # a calmer frame rate on the wallpaper
        h.addWidget(self.orb)
        col = QVBoxLayout()
        col.setSpacing(2)
        col.addStretch(1)
        self.clock = QLabel()
        self.clock.setFont(QFont("Segoe UI Light", 30))
        col.addWidget(self.clock)
        self.date = QLabel()
        self.date.setFont(QFont("Segoe UI", 10))
        col.addWidget(self.date)
        col.addSpacing(6)
        self.info: list[QLabel] = []
        for _ in range(3):
            lb = QLabel()
            lb.setTextFormat(Qt.TextFormat.PlainText)  # meeting titles come from other people's invites
            lb.setFont(QFont("Segoe UI", 9))
            col.addWidget(lb)
            self.info.append(lb)
        col.addStretch(1)
        h.addLayout(col, 1)
        self._style()
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self.tick)
        self._clock_timer.start(1000)
        self._info_at = 0.0
        self.tick()

    def _style(self) -> None:
        # sizes live here (not only in setFont) so the app's theme stylesheet can't shrink them
        self.clock.setStyleSheet("color: #ffffff; background: transparent; font-family: 'Segoe UI Light';"
                                 " font-size: 30pt;")
        self.date.setStyleSheet(f"color: {self.accent}; background: transparent; letter-spacing: 1px;"
                                " font-size: 10pt;")
        for lb in self.info:
            lb.setStyleSheet("color: #e3edf3; background: transparent; font-size: 9pt;")

    def set_accent(self, accent: str) -> None:
        self.accent = accent
        self.orb.set_accent(accent)
        self._style()
        self.update()

    def tick(self) -> None:
        now = datetime.now()
        self.clock.setText(now.strftime("%I:%M %p").lstrip("0"))
        self.date.setText(now.strftime("%A, %d %B").upper())
        if time.time() - self._info_at > 10:
            self.refresh_info()

    def refresh_info(self) -> None:
        self._info_at = time.time()
        try:
            lines = lines_for(self.app)
        except Exception:
            lines = []
        for lb, text in zip(self.info, lines + [""] * 3):
            lb.setText(text[:48] + ("…" if len(text) > 48 else ""))
            lb.setVisible(bool(text))

    def place(self, corner: str | None = None) -> None:
        if corner:
            self.corner = corner if corner in CORNERS else self.corner
        screen = QApplication.primaryScreen()
        area = screen.availableGeometry() if screen else None
        if area is None:
            return
        x = area.left() + MARGIN if "left" in self.corner else area.right() - self.width() - MARGIN
        y = area.top() + MARGIN if "top" in self.corner else area.bottom() - self.height() - MARGIN
        self.move(x, y)

    def show_on_desktop(self) -> None:
        self.place()
        self.orb._timer.start()
        self.show()
        self.lower()

    def hide_from_desktop(self) -> None:
        self.orb._timer.stop()  # no drawing while hidden
        self.hide()

    def paintEvent(self, _):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setBrush(QColor(8, 13, 18, 150))  # glass that keeps text readable on any wallpaper
        edge = QColor(self.accent)
        edge.setAlpha(110)
        p.setPen(QPen(edge, 1.2))
        p.drawRoundedRect(r, 22, 22)
        p.end()
