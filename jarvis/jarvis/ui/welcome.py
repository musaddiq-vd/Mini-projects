"""The welcome: a fresh greeting every start (and every few minutes), never the same twice in a row.

- an empty chat gets the full card: gradient greeting with a shimmer, a status line, a "did you know" tip and
  four suggestions that fit the time of day
- a chat with history gets a slim "welcome back" banner at the end instead
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from datetime import datetime

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame, QGraphicsOpacityEffect, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout,
)

from jarvis.ui import theme

REFRESH_MS = 8 * 60 * 1000  # a new greeting and tip every few minutes while it's on screen

GREETINGS = {
    "morning": ["Good morning{n} ☀️", "Rise and shine{n}", "Fresh day, fresh start{n}", "Morning{n} — let's make it count",
                "Coffee first, then the world{n} ☕"],
    "afternoon": ["Good afternoon{n}", "Hey{n} — how's the day going?", "Afternoon check-in{n} 👋",
                  "Halfway through the day{n} 💪", "Back at it{n}?"],
    "evening": ["Good evening{n} 🌆", "Evening{n} — let's wrap up the day", "Welcome back{n} ✨",
                "The evening shift begins{n}", "Winding down{n}?"],
    "night": ["Burning the midnight oil{n}? 🌙", "Late-night mode: on{n}", "Still up{n}? I've got you",
              "Quiet hours{n} — I'm right here", "Night owl spotted{n} 🦉"],
}
TAGLINES = [
    "I'm {a}. Talk in English, Hindi, Urdu or Hinglish — or just type.",
    "Systems online. Say my name, or type below.",
    "Your AI co-pilot is ready. What's first?",
    "All systems green. Ask anything, or pick one below.",
    "I'm {a} — voice, screen, files, reminders, all in one place.",
]
TIPS = [
    "Say “is error ko theek karo” and I'll read the error in your window and show the fix.",
    "“Kya pehle karun?” ranks your to-do list: overdue, due today, then what matters.",
    "Say “right corner me jao” or “dheere dheere neeche chalo … ruko” to move me by voice.",
    "No internet? Timers, reminders, notes and opening apps still work.",
    "Drop a PDF on this window, then ask anything about it.",
    "“Is hafte ka report” shows your money, habits and focus for the week.",
    "Paste a private iCal link in Settings and I'll remind you before meetings.",
    "Ctrl+Alt+J brings me up from any app.",
    "Say “desktop widget lagao” for a live card on your wallpaper.",
    "“100 ki chai pi” saves an expense — “is mahine kitna kharcha hua?” totals it.",
    "“Backup bana do” saves everything I keep in one file.",
    "Say “mujhse Hindi me baat karo” to choose the language I speak.",
    "The mini mode (Ctrl+Shift+M) has its own little chat — tap 💬.",
    "Press F1 any time to see everything I can do.",
]
SUGGESTIONS = {
    "morning": ["Plan my day", "What's on my calendar today?", "What's the weather today?", "Kya pehle karun?",
                "Top news in two lines"],
    "afternoon": ["Kya pehle karun?", "Focus mode 25 minutes", "What's on my screen?", "5 minute ka timer lagao",
                  "Is error ko theek karo"],
    "evening": ["Aaj ka recap do", "Is hafte ka report", "What's on my calendar tomorrow?", "Tell me a joke",
                "Meri list batao"],
    "night": ["Remind me at 7 am to wake up", "Aaj ka recap do", "Tell me a joke", "Play some calm music",
              "Kal ka plan banao"],
}


def part_of_day(now: datetime | None = None) -> str:
    h = (now or datetime.now()).hour
    if h < 5 or h >= 22:
        return "night"
    return "morning" if h < 12 else "afternoon" if h < 17 else "evening"


def pick(state, user: str, assistant: str, now: datetime | None = None, rnd: random.Random | None = None) -> dict:
    """A greeting, tagline, tip and suggestions - different from last time (remembered in state)."""
    rnd = rnd or random.Random()
    part = part_of_day(now)
    last = (state.get("welcome_last") if state is not None else None) or {}
    first = user.split()[0].title() if user and user.strip() else ""
    greetings = GREETINGS[part]
    choices = [i for i in range(len(greetings)) if i != last.get("greeting")] or list(range(len(greetings)))
    g = rnd.choice(choices)
    tagline = rnd.choice([i for i in range(len(TAGLINES)) if i != last.get("tagline")] or [0])
    tip = (int(last.get("tip", -1)) + 1) % len(TIPS)  # go through every tip in turn
    sugg = rnd.sample(SUGGESTIONS[part], 4)
    if state is not None:
        state.set("welcome_last", {"greeting": g, "tagline": tagline, "tip": tip})
    return {"part": part, "title": greetings[g].replace("{n}", f", {first}" if first else ""),
            "tagline": TAGLINES[tagline].replace("{a}", assistant), "tip": TIPS[tip], "suggestions": sugg}


def status_line(app, now: float | None = None) -> str:
    """'Sunday 9 Nov · 2 reminders today · 3 to-dos' (only what's there)."""
    now = now or time.time()
    today = datetime.fromtimestamp(now).date()
    parts = [datetime.fromtimestamp(now).strftime("%A %d %b").replace(" 0", " ")]
    st = getattr(app, "state", None)
    if st is not None:
        rem = [r for r in st.get("reminders", []) or []
               if r.get("due", 0) > now and datetime.fromtimestamp(r["due"]).date() == today]
        if rem:
            parts.append(f"{len(rem)} reminder{'s' if len(rem) != 1 else ''} today")
        todos = [n for n in st.get("notes", []) or [] if n.get("todo") and not n.get("done")]
        if todos:
            parts.append(f"{len(todos)} to-do{'s' if len(todos) != 1 else ''}")
    return "  ·  ".join(parts)


class GradientTitle(QLabel):
    """Big greeting text filled with the accent gradient, with a soft shimmer sweeping across now and then."""

    def __init__(self, text: str, size: int = 21, parent=None):
        super().__init__(text, parent)
        self._size = size
        f = QFont(self.font())
        f.setPointSize(size)
        f.setWeight(QFont.Weight.Bold)
        self.setFont(f)
        # also in its own stylesheet, so the app theme's label font can't shrink it
        self.setStyleSheet(f"font-size: {size}pt; font-weight: 700; background: transparent;")
        self.setWordWrap(True)
        self.setMinimumHeight(int(size * 1.9))
        self._shine = -1.0  # -1 = resting; 0..1 = the shimmer's position
        self._t0 = time.monotonic()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def _tick(self) -> None:
        cycle = (time.monotonic() - self._t0) % 6.0  # a 1.2 s shimmer every 6 s
        self._shine = cycle / 1.2 if cycle < 1.2 else -1.0
        self.update()

    def paintEvent(self, _):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        r = QRectF(self.rect())
        grad = QLinearGradient(r.topLeft(), r.topRight())
        a, b = QColor(theme.readable(theme.ACCENT)), QColor(theme.SECOND)
        if theme.P.light:
            b = b.darker(120)
        grad.setColorAt(0.0, a)
        grad.setColorAt(1.0, b)
        if 0 <= self._shine <= 1:
            x = max(0.02, min(0.98, self._shine))
            grad.setColorAt(max(0.0, x - 0.08), a if x < 0.5 else b)
            grad.setColorAt(x, QColor("#ffffff") if not theme.P.light else a.lighter(140))
            grad.setColorAt(min(1.0, x + 0.08), b if x > 0.5 else a)
        p.setPen(QPen(grad, 1))
        p.setFont(self.font())
        p.drawText(r, int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
                   self.text())
        p.end()


def _chip(text: str, send: Callable[[str], None]) -> QPushButton:
    b = QPushButton(text, objectName="chip")
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setToolTip(text)
    b.setMinimumWidth(0)
    b.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    b.setStyleSheet("text-align: left;")
    b.clicked.connect(lambda _, t=text: send(t))
    return b


class WelcomeCard(QFrame):
    """The full welcome for an empty chat."""

    def __init__(self, app, send: Callable[[str], None], compact: bool = False):
        super().__init__(objectName="card")
        self.app, self.send, self.compact = app, send, compact
        s = app.settings
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 16, 20, 16)
        v.setSpacing(6 if compact else 8)
        self.title = GradientTitle("", 15 if compact else 21)
        v.addWidget(self.title)
        self.status = QLabel(objectName="muted")
        self.status.setWordWrap(True)
        v.addWidget(self.status)
        self.tagline = QLabel()
        self.tagline.setWordWrap(True)
        if not compact:
            v.addWidget(self.tagline)
        self.grid = QGridLayout()
        self.grid.setSpacing(6)
        if not compact:
            v.addLayout(self.grid)
        tip_row = QHBoxLayout()
        tip_row.setSpacing(8)
        self.tip = QLabel(objectName="muted")
        self.tip.setWordWrap(True)
        tip_row.addWidget(self.tip, 1)
        more = QPushButton("↻", objectName="small")
        more.setToolTip("Another tip")
        more.setCursor(Qt.CursorShape.PointingHandCursor)
        more.clicked.connect(self.next_tip)
        tip_row.addWidget(more, 0, Qt.AlignmentFlag.AlignTop)
        v.addLayout(tip_row)
        self._fx = QGraphicsOpacityEffect(self)
        self._fx.setOpacity(1.0)
        self.setGraphicsEffect(self._fx)
        self._user, self._assistant = s.user_name, s.assistant_name
        self.fill(pick(app.state, s.user_name, s.assistant_name))
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(REFRESH_MS)

    def fill(self, chosen: dict) -> None:
        self.chosen = chosen
        self.title.setText(chosen["title"])
        self.status.setText(status_line(self.app))
        self.tagline.setText(chosen["tagline"])
        self.tip.setText(f"💡  {chosen['tip']}")
        while self.grid.count():
            item = self.grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        if not self.compact:
            for i, text in enumerate(chosen["suggestions"]):
                self.grid.addWidget(_chip(text, self.send), i // 2, i % 2)

    def refresh(self) -> None:
        """A new greeting with a quick fade."""
        chosen = pick(self.app.state, self._user, self._assistant)
        anim = QPropertyAnimation(self._fx, b"opacity", self)
        anim.setDuration(500)
        anim.setKeyValueAt(0.0, 1.0)
        anim.setKeyValueAt(0.45, 0.0)
        anim.setKeyValueAt(1.0, 1.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        QTimer.singleShot(230, lambda: self.fill(chosen))
        anim.start()
        self._anim = anim

    def next_tip(self) -> None:
        last = (self.app.state.get("welcome_last") or {}) if self.app.state is not None else {}
        i = (int(last.get("tip", -1)) + 1) % len(TIPS)
        if self.app.state is not None:
            self.app.state.set("welcome_last", {**last, "tip": i})
        self.tip.setText(f"💡  {TIPS[i]}")
