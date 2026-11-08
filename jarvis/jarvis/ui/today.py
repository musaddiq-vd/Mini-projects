"""The 'Today' strip under the orb: next reminder, running timer, open to-dos, streak, badges.

Each chip is clickable and asks Jarvis about that item.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QWidget


def _left(seconds: float) -> str:
    seconds = max(0, int(seconds + 0.999))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _when(ts: float) -> str:
    d = datetime.fromtimestamp(ts)
    stamp = d.strftime("%I:%M %p").lstrip("0")
    return stamp if d.date() == datetime.now().date() else d.strftime("%a ") + stamp


class TodayStrip(QWidget):
    def __init__(self, app, timers: Callable[[], dict], send: Callable[[str], None], parent=None):
        super().__init__(parent)
        self.app = app
        self.timers = timers  # id -> (ends, total, label), from the orb
        self.send = send
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.chips: dict[str, QPushButton] = {}
        for key, ask in (("meeting", "What's on my calendar today?"), ("reminder", "What reminders do I have?"), ("timer", "How much time is left on my timers?"),
                         ("todo", "What's on my to-do list?"), ("streak", None), ("badges", None)):
            b = QPushButton(objectName="chip")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            if ask:
                b.clicked.connect(lambda _, q=ask: self.send(q))
            h.addWidget(b)
            self.chips[key] = b
        h.addStretch(1)
        self.on_badges: Callable[[], None] | None = None
        self.chips["badges"].clicked.connect(lambda: self.on_badges and self.on_badges())
        self.chips["streak"].clicked.connect(lambda: self.on_badges and self.on_badges())
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(1000)
        self.refresh()

    def refresh(self) -> None:
        st = self.app.state
        now = time.time()
        self._meeting(now)
        upcoming = sorted((r for r in st.get("reminders", []) or [] if r.get("due", 0) > now), key=lambda r: r["due"])
        if upcoming:
            r = upcoming[0]
            msg = r.get("message", "")
            self._set("reminder", f"⏰ {_when(r['due'])} · {msg[:18] + ('…' if len(msg) > 18 else '')}",
                      f"Next reminder: {msg} at {_when(r['due'])}" + (f"  (+{len(upcoming) - 1} more)" if len(upcoming) > 1 else ""))
        else:
            self._set("reminder", None)
        timers = sorted(self.timers().values(), key=lambda t: t[0])
        if timers:
            ends, _, label = timers[0]
            self._set("timer", f"⏱ {_left(ends - now)}" + (f" {label[:10]}" if label else ""),
                      f"{len(timers)} timer(s) running")
        else:
            self._set("timer", None)
        todos = [n for n in st.get("notes", []) or [] if n.get("todo") and not n.get("done")]
        self._set("todo", f"📝 {len(todos)} to-do{'s' if len(todos) != 1 else ''}" if todos else None,
                  "\n".join("• " + n["text"] for n in todos[:8]))
        ach = getattr(self.app, "achievements", None)
        if ach is not None:
            streak = ach.streak()
            self._set("streak", f"🔥 {streak}" if streak >= 2 else None, f"{streak}-day streak")
            got = len(ach.unlocked())
            self._set("badges", f"🏆 {got}" if got else None, f"{got} achievements unlocked - click to see them")
        if not any(b.isVisibleTo(self) for b in self.chips.values()):
            self._set("todo", "✨ All clear today", "No reminders, timers or to-dos - enjoy!")

    def _meeting(self, now: float) -> None:
        try:
            from jarvis.tools.calendar_sync import next_event

            e = next_event(self.app.settings, now)
        except Exception:
            e = None
        if not e:
            self._set("meeting", None)
            return
        title = e["title"]
        short = title[:18] + ("…" if len(title) > 18 else "")
        if e["start"] <= now:
            text, tip = f"📅 Now · {short}", f"In progress: {title} (until {_when(e['end'])})"
        elif e["start"] - now < 3600:
            text, tip = f"📅 in {max(1, int((e['start'] - now) // 60))} min · {short}", f"{title} at {_when(e['start'])}"
        else:
            text, tip = f"📅 {_when(e['start'])} · {short}", f"Next meeting: {title} at {_when(e['start'])}"
        if e.get("location"):
            tip += f"\n📍 {e['location']}"
        self._set("meeting", text, tip)

    def _set(self, key: str, text: str | None, tip: str = "") -> None:
        b = self.chips[key]
        if text is None:
            b.hide()
            return
        if b.text() != text:
            b.setText(text)
        b.setToolTip(tip)
        b.show()
