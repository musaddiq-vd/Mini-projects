"""Planner: your to-dos, reminders, timers and routines in one place - tick, add, cancel, run.

Every change goes through the same tools Jarvis uses (so guardrails and undo apply), then the panel
refreshes. Opens with the 📋 button or Ctrl+P.
"""

from __future__ import annotations

import html
import re
import time
from collections.abc import Callable
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

from jarvis.ui import theme


def _when(ts: float) -> str:
    d = datetime.fromtimestamp(ts)
    stamp = d.strftime("%I:%M %p").lstrip("0")
    if d.date() == datetime.now().date():
        return stamp
    return d.strftime("%a %d %b, ") + stamp


def _left(seconds: float) -> str:
    seconds = max(0, int(seconds + 0.999))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_when(text: str) -> dict | None:
    """'20 min' / '1 hour' -> minutes; '6:30 pm' / '18:30' -> at; else None."""
    t = text.strip().lower()
    m = re.fullmatch(r"(?:in\s*)?(\d+(?:\.\d+)?)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours)", t)
    if m:
        n = float(m.group(1))
        return {"minutes": n * 60 if m.group(2).startswith("h") else n}
    if re.search(r"\d", t):
        return {"at": t}
    return None


class _Section(QFrame):
    def __init__(self, icon: str, title: str):
        super().__init__(objectName="card")
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(14, 12, 14, 12)
        self.v.setSpacing(8)
        head = QHBoxLayout()
        self.title = QLabel(f"{icon}  {title}", objectName="h2")
        head.addWidget(self.title)
        head.addStretch(1)
        self.count = QLabel("", objectName="muted")
        head.addWidget(self.count)
        self.v.addLayout(head)
        self.body = QVBoxLayout()
        self.body.setSpacing(4)
        self.v.addLayout(self.body)


    def clear(self) -> None:
        while self.body.count():
            item = self.body.takeAt(0)
            if item.widget():
                item.widget().hide()  # gone at once, not one frame later
                item.widget().deleteLater()
            elif item.layout():
                _clear(item.layout())

    def empty(self, text: str) -> None:
        self.body.addWidget(QLabel(text, objectName="muted"))


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().hide()
            item.widget().deleteLater()
        elif item.layout():
            _clear(item.layout())


def _small(text: str, tip: str, fn) -> QPushButton:
    b = QPushButton(text, objectName="small")
    b.setToolTip(tip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.clicked.connect(fn)
    return b


class PlannerPanel(QFrame):
    def __init__(self, parent: QWidget, app, timers: Callable[[], dict], send: Callable[[str], None]):
        super().__init__(parent)
        self.setObjectName("plannerPanel")
        self.app = app
        self.timers = timers
        self.send = send
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        head = QHBoxLayout()
        head.addWidget(QLabel("PLANNER", objectName="section"))
        head.addStretch(1)
        self.summary = QLabel("", objectName="accent")
        head.addWidget(self.summary)
        head.addWidget(_small("✕", "Close (Esc)", self.hide))
        outer.addLayout(head)

        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setSpacing(10)

        # to-dos (left column)
        self.todo = _Section("📝", "To-do")
        add = QHBoxLayout()
        self.todo_input = QLineEdit()
        self.todo_input.setPlaceholderText("Add a to-do and press Enter...")
        self.todo_input.returnPressed.connect(self._add_todo)
        add.addWidget(self.todo_input, 1)
        add.addWidget(_small("＋", "Add", self._add_todo))
        self.todo.v.insertLayout(1, add)
        self.done_toggle = QCheckBox("Show finished")
        self.done_toggle.toggled.connect(lambda _: self.refresh())
        self.todo.v.addWidget(self.done_toggle)
        grid.addWidget(self.todo, 0, 0)

        # reminders
        self.rem = _Section("⏰", "Reminders")
        row = QHBoxLayout()
        self.rem_text = QLineEdit()
        self.rem_text.setPlaceholderText("Remind me to...")
        self.rem_when = QLineEdit()
        self.rem_when.setPlaceholderText("6:30 pm / 20 min")
        self.rem_when.setMaximumWidth(140)
        self.rem_when.returnPressed.connect(self._add_reminder)
        self.rem_text.returnPressed.connect(self._add_reminder)
        row.addWidget(self.rem_text, 1)
        row.addWidget(self.rem_when)
        row.addWidget(_small("＋", "Add reminder", self._add_reminder))
        self.rem.v.insertLayout(1, row)
        self.rem_error = QLabel("", objectName="bad")
        self.rem_error.hide()
        self.rem.v.insertWidget(2, self.rem_error)
        grid.addWidget(self.rem, 1, 0)

        # timers
        self.tim = _Section("⏱", "Timers")
        quick = QHBoxLayout()
        for minutes in (1, 5, 10, 25):
            b = QPushButton(f"{minutes} min", objectName="chip")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _, m=minutes: self._call("set_timer", {"minutes": m}))
            quick.addWidget(b)
        focus = QPushButton("🎯 Focus 25", objectName="chip")
        focus.setToolTip("A 25-minute focus session: no check-ins or alerts until it ends")
        focus.clicked.connect(lambda: self._call("focus_mode", {"minutes": 25}))
        quick.addWidget(focus)
        quick.addStretch(1)
        self.tim.v.insertLayout(1, quick)
        grid.addWidget(self.tim, 2, 0)

        # routines
        self.rou = _Section("🔁", "Routines")
        grid.addWidget(self.rou, 3, 0)
        grid.setRowStretch(4, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._refresh_timers)
        self.hide()

    # -- open / close ------------------------------------------------------------------------
    def toggle(self) -> None:
        if self.isVisible():
            self.hide()
            return
        self.setStyleSheet(f"QFrame#plannerPanel {{ background: {theme.BG}; border: 1px solid {theme.ACCENT}; "
                           "border-radius: 16px; }")
        self.setGeometry(self.parentWidget().rect().adjusted(8, 8, -8, -8))
        self.refresh()
        self.show()
        self.raise_()
        self._tick.start(1000)
        self.todo_input.setFocus()

    def hideEvent(self, event) -> None:  # noqa: N802
        if getattr(self, "_tick", None) is not None:
            self._tick.stop()
        super().hideEvent(event)

    # -- actions --------------------------------------------------------------------------------
    def _call(self, tool: str, args: dict) -> dict:
        result = self.app.registry.call(tool, args, self.app.ctx)
        if "error" in result and self.app.events is not None:
            self.app.events.emit("log", text=f"⚠ {result['error'][:160]}")
        self.refresh()
        return result

    def _add_todo(self) -> None:
        text = self.todo_input.text().strip()
        if text:
            self.todo_input.clear()
            self._call("add_note", {"text": text, "todo": True})

    def _add_reminder(self) -> None:
        text, when = self.rem_text.text().strip(), parse_when(self.rem_when.text())
        if not text or when is None:
            self.rem_error.setText("Write what to remember and when (like '6:30 pm' or '20 min').")
            self.rem_error.show()
            return
        result = self._call("set_reminder", {"message": text, **when})
        if "error" in result:
            self.rem_error.setText(result["error"][:140])
            self.rem_error.show()
        else:
            self.rem_error.hide()
            self.rem_text.clear()
            self.rem_when.clear()

    # -- drawing ----------------------------------------------------------------------------------
    def refresh(self) -> None:
        st = self.app.state
        notes = st.get("notes", []) or []
        todos = [n for n in notes if n.get("todo")]
        open_ = [n for n in todos if not n.get("done")]
        self.todo.clear()
        shown = todos if self.done_toggle.isChecked() else open_
        for n in shown:
            row = QHBoxLayout()
            box = QCheckBox(n["text"].replace("&", "&&"))  # '&' would otherwise become a shortcut key
            box.setChecked(bool(n.get("done")))
            box.setEnabled(not n.get("done"))
            box.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            box.setToolTip(n["text"])
            box.toggled.connect(lambda on, i=n["id"]: on and self._call("complete_note", {"item": str(i)}))
            row.addWidget(box, 1)
            row.addWidget(_small("🗑", "Delete", lambda _, i=n["id"]: self._call("delete_note", {"item": str(i)})))
            self.todo.body.addLayout(row)
        if not shown:
            self.todo.empty("Nothing to do - nice! ✨" if not todos else "All done! Tick 'Show finished' to see them.")
        self.todo.count.setText(f"{len(open_)} open")

        self.rem.clear()
        now = time.time()
        reminders = sorted(st.get("reminders", []) or [], key=lambda r: r.get("due", 0))
        for r in reminders:
            row = QHBoxLayout()
            repeat = {"daily": " · every day", "weekdays": " · weekdays"}.get(r.get("repeat", ""), "")
            lab = QLabel(f"<b>{_when(r['due'])}</b>{repeat}  ·  {html.escape(r['message'])}")
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setWordWrap(True)
            if r["due"] < now:
                lab.setStyleSheet(f"color: {theme.MUTED};")
            row.addWidget(lab, 1)
            row.addWidget(_small("✕", "Cancel", lambda _, i=r["id"]: self._call("cancel_reminder", {"id": i})))
            self.rem.body.addLayout(row)
        if not reminders:
            self.rem.empty("No reminders.")
        self.rem.count.setText(f"{len(reminders)}")

        self._refresh_timers()

        self.rou.clear()
        routines = st.get("routines", {}) or {}
        for name, steps in sorted(routines.items()):
            row = QHBoxLayout()
            lab = QLabel(f"<b>{html.escape(name)}</b>  ·  {len(steps)} step{'s' if len(steps) != 1 else ''}")
            lab.setToolTip("\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
            row.addWidget(lab, 1)
            row.addWidget(_small("▶", "Run it", lambda _, n=name: self.send(f"Run my routine '{n}'.")))
            row.addWidget(_small("🗑", "Delete", lambda _, n=name: self._call("delete_routine", {"name": n})))
            self.rou.body.addLayout(row)
        if not routines:
            self.rou.empty("No routines yet. Say: \"save a routine called work mode: open VS Code and Chrome\".")
        self.rou.count.setText(f"{len(routines)}")

        self.summary.setText(f"📝 {len(open_)}   ⏰ {len([r for r in reminders if r['due'] > now])}   "
                             f"⏱ {len(self.timers())}   🔁 {len(routines)}")

    def _refresh_timers(self) -> None:
        self.tim.clear()
        now = time.time()
        timers = sorted(self.timers().items(), key=lambda kv: kv[1][0])
        for tid, (ends, total, label) in timers:
            row = QHBoxLayout()
            frac = max(0.0, min(1.0, (ends - now) / total)) if total else 0
            lab = QLabel(f"<b>{_left(ends - now)}</b>  ·  {html.escape(label or 'timer')}")
            row.addWidget(lab, 1)
            bar = QFrame()
            bar.setFixedSize(int(80 * frac) + 2, 6)
            bar.setStyleSheet(f"background: {theme.ACCENT}; border-radius: 3px;")
            row.addWidget(bar)
            row.addWidget(_small("■", "Stop", lambda _, i=tid: self._call("cancel_timer", {"id": i})))
            self.tim.body.addLayout(row)
        if not timers:
            self.tim.empty("No timers running.")
        self.tim.count.setText(f"{len(timers)}")
