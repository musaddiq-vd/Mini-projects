"""The red "REC" pill next to the connection pill while a meeting is recorded (privacy at a glance)."""

from __future__ import annotations

import time

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QLabel

from jarvis.ui.theme import DANGER, WARN


def rec_text(title: str, seconds: float) -> str:
    s = max(0, int(seconds))
    clock = f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"
    short = title if len(title) <= 22 else title[:21] + "…"
    return f"● REC {clock}  ·  {short}"


class MeetingPill(QLabel):
    """Hidden until a `meeting` event says recording started; ticks every second."""

    def __init__(self, parent=None):
        super().__init__("", parent)
        self.setVisible(False)
        self.state = ""
        self.title = ""
        self.started = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

    def _style(self, color: str) -> None:
        self.setStyleSheet(f"color: {color}; border: 1px solid {color}; border-radius: 9px; padding: 1px 10px; "
                           "font-size: 10px; font-weight: 700; background: transparent;")

    def on_event(self, data: dict) -> None:
        state = data.get("state", "")
        self.state = state
        if state == "recording":
            self.title = data.get("title", "Meeting")
            self.started = float(data.get("started") or time.time())
            sources = " + ".join(data.get("sources") or ["microphone"])
            self.setToolTip(f"Recording '{self.title}' from: {sources}. Say “stop the meeting” to stop.")
            self._style(DANGER)
            self._tick()
            self.setVisible(True)
            self._timer.start()
        elif state == "processing":
            self._timer.stop()
            self._style(WARN)
            self.setText("● Writing meeting notes…")
            self.setToolTip("The recording has stopped; Jarvis is writing the notes.")
            self.setVisible(True)
        else:  # stopped / deleted
            self._timer.stop()
            self.setVisible(False)

    def _tick(self) -> None:
        self.setText(rec_text(self.title, time.time() - self.started))
