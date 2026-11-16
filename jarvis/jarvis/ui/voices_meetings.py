"""Settings > Privacy: voice profiles (who Jarvis recognises) and recorded meetings, each with Delete."""

from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QCheckBox, QGridLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from jarvis import meetings, voices

VOICE_HINT = ("Jarvis can tell family members apart by voice and use each person's name, language and style. "
              "Say “learn my voice” to add someone. Only voice numbers are kept, never recordings. It's a "
              "convenience, not a security check.")
MEETING_HINT = ("Meetings are recorded only when you ask (“take notes of this meeting”), with a red REC pill "
                "while it runs. Turn recording off completely with the Meetings permission above.")


def _human(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB" if n >= 1024 * 1024 else f"{n / 1024:.0f} KB"


class VoicesMeetingsBox(QWidget):
    def __init__(self, app, settings, parent=None):
        super().__init__(parent)
        self.app = app
        self.s = settings
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        v.addWidget(QLabel("VOICE PROFILES", objectName="section"))
        hint = QLabel(VOICE_HINT, objectName="muted")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.voice_id = QCheckBox("Recognise who is speaking (uses the saved voice profiles)")
        self.voice_id.setChecked(bool(getattr(settings, "voice_id", True)))
        v.addWidget(self.voice_id)
        self.guest_tools = QCheckBox("Let guests (unknown voices) use WhatsApp, contacts and money")
        self.guest_tools.setChecked(bool(getattr(settings, "guest_private_tools", False)))
        v.addWidget(self.guest_tools)
        self.voice_grid = QGridLayout()
        self.voice_grid.setHorizontalSpacing(12)
        v.addLayout(self.voice_grid)

        v.addSpacing(8)
        v.addWidget(QLabel("MEETINGS", objectName="section"))
        hint = QLabel(MEETING_HINT, objectName="muted")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.meeting_offer = QCheckBox("When a calendar meeting starts, ask “Shall I take notes?”")
        self.meeting_offer.setChecked(bool(getattr(settings, "meeting_offer", True)))
        v.addWidget(self.meeting_offer)
        self.meeting_grid = QGridLayout()
        self.meeting_grid.setHorizontalSpacing(12)
        v.addLayout(self.meeting_grid)
        self.refresh()

    def apply_to(self, s) -> None:
        s.voice_id = self.voice_id.isChecked()
        s.guest_private_tools = self.guest_tools.isChecked()
        s.meeting_offer = self.meeting_offer.isChecked()

    @staticmethod
    def _clear(grid: QGridLayout) -> None:
        while grid.count():
            item = grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def refresh(self) -> None:
        self._clear(self.voice_grid)
        profiles = voices.load_profiles()
        if not profiles:
            self.voice_grid.addWidget(QLabel("No voice profiles yet.", objectName="muted"), 0, 0)
        for row, p in enumerate(profiles):
            badge = "  👑 owner" if p.get("owner") else ""
            self.voice_grid.addWidget(QLabel(f"🗣  {p['name']}{badge}"), row, 0)
            info = f"language {p.get('language', 'auto')} · style {p.get('style') or 'default'} · " \
                   f"facts: {p.get('memory', 'all')}"
            self.voice_grid.addWidget(QLabel(info, objectName="muted"), row, 1)
            btn = QPushButton("Delete")
            btn.clicked.connect(lambda _, name=p["name"]: self._delete_voice(name))
            self.voice_grid.addWidget(btn, row, 2)

        self._clear(self.meeting_grid)
        rows = meetings.all_meetings()
        if not rows:
            self.meeting_grid.addWidget(QLabel("No meetings recorded.", objectName="muted"), 0, 0)
        running = meetings.current()
        for row, m in enumerate(rows[:30]):
            self.meeting_grid.addWidget(QLabel(f"🎙  {m.get('title', m['id'])}"), row, 0)
            size = _human(meetings.folder_size(meetings.meetings_dir() / m["id"]))
            self.meeting_grid.addWidget(QLabel(f"{meetings.when_text(m)} · {size}", objectName="muted"), row, 1)
            folder = QPushButton("Open")
            folder.clicked.connect(lambda _, f=m["folder"]: self._open(f))
            self.meeting_grid.addWidget(folder, row, 2)
            btn = QPushButton("Delete")
            recording = running is not None and running.meeting.id == m["id"]
            btn.setEnabled(not recording)
            if recording:
                btn.setToolTip("Still recording - stop it first.")
            btn.clicked.connect(lambda _, mid=m["id"], t=m.get("title", ""): self._delete_meeting(mid, t))
            self.meeting_grid.addWidget(btn, row, 3)

    def _ask(self, what: str) -> bool:
        answer = QMessageBox.question(self, "Delete?", f"Permanently delete {what}?\nThis can't be undone.",
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def _delete_voice(self, name: str) -> None:
        if self._ask(f"{name}'s voice profile"):
            voices.delete_profile(name)
            self.refresh()

    def _delete_meeting(self, mid: str, title: str) -> None:
        if self._ask(f"the meeting '{title or mid}' (audio, transcript and notes)"):
            meetings.delete(meetings.meetings_dir() / mid)
            self.refresh()

    @staticmethod
    def _open(folder: str) -> None:
        try:
            os.startfile(folder)  # type: ignore[attr-defined]  # Windows
        except (AttributeError, OSError):
            pass

    def delete_all(self) -> None:
        """Used by "Delete everything": every voice profile and every meeting that isn't recording."""
        for p in voices.load_profiles():
            voices.delete_profile(p["name"])
        running = meetings.current()
        for m in meetings.all_meetings():
            if running is None or running.meeting.id != m["id"]:
                meetings.delete(meetings.meetings_dir() / m["id"])
        self.refresh()
