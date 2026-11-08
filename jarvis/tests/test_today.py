"""The Today strip under the orb."""

import sys
import time

import pytest


@pytest.fixture
def strip(monkeypatch, isolated_home):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.today import TodayStrip

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings())
    timers = {}
    sent = []
    s = TodayStrip(app, lambda: timers, sent.append)
    s.timers_dict, s.sent, s.qt = timers, sent, qt
    return s, app


def visible(s):
    return {k: b.text() for k, b in s.chips.items() if b.isVisibleTo(s)}


def test_all_clear(strip):
    s, app = strip
    s.refresh()
    assert visible(s) == {"todo": "✨ All clear today"}


def test_shows_reminder_timer_todos_and_badges(strip):
    s, app = strip
    app.state.set("reminders", [{"id": 1, "due": time.time() + 600, "message": "Call mom", "repeat": ""},
                                {"id": 2, "due": time.time() - 5, "message": "past", "repeat": ""}])
    app.state.set("notes", [{"id": 1, "text": "milk", "todo": True, "done": False},
                            {"id": 2, "text": "bank", "todo": True, "done": True},
                            {"id": 3, "text": "idea", "todo": False, "done": False}])
    s.timers_dict[1] = (time.time() + 65, 120, "tea")
    app.events.emit("user", text="hello")  # unlocks the first badge
    s.refresh()
    v = visible(s)
    assert "Call mom" in v["reminder"] and v["timer"].startswith("⏱ 1:0") and "tea" in v["timer"]
    got = len(app.achievements.unlocked())  # "hello" unlocks one; late at night "night owl" too
    assert v["todo"] == "📝 1 to-do" and v["badges"] == f"🏆 {got}" and got >= 1


def test_chips_ask_jarvis(strip):
    s, app = strip
    app.state.set("notes", [{"id": 1, "text": "milk", "todo": True, "done": False}])
    s.refresh()
    s.chips["todo"].click()
    assert s.sent == ["What's on my to-do list?"]
