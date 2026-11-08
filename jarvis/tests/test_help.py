"""'What can I say?' panel and the welcome card."""

import sys

import pytest


@pytest.fixture
def window(monkeypatch, isolated_home):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings(user_name="musaddique khan"))
    w = MainWindow(app, Bridge())
    w.show()
    w.qt = qt
    return w, app


def test_help_opens_closes_and_sends_examples(window):
    from PyQt6.QtWidgets import QPushButton

    w, app = window
    sent = []
    w._send = sent.append
    w.toggle_help()
    assert w.help.isVisible()
    w._escape()
    assert not w.help.isVisible()
    w.toggle_help()
    w.help.send = sent.append
    button = next(b for b in w.help.findChildren(QPushButton) if "Open Spotify" in b.text())
    button.click()
    assert sent == ["Open Spotify"] and not w.help.isVisible()


def test_every_topic_has_examples():
    from jarvis.ui.help import TOPICS

    assert len(TOPICS) >= 8 and all(len(examples) >= 2 for _, _, examples in TOPICS)


def test_welcome_card_greets_by_first_name(window):
    w, app = window
    text = w.chat.toPlainText()
    assert "Musaddique" in text and "💡" in text and w.welcome.title.text().count("Musaddique") == 1


def test_no_welcome_when_there_is_history(monkeypatch, isolated_home):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings())
    app.events.emit("user", text="hello")
    w = MainWindow(app, Bridge())
    assert qt is not None and w.welcome.compact  # with history: only the slim "welcome back" banner


def test_welcome_card_goes_away_when_the_chat_starts(window):
    w, app = window
    assert w._welcome_row is not None
    app.events.emit("user", text="hello")
    w.qt.processEvents()
    w.qt.processEvents()
    assert w._welcome_row is None


def test_welcome_changes_every_time():
    import random
    from datetime import datetime

    from jarvis.ui.welcome import GREETINGS, TIPS, part_of_day, pick

    class State(dict):
        def set(self, k, v):
            self[k] = v

    st = State()
    evening = datetime(2026, 11, 9, 19, 0)
    seen = [pick(st, "Musaddiq Khan", "Jarvis", evening, random.Random(i)) for i in range(12)]
    assert all(a["title"] != b["title"] for a, b in zip(seen, seen[1:]))  # never twice in a row
    assert all(", Musaddiq" in s["title"] for s in seen) and seen[0]["part"] == "evening"
    assert [s["tip"] for s in seen[:3]] == TIPS[:3]  # tips go round in turn
    assert all(len(set(s["suggestions"])) == 4 for s in seen)
    assert "{n}" not in pick(State(), "", "Jarvis", evening)["title"]  # no name yet: no stray comma
    assert part_of_day(datetime(2026, 1, 1, 6)) == "morning" and part_of_day(datetime(2026, 1, 1, 23)) == "night"
    assert part_of_day(datetime(2026, 1, 1, 2)) == "night" and all(len(v) >= 5 for v in GREETINGS.values())


def test_welcome_status_and_refresh(window):
    import time as _t

    w, app = window
    app.state.set("notes", [{"id": 1, "text": "milk", "todo": True, "done": False}])
    app.state.set("reminders", [{"id": 1, "due": _t.time() + 60, "message": "x", "repeat": ""}])
    w.welcome.fill(__import__("jarvis.ui.welcome", fromlist=["pick"]).pick(app.state, "", "Jarvis"))
    assert "1 to-do" in w.welcome.status.text() and "1 reminder today" in w.welcome.status.text()
    tip = w.welcome.tip.text()
    w.welcome.next_tip()
    assert w.welcome.tip.text() != tip
