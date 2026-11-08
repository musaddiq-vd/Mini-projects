"""Mini mode: floating orb with a fold-out chat and what's next."""

import sys
import time

import pytest


@pytest.fixture
def window(monkeypatch, isolated_home, sandbox):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    QSettings("MusaddiqKhan", "Jarvis").setValue("mini_chat", False)
    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings(allowed_roots=[str(sandbox)]))
    w = MainWindow(app, Bridge())
    w.show()
    yield w, app, qt
    w._leave_mini()
    w.reminders.close_all()
    QSettings("MusaddiqKhan", "Jarvis").setValue("mini_chat", False)


def bubble_texts(mini):
    from PyQt6.QtWidgets import QLabel

    return [lb.text() for lb in mini.panel.findChildren(QLabel) if lb.objectName().startswith("bubble")]


def test_chat_folds_out_and_shows_messages(window):
    w, app, qt = window
    app.events.emit("user", text="earlier question")
    w.toggle_mini()
    mini = w.mini
    assert not mini.chat_open and not mini.panel.isVisibleTo(mini) and mini.height() == 290
    mini.toggle_chat()
    assert mini.chat_open and mini.panel.isVisibleTo(mini) and mini.height() > 290
    for i in range(6):
        app.events.emit("assistant", text=f"answer {i}")
    app.events.emit("assistant", text="(system: hidden)")
    qt.processEvents()
    texts = bubble_texts(mini)
    assert texts == ["answer 2", "answer 3", "answer 4", "answer 5"]  # only the last few
    mini.toggle_chat()
    assert not mini.chat_open and mini.height() == 290


def test_typing_in_mini_sends_to_jarvis(window, monkeypatch):
    w, app, qt = window
    sent = []
    monkeypatch.setattr(w, "_send", sent.append)
    w.mini = None
    w.toggle_mini()
    w.mini.toggle_chat()
    w.mini.input.setText("  open notepad  ")
    w.mini._submit()
    w.mini.input.setText("   ")
    w.mini._submit()
    assert sent == ["open notepad"] and w.mini.input.text() == ""


def test_next_up_picks_the_sooner_thing(window):
    from jarvis.ui.mini import next_up

    w, app, qt = window
    now = time.time()
    assert next_up(app, now) is None
    app.state.set("reminders", [{"id": 1, "due": now + 3 * 3600, "message": "Late one", "repeat": ""},
                                {"id": 2, "due": now + 25 * 60, "message": "Call mom", "repeat": ""},
                                {"id": 3, "due": now - 60, "message": "Gone", "repeat": ""}])
    icon, text = next_up(app, now)
    assert icon == "⏰" and text == "in 25 min · Call mom"
    w.toggle_mini()
    w.mini.toggle_chat()
    w.mini.refresh_next()
    assert w.mini.next_chip.isVisibleTo(w.mini) and "Call mom" in w.mini.next_chip.text()


def test_opening_the_chat_stays_on_screen(window):
    w, app, qt = window
    w.toggle_mini()
    mini = w.mini
    area = mini.screen().availableGeometry()
    mini.move(area.right() - mini.width(), area.bottom() - mini.height())
    mini.toggle_chat()
    assert mini.geometry().bottom() <= area.bottom() and mini.y() >= area.top()


def test_choice_is_remembered(window):
    from PyQt6.QtCore import QSettings

    w, app, qt = window
    w.toggle_mini()
    w.mini.toggle_chat()
    assert QSettings("MusaddiqKhan", "Jarvis").value("mini_chat", type=bool) is True
