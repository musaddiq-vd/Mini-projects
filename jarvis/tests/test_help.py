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
    assert "Hi Musaddique! I'm Jarvis." in text and "F1" in text


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
    assert qt is not None and "I'm Jarvis." not in w.chat.toPlainText()
