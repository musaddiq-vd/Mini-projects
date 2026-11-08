"""Fast start: the window opens at once; the assistant loads on a background thread."""

import sys
import time

import pytest


class SlowAssistant:
    made = []

    def __init__(self, app):
        time.sleep(0.6)  # loading Gemini takes a while
        self.app = app
        self.sent = []
        self.awake = True
        SlowAssistant.made.append(self)

    def run_in_thread(self):
        return None

    def send_text(self, text):
        self.sent.append(text)


@pytest.fixture
def window(monkeypatch, isolated_home, sandbox):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings(api_key="k", allowed_roots=[str(sandbox)]))
    w = MainWindow(app, Bridge())
    yield w, app, qt
    w.reminders.close_all()


def pump_until(qt, cond, secs=4.0):
    end = time.time() + secs
    while time.time() < end and not cond():
        qt.processEvents()
        time.sleep(0.01)
    return cond()


def test_window_does_not_wait_for_the_assistant(window, monkeypatch):
    import jarvis.live

    w, app, qt = window
    SlowAssistant.made.clear()
    monkeypatch.setattr(jarvis.live, "LiveAssistant", SlowAssistant)
    t = time.perf_counter()
    w.start_assistant()
    assert time.perf_counter() - t < 0.3  # returned straight away
    w.start_assistant()  # a second call while booting does nothing
    w._send("open notepad")  # typed while it's still loading
    assert w.assistant is None
    assert pump_until(qt, lambda: w.assistant is not None)
    assert len(SlowAssistant.made) == 1 and w.assistant.sent == ["open notepad"]


def test_a_failed_start_is_shown(window, monkeypatch):
    import jarvis.live
    from PyQt6.QtWidgets import QLabel

    w, app, qt = window

    def broken(app):
        raise OSError("no audio device")

    monkeypatch.setattr(jarvis.live, "LiveAssistant", broken)
    w.start_assistant()
    assert pump_until(qt, lambda: not w._booting)
    text = " ".join(lb.text() for lb in w.chat.findChildren(QLabel))
    assert "Couldn't start" in text and "no audio device" in text and w.assistant is None
