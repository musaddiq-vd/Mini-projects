"""AI core drawing, toasts, typewriter text, glow and the animated background."""

import sys
import time

import pytest


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


@pytest.mark.parametrize("state", ["connecting", "listening", "speaking", "thinking", "muted", "sleeping", "offline"])
@pytest.mark.parametrize("feeling", ["", "love", "sad", "wink", "surprised"])
def test_ai_core_paints_every_state_and_feeling(qt, state, feeling):
    from jarvis.ui.orb import Orb

    orb = Orb("#00d4ff")
    orb.resize(300, 330)
    orb.boot = 0.5 if state == "connecting" else 1.0
    orb.set_state(state)
    orb.set_level(0.6)
    if feeling:
        orb.emote(feeling)
    orb.set_timer(1, time.time() + 30, 60, "tea")
    orb._tick()
    image = orb.grab().toImage()  # runs paintEvent; any drawing error raises here
    assert not image.isNull()


def test_typewriter_reveals_then_finishes(qt):
    from PyQt6.QtWidgets import QLabel

    from jarvis.ui.fx import Typewriter

    label = QLabel()
    tw = Typewriter(label, "Hello there, this is Jarvis.")
    end = time.time() + 3
    while label.text() != tw.text and time.time() < end:
        qt.processEvents()
        time.sleep(0.01)
    assert label.text() == "Hello there, this is Jarvis."


def test_toasts_stack_and_go_away(qt):
    from PyQt6.QtWidgets import QWidget

    from jarvis.ui.fx import ToastHost

    host_widget = QWidget()
    host_widget.resize(900, 600)
    host_widget.show()
    host = ToastHost(host_widget)
    a = host.show("🔔", "first", "#ffb547", seconds=30)
    b = host.show("⏱", "second", "#3ddc97", seconds=30)
    assert a.isVisible() and b.isVisible()
    assert b.y() > a.y() or b._in is not None  # second one sits below the first
    a._gone_now()  # what the fade-out ends with
    assert not _alive(a) and host.toasts == [b]


def _alive(w):
    from jarvis.ui.fx import _alive as alive

    return alive(w)


def test_window_uses_background_glow_and_toasts(qt):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.fx import AnimatedBackground
    from jarvis.ui.window import Bridge, MainWindow

    app = create_app(Settings())
    w = MainWindow(app, Bridge())
    w.show()
    assert isinstance(w.centralWidget(), AnimatedBackground)
    assert len(w._glows) >= 5
    app.events.emit("log", text="🔔 Reminder: tea")
    qt.processEvents()
    assert any(_alive(t) for t in w.toasts.toasts)
    assert not w.background.grab().isNull()
