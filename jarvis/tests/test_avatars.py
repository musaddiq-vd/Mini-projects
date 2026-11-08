"""Avatar packs for the orb."""

import sys
import time

import pytest


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


@pytest.mark.parametrize("kind", ["ai", "robot", "cat", "anime", "minimal"])
def test_every_avatar_draws_every_mood(qt, kind):
    from PyQt6.QtCore import QPoint
    from PyQt6.QtGui import QColor, QPainter, QPixmap

    from jarvis.ui.orb import Orb

    o = Orb("#00d4ff")
    o.resize(260, 300)
    o.set_avatar(kind)
    assert o.avatar == kind
    for state, feel, level in [("listening", "", 0), ("speaking", "", 0.6), ("thinking", "", 0),
                               ("sleeping", "", 0), ("listening", "love", 0), ("listening", "happy", 0),
                               ("listening", "sad", 0), ("offline", "", 0)]:
        o.set_state(state)
        o.level = level
        o.emotion, o.emotion_until = feel, time.monotonic() + 5
        pm = QPixmap(o.size())
        pm.fill(QColor("black"))
        painter = QPainter(pm)
        o.render(painter, QPoint(0, 0))
        painter.end()
    o.close()


def test_unknown_avatar_falls_back(qt):
    from jarvis.ui.orb import Orb

    o = Orb("#00d4ff")
    o.set_avatar("dragon")
    assert o.avatar == "ai"
    o.close()


def test_change_avatar_by_voice(qt, isolated_home, sandbox):
    from jarvis.app import create_app
    from jarvis.config import Settings, load_settings
    from jarvis.ui.window import Bridge, MainWindow

    app = create_app(Settings(allowed_roots=[str(sandbox)], orb_face=False))
    w = MainWindow(app, Bridge())
    assert "error" in app.registry.call("jarvis_display", {"action": "avatar", "avatar": "dragon"}, app.ctx)
    app.registry.call("jarvis_display", {"action": "avatar", "avatar": "cat"}, app.ctx)
    qt.processEvents()
    assert w.big_orb.avatar == "cat" and w.big_orb.face  # the face is turned on so the avatar shows
    assert load_settings().avatar == "cat"
    w.toggle_mini()
    assert w.mini.orb.avatar == "cat"
    w._leave_mini()
    w.reminders.close_all()
