"""The power-up wake animation and sound."""

import sys
import time

import pytest


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def test_curves():
    from jarvis.ui import powerup

    assert powerup.core_scale(0) == 1 and powerup.core_scale(1) == 1
    assert powerup.core_scale(0.2) < 1 < powerup.core_scale(0.35)  # shrink while charging, pop on the burst
    assert abs(powerup.core_scale(0.99) - 1) < 0.05  # settled
    assert powerup.glow_boost(0.3) > powerup.glow_boost(0.9) and powerup.text_alpha(0.1) == 0


def test_orb_runs_and_ends_the_animation(qt):
    from PyQt6.QtGui import QColor, QPainter, QPixmap

    from jarvis.ui import powerup
    from jarvis.ui.orb import Orb

    o = Orb("#00d4ff")
    o.resize(320, 380)
    o.transform("love", 5)
    o.power_up()
    assert o.power and o.cute_until == 0  # the cute face steps aside
    for t in (0.1, 0.3, 0.31, 0.5, 0.9):  # every phase paints without errors
        o.power["t"] = t
        pm = QPixmap(o.size())
        pm.fill(QColor("black"))
        p = QPainter(pm)
        o.render(p)
        p.end()
    o.power["t"] = 0.999
    o._last = time.monotonic() - 0.05
    o._tick()
    assert o.power is None
    assert powerup.DURATION < 2.5  # short enough not to get in the way


def test_window_uses_power_up_on_wake(qt, isolated_home, sandbox, monkeypatch):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    app = create_app(Settings(allowed_roots=[str(sandbox)]))
    w = MainWindow(app, Bridge())
    w._on_state("sleeping")
    w._on_state("listening")
    assert w.big_orb.power is not None
    app.settings.power_up_wake = False
    w.big_orb.power = None
    w._on_state("sleeping")
    w._on_state("listening")
    assert w.big_orb.power is None and w.big_orb.flash_t > 0
    w.reminders.close_all()


def test_power_up_sound():
    from jarvis.sfx import SoundFX, build_sounds

    snd = build_sounds()["powerup"]
    assert 0.8 < len(snd) / 24000 < 2.0
    fx = SoundFX(enabled=False)
    played = []
    fx.play = played.append
    fx.on_event("state", {"state": "sleeping"})
    fx._started = True
    fx.on_event("state", {"state": "listening"})
    fx.power_up = False
    fx.on_event("state", {"state": "sleeping"})
    fx.on_event("state", {"state": "listening"})
    assert "powerup" in played and played[-1] == "wake"
