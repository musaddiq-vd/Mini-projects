"""Day / night mode and themes."""

import sys
from datetime import datetime

import pytest

from jarvis.ui import theme


def test_auto_mode_follows_the_clock():
    assert theme.resolve_mode("auto", datetime(2026, 1, 1, 10)) == "light"
    assert theme.resolve_mode("auto", datetime(2026, 1, 1, 22)) == "dark"
    assert theme.resolve_mode("light") == "light" and theme.resolve_mode("weird") == "dark"


def test_theme_accent_or_custom():
    assert theme.accent_for("Sunset", "#00d4ff") == "#ff8a3d"  # default accent -> the theme's own
    assert theme.accent_for("Sunset", "#123456") == "#123456"  # a real custom colour wins
    assert theme.accent_for("Nope", "") == theme.THEMES["Jarvis Blue"][0]


@pytest.mark.parametrize("mode", ["dark", "light"])
@pytest.mark.parametrize("name", list(theme.THEMES))
def test_every_theme_builds(mode, name):
    css = theme.apply(name, mode, theme.accent_for(name))
    assert "QFrame#panel" in css and theme.P.light == (mode == "light")
    assert theme.TEXT == (theme.LIGHT.text if mode == "light" else theme.DARK.text)
    theme.apply("Jarvis Blue", "dark", "#00d4ff")


def test_readable_accent_is_darker_on_light():
    theme.apply("Jarvis Blue", "light", "#00d4ff")
    assert theme.readable("#00d4ff") != "#00d4ff"
    theme.apply("Jarvis Blue", "dark", "#00d4ff")
    assert theme.readable("#00d4ff") == "#00d4ff"


def test_window_cycles_day_night(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings, load_settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)  # keep a reference or Qt crashes
    app = create_app(Settings())
    w = MainWindow(app, Bridge())
    assert qt is not None
    w.cycle_theme_mode()
    assert app.settings.theme_mode == "light" and theme.P.light and w.mode_btn.text() == "☀"
    assert load_settings().theme_mode == "light"  # remembered
    w.cycle_theme_mode()
    assert app.settings.theme_mode == "auto"
    w.cycle_theme_mode()
    assert app.settings.theme_mode == "dark" and not theme.P.light
