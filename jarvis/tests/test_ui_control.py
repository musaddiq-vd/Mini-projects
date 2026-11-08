"""Voice control of Jarvis's own window."""

import sys

import pytest

from jarvis.events import Events


def test_tool_sends_a_ui_command(registry, ctx):
    ctx.events = Events()
    seen = []
    ctx.events.subscribe(lambda k, d: seen.append((d["action"], d["theme"])) if k == "ui_command" else None)
    assert registry.call("jarvis_display", {"action": "night_mode"}, ctx)["result"] == "Night mode is on."
    assert registry.call("jarvis_display", {"action": "theme", "theme": "Sunset"}, ctx)["result"] == "Sunset theme is on."
    assert seen == [("night_mode", ""), ("theme", "Sunset")]


def test_tool_rejects_bad_input(registry, ctx):
    ctx.events = Events()
    assert "error" in registry.call("jarvis_display", {"action": "explode"}, ctx)
    assert "error" in registry.call("jarvis_display", {"action": "theme", "theme": "Rainbow"}, ctx)
    ctx.events = None
    assert "error" in registry.call("jarvis_display", {"action": "show_help"}, ctx)


def test_prompt_mentions_it(ctx):
    from jarvis.app import system_prompt

    assert "jarvis_display" in system_prompt(ctx.settings, ctx.memory)


@pytest.fixture
def window(monkeypatch, isolated_home):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings())
    w = MainWindow(app, Bridge())
    w.show()
    w.qt = qt
    return w, app


def test_window_obeys(window):
    from jarvis.config import load_settings
    from jarvis.ui import theme

    w, app = window
    app.events.emit("ui_command", action="day_mode", theme="")
    assert app.settings.theme_mode == "light" and theme.P.light and load_settings().theme_mode == "light"
    app.events.emit("ui_command", action="theme", theme="Emerald")
    assert app.settings.theme_name == "Emerald" and w.accent == theme.THEMES["Emerald"][0]
    app.events.emit("ui_command", action="show_help", theme="")
    assert w.help.isVisible()
    app.events.emit("ui_command", action="show_achievements", theme="")
    assert w.badges.isVisible()
    app.events.emit("ui_command", action="mini_mode", theme="")
    assert w.mini.isVisible() and not w.isVisible()
    app.events.emit("ui_command", action="full_window", theme="")
    assert w.isVisible() and not w.mini.isVisible()
    app.events.emit("ui_command", action="night_mode", theme="")
    assert not theme.P.light
    w._leave_mini()
