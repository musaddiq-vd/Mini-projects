"""The desktop widget on the wallpaper."""

import sys
import time
from datetime import date

import pytest


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def app(isolated_home, sandbox):
    from jarvis.app import create_app
    from jarvis.config import Settings

    return create_app(Settings(allowed_roots=[str(sandbox)]))


def test_lines_show_whats_next(app):
    from jarvis.tools.habits import habits
    from jarvis.ui.desktop_widget import lines_for

    assert lines_for(app) == ["✨  Nothing planned - enjoy your day"]
    app.state.set("reminders", [{"id": 1, "due": time.time() + 600, "message": "Tea", "repeat": ""}])
    app.state.set("notes", [{"text": "Milk", "todo": True, "done": False}, {"text": "x", "todo": True, "done": True}])
    habits().save({"habits": [{"id": 1, "name": "Gym", "target": 1, "days": "daily", "icon": "💪",
                               "log": {date.today().isoformat(): 1}, "created_day": "2026-01-01"}], "next_id": 2})
    lines = lines_for(app)
    assert lines[0].startswith("⏰") and "Tea" in lines[0]
    assert lines[1] == "📝  1 to-do · Milk" and lines[2] == "✅  1/1 habits today  🎉"


def test_widget_never_takes_clicks_or_focus(qt, app):
    from PyQt6.QtCore import Qt

    from jarvis.ui.desktop_widget import DesktopWidget

    d = DesktopWidget(app, "#00d4ff")
    flags = d.windowFlags()
    assert flags & Qt.WindowType.WindowTransparentForInput and flags & Qt.WindowType.WindowStaysOnBottomHint
    assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
    d.show_on_desktop()
    area = d.screen().availableGeometry()
    assert d.x() + d.width() <= area.right() + 1 and d.y() >= area.top()
    d.place("bottom_left")
    assert d.x() == area.left() + 28 and d.y() + d.height() <= area.bottom() + 1
    d.place("nowhere")  # unknown corner: stays where it was
    assert d.corner == "bottom_left"
    assert d.clock.text() and d.date.text().isupper()
    d.hide_from_desktop()
    assert not d.isVisible() and not d.orb._timer.isActive()
    d.close()


def test_voice_and_settings_turn_it_on_and_off(qt, app):
    from jarvis.config import load_settings
    from jarvis.ui.window import Bridge, MainWindow

    w = MainWindow(app, Bridge())
    app.registry.call("jarvis_display", {"action": "widget_on", "place": "bottom_right"}, app.ctx)
    assert w.desk.isVisible() and w.desk.corner == "bottom_right"
    assert load_settings().desktop_widget is True and load_settings().widget_corner == "bottom_right"
    assert w.desk.orb in w.orb.orbs  # the widget's orb follows Jarvis' state
    app.events.emit("state", state="speaking")
    qt.processEvents()
    assert w.desk.orb.state == "speaking"
    w._toggle_desktop_widget()
    assert not w.desk.isVisible() and load_settings().desktop_widget is False
    w.desk.close()
    w.reminders.close_all()


def test_invite_titles_are_plain_text(qt, app):
    from PyQt6.QtCore import Qt

    from jarvis.ui.desktop_widget import DesktopWidget
    from jarvis.ui.reminder_popup import ReminderPopup

    d = DesktopWidget(app, "#00d4ff")
    assert all(lb.textFormat() == Qt.TextFormat.PlainText for lb in d.info)
    p = ReminderPopup("due", 1, "<b>hi</b><img src='http://x/y.png'>", time.time(), "#00d4ff")
    assert p.text.textFormat() == Qt.TextFormat.PlainText
    d.close()
    p.close()
