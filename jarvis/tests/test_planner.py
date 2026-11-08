"""Planner panel and the connection pill."""

import sys
import time

import pytest

from jarvis.ui.planner import parse_when


@pytest.mark.parametrize("text, expect", [
    ("20 min", {"minutes": 20.0}), ("in 5 minutes", {"minutes": 5.0}), ("1 hour", {"minutes": 60.0}),
    ("2h", {"minutes": 120.0}), ("6:30 pm", {"at": "6:30 pm"}), ("18:30", {"at": "18:30"}),
    ("tomorrow", None), ("", None),
])
def test_parse_when(text, expect):
    assert parse_when(text) == expect


@pytest.fixture
def window(monkeypatch, isolated_home):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    import jarvis.tools.reminders as rem
    import jarvis.tools.timers as tim
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings())
    w = MainWindow(app, Bridge())
    w.show()
    w.qt = qt
    yield w, app
    for t in list(rem._timers.values()):
        t.cancel()
    rem._timers.clear()
    for info in list(tim._timers.values()):
        info["timer"].cancel()
    tim._timers.clear()


def labels(widget):
    from PyQt6.QtWidgets import QApplication, QCheckBox, QLabel

    QApplication.processEvents()
    return " | ".join(w.text() for w in widget.findChildren((QLabel, QCheckBox)) if w.isVisibleTo(widget))


def test_planner_adds_ticks_and_deletes_todos(window):
    from PyQt6.QtWidgets import QCheckBox

    w, app = window
    w.toggle_planner()
    p = w.planner
    p.todo_input.setText("Buy milk & bread")
    p._add_todo()
    assert "Buy milk && bread" in labels(p.todo)  # shown with the '&' intact
    box = next(b for b in p.todo.findChildren(QCheckBox) if "milk" in b.text())
    box.setChecked(True)
    assert app.state.get("notes")[0]["done"] is True
    p.done_toggle.setChecked(True)
    assert "milk" in labels(p.todo)


def test_planner_reminders_timers_routines(window):
    w, app = window
    w.toggle_planner()
    p = w.planner
    p.rem_text.setText("Call mom")
    p.rem_when.setText("20 min")
    p._add_reminder()
    assert "Call mom" in labels(p.rem) and app.state.get("reminders")[0]["message"] == "Call mom"
    p.rem_text.setText("bad")
    p.rem_when.setText("someday")
    p._add_reminder()
    assert not p.rem_error.isHidden()
    p._call("set_timer", {"minutes": 5, "label": "tea"})
    w.qt.processEvents()
    assert 1 <= len(w.big_orb.timers) and "tea" in labels(p.tim)
    app.registry.call("save_routine", {"name": "work mode", "steps": ["open VS Code"]}, app.ctx)
    p.refresh()
    assert "work mode" in labels(p.rou)


def test_planner_never_saves_secrets(window):
    w, app = window
    w.toggle_planner()
    w.planner.todo_input.setText("card 4111 1111 1111 1111")
    w.planner._add_todo()
    assert not app.state.get("notes")  # guardrails apply here too


def test_planner_opens_with_ctrl_p_and_closes_with_esc(window):
    w, app = window
    w.toggle_planner()
    assert w.planner.isVisible()
    w._escape()
    assert not w.planner.isVisible()


def test_connection_pill(window):
    w, app = window
    app.events.emit("state", state="listening")
    assert w.net_pill.text().startswith("● Online")
    app.events.emit("log", text="[connection lost] boom - retrying in 1s")
    assert "Reconnecting" in w.net_pill.text()
    app.events.emit("state", state="offline")
    assert "Offline" in w.net_pill.text()


def test_timers_show_in_planner_countdown(window):
    w, app = window
    w.big_orb.set_timer(9, time.time() + 90, 120, "eggs")
    w.toggle_planner()
    assert "eggs" in labels(w.planner.tim) and "1:3" in labels(w.planner.tim)
