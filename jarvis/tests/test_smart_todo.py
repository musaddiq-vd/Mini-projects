"""Smart to-do: deadlines, priority, what to do first, and nudges."""

import sys
from datetime import date, datetime, timedelta

import pytest

from jarvis.events import Events
from jarvis.state import StateStore


@pytest.fixture
def tctx(ctx, tmp_path):
    ctx.state = StateStore(tmp_path / "state.json")
    ctx.events = Events()
    return ctx


def test_parse_due_and_labels():
    from jarvis.tools.notes import due_label, parse_due

    today = date(2026, 11, 11)  # a Wednesday
    assert parse_due("kal tak", today) == "2026-11-12" and parse_due("friday", today) == "2026-11-13"
    assert parse_due("12 dec", today) == "2026-12-12" and parse_due("", today) == "" and parse_due("hatao") == ""
    assert parse_due("blue moon", today) is None
    assert due_label("2026-11-11", today) == "today" and due_label("2026-11-12", today) == "tomorrow"
    assert due_label("2026-11-09", today) == "overdue 2 days" and due_label("2026-11-10", today) == "overdue 1 day"
    assert due_label("2026-11-14", today) == "Sat" and due_label("2026-12-25", today) == "25 Dec"


def test_rank_order():
    from jarvis.tools.notes import rank

    today = date(2026, 11, 11)
    items = [
        {"id": 1, "text": "someday", "todo": True, "created": 1},
        {"id": 2, "text": "low", "todo": True, "priority": "low", "created": 2},
        {"id": 3, "text": "urgent", "todo": True, "priority": "high", "created": 3},
        {"id": 4, "text": "today", "todo": True, "due": "2026-11-11", "created": 4},
        {"id": 5, "text": "late", "todo": True, "due": "2026-11-08", "created": 5},
        {"id": 6, "text": "soon", "todo": True, "due": "2026-11-12", "created": 6},
        {"id": 7, "text": "done", "todo": True, "done": True, "due": "2026-11-01"},
        {"id": 8, "text": "a note", "todo": False},
    ]
    ranked = rank(items, today)
    assert [n["text"] for n, _ in ranked] == ["late", "today", "urgent", "soon", "someday", "low"]
    assert ranked[0][1] == "overdue by 3 days" and ranked[1][1] == "due today"


def test_add_update_and_list(registry, tctx):
    out = registry.call("add_note", {"text": "Pay electricity bill", "due": "kal tak", "priority": "high"}, tctx)
    assert "due tomorrow" in out["result"] and "high priority" in out["result"]
    saved = tctx.state.get("notes")[0]
    assert saved["due"] == (date.today() + timedelta(days=1)).isoformat() and saved["priority"] == "high"
    assert "error" in registry.call("add_note", {"text": "x", "due": "blue moon"}, tctx)
    registry.call("add_note", {"text": "Buy milk"}, tctx)
    assert "priority" not in tctx.state.get("notes")[1] and "due" not in tctx.state.get("notes")[1]
    registry.call("update_todo", {"item": "milk", "due": "aaj", "priority": "low"}, tctx)
    milk = tctx.state.get("notes")[1]
    assert milk["due"] == date.today().isoformat() and milk["priority"] == "low"
    registry.call("update_todo", {"item": "milk", "due": "none", "priority": "normal"}, tctx)
    assert "due" not in tctx.state.get("notes")[1] and "priority" not in tctx.state.get("notes")[1]
    assert "error" in registry.call("update_todo", {"item": "milk"}, tctx)
    assert "error" in registry.call("update_todo", {"item": "zebra", "priority": "high"}, tctx)
    rows = registry.call("list_notes", {}, tctx)["items"]
    assert rows[0]["text"] == "Pay electricity bill" and rows[0]["due"] == "tomorrow"
    registry.call("undo", {}, tctx)  # the last change can be undone
    assert tctx.state.get("notes")[1]["priority"] == "low"


def test_what_first(registry, tctx):
    assert "empty" in registry.call("what_first", {}, tctx)["result"]
    registry.call("add_note", {"text": "Gym"}, tctx)
    registry.call("add_note", {"text": "Submit report", "due": "aaj"}, tctx)
    out = registry.call("what_first", {}, tctx)
    assert out["do_first"][0]["text"] == "Submit report" and out["do_first"][0]["why"] == "due today"
    assert out["due_today"] == 1 and out["open"] == 2


def test_background_nudges_once(isolated_home, sandbox):
    from jarvis.app import create_app
    from jarvis.background import BackgroundServices
    from jarvis.config import Settings

    app = create_app(Settings(allowed_roots=[str(sandbox)]))
    today = date.today()
    app.state.set("notes", [{"id": 1, "text": "Submit report", "todo": True, "due": today.isoformat(),
                             "priority": "high"},
                            {"id": 2, "text": "Old task", "todo": True, "due": (today - timedelta(days=2)).isoformat()}])
    bg = BackgroundServices(app, lambda t: None)
    at = lambda h: datetime.combine(today, datetime.min.time()).replace(hour=h)  # noqa: E731
    assert bg.todo_nudge(at(9)) is None
    first = bg.todo_nudge(at(10))
    assert "overdue" in first and "Submit report" in first
    assert bg.todo_nudge(at(11)) is None  # once in the morning
    later = bg.todo_nudge(at(16))
    assert "Submit report" in later and "still open" in later
    assert bg.todo_nudge(at(17)) is None


def test_planner_shows_deadlines(isolated_home, sandbox, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings(allowed_roots=[str(sandbox)]))
    w = MainWindow(app, Bridge())
    app.registry.call("add_note", {"text": "Later thing"}, app.ctx)
    app.registry.call("add_note", {"text": "Due now", "due": "aaj"}, app.ctx)
    w.toggle_planner()
    qt.processEvents()
    labels = [lb.text() for lb in w.planner.todo.findChildren(QLabel)]
    assert "today" in labels
    boxes = [b.text() for b in w.planner.todo.findChildren(QPushButton) if b.text() in ("⚑", "🚩")]
    assert boxes == ["⚑", "⚑"]
    w.reminders.close_all()
