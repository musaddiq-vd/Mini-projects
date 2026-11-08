"""The weekly report."""

import sys
import time
from datetime import date, datetime, timedelta

import pytest


def seed(app, today: date):
    """A week of life: money this week and last, a habit, a goal step, tasks, focus."""
    from jarvis.tools.goals import goals
    from jarvis.tools.habits import habits
    from jarvis.tools.money import expenses

    monday = today - timedelta(days=today.weekday())
    d = lambda n: (monday + timedelta(days=n)).isoformat()  # noqa: E731
    expenses().save({"items": [
        {"id": 1, "amount": 300, "category": "food", "note": "", "date": d(0)},
        {"id": 2, "amount": 200, "category": "travel", "note": "", "date": d(0)},
        {"id": 3, "amount": 1000, "category": "food", "note": "", "date": (monday - timedelta(days=7)).isoformat()},
    ], "budgets": {}, "next_id": 4})
    habits().save({"habits": [{"id": 1, "name": "Gym", "target": 1, "days": "daily", "icon": "💪",
                               "log": {d(i): 1 for i in range(today.weekday() + 1)},
                               "created_day": (monday - timedelta(days=30)).isoformat()}], "next_id": 2})
    noon = datetime.combine(monday, datetime.min.time()).replace(hour=12).timestamp()
    goals().save({"goals": [{"id": 1, "title": "Learn Python", "deadline": d(60), "why": "", "done": False,
                             "steps": [{"text": "Basics", "due": d(1), "done": True, "done_at": noon},
                                       {"text": "Project", "due": d(9), "done": False}]}], "next_id": 2})
    app.state.set("notes", [{"text": "Milk", "todo": True, "done": True, "done_at": noon}])
    app.state.set("focus_log", [{"date": d(0), "minutes": 150}])
    app.state.set("reminders_done", [{"message": "Call", "at": noon}])


@pytest.fixture
def app(isolated_home, sandbox):
    from jarvis.app import create_app
    from jarvis.config import Settings

    return create_app(Settings(allowed_roots=[str(sandbox)]))


def test_week_bounds():
    from jarvis.tools.report import week_bounds

    wed = date(2026, 11, 11)
    assert week_bounds("this", wed) == (date(2026, 11, 9), date(2026, 11, 15))
    assert week_bounds("last", wed) == (date(2026, 11, 2), date(2026, 11, 8))


def test_report_numbers(app):
    from jarvis.tools.report import build

    today = date.today()
    seed(app, today)
    r = build(app.ctx, "this", today)
    assert r["spent"] == 500 and r["top_categories"][0] == ("food", 300)
    assert r["spent_change_percent"] == -50  # 1000 on the same days last week
    assert r["habits_percent"] == 100 and r["best_habit"] == "Gym"
    assert r["goal_steps_done"] == 1 and r["goals"] == [{"title": "Learn Python", "percent": 50}]
    assert r["tasks_done"] == 2 and r["focus"] == "2h 30m" and r["meetings_next_week"] is None
    assert "habits 100% done" in r["wins"] and "spent 50% less than last week" in r["wins"]


def test_empty_week_and_last_week(app):
    from jarvis.tools.report import build

    r = build(app.ctx, "last")
    assert r["spent"] == 0 and r["habits_percent"] is None and r["wins"] == [] and r["which"] == "last"


def test_tool_and_card(app, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QLabel

    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    seed(app, date.today())
    w = MainWindow(app, Bridge())
    out = app.registry.call("weekly_report", {}, app.ctx)
    assert "suggestion" in out["result"]
    app.events.emit("tool", name="weekly_report", args={}, result=out)
    qt.processEvents()
    texts = " ".join(lb.text() for lb in w.chat.findChildren(QLabel))
    assert "Your week" in texts and "₹500" in texts and "Learn Python" in texts
    w.reminders.close_all()


def test_sunday_evening_once(app):
    from jarvis.background import BackgroundServices

    bg = BackgroundServices(app, lambda t: None)
    sunday = datetime(2026, 11, 15, 19, 0)
    assert bg.weekly_due(sunday) and not bg.weekly_due(sunday.replace(hour=12))
    assert not bg.weekly_due(datetime(2026, 11, 14, 19, 0))  # Saturday
    app.state.set("last_weekly_report", "2026-W46")
    assert not bg.weekly_due(sunday)
    app.state.set("last_weekly_report", "2026-W45")
    bg.sleeping = True
    assert not bg.weekly_due(sunday)
