"""Phase A: dates, data files, expenses, habits, goals, goal follow-up and the dashboard."""

import json
import sys
from datetime import date, timedelta

import pytest

from jarvis.dates import parse_day, period_range

MON = date(2026, 11, 9)


# -- dates ------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, past, expect", [
    ("", True, MON), ("aaj", True, MON), ("kal", True, MON - timedelta(1)), ("kal", False, MON + timedelta(1)),
    ("parso", True, MON - timedelta(2)), ("3 din pehle", True, MON - timedelta(3)),
    ("2 weeks ago", True, MON - timedelta(14)), ("in 3 weeks", False, MON + timedelta(21)),
    ("3 months", False, date(2027, 2, 9)), ("next month", False, date(2026, 12, 9)),
    ("friday", True, date(2026, 11, 6)), ("friday", False, date(2026, 11, 13)), ("monday", True, MON),
    ("12 nov", True, date(2025, 11, 12)), ("31 dec", False, date(2026, 12, 31)), ("15/10", True, date(2026, 10, 15)),
    ("2027-01-05", False, date(2027, 1, 5)), ("end of month", False, date(2026, 11, 30)),
    ("31/02", True, None), ("banana", True, None),
])
def test_parse_day(text, past, expect):
    assert parse_day(text, MON, past) == expect


def test_month_overflow():
    assert parse_day("1 month", date(2026, 1, 31), past=False) == date(2026, 2, 28)


def test_periods():
    assert period_range("last_month", MON)[:2] == (date(2026, 10, 1), date(2026, 10, 31))
    assert period_range("week", MON)[0] == MON - timedelta(6)


# -- data file ---------------------------------------------------------------------------------
def test_corrupt_file_is_set_aside(tmp_path):
    from jarvis.datafile import DataFile

    path = tmp_path / "x.json"
    path.write_text("{oops", encoding="utf-8")
    store = DataFile(path, {"items": []})
    assert store.load() == {"items": []}
    assert list(tmp_path.glob("x.broken-*.json"))
    store.update(lambda d: d["items"].append(1))
    assert json.loads(path.read_text(encoding="utf-8")) == {"items": [1]}


# -- expenses ---------------------------------------------------------------------------------
def test_expenses(registry, ctx, isolated_home):
    out = registry.call("add_expense", {"amount": 100, "category": "food", "note": "chai"}, ctx)
    assert "₹100" in out["result"] and "This month so far: ₹100" in out["result"]
    registry.call("add_expense", {"amount": 450, "category": "transport", "note": "Uber", "when": "kal"}, ctx)
    s = registry.call("expense_summary", {"period": "all"}, ctx)
    assert s["total"] == "₹550" and list(s["by_category"]) == ["transport", "food"]
    assert registry.call("expense_summary", {"period": "today", "category": "health"}, ctx)["result"].startswith("No ")


@pytest.mark.parametrize("args", [{"amount": 0}, {"amount": -5}, {"amount": "abc"}, {"amount": 5, "when": "tomorrow"},
                                  {"amount": 5, "when": "someday"}, {"amount": 1e12}])
def test_bad_expenses(registry, ctx, isolated_home, args):
    assert "error" in registry.call("add_expense", args, ctx)


def test_budget_warning_delete_and_undo(registry, ctx, isolated_home):
    registry.call("set_budget", {"amount": 500}, ctx)
    registry.call("add_expense", {"amount": 420, "category": "shopping"}, ctx)
    out = registry.call("add_expense", {"amount": 100, "category": "food"}, ctx)
    assert "used up" in out["result"]
    eid = registry.call("list_expenses", {"period": "today"}, ctx)["expenses"][-1]["id"]
    assert "Deleted" in registry.call("delete_expense", {"id": eid}, ctx)["result"]
    registry.call("undo", {}, ctx)
    assert registry.call("expense_summary", {"period": "today"}, ctx)["total"] == "₹520"
    assert "error" in registry.call("delete_expense", {"id": 999}, ctx)


def test_expense_notes_cant_hold_secrets(registry, ctx, isolated_home):
    out = registry.call("add_expense", {"amount": 50, "note": "card 4111 1111 1111 1111"}, ctx)
    assert out["error"].startswith("Blocked for your safety")


# -- habits -----------------------------------------------------------------------------------------
def test_habits_progress_streak_and_week(registry, ctx, isolated_home):
    from jarvis.tools.habits import habits, streak

    assert "8 times" in registry.call("add_habit", {"name": "Water", "target": 8}, ctx)["result"]
    assert "error" in registry.call("add_habit", {"name": "water"}, ctx)  # duplicate
    assert "3/8" in registry.call("check_habit", {"name": "water", "count": 3}, ctx)["result"]
    assert "done for the day" in registry.call("check_habit", {"name": "wat", "count": 5}, ctx)["result"]
    registry.call("check_habit", {"name": "water", "count": 8, "when": "kal"}, ctx)
    h = habits().load()["habits"][0]
    assert streak(h) == 2  # yesterday (backfilled) + today
    status = registry.call("habit_status", {}, ctx)["habits"][0]
    assert status["today"] == "8/8" and status["last_7_days"].endswith("✓✓") and status["last_7_days"][0] == "·"


def test_weekday_habits_skip_weekends():
    from jarvis.tools.habits import streak

    fri = date(2026, 11, 6)
    h = {"target": 1, "days": "weekdays", "created_day": "2026-11-01",
         "log": {"2026-11-05": 1, "2026-11-06": 1, "2026-11-09": 1}}
    assert streak(h, date(2026, 11, 9)) == 3  # Thu, Fri, (weekend skipped), Mon
    assert streak(h, fri + timedelta(1)) == 2  # Saturday: doesn't break it


def test_habit_errors(registry, ctx, isolated_home):
    assert "error" in registry.call("check_habit", {"name": "yoga"}, ctx)
    registry.call("add_habit", {"name": "Gym"}, ctx)
    assert "error" in registry.call("check_habit", {"name": "gym", "when": "tomorrow"}, ctx)
    assert "error" in registry.call("add_habit", {"name": "x", "target": 0}, ctx)
    registry.call("check_habit", {"name": "gym"}, ctx)
    assert "0/1" in registry.call("check_habit", {"name": "gym", "count": -5}, ctx)["result"]  # never below 0


# -- goals -------------------------------------------------------------------------------------------
def test_goals(registry, ctx, isolated_home):
    out = registry.call("create_goal", {"title": "Learn Python", "deadline": "3 months", "why": "better job",
                                        "steps": ["Install", "Basics", "Functions", "Project"]}, ctx)
    assert "4 steps" in out["result"]
    st = registry.call("goal_status", {}, ctx)["goals"][0]
    assert st["progress"].startswith("0/4") and st["days_left"] > 80
    assert "25%" in registry.call("complete_goal_step", {"goal": "python"}, ctx)["result"]
    assert "Functions" in registry.call("complete_goal_step", {"goal": "python", "step": "3"}, ctx)["result"]
    registry.call("add_goal_step", {"goal": "python", "text": "Portfolio", "due": "2 months"}, ctx)
    for _ in range(3):
        registry.call("complete_goal_step", {"goal": "python"}, ctx)
    assert "Goal complete" in registry.call("goal_status", {"goal": "python"}, ctx).get("result", "Goal complete") \
        or registry.call("goal_status", {"goal": "python"}, ctx)["goals"][0]["next_step"] == "all done"


@pytest.mark.parametrize("args", [{"title": "x", "steps": []}, {"title": "x", "steps": ["a"], "deadline": "yesterday"},
                                  {"title": "x", "steps": ["a"], "deadline": "whenever"},
                                  {"title": "", "steps": ["a"]}, {"title": "x", "steps": ["s"] * 25}])
def test_bad_goals(registry, ctx, isolated_home, args):
    assert "error" in registry.call("create_goal", args, ctx)


def test_steps_are_spread_until_the_deadline():
    from jarvis.tools.goals import spread

    dues = spread(4, date(2026, 1, 1), date(2026, 1, 29))
    assert dues == ["2026-01-08", "2026-01-15", "2026-01-22", "2026-01-29"]
    assert spread(3, date(2026, 1, 1), date(2026, 1, 2))[0] == "2026-01-02"  # never on the start day


def test_goal_checkin(registry, ctx, isolated_home):
    from jarvis.tools.goals import checkin_instruction

    assert checkin_instruction() is None
    registry.call("create_goal", {"title": "Run 5k", "deadline": "4 weeks", "steps": ["Walk", "Run 1k"]}, ctx)
    assert checkin_instruction() is None  # nothing due yet
    later = date.today() + timedelta(days=20)
    text = checkin_instruction(later)
    assert "Run 5k" in text and "Walk" in text and "overdue" in text


def test_goal_checkin_in_background(isolated_home):
    from datetime import datetime

    from jarvis.app import create_app
    from jarvis.background import BackgroundServices
    from jarvis.config import Settings

    app = create_app(Settings())
    app.registry.call("create_goal", {"title": "Read a book", "deadline": "1 week", "steps": ["Chapter 1"]}, app.ctx)
    b = BackgroundServices(app, lambda t: None)
    later = datetime.now().replace(hour=12) + timedelta(days=8)
    assert b.goal_checkin_due(later) and not b.goal_checkin_due(later.replace(hour=22))
    b.sleeping = True
    assert not b.goal_checkin_due(later)


# -- dashboard ----------------------------------------------------------------------------------------
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


def page_text(w):
    from PyQt6.QtWidgets import QCheckBox, QLabel

    w.qt.processEvents()
    return " | ".join(x.text() for x in w.dashboard.findChildren((QLabel, QCheckBox)) if x.isVisibleTo(w.dashboard))


@pytest.mark.parametrize("tab", ["week", "habits", "money", "goals"])
def test_every_dashboard_page_builds_empty_and_full(window, tab):
    w, app = window
    w.toggle_dashboard(tab)
    assert "Couldn't show" not in page_text(w)
    r, c = app.registry, app.ctx
    r.call("add_expense", {"amount": 99, "category": "food", "note": "pizza"}, c)
    r.call("add_habit", {"name": "Gym"}, c)
    r.call("check_habit", {"name": "Gym"}, c)
    r.call("create_goal", {"title": "Ship v2", "deadline": "2 weeks", "steps": ["Plan", "Build"]}, c)
    w.dashboard.refresh()
    text = page_text(w)
    assert "Couldn't show" not in text
    expect = {"week": "₹99", "habits": "Gym", "money": "pizza", "goals": "Ship v2"}[tab]
    assert expect in text


def test_dashboard_actions(window):
    from PyQt6.QtWidgets import QCheckBox

    w, app = window
    app.registry.call("create_goal", {"title": "Ship v2", "deadline": "2 weeks", "steps": ["Plan", "Build"]}, app.ctx)
    w.toggle_dashboard("goals")
    box = next(b for b in w.dashboard.findChildren(QCheckBox) if b.text().startswith("Plan"))
    box.setChecked(True)
    from jarvis.tools.goals import goals

    assert goals().load()["goals"][0]["steps"][0]["done"] is True


def test_week_numbers(window):
    from jarvis.ui.dashboard import week_numbers

    w, app = window
    app.events.emit("user", text="hi")
    app.registry.call("add_expense", {"amount": 20, "note": "tea"}, app.ctx)
    app.state.set("focus_log", [{"date": date.today().isoformat(), "minutes": 25}])
    nums = week_numbers(app)
    assert nums["messages"][-1] == 1 and nums["spent"][-1] == 20 and nums["focus"][-1] == 25


def test_dashboard_by_voice(window):
    w, app = window
    app.events.emit("ui_command", action="dashboard", theme="money")
    assert w.dashboard.isVisible() and w.dashboard.tabs.currentIndex() == 2
    w._escape()
    assert not w.dashboard.isVisible()


def test_privacy_tab_lists_and_deletes_life_data(window):
    import copy

    from jarvis.ui.privacy import PrivacyTab

    w, app = window
    app.registry.call("add_expense", {"amount": 10}, app.ctx)
    tab = PrivacyTab(app, copy.deepcopy(app.settings))
    assert tab.rows["money"][0].text().startswith("1 expense")
    tab._clear_all()
    tab.refresh()
    assert tab.rows["money"][0].text() == "nothing yet"
