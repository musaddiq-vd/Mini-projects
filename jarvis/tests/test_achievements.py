"""Achievements, confetti and the gallery."""

import sys
from datetime import datetime, timedelta

import pytest

from jarvis.achievements import BADGES, BY_ID, Achievements
from jarvis.events import Events
from jarvis.state import StateStore


@pytest.fixture
def ach(tmp_path):
    clock = {"now": datetime(2026, 11, 9, 14, 0)}
    events = Events()
    unlocked = []
    events.subscribe(lambda k, d: unlocked.append(d["id"]) if k == "achievement" else None)
    a = Achievements(StateStore(tmp_path / "state.json"), events, now=lambda: clock["now"])
    a.clock, a.seen, a.ev = clock, unlocked, events
    return a


def test_first_words_and_first_action(ach):
    ach.ev.emit("user", text="hello")
    ach.ev.emit("tool", name="get_time", args={}, result={"result": "3 pm"})
    assert ach.seen == ["hello", "first_action"]
    ach.ev.emit("user", text="again")
    assert ach.seen == ["hello", "first_action"]  # each badge only once


def test_failed_tools_and_system_messages_dont_count(ach):
    ach.ev.emit("tool", name="get_time", args={}, result={"error": "x"})
    ach.ev.emit("user", text="(system: asleep) ...")
    assert ach.seen == []


def test_counters_reach_goals(ach):
    for _ in range(10):
        ach.ev.emit("tool", name="add_note", args={}, result={"result": "ok"})
    for _ in range(10):
        ach.ev.emit("timer_end", id=1, label="tea", done=True)
    ach.ev.emit("timer_end", id=2, label="Focus: maths", done=True)
    ach.ev.emit("timer_end", id=3, label="tea", done=False)  # cancelled: no credit
    assert {"list_maker", "timer_master", "focused"} <= set(ach.seen)


def test_explorer_counts_different_abilities(ach):
    for i in range(14):
        ach.ev.emit("tool", name=f"tool_{i}", args={}, result={"result": "ok"})
    assert "explorer" not in ach.seen
    ach.ev.emit("tool", name="tool_0", args={}, result={"result": "ok"})  # same one again
    assert "explorer" not in ach.seen
    ach.ev.emit("tool", name="tool_14", args={}, result={"result": "ok"})
    assert "explorer" in ach.seen


def test_time_of_day_badges(ach):
    ach.clock["now"] = datetime(2026, 11, 9, 5, 30)
    ach.ev.emit("user", text="good morning")
    ach.clock["now"] = datetime(2026, 11, 10, 1, 15)
    ach.ev.emit("user", text="can't sleep")
    assert "early_bird" in ach.seen and "night_owl" in ach.seen


def test_streaks(ach):
    start = datetime(2026, 11, 1, 12)
    for d in range(7):
        ach.clock["now"] = start + timedelta(days=d)
        ach.ev.emit("user", text="hi")
    assert "streak3" in ach.seen and "streak7" in ach.seen


def test_thanks_jokes_guardrail_and_perfect_practice(ach):
    for _ in range(10):
        ach.ev.emit("user", text="thank you!")
    for _ in range(5):
        ach.ev.emit("user", text="tell me a joke")
    ach.ev.emit("guardrail", action="blocked", tool="type_text", text="x")
    ach.ev.emit("practice_progress", task_id="d1-vocab", score=5.0, total=5.0)
    assert {"polite", "comedian", "safe", "scholar"} <= set(ach.seen)


def test_progress_list_and_persistence(ach, tmp_path):
    ach.ev.emit("user", text="hello")
    rows = ach.progress()
    assert len(rows) == len(BADGES)
    hello = next(r for r in rows if r[0].id == "hello")
    assert hello[2] is not None
    again = Achievements(StateStore(tmp_path / "state.json"), Events())
    assert "hello" in again.unlocked()


def test_badge_ids_unique():
    assert len(BY_ID) == len(BADGES)


# -- window ---------------------------------------------------------------------------------------
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


def test_unlock_celebrates(window):
    from jarvis.ui.fx import Confetti

    w, app = window
    app.events.emit("user", text="hello")
    w.qt.processEvents()
    assert w.centralWidget().findChildren(Confetti)
    assert w.big_orb.cute_kind == "star" and "Achievement unlocked" in w.chat.toPlainText()


def test_gallery_toggles(window):
    w, app = window
    w.toggle_badges()
    assert w.badges.isVisible()
    w._escape()
    assert not w.badges.isVisible()


def test_toasts_survive_window_teardown(window):
    w, app = window
    w.toast("🔔", "hello", "#fff", 0.05)
    w.toasts._restack()
    w.close()
    w.deleteLater()
    w.qt.processEvents()  # must not raise when toasts are destroyed after the window
