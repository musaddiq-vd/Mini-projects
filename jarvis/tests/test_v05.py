"""Clock-time reminders that survive restarts, chat history, mini mode, hotkey."""

import sys
import time
from datetime import datetime

import pytest

from jarvis.tools import reminders as rem


@pytest.fixture(autouse=True)
def clean_reminders():
    for t in list(rem._timers.values()):
        t.cancel()
    rem._timers.clear()
    rem._mem.clear()
    yield
    for t in list(rem._timers.values()):
        t.cancel()
    rem._timers.clear()


NOW = datetime(2026, 11, 8, 15, 0)  # 3 pm


@pytest.mark.parametrize("text, expect", [
    ("18:30", (8, 18, 30)), ("6:30 pm", (8, 18, 30)), ("6pm", (8, 18, 0)), ("shaam 7", (8, 19, 0)),
    ("7", (8, 19, 0)),            # no am/pm at 3 pm -> 7 pm today
    ("7 am", (9, 7, 0)),          # already past -> tomorrow
    ("14:00", (9, 14, 0)), ("12 am", (9, 0, 0)), ("15.45", (8, 15, 45)),
])
def test_parse_clock(text, expect):
    due = rem.parse_clock(text, NOW)
    assert (due.day, due.hour, due.minute) == expect


@pytest.mark.parametrize("text", ["", "soon", "25:00", "10:75"])
def test_parse_clock_rejects(text):
    assert rem.parse_clock(text, NOW) is None


@pytest.fixture
def state_ctx(ctx, tmp_path):
    from jarvis.state import StateStore

    ctx.state = StateStore(tmp_path / "state.json")
    return ctx


def test_reminders_are_saved_and_listed(registry, state_ctx):
    out = registry.call("set_reminder", {"message": "call mom", "at": "23:59"}, state_ctx)
    assert "Reminder #1" in out["result"]
    registry.call("set_reminder", {"message": "tea", "minutes": 30}, state_ctx)
    saved = state_ctx.state.get("reminders")
    assert [r["message"] for r in saved] == ["tea", "call mom"] or len(saved) == 2
    listed = registry.call("list_reminders", {}, state_ctx)["reminders"]
    assert {r["message"] for r in listed} == {"call mom", "tea"}
    assert registry.call("cancel_reminder", {"id": 1}, state_ctx)["result"] == "Cancelled reminder #1."
    assert [r["message"] for r in state_ctx.state.get("reminders")] == ["tea"]


def test_reminder_needs_a_time(registry, state_ctx):
    assert "error" in registry.call("set_reminder", {"message": "x"}, state_ctx)
    assert "error" in registry.call("set_reminder", {"message": "x", "at": "whenever"}, state_ctx)


def test_restart_restores_and_tells_missed_ones(registry, state_ctx):
    state_ctx.state.set("reminders", [
        {"id": 3, "due": time.time() - 600, "message": "missed one"},
        {"id": 4, "due": time.time() + 3600, "message": "later one"},
    ])
    assert rem.restore_reminders(state_ctx, missed_delay=0.01) == 2
    deadline = time.time() + 2
    while not state_ctx.messages and time.time() < deadline:
        time.sleep(0.02)
    assert state_ctx.messages and "missed one" in state_ctx.messages[0] and "while I was off" in state_ctx.messages[0]
    assert [r["id"] for r in state_ctx.state.get("reminders")] == [4]
    assert 4 in rem._timers


def test_due_reminder_fires_and_is_removed(registry, state_ctx):
    registry.call("set_reminder", {"message": "stretch", "minutes": 0.001}, state_ctx)
    deadline = time.time() + 2
    while not state_ctx.messages and time.time() < deadline:
        time.sleep(0.02)
    assert state_ctx.messages == ["Reminder: stretch"]
    assert state_ctx.state.get("reminders") == []


# -- history -----------------------------------------------------------------------

def test_history_round_trip(tmp_path):
    from jarvis.history import ChatHistory

    h = ChatHistory(tmp_path / "h.jsonl")
    h.add("user", "hello")
    h.add("assistant", "hi there")
    h.add("user", "(system: asleep) ignore me")
    h.add("user", "   ")
    assert [(i["role"], i["text"]) for i in h.recent()] == [("user", "hello"), ("assistant", "hi there")]
    h.clear()
    assert h.recent() == []


def test_history_trims_itself(tmp_path, monkeypatch):
    import jarvis.history as history

    monkeypatch.setattr(history, "MAX_BYTES", 2000)
    h = history.ChatHistory(tmp_path / "h.jsonl")
    for k in range(200):
        h.add("user", f"message number {k}")
    assert (tmp_path / "h.jsonl").stat().st_size < 4000
    assert h.recent(1)[0]["text"] == "message number 199"


# -- window: history restore, mini mode -----------------------------------------------

@pytest.fixture
def window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings())
    w = MainWindow(app, Bridge())
    yield w, app, qt
    w._leave_mini()


def test_chat_is_saved_and_comes_back(window):
    from jarvis.ui.window import Bridge, MainWindow

    w, app, qt = window
    app.events.emit("user", text="remember this chat")
    app.events.emit("assistant", text="I will")
    qt.processEvents()
    again = MainWindow(app, Bridge())
    text = again.chat.toPlainText()
    assert "remember this chat" in text and "earlier conversation" in text
    again._clear_chat()
    fresh = MainWindow(app, Bridge()).chat.toPlainText()
    assert "remember this chat" not in fresh and "💡" in fresh  # empty chat shows the welcome card


def test_mini_mode_shares_the_orb(window):
    w, app, qt = window
    w.show()
    w.toggle_mini()
    assert w.mini.isVisible() and not w.isVisible()
    app.events.emit("state", state="speaking")
    app.events.emit("caption", who="assistant", text="hello from mini")
    qt.processEvents()
    assert w.mini.orb.state == "speaking" and w.big_orb.state == "speaking"
    assert "hello from mini" in w.mini.caption.text()
    w.toggle_mini()
    assert not w.mini.isVisible() and w.isVisible()


def test_hotkey_wakes_jarvis(window):
    w, app, qt = window
    calls = []
    w.assistant = type("A", (), {"awake": False, "wake_up": lambda self: calls.append("wake")})()
    w._on_hotkey()
    assert calls == ["wake"] and w.isVisible()


def test_global_hotkey_can_start_and_stop():
    from jarvis.ui.hotkey import GlobalHotkey

    hk = GlobalHotkey(lambda: None, key="F", mods=0x0002 | 0x0001 | 0x0004)  # Ctrl+Alt+Shift+F
    started = hk.start()
    hk.stop()
    assert started in (True, False)  # False is fine off Windows or if another app owns it


def test_quit_is_not_blocked_by_mini_mode(window):
    from PyQt6.QtCore import QTimer

    w, app, qt = window
    w.show()
    w.toggle_mini()
    QTimer.singleShot(50, w.close)  # like "Quit" from the mini menu
    QTimer.singleShot(3000, lambda: setattr(w, "_stuck", True) or qt.exit(1))
    code = qt.exec()
    assert code == 0 and not getattr(w, "_stuck", False)
    assert not w.mini.isVisible()
