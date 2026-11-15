"""Crash recovery: the running marker, the notice after an unexpected exit, and missed reminders told once."""

import json
import os
import sys
import time

import psutil
import pytest

from jarvis import session
from jarvis.state import StateStore
from jarvis.tools import reminders as rem


@pytest.fixture
def marker(isolated_home):
    return session.marker_path()


def test_clean_run_leaves_no_marker(marker):
    assert session.begin() is None  # first start ever: nothing to recover
    assert json.loads(marker.read_text(encoding="utf-8"))["pid"] == os.getpid()
    session.end()
    assert not marker.exists()
    assert session.begin() is None  # after a clean quit: still nothing


def test_marker_left_behind_means_a_crash(marker):
    marker.write_text(json.dumps({"pid": 999_999_999, "started": time.time() - 60}), encoding="utf-8")
    previous = session.begin()
    assert previous is not None and previous["pid"] == 999_999_999
    assert json.loads(marker.read_text(encoding="utf-8"))["pid"] == os.getpid()  # this run is marked now


def test_half_written_marker_still_counts(marker):
    marker.write_text("{not json", encoding="utf-8")
    assert session.begin() == {}


def test_a_run_still_closing_after_a_restart_is_not_a_crash(marker):
    parent = psutil.Process(os.getppid())  # a process that is surely alive
    marker.write_text(json.dumps({"pid": parent.pid, "started": parent.create_time()}), encoding="utf-8")
    assert session.begin() is None


def test_a_reused_pid_is_not_mistaken_for_the_old_run(marker):
    parent = psutil.Process(os.getppid())  # alive, but started long after the marker says
    marker.write_text(json.dumps({"pid": parent.pid, "started": parent.create_time() - 3600}), encoding="utf-8")
    assert session.begin() is not None


def test_quitting_old_run_keeps_the_new_runs_marker(marker):
    marker.write_text(json.dumps({"pid": os.getppid(), "started": time.time()}), encoding="utf-8")
    session.end()  # another process owns it
    assert marker.exists()


@pytest.fixture
def state_ctx(ctx, isolated_home):
    ctx.state = StateStore(isolated_home / "state.json")
    yield ctx
    for rid in (9101, 9102):
        timer = rem._timers.pop(rid, None)
        if timer:
            timer.cancel()


def _wait(pred, seconds=2.0):
    deadline = time.time() + seconds
    while not pred() and time.time() < deadline:
        time.sleep(0.02)


def test_recover_shows_notice_and_tells_a_missed_reminder_once(state_ctx, monkeypatch):
    state_ctx.state.set("reminders", [
        {"id": 9101, "due": time.time() - 900, "message": "take medicine"},
        {"id": 9102, "due": time.time() + 3600, "message": "call the bank"},
    ])
    real = rem.restore_reminders
    monkeypatch.setattr(rem, "restore_reminders", lambda ctx: real(ctx, missed_delay=0.05))

    class App:
        ctx = state_ctx

    shown = []
    assert session.recover(App(), shown.append) == 2
    assert shown == [session.CRASH_NOTICE]
    assert rem.restore_reminders(state_ctx) == 2  # the voice session connects a moment later and restores again
    _wait(lambda: state_ctx.messages)
    time.sleep(0.2)
    assert len(state_ctx.messages) == 1 and "take medicine" in state_ctx.messages[0]
    assert "while I was off" in state_ctx.messages[0]
    assert rem.restore_reminders(state_ctx) == 1  # after it rang: only the later one is left, still armed once
    time.sleep(0.2)
    assert len(state_ctx.messages) == 1
    assert [r["id"] for r in state_ctx.state.get("reminders")] == [9102]


@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def test_window_after_a_crash_has_the_chat_back_and_the_notice(qt, isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    app = create_app(Settings())
    app.history.add("user", "add milk to my shopping list")
    app.history.add("assistant", "Added milk to your shopping list.")
    w = MainWindow(app, Bridge())
    session.recover(app, w._add_system)
    qt.processEvents()
    text = w.chat.toPlainText()
    assert "add milk to my shopping list" in text and "Added milk" in text
    assert text.index("Added milk") < text.index("closed unexpectedly")
