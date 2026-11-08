"""Reminder pop-ups: a card when a reminder is set and when it rings, with Done / Snooze / Cancel."""

import sys
import time

import pytest

from jarvis.events import Events
from jarvis.state import StateStore


@pytest.fixture
def sctx(ctx, tmp_path):
    ctx.state = StateStore(tmp_path / "state.json")
    ctx.events = Events()
    ctx.seen = []
    ctx.events.subscribe(lambda k, d: ctx.seen.append((k, d)))
    return ctx


def wait_for(cond, secs=3.0):
    end = time.time() + secs
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def kinds(ctx, kind):
    return [d for k, d in ctx.seen if k == kind]


def ring_now(ctx, rid, message):
    import jarvis.tools.reminders as rem

    rem._timers.pop(rid).cancel()
    rem._schedule(ctx, rid, time.time(), message)
    assert wait_for(lambda: any(d["id"] == rid for d in kinds(ctx, "reminder_due")))


def test_set_and_ring_send_events_and_speak(registry, sctx):
    registry.call("set_reminder", {"message": "Call mom", "minutes": 30}, sctx)
    (made,) = kinds(sctx, "reminder_set")
    assert made["message"] == "Call mom" and made["due"] > time.time() + 29 * 60 and made["repeat"] == ""
    ring_now(sctx, made["id"], "Call mom")
    due = kinds(sctx, "reminder_due")[0]
    assert due["message"] == "Call mom" and not due["late"]
    assert wait_for(lambda: any("Reminder: Call mom" in m for m in sctx.messages))  # Jarvis says it
    assert sctx.state.get("reminders_rung")[-1]["message"] == "Call mom"


def test_done_marks_the_latest_and_closes_the_card(registry, sctx):
    registry.call("set_reminder", {"message": "Pay bill", "minutes": 5}, sctx)
    rid = kinds(sctx, "reminder_set")[0]["id"]
    ring_now(sctx, rid, "Pay bill")
    out = registry.call("reminder_done", {}, sctx)
    assert "Pay bill" in out["result"]
    assert kinds(sctx, "reminder_closed")[-1]["id"] == rid
    assert sctx.state.get("reminders_done")[-1]["message"] == "Pay bill"
    assert "error" in registry.call("reminder_done", {}, sctx)  # nothing waiting any more


def test_snooze_books_it_again(registry, sctx):
    import jarvis.tools.reminders as rem

    registry.call("set_reminder", {"message": "Stretch", "minutes": 5}, sctx)
    rid = kinds(sctx, "reminder_set")[0]["id"]
    ring_now(sctx, rid, "Stretch")
    out = registry.call("snooze_reminder", {"minutes": 15}, sctx)
    assert "Snoozed" in out["result"] and "Stretch" in out["result"]
    again = kinds(sctx, "reminder_set")[-1]
    assert again["message"] == "Stretch" and 14 * 60 < again["due"] - time.time() <= 15 * 60
    assert kinds(sctx, "reminder_closed")[-1]["id"] == rid
    assert "error" in registry.call("snooze_reminder", {}, sctx)  # it's no longer waiting
    assert "error" in registry.call("snooze_reminder", {"minutes": 0}, sctx)
    rem._timers.pop(again["id"]).cancel()


def test_cancel_closes_the_card(registry, sctx):
    import jarvis.tools.reminders as rem

    registry.call("set_reminder", {"message": "Tea", "minutes": 5}, sctx)
    rid = kinds(sctx, "reminder_set")[0]["id"]
    registry.call("cancel_reminder", {"id": rid}, sctx)
    assert kinds(sctx, "reminder_closed")[-1]["id"] == rid and rid not in rem._timers


def test_missed_reminder_is_marked_late(registry, sctx):
    import jarvis.tools.reminders as rem

    sctx.state.set("reminders", [{"id": 7, "due": time.time() - 600, "message": "Old task", "repeat": ""}])
    rem.restore_reminders(sctx, missed_delay=0)
    assert wait_for(lambda: kinds(sctx, "reminder_due"))
    assert kinds(sctx, "reminder_due")[0]["late"] is True


# -- the pop-ups ------------------------------------------------------------------------------
@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def settle(qt, ms=50):
    end = time.time() + ms / 1000
    while time.time() < end:
        qt.processEvents()


def make(calls):
    from jarvis.ui.reminder_popup import ReminderPopups

    return ReminderPopups(lambda: "#00d4ff", on_done=lambda r: calls.append(("done", r)),
                          on_snooze=lambda r, m: calls.append(("snooze", r, m)),
                          on_cancel=lambda r: calls.append(("cancel", r)), on_open=lambda: calls.append("open"))


def test_due_card_buttons(qt):
    calls = []
    rp = make(calls)
    card = rp.show("due", 3, "Mummy ko call karna", time.time())
    settle(qt)
    assert card.isVisible() and "REMINDER" in card.head.text() and card.text.text() == "Mummy ko call karna"
    done, snooze, more = card.buttons
    done.click()
    assert calls == [("done", 3)] and rp.popups == []
    card = rp.show("due", 4, "Bill", time.time())
    card.buttons[1].click()
    assert calls[-1] == ("snooze", 4, 10)
    card = rp.show("due", 5, "Gym", time.time())
    card.buttons[2].menu().actions()[2].trigger()  # 30 minutes
    assert calls[-1] == ("snooze", 5, 30)
    rp.close_all()


def test_set_card_cancel_and_auto_close(qt, monkeypatch):
    from jarvis.ui import reminder_popup

    monkeypatch.setattr(reminder_popup, "SET_SECONDS", 0.05)
    calls = []
    rp = make(calls)
    card = rp.show("set", 8, "Gym jaana", time.time() + 25 * 60, repeat="daily")
    assert "DAILY" in card.head.text() and "in 25 min" in card.sub.text()
    settle(qt, 400)
    assert rp.popups == []  # went away by itself
    card = rp.show("set", 9, "Tea", time.time() + 60)
    card.buttons[0].click()
    assert calls[-1] == ("cancel", 9)


def test_cards_stack_upwards_and_ringing_replaces_set(qt):
    rp = make([])
    a = rp.show("set", 1, "One", time.time() + 600)
    b = rp.show("set", 2, "Two", time.time() + 600)
    settle(qt, 600)
    assert b.y() < a.y() and a.x() == b.x()
    rp.show("due", 1, "One", time.time())
    assert [(p.rid, p.kind) for p in rp.popups] == [(2, "set"), (1, "due")]
    for i in range(10, 16):
        rp.show("due", i, f"Task {i}", time.time())
    assert len([p for p in rp.popups if not p._closing]) <= 4  # never a wall of cards
    rp.close(15)
    assert all(p.rid != 15 for p in rp.popups)
    rp.close_all()


def test_long_text_and_time_words():
    from jarvis.ui.reminder_popup import until

    now = time.time()
    assert until(now + 20, now) == "in under a minute" and until(now + 25 * 60, now) == "in 25 min"
    assert until(now + 90 * 60, now) == "in 1 h 30 min" and until(now + 3 * 86400, now) == "in 3 days"


def test_window_shows_and_closes_cards(qt, isolated_home, sandbox):
    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    app = create_app(Settings(allowed_roots=[str(sandbox)]))
    w = MainWindow(app, Bridge())
    out = app.registry.call("set_reminder", {"message": "Drink water", "minutes": 20}, app.ctx)
    settle(qt)
    assert "Reminder #" in out["result"]
    (card,) = w.reminders.popups
    assert card.kind == "set" and card.text.text() == "Drink water"
    app.registry.call("cancel_reminder", {"id": card.rid}, app.ctx)
    settle(qt)
    assert w.reminders.popups == []
    w.reminders.close_all()
