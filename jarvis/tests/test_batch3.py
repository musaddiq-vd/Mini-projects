"""Contacts and birthdays, "tell me when" alerts and focus stats."""

from datetime import date, timedelta

import pytest

from jarvis.state import StateStore
from jarvis.tools import smart_alerts
from jarvis.tools.contacts import _birthday, _phone, next_birthday
from jarvis.tools.focus_stats import summarize


@pytest.fixture
def sctx(ctx):
    ctx.state = StateStore(ctx.memory.path.parent / "state.json")
    return ctx


# -- contacts -----------------------------------------------------------------------------------------
def test_phone_and_birthday_parsing():
    assert _phone("+91 98765 43210") == "919876543210"
    assert _phone("098765 43210") == "919876543210"
    assert _phone("9876543210") == "919876543210"
    assert _phone("0044 20 7946 0958") == "442079460958"
    assert _phone("123") is None
    assert _birthday("1995-03-12") == "03-12"
    assert _birthday("12 march") == "03-12"
    assert next_birthday("02-29", date(2027, 1, 1))[0] == date(2027, 2, 28)


def test_save_find_and_whatsapp_uses_it(registry, sctx):
    out = registry.call("save_contact", {"name": "Mom", "phone": "98765 43210", "relation": "mother"}, sctx)
    assert out["saved"]["phone"] == "+919876543210"
    registry.call("save_contact", {"name": "mom", "birthday": "12 march"}, sctx)  # only adds the birthday
    found = registry.call("find_contact", {"query": "mom"}, sctx)["contacts"][0]
    assert found["phone"] == "+919876543210" and found["relation"] == "mother" and "March" in found["birthday"]
    assert registry.call("find_contact", {"query": "mother"}, sctx)["count"] == 1  # by relation too
    from jarvis.tools.messaging import _resolve_phone

    assert _resolve_phone(sctx, "Mom") == "919876543210"


def test_contact_errors_undo_and_delete(registry, sctx):
    assert "phone number" in registry.call("save_contact", {"name": "X", "phone": "12"}, sctx)["error"]
    assert "e-mail" in registry.call("save_contact", {"name": "X", "email": "nope"}, sctx)["error"]
    assert "something to save" in registry.call("save_contact", {"name": "X"}, sctx)["error"]
    registry.call("save_contact", {"name": "Ali", "phone": "9876500000"}, sctx)
    registry.call("save_contact", {"name": "Ali", "phone": "9876511111"}, sctx)
    sctx.undo.undo()  # back to the first number
    assert sctx.memory.get("contact_ali").value == "919876500000"
    assert "Deleted" in registry.call("delete_contact", {"name": "Ali"}, sctx)["result"]
    assert sctx.memory.get("contact_ali") is None
    sctx.undo.undo()
    assert sctx.memory.get("contact_ali").value == "919876500000"
    assert "error" in registry.call("delete_contact", {"name": "Nobody"}, sctx)


def test_upcoming_birthdays_and_digest(registry, sctx):
    soon = date.today() + timedelta(days=3)
    registry.call("save_contact", {"name": "Sara", "birthday": soon.isoformat()}, sctx)
    registry.call("save_contact", {"name": "Zed", "birthday": (date.today() + timedelta(days=60)).isoformat()}, sctx)
    rows = registry.call("upcoming_birthdays", {"days": 30}, sctx)["birthdays"]
    assert [r["name"] for r in rows] == ["Sara"] and rows[0]["in_days"] == 3
    digest = registry.call("daily_digest", {"weather": False}, sctx)
    assert digest["birthdays"] and digest["birthdays"][0].startswith("Sara")


# -- tell me when -------------------------------------------------------------------------------------
@pytest.fixture
def pc(monkeypatch):
    now = {"battery": (50, True), "online": True, "apps": {"chrome", "explorer"}, "free_gb": 50.0, "ram": 70.0}
    monkeypatch.setitem(smart_alerts.PROBES, "battery", lambda: now["battery"])
    monkeypatch.setitem(smart_alerts.PROBES, "online", lambda: now["online"])
    monkeypatch.setitem(smart_alerts.PROBES, "apps", lambda: now["apps"])
    monkeypatch.setitem(smart_alerts.PROBES, "free_gb", lambda drive: now["free_gb"])
    monkeypatch.setitem(smart_alerts.PROBES, "ram", lambda: now["ram"])
    return now


def test_one_shot_alerts_ring_once(registry, sctx, pc):
    registry.call("add_alert", {"kind": "battery_full"}, sctx)
    registry.call("add_alert", {"kind": "app_closed", "value": "Chrome"}, sctx)
    registry.call("add_alert", {"kind": "internet_back", "message": "Internet is back - resume the upload"}, sctx)
    assert smart_alerts.check(sctx.state) == ["Internet is back - resume the upload"]
    assert smart_alerts.check(sctx.state) == []
    pc["battery"] = (100, True)
    pc["apps"] = {"explorer"}
    msgs = smart_alerts.check(sctx.state)
    assert len(msgs) == 2 and any("battery full" in m for m in msgs) and any("Chrome closed" in m for m in msgs)
    assert registry.call("list_alerts", {}, sctx)["result"] == "No alerts set."


def test_repeating_alert_rearms(registry, sctx, pc):
    registry.call("add_alert", {"kind": "battery_below", "value": "20", "repeat": True}, sctx)
    pc["battery"] = (15, False)
    assert len(smart_alerts.check(sctx.state)) == 1
    assert smart_alerts.check(sctx.state) == []  # still low: not again
    pc["battery"] = (60, True)
    smart_alerts.check(sctx.state)
    pc["battery"] = (18, False)
    assert len(smart_alerts.check(sctx.state)) == 1  # dropped again: rings again


def test_alert_errors_list_cancel_and_undo(registry, sctx, pc):
    assert "Which app" in registry.call("add_alert", {"kind": "app_closed"}, sctx)["error"]
    assert "isn't running" in registry.call("add_alert", {"kind": "app_closed", "value": "spotify"}, sctx)["error"]
    assert "number" in registry.call("add_alert", {"kind": "disk_below", "value": "lots"}, sctx)["error"]
    out = registry.call("add_alert", {"kind": "disk_below", "value": "10", "drive": "D:"}, sctx)
    assert "D:" in out["result"]
    registry.call("add_alert", {"kind": "internet_lost"}, sctx)
    assert len(registry.call("list_alerts", {}, sctx)["alerts"]) == 2
    sctx.undo.undo()
    assert len(registry.call("list_alerts", {}, sctx)["alerts"]) == 1
    assert "error" in registry.call("cancel_alert", {"id": 99}, sctx)
    registry.call("cancel_alert", {"id": 0}, sctx)
    assert registry.call("list_alerts", {}, sctx)["result"] == "No alerts set."


def test_a_failing_sensor_keeps_the_alert(registry, sctx, pc, monkeypatch):
    registry.call("add_alert", {"kind": "charger_unplugged"}, sctx)
    monkeypatch.setitem(smart_alerts.PROBES, "battery", lambda: 1 / 0)
    assert smart_alerts.check(sctx.state) == []
    assert len(registry.call("list_alerts", {}, sctx)["alerts"]) == 1


# -- focus stats --------------------------------------------------------------------------------------
def test_focus_summary():
    today = date(2026, 11, 19)  # a Thursday
    d = lambda n: (today - timedelta(days=n)).isoformat()  # noqa: E731
    log = [{"date": d(0), "minutes": 25, "task": "Python"}, {"date": d(1), "minutes": 50, "task": "python"},
           {"date": d(2), "minutes": 25}, {"date": d(8), "minutes": 30, "task": "Maths"}, {"bad": 1}]
    s = summarize(log, today)
    assert s["today"] == 25 and s["week"] == 100 and s["last_week"] == 30
    assert s["streak_days"] == 3 and s["best_day"] == (d(1), 50)
    assert s["top_tasks"][0] == ("python", 75)


def test_focus_stats_tool_and_task_is_logged(registry, sctx, monkeypatch):
    assert "No focus" in registry.call("focus_stats", {}, sctx)["result"]
    from jarvis.tools import timers

    started = {}

    class FakeTimer:
        def __init__(self, secs, fn):
            started["fn"] = fn
            self.daemon = True

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(timers.threading, "Timer", FakeTimer)
    registry.call("focus_mode", {"minutes": 25, "task": "Physics"}, sctx)
    started["fn"]()  # the session ends
    out = registry.call("focus_stats", {}, sctx)
    assert out["today"] == "25 min" and out["top_subjects"] == ["Physics 25 min"] and out["streak_days"] == 1
