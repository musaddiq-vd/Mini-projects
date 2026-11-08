"""Timers & focus, notes, routines, repeating reminders, chat search, orb feelings."""

import sys
import time
from datetime import datetime

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


# -- timers ---------------------------------------------------------------------------

def test_timer_runs_and_rings(registry, sctx):
    out = registry.call("set_timer", {"seconds": 0.05, "label": "eggs"}, sctx)
    assert "started" in out["result"]
    assert any(k == "timer" and d["label"] == "eggs" for k, d in sctx.seen)
    assert wait_for(lambda: sctx.messages)
    assert sctx.messages == ["Timer done: eggs"]
    assert any(k == "timer_end" and d["done"] for k, d in sctx.seen)


def test_timer_list_and_cancel(registry, sctx):
    registry.call("set_timer", {"minutes": 10, "label": "tea"}, sctx)
    assert registry.call("list_timers", {}, sctx)["timers"][0]["label"] == "tea"
    assert "Stopped" in registry.call("cancel_timer", {}, sctx)["result"]
    assert registry.call("list_timers", {}, sctx)["result"] == "No timers running."
    assert any(k == "timer_end" and not d["done"] for k, d in sctx.seen)


def test_timer_limits(registry, sctx):
    assert "error" in registry.call("set_timer", {}, sctx)
    assert "error" in registry.call("set_timer", {"minutes": 2000}, sctx)


def test_focus_mode_keeps_jarvis_quiet(registry, sctx):
    from jarvis.tools.timers import focus_active

    registry.call("focus_mode", {"minutes": 25, "task": "maths"}, sctx)
    assert focus_active(sctx)
    registry.call("cancel_timer", {}, sctx)
    assert not focus_active(sctx)


# -- notes ----------------------------------------------------------------------------

def test_notes_lifecycle(registry, sctx):
    registry.call("add_note", {"text": "buy milk"}, sctx)
    registry.call("add_note", {"text": "call the bank"}, sctx)
    registry.call("add_note", {"text": "wifi password is on the router", "todo": False}, sctx)
    assert registry.call("list_notes", {}, sctx)["count"] == 3
    assert registry.call("list_notes", {"show": "todo"}, sctx)["count"] == 2
    out = registry.call("complete_note", {"item": "milk"}, sctx)
    assert "buy milk" in out["result"] and "1 left" in out["result"]
    out = registry.call("complete_note", {"item": "2"}, sctx)
    assert "All done" in out["result"]
    assert registry.call("list_notes", {"show": "done"}, sctx)["count"] == 2
    assert "Cleared 2" in registry.call("delete_note", {"item": "done"}, sctx)["result"]
    assert registry.call("list_notes", {"show": "all"}, sctx)["count"] == 1


def test_notes_undo(registry, sctx):
    registry.call("add_note", {"text": "secret plan"}, sctx)
    registry.call("delete_note", {"item": "secret"}, sctx)
    assert registry.call("list_notes", {"show": "all"}, sctx)["result"] == "Nothing here yet."
    registry.call("undo", {}, sctx)
    assert registry.call("list_notes", {"show": "all"}, sctx)["count"] == 1


def test_notes_errors(registry, sctx):
    assert "error" in registry.call("add_note", {"text": "  "}, sctx)
    assert "error" in registry.call("complete_note", {"item": "nothing like this"}, sctx)


# -- routines -------------------------------------------------------------------------

def test_routines(registry, sctx):
    out = registry.call("save_routine", {"name": "Work Mode", "steps": ["open VS Code", "open Chrome", " "]}, sctx)
    assert "2 step" in out["result"]
    run = registry.call("run_routine", {"name": "work mode"}, sctx)
    assert run["steps"] == ["open VS Code", "open Chrome"] and "1) open VS Code" in run["result"]
    assert registry.call("run_routine", {"name": "work"}, sctx)["routine"] == "work mode"  # forgiving
    assert "error" in registry.call("run_routine", {"name": "party"}, sctx)
    assert "work mode" in registry.call("list_routines", {}, sctx)["routines"]
    registry.call("delete_routine", {"name": "work mode"}, sctx)
    assert "No routines" in registry.call("list_routines", {}, sctx)["result"]
    registry.call("undo", {}, sctx)
    assert "work mode" in registry.call("list_routines", {}, sctx)["routines"]


def test_routine_names_reach_the_prompt(tmp_path, isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings

    app = create_app(Settings())
    app.registry.call("save_routine", {"name": "good night", "steps": ["set volume to 10"]}, app.ctx)
    assert "Saved routines: good night" in app.prompt()


def test_routine_limits(registry, sctx):
    assert "error" in registry.call("save_routine", {"name": "x", "steps": []}, sctx)
    assert "error" in registry.call("save_routine", {"name": "x", "steps": ["a"] * 20}, sctx)


# -- repeating reminders --------------------------------------------------------------

def test_next_repeat():
    from jarvis.tools.reminders import next_repeat

    fri = datetime(2026, 11, 6, 9, 0).timestamp()  # a Friday
    now = fri + 60
    assert datetime.fromtimestamp(next_repeat(fri, "daily", now)).day == 7
    assert datetime.fromtimestamp(next_repeat(fri, "weekdays", now)).weekday() == 0  # Monday


def test_daily_reminder_books_the_next_day(registry, sctx):
    import jarvis.tools.reminders as rem

    out = registry.call("set_reminder", {"message": "vitamins", "at": "9:00 am", "repeat": "daily"}, sctx)
    assert "every day" in out["result"]
    saved = sctx.state.get("reminders")[0]
    assert saved["repeat"] == "daily"
    rid = saved["id"]
    rem._timers.pop(rid).cancel()
    rem._schedule(sctx, rid, time.time(), "vitamins")  # make it ring now
    assert wait_for(lambda: sctx.messages)
    again = sctx.state.get("reminders")[0]
    assert again["due"] > time.time() + 3600  # booked for the next morning
    rem._timers.pop(rid).cancel()


def test_repeat_needs_a_clock_time(registry, sctx):
    assert "error" in registry.call("set_reminder", {"message": "x", "minutes": 5, "repeat": "daily"}, sctx)
    assert "error" in registry.call("set_reminder", {"message": "x", "at": "9 am", "repeat": "hourly"}, sctx)


# -- chat search ------------------------------------------------------------------------

def test_search_chat_history(registry, ctx, isolated_home):
    from jarvis.history import ChatHistory

    h = ChatHistory(isolated_home / "history.jsonl")
    h.add("user", "plan a trip to Goa in December")
    h.add("assistant", "Goa in December is lovely, book early.")
    h.add("user", "what is 2+2")
    h.add("assistant", "4")
    out = registry.call("search_chat_history", {"query": "humne goa trip ke baare mein kya baat ki thi"}, ctx)
    texts = [m["text"] for m in out["matches"]]
    assert texts[0].startswith("plan a trip") and "book early" in texts[1]
    assert len(registry.call("search_chat_history", {}, ctx)["matches"]) == 4
    assert "Nothing" in registry.call("search_chat_history", {"query": "bitcoin"}, ctx)["result"]


def test_core_saves_history_without_a_window(isolated_home):
    from jarvis.app import create_app
    from jarvis.config import Settings

    app = create_app(Settings())
    app.events.emit("user", text="hello from the terminal")
    assert app.history.recent()[-1]["text"] == "hello from the terminal"


# -- orb feelings and window reactions --------------------------------------------------

@pytest.fixture
def window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings())
    return MainWindow(app, Bridge()), app, qt


def test_thanks_shows_love_and_jokes_get_a_wink(window):
    w, app, qt = window
    feelings = []
    app.events.subscribe(lambda k, d: feelings.append(d["feeling"]) if k == "emote" else None)
    w._react_to_user("Thank you Jarvis!")  # called directly so an error here fails the test
    assert w.big_orb.emotion == "love" and w.big_orb.hearts and feelings == ["love"]
    app.events.emit("user", text="ek chutkula sunao")
    app.events.emit("assistant", text="Why did the computer go to the doctor? It had a virus!")
    qt.processEvents()
    assert w.big_orb.emotion == "wink"


def test_errors_look_sad_and_timers_show_on_the_orb(window):
    w, app, qt = window
    app.events.emit("tool", name="get_weather", args={}, result={"error": "offline"})
    qt.processEvents()
    assert w.big_orb.emotion == "sad"
    app.events.emit("timer", id=7, label="tea", ends=time.time() + 60, total=60)
    qt.processEvents()
    assert 7 in w.big_orb.timers
    app.events.emit("timer_end", id=7, label="tea", done=True)
    qt.processEvents()
    assert 7 not in w.big_orb.timers and w.big_orb.emotion == "surprised"


def test_new_sounds(monkeypatch):
    from jarvis.sfx import SoundFX

    fx = SoundFX()
    played = []
    monkeypatch.setattr(fx, "play", played.append)
    fx.on_event("timer_end", {"done": True})
    fx.on_event("timer_end", {"done": False})
    fx.on_event("emote", {"feeling": "love"})
    assert played == ["alarm", "love"]


def _walk_schema(schema, path, problems):
    """Rules the Gemini Live API enforces on tool declarations (a bad one blocks connecting)."""
    if not isinstance(schema, dict):
        return
    t = schema.get("type")
    if t is None:
        problems.append(f"{path}: no type")
    if "enum" in schema:
        if not schema["enum"] or any(not isinstance(v, str) or v == "" for v in schema["enum"]):
            problems.append(f"{path}: enum values must be non-empty strings")
    if t == "ARRAY" and "items" not in schema:
        problems.append(f"{path}: array without items")
    for name, sub in (schema.get("properties") or {}).items():
        _walk_schema(sub, f"{path}.{name}", problems)
    if "items" in schema:
        _walk_schema(schema["items"], f"{path}[]", problems)
    for req in schema.get("required", []):
        if req not in (schema.get("properties") or {}):
            problems.append(f"{path}: required '{req}' is not a property")


def test_every_tool_declaration_is_valid_for_gemini(registry):
    from google.genai import types

    problems = []
    for decl in registry.declarations():
        _walk_schema(decl["parameters"], decl["name"], problems)
        types.FunctionDeclaration(**decl)  # the SDK must accept it too
    assert problems == []
