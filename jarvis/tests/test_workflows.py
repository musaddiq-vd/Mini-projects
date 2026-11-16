"""Workflows: validation against the registry, deterministic runs, conditions, variables, waits and triggers."""

from datetime import datetime

import pytest

from jarvis.agents import workflows as wf
from jarvis.events import Events
from jarvis.state import StateStore
from jarvis.tools import Tool, ToolRegistry


def fake_tool(name, params, result, calls, required=()):
    def handler(ctx, **kw):
        calls.append((name, kw))
        return result(kw) if callable(result) else result

    return Tool(name, name, handler, {p: {"type": "string"} for p in params}, list(required))


@pytest.fixture
def reg():
    r = ToolRegistry()
    r.calls = []
    r.add(fake_tool("system_status", [], {"battery_percent": 22, "charging": False}, r.calls))
    r.add(fake_tool("open_app", ["name"], lambda kw: {"result": f"Opened {kw['name']}"}, r.calls, ["name"]))
    r.add(fake_tool("volume", ["action"], {"result": "ok"}, r.calls, ["action"]))
    r.add(fake_tool("add_note", ["text"], {"result": "noted"}, r.calls, ["text"]))
    r.add(fake_tool("broken", [], {"error": "it broke"}, r.calls))
    r.add(fake_tool("write_file", ["path", "text"], {"result": "written"}, r.calls))
    r.add(fake_tool("run_agents", ["goal"], {"result": "x"}, r.calls))
    return r


@pytest.fixture
def state(tmp_path, ctx):
    ctx.state = StateStore(tmp_path / "state.json")
    return ctx.state


def make(state, name, steps, trigger=None):
    wf.save(state, name, {"steps": steps, "trigger": trigger or {"type": "command"}})


# -- validation ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("steps, message", [
    ([], "at least one step"),
    (["open chrome"], "sentence"),
    ([{"tool": "nope"}], "no tool called 'nope'"),
    ([{"tool": "open_app"}], "needs name"),
    ([{"tool": "open_app", "args": {"name": "x", "colour": "red"}}], "no argument(s) colour"),
    ([{"tool": "run_agents", "args": {"goal": "x"}}], "can't be used inside a workflow"),
    ([{"tool": "open_app", "args": {"name": "{pc.app}"}}], "isn't saved by an earlier step"),
    ([{"tool": "volume", "args": {"action": "up"}, "if": "the moon looks round"}], "can't read the condition"),
    ([{"wait": 5000}], "between 1 and"),
    ([{"tool": "open_app", "args": {"name": "x"}, "save_as": "two words"}], "simple name"),
    ([{"wait": 600}, {"wait": 600}, {"wait": 600}, {"wait": 600}], "more than"),
])
def test_bad_steps_are_explained(reg, steps, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        wf.validate(steps, reg)


def test_good_steps_are_cleaned(reg):
    steps = wf.validate([{"tool": "system_status", "save_as": "pc"}, {"wait": "2"},
                         {"tool": "open_app", "args": {"name": "{pc.battery_percent}"}, "if": "battery < 30"}], reg)
    assert steps[1] == {"wait": 2.0}
    assert steps[2]["if"] == "battery < 30"


def test_builtin_examples_use_real_tools(registry):
    for name, flow in wf.BUILTIN.items():
        assert wf.validate(flow["steps"], registry), name


@pytest.mark.parametrize("raw, out", [
    (None, {"type": "command"}), ("startup", {"type": "startup"}), ("daily at 9:05", {"type": "daily", "at": "09:05"}),
    ({"type": "weekday", "at": "18.30"}, {"type": "weekdays", "at": "18:30"}),
])
def test_triggers(raw, out):
    assert wf.normalise_trigger(raw) == out


@pytest.mark.parametrize("raw", ["hourly", {"type": "daily"}, {"type": "daily", "at": "25:00"}])
def test_bad_triggers(raw):
    with pytest.raises(ValueError):
        wf.normalise_trigger(raw)


# -- conditions and variables --------------------------------------------------------------------------------
@pytest.mark.parametrize("cond, expected", [
    ("battery < 30", True), ("battery >= 30", False), ("not battery < 30", False),
    ("hour >= 18", True), ("time > 18:30", True), ("time < 9:00", False), ("weekday", True), ("weekend", False),
    ("charging", False), ("charging == false", True), ("day == monday", True), ("evening", True),
    ("{pc.battery_percent} < 30", True), ("{pc.charging} == false", True), ("{note.result} contains noted", True),
    ("{missing}", False), ("not {missing}", True), ("online", True),
])
def test_conditions(cond, expected):
    facts = {"battery": 22, "charging": False, "hour": 19, "minute": 0, "time": "19:00", "day": "monday",
             "weekday": True, "online": True}
    variables = {"pc": {"battery_percent": 22, "charging": False}, "note": {"result": "Noted it"}}
    assert wf.evaluate(cond, variables, facts) is expected


def test_unknown_battery_skips_the_step():
    assert wf.evaluate("battery < 30", {}, {"hour": 10}) is False


def test_variables_keep_their_type_alone_and_become_text_inside():
    v = {"pc": {"battery_percent": 22, "apps": ["a", "b"]}}
    assert wf.resolve({"n": "{pc.battery_percent}", "t": "Battery {pc.battery_percent}% {pc.apps.1}"}, v) == \
        {"n": 22, "t": "Battery 22% b"}


# -- running ------------------------------------------------------------------------------------------------
FACTS = {"battery": 50, "hour": 10, "time": "10:00", "day": "monday", "weekday": True}


def test_a_run_is_deterministic_with_variables_conditions_and_waits(ctx, reg, state):
    make(state, "Morning", [
        {"tool": "system_status", "save_as": "pc"},
        {"tool": "volume", "args": {"action": "mute"}, "if": "hour >= 18"},
        {"wait": 3},
        {"tool": "add_note", "args": {"text": "Battery at {pc.battery_percent}%"}, "if": "{pc.battery_percent} < 30"},
        {"tool": "open_app", "args": {"name": "Chrome"}},
    ])
    ctx.events = Events()
    plans = []
    ctx.events.subscribe(lambda kind, data: plans.append(data) if kind == "agents" else None)
    slept = []
    out = wf.run(ctx, "morning", reg, sleep=slept.append, facts=FACTS)
    assert out["status"] == "done" and slept == [3]
    assert [c[0] for c in reg.calls] == ["system_status", "add_note", "open_app"]
    assert reg.calls[1] == ("add_note", {"text": "Battery at 22%"})
    assert [x["status"] for x in out["log"]] == ["ok", "skipped", "waited", "ok", "ok"]
    assert plans[-1]["kind"] == "workflow" and [s["status"] for s in plans[-1]["steps"]] == \
        ["done", "skipped", "done", "done", "done"]
    assert state.get(wf.LAST_RUN_KEY)["morning"]["status"] == "done"


def test_a_run_stops_at_the_first_error(ctx, reg, state):
    make(state, "fails", [{"tool": "open_app", "args": {"name": "A"}}, {"tool": "broken"},
                          {"tool": "open_app", "args": {"name": "B"}}])
    out = wf.run(ctx, "fails", reg, sleep=lambda s: None, facts=FACTS)
    assert out["status"] == "failed" and "it broke" in out["error"]
    assert [c[1].get("name") for c in reg.calls if c[0] == "open_app"] == ["A"]


def test_a_declined_confirmation_stops_the_run(ctx, reg, state, answers):
    ctx.settings.permissions = {"files": "ask"}
    answers.append(False)
    make(state, "save", [{"tool": "write_file", "args": {"path": "a.txt", "text": "x"}},
                         {"tool": "open_app", "args": {"name": "B"}}])
    out = wf.run(ctx, "save", reg, sleep=lambda s: None, facts=FACTS)
    assert out["status"] == "declined" and "error" not in out
    assert reg.calls == []


def test_off_permission_is_an_error_that_stops_the_run(ctx, reg, state):
    ctx.settings.permissions = {"files": "off"}
    make(state, "save", [{"tool": "write_file", "args": {"path": "a.txt", "text": "x"}}])
    out = wf.run(ctx, "save", reg, sleep=lambda s: None, facts=FACTS)
    assert out["status"] == "failed" and "Not allowed" in out["error"]


def test_cancel_during_a_wait(ctx, reg, state):
    from jarvis.agents import team

    make(state, "slow", [{"wait": 60}, {"tool": "open_app", "args": {"name": "A"}}])
    out = wf.run(ctx, "slow", reg, sleep=lambda s: team.cancel("test"), facts=FACTS)
    assert out["status"] == "cancelled" and reg.calls == []
    assert wf.running() is None


def test_unknown_and_forgiving_names(ctx, reg, state):
    make(state, "night routine", [{"tool": "volume", "args": {"action": "down"}}])
    assert "No workflow" in wf.run(ctx, "party", reg)["error"]
    assert wf.run(ctx, "night", reg, sleep=lambda s: None, facts=FACTS)["workflow"] == "night routine"


# -- triggers -------------------------------------------------------------------------------------------------
def test_due_workflows(state):
    make(state, "standup", [{"wait": 1}], {"type": "weekdays", "at": "09:00"})
    make(state, "plants", [{"wait": 1}], {"type": "daily", "at": "18:00"})
    make(state, "hello", [{"wait": 1}], {"type": "startup"})
    monday_9 = datetime(2026, 11, 16, 9, 4)
    saturday_9 = datetime(2026, 11, 21, 9, 4)
    assert wf.due(state, monday_9) == ["standup"]
    assert wf.due(state, saturday_9) == []
    assert wf.due(state, datetime(2026, 11, 16, 9, 30)) == []  # window passed
    assert wf.due(state, datetime(2026, 11, 21, 18, 0)) == ["plants"]
    state.set(wf.LAST_RUN_KEY, {"standup": {"date": "2026-11-16"}})
    assert wf.due(state, monday_9) == []
    assert wf.at_startup(state) == ["hello"]


def test_scheduler_tick_runs_due_workflows(ctx, reg, state, monkeypatch):
    make(state, "standup", [{"tool": "open_app", "args": {"name": "Teams"}}], {"type": "daily", "at": "09:00"})
    told = []
    sched = wf.Scheduler(ctx, reg, notify=told.append)
    assert sched.tick(datetime(2026, 11, 16, 9, 1)) == ["standup"]
    assert reg.calls == [("open_app", {"name": "Teams"})] and "standup" in told[0]
    assert sched.tick(datetime(2026, 11, 16, 9, 2)) == []  # once a day


# -- the tools ---------------------------------------------------------------------------------------------------
def test_create_list_edit_run_delete_and_undo(ctx, registry, state, monkeypatch):
    from jarvis import agents

    monkeypatch.setattr(agents, "_registry", [registry])
    steps = [{"tool": "get_time", "save_as": "now"}, {"tool": "calculate", "args": {"expression": "2+2"}}]
    out = registry.call("create_workflow", {"name": "Quick Check", "steps": steps,
                                            "trigger": {"type": "daily", "at": "8:00"}}, ctx)
    assert "Created workflow 'quick check'" in out["result"] and "daily at 08:00" in out["result"]
    assert "already exists" in registry.call("create_workflow", {"name": "quick check", "steps": steps}, ctx)["error"]
    assert "no tool called" in registry.call("create_workflow", {"name": "x", "steps": [{"tool": "fly"}]}, ctx)["error"]
    listed = registry.call("list_workflows", {}, ctx)
    assert {"quick check", "work mode", "study mode"} <= {w["name"] for w in listed["workflows"]}
    edited = registry.call("edit_workflow", {"name": "quick", "trigger": {"type": "startup"}}, ctx)
    assert "Changed" in edited["result"] and wf.find(state, "quick check")[1]["trigger"] == {"type": "startup"}
    ran = registry.call("run_workflow", {"name": "quick check"}, ctx)
    assert ran["status"] == "done" and len(ran["log"]) == 2
    assert "Deleted" in registry.call("delete_workflow", {"name": "quick check"}, ctx)["result"]
    assert wf.find(state, "quick check") is None
    ctx.undo.undo()
    assert wf.find(state, "quick check") is not None


def test_builtins_can_be_edited_but_not_deleted(ctx, registry, state, monkeypatch):
    from jarvis import agents

    monkeypatch.setattr(agents, "_registry", [registry])
    assert "built-in" in registry.call("delete_workflow", {"name": "work mode"}, ctx)["error"]
    registry.call("edit_workflow", {"name": "work mode", "steps": [{"tool": "get_time"}]}, ctx)
    assert wf.find(state, "work mode")[1]["steps"] == [{"tool": "get_time", "args": {}}]
    assert "is back" in registry.call("delete_workflow", {"name": "work mode"}, ctx)["result"]
    assert wf.find(state, "work mode")[1]["builtin"]


def test_workflow_steps_are_checked_for_secrets(registry):
    from jarvis import guardrails

    steps = [{"tool": "add_note", "args": {"text": "my password is hunter2!"}}]
    assert guardrails.check_tool("create_workflow", {"name": "x", "steps": steps})


def test_old_routines_keep_working(ctx, registry, state):
    out = registry.call("save_routine", {"name": "work mode", "steps": ["open VS Code"]}, ctx)
    assert "Saved routine" in out["result"]
    assert registry.call("run_routine", {"name": "work mode"}, ctx)["steps"] == ["open VS Code"]
