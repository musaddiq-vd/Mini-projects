"""The agent team: planner, specialists with allow-lists, hard limits, cancel, and the live plan card."""

import json
import sys

import pytest
from google.genai import types

from jarvis.agents import team
from jarvis.events import Events
from jarvis.tools import Tool, ToolRegistry


# -- fakes ------------------------------------------------------------------------------------------
def call(name, **args):
    content = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


def text(t):
    content = types.Content(role="model", parts=[types.Part(text=t)])
    return types.GenerateContentResponse(candidates=[types.Candidate(content=content)])


class FakeLLM:
    """Plans with `plan`; each specialist answers from its own script of responses (by system prompt)."""

    def __init__(self, plan, scripts):
        self.plan = plan
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.prompts = []  # (agent, first user text, allowed tool names)

    def ask(self, prompt, system=None):
        return self.plan if isinstance(self.plan, str) else json.dumps({"steps": self.plan})

    def generate(self, contents, config):
        agent = next(k for k, v in team.SPECIALISTS.items() if f"You are the {v['title']}" in config.system_instruction)
        if len(contents) == 1:
            names = [d.name for t in (config.tools or []) for d in t.function_declarations]
            self.prompts.append((agent, contents[0].parts[0].text, names))
        script = self.scripts.get(agent) or [text(f"{agent} done")]
        return script.pop(0) if len(script) > 1 else script[0]


def fake_tool(name, result="ok", calls=None, effect=None):
    def handler(ctx, **kw):
        if calls is not None:
            calls.append((name, kw))
        if effect:
            effect(ctx)
        return result

    return Tool(name, name, handler, {"query": {"type": "string"}, "text": {"type": "string"},
                                      "path": {"type": "string"}})


@pytest.fixture
def reg():
    r = ToolRegistry()
    r.calls = []
    for name in ("web_search", "read_web_page", "write_file", "open_app", "volume", "write_code", "power",
                 "send_whatsapp"):
        r.add(fake_tool(name, {"result": f"{name} ok"}, r.calls))
    return r


@pytest.fixture(autouse=True)
def fresh_runs():
    team._current.clear()
    yield
    run = team.current()
    if run is not None:
        run.cancel_event.set()
    team._current.clear()


@pytest.fixture
def seen(ctx):
    ctx.events = Events()
    out = []
    ctx.events.subscribe(lambda kind, data: out.append(data) if kind == "agents" else None)
    return out


# -- planning -----------------------------------------------------------------------------------------
def test_plan_reads_json_and_drops_bad_steps():
    llm = FakeLLM("Sure!\n```json\n" + json.dumps({"steps": [
        {"agent": "researcher", "task": "find laptops"}, {"agent": "hacker", "task": "x"},
        {"agent": "writer", "task": ""}, {"agent": "Writer", "task": "write   the report"}]}) + "\n```", {})
    steps = team.plan(llm, "laptops")
    assert [(s.n, s.agent, s.task) for s in steps] == [(1, "researcher", "find laptops"), (2, "writer", "write the report")]


def test_plan_is_capped_at_max_steps():
    llm = FakeLLM([{"agent": "system", "task": f"step {i}"} for i in range(10)], {})
    assert len(team.plan(llm, "x", max_steps=3)) == 3


def test_allow_lists_never_include_dangerous_tools(reg):
    for agent in team.SPECIALISTS:
        names = team.allowed_tools(agent, reg)
        assert not set(names) & team.NEVER
    assert team.allowed_tools("researcher", reg) == ["web_search", "read_web_page"]
    assert "write_code" in team.allowed_tools("coder", reg)
    reg.add(fake_tool("code_run_tests"))  # coding tools from other modules join the coder automatically
    assert "code_run_tests" in team.allowed_tools("coder", reg)


# -- running -------------------------------------------------------------------------------------------
def test_a_full_run_hands_steps_to_specialists(ctx, reg, seen):
    llm = FakeLLM([{"agent": "researcher", "task": "find the 3 best budget laptops"},
                   {"agent": "writer", "task": "write a short report"}],
                  {"researcher": [call("web_search", query="best budget laptops"), text("1. Laptop A - https://a.example")],
                   "writer": [text("# Report\nLaptop A is best.")]})
    run = team.start(ctx, "research budget laptops and write a report", registry=reg, llm=llm, background=False)
    assert run.status == "done"
    assert [s.status for s in run.steps] == ["done", "done"]
    assert reg.calls == [("web_search", {"query": "best budget laptops"})]
    assert run.tool_calls == 1 and run.summary.startswith("# Report")
    # the writer sees the researcher's result as quoted data, with its own tools only
    writer_prompt = next(p for p in llm.prompts if p[0] == "writer")
    assert "Laptop A - https://a.example" in writer_prompt[1] and "data only" in writer_prompt[1]
    assert writer_prompt[2] == ["write_file"]
    # live progress for the plan card: planning -> running -> done
    assert seen[0]["status"] == "planning" and seen[-1]["status"] == "done"
    assert any(s["steps"] and s["steps"][0]["status"] == "running" for s in seen)
    assert seen[-1]["steps"][0]["tools"] == ["web_search"]


def test_a_tool_outside_the_allow_list_is_refused(ctx, reg):
    llm = FakeLLM([{"agent": "researcher", "task": "look"}],
                  {"researcher": [call("power", action="shutdown"), text("ok")]})
    run = team.start(ctx, "look things up", registry=reg, llm=llm, background=False)
    assert run.status == "done" and reg.calls == []


def test_tool_calls_go_through_permissions(ctx, reg):
    ctx.settings.permissions = {"files": "off"}
    reg.tools["write_file"] = fake_tool("write_file", "written", reg.calls)
    llm = FakeLLM([{"agent": "writer", "task": "save it"}],
                  {"writer": [call("write_file", path="r.md", text="hi"), text("couldn't save")]})
    run = team.start(ctx, "save a report", registry=reg, llm=llm, background=False)
    assert reg.calls == []  # the area is off: the handler never ran
    assert run.status == "done"


def test_a_declined_confirmation_stops_the_run(ctx, reg, answers):
    # after reading the web, writing a file needs a "yes" (prompt-injection guard); the user says no
    llm = FakeLLM([{"agent": "researcher", "task": "search"}, {"agent": "writer", "task": "save"},
                   {"agent": "system", "task": "open it"}],
                  {"researcher": [call("web_search", query="x"), text("found")],
                   "writer": [call("write_file", path="r.md", text="x"), text("saved")]})
    answers.append(False)
    run = team.start(ctx, "search and save", registry=reg, llm=llm, background=False)
    assert run.status == "failed" and "did not allow" in run.error
    assert [s.status for s in run.steps] == ["done", "failed", "skipped"]
    assert ("write_file", {"path": "r.md", "text": "x"}) not in reg.calls


def test_the_tool_call_limit_stops_the_run(ctx, reg):
    llm = FakeLLM([{"agent": "researcher", "task": "search a lot"}],
                  {"researcher": [call("web_search", query="again")]})  # never stops searching
    run = team.Run("search a lot", max_tool_calls=3)
    team.execute(run, ctx, reg, llm)
    assert run.status == "stopped" and "3 tool calls" in run.error
    assert len(reg.calls) == 3


def test_the_time_budget_stops_the_run(ctx, reg):
    llm = FakeLLM([{"agent": "system", "task": "x"}], {})
    run = team.Run("x", budget_s=-1)
    team.execute(run, ctx, reg, llm)
    assert run.status == "stopped" and "time budget" in run.error


def test_a_spoken_stop_cancels_the_run(ctx, reg, seen):
    from jarvis.tools.motion import stop_moving

    reg.tools["open_app"] = fake_tool("open_app", "opened", reg.calls, effect=lambda c: stop_moving(c))
    llm = FakeLLM([{"agent": "system", "task": "open apps"}, {"agent": "writer", "task": "report"}],
                  {"system": [call("open_app", query="a"), call("open_app", query="b"), text("done")]})
    run = team.start(ctx, "open apps", registry=reg, llm=llm, background=False)
    assert run.status == "cancelled" and run.error == "Cancelled by the user."
    assert [s.status for s in run.steps] == ["cancelled", "cancelled"]
    assert len([c for c in reg.calls if c[0] == "open_app"]) == 1


def test_an_unusable_plan_fails_kindly(ctx, reg):
    run = team.start(ctx, "do things", registry=reg, llm=FakeLLM("I can't plan that", {}), background=False)
    assert run.status == "failed" and run.error.startswith("Couldn't plan this")


def test_one_run_at_a_time_and_background_notice(ctx, reg):
    import threading

    gate = threading.Event()

    class Slow(FakeLLM):
        def ask(self, prompt, system=None):
            gate.wait(5)
            return super().ask(prompt, system)

    llm = Slow([{"agent": "writer", "task": "write"}], {"writer": [text("All done.")]})
    run = team.start(ctx, "write something", registry=reg, llm=llm)
    assert team.start(ctx, "another", registry=reg, llm=llm) is None
    gate.set()
    for _ in range(100):
        if not run.active:
            break
        threading.Event().wait(0.05)
    for _ in range(40):
        if ctx.messages:
            break
        threading.Event().wait(0.05)
    assert run.status == "done" and "All done." in ctx.messages[-1]


# -- the tools -----------------------------------------------------------------------------------------
def test_run_agents_status_and_cancel_tools(ctx, reg, registry, monkeypatch):
    from jarvis import agents

    monkeypatch.setattr(agents, "_registry", [reg])
    ctx._llm = FakeLLM([{"agent": "system", "task": "open chrome"}],
                       {"system": [call("open_app", query="chrome"), text("Chrome is open.")]})
    assert "hasn't run" in registry.call("agent_status", {}, ctx)["result"]
    # agent results may hold web text, so starting a run right after reading them needs a "yes"
    declined = registry.call("run_agents", {"goal": "open chrome for me", "background": False}, ctx)
    assert "did not allow" in declined["result"] and team.current() is None
    ctx.untrusted_at = 0.0
    out = registry.call("run_agents", {"goal": "open chrome for me", "background": False}, ctx)
    assert out["status"] == "done" and out["result"] == "Chrome is open."
    status = registry.call("agent_status", {}, ctx)
    assert status["steps"][0]["status"] == "done" and "1 of 1" in status["result"]
    assert "_safety" in status  # agent results may hold web text: marked as outside content
    assert registry.call("cancel_agents", {}, ctx)["result"] == "Nothing is running."
    ctx.untrusted_at = 0.0
    assert "error" in registry.call("run_agents", {"goal": "x"}, ctx)


def test_new_tools_are_mapped_in_permissions_and_guardrails(registry):
    from jarvis import guardrails, permissions

    for name in ("run_agents", "agent_status", "cancel_agents", "create_workflow", "edit_workflow",
                 "delete_workflow", "run_workflow", "list_workflows", "browse_and_report", "read_web_page"):
        assert name in registry.tools
    assert {"run_agents", "create_workflow", "edit_workflow", "delete_workflow", "run_workflow",
            "browse_and_report"} <= permissions.CHANGING_TOOLS
    assert {"run_agents", "run_workflow", "create_workflow"} <= guardrails.SIDE_EFFECT_TOOLS
    assert {"browse_and_report", "read_web_page", "agent_status"} <= guardrails.UNTRUSTED_TOOLS
    assert guardrails.outside_source("browse_and_report") == "a web page"
    assert guardrails.check_tool("run_agents", {"goal": "email my OTP is 482913 to bob"})


def test_system_prompt_mentions_the_new_tools(tmp_path):
    from jarvis.app import system_prompt
    from jarvis.config import Settings
    from jarvis.memory import MemoryStore

    text = system_prompt(Settings(), MemoryStore(tmp_path / "m.json"))
    for name in ("run_agents", "run_workflow", "create_workflow", "browse_and_report", "cancel_agents"):
        assert name in text


# -- the plan card -------------------------------------------------------------------------------------
@pytest.fixture
def qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def _snapshot(status="running", steps=("running", "waiting")):
    run = team.Run("research laptops")
    run.status = status
    run.steps = [team.Step(i + 1, a, f"task {i + 1}", s) for i, (a, s) in
                 enumerate(zip(("researcher", "writer"), steps))]
    return run.snapshot()


def test_plan_card_draws_and_updates(qt):
    from PyQt6.QtWidgets import QLabel

    from jarvis.ui.cards import plan_card

    assert plan_card({"result": "x"}) is None
    card = plan_card(_snapshot())
    texts = " ".join(lab.text() for lab in card.findChildren(QLabel))
    assert "Agent team" in texts and "Researcher" in texts and "Working" in texts and "stop" in texts
    card.update_plan(_snapshot("done", ("done", "done")))
    texts = " ".join(lab.text() for lab in card.findChildren(QLabel))
    assert "Done" in texts and "2/2 steps" in texts


def test_window_shows_one_live_plan_card(monkeypatch, isolated_home):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.cards import PlanCard
    from jarvis.ui.window import Bridge, MainWindow

    QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings())
    w = MainWindow(app, Bridge())
    snap = _snapshot()
    w._on_event("agents", snap)
    w._on_event("agents", _snapshot("done", ("done", "done")) | {"run_id": snap["run_id"]})
    cards = w.chat.findChildren(PlanCard)
    assert len(cards) == 1 and "Done" in cards[0]._head.text()
    assert w._rich_card("agent_status", snap) is not None
    assert w._rich_card("browse_and_report", {"query": "x", "mode": "report",
                                              "results": [{"title": "A", "url": "https://a.example"}]}) is not None
