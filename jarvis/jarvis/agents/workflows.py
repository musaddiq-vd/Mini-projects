"""Workflows: saved sequences of real tool calls that run the same way every time.

A routine (``tools/routines.py``) is a list of sentences the voice model reads and carries out. A workflow is
stricter: each step names a tool and its arguments, and ``run`` executes the steps itself, in order, through
``ToolRegistry.call`` (permissions, guardrails, confirmations and the audit log all apply). A step can have:

- ``tool`` + ``args``: the call; string arguments may use variables like ``{pc.battery_percent}``
- ``if``: a condition - "battery < 30", "hour >= 18", "weekday", "weekend", "charging", "online",
  "{pc.battery_percent} < 30", "{search.query} contains python", with "not ..." to turn it around
- ``wait``: seconds to wait before the step (a step may be just a wait)
- ``save_as``: a name for the step's result, used by later steps as ``{name}`` or ``{name.field}``

A run stops at the first error, or when the user declines a confirmation. Triggers: said by name ("command"),
"daily" or "weekdays" at a time, or "startup" (when Jarvis starts).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import date, datetime

log = logging.getLogger("jarvis.workflows")

KEY = "workflows"
LAST_RUN_KEY = "workflow_last_run"
MAX_STEPS = 20
MAX_WAIT_S = 600
MAX_TOTAL_WAIT_S = 1800
TRIGGERS = ("command", "daily", "weekdays", "startup")
# a workflow may not start agents, other workflows or change automations
FORBIDDEN = {"run_agents", "cancel_agents", "run_workflow", "create_workflow", "edit_workflow", "delete_workflow",
             "save_routine", "delete_routine", "run_routine", "power"}
_mem: dict[str, dict] = {}

BUILTIN: dict[str, dict] = {
    "work mode": {
        "description": "Open VS Code and Chrome, lofi music, and dark mode in the evening.",
        "trigger": {"type": "command"},
        "steps": [
            {"tool": "system_status", "save_as": "pc"},
            {"tool": "open_app", "args": {"name": "VS Code"}},
            {"tool": "open_app", "args": {"name": "Chrome"}},
            {"tool": "dark_mode", "args": {"mode": "on"}, "if": "hour >= 18"},
            {"tool": "play_youtube", "args": {"query": "lofi hip hop for coding"}},
        ],
        "builtin": True,
    },
    "study mode": {
        "description": "Close distractions, start a 25 minute focus session and show the to-do list.",
        "trigger": {"type": "command"},
        "steps": [
            {"tool": "close_app", "args": {"name": "YouTube"}},
            {"tool": "close_app", "args": {"name": "WhatsApp"}},
            {"tool": "focus_mode", "args": {"minutes": 25, "task": "study"}},
            {"tool": "list_notes", "save_as": "todo"},
        ],
        "builtin": True,
    },
}


def _key(name: str) -> str:
    return " ".join(str(name).lower().split())


# -- storage ----------------------------------------------------------------------------------------
def _saved(state) -> dict[str, dict]:
    if state is None:
        return dict(_mem)
    data = state.get(KEY, {})
    return dict(data) if isinstance(data, dict) else {}


def _store(state, data: dict[str, dict]) -> None:
    if state is None:
        _mem.clear()
        _mem.update(data)
    else:
        state.set(KEY, data)


def all_workflows(state) -> dict[str, dict]:
    """Built-in examples plus the user's own (a saved one with the same name replaces the example)."""
    out = {k: dict(v) for k, v in BUILTIN.items()}
    out.update(_saved(state))
    return out


def find(state, name: str) -> tuple[str, dict] | None:
    flows = all_workflows(state)
    key = _key(name)
    if key in flows:
        return key, flows[key]
    close = [k for k in flows if key and (key in k or k in key)]
    if len(close) == 1:
        return close[0], flows[close[0]]
    return None


def names(state) -> list[str]:
    try:
        return sorted(all_workflows(state))
    except Exception:
        return []


# -- validation -------------------------------------------------------------------------------------
_TIME = re.compile(r"^([01]?\d|2[0-3])[:.]([0-5]\d)$")


def normalise_trigger(trigger) -> dict:
    if trigger in (None, "", {}):
        return {"type": "command"}
    if isinstance(trigger, str):
        parts = trigger.lower().replace(" at ", " ").split()
        trigger = {"type": parts[0], "at": parts[1] if len(parts) > 1 else ""}
    if not isinstance(trigger, dict):
        raise ValueError("trigger must be 'command', 'startup', or daily/weekdays with a time like 09:00")
    kind = str(trigger.get("type", "command")).lower().strip()
    if kind in ("weekday", "workdays"):
        kind = "weekdays"
    if kind not in TRIGGERS:
        raise ValueError(f"unknown trigger '{kind}'; use {', '.join(TRIGGERS)}")
    out = {"type": kind}
    if kind in ("daily", "weekdays"):
        m = _TIME.match(str(trigger.get("at", "")).strip())
        if not m:
            raise ValueError(f"a {kind} trigger needs a time like 09:00")
        out["at"] = f"{int(m.group(1)):02d}:{m.group(2)}"
    return out


def validate(steps, registry) -> list[dict]:
    """Clean steps, or ValueError with a message the model can fix. Every tool must exist in the registry."""
    if not isinstance(steps, list) or not steps:
        raise ValueError("a workflow needs at least one step")
    if len(steps) > MAX_STEPS:
        raise ValueError(f"keep a workflow to {MAX_STEPS} steps or fewer")
    clean: list[dict] = []
    total_wait = 0.0
    saved: set[str] = set()
    for i, raw in enumerate(steps, 1):
        if isinstance(raw, str):
            raise ValueError(f"step {i} is a sentence; each step must be {{'tool': name, 'args': {{...}}}} "
                             "(sentences belong in save_routine)")
        if not isinstance(raw, dict):
            raise ValueError(f"step {i} must be an object")
        step: dict = {}
        if raw.get("wait") not in (None, "", 0):
            try:
                wait = float(raw["wait"])
            except (TypeError, ValueError):
                raise ValueError(f"step {i}: wait must be a number of seconds") from None
            if not 0 < wait <= MAX_WAIT_S:
                raise ValueError(f"step {i}: wait must be between 1 and {MAX_WAIT_S} seconds")
            total_wait += wait
            step["wait"] = wait
        name = str(raw.get("tool") or "").strip()
        if name:
            if name in FORBIDDEN:
                raise ValueError(f"step {i}: '{name}' can't be used inside a workflow")
            t = registry.tools.get(name)
            if t is None:
                raise ValueError(f"step {i}: there is no tool called '{name}'")
            args = raw.get("args") or {}
            if not isinstance(args, dict):
                raise ValueError(f"step {i}: args must be an object")
            unknown = [a for a in args if a not in t.parameters]
            if unknown and t.parameters:
                raise ValueError(f"step {i}: {name} has no argument(s) {', '.join(unknown)}; "
                                 f"it takes {', '.join(t.parameters)}")
            missing = [r for r in t.required if r not in args]
            if missing:
                raise ValueError(f"step {i}: {name} needs {', '.join(missing)}")
            for ref in _refs(args):
                if ref.split(".")[0] not in saved:
                    raise ValueError(f"step {i}: variable {{{ref}}} isn't saved by an earlier step (use save_as)")
            step["tool"], step["args"] = name, args
        elif "wait" not in step:
            raise ValueError(f"step {i} needs a tool or a wait")
        if raw.get("if"):
            cond = str(raw["if"]).strip()
            parse_condition(cond)  # raises on a bad condition
            step["if"] = cond
        if raw.get("save_as"):
            var = str(raw["save_as"]).strip()
            if not re.fullmatch(r"[A-Za-z_]\w{0,30}", var):
                raise ValueError(f"step {i}: save_as must be a simple name like 'pc'")
            if not name:
                raise ValueError(f"step {i}: only a tool step can save_as")
            step["save_as"] = var
            saved.add(var)
        clean.append(step)
    if total_wait > MAX_TOTAL_WAIT_S:
        raise ValueError(f"waits add up to more than {MAX_TOTAL_WAIT_S // 60} minutes")
    return clean


# -- variables and conditions --------------------------------------------------------------------------
_REF = re.compile(r"\{([A-Za-z_]\w*(?:\.\w+)*)\}")


def _refs(value) -> list[str]:
    if isinstance(value, str):
        return _REF.findall(value)
    if isinstance(value, dict):
        return [r for v in value.values() for r in _refs(v)]
    if isinstance(value, list):
        return [r for v in value for r in _refs(v)]
    return []


def lookup(variables: dict, ref: str):
    parts = ref.split(".")
    value = variables.get(parts[0])
    for part in parts[1:]:
        if isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def resolve(value, variables: dict):
    """Put saved results into arguments: "{pc.battery_percent}" alone keeps its type; inside text it's text."""
    if isinstance(value, str):
        whole = _REF.fullmatch(value.strip())
        if whole:
            found = lookup(variables, whole.group(1))
            return "" if found is None else found
        return _REF.sub(lambda m: "" if lookup(variables, m.group(1)) is None
                        else str(lookup(variables, m.group(1)))[:2000], value)
    if isinstance(value, dict):
        return {k: resolve(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, variables) for v in value]
    return value


_COND = re.compile(r"^(.+?)\s*(<=|>=|==|!=|<|>|=| contains | is )\s*(.+)$", re.IGNORECASE)
FLAGS = ("weekday", "weekend", "charging", "online", "offline", "morning", "afternoon", "evening", "night")
FACTS = ("battery", "charging", "hour", "minute", "time", "day")


def parse_condition(text: str) -> tuple[bool, str, str | None, str | None]:
    """(negated, left, operator, right). ValueError when it can't be understood."""
    t = " ".join(str(text).strip().lower().split())
    t = t.removeprefix("if ").strip()
    negated = t.startswith("not ")
    if negated:
        t = t[4:].strip()
    if t in FLAGS or re.fullmatch(r"\{[\w.]+\}", t):
        return negated, t, None, None
    m = _COND.match(t)
    if not m:
        raise ValueError(f"I can't read the condition '{text}'. Examples: 'battery < 30', 'hour >= 18', "
                         "'weekday', '{pc.charging} == false'")
    left, op, right = m.group(1).strip(), m.group(2).strip(), m.group(3).strip().strip("'\"")
    op = {"=": "==", "is": "=="}.get(op, op)
    if left not in FACTS and not re.fullmatch(r"\{[\w.]+\}", left):
        raise ValueError(f"unknown value '{left}' in condition; use battery, hour, time, day or a {{variable}}")
    return negated, left, op, right


def current_facts() -> dict:
    """What conditions can test, read now."""
    now = datetime.now()
    facts = {"hour": now.hour, "minute": now.minute, "time": now.strftime("%H:%M"),
             "day": now.strftime("%A").lower(), "weekday": now.weekday() < 5}
    try:
        import psutil

        battery = psutil.sensors_battery()
        if battery is not None:
            facts["battery"] = round(battery.percent)
            facts["charging"] = bool(battery.power_plugged)
    except Exception:
        pass
    return facts


def _online() -> bool:
    import socket

    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2).close()
        return True
    except OSError:
        return False


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate(text: str, variables: dict, facts: dict) -> bool:
    negated, left, op, right = parse_condition(text)
    if op is None:
        if left.startswith("{"):
            value = bool(lookup(variables, left[1:-1]))
        elif left == "weekday":
            value = bool(facts.get("weekday"))
        elif left == "weekend":
            value = not facts.get("weekday", True)
        elif left == "charging":
            value = bool(facts.get("charging"))
        elif left in ("online", "offline"):
            up = facts["online"] if "online" in facts else _online()
            value = up if left == "online" else not up
        else:
            hour = int(facts.get("hour", 0))
            value = {"morning": 5 <= hour < 12, "afternoon": 12 <= hour < 17, "evening": 17 <= hour < 21,
                     "night": hour >= 21 or hour < 5}[left]
        return value != negated
    actual = lookup(variables, left[1:-1]) if left.startswith("{") else facts.get(left)
    if actual is None:
        result = False  # unknown (no battery on a desktop): the step is skipped
    elif op == "contains":
        result = right in str(actual).lower()
    else:
        a, b = _num(actual), _num(right)
        if left == "time":
            a, b = str(actual), right.zfill(5)
        elif a is None or b is None:
            a, b = str(actual).lower(), right
            if op not in ("==", "!="):
                return False
        result = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[op]
    return result != negated


# -- running ------------------------------------------------------------------------------------------
_active: list = []
_lock = threading.Lock()


def running():
    with _lock:
        return _active[0] if _active else None


def _short(result) -> str:
    if not isinstance(result, dict):
        return str(result)[:160]
    for key in ("error", "result", "message"):
        if result.get(key):
            return str(result[key])[:160]
    return ", ".join(f"{k}: {str(v)[:40]}" for k, v in list(result.items())[:3] if not k.startswith("_"))[:160]


def run(ctx, name: str, registry, sleep: Callable[[float], None] | None = None, facts: dict | None = None,
        trigger: str = "command") -> dict:
    """Run a workflow now, step by step. Returns the log and the final status (never raises)."""
    from jarvis.agents import team

    found = find(getattr(ctx, "state", None), name)
    if found is None:
        return {"error": f"No workflow called '{name}'. Workflows: {', '.join(names(ctx.state)) or 'none'}."}
    key, flow = found
    try:
        steps = validate(flow.get("steps"), registry)
    except ValueError as exc:
        return {"error": f"Workflow '{key}' can't run: {exc}. Fix it with edit_workflow."}
    with _lock:
        if _active and _active[0].active:
            return {"error": "Another workflow is running; wait for it or say stop."}
        live = team.Run(goal=key, kind="workflow", status=team.RUNNING, max_steps=len(steps),
                        max_tool_calls=len(steps), budget_s=MAX_TOTAL_WAIT_S + 300)
        live.steps = [team.Step(i, "workflow", _describe(s)) for i, s in enumerate(steps, 1)]
        _active[:] = [live]
    team.watch_stop(getattr(ctx, "events", None))
    facts = facts if facts is not None else current_facts()
    sleep = sleep or (lambda s: live.cancel_event.wait(s))
    variables: dict = {}
    log_lines: list[dict] = []
    status = "done"
    team.publish(ctx, live)
    for i, (step, view) in enumerate(zip(steps, live.steps), 1):
        if live.cancel_event.is_set():
            status = "cancelled"
            break
        view.status = team.RUNNING
        team.publish(ctx, live)
        try:
            cond = step.get("if")
            if cond and not evaluate(cond, variables, facts):
                view.status, view.output = team.SKIPPED, f"skipped (not {cond})"
                log_lines.append({"step": i, "status": "skipped", "why": cond})
                continue
        except ValueError as exc:
            view.status, view.output = team.FAILED, str(exc)
            log_lines.append({"step": i, "status": "error", "result": str(exc)})
            status = "failed"
            break
        if step.get("wait"):
            sleep(step["wait"])
            if live.cancel_event.is_set():
                view.status = team.CANCELLED
                status = "cancelled"
                break
        if not step.get("tool"):
            view.status, view.output = team.DONE, f"waited {step['wait']:g} s"
            log_lines.append({"step": i, "status": "waited", "seconds": step["wait"]})
            continue
        args = resolve(step.get("args", {}), variables)
        result = registry.call(step["tool"], args, ctx)
        live.tool_calls += 1
        view.tools.append(step["tool"])
        view.output = _short(result)
        if step.get("save_as"):
            variables[step["save_as"]] = result
        declined = isinstance(result, dict) and str(result.get("result", "")).startswith("Cancelled - the user did not")
        if isinstance(result, dict) and "error" in result:
            view.status = team.FAILED
            log_lines.append({"step": i, "tool": step["tool"], "status": "error", "result": view.output})
            status = "failed"
            break
        if declined:
            view.status = team.CANCELLED
            log_lines.append({"step": i, "tool": step["tool"], "status": "declined"})
            status = "declined"
            break
        view.status = team.DONE
        log_lines.append({"step": i, "tool": step["tool"], "status": "ok", "result": view.output})
    _finish(live)
    for view in live.steps:
        if view.status in (team.WAITING, team.RUNNING):
            view.status = team.CANCELLED if status == "cancelled" else team.SKIPPED
    live.status = {"done": team.DONE, "cancelled": team.CANCELLED}.get(status, team.FAILED)
    live.finished = time.monotonic()
    words = {"done": "finished", "failed": "stopped at an error", "declined": "stopped because you said no",
             "cancelled": "was cancelled"}[status]
    ran = sum(1 for x in log_lines if x["status"] == "ok")
    live.summary = f"Workflow '{key}' {words} ({ran} of {len(steps)} step(s) ran)."
    team.publish(ctx, live)
    _remember_run(ctx, key, status)
    out = {"workflow": key, "status": status, "trigger": trigger, "log": log_lines, "result": live.summary}
    if status == "failed":
        out["error"] = f"{live.summary} {log_lines[-1].get('result', '')}".strip()
    return out


def _finish(live) -> None:
    with _lock:
        if _active and _active[0] is live:
            _active.clear()


def cancel() -> bool:
    """Stop the running workflow (``cancel_agents`` and a spoken "stop" call this too)."""
    live = running()
    if live is None or not live.active:
        return False
    live.cancel_event.set()
    return True


def _describe(step: dict) -> str:
    parts = []
    if step.get("wait"):
        parts.append(f"wait {step['wait']:g} s")
    if step.get("tool"):
        args = ", ".join(f"{k}={v}" for k, v in (step.get("args") or {}).items())
        parts.append(f"{step['tool']}({args})" if args else step["tool"])
    if step.get("if"):
        parts.append(f"if {step['if']}")
    return " · ".join(parts)


def _remember_run(ctx, key: str, status: str) -> None:
    state = getattr(ctx, "state", None)
    if state is None:
        return
    try:
        last = dict(state.get(LAST_RUN_KEY, {}) or {})
        last[key] = {"date": date.today().isoformat(), "status": status}
        state.set(LAST_RUN_KEY, last)
    except Exception:
        log.exception("couldn't save the workflow run time")


# -- triggers ------------------------------------------------------------------------------------------
def due(state, now: datetime | None = None) -> list[str]:
    """Workflows whose daily / weekdays time has come and that haven't run today (10 minute window)."""
    now = now or datetime.now()
    last = dict((state.get(LAST_RUN_KEY, {}) if state is not None else {}) or {})
    out = []
    for key, flow in all_workflows(state).items():
        trig = flow.get("trigger") or {}
        if trig.get("type") not in ("daily", "weekdays") or not trig.get("at"):
            continue
        if trig["type"] == "weekdays" and now.weekday() >= 5:
            continue
        hh, mm = (int(x) for x in trig["at"].split(":"))
        minutes = (now.hour * 60 + now.minute) - (hh * 60 + mm)
        if 0 <= minutes <= 10 and (last.get(key) or {}).get("date") != now.date().isoformat():
            out.append(key)
    return out


def at_startup(state) -> list[str]:
    return [k for k, f in all_workflows(state).items() if (f.get("trigger") or {}).get("type") == "startup"]


class Scheduler:
    """Runs time-triggered workflows, and startup ones once, on a quiet background thread."""

    def __init__(self, ctx, registry, notify: Callable[[str], None] | None = None, tick_s: float = 30.0):
        self.ctx, self.registry, self.tick_s = ctx, registry, tick_s
        self.notify = notify or ctx.notify
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._loop, name="workflows", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self, key: str, trigger: str) -> None:
        out = run(self.ctx, key, self.registry, trigger=trigger)
        self.notify(f"Workflow '{key}' ran on its {trigger} trigger: {out.get('result') or out.get('error')}")

    def tick(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now()
        ran = due(self.ctx.state, now)
        for key in ran:
            self._run(key, "scheduled")
            state = self.ctx.state
            if state is not None:  # count the run on the day it was due (matters just after midnight, and in tests)
                last = dict(state.get(LAST_RUN_KEY, {}) or {})
                last[key] = {**(last.get(key) or {}), "date": now.date().isoformat()}
                state.set(LAST_RUN_KEY, last)
        return ran

    def _loop(self) -> None:
        if self._stop.wait(20):  # let Jarvis finish starting first
            return
        for key in at_startup(self.ctx.state):
            try:
                self._run(key, "startup")
            except Exception:
                log.exception("startup workflow %s failed", key)
        while not self._stop.wait(self.tick_s):
            try:
                self.tick()
            except Exception:  # one bad tick must not end the loop
                log.exception("workflow scheduler tick failed")


# -- editing (used by the tools) ------------------------------------------------------------------------
def save(state, name: str, flow: dict) -> dict | None:
    """Store a workflow. Returns the previous version (for undo)."""
    data = _saved(state)
    key = _key(name)
    previous = data.get(key)
    data[key] = flow
    _store(state, data)
    return previous


def remove(state, name: str) -> dict | None:
    data = _saved(state)
    flow = data.pop(_key(name), None)
    if flow is not None:
        _store(state, data)
    return flow


def restore(state, name: str, flow: dict | None) -> None:
    data = _saved(state)
    if flow is None:
        data.pop(_key(name), None)
    else:
        data[_key(name)] = flow
    _store(state, data)


# public names for the tools module
key = _key
saved = _saved
describe = _describe
