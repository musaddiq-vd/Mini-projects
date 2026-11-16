"""Voice tools for workflows: saved tool-call sequences with conditions, waits, variables and triggers.

See ``jarvis.agents.workflows`` for the step format. Old routines (``routines.py``) keep working as before.
"""

from __future__ import annotations

from jarvis.agents import workflows as wf
from jarvis.tools import tool

STEP_HELP = ("steps: a list of objects, each {'tool': '<tool name>', 'args': {...}} with optional 'if' "
             "('battery < 30', 'hour >= 18', 'weekday', '{pc.charging} == false'), 'wait' (seconds before the "
             "step) and 'save_as' (name the result; later args can use '{name.field}'). A step can be only a wait.")
TRIGGER_HELP = ("trigger: 'command' (default: run when the user says the name), 'startup' (when Jarvis starts), "
                "or {'type': 'daily' or 'weekdays', 'at': 'HH:MM'}.")
STEP_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "tool": {"type": "string"}, "args": {"type": "object"}, "if": {"type": "string"},
    "wait": {"type": "number"}, "save_as": {"type": "string"}}}}
TRIGGER_SCHEMA = {"type": "object", "properties": {
    "type": {"type": "string", "enum": list(wf.TRIGGERS)}, "at": {"type": "string", "description": "HH:MM"}}}


def _registry():
    from jarvis.agents import get_registry

    return get_registry()


def _summary(key: str, flow: dict) -> dict:
    trig = flow.get("trigger") or {"type": "command"}
    when = trig["type"] + (f" at {trig['at']}" if trig.get("at") else "")
    return {"name": key, "trigger": when, "steps": [wf.describe(s) for s in flow.get("steps", [])],
            "built_in": bool(flow.get("builtin")), "description": flow.get("description", "")}


def _store(ctx, name: str, steps, trigger, description: str, verb: str):
    try:
        clean = wf.validate(steps, _registry())
        trig = wf.normalise_trigger(trigger)
    except ValueError as exc:
        return {"error": f"Can't {verb} that workflow: {exc}"}
    key = wf.key(name)
    if not key:
        return {"error": "A workflow needs a name."}
    flow = {"steps": clean, "trigger": trig, "description": str(description or "")[:200]}
    previous = wf.save(ctx.state, key, flow)
    ctx.undo.push(f"{verb} workflow {key}", lambda: wf.restore(ctx.state, key, previous))
    when = {"command": f"Say '{key}' to run it.", "startup": "It runs when Jarvis starts."}.get(
        trig["type"], f"It runs {trig['type']} at {trig.get('at')}.")
    return {"workflow": _summary(key, flow),
            "result": f"{'Created' if verb == 'create' else 'Changed'} workflow '{key}' with {len(clean)} step(s). "
                      f"{when}"}


@tool(
    "create_workflow",
    "Create a workflow: a named list of real tool calls Jarvis runs exactly, in order, stopping on an error. "
    "Better than a routine when the user wants conditions, waits, a schedule or values passed between steps. "
    + STEP_HELP + " " + TRIGGER_HELP,
    {"name": {"type": "string"}, "steps": STEP_SCHEMA, "trigger": TRIGGER_SCHEMA,
     "description": {"type": "string"}},
    required=["name", "steps"],
)
def create_workflow(ctx, name: str, steps: list, trigger=None, description: str = ""):
    if wf.key(name) in wf.saved(ctx.state):
        return {"error": f"A workflow called '{wf.key(name)}' already exists; use edit_workflow to change it."}
    return _store(ctx, name, steps, trigger, description, "create")


@tool(
    "edit_workflow",
    "Change a workflow: give the whole new list of steps and/or a new trigger. Editing a built-in example "
    "(work mode, study mode) saves your own version. " + STEP_HELP,
    {"name": {"type": "string"}, "steps": STEP_SCHEMA, "trigger": TRIGGER_SCHEMA,
     "description": {"type": "string"}},
    required=["name"],
)
def edit_workflow(ctx, name: str, steps: list | None = None, trigger=None, description: str | None = None):
    found = wf.find(ctx.state, name)
    if found is None:
        return {"error": f"No workflow called '{name}'. Workflows: {', '.join(wf.names(ctx.state))}."}
    key, flow = found
    return _store(ctx, key, steps if steps else flow.get("steps"), trigger or flow.get("trigger"),
                  flow.get("description", "") if description is None else description, "edit")


@tool(
    "run_workflow",
    "Run a workflow now when the user says its name ('work mode', 'study mode'). Jarvis carries out every step "
    "itself and returns a step-by-step log; just tell the user how it went in a sentence.",
    {"name": {"type": "string"}},
    required=["name"],
    slow=True,
)
def run_workflow(ctx, name: str):
    return wf.run(ctx, name, _registry())


@tool("list_workflows", "List workflows (built-in examples and the user's own) with their triggers and steps.")
def list_workflows(ctx):
    flows = wf.all_workflows(ctx.state)
    return {"workflows": [_summary(k, f) for k, f in sorted(flows.items())],
            "result": f"{len(flows)} workflow(s): {', '.join(sorted(flows))}."}


@tool("delete_workflow", "Delete one of the user's workflows by name.", {"name": {"type": "string"}},
      required=["name"])
def delete_workflow(ctx, name: str):
    key = wf.key(name)
    flow = wf.remove(ctx.state, key)
    if flow is None:
        if key in wf.BUILTIN:
            return {"error": f"'{key}' is a built-in example and can't be deleted; edit_workflow can change it."}
        return {"error": f"No workflow of yours called '{name}'."}
    ctx.undo.push(f"delete workflow {key}", lambda: wf.restore(ctx.state, key, flow))
    back = " The built-in example of that name is back." if key in wf.BUILTIN else ""
    return f"Deleted workflow '{key}'.{back}"
