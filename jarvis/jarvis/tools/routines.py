"""Routines: one phrase runs several actions. "Jab main 'work mode' bolu to VS Code aur Chrome kholo
aur lofi chalao" saves a routine; saying "work mode" later runs it. Steps are plain sentences that the
model carries out with its normal tools, so every safety check (confirmations etc.) still applies."""

from __future__ import annotations

from jarvis.tools import tool

KEY = "routines"
MAX_STEPS = 12
_mem: dict[str, list[str]] = {}


def _load(ctx) -> dict[str, list[str]]:
    return dict(ctx.state.get(KEY, {})) if ctx.state is not None else dict(_mem)


def _save(ctx, data: dict[str, list[str]]) -> None:
    if ctx.state is not None:
        ctx.state.set(KEY, data)
    else:
        _mem.clear()
        _mem.update(data)


def _key(name: str) -> str:
    return " ".join(name.lower().split())


def routine_names(state) -> list[str]:
    try:
        return sorted(state.get(KEY, {})) if state is not None else []
    except Exception:
        return []


@tool(
    "save_routine",
    "Save a routine: a name the user will say, and the steps to do (short sentences like "
    "'open VS Code', 'play lofi on YouTube', 'set volume to 30'). Saving again replaces it.",
    {"name": {"type": "string"},
     "steps": {"type": "array", "items": {"type": "string"}}},
    required=["name", "steps"],
)
def save_routine(ctx, name: str, steps: list):
    steps = [str(s).strip() for s in (steps or []) if str(s).strip()]
    if not name.strip() or not steps:
        return {"error": "A routine needs a name and at least one step."}
    if len(steps) > MAX_STEPS:
        return {"error": f"Keep a routine to {MAX_STEPS} steps or fewer."}
    data = _load(ctx)
    previous = data.get(_key(name))
    data[_key(name)] = steps
    _save(ctx, data)

    def revert():
        d = _load(ctx)
        if previous is None:
            d.pop(_key(name), None)
        else:
            d[_key(name)] = previous
        _save(ctx, d)

    ctx.undo.push(f"save routine {name}", revert)
    return f"Saved routine '{_key(name)}' with {len(steps)} step(s). Say '{_key(name)}' any time to run it."


@tool(
    "run_routine",
    "Run a saved routine when the user says its name ('work mode', 'good night routine').",
    {"name": {"type": "string"}},
    required=["name"],
)
def run_routine(ctx, name: str):
    data = _load(ctx)
    key = _key(name)
    steps = data.get(key)
    if steps is None:  # forgiving match: 'work' -> 'work mode'
        close = [k for k in data if key in k or k in key]
        if len(close) == 1:
            key, steps = close[0], data[close[0]]
    if steps is None:
        names = ", ".join(sorted(data)) or "none yet"
        return {"error": f"No routine called '{name}'. Saved routines: {names}."}
    numbered = " ".join(f"{i}) {s}" for i, s in enumerate(steps, 1))
    return {"routine": key, "steps": steps,
            "result": f"Now do these steps in order with your tools, without asking again: {numbered}. "
                      "Then confirm in one short sentence."}


@tool("list_routines", "List the user's saved routines and their steps.")
def list_routines(ctx):
    data = _load(ctx)
    return {"routines": data} if data else "No routines yet. Example: \"save a routine called work mode: open VS Code and Chrome\"."


@tool("delete_routine", "Delete a saved routine by name.", {"name": {"type": "string"}}, required=["name"])
def delete_routine(ctx, name: str):
    data = _load(ctx)
    steps = data.pop(_key(name), None)
    if steps is None:
        return {"error": f"No routine called '{name}'."}
    _save(ctx, data)
    ctx.undo.push(f"delete routine {name}", lambda: _save(ctx, {**_load(ctx), _key(name): steps}))
    return f"Deleted routine '{_key(name)}'."
