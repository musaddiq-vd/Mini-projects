"""Tool registry.

A tool is a plain function decorated with ``@tool``. It receives a
``ToolContext`` first, then the arguments the model sent::

    @tool("get_time", "Current local date and time.")
    def get_time(ctx):
        return datetime.now().isoformat()

Built-in tools live in ``jarvis/tools/*.py``. Your own tools go in a
``plugins/`` folder (in the repo or in JARVIS_HOME) and use the same decorator.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jarvis.config import Settings
from jarvis.confirm import ConfirmGate
from jarvis.memory import MemoryStore
from jarvis.undo import UndoStack


@dataclass
class ToolContext:
    settings: Settings
    memory: MemoryStore
    undo: UndoStack
    confirm: ConfirmGate
    notify: Callable[[str], None] = print  # show/speak a message outside a tool reply
    state: Any = None  # jarvis.state.StateStore
    events: Any = None  # jarvis.events.Events
    # Send an image (JPEG bytes) to the model with the next tool reply; None = not supported
    attach_image: Callable[[bytes], None] | None = None
    assistant: Any = None  # the running LiveAssistant (sleep/wake_up), None in text mode
    _llm: Any = None

    @property
    def llm(self):
        """Shared text model for tools that need to write or summarise text."""
        if self._llm is None:
            from jarvis.llm import TextModel

            self._llm = TextModel(self.settings)
        return self._llm


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    parameters: dict = field(default_factory=dict)  # JSON-schema "properties"
    required: list[str] = field(default_factory=list)
    slow: bool = False  # model should say a short "on it" before calling

    def schema(self) -> dict:
        return {"type": "object", "properties": self.parameters, "required": self.required}


_ATTR = "__jarvis_tool__"


def tool(
    name: str,
    description: str,
    parameters: dict | None = None,
    required: list[str] | None = None,
    slow: bool = False,
):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, _ATTR, Tool(name, description, fn, parameters or {}, required or [], slow))
        return fn

    return decorator


def _to_gemini_schema(schema: dict) -> dict:
    """JSON-schema-ish dict -> Gemini Schema dict (types are upper case)."""
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type":
            out["type"] = str(value).upper()
        elif key == "properties":
            out["properties"] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items":
            out["items"] = _to_gemini_schema(value)
        else:
            out[key] = value
    if out.get("type") == "OBJECT" and not out.get("properties"):
        out.pop("properties", None)
        out.pop("required", None)
    return out


class ToolRegistry:
    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}
        self.errors: list[str] = []

    def add(self, t: Tool) -> None:
        if t.name in self.tools:
            self.errors.append(f"duplicate tool name '{t.name}' ignored")
            return
        self.tools[t.name] = t

    def add_module(self, module) -> None:
        for value in vars(module).values():
            t = getattr(value, _ATTR, None)
            if isinstance(t, Tool):
                self.add(t)

    def load_builtins(self) -> None:
        import jarvis.tools as pkg

        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name.startswith("_"):
                continue
            try:
                module = importlib.import_module(f"jarvis.tools.{info.name}")
            except Exception as exc:
                self.errors.append(f"builtin '{info.name}': {exc}")
                continue
            self.add_module(module)

    def load_plugins(self, folders: list[Path]) -> None:
        for folder in folders:
            if not folder.is_dir():
                continue
            for file in sorted(folder.glob("*.py")):
                if file.name.startswith("_"):
                    continue
                mod_name = f"jarvis_plugin_{file.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(mod_name, file)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = module
                    spec.loader.exec_module(module)
                except Exception as exc:
                    sys.modules.pop(mod_name, None)
                    self.errors.append(f"plugin '{file.name}': {exc}")
                    continue
                self.add_module(module)

    def declarations(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description + (" (slow: say a short 'on it' first)" if t.slow else ""),
                "parameters": _to_gemini_schema(t.schema()),
            }
            for t in self.tools.values()
        ]

    def call(self, name: str, args: dict | None, ctx: ToolContext) -> dict:
        """Run a tool. Always returns a dict; errors never escape."""
        t = self.tools.get(name)
        if t is None:
            return {"error": f"Unknown tool '{name}'."}
        args = dict(args or {})
        accepted = inspect.signature(t.handler).parameters
        if not any(p.kind is p.VAR_KEYWORD for p in accepted.values()):
            args = {k: v for k, v in args.items() if k in accepted}
        missing = [r for r in t.required if r not in args]
        if missing:
            return {"error": f"Missing argument(s): {', '.join(missing)}"}
        try:
            result = t.handler(ctx, **args)
        except Exception as exc:
            traceback.print_exc()
            return {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(result, dict):
            return result
        return {"result": "Done." if result is None else str(result)}


def build_registry(plugin_dirs: list[Path] | None = None) -> ToolRegistry:
    reg = ToolRegistry()
    reg.load_builtins()
    reg.load_plugins(plugin_dirs or [])
    return reg
