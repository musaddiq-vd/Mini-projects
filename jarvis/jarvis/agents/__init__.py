"""Bigger jobs than one tool call: a small team of agents, workflows and a careful web researcher.

- ``team``: a Planner splits a request into steps and hands each to a specialist (Researcher, Coder, System,
  Writer). Every specialist is a text-model call with its own instructions and an allow-list of tools.
- ``workflows``: saved, deterministic tool-call sequences with conditions, waits, variables and triggers.
- ``browser``: reads web pages politely (robots.txt, timeouts, size caps) and writes a sourced report.

Every tool call any of these make goes through ``ToolRegistry.call``, so permissions, guardrails,
confirmations and the audit log apply exactly as they do for the voice model.
"""

from __future__ import annotations

import threading

_registry: list = []
_lock = threading.Lock()


def bind_registry(registry) -> None:
    """Called once by ``create_app`` so the agents use the same tools (and plugins) as the voice model."""
    with _lock:
        _registry[:] = [registry]


def get_registry():
    """The app's tool registry; a fresh one when no app bound it (the CLI or a test)."""
    with _lock:
        if _registry:
            return _registry[0]
    from jarvis.tools import build_registry

    registry = build_registry()
    bind_registry(registry)
    return registry
