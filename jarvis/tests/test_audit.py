"""The audit log: every tool call is written down, secrets never are."""

from jarvis import audit
from jarvis.tools import Tool, ToolRegistry


def _registry():
    reg = ToolRegistry()
    reg.add(Tool("echo", "Echo.", lambda ctx, text="": text, {"text": {"type": "string"}}))
    reg.add(Tool("boom", "Fails.", lambda ctx: 1 / 0))
    reg.add(Tool("ask", "Asks.", lambda ctx: "Cancelled - the user did not confirm."))
    return reg


def test_every_call_is_recorded(ctx):
    reg = _registry()
    reg.call("echo", {"text": "hi"}, ctx)
    reg.call("boom", {}, ctx)
    reg.call("ask", {}, ctx)
    reg.call("nope", {}, ctx)
    got = audit.recent(10)
    assert [e["tool"] for e in got] == ["echo", "boom", "ask", "nope"]
    assert [e["outcome"] for e in got] == ["ok", "error", "cancelled", "error"]
    assert got[0]["args"] == {"text": "hi"} and got[0]["source"] == "pc"
    assert "ZeroDivisionError" in got[1]["detail"]


def test_secrets_are_masked_and_values_cut(ctx):
    reg = _registry()
    reg.call("echo", {"text": "my card 4111 1111 1111 1111 " + "x" * 500}, ctx)
    line = audit.audit_path().read_text(encoding="utf-8")
    assert "4111 1111 1111 1111" not in line
    assert len(audit.recent(1)[0]["args"]["text"]) <= audit.MAX_VALUE + 1


def test_blocked_calls_are_recorded(registry, ctx):
    registry.call("remember", {"key": "pin", "value": "card 4111 1111 1111 1111"}, ctx)
    e = audit.recent(1)[0]
    assert e["tool"] == "remember" and e["outcome"] == "blocked"
    assert "4111" not in str(e)


def test_size_is_capped_and_clear(ctx, monkeypatch):
    monkeypatch.setattr(audit, "MAX_BYTES", 2000)
    reg = _registry()
    for i in range(100):
        reg.call("echo", {"text": f"line {i}"}, ctx)
    assert audit.audit_path().stat().st_size <= 2000
    assert audit.recent(1)[0]["args"]["text"] == "line 99"
    audit.clear()
    assert audit.recent() == []


def test_a_broken_disk_never_breaks_a_tool(ctx, monkeypatch):
    monkeypatch.setattr(audit, "audit_path", lambda: (_ for _ in ()).throw(OSError("disk")))
    assert _registry().call("echo", {"text": "hi"}, ctx) == {"result": "hi"}
