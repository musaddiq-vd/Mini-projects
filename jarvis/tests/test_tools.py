from pathlib import Path

from jarvis.tools import ToolRegistry, build_registry


def test_builtins_load_without_errors(registry):
    assert not registry.errors, registry.errors
    for name in ["get_time", "open_app", "web_search", "remember", "undo", "move_file", "power"]:
        assert name in registry.tools


def test_declarations_are_gemini_shaped(registry):
    for decl in registry.declarations():
        assert decl["name"] and decl["description"]
        assert decl["parameters"]["type"] == "OBJECT"
        for prop in decl["parameters"].get("properties", {}).values():
            assert prop["type"].isupper()


def test_declarations_accepted_by_sdk(registry):
    from google.genai import types

    types.Tool(function_declarations=registry.declarations())


def test_call_handles_unknown_missing_and_extra_args(registry, ctx):
    assert "Unknown tool" in registry.call("nope", {}, ctx)["error"]
    assert "Missing" in registry.call("remember", {"key": "a"}, ctx)["error"]
    out = registry.call("remember", {"key": "a", "value": "b", "junk": 1}, ctx)
    assert out == {"result": "Saved a."}


def test_tool_exception_becomes_error(ctx):
    reg = ToolRegistry()
    from jarvis.tools import Tool

    reg.add(Tool("boom", "x", lambda c: 1 / 0))
    assert "ZeroDivisionError" in reg.call("boom", {}, ctx)["error"]


def test_plugin_loading(tmp_path, ctx):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "hello.py").write_text(
        "from jarvis.tools import tool\n"
        "@tool('hello', 'Say hi', {'who': {'type': 'string'}}, required=['who'])\n"
        "def hello(ctx, who):\n    return f'hi {who}'\n"
    )
    (plugins / "broken.py").write_text("raise RuntimeError('bad plugin')\n")
    (plugins / "_skip.py").write_text("raise RuntimeError('should not load')\n")
    reg = build_registry([plugins])
    assert reg.call("hello", {"who": "Musaddiq"}, ctx) == {"result": "hi Musaddiq"}
    assert any("broken.py" in e for e in reg.errors)
    assert not any("_skip" in e for e in reg.errors)


def test_example_plugin_works(ctx):
    repo_plugins = Path(__file__).resolve().parent.parent / "plugins"
    reg = ToolRegistry()
    reg.load_plugins([repo_plugins])
    assert "roll_dice" not in reg.tools  # underscore files are skipped


def test_files_stay_inside_allowed_folder(registry, ctx, sandbox, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("no")
    assert "outside" in registry.call("read_file", {"path": str(outside)}, ctx)["error"]
    assert "outside" in registry.call("read_file", {"path": str(sandbox / ".." / "secret.txt")}, ctx)["error"]


def test_write_move_folder_and_undo(registry, ctx, sandbox):
    note = sandbox / "note.txt"
    registry.call("write_file", {"path": str(note), "content": "v1"}, ctx)
    registry.call("write_file", {"path": str(note), "content": "v2"}, ctx)
    registry.call("create_folder", {"path": str(sandbox / "box")}, ctx)
    registry.call("move_file", {"source": str(note), "destination": str(sandbox / "box")}, ctx)
    assert (sandbox / "box" / "note.txt").read_text() == "v2"

    registry.call("undo", {}, ctx)  # move back
    assert note.exists()
    registry.call("undo", {}, ctx)  # remove empty folder
    assert not (sandbox / "box").exists()
    registry.call("undo", {}, ctx)  # back to v1
    assert note.read_text() == "v1"
    registry.call("undo", {}, ctx)  # file did not exist before
    assert not note.exists()


def test_list_folder(registry, ctx, sandbox):
    (sandbox / "a.txt").write_text("x")
    (sandbox / "sub").mkdir()
    out = registry.call("list_folder", {"path": str(sandbox)}, ctx)
    assert out["items"] == ["sub/", "a.txt"]


def test_delete_needs_confirmation(registry, ctx, sandbox, answers, monkeypatch):
    import send2trash

    trashed = []
    monkeypatch.setattr(send2trash, "send2trash", trashed.append)
    f = sandbox / "old.txt"
    f.write_text("x")
    answers.append(False)
    assert "Cancelled" in registry.call("delete_file", {"path": str(f)}, ctx)["result"]
    assert not trashed
    answers.append(True)
    registry.call("delete_file", {"path": str(f)}, ctx)
    assert trashed == [str(f)]
    assert "Refusing" in registry.call("delete_file", {"path": str(sandbox)}, ctx)["error"]


def test_power_shutdown_needs_confirmation(registry, ctx, answers, monkeypatch):
    import jarvis.tools.system as system

    launched = []
    monkeypatch.setattr(system.subprocess, "Popen", launched.append)
    answers.append(False)
    assert "Cancelled" in registry.call("power", {"action": "shutdown"}, ctx)["result"]
    assert not launched
    registry.call("power", {"action": "lock"}, ctx)  # lock needs no confirmation
    assert len(launched) == 1


def test_open_website_rejects_non_http(registry, ctx, monkeypatch):
    import jarvis.tools.apps as apps

    opened = []
    monkeypatch.setattr(apps.webbrowser, "open", opened.append)
    assert "error" in registry.call("open_website", {"url": "file:///C:/Windows"}, ctx)
    assert "error" in registry.call("open_website", {"url": "javascript:alert(1)"}, ctx)
    registry.call("open_website", {"url": "github.com"}, ctx)
    assert opened == ["https://github.com"]


def test_reminder_fires(registry, ctx):
    import time

    registry.call("set_reminder", {"minutes": 0.002, "message": "drink water"}, ctx)
    time.sleep(0.5)
    assert ctx.messages == ["Reminder: drink water"]


def test_memory_tool_undo(registry, ctx):
    registry.call("remember", {"key": "city", "value": "Pune"}, ctx)
    registry.call("remember", {"key": "city", "value": "Delhi"}, ctx)
    registry.call("undo", {}, ctx)
    assert ctx.memory.get("city").value == "Pune"


def test_system_prompt_mentions_author_and_memory(ctx):
    from jarvis.app import system_prompt

    ctx.memory.remember("name", "Musaddiq", "identity")
    text = system_prompt(ctx.settings, ctx.memory)
    assert "Musaddiq Khan" in text and "name: Musaddiq" in text
