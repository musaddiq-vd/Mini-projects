import time

from jarvis.config import Settings, load_settings, save_settings
from jarvis.confirm import ConfirmGate
from jarvis.memory import MemoryStore
from jarvis.undo import UndoStack


def test_settings_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    save_settings(Settings(user_name="Musaddiq", voice="Puck"), path)
    loaded = load_settings(path)
    assert loaded.user_name == "Musaddiq"
    assert loaded.voice == "Puck"


def test_settings_ignore_unknown_and_broken(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"user_name": "A", "nonsense": 1}')
    assert load_settings(path).user_name == "A"
    path.write_text("{not json")
    assert load_settings(path).user_name == ""


def test_env_api_key_wins(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert Settings(api_key="from-file").effective_api_key == "from-env"


def test_memory_persists_and_recalls(tmp_path):
    path = tmp_path / "m.json"
    mem = MemoryStore(path)
    mem.remember("Sister Name", "Ayesha", "relationship")
    mem.remember("city", "Mumbai", "identity")
    again = MemoryStore(path)
    assert again.get("sister_name").value == "Ayesha"
    assert again.recall("sister")[0].key == "sister_name"
    assert again.recall("mumbai")[0].key == "city"
    assert again.forget("city") and not again.forget("city")


def test_memory_prompt_lists_keys_that_do_not_fit(tmp_path):
    mem = MemoryStore(tmp_path / "m.json")
    mem.remember("name", "Musaddiq", "identity")
    for i in range(40):
        mem.remember(f"fact_{i}", "x" * 50)
    block = mem.prompt_block(budget=300)
    assert "name: Musaddiq" in block  # identity always first
    assert "recall_memory" in block and "fact_0" in block


def test_undo_order_and_errors():
    stack, log = UndoStack(), []
    stack.push("a", lambda: log.append("a"))
    stack.push("b", lambda: log.append("b"))
    assert stack.undo() == "Undone: b"
    assert log == ["b"]
    stack.push("bad", lambda: 1 / 0)
    assert "Could not undo" in stack.undo()
    stack.undo()
    assert stack.undo() == "Nothing to undo."


def test_confirm_gate_times_out_as_no():
    gate = ConfirmGate(lambda q: time.sleep(5) or True, timeout=0.2)
    assert gate.confirm("?") is False
    assert ConfirmGate(lambda q: True).confirm("?") is True
    assert ConfirmGate(lambda q: 1 / 0).confirm("?") is False


def test_outdated_models_are_replaced_and_only_changes_are_saved(tmp_path):
    import json

    path = tmp_path / "config.json"
    path.write_text('{"text_model": "gemini-2.5-flash", "live_model": "gemini-3.1-flash-live-preview", "voice": "Puck"}')
    s = load_settings(path)
    assert s.text_model == Settings().text_model and s.live_model == Settings().live_model
    save_settings(s, path)
    assert json.loads(path.read_text()) == {"voice": "Puck"}


def test_prefer_ipv4_orders_addresses(monkeypatch):
    import socket

    import jarvis.net as net

    fake = [(socket.AF_INET6, 1, 6, "", ("::1", 443, 0, 0)), (socket.AF_INET, 1, 6, "", ("1.2.3.4", 443))]
    monkeypatch.setattr(net, "_original_getaddrinfo", lambda *a, **k: fake)
    assert net._ipv4_first("x", 443)[0][0] == socket.AF_INET


def test_prompt_includes_user_name(tmp_path):
    from jarvis.app import system_prompt

    text = system_prompt(Settings(user_name="Musaddiq"), MemoryStore(tmp_path / "m.json"))
    assert "The user's name is Musaddiq" in text


def test_old_orion_home_is_carried_over(tmp_path, monkeypatch):
    import jarvis.config as config

    monkeypatch.delenv("JARVIS_HOME")
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)
    (tmp_path / ".orion").mkdir()
    (tmp_path / ".orion" / "config.json").write_text('{"user_name": "Musaddiq"}')
    home = config.home_dir()
    assert home == tmp_path / ".jarvis"
    assert config.load_settings(home / "config.json").user_name == "Musaddiq"
