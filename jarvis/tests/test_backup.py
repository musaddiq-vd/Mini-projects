"""Backup and restore."""

import json
import time
import zipfile

import pytest

from jarvis import backup


def fill(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "memory.json").write_text(json.dumps({"facts": [{"key": "city", "value": "Pune"}]}), encoding="utf-8")
    (home / "state.json").write_text(json.dumps({"notes": [{"id": 1, "text": "milk"}], "reminders": []}),
                                     encoding="utf-8")
    (home / "history.jsonl").write_text('{"t": 1, "role": "user", "text": "hi"}\n', encoding="utf-8")
    (home / "expenses.json").write_text(json.dumps({"items": [{"amount": 50}], "budgets": {}, "next_id": 2}),
                                        encoding="utf-8")
    (home / "config.json").write_text(json.dumps({"api_key": "dpapi:SECRET", "calendar_urls": ["dpapi:CAL"],
                                                  "theme_name": "Sunset", "user_name": "Musaddiq"}),
                                      encoding="utf-8")


def test_round_trip_to_another_pc(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    fill(old)
    dest = tmp_path / "b.zip"
    manifest = backup.make_backup(dest, old)
    assert manifest["counts"]["messages"] == 1 and manifest["counts"]["notes"] == 1
    with zipfile.ZipFile(dest) as z:
        settings = json.loads(z.read("settings.json"))
        assert "api_key" not in settings and "calendar_urls" not in settings  # secrets stay on the old PC
        assert settings["theme_name"] == "Sunset"
    new.mkdir()
    (new / "config.json").write_text(json.dumps({"api_key": "dpapi:NEWKEY"}), encoding="utf-8")
    (new / "goals.json").write_text(json.dumps({"goals": [{"title": "old"}]}), encoding="utf-8")
    out = backup.restore(dest, new)
    assert json.loads((new / "memory.json").read_text(encoding="utf-8"))["facts"][0]["value"] == "Pune"
    cfg = json.loads((new / "config.json").read_text(encoding="utf-8"))
    assert cfg["api_key"] == "dpapi:NEWKEY" and cfg["theme_name"] == "Sunset"  # this PC's key is kept
    assert not (new / "goals.json").exists()  # nothing old mixes in
    assert (new / "backups").is_dir() and out["saved_current"].endswith(".zip")  # current data saved first


@pytest.mark.parametrize("make, message", [
    (lambda p: p.write_text("not a zip"), "not a zip"),
    (lambda p: zipfile.ZipFile(p, "w").writestr("memory.json", "{}"), "no manifest"),
    (lambda p: zipfile.ZipFile(p, "w").writestr("manifest.json", '{"app": "other"}'), "isn't a Jarvis backup"),
    (lambda p: zipfile.ZipFile(p, "w").writestr("manifest.json", '{"app": "jarvis", "format": 99}'), "newer Jarvis"),
])
def test_bad_files_are_refused(tmp_path, make, message):
    p = tmp_path / "x.zip"
    make(p)
    with pytest.raises(backup.BackupError, match=message):
        backup.read_backup(p)


def test_damaged_and_unexpected_entries(tmp_path):
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("manifest.json", json.dumps({"app": "jarvis", "format": 1}))
        z.writestr("../evil.py", "print('x')")
        z.writestr("plugins/hack.py", "x")
        z.writestr("state.json", '{"ok": 1}')
    manifest, files = backup.read_backup(p)
    assert list(files) == ["state.json"]  # unknown names and paths are never written
    with zipfile.ZipFile(p, "a") as z:
        z.writestr("memory.json", "{broken")
    with pytest.raises(backup.BackupError, match="damaged"):
        backup.read_backup(p)


def test_auto_backup_weekly_and_keeps_four(tmp_path):
    home = tmp_path / "h"
    assert backup.auto_backup(home) is None  # nothing to back up yet
    fill(home)
    first = backup.auto_backup(home)
    assert first and first.exists()
    assert backup.auto_backup(home) is None  # not again this week
    now = time.time()
    for i in range(1, 7):
        backup.auto_backup(home, now=now + i * 8 * 86400)
        newest = sorted((home / "backups").glob("auto-*.zip"))[-1]
        import os

        os.utime(newest, (now + i * 8 * 86400, now + i * 8 * 86400))
    assert len(list((home / "backups").glob("auto-*.zip"))) == 4


def test_voice_backup_and_reload(isolated_home, sandbox, monkeypatch, tmp_path):
    from jarvis.app import create_app
    from jarvis.config import Settings

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    app = create_app(Settings(allowed_roots=[str(sandbox)]))
    app.state.set("notes", [{"id": 1, "text": "first"}])
    out = app.registry.call("make_backup", {}, app.ctx)
    assert "Documents/Jarvis Backups" in out["result"]
    again = app.registry.call("make_backup", {}, app.ctx)
    assert again["path"] != out["path"]  # never overwrites the earlier one today
    backup.restore(out["path"])
    app.state.set("x", 1)  # without a reload this would be lost ... or overwrite the restored file
    app.reload_stores()
    assert app.state.get("notes") == [{"id": 1, "text": "first"}]
