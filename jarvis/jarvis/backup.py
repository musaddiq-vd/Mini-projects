"""Backup and restore everything Jarvis keeps about you, as one .zip file.

In the backup: memory, chat history, notes / reminders / app state, expenses, habits, goals, practice
progress and your settings. Left out: the API key and calendar links (they are encrypted for this Windows
account and wouldn't work on another PC anyway), logs, image cache and plugins (code is never restored).

Restoring checks the file first (only known names, size limits, valid JSON), saves a copy of the current
data to ~/.jarvis/backups, then writes the new files. The app then restarts so every part reloads.
"""

from __future__ import annotations

import json
import time
import zipfile
from datetime import datetime
from pathlib import Path

from jarvis.config import home_dir

FORMAT = 1
DATA_FILES = ("memory.json", "state.json", "history.jsonl", "english.json", "expenses.json", "habits.json",
              "goals.json")
SETTINGS_NAME = "settings.json"
MANIFEST = "manifest.json"
ALLOWED = set(DATA_FILES) | {SETTINGS_NAME, MANIFEST}
SECRET_SETTINGS = ("api_key", "calendar_urls", "ha_token", "tuya_devices", "telegram_token", "telegram_chat_id",
                   "telegram_user")
MAX_FILE = 60 * 1024 * 1024
MAX_TOTAL = 200 * 1024 * 1024
KEEP_AUTO = 4


class BackupError(ValueError):
    """A friendly reason the backup can't be made or used."""


def _settings_without_secrets(home: Path) -> dict:
    path = home / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if k not in SECRET_SETTINGS} if isinstance(data, dict) else {}


def _counts(home: Path) -> dict:
    out = {}
    try:
        out["memories"] = len(json.loads((home / "memory.json").read_text(encoding="utf-8")).get("facts", []))
    except (OSError, ValueError, AttributeError):
        pass
    try:
        out["messages"] = sum(1 for _ in (home / "history.jsonl").open(encoding="utf-8"))
    except OSError:
        pass
    for name, key in (("expenses", "items"), ("habits", "habits"), ("goals", "goals")):
        try:
            out[name] = len(json.loads((home / f"{name}.json").read_text(encoding="utf-8")).get(key, []))
        except (OSError, ValueError, AttributeError):
            pass
    try:
        st = json.loads((home / "state.json").read_text(encoding="utf-8"))
        out["notes"] = len(st.get("notes", []))
        out["reminders"] = len(st.get("reminders", []))
    except (OSError, ValueError, AttributeError):
        pass
    return out


def make_backup(dest: Path, home: Path | None = None) -> dict:
    """Write the backup zip to `dest`. Returns the manifest."""
    home = Path(home or home_dir())
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    files = [n for n in DATA_FILES if (home / n).is_file()]
    from jarvis import __version__

    manifest = {"app": "jarvis", "format": FORMAT, "version": __version__,
                "created": datetime.now().isoformat(timespec="seconds"), "files": files + [SETTINGS_NAME],
                "counts": _counts(home)}
    tmp = dest.with_name(dest.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False))
        for name in files:
            z.write(home / name, name)
        z.writestr(SETTINGS_NAME, json.dumps(_settings_without_secrets(home), indent=2, ensure_ascii=False))
    tmp.replace(dest)
    return manifest


def read_backup(path: Path) -> tuple[dict, dict[str, bytes]]:
    """(manifest, {name: bytes}) after checking everything. Raises BackupError."""
    path = Path(path)
    if not path.is_file():
        raise BackupError("That backup file doesn't exist.")
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise BackupError("That isn't a Jarvis backup (not a zip file).") from None
    with z:
        infos = z.infolist()
        if sum(i.file_size for i in infos) > MAX_TOTAL:
            raise BackupError("That backup is too big to be a Jarvis backup.")
        names = {i.filename for i in infos}
        if MANIFEST not in names:
            raise BackupError("That zip file isn't a Jarvis backup (no manifest).")
        out: dict[str, bytes] = {}
        for info in infos:
            name = info.filename
            if name not in ALLOWED or "/" in name or "\\" in name:
                continue  # anything unexpected is ignored, never written
            if info.file_size > MAX_FILE:
                raise BackupError(f"{name} in the backup is too big.")
            out[name] = z.read(info)
    try:
        manifest = json.loads(out.pop(MANIFEST).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise BackupError("The backup's manifest is damaged.") from None
    if not isinstance(manifest, dict) or manifest.get("app") != "jarvis":
        raise BackupError("That zip file isn't a Jarvis backup.")
    if int(manifest.get("format", 0)) > FORMAT:
        raise BackupError("That backup was made by a newer Jarvis. Update Jarvis first.")
    for name, data in out.items():
        try:
            text = data.decode("utf-8")
            if name.endswith(".json"):
                if not isinstance(json.loads(text), dict):
                    raise ValueError
            else:
                for line in text.splitlines()[:50]:
                    if line.strip():
                        json.loads(line)
        except (ValueError, UnicodeDecodeError):
            raise BackupError(f"{name} in the backup is damaged.") from None
    return manifest, out


def backups_dir(home: Path | None = None) -> Path:
    return Path(home or home_dir()) / "backups"


def restore(path: Path, home: Path | None = None) -> dict:
    """Replace the current data with the backup's. Returns {'manifest', 'saved_current'}."""
    home = Path(home or home_dir())
    manifest, files = read_backup(path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safety = backups_dir(home) / f"before-restore-{stamp}.zip"
    make_backup(safety, home)  # the current data, in case the user changes their mind
    for name, data in files.items():
        if name == SETTINGS_NAME:
            _merge_settings(home, data)
            continue
        target = home / name
        tmp = target.with_name(target.name + ".restoring")
        tmp.write_bytes(data)
        tmp.replace(target)
    for name in DATA_FILES:  # files the backup didn't have are cleared, so nothing old mixes in
        if name not in files and (home / name).exists():
            (home / name).unlink()
    return {"manifest": manifest, "saved_current": str(safety)}


def _merge_settings(home: Path, data: bytes) -> None:
    """Take the backup's choices but keep this PC's API key and calendar links."""
    try:
        incoming = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return
    path = home / "config.json"
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        current = {}
    merged = {k: v for k, v in incoming.items() if k not in SECRET_SETTINGS}
    for key in SECRET_SETTINGS:
        if key in current:
            merged[key] = current[key]
    tmp = path.with_name("config.json.restoring")
    tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    tmp.replace(path)


def auto_backup(home: Path | None = None, now: float | None = None, every_days: int = 7) -> Path | None:
    """A weekly backup in ~/.jarvis/backups (keeps the last few). Returns the new file, or None."""
    home = Path(home or home_dir())
    folder = backups_dir(home)
    now = now or time.time()
    autos = sorted(folder.glob("auto-*.zip"))
    if autos and now - autos[-1].stat().st_mtime < every_days * 86400:
        return None
    if not any((home / n).exists() for n in DATA_FILES):
        return None  # nothing to back up yet
    dest = folder / f"auto-{datetime.fromtimestamp(now):%Y%m%d-%H%M%S}.zip"
    make_backup(dest, home)
    for old in sorted(folder.glob("auto-*.zip"))[:-KEEP_AUTO]:
        old.unlink(missing_ok=True)
    return dest


def describe(manifest: dict) -> str:
    c = manifest.get("counts", {})
    parts = [f"{c[k]} {k}" for k in ("messages", "memories", "notes", "reminders", "expenses", "habits", "goals")
             if c.get(k)]
    when = manifest.get("created", "")[:16].replace("T", " ")
    return f"Made {when}" + (": " + ", ".join(parts) if parts else "")


def default_name() -> str:
    return f"Jarvis-backup-{datetime.now():%Y-%m-%d}.zip"
