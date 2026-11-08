"""Make a backup by voice ("back up my data")."""

from __future__ import annotations

from pathlib import Path

from jarvis.tools import tool


@tool(
    "make_backup",
    "Save a backup of everything Jarvis keeps (memory, chat, lists, money, habits, goals, practice, settings - "
    "not the API key) as one file in Documents/Jarvis Backups. Restoring is done in Settings > Privacy.",
    slow=True,
)
def make_backup(ctx):
    from jarvis.backup import default_name, describe
    from jarvis.backup import make_backup as write

    folder = Path.home() / "Documents" / "Jarvis Backups"
    dest = folder / default_name()
    n = 2
    while dest.exists():
        dest = folder / default_name().replace(".zip", f"-{n}.zip")
        n += 1
    try:
        manifest = write(dest)
    except OSError as exc:
        return {"error": f"Couldn't save the backup: {exc}"}
    return {"result": f"Backup saved: {dest.name} in Documents/Jarvis Backups. {describe(manifest)}.",
            "path": str(dest), "tip": "To restore on another PC: Settings > Privacy > Restore from a backup."}
