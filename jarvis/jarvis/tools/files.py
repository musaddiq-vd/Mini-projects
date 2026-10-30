"""File tools. Every path must be inside the allowed folders (default: home)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from jarvis.tools import tool

MAX_READ_CHARS = 20_000
MAX_UNDO_BYTES = 1_000_000

SHORTCUTS = {
    "desktop": Path.home() / "Desktop",
    "documents": Path.home() / "Documents",
    "downloads": Path.home() / "Downloads",
    "pictures": Path.home() / "Pictures",
    "music": Path.home() / "Music",
    "videos": Path.home() / "Videos",
    "home": Path.home(),
}


class PathNotAllowed(Exception):
    pass


def resolve(ctx, raw: str) -> Path:
    """Turn what the model sent into a safe absolute path."""
    raw = (raw or "").strip().strip('"')
    first, _, rest = raw.replace("\\", "/").partition("/")
    if first.lower() in SHORTCUTS:
        path = SHORTCUTS[first.lower()] / rest
    else:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = SHORTCUTS["desktop"] / path
    path = path.resolve()
    for root in ctx.settings.roots:
        if path == root or root in path.parents:
            return path
    raise PathNotAllowed(f"'{path}' is outside the folders Jarvis may use.")


def _guard(fn):
    def wrapper(ctx, *args, **kwargs):
        try:
            return fn(ctx, *args, **kwargs)
        except PathNotAllowed as exc:
            return {"error": str(exc)}
        except FileNotFoundError as exc:
            return {"error": f"Not found: {exc.filename or exc}"}

    wrapper.__name__ = fn.__name__
    wrapper.__wrapped__ = fn
    import inspect

    wrapper.__signature__ = inspect.signature(fn)
    return wrapper


@tool(
    "list_folder",
    "List files and folders. Paths can start with desktop/, documents/, downloads/ etc.",
    {"path": {"type": "string", "description": "Folder, e.g. 'downloads'"}},
    required=["path"],
)
@_guard
def list_folder(ctx, path: str):
    folder = resolve(ctx, path)
    if not folder.is_dir():
        return {"error": f"'{folder}' is not a folder."}
    entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    items = [p.name + ("/" if p.is_dir() else "") for p in entries[:200]]
    return {"folder": str(folder), "count": len(entries), "items": items}


@tool(
    "read_file",
    "Read a text file (first 20,000 characters).",
    {"path": {"type": "string"}},
    required=["path"],
    slow=True,
)
@_guard
def read_file(ctx, path: str):
    file = resolve(ctx, path)
    text = file.read_text(encoding="utf-8", errors="replace")
    truncated = len(text) > MAX_READ_CHARS
    return {"path": str(file), "content": text[:MAX_READ_CHARS], "truncated": truncated}


@tool(
    "write_file",
    "Create or overwrite a text file. Can be undone.",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    required=["path", "content"],
)
@_guard
def write_file(ctx, path: str, content: str):
    file = resolve(ctx, path)
    existed = file.exists()
    old = None
    if existed:
        if file.stat().st_size > MAX_UNDO_BYTES:
            return {"error": "File is larger than 1 MB; refusing to overwrite it."}
        old = file.read_bytes()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")

    def revert():
        if old is None:
            file.unlink(missing_ok=True)
        else:
            file.write_bytes(old)

    ctx.undo.push(f"write {file.name}", revert)
    return f"{'Updated' if existed else 'Created'} {file}."


@tool(
    "create_folder",
    "Create a new folder. Can be undone while it is still empty.",
    {"path": {"type": "string"}},
    required=["path"],
)
@_guard
def create_folder(ctx, path: str):
    folder = resolve(ctx, path)
    if folder.exists():
        return f"'{folder}' already exists."
    folder.mkdir(parents=True)

    def revert():
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
        else:
            raise RuntimeError("folder is no longer empty")

    ctx.undo.push(f"create folder {folder.name}", revert)
    return f"Created folder {folder}."


@tool(
    "move_file",
    "Move or rename a file or folder. Can be undone.",
    {"source": {"type": "string"}, "destination": {"type": "string"}},
    required=["source", "destination"],
)
@_guard
def move_file(ctx, source: str, destination: str):
    src = resolve(ctx, source)
    dst = resolve(ctx, destination)
    if not src.exists():
        raise FileNotFoundError(str(src))
    if dst.is_dir():
        dst = dst / src.name
    if dst.exists():
        return {"error": f"'{dst}' already exists."}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    ctx.undo.push(f"move {src.name}", lambda: shutil.move(str(dst), str(src)))
    return f"Moved to {dst}."


@tool(
    "delete_file",
    "Send a file or folder to the Recycle Bin / Trash. Asks the user to confirm.",
    {"path": {"type": "string"}},
    required=["path"],
)
@_guard
def delete_file(ctx, path: str):
    from send2trash import send2trash

    target = resolve(ctx, path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    if target in ctx.settings.roots:
        return {"error": "Refusing to delete a top-level allowed folder."}
    if not ctx.confirm.confirm(f"Move '{target}' to the trash?"):
        return "Cancelled - the user did not confirm."
    send2trash(os.fspath(target))
    return f"Moved {target.name} to the trash."
