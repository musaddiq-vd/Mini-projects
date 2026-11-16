"""Write code files or small projects with the text model, then open them in VS Code."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from jarvis.tools import tool
from jarvis.tools.files import SHORTCUTS, PathNotAllowed, resolve

SYSTEM = (
    "You are an expert programmer. Return ONLY a JSON object of the form "
    '{"files": [{"path": "relative/path.ext", "content": "..."}], "run": "command to run it"} '
    "with complete, working, well-commented code. Use relative paths only. No markdown fences."
)
MAX_FILES = 20


def _parse(reply: str) -> dict:
    reply = reply.strip()
    reply = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply)
    start, end = reply.find("{"), reply.rfind("}")
    return json.loads(reply[start:end + 1])


def _open_in_editor(path) -> bool:
    code = shutil.which("code")
    if not code:
        return False
    if code.lower().endswith(".cmd"):
        # Run Code.exe directly: arguments to a .cmd file pass through cmd.exe parsing.
        exe = Path(code).resolve().parent.parent / "Code.exe"
        if not exe.exists():
            return False
        code = str(exe)
    subprocess.Popen([code, str(path)])
    return True


@tool(
    "write_code",
    "Write a program, script or small project from a description and open it in VS Code. "
    "Saved in Desktop/<folder> unless another folder is given.",
    {
        "description": {"type": "string", "description": "What to build, language, features"},
        "folder": {"type": "string", "description": "Project folder name or path"},
    },
    required=["description", "folder"],
    slow=True,
)
def write_code(ctx, description: str, folder: str):
    try:
        target = resolve(ctx, folder if any(c in folder for c in "/\\") else str(SHORTCUTS["desktop"] / folder))
    except PathNotAllowed as exc:
        return {"error": str(exc)}
    try:
        plan = _parse(ctx.llm.ask(description, system=SYSTEM))
    except (ValueError, KeyError) as exc:
        return {"error": f"The code model returned something unreadable: {exc}"}
    files = plan.get("files", [])[:MAX_FILES]
    written = []
    for item in files:
        rel = str(item.get("path", "")).replace("\\", "/").lstrip("/")
        if not rel or ".." in rel.split("/"):
            continue
        path = (target / rel).resolve()
        if target not in path.parents:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(item.get("content", "")), encoding="utf-8")
        written.append(rel)
    if not written:
        return {"error": "No files were produced."}
    opened = _open_in_editor(target)
    return {
        "folder": str(target),
        "files": written,
        "how_to_run": plan.get("run", ""),
        "opened_in_vscode": opened,
    }


# explain_code, review_code, debug_code and the other coding tools live in jarvis/tools/coding.py.
