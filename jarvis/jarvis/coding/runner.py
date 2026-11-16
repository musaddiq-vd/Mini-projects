"""Run a file or the tests of the active project.

Safety rules:
- Only commands built here from the detected runner (python, node, pytest, unittest, npm/jest/vitest, go test,
  cargo test). Never a command the model wrote, and never through a shell.
- Every path is inside the project; a test name may only hold plain characters and can't start with "-".
- A timeout; output is capped; Jarvis's own API keys are not passed to the program.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.coding.project import Project

DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 300
MAX_OUTPUT = 30_000
HIDDEN_ENV = {"GEMINI_API_KEY", "GOOGLE_API_KEY", "JARVIS_TELEGRAM_TOKEN"}
_SAFE_NAME = re.compile(r"^[\w./\\:\[\]\- ]+$")


class RunError(Exception):
    pass


@dataclass
class RunResult:
    command: list[str]
    code: int | None          # exit code; None when it timed out
    output: str
    timed_out: bool = False
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)  # failing test names

    @property
    def ok(self) -> bool:
        return self.code == 0 and not self.timed_out

    def tail(self, chars: int = 3000) -> str:
        return self.output[-chars:]


def python_for(root: Path) -> str:
    """The project's own virtual environment if it has one, otherwise the Python running Jarvis."""
    for name in (".venv", "venv", "env"):
        for exe in (root / name / "Scripts" / "python.exe", root / name / "bin" / "python"):
            if exe.is_file():
                return str(exe)
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RunError(f"'{name}' isn't installed or isn't on the PATH, so I can't run this project's tests.")
    return found


def check_target(project: Project, target: str) -> str:
    """A test file, folder or test id inside the project, as a path relative to the project root."""
    target = (target or "").strip().strip('"')
    if not target:
        return ""
    if target.startswith("-") or not _SAFE_NAME.match(target):
        raise RunError("That test name has characters I won't pass to a command.")
    path_part, sep, test_part = target.partition("::")
    path = Path(path_part)
    path = (path if path.is_absolute() else project.root / path).resolve()
    if not project.contains(path):
        raise RunError("That test is outside the project.")
    if not path.exists():
        raise RunError(f"Not found in the project: {path_part}")
    rel = path.relative_to(project.root).as_posix() or "."
    return rel + (sep + test_part if test_part else "")


def test_command(project: Project, target: str = "") -> list[str]:
    runner = project.test_runner
    target = check_target(project, target)
    if runner == "pytest":
        return [python_for(project.root), "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header", "-rfE",
                *([target] if target else [])]
    if runner == "unittest":
        if target and target.endswith(".py"):
            module = target[:-3].replace("/", ".")
            return [python_for(project.root), "-m", "unittest", "-v", module]
        return [python_for(project.root), "-m", "unittest", "discover", "-v", *(["-s", target] if target else [])]
    if runner in {"npm", "jest", "vitest", "mocha"}:
        return [_tool("npm"), "test", "--silent", *(["--", target] if target else [])]
    if runner == "go":
        pkg = "./..." if not target else "./" + (target if Path(project.root / target).is_dir()
                                                 else str(Path(target).parent.as_posix()))
        return [_tool("go"), "test", pkg]
    if runner == "cargo":
        return [_tool("cargo"), "test", *([Path(target).stem] if target else [])]
    raise RunError("I couldn't find a test runner in this project (pytest, unittest, npm test, go test or "
                   "cargo test).")


def file_command(project: Project, file: Path) -> list[str]:
    """How to run one file: Python and JavaScript files only."""
    if not project.contains(file) or not file.is_file():
        raise RunError("That file isn't inside the project.")
    suffix = file.suffix.lower()
    if suffix in {".py", ".pyw"}:
        return [python_for(project.root), str(file)]
    if suffix in {".js", ".mjs", ".cjs"}:
        return [_tool("node"), str(file)]
    raise RunError(f"I can run Python and JavaScript files directly; for {suffix} files I run the tests instead.")


def is_test_file(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return bool(re.match(r"(test_.*\.py|.*_test\.(py|go)|.*\.(test|spec)\.[jt]sx?)$", name)) or "/tests/" in "/" + rel


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() not in HIDDEN_ENV}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("CI", "1")  # test runners stay non-interactive (no watch mode)
    return env


def run(command: list[str], cwd: Path, timeout: int = DEFAULT_TIMEOUT) -> RunResult:
    timeout = max(5, min(int(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(command, cwd=str(cwd), capture_output=True, timeout=timeout, env=_env(),
                              stdin=subprocess.DEVNULL, creationflags=flags, shell=False)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"") + (exc.stderr or b"")
        text = out.decode("utf-8", errors="replace") if isinstance(out, bytes) else str(out)
        return RunResult(command, None, text[-MAX_OUTPUT:] + f"\n[stopped after {timeout} seconds]", True)
    except OSError as exc:
        raise RunError(f"Couldn't start {Path(command[0]).name}: {exc}") from exc
    text = (proc.stdout or b"").decode("utf-8", errors="replace")
    err = (proc.stderr or b"").decode("utf-8", errors="replace")
    if err.strip():
        text = (text + "\n" + err).strip()
    if len(text) > MAX_OUTPUT:  # the end holds the summary and the traceback
        text = "[... start cut ...]\n" + text[-MAX_OUTPUT:]
    return summarize(RunResult(command, proc.returncode, text))


_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|deselected|xfailed)")


def summarize(result: RunResult) -> RunResult:
    """Fill in passed / failed counts and failing test names from the runner's output."""
    out = result.output
    tail = out[-4000:]
    for n, word in _COUNT.findall(tail):  # pytest: "2 failed, 5 passed in 0.3s"
        n = int(n)
        if word == "passed":
            result.passed = n
        elif word == "failed":
            result.failed = n
        elif word.startswith("error"):
            result.errors = n
        elif word == "skipped":
            result.skipped = n
    names = re.findall(r"^(?:FAILED|ERROR) (\S+)", out, re.M)  # pytest -rfE
    m = re.search(r"Ran (\d+) tests?", tail)  # unittest
    if m:
        total = int(m.group(1))
        fm = re.search(r"FAILED \((?:failures=(\d+))?(?:, )?(?:errors=(\d+))?", tail)
        result.failed = int(fm.group(1) or 0) if fm else 0
        result.errors = int(fm.group(2) or 0) if fm else 0
        result.passed = max(0, total - result.failed - result.errors)
        names = re.findall(r"^(?:FAIL|ERROR): (\S+ \([^)]*\))", out, re.M)
    jest = re.search(r"Tests:\s+(?:(\d+) failed, )?(?:(\d+) skipped, )?(?:(\d+) passed, )?(\d+) total", tail)
    if jest:
        result.failed, result.skipped, result.passed = (int(jest.group(i) or 0) for i in (1, 2, 3))
        names = re.findall(r"^\s*●\s+(.+?)\s*$", out, re.M)
    if not names:
        names = re.findall(r"^--- FAIL: (\S+)", out, re.M)  # go test
    if not names:
        names = re.findall(r"^test (\S+) \.\.\. FAILED", out, re.M)  # cargo test
    if names and not result.failed:
        result.failed = len(names)
    result.failures = list(dict.fromkeys(names))[:20]
    return result


# -- tracebacks ----------------------------------------------------------------------------------------
_PY_FRAME = re.compile(r'File "([^"]+)", line (\d+)')
_GENERIC_FRAME = re.compile(r"((?:[A-Za-z]:)?[\w./\\-]+\.(?:py|js|mjs|cjs|ts|tsx|jsx|go|rs|java|rb|php)):(\d+)")


def frames(output: str, project: Project) -> list[tuple[str, int]]:
    """(relative file, line) pairs from a traceback that point into the project, innermost last."""
    found: list[tuple[str, int]] = []
    for pattern in (_PY_FRAME, _GENERIC_FRAME):
        for m in pattern.finditer(output):
            raw, line = m.group(1).strip(), int(m.group(2))
            path = Path(raw)
            path = path if path.is_absolute() else project.root / path
            try:
                path = path.resolve()
            except OSError:
                continue
            if not project.contains(path) or not path.is_file():
                continue
            if any(part in {"site-packages", "node_modules", ".venv", "venv"} for part in path.parts):
                continue
            item = (project.rel(path), line)
            if item in found:
                found.remove(item)
            found.append(item)
        if found:
            break
    return found


def error_line(output: str) -> str:
    """The most telling line of an error output: 'ValueError: ...', 'AssertionError ...', 'Error: ...'."""
    candidates = re.findall(r"^\s*(?:E\s+)?((?:[\w.]+(?:Error|Exception|Exit)|AssertionError|panic|error(?:\[\w+\])?)"
                            r"\b.*)$", output, re.M)
    return candidates[-1].strip()[:300] if candidates else ""
