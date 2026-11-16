"""Coding assistant tools: open a project, explain, debug and fix, review, write and run tests, refactor, search.

The heavy lifting is in ``jarvis.coding``. Results carry a ``card`` (shown as a code card in the chat) and a short
``spoken`` line for voice. Files are read only inside the allowed folders; code runs only through the project's
detected runner; changes are written only after the user says yes, with a backup and undo.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from jarvis.coding import brain, edits, runner
from jarvis.coding import project as proj
from jarvis.coding.highlight import guess_language
from jarvis.coding.snippets import BUDGET, CodeError, around, file_snippet, outline, read_code, symbol_snippet, untrusted
from jarvis.tools import tool
from jarvis.tools.files import PathNotAllowed, resolve

CARD_CODE_CHARS = 12_000
_last_run: dict = {}  # the newest failing run: {"root": Path, "output": str, "when": float}


class _Stop(Exception):
    """Stops a tool with a message for the user."""


# -- shared helpers ------------------------------------------------------------------------------------------
def _allowed(ctx, path: Path) -> Path:
    try:
        return resolve(ctx, str(path))
    except PathNotAllowed as exc:
        raise _Stop(str(exc)) from exc


def _active(ctx) -> proj.Project | None:
    root = proj.active_root()
    if root is None:
        return None
    try:
        root = resolve(ctx, str(root))
    except PathNotAllowed:
        return None
    return proj.load(root)


def _need_project(ctx) -> proj.Project:
    project = _active(ctx)
    if project is None:
        raise _Stop("No project is open. Ask the user which folder their project is in, then call open_project.")
    return project


def _file(ctx, path: str) -> tuple[proj.Project, Path]:
    """The file the user means and its project (a relative path or a bare file name is looked up in the open
    project first). Opening a file from another project makes that project the active one."""
    from jarvis.guardrails import sensitive_path

    raw = (path or "").strip().strip('"')
    if not raw:
        raise _Stop("Which file? Give a path or a file name in the open project.")
    project = _active(ctx)
    file: Path | None = None
    if project is not None and not Path(raw).is_absolute():
        candidate = (project.root / raw).resolve()
        if project.contains(candidate) and candidate.exists():
            file = candidate
        else:
            name = raw.replace("\\", "/").lstrip("./").lower()
            hits = [f for f in project.files if f.lower() == name or f.lower().endswith("/" + name)]
            if len(hits) == 1:
                file = project.root / hits[0]
            elif len(hits) > 1:
                raise _Stop(f"More than one file matches '{raw}': {', '.join(hits[:6])}. Which one?")
    if file is None:
        try:
            file = resolve(ctx, raw)
        except PathNotAllowed as exc:
            raise _Stop(str(exc)) from exc
    file = _allowed(ctx, file)
    reason = sensitive_path(file)
    if reason:
        raise _Stop(f"I don't read that file because {reason}.")
    if not file.exists():
        raise _Stop(f"Not found: {raw}")
    if project is None or not project.contains(file):
        root = proj.find_root(file, ctx.settings.roots)
        project = proj.load(_allowed(ctx, root))
        proj.set_active(project.root)
    return project, file


def _ask(ctx, prompt: str, system: str) -> str:
    try:
        return ctx.llm.ask(prompt, system=system)
    except Exception as exc:
        raise _Stop(f"Couldn't reach the code model ({type(exc).__name__}). Try again in a moment.") from exc


def _guarded(fn):
    """Turn _Stop / CodeError / RunError / EditError into a friendly error reply."""
    import functools

    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        try:
            return fn(ctx, *args, **kwargs)
        except (_Stop, CodeError, runner.RunError, edits.EditError) as exc:
            return {"error": str(exc)}

    return wrapper


def _level(level: str) -> str:
    level = (level or "normal").strip().lower()
    return {"simple": "beginner", "easy": "beginner", "basic": "beginner", "intermediate": "normal",
            "advanced": "expert", "deep": "expert"}.get(level, level if level in brain.LEVELS else "normal")


def _card_code(text: str) -> str:
    return text if len(text) <= CARD_CODE_CHARS else text[:CARD_CODE_CHARS] + "\n... (cut, open the file for the rest)"


def _snippet(project: proj.Project, file: Path, symbol: str):
    text = read_code(file)
    rel = project.rel(file)
    snip = symbol_snippet(file, rel, text, symbol) if symbol else file_snippet(file, rel, text)
    return text, snip


def _project_line(project: proj.Project) -> str:
    bits = [project.language]
    if project.framework:
        bits.append(project.framework)
    if project.test_runner:
        bits.append(f"tests: {project.test_runner}")
    return f"Project '{project.name}' ({', '.join(bits)})."


def _run_summary(result: runner.RunResult) -> str:
    if result.timed_out:
        return "Stopped: it ran past the time limit."
    parts = [f"{n} {word}" for n, word in ((result.passed, "passed"), (result.failed, "failed"),
                                            (result.errors, "errors"), (result.skipped, "skipped")) if n]
    if parts:
        return ", ".join(parts)
    return "Finished without errors." if result.ok else f"Failed (exit code {result.code})."


# -- tools ---------------------------------------------------------------------------------------------------
@tool(
    "open_project",
    "Open a coding project folder so Jarvis knows it: detects the language, framework and test runner and maps "
    "its files and functions. Later coding requests ('explain parse', 'run the tests') use this project.",
    {"path": {"type": "string", "description": "The project folder (or any file in it)"}},
    required=["path"],
)
@_guarded
def open_project(ctx, path: str):
    try:
        target = resolve(ctx, path)
    except PathNotAllowed as exc:
        return {"error": str(exc)}
    if not target.exists():
        return {"error": f"Not found: {target}"}
    root = target if target.is_dir() else proj.find_root(target, ctx.settings.roots)
    project = proj.load(_allowed(ctx, root), fresh=True)
    proj.set_active(project.root)
    info = project.summary()
    card = {"icon": "📂", "title": f"Project · {project.name}", "subtitle": str(project.root),
            "summary": _project_line(project),
            "sections": [{"title": "Files", "items": project.files[:15] + (["…"] if len(project.files) > 15 else [])}]}
    return {**info, "card": card,
            "result": f"{_project_line(project)} {len(project.files)} files mapped. Say this in one short line."}


@tool(
    "explain_code",
    "Explain code really well: a file, or one function/class in it, at the user's level (beginner, normal or "
    "expert), with a summary, step by step walk-through, data flow, key ideas with everyday analogies, gotchas and "
    "a usage example. Shows a code card; speak only the short version. Use for 'explain main.py', 'what does "
    "parse do', 'samjhao yeh code', 'explain it simply'.",
    {"path": {"type": "string", "description": "File path or name in the open project (optional if symbol given)"},
     "symbol": {"type": "string", "description": "A function or class name, e.g. 'parse' or 'Parser.parse'"},
     "level": {"type": "string", "description": "beginner, normal or expert"},
     "question": {"type": "string", "description": "What the user specifically wants to know"}},
    slow=True,
)
@_guarded
def explain_code(ctx, path: str = "", symbol: str = "", level: str = "normal", question: str = ""):
    symbol = (symbol or "").strip()
    if not path and symbol:
        project = _need_project(ctx)
        hits = project.find_symbol(symbol)
        if not hits:
            return {"error": f"I couldn't find '{symbol}' in project {project.name}."}
        file = _allowed(ctx, project.root / hits[0].file)
        symbol = hits[0].name
    else:
        project, file = _file(ctx, path)
    if file.is_dir():
        return {"error": "That's a folder. Say which file (or call open_project for an overview)."}
    level = _level(level)
    text, snip = _snippet(project, file, symbol)
    language = guess_language(snip.rel) or project.language
    extra = ""
    if not symbol and snip.truncated:
        extra = "\nThe file is long; only its start is shown. Its outline:\n" + outline(
            proj.symbols_for(text, snip.rel))
    target = f"{snip.symbol} in {snip.rel}" if snip.symbol else snip.rel
    prompt = (f"Explain {target} (lines {snip.start}-{snip.end}), language: {language}.\n"
              f"Level: {level}. {brain.LEVELS[level]}\n{_project_line(project)}\n"
              + (f"The user asks: {question}\n" if question else "")
              + untrusted("CODE", snip.numbered()) + extra)
    exp = brain.explanation(_ask(ctx, prompt, brain.EXPLAIN_SYSTEM))
    if not exp["summary"] and not exp["steps"]:
        return {"error": "The code model's answer was unreadable. Try again."}
    sections = [
        {"title": "Step by step", "items": exp["steps"]},
        {"title": "How the data flows", "items": [exp["data_flow"]] if exp["data_flow"] else []},
        {"title": "Key ideas", "items": [f"`{c['name']}`: {c['meaning']}" + (f"  Think of it like this: {c['analogy']}"
                                                                           if c["analogy"] else "")
                                         for c in exp["concepts"]]},
        {"title": "Watch out for", "items": exp["gotchas"]},
        {"title": "Cost", "items": [exp["complexity"]] if exp["complexity"] else []},
    ]
    card = {"icon": "📘", "title": f"Explain · {snip.symbol or file.name}",
            "subtitle": f"{snip.rel} · lines {snip.start}-{snip.end} · {level}",
            "summary": exp["summary"], "sections": [s for s in sections if s["items"]],
            "code": _card_code(snip.text), "language": language, "start_line": snip.start,
            "example": exp["example"]}
    spoken = exp["spoken"] or exp["summary"]
    return {"path": str(file), "symbol": snip.symbol, "level": level, "summary": exp["summary"],
            "spoken": spoken, "card": card,
            "result": ("The full explanation is on a code card. Say only the short version in your own words: "
                       f"{spoken} Then offer to go simpler or deeper, or into one step.")}


@tool(
    "debug_code",
    "Debug code: runs the file or failing test (inside the project, with a time limit) or reads a pasted error, "
    "finds the lines involved, explains the root cause and proposes a minimal fix as a diff (applied only with "
    "apply_fix after the user agrees). With no arguments it debugs the last failing test run.",
    {"path": {"type": "string", "description": "File or test file to run"},
     "error": {"type": "string", "description": "Error text or traceback the user pasted"},
     "test": {"type": "string", "description": "A test id like tests/test_x.py::test_y"},
     "timeout": {"type": "integer", "description": "Seconds to let it run (default 60)"}},
    slow=True,
)
@_guarded
def debug_code(ctx, path: str = "", error: str = "", test: str = "", timeout: int = runner.DEFAULT_TIMEOUT):
    file = None
    if path:
        project, file = _file(ctx, path)
    else:
        project = _need_project(ctx)
    ran = None
    if error.strip():
        output = error.strip()[-runner.MAX_OUTPUT:]
    elif test or file is not None:
        if test or runner.is_test_file(project.rel(file)):
            command = runner.test_command(project, test or project.rel(file))
        else:
            command = runner.file_command(project, file)
        ran = runner.run(command, project.root, timeout)
        output = ran.output
        if ran.ok:
            return {"ran": " ".join([Path(command[0]).name, *command[1:]]),
                    "summary": _run_summary(ran), "output_tail": ran.tail(1500),
                    "result": "It ran without errors. Tell the user, and ask what they expected to happen instead."}
    elif _last_run.get("root") == project.root and time.time() - _last_run.get("when", 0) < 3600:
        output = _last_run["output"]
    else:
        return {"error": "Nothing to debug yet: give a file to run, a test, or paste the error."}

    frames = runner.frames(output, project)
    shown: dict[str, str] = {}
    budget = BUDGET
    for rel, line in reversed(frames[-4:]):  # innermost frame first
        if budget <= 0:
            break
        try:
            text = read_code(project.root / rel)
        except CodeError:
            continue
        block, _ = around(text, line)
        shown.setdefault(rel, "")
        shown[rel] += (f"\n... (around line {line})\n" if shown[rel] else "") + block
        budget -= len(block)
    if not shown and file is not None and file.is_file():
        snip = file_snippet(file, project.rel(file), read_code(file), BUDGET // 2)
        shown[snip.rel] = snip.numbered()
    code_parts = "\n".join(untrusted(f"FILE {rel}", block) for rel, block in shown.items())
    prompt = (f"{_project_line(project)}\n" + untrusted("ERROR OUTPUT", output[-8000:]) + "\n"
              + (code_parts or "(No project file appears in the error; reason from the output.)"))
    diag = brain.diagnosis(_ask(ctx, prompt, brain.DEBUG_SYSTEM))
    problem = diag["problem"] or runner.error_line(output) or "The program failed."
    diffs, note = [], ""
    target = diag["file"].lstrip("./")
    if target and diag["edits"]:
        path_to_fix = (project.root / target).resolve()
        if not project.contains(path_to_fix) or not path_to_fix.is_file():
            note = "The proposed fix pointed outside the project, so it was dropped."
        else:
            try:
                fix = edits.propose(_allowed(ctx, path_to_fix), project.rel(path_to_fix), read_code(path_to_fix),
                                    diag["edits"], problem)
                diffs.append({"title": f"Fix in {fix.rel}", "diff": fix.diff, "fix_id": fix.id})
            except (edits.EditError, CodeError) as exc:
                note = f"I couldn't turn the suggested fix into a clean change ({exc})"
    error_lines = output.strip().splitlines()[-25:]
    card = {"icon": "🐞", "title": "Debug", "subtitle": f"{project.name} · confidence {diag['confidence']}",
            "summary": problem, "tone": "bad",
            "sections": [s for s in [
                {"title": "Root cause", "items": [diag["root_cause"]] if diag["root_cause"] else []},
                {"title": "What happens", "items": [diag["explanation"]] if diag["explanation"] else []},
                {"title": "Next steps", "items": diag["steps"] + ([note] if note else [])}] if s["items"]],
            "code": "\n".join(error_lines), "language": "text", "code_title": "Error",
            "diffs": diffs}
    out = {"problem": problem, "root_cause": diag["root_cause"], "confidence": diag["confidence"],
           "files_involved": list(shown), "card": card}
    if ran is not None:
        out["summary"] = _run_summary(ran)
    if diffs:
        out["fix_id"] = diffs[0]["fix_id"]
        out["result"] = ("A debug card shows the cause and the proposed diff. Say the root cause in one or two "
                         f"sentences, then ask if they want the fix applied (apply_fix with fix_id "
                         f"{diffs[0]['fix_id']}). Don't read code aloud.")
    else:
        out["result"] = "A debug card shows the cause and next steps. Say the root cause in one or two sentences."
    return out


@tool(
    "apply_fix",
    "Apply a fix or refactoring that debug_code or refactor_code proposed: shows the diff and asks the user "
    "first, keeps a backup, and can be undone with 'undo'.",
    {"fix_id": {"type": "string", "description": "The fix_id from debug_code / refactor_code (default: newest)"}},
)
@_guarded
def apply_fix(ctx, fix_id: str = ""):
    fix = edits.get(fix_id)
    if fix is None:
        return {"error": "There's no proposed fix to apply. Run debug_code or refactor_code first."}
    path = _allowed(ctx, fix.path)
    if not path.is_file():
        return {"error": f"{fix.rel} no longer exists."}
    current = read_code(path)
    if edits.digest(current) != fix.old_hash:
        edits.discard(fix.id)
        return {"error": f"{fix.rel} changed since the fix was proposed. Debug it again for a fresh fix."}
    added, removed = edits.diff_stats(fix.diff)
    preview = fix.diff if len(fix.diff) < 1800 else fix.diff[:1800] + "\n..."
    if not ctx.confirm.confirm(f"Apply this change to {fix.rel}? (+{added} / -{removed} lines)\n\n{preview}"):
        return {"result": "Cancelled - the file was not changed."}
    backup = edits.write_with_backup(path, fix.new_text)
    ctx.undo.push(f"code change in {fix.rel}", lambda: edits.restore(backup, path))
    edits.discard(fix.id)
    proj.forget_cache()
    return {"result": f"Applied to {fix.rel} (+{added} / -{removed} lines). A backup is kept; say 'undo' to put "
                      "it back. Offer to run the tests again.",
            "file": str(path), "backup": str(backup)}


@tool(
    "review_code",
    "Review a code file (or one function) and give prioritised findings with line numbers: bugs, security, "
    "performance, readability. Shows a review card.",
    {"path": {"type": "string"}, "symbol": {"type": "string", "description": "Only this function/class"},
     "focus": {"type": "string", "description": "e.g. security, performance"}},
    required=["path"],
    slow=True,
)
@_guarded
def review_code(ctx, path: str, symbol: str = "", focus: str = ""):
    project, file = _file(ctx, path)
    text, snip = _snippet(project, file, symbol)
    language = guess_language(snip.rel) or project.language
    prompt = (f"Review {snip.symbol or snip.rel} ({language}). {_project_line(project)}\n"
              + (f"Focus especially on: {focus}\n" if focus else "")
              + ("Only the start of the file is shown.\n" if snip.truncated else "")
              + untrusted("CODE", snip.numbered()))
    rev = brain.review(_ask(ctx, prompt, brain.REVIEW_SYSTEM), len(text.splitlines()))
    marks = {"high": "🔴", "medium": "🟠", "low": "🟡"}
    items = [f"{marks[f['severity']]} {f['severity'].title()} · {f['category']}"
             + (f" · line {f['line']}" if f["line"] else "") + f": {f['title']}. {f['detail']}"
             + (f" Fix: {f['suggestion']}" if f["suggestion"] else "") for f in rev["findings"]]
    counts = {s: sum(1 for f in rev["findings"] if f["severity"] == s) for s in ("high", "medium", "low")}
    card = {"icon": "🔍", "title": f"Review · {snip.symbol or file.name}",
            "subtitle": f"{snip.rel} · {counts['high']} high, {counts['medium']} medium, {counts['low']} low",
            "summary": rev["summary"],
            "sections": [s for s in [{"title": "Findings", "items": items},
                                     {"title": "Done well", "items": rev["strengths"]}] if s["items"]]}
    spoken = rev["spoken"] or rev["summary"]
    return {"path": str(file), "findings": rev["findings"], "counts": counts, "spoken": spoken, "card": card,
            "result": f"A review card lists the findings. Say briefly: {spoken} Mention the most serious one."}


def _test_location(project: proj.Project, file: Path) -> tuple[Path, str]:
    """Where a new test file goes, and how to import the code under test."""
    rel = project.rel(file)
    stem, suffix = file.stem, file.suffix.lower()
    root = project.root
    if suffix in {".py", ".pyw"}:
        folder = next((root / d for d in ("tests", "test") if (root / d).is_dir()), root / "tests")
        module = rel.rsplit(".", 1)[0]
        hint = f"Tests run from the project root with {project.test_runner or 'pytest'}. "
        if module.startswith("src/"):
            module = module[4:]
            hint += ("The code is in a src/ folder: at the top of the test add "
                     "`sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))`. ")
        hint += f"Import it as `{module.replace('/', '.')}`."
        if project.test_runner == "unittest":
            hint += " Use the unittest module (unittest.TestCase)."
        else:
            hint += " Use plain pytest test functions."
        return folder / f"test_{stem}.py", hint
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        tests_dir = root / "__tests__"
        folder = tests_dir if tests_dir.is_dir() else file.parent
        imp = Path(os.path.relpath(file.with_suffix(""), folder)).as_posix()
        imp = imp if imp.startswith(".") else "./" + imp
        kind = project.test_runner if project.test_runner in {"jest", "vitest", "mocha"} else "jest"
        return folder / f"{stem}.test{suffix}", f"Use {kind}. Import the module from '{imp}'."
    if suffix == ".go":
        return file.parent / f"{stem}_test.go", "Use the standard testing package, same package as the file."
    if suffix == ".rs":
        return root / "tests" / f"{stem}_tests.rs", "Write an integration test file using the crate's public API."
    raise _Stop(f"I can write tests for Python, JavaScript/TypeScript, Go and Rust files, not {suffix} files.")


def _free(path: Path) -> Path:
    """`path`, or a numbered sibling if it exists (never overwrite)."""
    if not path.exists():
        return path
    base, ext = path.name.split(".", 1) if "." in path.name else (path.name, "")
    for n in range(1, 100):
        candidate = path.with_name(f"{base}_generated{'' if n == 1 else n}" + (f".{ext}" if ext else ""))
        if not candidate.exists():
            return candidate
    raise _Stop("Too many generated test files already exist there.")


@tool(
    "generate_tests",
    "Write unit tests for a file (or one function) in the project's test framework, save them in the test folder "
    "as a NEW file (asks first, never overwrites), run them and report pass/fail.",
    {"path": {"type": "string"}, "symbol": {"type": "string", "description": "Only this function/class"}},
    required=["path"],
    slow=True,
)
@_guarded
def generate_tests(ctx, path: str, symbol: str = ""):
    project, file = _file(ctx, path)
    _, snip = _snippet(project, file, symbol)
    target, hint = _test_location(project, file)
    target = _free(_allowed(ctx, target))
    test_rel = target.relative_to(project.root).as_posix()
    language = guess_language(snip.rel) or project.language
    prompt = (f"Write tests for {snip.symbol or snip.rel} ({language}). The test file will be {test_rel}. {hint}\n"
              + untrusted("CODE", snip.numbered()))
    made = brain.test_file(_ask(ctx, prompt, brain.TESTS_SYSTEM))
    if not made["code"].strip():
        return {"error": "The code model didn't return a test file. Try again."}
    if not ctx.confirm.confirm(f"Write {len(made['cases']) or 'new'} tests to {test_rel}? (a new file)"):
        return {"result": "Cancelled - no test file was written."}
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "x", encoding="utf-8", newline="\n") as fh:  # "x": never overwrite
        fh.write(made["code"].rstrip() + "\n")
    ctx.undo.push(f"new test file {test_rel}", lambda: target.unlink(missing_ok=True))
    proj.forget_cache()
    project = proj.load(project.root, fresh=True)
    card = {"icon": "🧪", "title": f"Tests · {snip.symbol or file.name}", "subtitle": test_rel,
            "sections": [{"title": "What they check", "items": made["cases"]}],
            "code": _card_code(made["code"]), "language": language, "code_title": test_rel}
    out = {"test_file": str(target), "cases": made["cases"], "card": card}
    if not project.test_runner and target.suffix == ".py":
        project.test_runner = "pytest"
    try:
        ran = runner.run(runner.test_command(project, test_rel), project.root)
    except runner.RunError as exc:
        card["summary"] = f"Written, but not run: {exc}"
        out["result"] = f"Tests written to {test_rel}, but I couldn't run them: {exc}"
        return out
    _remember(project, ran)
    card["summary"] = _run_summary(ran)
    card["tone"] = "ok" if ran.ok else "bad"
    if not ran.ok:
        card["sections"].append({"title": "Failing", "items": ran.failures or [runner.error_line(ran.output)]})
    out.update(passed=ran.passed, failed=ran.failed + ran.errors, summary=_run_summary(ran))
    out["result"] = (f"Wrote tests to {test_rel} and ran them: {_run_summary(ran)}. "
                     + ("Say so briefly." if ran.ok else "A failing generated test may mean a bug or a wrong test; "
                                                         "offer debug_code."))
    return out


def _remember(project: proj.Project, ran: runner.RunResult) -> None:
    if ran.ok:
        _last_run.clear()
    else:
        _last_run.update(root=project.root, output=ran.output, when=time.time())


@tool(
    "run_tests",
    "Run the open project's tests (all, or one file / test id) with its own test runner and summarise the "
    "failures. Use for 'run the tests', 'tests chalao', 'did my tests pass'.",
    {"target": {"type": "string", "description": "Test file, folder or id; empty for all"},
     "timeout": {"type": "integer", "description": "Seconds (default 60)"}},
    slow=True,
)
@_guarded
def run_tests(ctx, target: str = "", timeout: int = runner.DEFAULT_TIMEOUT):
    project = _need_project(ctx)
    ran = runner.run(runner.test_command(project, target), project.root, timeout)
    _remember(project, ran)
    summary = _run_summary(ran)
    card = {"icon": "🧪", "title": f"Tests · {project.name}", "subtitle": target or "whole suite",
            "summary": summary, "tone": "ok" if ran.ok else "bad",
            "sections": [{"title": "Failing", "items": ran.failures}] if ran.failures else []}
    if not ran.ok:
        card.update(code=ran.tail(2500), language="text", code_title="Output (end)")
    out = {"passed": ran.passed, "failed": ran.failed, "errors": ran.errors, "skipped": ran.skipped,
           "timed_out": ran.timed_out, "failures": ran.failures, "summary": summary, "card": card}
    if ran.ok:
        out["result"] = f"All good: {summary}. Say it in a few words."
    else:
        out["error_line"] = runner.error_line(ran.output)
        out["result"] = (f"{summary}. Name the failing tests briefly and offer to debug them (debug_code with no "
                         "arguments debugs this run).")
    return out


@tool(
    "refactor_code",
    "Suggest refactorings for a file or function (clearer, safer or faster, same behaviour) as diffs; each can be "
    "applied with apply_fix after the user agrees.",
    {"path": {"type": "string"}, "symbol": {"type": "string"},
     "goal": {"type": "string", "description": "e.g. 'make it readable', 'remove duplication', 'faster'"}},
    required=["path"],
    slow=True,
)
@_guarded
def refactor_code(ctx, path: str, symbol: str = "", goal: str = ""):
    project, file = _file(ctx, path)
    text, snip = _snippet(project, file, symbol)
    language = guess_language(snip.rel) or project.language
    prompt = (f"Suggest refactorings for {snip.symbol or snip.rel} ({language}).\n"
              + (f"The user's goal: {goal}\n" if goal else "") + untrusted("CODE", snip.text))
    ref = brain.refactorings(_ask(ctx, prompt, brain.REFACTOR_SYSTEM))
    diffs, skipped = [], 0
    for s in ref["suggestions"]:
        try:
            fix = edits.propose(file, snip.rel, text, s["edits"], s["title"])
        except edits.EditError:
            skipped += 1
            continue
        diffs.append({"title": s["title"], "why": s["why"], "diff": fix.diff, "fix_id": fix.id})
    if not diffs:
        return {"result": ref["summary"] or "No clear refactoring to suggest - the code looks fine as it is."
                + (" (Some suggestions didn't match the file and were dropped.)" if skipped else "")}
    card = {"icon": "🧹", "title": f"Refactor · {snip.symbol or file.name}", "subtitle": snip.rel,
            "summary": ref["summary"], "diffs": diffs}
    return {"path": str(file), "suggestions": [{"title": d["title"], "why": d["why"], "fix_id": d["fix_id"]}
                                               for d in diffs], "card": card,
            "result": ("A card shows the suggested changes as diffs. Name them in one or two sentences and ask which "
                       "to apply (apply_fix with its fix_id).")}


@tool(
    "code_search",
    "Search the open project's code: function and class names first, then lines that contain the text.",
    {"query": {"type": "string"}, "limit": {"type": "integer", "description": "Max matching lines (default 30)"}},
    required=["query"],
)
@_guarded
def code_search(ctx, query: str, limit: int = 30):
    project = _need_project(ctx)
    query = (query or "").strip()
    if len(query) < 2:
        return {"error": "Give at least two characters to search for."}
    limit = max(1, min(int(limit or 30), 100))
    low = query.lower()
    symbols = [{"name": s.name, "kind": s.kind, "file": s.file, "line": s.line}
               for s in project.symbols if low in s.name.lower()][:20]
    matches, read = [], 0
    for rel in project.files:
        if len(matches) >= limit or read > 8_000_000:
            break
        path = project.root / rel
        try:
            if path.stat().st_size > proj.MAX_PARSE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        read += len(text)
        for n, line in enumerate(text.splitlines(), 1):
            if low in line.lower():
                matches.append({"file": rel, "line": n, "text": line.strip()[:200]})
                if len(matches) >= limit:
                    break
    if not symbols and not matches:
        return {"result": f"Nothing in {project.name} matches '{query}'."}
    card = {"icon": "🔎", "title": f"Search · {query}", "subtitle": project.name,
            "summary": f"{len(symbols)} definitions, {len(matches)} matching lines",
            "sections": [s for s in [
                {"title": "Definitions", "items": [f"{s['kind']} `{s['name']}` · {s['file']}:{s['line']}"
                                                   for s in symbols]},
                {"title": "Lines", "items": [f"{m['file']}:{m['line']}  `{m['text']}`" for m in matches[:20]]}]
                if s["items"]]}
    return {"symbols": symbols, "matches": matches, "card": card,
            "result": "A card lists the matches. Say where the main ones are in a sentence."}
