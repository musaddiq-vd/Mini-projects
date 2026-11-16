"""The active coding project: where it is, what it is written in, how its tests run, and a light map of it."""

from __future__ import annotations

import ast
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Folders that are never part of the map: dependencies, builds, caches and version control.
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env", ".env", "__pycache__", ".mypy_cache",
             ".pytest_cache", ".ruff_cache", ".tox", ".nox", "dist", "build", "target", ".idea", ".vscode",
             ".next", ".nuxt", "out", "coverage", "htmlcov", ".gradle", "bin", "obj", "vendor", ".eggs",
             "site-packages", ".cache", ".parcel-cache", ".svelte-kit"}
LANGUAGES = {
    ".py": "python", ".pyw": "python", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript", ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".rs": "rust",
    ".java": "java", ".kt": "kotlin", ".cs": "csharp", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".cc": "cpp", ".rb": "ruby", ".php": "php", ".swift": "swift", ".dart": "dart", ".lua": "lua",
    ".sh": "bash", ".ps1": "powershell", ".sql": "sql", ".html": "html", ".css": "css", ".scss": "css",
    ".vue": "vue", ".svelte": "svelte",
}
TEXT_EXTRA = {".json", ".toml", ".yaml", ".yml", ".md", ".txt", ".cfg", ".ini", ".xml", ".gradle"}
MAX_FILES = 2000          # files listed in the map
MAX_PARSE_BYTES = 300_000  # bigger files are listed but not parsed for symbols
MAX_SYMBOLS = 6000
MAP_SECONDS = 60.0        # a map older than this is rebuilt on the next use
MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "package.json", "go.mod", "Cargo.toml",
           "pytest.ini", "tox.ini", ".git", "manage.py")


@dataclass
class Symbol:
    name: str     # "parse" or "Parser.parse"
    kind: str     # function, class, method, ...
    file: str     # relative path, "/" separated
    line: int     # 1-based first line
    end: int      # 1-based last line (best effort for non-Python files)


@dataclass
class Project:
    root: Path
    language: str = "unknown"
    languages: dict[str, int] = field(default_factory=dict)  # language -> number of files
    framework: str = ""
    test_runner: str = ""     # pytest, unittest, jest, vitest, mocha, npm, go, cargo or ""
    files: list[str] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    truncated: bool = False
    built: float = 0.0

    @property
    def name(self) -> str:
        return self.root.name

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def contains(self, path: Path) -> bool:
        path = path.resolve()
        return path == self.root or self.root in path.parents

    def find_symbol(self, name: str) -> list[Symbol]:
        """Symbols called `name` ("parse", "Parser.parse" or "Parser"); exact matches first."""
        name = name.strip()
        exact = [s for s in self.symbols if s.name == name or s.name.endswith("." + name)]
        if exact:
            return exact
        low = name.lower()
        return [s for s in self.symbols if low in s.name.lower()][:20]

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for s in self.symbols:
            kinds[s.kind] = kinds.get(s.kind, 0) + 1
        return {
            "project": self.name, "path": str(self.root), "language": self.language,
            "languages": dict(sorted(self.languages.items(), key=lambda kv: -kv[1])[:5]),
            "framework": self.framework or "none detected", "test_runner": self.test_runner or "none detected",
            "files": len(self.files), "symbols": kinds, "map_truncated": self.truncated,
            "top_files": self.files[:25],
        }


# -- detection --------------------------------------------------------------------------------------
def _read(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return ""


def find_root(start: Path, roots: list[Path]) -> Path:
    """The project folder around a file: the nearest parent with a project marker, never above an allowed root."""
    start = start.resolve()
    folder = start if start.is_dir() else start.parent
    probe = folder
    while True:
        if any((probe / m).exists() for m in MARKERS):
            return probe
        if probe in roots or probe.parent == probe or not any(r in probe.parents for r in roots):
            return folder
        probe = probe.parent


def _python_runner(root: Path, files: list[str]) -> str:
    config = " ".join(_read(root / n) for n in ("pyproject.toml", "setup.cfg", "tox.ini", "requirements.txt",
                                                "requirements-dev.txt", "Pipfile"))
    if (root / "pytest.ini").exists() or (root / "conftest.py").exists() or "pytest" in config:
        return "pytest"
    tests = [f for f in files if re.search(r"(^|/)(test_[^/]*|[^/]*_test)\.py$", f)]
    if not tests:
        return "pytest" if any(f.startswith("tests/") for f in files) else ""
    sample = " ".join(_read(root / f, 4000) for f in tests[:5])
    if "import pytest" in sample or "unittest" not in sample:
        return "pytest"
    return "unittest"


def _node_info(root: Path) -> tuple[str, str]:
    try:
        pkg = json.loads(_read(root / "package.json"))
    except ValueError:
        pkg = {}
    deps = {**(pkg.get("dependencies") or {}), **(pkg.get("devDependencies") or {})}
    framework = next((name for key, name in (("next", "Next.js"), ("react", "React"), ("vue", "Vue"),
                                              ("@angular/core", "Angular"), ("svelte", "Svelte"),
                                              ("express", "Express"), ("fastify", "Fastify")) if key in deps), "")
    script = str((pkg.get("scripts") or {}).get("test", ""))
    if not script or "no test specified" in script:
        runner = ""
    elif "vitest" in script or "vitest" in deps:
        runner = "vitest"
    elif "jest" in script or "jest" in deps:
        runner = "jest"
    elif "mocha" in script:
        runner = "mocha"
    else:
        runner = "npm"
    return framework, runner


def _python_framework(root: Path) -> str:
    if (root / "manage.py").exists():
        return "Django"
    text = " ".join(_read(root / n).lower() for n in ("pyproject.toml", "requirements.txt", "setup.py"))
    for key, name in (("django", "Django"), ("fastapi", "FastAPI"), ("flask", "Flask"), ("pyqt6", "PyQt6"),
                      ("streamlit", "Streamlit"), ("pandas", "pandas")):
        if key in text:
            return name
    return ""


def detect(project: Project) -> None:
    root = project.root
    counts: dict[str, int] = {}
    for f in project.files:
        lang = LANGUAGES.get(Path(f).suffix.lower())
        if lang and lang not in {"html", "css"}:
            counts[lang] = counts.get(lang, 0) + 1
    project.languages = counts
    if (root / "Cargo.toml").exists():
        project.language, project.test_runner = "rust", "cargo"
    elif (root / "go.mod").exists():
        project.language, project.test_runner = "go", "go"
    elif (root / "package.json").exists():
        project.language = "typescript" if counts.get("typescript", 0) > counts.get("javascript", 0) else "javascript"
        project.framework, project.test_runner = _node_info(root)
    elif counts:
        project.language = max(counts, key=counts.get)
    if project.language == "python":
        project.test_runner = _python_runner(root, project.files)
        project.framework = _python_framework(root)


# -- symbols ----------------------------------------------------------------------------------------
def python_symbols(text: str, rel: str) -> list[Symbol]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    out: list[Symbol] = []

    def walk(nodes, prefix: str = "", in_class: bool = False) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                kind = "class" if isinstance(node, ast.ClassDef) else ("method" if in_class else "function")
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                name = prefix + node.name
                out.append(Symbol(name, kind, rel, start, getattr(node, "end_lineno", None) or node.lineno))
                if isinstance(node, ast.ClassDef):
                    walk(node.body, name + ".", True)

    walk(tree.body)
    return out


_GENERIC = [  # (kind, pattern with a "name" group) for languages without a parser here
    ("class", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?(?:public\s+|private\s+|internal\s+)?"
                         r"(?:static\s+)?(?:class|interface|struct|enum|trait|impl|type)\s+(?P<name>[A-Za-z_]\w*)")),
    ("function", re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)")),
    ("function", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
                            r"(?:async\s+)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>")),
    ("function", re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?(?P<name>[A-Za-z_]\w*)\s*\(")),
    ("function", re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(?P<name>[A-Za-z_]\w*)")),
    ("function", re.compile(r"^\s*def\s+(?P<name>[A-Za-z_]\w*[?!]?)")),  # ruby
    ("method", re.compile(r"^\s*(?:public|private|protected|internal)\s+(?:static\s+)?(?:async\s+)?(?:final\s+)?"
                          r"[\w<>\[\],.? ]+\s+(?P<name>[A-Za-z_]\w*)\s*\([^;]*$")),
]


def block_end(lines: list[str], start: int, limit: int = 400) -> int:
    """Last line (1-based) of a brace block that starts at line `start`; the start line if there is no block."""
    depth, opened = 0, False
    for i in range(start - 1, min(len(lines), start - 1 + limit)):
        line = re.sub(r"(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|//.*$)", "", lines[i])
        for ch in line:
            if ch == "{":
                depth, opened = depth + 1, True
            elif ch == "}":
                depth -= 1
        if opened and depth <= 0:
            return i + 1
        if not opened and i > start + 2:  # no brace near the start: a one-liner or a declaration
            return start
    return min(len(lines), start - 1 + limit)


def generic_symbols(text: str, rel: str) -> list[Symbol]:
    lines = text.splitlines()
    out: list[Symbol] = []
    for n, line in enumerate(lines, 1):
        if len(line) > 300:
            continue
        for kind, pattern in _GENERIC:
            m = pattern.match(line)
            if m and m.group("name") not in {"if", "for", "while", "switch", "return", "new", "catch"}:
                out.append(Symbol(m.group("name"), kind, rel, n, block_end(lines, n)))
                break
    return out


def symbols_for(text: str, rel: str) -> list[Symbol]:
    return python_symbols(text, rel) if rel.endswith((".py", ".pyw")) else generic_symbols(text, rel)


# -- map --------------------------------------------------------------------------------------------
def is_code(path: Path) -> bool:
    return path.suffix.lower() in LANGUAGES or path.suffix.lower() in TEXT_EXTRA


def walk_files(root: Path, limit: int = MAX_FILES) -> tuple[list[Path], bool]:
    """Code and text files in the project, skipping dependencies, builds and secret files. (files, truncated)"""
    from jarvis.guardrails import sensitive_path

    found: list[Path] = []
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        subdirs = []
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name not in SKIP_DIRS and not entry.name.endswith(".egg-info"):
                    subdirs.append(entry)
            elif is_code(entry) and not sensitive_path(entry):
                found.append(entry)
                if len(found) >= limit:
                    return found, True
        stack.extend(reversed(subdirs))
    return found, False


def build(root: Path) -> Project:
    project = Project(root.resolve())
    paths, project.truncated = walk_files(project.root, MAX_FILES)
    project.files = [project.rel(p) for p in paths]
    for path, rel in zip(paths, project.files):
        if path.suffix.lower() not in LANGUAGES or len(project.symbols) >= MAX_SYMBOLS:
            continue
        try:
            if path.stat().st_size > MAX_PARSE_BYTES:
                continue
        except OSError:
            continue
        project.symbols.extend(symbols_for(_read(path, MAX_PARSE_BYTES), rel))
    project.symbols = project.symbols[:MAX_SYMBOLS]
    detect(project)
    project.built = time.monotonic()
    return project


# -- the active project ----------------------------------------------------------------------------
_lock = threading.Lock()
_cache: dict[Path, Project] = {}


def _store():
    from jarvis.datafile import store

    return store("coding_project", {"path": ""})


def load(root: Path, fresh: bool = False) -> Project:
    """The (cached) map of a project folder."""
    root = root.resolve()
    with _lock:
        cached = _cache.get(root)
        if cached and not fresh and time.monotonic() - cached.built < MAP_SECONDS:
            return cached
    project = build(root)
    with _lock:
        _cache[root] = project
        while len(_cache) > 4:  # keep only a few maps in memory
            _cache.pop(next(iter(_cache)))
    return project


def set_active(root: Path) -> None:
    _store().save({"path": str(root.resolve())})


def active_root() -> Path | None:
    raw = _store().load().get("path") or ""
    path = Path(raw) if raw else None
    return path if path and path.is_dir() else None


def forget_cache() -> None:
    with _lock:
        _cache.clear()
