"""No key-shaped text in the repository: GitHub's secret scanning flags even fake keys in tests.

Tests that need a fake secret build it from pieces at run time (see FAKE_GOOGLE_KEY in test_guardrails.py).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the patterns are themselves built from pieces, so this file doesn't trip the scanner either
PATTERNS = {
    "Google API key": "AI" + r"za[0-9A-Za-z_\-]{35}",
    "OpenAI key": "sk" + r"-(?:proj-)?[A-Za-z0-9]{20,}",
    "GitHub token": "gh" + r"p_[A-Za-z0-9]{36}",
    "AWS key": "AK" + r"IA[0-9A-Z]{16}",
    "private key": "-----BEGIN [A-Z ]*PRI" + "VATE KEY-----",
    "Telegram bot token": r"\b\d{8,10}:" + "AA" + r"[A-Za-z0-9_-]{33}\b",
    "Slack token": "xo" + r"x[abprs]-[A-Za-z0-9-]{10,}",
}
SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", ".pytest_cache", ".ruff_cache", "build", "dist"}
TEXT = {".py", ".pyw", ".md", ".txt", ".toml", ".json", ".cfg", ".ini", ".yml", ".yaml", ".iss", ".ps1", ".bat"}


def test_no_key_shaped_text_in_the_repo():
    found = []
    for path in ROOT.rglob("*"):
        if path.is_dir() or path.suffix.lower() not in TEXT or SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            for m in re.finditer(pattern, text):
                line = text.count("\n", 0, m.start()) + 1
                found.append(f"{path.relative_to(ROOT)}:{line} looks like a {name}")
    assert not found, "Build fake secrets from pieces instead:\n" + "\n".join(found)
