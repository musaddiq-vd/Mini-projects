"""Open apps, websites and YouTube - without passing anything through a shell."""

from __future__ import annotations

import difflib
import os
import platform
import re
import shutil
import subprocess
import urllib.parse
import webbrowser
from pathlib import Path

from jarvis.tools import tool

OS = platform.system()

# Friendly names -> what to launch (Windows). URIs open through the OS handler.
WINDOWS_ALIASES = {
    "calculator": "calc.exe",
    "notepad": "notepad.exe",
    "paint": "mspaint.exe",
    "file explorer": "explorer.exe",
    "explorer": "explorer.exe",
    "command prompt": "cmd.exe",
    "terminal": "wt.exe",
    "task manager": "taskmgr.exe",
    "settings": "ms-settings:",
    "camera": "microsoft.windows.camera:",
    "store": "ms-windows-store:",
    "mail": "outlookmail:",
    "calendar": "outlookcal:",
}


def _start_menu_shortcuts() -> dict[str, Path]:
    roots = [
        Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
        Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    ]
    found: dict[str, Path] = {}
    for root in roots:
        if root.is_dir():
            for lnk in root.rglob("*.lnk"):
                found.setdefault(lnk.stem.lower(), lnk)
    return found


def _open_windows(name: str) -> str | None:
    key = name.lower().strip()
    target = WINDOWS_ALIASES.get(key)
    if target:
        os.startfile(target)  # noqa: S606 - fixed, known targets
        return name
    shortcuts = _start_menu_shortcuts()
    match = shortcuts.get(key)
    if not match:
        close = difflib.get_close_matches(key, shortcuts.keys(), n=1, cutoff=0.6)
        if not close:
            close = [k for k in shortcuts if key in k][:1]
        match = shortcuts[close[0]] if close else None
    if match:
        os.startfile(match)  # noqa: S606 - a Start Menu shortcut
        return match.stem
    exe = shutil.which(key) or shutil.which(key.replace(" ", ""))
    if exe:
        subprocess.Popen([exe])
        return Path(exe).stem
    return None


def _open_mac(name: str) -> str | None:
    result = subprocess.run(["open", "-a", name], capture_output=True)
    return name if result.returncode == 0 else None


def _open_linux(name: str) -> str | None:
    for candidate in (name, name.lower(), name.lower().replace(" ", "-")):
        exe = shutil.which(candidate)
        if exe:
            subprocess.Popen([exe], start_new_session=True)
            return candidate
    if shutil.which("gtk-launch"):
        result = subprocess.run(["gtk-launch", name.lower()], capture_output=True)
        if result.returncode == 0:
            return name
    return None


@tool(
    "open_app",
    "Open an installed application by name, e.g. 'chrome', 'spotify', 'calculator'.",
    {"name": {"type": "string", "description": "App name as the user said it"}},
    required=["name"],
)
def open_app(ctx, name: str):
    opener = {"Windows": _open_windows, "Darwin": _open_mac}.get(OS, _open_linux)
    opened = opener(name)
    if not opened:
        return {"error": f"Could not find an app called '{name}'."}
    return f"Opened {opened}."


@tool(
    "open_website",
    "Open a website or URL in the default browser.",
    {"url": {"type": "string", "description": "Full URL or a domain like 'github.com'"}},
    required=["url"],
)
def open_website(ctx, url: str):
    url = url.strip()
    if "://" not in url and re.match(r"^[a-zA-Z][\w+.-]*:(?!\d)", url):
        return {"error": "Only http(s) web addresses can be opened."}  # e.g. javascript:, mailto:
    parsed = urllib.parse.urlparse(url if "://" in url else f"https://{url}")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"error": "Only http(s) web addresses can be opened."}
    webbrowser.open(parsed.geturl())
    return f"Opened {parsed.netloc}."


def first_youtube_video(query: str) -> str | None:
    """Video id of the top YouTube search result (no API key needed)."""
    import requests

    r = requests.get(
        "https://www.youtube.com/results",
        params={"search_query": query},
        headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en"},
        timeout=10,
    )
    match = re.search(r'"videoId":"([\w-]{11})"', r.text)
    return match.group(1) if match else None


@tool(
    "play_youtube",
    "Play a video or song on YouTube: opens the top result directly. Set results_only to just show the search.",
    {"query": {"type": "string"}, "results_only": {"type": "boolean"}},
    required=["query"],
)
def play_youtube(ctx, query: str, results_only: bool = False):
    video = None
    if not results_only:
        try:
            video = first_youtube_video(query)
        except Exception:
            video = None
    if video:
        webbrowser.open(f"https://www.youtube.com/watch?v={video}")
        return f"Playing '{query}' on YouTube."
    webbrowser.open("https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query))
    return f"Opened YouTube results for '{query}'."
