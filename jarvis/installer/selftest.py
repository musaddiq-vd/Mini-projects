"""Check a packaged or installed Jarvis without opening anything on screen.

    runtime\\python.exe app\\installer\\selftest.py report.json

Loads every tool and the heavy libraries, builds the main window off screen, and writes what worked to
report.json (exit code 0 = all good). Run with QT_QPA_PLATFORM=offscreen and a temporary JARVIS_HOME so it
never touches the user's data.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LIBRARIES = ("PyQt6.QtWidgets", "PyQt6.QtNetwork", "google.genai", "numpy", "sounddevice", "openwakeword", "pypdf",
             "docx", "openpyxl", "pptx", "PIL", "mss", "pyautogui", "tinytuya", "ddgs", "psutil", "requests")


def run(path: str) -> int:
    report: dict = {"errors": []}
    app_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(app_dir))
    try:
        from jarvis import __version__
        from jarvis.tools import build_registry

        reg = build_registry()
        report.update(version=__version__, python=sys.version.split()[0], tools=len(reg.tools),
                      tool_errors=reg.errors)
        for mod in LIBRARIES:
            try:
                __import__(mod)
            except Exception as exc:
                report["errors"].append(f"{mod}: {type(exc).__name__}: {exc}")
        from zoneinfo import ZoneInfo

        ZoneInfo("Asia/Kolkata")  # time zones come from the tzdata package
        import openwakeword

        report["wake_models"] = bool(openwakeword.get_pretrained_model_paths())
        from jarvis.app import REPO_PLUGINS, create_app

        report["plugins_folder"] = REPO_PLUGINS.is_dir()
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication

        from jarvis.config import Settings
        from jarvis.ui.window import Bridge, MainWindow

        qt = QApplication.instance() or QApplication(sys.argv)
        window = MainWindow(create_app(Settings()), Bridge())
        window.show()
        qt.processEvents()
        report["window"] = window.isVisible()
        window.reminders.close_all()
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["ok"] = (not report["errors"] and not report.get("tool_errors") and report.get("tools", 0) >= 90
                    and report.get("wake_models") and report.get("plugins_folder") and report.get("window"))
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else "selftest.json"))
