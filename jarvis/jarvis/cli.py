"""Command line entry point:  python -m jarvis [--text | --setup | --tools | --devices]"""

from __future__ import annotations

import argparse
import sys

from jarvis import __author__, __version__
from jarvis.config import VOICES, load_settings, save_settings, settings_path


def _setup() -> int:
    try:
        return _setup_steps()
    except (KeyboardInterrupt, EOFError):
        print("\nSetup cancelled - nothing was saved.")
        return 1


def _setup_steps() -> int:
    s = load_settings()
    print(f"Jarvis setup - settings are saved to {settings_path()}\n")
    current = f" [current: ...{s.api_key[-4:]}]" if s.api_key else ""
    # Plain input (not getpass): pasting into a hidden prompt fails in many Windows terminals.
    key = input(f"Gemini API key - paste it and press Enter{current}: ").strip()
    if key:
        s.api_key = key
        print("\033[1A\033[2K" + f"Gemini API key: saved (...{key[-4:]})")  # hide the key from the screen
    s.user_name = input(f"Your name [{s.user_name}]: ").strip() or s.user_name
    s.assistant_name = input(f"Assistant name [{s.assistant_name}]: ").strip() or s.assistant_name
    voice = input(f"Voice {VOICES} [{s.voice}]: ").strip()
    if voice in VOICES:
        s.voice = voice
    wake = input(f"Wake word model, e.g. hey_jarvis (blank = always listening) [{s.wake_word}]: ").strip()
    if wake:
        s.wake_word = "" if wake.lower() in {"off", "none", "-"} else wake
    save_settings(s)
    print("\nSaved. Run:  python -m jarvis")
    return 0


def _list_tools() -> int:
    from jarvis.app import create_app

    app = create_app()
    for t in app.registry.tools.values():
        print(f"{t.name:18} {t.description}")
    for err in app.registry.errors:
        print(f"[warning] {err}")
    return 0


def _list_devices() -> int:
    from jarvis.audio import list_devices

    for d in list_devices():
        kind = ("in " if d["inputs"] else "   ") + ("out" if d["outputs"] else "   ")
        star = "*" if d["default"] else " "
        print(f"{star} {d['index']:3} [{kind}] {d['name']}  ({d['host']})")
    print("\nPut a name (or part of it) in input_device / output_device in config.json.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description=f"Jarvis voice assistant by {__author__}")
    parser.add_argument("--cli", action="store_true", help="voice in the terminal, no window")
    parser.add_argument("--text", action="store_true", help="typed chat in the terminal")
    parser.add_argument("--setup", action="store_true", help="set API key, names and voice")
    parser.add_argument("--tools", action="store_true", help="list available tools")
    parser.add_argument("--devices", action="store_true", help="list microphones and speakers")
    parser.add_argument("--interrupt", action="store_true",
                        help="let you talk over Jarvis (use headphones, or it hears itself)")
    parser.add_argument("--version", action="version", version=f"jarvis {__version__}")
    args = parser.parse_args(argv)

    if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(errors="replace")  # never crash on an emoji
        except AttributeError:
            pass

    if args.setup:
        return _setup()
    if args.tools:
        return _list_tools()
    if args.devices:
        return _list_devices()

    from jarvis.net import prefer_ipv4

    prefer_ipv4()
    if not (args.cli or args.text):
        from jarvis.ui.window import run_window

        return run_window()

    from jarvis.app import create_app

    app = create_app()
    if not app.settings.effective_api_key:
        print("No Gemini API key yet. Run:  python -m jarvis --setup")
        print("(or set the GEMINI_API_KEY environment variable)")
        return 1
    if args.interrupt:
        app.settings.allow_interrupt = True
    for err in app.registry.errors:
        print(f"[warning] {err}")

    if args.text:
        from jarvis.text_chat import run_text_chat

        run_text_chat(app)
    else:
        from jarvis.live import run_voice

        run_voice(app)
    return 0
