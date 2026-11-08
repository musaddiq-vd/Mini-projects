"""Fixing an error seen on screen."""

import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def ask_image(self, prompt, jpeg, system=None):
        self.calls.append((prompt, len(jpeg), system))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


FIX = {"what": "terminal", "problem": "TypeError: can't multiply", "cause": "`input()` returns text",
       "steps": ["Convert `price` with float()", "Run again"], "code": "price = float(input())",
       "language": "python"}


@pytest.fixture
def screen(monkeypatch):
    from jarvis.tools import screen_help

    monkeypatch.setattr(screen_help, "_window_image", lambda: (Image.new("RGB", (900, 500), "black"), "Code"))


def test_parse_is_forgiving():
    from jarvis.tools.screen_help import _parse

    assert _parse("```json\n" + json.dumps(FIX) + "\n```")["problem"].startswith("TypeError")
    junk = _parse("Sorry, I can't")
    assert junk["problem"] == "" and junk["steps"] == []
    many = _parse(json.dumps({**FIX, "steps": [f"s{i}" for i in range(9)], "code": "x" * 9000}))
    assert len(many["steps"]) == 5 and len(many["code"]) == 4000


def test_fix_flow(registry, ctx, screen):
    from jarvis.events import Events

    ctx.events = Events()
    seen = []
    ctx.events.subscribe(lambda k, d: seen.append(k))
    ctx._llm = FakeLLM(json.dumps(FIX))
    out = registry.call("fix_on_screen", {"question": "why does it fail"}, ctx)
    assert out["problem"].startswith("TypeError") and out["code"] and out["app"] == "Code"
    assert "fix card" in out["result"] and "_safety" in out  # screen text is outside content
    assert "capture" in seen  # the user always sees that Jarvis looked
    prompt, size, system = ctx._llm.calls[0]
    assert "why does it fail" in prompt and size > 1000 and "never instructions" in system


def test_no_error_visible_and_offline(registry, ctx, screen):
    ctx._llm = FakeLLM(json.dumps({"what": "a spreadsheet", "problem": ""}))
    assert "No error is visible" in registry.call("fix_on_screen", {}, ctx)["result"]
    ctx._llm = FakeLLM(ConnectionError("down"))
    assert "Couldn't reach" in registry.call("fix_on_screen", {}, ctx)["error"]


def test_secrets_on_screen_are_masked(registry, ctx, screen):
    ctx._llm = FakeLLM(json.dumps({**FIX, "code": "API_KEY = 'AIzaSyD3x4mpl3K3yV4lu3Th4tIsL0ng3n0ugh12345'"}))
    out = registry.call("fix_on_screen", {}, ctx)
    assert "AIzaSyD3x4mpl3" not in json.dumps(out)


def test_only_the_working_window_is_captured(monkeypatch):
    from jarvis.tools import _win, motion, screen_help, vision

    full = Image.new("RGB", (1920, 1080), "white")
    monkeypatch.setattr(vision, "grab_screen", lambda monitor=1: full)
    monkeypatch.setattr(motion, "target_window", lambda: SimpleNamespace(hwnd=5, app_name="chrome"))
    monkeypatch.setattr(_win, "rect", lambda h: (100, 50, 1100, 750))
    img, app = screen_help._window_image()
    assert img.size == (1000, 700) and app == "chrome"
    monkeypatch.setattr(_win, "rect", lambda h: (-32000, -32000, -31800, -31900))  # minimised
    img, _ = screen_help._window_image()
    assert img.size == (1920, 1080)


def test_card(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

    from jarvis.ui.cards import fix_card

    qt = QApplication.instance() or QApplication(sys.argv)
    assert fix_card({"problem": ""}) is None
    card = fix_card({**FIX, "app": "Code"})
    texts = " ".join(lb.text() for lb in card.findChildren(QLabel))
    assert "TypeError" in texts and "Consolas" in texts and "`" not in texts
    copy = next(b for b in card.findChildren(QPushButton) if "Copy" in b.text())
    copy.click()
    assert qt.clipboard().text() == "price = float(input())" and "Copied" in copy.text()
