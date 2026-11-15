"""Clipboard history, PDF tools, reading text in pictures and the daily digest."""

import io

import pytest

from jarvis.tools import clipboard_history as ch
from jarvis.tools.pdf_tools import free_name, parse_pages


@pytest.fixture(autouse=True)
def fresh_history():
    ch.clear()
    yield
    ch.clear()


# -- clipboard history --------------------------------------------------------------------------------
def test_history_keeps_recent_and_skips_secrets_and_repeats():
    assert ch.note("hello")
    assert not ch.note("hello")  # same as the last one
    assert not ch.note("card 4111 1111 1111 1111")
    assert not ch.note("   ")
    assert not ch.note("x" * (ch.MAX_ITEM + 1))
    ch.note("world")
    ch.note("hello")  # copied again: moves to the top
    assert [i["text"] for i in ch.items()] == ["hello", "world"]
    for n in range(30):
        ch.note(f"item {n}")
    assert len(ch.items()) == ch.KEEP


def test_history_tools(registry, ctx, monkeypatch):
    board = {"text": "now"}
    monkeypatch.setattr(ch, "_paste", lambda: board["text"])
    monkeypatch.setattr(ch, "_copy", lambda t: board.update(text=t))
    assert "Nothing" in registry.call("clipboard_history", {}, ctx)["result"]
    ch.note("first")
    ch.note("second")
    out = registry.call("clipboard_history", {}, ctx)
    assert [i["text"] for i in out["items"]] == ["second", "first"] and "_safety" in out
    assert "item 2" in registry.call("copy_from_history", {"number": 2}, ctx)["result"]
    assert board["text"] == "first"
    ctx.undo.undo()
    assert board["text"] == "now"
    assert "error" in registry.call("copy_from_history", {"number": 9}, ctx)
    registry.call("clear_clipboard_history", {}, ctx)
    assert ch.items() == []


# -- PDFs ---------------------------------------------------------------------------------------------
def _make_pdf(path, pages):
    from pypdf import PdfWriter

    w = PdfWriter()
    for i in range(pages):
        w.add_blank_page(width=200 + i, height=300)
    with open(path, "wb") as f:
        w.write(f)
    return path


def _pages(path):
    from pypdf import PdfReader

    return len(PdfReader(str(path)).pages)


def test_parse_pages():
    assert parse_pages("1-3, 5", 10) == [0, 1, 2, 4]
    assert parse_pages("8-", 10) == [7, 8, 9]
    assert parse_pages("last", 4) == [3]
    assert parse_pages("odd", 5) == [0, 2, 4]
    assert parse_pages("2 to 3", 5) == [1, 2]
    assert parse_pages("all", 3) == [0, 1, 2]
    with pytest.raises(ValueError, match="outside"):
        parse_pages("4-9", 5)
    with pytest.raises(ValueError, match="understand"):
        parse_pages("page one", 5)


def test_merge_split_rotate_compress(registry, ctx, sandbox):
    from pypdf import PdfReader

    a = _make_pdf(sandbox / "a.pdf", 2)
    b = _make_pdf(sandbox / "b.pdf", 3)
    out = registry.call("merge_pdfs", {"files": [str(a), str(b)]}, ctx)
    assert out["pages"] == 5 and _pages(out["path"]) == 5
    again = registry.call("merge_pdfs", {"files": [str(a), str(b)]}, ctx)
    assert again["path"].endswith("merged (2).pdf")  # never overwrites
    part = registry.call("extract_pdf_pages", {"path": str(b), "pages": "2-3"}, ctx)
    assert _pages(part["path"]) == 2
    each = registry.call("extract_pdf_pages", {"path": str(a), "every_page": True}, ctx)
    assert len(each["files"]) == 2
    rot = registry.call("rotate_pdf", {"path": str(a), "degrees": 90, "pages": "1"}, ctx)
    r = PdfReader(rot["path"])
    assert r.pages[0].rotation == 90 and r.pages[1].rotation == 0
    small = registry.call("compress_pdf", {"path": str(b)}, ctx)
    assert small["path"].endswith("smaller.pdf") and "saved_percent" in small


def test_pdf_errors_and_folders(registry, ctx, sandbox, tmp_path):
    a = _make_pdf(sandbox / "a.pdf", 1)
    assert "two" in registry.call("merge_pdfs", {"files": [str(a)]}, ctx)["error"]
    (sandbox / "notes.txt").write_text("hi")
    assert "isn't a PDF" in registry.call("rotate_pdf", {"path": str(sandbox / "notes.txt")}, ctx)["error"]
    outside = _make_pdf(tmp_path / "outside.pdf", 1)
    assert "outside" in registry.call("compress_pdf", {"path": str(outside)}, ctx)["error"]
    assert "outside this PDF" in registry.call("extract_pdf_pages", {"path": str(a), "pages": "3"}, ctx)["error"]
    assert "Not found" in registry.call("compress_pdf", {"path": str(sandbox / "nope.pdf")}, ctx)["error"]


def test_pdf_undo_moves_new_file_away(registry, ctx, sandbox, monkeypatch):
    import send2trash

    gone = []
    monkeypatch.setattr(send2trash, "send2trash", lambda p: gone.append(p))
    a = _make_pdf(sandbox / "a.pdf", 2)
    out = registry.call("rotate_pdf", {"path": str(a)}, ctx)
    ctx.undo.undo()
    assert gone == [out["path"]]


def test_free_name(tmp_path):
    p = tmp_path / "x.pdf"
    assert free_name(p) == p
    p.write_text("")
    assert free_name(p).name == "x (2).pdf"


# -- text in pictures ---------------------------------------------------------------------------------
class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.got = None

    def ask_image(self, prompt, jpeg, system=None):
        self.got = jpeg
        return self.answer


def test_read_image_text(registry, ctx, sandbox, monkeypatch):
    import pyperclip
    from PIL import Image

    Image.new("RGB", (4000, 3000), "white").save(sandbox / "bill.png")
    ctx._llm = FakeLLM("Total: 450\nCard 4111 1111 1111 1111 was used")
    copied = {}
    monkeypatch.setattr(pyperclip, "copy", lambda t: copied.update(text=t))
    out = registry.call("read_image_text", {"path": str(sandbox / "bill.png"), "copy": True}, ctx)
    assert out["text"].startswith("Total: 450") and "_safety" in out
    assert "4111" not in out["text"] and "4111" not in copied["text"]
    assert max(Image.open(io.BytesIO(ctx._llm.got)).size) <= 2400  # scaled down before sending
    (sandbox / "a.pdf").write_text("x")
    assert "ask_document" in registry.call("read_image_text", {"path": str(sandbox / "a.pdf")}, ctx)["error"]


def test_read_image_text_from_clipboard(registry, ctx, monkeypatch):
    from PIL import Image

    from jarvis.tools import image_text

    ctx._llm = FakeLLM("")
    monkeypatch.setattr(image_text, "_clipboard_image", lambda: None)
    assert "no picture" in registry.call("read_image_text", {"source": "clipboard"}, ctx)["error"]
    monkeypatch.setattr(image_text, "_clipboard_image", lambda: Image.new("RGB", (50, 50)))
    assert "couldn't find any text" in registry.call("read_image_text", {"source": "clipboard"}, ctx)["result"]


# -- daily digest -------------------------------------------------------------------------------------
def _with_state(ctx):
    from jarvis.state import StateStore

    ctx.state = StateStore(ctx.memory.path.parent / "state.json")
    return ctx


def test_digest_empty_day(registry, ctx):
    out = registry.call("daily_digest", {"weather": False}, _with_state(ctx))
    assert out["digest"] and "clear day" in out["result"]


def test_digest_collects_the_day(registry, ctx, monkeypatch):
    from jarvis.tools import web

    _with_state(ctx)
    ctx.settings.user_name = "Musaddiq"
    ctx.memory.remember("city", "Pune")
    monkeypatch.setattr(web, "get_weather", lambda ctx, city: {"place": city, "now": "clear sky",
                                                               "temperature_c": 28, "today_max_c": 31,
                                                               "today_min_c": 19, "rain_chance_percent": 0})
    registry.call("add_note", {"text": "Pay the bill", "due": "today", "priority": "high"}, ctx)
    out = registry.call("daily_digest", {}, ctx)
    assert out["greeting"].endswith("Musaddiq")
    assert out["weather"]["place"] == "Pune"
    assert out["todos"]["first"] == ["Pay the bill"] and out["todos"]["due_today"] == 1
    assert "result" not in out


@pytest.fixture
def qt():
    import sys

    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def test_digest_card_draws(qt):
    from PyQt6.QtWidgets import QLabel

    from jarvis.ui.cards import digest_card

    card = digest_card({"digest": "Sunday 15 November", "greeting": "Good morning",
                        "weather": {"place": "Pune", "now": "clear sky", "temperature_c": 28},
                        "meetings": [{"title": "Standup", "time": "10:00 AM - 10:15 AM"}],
                        "todos": {"first": ["Pay the bill"], "open": 3, "overdue": 1, "due_today": 1},
                        "reminders": ["6:30 PM: Call mom"], "habits": [{"habit": "✅ Gym", "today": "0/1", "streak": 4}],
                        "spent_today": "₹450"})
    text = " ".join(lbl.text() for lbl in card.findChildren(QLabel))
    assert "Standup" in text and "Pay the bill" in text and "₹450" in text and "1 overdue" in text
    assert digest_card({"no": "digest"}) is None
