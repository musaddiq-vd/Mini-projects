"""Chat panel: safe formatting, day lines, search, copy, export."""

import sys
from datetime import datetime, timedelta

import pytest

from jarvis.ui.richtext import has_markup, plain, to_html


def test_formatting():
    html = to_html("Use **sorted()** or `list.sort()`\n- one\n2) two\nsee https://example.com/a?b=1.")
    assert "<b>sorted()</b>" in html and "Consolas" in html and "list.sort()" in html
    assert "&nbsp;•&nbsp;one" in html and "2)&nbsp;two" in html
    assert '<a href="https://example.com/a?b=1"' in html  # trailing full stop not part of the link


def test_formatting_cannot_inject_html():
    html = to_html('<img src=x onerror=alert(1)> **<b>hi</b>**')
    assert "<img" not in html and "&lt;img" in html and "&lt;b&gt;hi&lt;/b&gt;" in html


def test_code_is_not_styled_inside():
    assert "<b>" not in to_html("`**not bold**`")


def test_has_markup_and_plain():
    assert not has_markup("just a normal sentence, 2 * 3 = 6")
    assert has_markup("**yes**") and has_markup("- item")
    assert plain("**bold** and `code`") == "bold and code"


@pytest.fixture
def window(monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings())
    w = MainWindow(app, Bridge())
    w.qt = qt
    return w, app


def test_day_lines(window):
    w, app = window
    w.chat.add_message("You", "old one", True, when=datetime.now() - timedelta(days=1))
    w.chat.add_message("You", "new one", True)
    w.chat.add_message("You", "another", True)
    text = w.chat.toPlainText()
    assert "YESTERDAY" in text and text.count("TODAY") == 1


def test_rich_message_copies_as_plain(window):
    w, app = window
    w.chat.add_message("Jarvis", "**Hello** see https://a.b", False)
    assert "**Hello** see https://a.b" in w.chat.toPlainText()


def test_search_filters_rows(window):
    w, app = window
    for t in ("buy milk", "call the bank", "milk prices"):
        w.chat.add_message("You", t, True)
    w.open_search()
    w.search.setText("milk")
    assert w.search_count.text() == "2 matches"
    visible = [r for i in range(1, w.chat.col.count()) if (r := w.chat.col.itemAt(i).widget()) and not r.isHidden()]
    assert len(visible) == 2
    w.close_search()
    assert all(not w.chat.col.itemAt(i).widget().isHidden() for i in range(1, w.chat.col.count()))


def test_export(window, tmp_path, monkeypatch):
    from pathlib import Path

    w, app = window
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert w.export_chat() is None  # nothing yet
    app.events.emit("user", text="hello")
    app.events.emit("assistant", text="hi there")
    path = w.export_chat()
    body = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path / "Documents" / "Jarvis Chats"
    assert "**You**" in body and "hi there" in body


def test_typewriter_ends_with_formatting(window):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QLabel

    from jarvis.ui.fx import Typewriter

    w, app = window
    label = QLabel()
    tw = Typewriter(label, "**bold**", final_html="<b>bold</b>")
    assert label.textFormat() == Qt.TextFormat.PlainText
    tw.finish()
    assert label.textFormat() == Qt.TextFormat.RichText and label.text() == "<b>bold</b>"
