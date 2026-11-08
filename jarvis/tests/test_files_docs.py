"""Phase B: smart file search and chatting with documents."""

import os
import sys
import time
from datetime import datetime, timedelta

import pytest


def make_pdf(path, pages: list[str]) -> None:
    """A tiny real PDF with one line of text per page (enough for pypdf to extract)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        content_id = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(out)


@pytest.fixture
def files(sandbox):
    import docx

    from jarvis.tools import filesearch

    filesearch._cache.clear()
    (sandbox / "Downloads").mkdir()
    make_pdf(sandbox / "Downloads" / "scan_0042.pdf", ["Invoice number 7781", "Total amount Rs 4500 due 30 Nov"])
    d = docx.Document()
    d.add_paragraph("Project proposal for the Goa trip budget")
    d.save(sandbox / "proposal.docx")
    (sandbox / "notes.txt").write_text("shopping list: milk, bread", encoding="utf-8")
    (sandbox / "holiday photo.jpg").write_bytes(b"\xff\xd8")
    secret = sandbox / ".ssh"
    secret.mkdir()
    (secret / "invoice_key.txt").write_text("invoice secret", encoding="utf-8")
    old = sandbox / "old_invoice.pdf"
    make_pdf(old, ["Invoice from last year"])
    ten_days = time.time() - 10 * 86400
    os.utime(old, (ten_days, ten_days))
    return sandbox


def names(out):
    return [f["name"] for f in out.get("files", [])]


def test_finds_by_text_inside_a_pdf(registry, ctx, files):
    out = registry.call("find_files", {"query": "kal wala invoice", "file_type": "pdf"}, ctx)
    assert names(out)[0] in ("scan_0042.pdf", "old_invoice.pdf")
    hit = next(f for f in out["files"] if f["name"] == "scan_0042.pdf")
    assert hit["matched"].startswith("inside:") and "Invoice" in hit["matched"]


def test_date_filter(registry, ctx, files):
    out = registry.call("find_files", {"query": "invoice", "modified": "today"}, ctx)
    assert "scan_0042.pdf" in names(out) and "old_invoice.pdf" not in names(out)
    out = registry.call("find_files", {"query": "invoice", "modified": "this month"}, ctx)
    when = (datetime.now() - timedelta(days=10)).month == datetime.now().month
    assert ("old_invoice.pdf" in names(out)) == when


def test_name_match_and_word_files(registry, ctx, files):
    assert names(registry.call("find_files", {"query": "proposal"}, ctx)) == ["proposal.docx"]
    assert "proposal.docx" in names(registry.call("find_files", {"query": "goa trip"}, ctx))
    assert names(registry.call("find_files", {"file_type": "image"}, ctx)) == ["holiday photo.jpg"]


def test_secret_folders_are_never_searched(registry, ctx, files):
    assert "invoice_key.txt" not in names(registry.call("find_files", {"query": "invoice"}, ctx))


def test_search_errors(registry, ctx, files):
    assert "error" in registry.call("find_files", {}, ctx)
    assert "error" in registry.call("find_files", {"query": "x", "modified": "blue moon"}, ctx)
    assert "No matching" in registry.call("find_files", {"query": "zebra unicorn"}, ctx)["result"]


# -- document chat -------------------------------------------------------------------------------
class FakeLLM:
    def __init__(self):
        self.prompts = []

    def ask(self, prompt, system=None):
        self.prompts.append(prompt)
        return "The total is Rs 4500 (p. 2)."


def test_open_and_ask(registry, ctx, files, tmp_path):
    from jarvis.events import Events
    from jarvis.state import StateStore

    ctx.state, ctx.events, ctx._llm = StateStore(tmp_path / "s.json"), Events(), FakeLLM()
    seen = []
    ctx.events.subscribe(lambda k, d: seen.append(d.get("name")) if k == "document" else None)
    pdf = str(files / "Downloads" / "scan_0042.pdf")
    opened = registry.call("open_document", {"path": pdf}, ctx)
    assert opened["document"] == "scan_0042.pdf" and seen == ["scan_0042.pdf"]
    out = registry.call("ask_document", {"question": "What is the total amount?"}, ctx)
    assert "4500" in out["answer"] and "p. 2" in out["pages"]
    assert "[p. 2]" in ctx._llm.prompts[-1]  # the right page was sent
    registry.call("close_document", {}, ctx)
    assert "error" in registry.call("ask_document", {"question": "total?"}, ctx) and seen[-1] == ""


def test_question_not_in_document(registry, ctx, files, tmp_path):
    from jarvis.state import StateStore

    ctx.state, ctx._llm = StateStore(tmp_path / "s.json"), FakeLLM()
    out = registry.call("ask_document", {"question": "Who won the football world cup?",
                                         "path": str(files / "proposal.docx")}, ctx)
    assert "doesn't seem to be in the document" in out["result"] and not ctx._llm.prompts


def test_summary_uses_spread_chunks():
    from jarvis.tools.docchat import rank

    chunks = [(f"p. {i}", f"text {i}") for i in range(1, 41)]
    picked = rank(chunks, "summarize this")
    assert len(picked) == 8 and picked[0][0] == "p. 1" and picked[-1][0] != "p. 8"


def test_scanned_or_empty_pdf(registry, ctx, files):
    make_pdf(files / "blank.pdf", [""])
    out = registry.call("open_document", {"path": str(files / "blank.pdf")}, ctx)
    assert "scanned" in out["error"]


def test_offline_gives_excerpts(registry, ctx, files, tmp_path):
    from jarvis.state import StateStore

    class Down:
        def ask(self, *a, **k):
            raise ConnectionError("offline")

    ctx.state, ctx._llm = StateStore(tmp_path / "s.json"), Down()
    out = registry.call("ask_document", {"question": "invoice number", "path": str(files / "Downloads" / "scan_0042.pdf")}, ctx)
    assert "7781" in out["excerpts"] and "Couldn't reach" in out["note"]


def test_documents_outside_or_secret_are_refused(registry, ctx, files, tmp_path):
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("x", encoding="utf-8")
    assert "error" in registry.call("open_document", {"path": str(outside)}, ctx)
    env = files / ".env"
    env.write_text("KEY=1", encoding="utf-8")
    assert registry.call("open_document", {"path": str(env)}, ctx)["error"].startswith("Blocked for your safety")


def test_chunks_are_cached_until_the_file_changes(files):
    from jarvis.tools.docchat import _cache, chunks_of

    f = files / "notes.txt"
    first = chunks_of(f)
    assert str(f) in _cache and chunks_of(f) is first
    time.sleep(0.01)
    f.write_text("new text entirely", encoding="utf-8")
    os.utime(f, (time.time() + 5, time.time() + 5))
    assert chunks_of(f) is not first


def test_window_document_chip(monkeypatch, isolated_home, files):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from jarvis.app import create_app
    from jarvis.config import Settings
    from jarvis.ui.window import Bridge, MainWindow

    qt = QApplication.instance() or QApplication(sys.argv)
    app = create_app(Settings(allowed_roots=[str(files)]))
    w = MainWindow(app, Bridge())
    w.show()
    app.registry.call("open_document", {"path": str(files / "proposal.docx")}, app.ctx)
    qt.processEvents()
    assert w.doc_chip.isVisible() and "proposal.docx" in w.doc_chip.text()
    w.doc_chip.click()
    qt.processEvents()
    assert not w.doc_chip.isVisible() and app.state.get("current_doc") == ""
