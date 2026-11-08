"""Chat with a document: open a PDF / Word / PowerPoint / Excel / text file, then ask questions about it.

Big documents are split into chunks; the chunks that best match the question go to the text model, which
answers only from them and cites pages. Works on the "current document" (set by drag & drop or
open_document) or any path.
"""

from __future__ import annotations

import re
from pathlib import Path

from jarvis.tools import tool

CHUNK = 1400
MAX_PAGES = 300
STOP = {"the", "a", "an", "is", "are", "was", "what", "which", "who", "how", "does", "do", "in", "on", "of", "to",
        "and", "or", "for", "this", "that", "it", "me", "my", "kya", "hai", "ka", "ki", "ke", "me", "mein", "is",
        "batao", "bata", "document", "file", "pdf", "about", "tell", "please"}
SUMMARY_WORDS = ("summar", "overview", "saar", "kya hai is", "what is this", "main points", "gist", "tldr", "short me")
_cache: dict[str, tuple[float, list[tuple[str, str]]]] = {}


def _resolve(ctx, path: str) -> Path:
    from jarvis.tools.files import resolve

    return resolve(ctx, path)


def chunks_of(path: Path) -> list[tuple[str, str]]:
    """[(where, text)] where 'where' is 'p. 3' / 'slide 2' / 'part 4'. Cached per file version."""
    from jarvis.tools.documents import extract_text

    key = str(path)
    mtime = path.stat().st_mtime
    hit = _cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    text = extract_text(path, max_pages=MAX_PAGES)
    parts = re.split(r"\[(page|slide) (\d+)\]", text)
    out: list[tuple[str, str]] = []
    if len(parts) > 1:
        for i in range(1, len(parts) - 1, 3):
            label = f"{'p.' if parts[i] == 'page' else 'slide'} {parts[i + 1]}"
            body = parts[i + 2].strip()
            for j in range(0, max(1, len(body)), CHUNK):
                piece = body[j:j + CHUNK].strip()
                if piece:
                    out.append((label, piece))
    else:
        paras, buf, n = text.split("\n"), "", 1
        for para in paras:
            if len(buf) + len(para) > CHUNK and buf:
                out.append((f"part {n}", buf.strip()))
                buf, n = "", n + 1
            buf += para + "\n"
        if buf.strip():
            out.append((f"part {n}", buf.strip()))
    _cache[key] = (mtime, out)
    return out


def rank(chunks: list[tuple[str, str]], question: str, k: int = 6) -> list[tuple[str, str]]:
    words = [w for w in re.findall(r"\w+", question.lower()) if w not in STOP and len(w) > 2]
    if any(s in question.lower() for s in SUMMARY_WORDS) or not words:
        step = max(1, len(chunks) // 8)
        return chunks[::step][:8]
    scored = []
    for i, (where, text) in enumerate(chunks):
        low = text.lower()
        score = sum(low.count(w) for w in words) + 3 * sum(1 for w in set(words) if w in low)
        if score:
            scored.append((score, i))
    if not scored:
        return []
    best = sorted(i for _, i in sorted(scored, reverse=True)[:k])
    return [chunks[i] for i in best]


def _current(ctx) -> str:
    return (ctx.state.get("current_doc") if ctx.state is not None else "") or ""


@tool(
    "open_document",
    "Open a document for questions (PDF, Word, PowerPoint, Excel, text). After this the user can just ask "
    "questions and you call ask_document.",
    {"path": {"type": "string"}},
    required=["path"],
    slow=True,
)
def open_document(ctx, path: str):
    try:
        file = _resolve(ctx, path)
    except Exception as exc:
        return {"error": str(exc)}
    if not file.is_file():
        return {"error": f"File not found: {file.name}. Try find_files first."}
    try:
        chunks = chunks_of(file)
    except Exception as exc:
        return {"error": f"Couldn't read {file.name}: {exc}"}
    words = sum(len(t.split()) for _, t in chunks)
    if words < 5:
        return {"error": f"{file.name} has no readable text - it may be a scanned image. I can look at it on the "
                         "screen instead (open it and ask me what's on the screen)."}
    if ctx.state is not None:
        ctx.state.set("current_doc", str(file))
    if ctx.events is not None:
        ctx.events.emit("document", path=str(file), name=file.name)
    pages = len({w for w, _ in chunks})
    return {"document": file.name, "sections": pages, "words": words, "start": chunks[0][1][:500],
            "result": f"Opened {file.name} ({words:,} words). Ask anything about it."}


@tool(
    "ask_document",
    "Answer a question from the current document (or the one at 'path'): facts, totals, dates, summaries. "
    "Answers come only from the document, with page numbers.",
    {"question": {"type": "string"}, "path": {"type": "string"}},
    required=["question"],
    slow=True,
)
def ask_document(ctx, question: str, path: str = ""):
    path = path or _current(ctx)
    if not path:
        return {"error": "No document is open. Drop a file on the window or ask me to find one first."}
    try:
        file = _resolve(ctx, path)
        chunks = chunks_of(file)
    except FileNotFoundError:
        return {"error": "That document isn't there any more."}
    except Exception as exc:
        return {"error": f"Couldn't read the document: {exc}"}
    if not chunks:
        return {"error": f"{file.name} has no readable text (maybe a scanned image)."}
    picked = rank(chunks, question)
    if not picked:
        return {"document": file.name, "answer": "", "result": "That doesn't seem to be in the document. "
                "Tell the user so, and offer to search for different words."}
    context = "\n\n".join(f"[{where}]\n{text}" for where, text in picked)
    pages = list(dict.fromkeys(w for w, _ in picked))
    try:
        answer = ctx.llm.ask(
            f"Document: {file.name}\n\nExcerpts:\n{context}\n\nQuestion: {question}",
            system="Answer only from the excerpts. Be short and exact; quote numbers and dates as written. Cite the "
                   "page or slide in brackets like (p. 3). If the excerpts don't contain the answer, say it isn't "
                   "in the document. Reply in the language of the question.")
    except Exception as exc:  # offline / quota: give the excerpts, the live model can still help
        return {"document": file.name, "excerpts": context[:6000], "pages": pages,
                "note": f"Couldn't reach the text model ({type(exc).__name__}); answer from these excerpts."}
    return {"document": file.name, "answer": answer, "pages": pages,
            "result": "Say the answer naturally and briefly; mention the page."}


@tool("close_document", "Stop using the current document.")
def close_document(ctx):
    name = Path(_current(ctx)).name if _current(ctx) else ""
    if ctx.state is not None:
        ctx.state.set("current_doc", "")
    if ctx.events is not None:
        ctx.events.emit("document", path="", name="")
    return f"Closed {name}." if name else "No document was open."
