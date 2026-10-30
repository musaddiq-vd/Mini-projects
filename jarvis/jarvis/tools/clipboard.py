"""Read and write the clipboard (translate / summarise / fix what the user copied)."""

from __future__ import annotations

from jarvis.tools import tool

MAX_CHARS = 15_000


@tool(
    "read_clipboard",
    "Read the text the user has copied, e.g. to translate, summarise, explain or fix it.",
)
def read_clipboard(ctx):
    import pyperclip

    text = pyperclip.paste() or ""
    if not text.strip():
        return "The clipboard is empty (or holds an image/file)."
    return {"text": text[:MAX_CHARS], "truncated": len(text) > MAX_CHARS}


@tool(
    "write_clipboard",
    "Put text on the clipboard so the user can paste it (e.g. a translation or fixed text).",
    {"text": {"type": "string"}},
    required=["text"],
)
def write_clipboard(ctx, text: str):
    import pyperclip

    old = pyperclip.paste()
    pyperclip.copy(text)
    ctx.undo.push("clipboard change", lambda: pyperclip.copy(old))
    return "Copied to the clipboard."
