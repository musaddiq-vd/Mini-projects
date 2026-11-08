"""Safe, tiny "markdown" for chat bubbles: **bold**, *italic*, `code`, links and bullet lines.

Everything is HTML-escaped first, so nothing a message contains can inject markup.
"""

from __future__ import annotations

import html
import re

_URL = re.compile(r"(https?://[^\s<]+[^\s<.,;:!?)\]'\"])")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])")
_CODE = re.compile(r"`([^`]+)`")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def has_markup(text: str) -> bool:
    return bool(_URL.search(text) or _BOLD.search(text) or _CODE.search(text) or _ITALIC.search(text)
                or any(_BULLET.match(line) for line in text.splitlines()))


def to_html(text: str, link_color: str = "#00d4ff", code_bg: str = "rgba(127,127,127,40)") -> str:
    lines = []
    for raw in (text or "").split("\n"):
        line = html.escape(raw)
        bullet = _BULLET.match(raw)
        if bullet:
            marker = raw[: bullet.end()].strip()
            line = html.escape(raw[bullet.end():])
            prefix = "•" if marker in {"-", "*", "•"} else html.escape(marker)
            line = f"&nbsp;&nbsp;{prefix}&nbsp;{line}"
        # code first, so its contents aren't styled
        codes: list[str] = []

        def keep_code(m: re.Match) -> str:
            codes.append(m.group(1))
            return f"\x00{len(codes) - 1}\x00"

        line = _CODE.sub(keep_code, line)
        line = _URL.sub(lambda m: f'<a href="{m.group(1)}" style="color:{link_color};">{m.group(1)}</a>', line)
        line = _BOLD.sub(r"<b>\1</b>", line)
        line = _ITALIC.sub(r"<i>\1</i>", line)
        for i, code in enumerate(codes):
            line = line.replace(f"\x00{i}\x00", f'<span style="font-family:Consolas,monospace; '
                                                f'background:{code_bg};">&nbsp;{code}&nbsp;</span>')
        lines.append(line)
    return "<br>".join(lines)


def plain(text: str) -> str:
    """The text without markup characters (for copying and for the typewriter)."""
    t = _BOLD.sub(r"\1", text or "")
    t = _CODE.sub(r"\1", t)
    return t
