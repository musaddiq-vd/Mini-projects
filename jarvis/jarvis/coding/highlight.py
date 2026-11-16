"""A small syntax highlighter: code -> HTML spans (keywords, strings, comments, numbers, diff lines).

Good enough for a chat card in Python, JavaScript/TypeScript, Go, Rust, Java, C-like languages and shell; no
extra dependency. Colors come from the caller so the card follows the theme.
"""

from __future__ import annotations

import html
import re

KEYWORDS = {
    "python": "and as assert async await break class continue def del elif else except finally for from global if "
              "import in is lambda nonlocal not or pass raise return try while with yield match case None True False "
              "self",
    "javascript": "async await break case catch class const continue debugger default delete do else export extends "
                  "finally for from function if import in instanceof let new of return static super switch this "
                  "throw try typeof var void while yield null undefined true false interface type enum implements "
                  "public private protected readonly",
    "go": "break case chan const continue default defer else fallthrough for func go goto if import interface map "
          "package range return select struct switch type var nil true false",
    "rust": "as async await break const continue crate else enum extern false fn for if impl in let loop match mod "
            "move mut pub ref return self Self static struct super trait true type unsafe use where while",
    "c": "auto break case char const continue default do double else enum extern float for goto if int long return "
         "short signed sizeof static struct switch typedef union unsigned void volatile while class public private "
         "protected new delete this try catch throw namespace using template virtual bool true false null nullptr "
         "string var final abstract interface extends implements package import boolean",
    "bash": "if then else elif fi for while do done case esac function in return export local echo",
}
ALIASES = {"typescript": "javascript", "jsx": "javascript", "tsx": "javascript", "js": "javascript",
           "ts": "javascript", "py": "python", "cpp": "c", "java": "c", "csharp": "c", "cs": "c", "kotlin": "c",
           "swift": "c", "dart": "c", "php": "c", "sh": "bash", "powershell": "bash", "ruby": "python"}
HASH_COMMENTS = {"python", "bash"}


def _pattern(lang: str) -> re.Pattern:
    words = KEYWORDS.get(lang, KEYWORDS["c"]).split()
    comment = r"#[^\n]*" if lang in HASH_COMMENTS else r"//[^\n]*|/\*[\s\S]*?\*/"
    strings = (r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|' if lang == "python" else r"`(?:\\.|[^`\\])*`|") + \
        r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''
    return re.compile(
        rf"(?P<comment>{comment})|(?P<string>{strings})|(?P<number>\b\d[\d_]*(?:\.\d+)?\b)"
        rf"|(?P<keyword>\b(?:{'|'.join(map(re.escape, words))})\b)|(?P<call>\b[A-Za-z_]\w*(?=\())")


def to_html(code: str, language: str, colors: dict[str, str]) -> str:
    """Highlighted, escaped HTML for `code` (keep it inside a <pre>)."""
    lang = (language or "").lower()
    lang = ALIASES.get(lang, lang)
    if lang == "diff":
        return diff_html(code, colors)
    pattern = _pattern(lang if lang in KEYWORDS else "c")
    out, pos = [], 0
    for m in pattern.finditer(code):
        out.append(html.escape(code[pos:m.start()]))
        kind = m.lastgroup or ""
        color = colors.get(kind)
        text = html.escape(m.group())
        out.append(f"<span style='color:{color}'>{text}</span>" if color else text)
        pos = m.end()
    out.append(html.escape(code[pos:]))
    return "".join(out)


def diff_html(diff: str, colors: dict[str, str]) -> str:
    rows = []
    for line in diff.splitlines():
        text = html.escape(line)
        if line.startswith(("+++", "---")):
            rows.append(f"<span style='color:{colors.get('comment', '')}'>{text}</span>")
        elif line.startswith("+"):
            rows.append(f"<span style='color:{colors.get('added', '')}'>{text}</span>")
        elif line.startswith("-"):
            rows.append(f"<span style='color:{colors.get('removed', '')}'>{text}</span>")
        elif line.startswith("@@"):
            rows.append(f"<span style='color:{colors.get('keyword', '')}'>{text}</span>")
        else:
            rows.append(text)
    return "\n".join(rows)


def guess_language(path: str) -> str:
    from jarvis.coding.project import LANGUAGES

    m = re.search(r"(\.[A-Za-z0-9]+)$", path or "")
    return LANGUAGES.get(m.group(1).lower(), "") if m else ""
