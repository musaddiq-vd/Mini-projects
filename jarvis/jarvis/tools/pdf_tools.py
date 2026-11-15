"""PDF tools: merge, split / pick pages, rotate and shrink PDF files.

Every path must be inside the folders Jarvis may use. A new file is always written next to the input (or
where the user said) and never overwrites anything; "undo" sends the new file to the Recycle Bin.
"""

from __future__ import annotations

import re
from pathlib import Path

from jarvis.tools import tool
from jarvis.tools.files import _guard, resolve

MAX_FILES = 30


def free_name(path: Path) -> Path:
    """path, or 'name (2).pdf', 'name (3).pdf' ... if it already exists."""
    if not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(path)


def parse_pages(spec: str, total: int) -> list[int]:
    """'1-3, 5, 8-' -> zero-based page numbers (in that order, no repeats). 'last' and 'odd'/'even' work too."""
    spec = (spec or "").lower().replace(" ", "").replace("to", "-")
    if spec in ("", "all"):
        return list(range(total))
    if spec in ("odd", "even"):
        return list(range(0 if spec == "odd" else 1, total, 2))
    out: list[int] = []
    for part in spec.split(","):
        if not part:
            continue
        part = part.replace("last", str(total))
        m = re.fullmatch(r"(\d+)?-(\d+)?|(\d+)", part)
        if not m:
            raise ValueError(f"I don't understand the pages '{part}'. Use something like '1-3, 5'.")
        if m.group(3):
            first = last = int(m.group(3))
        else:
            first, last = int(m.group(1) or 1), int(m.group(2) or total)
        if first < 1 or last > total or first > last:
            raise ValueError(f"Pages {part} are outside this PDF, which has {total} page{'s' if total != 1 else ''}.")
        for n in range(first, last + 1):
            if n - 1 not in out:
                out.append(n - 1)
    if not out:
        raise ValueError("No pages chosen.")
    return out


def _pdf(ctx, raw: str) -> Path:
    path = resolve(ctx, raw)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"'{path.name}' isn't a PDF.")
    return path


def _out(ctx, raw: str, default: Path) -> Path:
    path = resolve(ctx, raw) if raw else default
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    return free_name(path)


def _write(ctx, writer, out: Path, what: str) -> dict:
    with out.open("wb") as f:
        writer.write(f)

    def revert():
        from send2trash import send2trash

        if out.exists():
            send2trash(str(out))

    ctx.undo.push(what, revert)
    return {"result": f"Saved {out.name}.", "path": str(out), "size_kb": round(out.stat().st_size / 1024)}


def _errors(fn):
    def wrapper(ctx, *args, **kwargs):
        try:
            return fn(ctx, *args, **kwargs)
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - a broken or locked PDF
            if type(exc).__name__ in ("PdfReadError", "FileNotDecryptedError", "DependencyError"):
                return {"error": "That PDF can't be read (it may be damaged or password protected)."}
            raise

    wrapper.__name__ = fn.__name__
    import inspect

    wrapper.__signature__ = inspect.signature(fn)
    return wrapper


@tool(
    "merge_pdfs",
    "Join several PDF files into one, in the given order. Paths may start with downloads/, documents/ etc.",
    {"files": {"type": "array", "items": {"type": "string"}},
     "output": {"type": "string", "description": "where to save, optional (default: 'merged.pdf' next to the first)"}},
    required=["files"],
    slow=True,
)
@_guard
@_errors
def merge_pdfs(ctx, files: list, output: str = ""):
    from pypdf import PdfWriter

    if not files or len(files) < 2:
        return {"error": "Give me at least two PDF files to merge."}
    if len(files) > MAX_FILES:
        return {"error": f"I can merge up to {MAX_FILES} files at once."}
    paths = [_pdf(ctx, f) for f in files]
    out = _out(ctx, output, paths[0].with_name("merged.pdf"))
    writer = PdfWriter()
    for p in paths:
        writer.append(str(p))
    result = _write(ctx, writer, out, f"merge into {out.name}")
    result["pages"] = len(writer.pages)
    return result


@tool(
    "extract_pdf_pages",
    "Save some pages of a PDF as a new PDF (split / pick pages), e.g. pages '1-3, 5', 'last', 'odd'. "
    "With every_page=true each page becomes its own file.",
    {"path": {"type": "string"}, "pages": {"type": "string"}, "every_page": {"type": "boolean"},
     "output": {"type": "string"}},
    required=["path"],
    slow=True,
)
@_guard
@_errors
def extract_pdf_pages(ctx, path: str, pages: str = "all", every_page: bool = False, output: str = ""):
    from pypdf import PdfReader, PdfWriter

    src = _pdf(ctx, path)
    reader = PdfReader(str(src))
    chosen = parse_pages(pages, len(reader.pages))
    if every_page:
        if len(chosen) > 200:
            return {"error": "That's too many separate files - choose fewer pages."}
        saved = []
        for i in chosen:
            w = PdfWriter()
            w.add_page(reader.pages[i])
            out = _out(ctx, "", src.with_name(f"{src.stem} - page {i + 1}.pdf"))
            saved.append(_write(ctx, w, out, f"save {out.name}")["path"])
        return {"result": f"Saved {len(saved)} files, one per page.", "files": saved}
    w = PdfWriter()
    for i in chosen:
        w.add_page(reader.pages[i])
    label = pages if pages and pages != "all" else "copy"
    out = _out(ctx, output, src.with_name(f"{src.stem} - pages {label}.pdf".replace(",", " ")))
    result = _write(ctx, w, out, f"save {out.name}")
    result["pages"] = len(chosen)
    return result


@tool(
    "rotate_pdf",
    "Turn pages of a PDF that are sideways or upside down. degrees: 90 (clockwise), 180 or 270. Saves a new file.",
    {"path": {"type": "string"}, "degrees": {"type": "integer", "description": "90, 180 or 270"},
     "pages": {"type": "string", "description": "which pages, default all"}},
    required=["path"],
)
@_guard
@_errors
def rotate_pdf(ctx, path: str, degrees: int = 90, pages: str = "all"):
    from pypdf import PdfReader, PdfWriter

    if int(degrees) % 90:
        return {"error": "Pages can only turn by 90, 180 or 270 degrees."}
    src = _pdf(ctx, path)
    reader = PdfReader(str(src))
    turn = set(parse_pages(pages, len(reader.pages)))
    w = PdfWriter()
    for i, page in enumerate(reader.pages):
        w.add_page(page)
        if i in turn:
            w.pages[-1].rotate(int(degrees))
    out = _out(ctx, "", src.with_name(f"{src.stem} - rotated.pdf"))
    result = _write(ctx, w, out, f"save {out.name}")
    result["rotated_pages"] = len(turn)
    return result


@tool(
    "compress_pdf",
    "Make a PDF file smaller (to email or upload it). Saves a new, smaller copy and says how much was saved.",
    {"path": {"type": "string"}},
    required=["path"],
    slow=True,
)
@_guard
@_errors
def compress_pdf(ctx, path: str):
    from pypdf import PdfReader, PdfWriter

    src = _pdf(ctx, path)
    writer = PdfWriter(clone_from=PdfReader(str(src)))
    for page in writer.pages:
        try:
            page.compress_content_streams()
        except Exception:  # noqa: BLE001 - some pages can't be compressed further
            pass
        for img in getattr(page, "images", []) or []:
            try:
                if img.image is not None and img.image.mode in ("RGB", "L"):
                    img.replace(img.image, quality=60)
            except Exception:  # noqa: BLE001 - unusual image formats stay as they are
                pass
    writer.compress_identical_objects(remove_identicals=True, remove_orphans=True)
    out = _out(ctx, "", src.with_name(f"{src.stem} - smaller.pdf"))
    result = _write(ctx, writer, out, f"save {out.name}")
    before = src.stat().st_size
    after = out.stat().st_size
    saved = max(0, before - after)
    result["before_kb"], result["after_kb"] = round(before / 1024), round(after / 1024)
    result["saved_percent"] = round(saved * 100 / before) if before else 0
    if after >= before:
        result["note"] = "This PDF was already as small as it gets."
    return result
