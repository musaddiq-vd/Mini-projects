"""Personal knowledge base: answers from the folders and files the user chose to add.

Nothing is indexed by default. The user names a folder or file ("documents/college notes", a PDF) and confirms;
Jarvis then reads it, splits the text into overlapping chunks, masks secrets, embeds the chunks with the Gemini
embedding API and keeps everything in ``~/.jarvis/kb/``:

- ``kb.sqlite``: the chosen sources, one row per indexed file (hash, mtime, size), one row per chunk (where it
  came from, its text and its vector as float32 bytes) and an embedding cache keyed by the chunk text's hash,
  so text that was embedded once is never sent again (a moved or re-added file costs nothing).
- vectors are loaded into one numpy matrix for search; questions are matched by cosine similarity plus a
  BM25-style keyword boost.

Files that hold secrets (``guardrails.sensitive_path``) are never indexed. A light watcher re-indexes changed
files (an mtime/size check every few minutes). Backups leave the index out: it can be rebuilt from the files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

log = logging.getLogger("jarvis.kb")

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBED_BATCH = 64
CHUNK = 1200  # characters per chunk
OVERLAP = 200  # characters shared with the next chunk
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_FILES = 5000  # per source
MAX_PAGES = 500
NOTES_SOURCE = "jarvis:notes"  # Jarvis's own notes and to-dos
WATCH_SECONDS = 300.0  # how often the watcher looks for changed files
MIN_SCORE = 0.25  # below this nothing is a real match
CODE_SUFFIXES = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs",
                 ".rb", ".php", ".kt", ".swift", ".sh", ".ps1", ".sql", ".r", ".m", ".scala", ".lua", ".dart",
                 ".html", ".css", ".scss", ".vue", ".toml", ".ini", ".cfg"}
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".log", ".xml", ".yaml", ".yml", ".tex"}
DOC_SUFFIXES = {".pdf", ".docx", ".xlsx", ".xlsm", ".pptx"}
SUPPORTED = DOC_SUFFIXES | TEXT_SUFFIXES | CODE_SUFFIXES
SKIP_DIRS = {"node_modules", "__pycache__", ".git", "venv", ".venv", "site-packages", "dist", "build", ".idea",
             ".vscode", "appdata", "$recycle.bin", ".cache", "cache", "temp", "tmp"}
STOP = {"the", "a", "an", "is", "are", "was", "were", "what", "which", "who", "how", "does", "do", "did", "in", "on",
        "of", "to", "and", "or", "for", "this", "that", "it", "me", "my", "i", "you", "be", "with", "as", "at", "by",
        "kya", "hai", "ka", "ki", "ke", "mein", "me", "batao", "bata", "about", "tell", "please", "from", "notes"}

Embedder = Callable[[list[str], str], list[list[float]]]  # (texts, "document"|"query") -> vectors


# -- embedding ----------------------------------------------------------------------------------
class GeminiEmbedder:
    """Batches texts to the Gemini embedding API."""

    def __init__(self, settings):
        from google import genai
        from google.genai import types

        self._types = types
        self.client = genai.Client(api_key=settings.effective_api_key,
                                   http_options=types.HttpOptions(timeout=60_000))

    def __call__(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        config = self._types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM)
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            reply = self.client.models.embed_content(model=EMBED_MODEL, contents=batch, config=config)
            out.extend(list(e.values) for e in reply.embeddings)
        return out


_embedder_factory: Callable[[object], Embedder] = GeminiEmbedder


def set_embedder_factory(factory: Callable[[object], Embedder]) -> None:
    """Swap the embedder (tests use a fake one that needs no network)."""
    global _embedder_factory
    _embedder_factory = factory


def _unit(vectors) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# -- reading and chunking -------------------------------------------------------------------------
def words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1 and w not in STOP]


def split_text(text: str, size: int = CHUNK, overlap: int = OVERLAP) -> list[str]:
    """Overlapping windows that end at a space or line break where possible."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind("\n", start + size // 2, end), text.rfind(" ", start + size // 2, end))
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return out


_SECTION = re.compile(r"\[(page|slide|sheet) ([^\]]+)\]")


def read_sections(path: Path) -> list[tuple[str, str]]:
    """[(where, text)]: where is 'p. 3', 'slide 2', 'sheet Budget' or '' for plain text."""
    suffix = path.suffix.lower()
    if suffix in CODE_SUFFIXES or suffix in TEXT_SUFFIXES:
        return [("", path.read_text(encoding="utf-8", errors="replace"))]
    from jarvis.tools.documents import extract_text

    text = extract_text(path, max_pages=MAX_PAGES)
    parts = _SECTION.split(text)
    if len(parts) == 1:
        return [("", text)]
    out = [("", parts[0])] if parts[0].strip() else []
    for i in range(1, len(parts) - 1, 3):
        kind, label, body = parts[i], parts[i + 1].strip(), parts[i + 2]
        out.append((f"{'p.' if kind == 'page' else kind} {label}", body))
    return out


def chunk_file(path: Path) -> list[tuple[str, str]]:
    """[(where, chunk text)] with secrets masked."""
    from jarvis.guardrails import redact

    out = []
    for where, body in read_sections(path):
        pieces = split_text(body)
        for n, piece in enumerate(pieces, 1):
            label = where or (f"part {n}" if len(pieces) > 1 else "")
            out.append((label, redact(piece)[0]))
    return out


def _skip_dir(name: str) -> bool:
    return name.startswith((".", "$")) or name.lower() in SKIP_DIRS


def candidate_files(source: Path, limit: int = MAX_FILES) -> list[Path]:
    """Readable files under a source (or the file itself); never secret files."""
    from jarvis.guardrails import sensitive_path

    if source.is_file():
        ok = source.suffix.lower() in SUPPORTED and not sensitive_path(source)
        return [source] if ok else []
    out: list[Path] = []
    for dirpath, dirs, files in os.walk(source):
        dirs[:] = sorted(d for d in dirs if not _skip_dir(d) and not sensitive_path(Path(dirpath) / d))
        for f in sorted(files):
            p = Path(dirpath) / f
            if p.suffix.lower() in SUPPORTED and not f.startswith("~$") and not sensitive_path(p):
                out.append(p)
                if len(out) >= limit:
                    return out
    return out


def estimate(source: Path) -> dict:
    """How much adding a source would index: file count, total size and a rough time."""
    files = candidate_files(source)
    size = 0
    for f in files:
        try:
            size += min(f.stat().st_size, MAX_FILE_BYTES)
        except OSError:
            pass
    return {"files": len(files), "bytes": size, "size": human(size), "minutes": max(1, round(size / 8_000_000))}


def human(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def notes_chunks(state) -> list[tuple[str, str]]:
    """Jarvis's own notes and to-dos as chunks ('note #3')."""
    from jarvis.guardrails import redact

    items = list(state.get("notes", [])) if state is not None else []
    out = []
    for n in items:
        text = str(n.get("text", "")).strip()
        if text:
            extra = " (done)" if n.get("done") else ""
            out.append((f"note #{n.get('id', '?')}", redact(text + extra)[0]))
    return out


# -- store ------------------------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (path TEXT PRIMARY KEY, added REAL);
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, source TEXT, hash TEXT, mtime REAL, size INTEGER,
                                  chunks INTEGER, indexed REAL);
CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT, place TEXT, text TEXT,
                                   vec BLOB);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE TABLE IF NOT EXISTS embcache (hash TEXT PRIMARY KEY, vec BLOB);
"""


def kb_dir() -> Path:
    from jarvis.config import home_dir

    return home_dir() / "kb"


class KnowledgeBase:
    def __init__(self, folder: Path | None = None, settings=None, state=None):
        self.folder = Path(folder or kb_dir())
        self.settings = settings
        self.state = state
        self._lock = threading.RLock()  # the in-memory search matrix and short database edits
        self._sync_lock = threading.Lock()  # one indexing run at a time; searches keep working meanwhile
        self._matrix: np.ndarray | None = None
        self._rows: list[tuple[int, str, str, str]] = []  # (id, path, place, text)
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._embedder: Embedder | None = None
        self.busy = False
        self.progress = ""
        self.last_error = ""

    # -- plumbing -------------------------------------------------------------
    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success and is always closed (Windows can't delete open files)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.folder / "kb.sqlite", timeout=30)
        try:
            db.executescript(_SCHEMA)
            yield db
            db.commit()
        finally:
            db.close()

    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = _embedder_factory(self.settings)
        return self._embedder

    def _changed(self) -> None:
        self._matrix = None

    # -- sources --------------------------------------------------------------
    def sources(self) -> list[str]:
        if not (self.folder / "kb.sqlite").exists():
            return []
        with self._lock, self._db() as db:
            return [r[0] for r in db.execute("SELECT path FROM sources ORDER BY added")]

    def add_source(self, path: str) -> None:
        with self._lock, self._db() as db:
            db.execute("INSERT OR IGNORE INTO sources VALUES (?, ?)", (path, time.time()))

    def remove_source(self, path: str, keep_files: bool = False) -> int:
        """Forget a source (or any indexed files under the path). Returns how many files were dropped.
        keep_files: only drop the source entry (a bigger folder that was just added takes its files over)."""
        with self._lock, self._db() as db:
            db.execute("DELETE FROM sources WHERE path = ?", (path,))
            if keep_files:
                return 0
            like = path.rstrip("\\/") + os.sep + "%"
            gone = [r[0] for r in db.execute("SELECT path FROM files WHERE path = ? OR path LIKE ? OR source = ?",
                                             (path, like, path))]
            for p in gone:
                db.execute("DELETE FROM chunks WHERE path = ?", (p,))
                db.execute("DELETE FROM files WHERE path = ?", (p,))
        self._changed()
        return len(gone)

    def clear(self) -> None:
        """Delete the whole index and the list of sources."""
        import shutil

        with self._lock:
            self._matrix, self._rows = None, []
            shutil.rmtree(self.folder, ignore_errors=True)

    # -- indexing ---------------------------------------------------------------
    def _embed(self, db: sqlite3.Connection, texts: list[str]) -> list[bytes]:
        """Vectors for the texts as float32 bytes; cached ones are not sent again."""
        keys = [_sha(t) for t in texts]
        have = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            marks = ",".join("?" * len(part))
            have.update(db.execute(f"SELECT hash, vec FROM embcache WHERE hash IN ({marks})", part).fetchall())
        todo = list(dict.fromkeys(k for k in keys if k not in have))
        if todo:
            text_of = dict(zip(keys, texts))
            vectors = self.embedder()([text_of[k] for k in todo], "document")
            for k, vec in zip(todo, _unit(vectors)):
                blob = vec.astype(np.float32).tobytes()
                have[k] = blob
                db.execute("INSERT OR REPLACE INTO embcache VALUES (?, ?)", (k, blob))
        return [have[k] for k in keys]

    def _store_chunks(self, db, path: str, source: str, chunks: list[tuple[str, str]], digest: str,
                      mtime: float, size: int) -> int:
        blobs = self._embed(db, [t for _, t in chunks]) if chunks else []
        db.execute("DELETE FROM chunks WHERE path = ?", (path,))
        db.executemany("INSERT INTO chunks (path, place, text, vec) VALUES (?, ?, ?, ?)",
                       [(path, where, text, blob) for (where, text), blob in zip(chunks, blobs)])
        db.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (path, source, digest, mtime, size, len(chunks), time.time()))
        return len(chunks)

    def index_file(self, db, path: Path, source: str, force: bool = False) -> str:
        """'new', 'updated', 'same' or 'skipped'. Unchanged files (mtime+size, then hash) are not re-read."""
        try:
            st = path.stat()
        except OSError:
            return "skipped"
        if st.st_size > MAX_FILE_BYTES:
            return "skipped"
        row = db.execute("SELECT hash, mtime, size, source FROM files WHERE path = ?", (str(path),)).fetchone()
        if row and row[3] != source:  # now covered by a bigger folder the user added
            db.execute("UPDATE files SET source = ? WHERE path = ?", (source, str(path)))
        if row and not force and row[1] == st.st_mtime and row[2] == st.st_size:
            return "same"
        digest = _file_hash(path)
        if row and not force and row[0] == digest:
            db.execute("UPDATE files SET mtime = ?, size = ? WHERE path = ?", (st.st_mtime, st.st_size, str(path)))
            return "same"
        try:
            chunks = chunk_file(path)
        except Exception as exc:  # unreadable or password-protected: remember it so it isn't retried every time
            log.info("kb: can't read %s: %s", path.name, exc)
            chunks = []
        self._store_chunks(db, str(path), source, chunks, digest, st.st_mtime, st.st_size)
        return "updated" if row else "new"

    def index_notes(self, db, force: bool = False) -> str:
        chunks = notes_chunks(self.state)
        digest = _sha(json.dumps(chunks))
        row = db.execute("SELECT hash FROM files WHERE path = ?", (NOTES_SOURCE,)).fetchone()
        if row and row[0] == digest and not force:
            return "same"
        self._store_chunks(db, NOTES_SOURCE, NOTES_SOURCE, chunks, digest, time.time(), 0)
        return "updated" if row else "new"

    def sync(self, only: str | None = None, force: bool = False) -> dict:
        """Bring the index in line with the sources: new and changed files in, deleted ones out."""
        from jarvis.guardrails import sensitive_path

        counts = Counter()
        with self._sync_lock:
            self.busy = True
            try:
                with self._db() as db:
                    sources = [only] if only else [r[0] for r in db.execute("SELECT path FROM sources")]
                    for source in sources:
                        if source == NOTES_SOURCE:
                            counts[self.index_notes(db, force)] += 1
                            continue
                        root = Path(source)
                        seen = set()
                        files = candidate_files(root) if root.exists() else []
                        for i, f in enumerate(files, 1):
                            self.progress = f"{i}/{len(files)} {f.name}"
                            seen.add(str(f))
                            counts[self.index_file(db, f, source, force)] += 1
                            db.commit()
                        for (p,) in db.execute("SELECT path FROM files WHERE source = ?", (source,)).fetchall():
                            if p not in seen or sensitive_path(p):
                                db.execute("DELETE FROM chunks WHERE path = ?", (p,))
                                db.execute("DELETE FROM files WHERE path = ?", (p,))
                                counts["removed"] += 1
                self.last_error = ""
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("kb sync failed: %s", self.last_error)
                raise
            finally:
                self.busy = False
                self.progress = ""
                with self._lock:
                    self._changed()
        return dict(counts)

    def needs_sync(self) -> bool:
        """Cheap check (stat only): did any file under the sources change, appear or disappear?"""
        if not (self.folder / "kb.sqlite").exists():
            return False
        with self._lock, self._db() as db:
            known = {p: (src, m, s) for p, src, m, s in db.execute("SELECT path, source, mtime, size FROM files")}
            sources = [r[0] for r in db.execute("SELECT path FROM sources")]
            notes_hash = db.execute("SELECT hash FROM files WHERE path = ?", (NOTES_SOURCE,)).fetchone()
        for source in sources:
            if source == NOTES_SOURCE:
                if not notes_hash or notes_hash[0] != _sha(json.dumps(notes_chunks(self.state))):
                    return True
                continue
            files = candidate_files(Path(source)) if Path(source).exists() else []
            if len(files) != sum(1 for src, _, _ in known.values() if src == source):
                return True
            for f in files:
                try:
                    st = f.stat()
                except OSError:
                    return True
                if known.get(str(f)) != (source, st.st_mtime, st.st_size):
                    return True
        return False

    # -- search ------------------------------------------------------------------
    def _load(self) -> None:
        if self._matrix is not None:
            return
        rows, vecs = [], []
        if (self.folder / "kb.sqlite").exists():
            with self._db() as db:
                for cid, path, place, text, blob in db.execute("SELECT id, path, place, text, vec FROM chunks"):
                    if blob:
                        rows.append((cid, path, place, text))
                        vecs.append(np.frombuffer(blob, dtype=np.float32))
        self._rows = rows
        dims = {len(v) for v in vecs}
        self._matrix = np.vstack(vecs) if vecs and len(dims) == 1 else np.zeros((0, 1), dtype=np.float32)
        if len(dims) > 1:  # embedding size changed: the index must be rebuilt
            self._rows = []
        self._tf = [Counter(words(r[3])) for r in self._rows]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())

    def _bm25(self, query_words: list[str]) -> np.ndarray:
        n = len(self._tf)
        scores = np.zeros(n, dtype=np.float32)
        if not n or not query_words:
            return scores
        avg = sum(self._len) / n or 1.0
        k1, b = 1.4, 0.75
        for w in set(query_words):
            df = self._df.get(w, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self._tf):
                f = tf.get(w)
                if f:
                    scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self._len[i] / avg))
        return scores

    def search(self, question: str, k: int = 6, keyword_fallback: bool = True) -> list[dict]:
        """Best chunks for a question: cosine similarity plus a keyword boost. [{file, path, where, text, score}]
        Offline (the embedder fails) it falls back to keywords only, unless keyword_fallback is False."""
        with self._lock:
            self._load()
            if not self._rows:
                return []
            try:
                q = _unit(self.embedder()([question], "query"))[0]
                cos = self._matrix @ q if q.shape[0] == self._matrix.shape[1] else None
            except Exception:
                if not keyword_fallback:
                    raise
                cos = None
            if cos is None:
                cos = np.zeros(len(self._rows), dtype=np.float32)
            kw = self._bm25(words(question))
            top_kw = float(kw.max()) if kw.size else 0.0
            boost = kw / top_kw if top_kw > 0 else kw
            score = cos + 0.25 * boost
            order = np.argsort(-score)[:k]
            out = []
            for i in order:
                if cos[i] < MIN_SCORE and boost[i] == 0:
                    continue
                _, path, place, text = self._rows[i]
                name = "Jarvis notes" if path == NOTES_SOURCE else Path(path).name
                out.append({"file": name, "path": path, "where": place, "text": text,
                            "score": round(float(score[i]), 3), "keyword": bool(boost[i] > 0)})
            return out

    # -- facts ---------------------------------------------------------------------
    def status(self) -> dict:
        db_file = self.folder / "kb.sqlite"
        if not db_file.exists():
            return {"sources": 0, "files": 0, "chunks": 0, "bytes": 0, "size": "empty"}
        with self._lock, self._db() as db:
            files = db.execute("SELECT COUNT(*), MAX(indexed) FROM files").fetchone()
            chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            sources = db.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        size = sum(f.stat().st_size for f in self.folder.glob("*") if f.is_file())
        return {"sources": sources, "files": files[0], "chunks": chunks, "bytes": size, "size": human(size),
                "last_indexed": time.strftime("%d %b %Y %I:%M %p", time.localtime(files[1])) if files[1] else ""}

    def files_of(self, source: str) -> int:
        with self._lock, self._db() as db:
            return db.execute("SELECT COUNT(*) FROM files WHERE source = ?", (source,)).fetchone()[0]


# -- the shared instance and background work ----------------------------------------------------------
_kb: KnowledgeBase | None = None
_kb_lock = threading.Lock()
_jobs: list[threading.Thread] = []
_watch_stop = threading.Event()
_watcher: threading.Thread | None = None


def get(ctx=None) -> KnowledgeBase:
    """The knowledge base in the current Jarvis home (made again when JARVIS_HOME changes)."""
    global _kb
    with _kb_lock:
        folder = kb_dir()
        if _kb is None or _kb.folder != folder:
            _kb = KnowledgeBase(folder)
        if ctx is not None:
            _kb.settings = ctx.settings
            _kb.state = ctx.state if ctx.state is not None else _kb.state
        return _kb


def run_in_background(fn: Callable[[], None]) -> threading.Thread:
    t = threading.Thread(target=fn, daemon=True, name="kb-index")
    _jobs.append(t)
    t.start()
    return t


def wait_idle(timeout: float = 30.0) -> bool:
    """Wait for background indexing to finish (used by tests and on quit)."""
    end = time.monotonic() + timeout
    for t in list(_jobs):
        t.join(max(0.0, end - time.monotonic()))
    _jobs[:] = [t for t in _jobs if t.is_alive()]
    return not _jobs


def size_text() -> str:
    """For Settings > Privacy: '12 files · 3.4 MB' or 'nothing yet'."""
    folder = kb_dir()
    if not (folder / "kb.sqlite").exists():
        return "nothing yet"
    s = get().status()
    return f"{s['files']} file{'s' if s['files'] != 1 else ''} · {s['size']}"


def clear_all() -> None:
    get().clear()


def start_watcher(ctx, every: float = WATCH_SECONDS) -> None:
    """Re-index changed files now and then. Does nothing while no sources were added."""
    global _watcher
    if _watcher is not None and _watcher.is_alive():
        return
    _watch_stop.clear()

    def loop() -> None:
        while not _watch_stop.wait(every):
            try:
                kb = get(ctx)
                if kb.busy or not kb.sources():
                    continue
                if kb.needs_sync():
                    counts = kb.sync()
                    log.info("kb watcher: %s", counts)
            except Exception:  # noqa: BLE001 - never stop the watcher over one failure
                log.exception("kb watcher failed")

    _watcher = threading.Thread(target=loop, daemon=True, name="kb-watch")
    _watcher.start()


def stop_watcher() -> None:
    _watch_stop.set()


def iter_sources_text(items: Iterable[str]) -> list[dict]:
    out = []
    kb = get()
    for s in items:
        name = "Jarvis notes and to-dos" if s == NOTES_SOURCE else s
        out.append({"source": name, "files": kb.files_of(s), "exists": s == NOTES_SOURCE or Path(s).exists()})
    return out
