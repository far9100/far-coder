"""Embedding-based hybrid code search.

A second retrieval channel that complements the FTS5 keyword index in
``memory.py``. The motivating use case: the user asks "where's the rate
limiter?" and no file is named ``rate_limiter`` — keyword search misses,
embedding search finds it via semantic similarity.

Design choices:
  * Index lives in the same SQLite DB (``~/.farcode_memory.db``) as the
    memory store, in a separate ``code_chunks`` table. This keeps the install
    surface tiny — we don't need ``sqlite-vec`` or ``faiss``.
  * Vectors are stored as JSON-serialized lists of floats (no numpy required).
    For typical projects (<2000 chunks × 768 dims), pure-Python cosine is
    well under 100ms — fast enough to be invisible to the user.
  * Embedding model defaults to ``nomic-embed-text`` (~140M, pulled separately
    via ``ollama pull nomic-embed-text``). Override with FARCODE_EMBED_MODEL.
  * Chunks are top-level Python defs/classes only in v1. Adding tree-sitter
    chunkers for other languages later is a straightforward extension.
  * Hybrid retrieval combines FTS5 + embedding rankings via Reciprocal Rank
    Fusion (RRF), the standard approach when no calibrated cross-encoder is
    available.
  * Index is lazy: ``recall_code`` triggers indexing of the current project on
    first call, then reuses cached vectors (busted by file mtime).
"""

from __future__ import annotations

import ast
import json
import math
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

EMBED_MODEL = os.environ.get("FARCODE_EMBED_MODEL", "nomic-embed-text")
# Limits to keep indexing cheap on big repos.
MAX_FILES_TO_INDEX = 400
MAX_CHUNK_CHARS = 4000
# Reciprocal Rank Fusion parameter (Cormack et al. 2009 default).
RRF_K = 60


# ── Schema ───────────────────────────────────────────────────────────────────

def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS code_chunks (
            id TEXT PRIMARY KEY,
            project_path TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_mtime REAL NOT NULL,
            chunk_name TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            body TEXT NOT NULL,
            embedding TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_project ON code_chunks(project_path)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_file ON code_chunks(file_path)"
    )
    conn.commit()


def _conn() -> sqlite3.Connection:
    """Borrow the memory store's connection so all our state lives in one DB."""
    from .memory import _get_conn
    c = _get_conn()
    _ensure_schema(c)
    return c


# ── Chunking ─────────────────────────────────────────────────────────────────

def _python_chunks(path: Path) -> list[tuple[str, int, int, str]]:
    """Return [(name, start_line, end_line, body), ...] for top-level defs/classes."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines(keepends=True)
    out: list[tuple[str, int, int, str]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", start)
        body = "".join(lines[start - 1:end])
        if len(body) > MAX_CHUNK_CHARS:
            body = body[:MAX_CHUNK_CHARS] + "\n# ...[truncated]"
        out.append((node.name, start, end, body))
    return out


# ── Embedding I/O ────────────────────────────────────────────────────────────

def _embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Call Ollama's embed endpoint. Returns None on any failure (model not
    pulled, ollama down, etc.)."""
    if not texts:
        return []
    try:
        import ollama
        # Ollama supports batched input via the embed() method.
        resp = ollama.embed(model=EMBED_MODEL, input=texts)
        # Newer ollama-python returns dict-like with "embeddings" key
        embs = resp.get("embeddings") if hasattr(resp, "get") else None
        if embs is None:
            embs = getattr(resp, "embeddings", None)
        if embs and len(embs) == len(texts):
            return [list(map(float, e)) for e in embs]
    except Exception:
        return None
    return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ── Indexing ─────────────────────────────────────────────────────────────────

def index_project(project_path: str | None = None, *, force: bool = False) -> int:
    """Index all .py files in the project. Returns number of chunks indexed.

    Skips files whose mtime matches what's already in the DB. Pass force=True
    to wipe and rebuild the project's chunks.
    """
    from .memory import current_project_path
    project = project_path or current_project_path()
    rootp = Path(project).resolve()
    conn = _conn()

    if force:
        conn.execute(
            "DELETE FROM code_chunks WHERE project_path = ?", (str(rootp),)
        )
        conn.commit()

    # Collect candidate files (Python only in v1)
    files: list[Path] = []
    for current, dirs, names in os.walk(rootp):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in {
                "node_modules", "__pycache__", "venv", "dist", "build",
                "site-packages", "target",
            }
        ]
        for n in names:
            if n.endswith(".py"):
                files.append(Path(current) / n)
                if len(files) >= MAX_FILES_TO_INDEX:
                    break
        if len(files) >= MAX_FILES_TO_INDEX:
            break

    # Find what's already indexed and skip unchanged files.
    cur = conn.execute(
        "SELECT file_path, file_mtime FROM code_chunks WHERE project_path = ?",
        (str(rootp),),
    )
    cached: dict[str, float] = {}
    for row in cur.fetchall():
        fp = row["file_path"] if hasattr(row, "keys") else row[0]
        mt = row["file_mtime"] if hasattr(row, "keys") else row[1]
        cached[fp] = max(cached.get(fp, 0.0), mt)

    new_chunks: list[tuple[Path, str, int, int, str]] = []
    files_to_clear: list[str] = []
    for fp in files:
        try:
            mtime = fp.stat().st_mtime
        except OSError:
            continue
        rel = str(fp.resolve())
        if cached.get(rel) == mtime:
            continue  # already indexed at this mtime
        if rel in cached:
            files_to_clear.append(rel)
        for name, s, e, body in _python_chunks(fp):
            new_chunks.append((fp, name, s, e, body))

    if files_to_clear:
        conn.executemany(
            "DELETE FROM code_chunks WHERE file_path = ?",
            [(p,) for p in files_to_clear],
        )
        conn.commit()

    if not new_chunks:
        return 0

    # Embed in batches
    texts_for_embedding: list[str] = []
    for fp, name, s, e, body in new_chunks:
        # Prefix with name so the embedding sees both signature and body.
        texts_for_embedding.append(f"{name}\n{body}")

    BATCH = 32
    rows: list[tuple] = []
    now = datetime.now().isoformat(timespec="seconds")
    for i in range(0, len(texts_for_embedding), BATCH):
        batch_texts = texts_for_embedding[i:i + BATCH]
        batch_chunks = new_chunks[i:i + BATCH]
        embs = _embed_texts(batch_texts)
        if embs is None:
            return 0  # Ollama unavailable — bail without writing
        for (fp, name, s, e, body), emb in zip(batch_chunks, embs):
            try:
                mtime = fp.stat().st_mtime
            except OSError:
                continue
            rows.append((
                uuid.uuid4().hex,
                str(rootp),
                str(fp.resolve()),
                mtime,
                name,
                s, e,
                body,
                json.dumps(emb),
                now,
            ))

    if rows:
        conn.executemany(
            "INSERT INTO code_chunks "
            "(id, project_path, file_path, file_mtime, chunk_name, "
            "start_line, end_line, body, embedding, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    return len(rows)


# ── Search ───────────────────────────────────────────────────────────────────

def embed_search(
    query: str,
    top_k: int = 5,
    project_path: str | None = None,
) -> list[dict]:
    """Embedding-only search over indexed code chunks. Returns ranked dicts.

    Returns [] if no index exists or the query embedding fails.
    """
    from .memory import current_project_path

    query = (query or "").strip()
    if not query:
        return []
    project = project_path or current_project_path()
    conn = _conn()

    cur = conn.execute(
        "SELECT * FROM code_chunks WHERE project_path = ?",
        (str(Path(project).resolve()),),
    )
    rows = cur.fetchall()
    if not rows:
        return []

    qemb_list = _embed_texts([query])
    if qemb_list is None or not qemb_list:
        return []
    qemb = qemb_list[0]

    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        try:
            emb = json.loads(row["embedding"])
        except (json.JSONDecodeError, TypeError):
            continue
        scored.append((_cosine(qemb, emb), row))

    scored.sort(key=lambda x: -x[0])
    out: list[dict] = []
    for score, row in scored[:top_k]:
        out.append({
            "score": score,
            "file_path": row["file_path"],
            "chunk_name": row["chunk_name"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "body": row["body"],
        })
    return out


def hybrid_search(
    query: str,
    top_k: int = 5,
    project_path: str | None = None,
) -> list[dict]:
    """Combine embedding search with FTS5 keyword search via RRF.

    Reciprocal Rank Fusion: ``score(item) = sum(1 / (RRF_K + rank_in_channel))``
    over each retrieval channel. No calibration needed, robust to scale
    differences between cosine and BM25.
    """
    embed_results = embed_search(query, top_k=top_k * 2, project_path=project_path)

    # Keyword channel: search the in-DB chunk bodies via LIKE (FTS5 is on the
    # memory table, not chunks, so we'd need a separate FTS table; LIKE is fine
    # for the v1 hybrid since chunk counts are small).
    conn = _conn()
    from .memory import current_project_path
    project = project_path or current_project_path()
    needle = f"%{query.lower()}%"
    cur = conn.execute(
        "SELECT file_path, chunk_name, start_line, end_line, body "
        "FROM code_chunks WHERE project_path = ? AND "
        "(lower(body) LIKE ? OR lower(chunk_name) LIKE ?) LIMIT ?",
        (str(Path(project).resolve()), needle, needle, top_k * 2),
    )
    keyword_rows = cur.fetchall()
    keyword_results = [
        {
            "file_path": r["file_path"],
            "chunk_name": r["chunk_name"],
            "start_line": r["start_line"],
            "end_line": r["end_line"],
            "body": r["body"],
            "score": 0.0,
        }
        for r in keyword_rows
    ]

    def _key(r: dict) -> tuple:
        return (r["file_path"], r["chunk_name"], r["start_line"])

    fused: dict[tuple, dict] = {}
    for rank, r in enumerate(embed_results, start=1):
        key = _key(r)
        fused.setdefault(key, dict(r))
        fused[key]["rrf"] = fused[key].get("rrf", 0.0) + 1.0 / (RRF_K + rank)

    for rank, r in enumerate(keyword_results, start=1):
        key = _key(r)
        if key in fused:
            fused[key]["rrf"] = fused[key].get("rrf", 0.0) + 1.0 / (RRF_K + rank)
        else:
            entry = dict(r)
            entry["rrf"] = 1.0 / (RRF_K + rank)
            fused[key] = entry

    ranked = sorted(fused.values(), key=lambda r: -r.get("rrf", 0.0))
    return ranked[:top_k]


# ── Tool wiring ──────────────────────────────────────────────────────────────

def format_results(results: list[dict]) -> str:
    if not results:
        return "No matching code chunks found."
    out: list[str] = [f"Found {len(results)} code chunk(s):\n"]
    for r in results:
        path = r.get("file_path", "")
        name = r.get("chunk_name", "")
        sl = r.get("start_line", 0)
        el = r.get("end_line", 0)
        body = (r.get("body", "") or "").rstrip()
        if len(body) > 800:
            body = body[:800] + "\n# ...[truncated]"
        out.append(f"### {path}:{sl}-{el}  ({name})\n```\n{body}\n```\n")
    return "\n".join(out)
