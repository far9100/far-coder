"""Persistent project-scoped memory backed by SQLite + FTS5.

Single-file database at ``~/.farcode_memory.db``. Every memory entry has:

  - ``kind``         — ``session_summary`` | ``task`` | ``decision``
  - ``project_path`` — git root or cwd at write time, used for project scoping
  - ``tags``         — JSON array of short string tags
  - ``files_touched``— JSON array of file paths the memory relates to
  - ``summary``      — the human-readable lesson

Search is driven by a contentless FTS5 virtual table that mirrors the rows
through INSERT/DELETE triggers. Queries are tokenized by SQLite's built-in
``unicode61`` tokenizer with ``_./-`` added as token characters so identifiers
and file paths survive intact, plus a camelCase splitter that injects an
extra "My Class Name" form alongside the original token. Result ranking is
the standard FTS5 BM25.

If FTS5 is unavailable (older SQLite builds), search degrades to a LIKE
scan over a derived ``search_blob`` column.

A one-time migration ingests the legacy ``~/.farcode_memory.jsonl`` /
``~/.ai_coder_memory.jsonl`` file (whichever exists) and renames it
``.jsonl.migrated`` so re-runs are idempotent.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

MEMORY_DB = Path.home() / ".farcode_memory.db"
LEGACY_JSONL_PATHS = [
    Path.home() / ".farcode_memory.jsonl",
    Path.home() / ".ai_coder_memory.jsonl",
]
_PROMPT_PER_ENTRY = 600
_PROMPT_TOTAL = 1500
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_conn_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_fts_available: bool | None = None


# ── Connection / schema ───────────────────────────────────────────────────────

def _fts5_supported() -> bool:
    global _fts_available
    if _fts_available is not None:
        return _fts_available
    try:
        probe = sqlite3.connect(":memory:")
        try:
            probe.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
            _fts_available = True
        finally:
            probe.close()
    except sqlite3.OperationalError:
        _fts_available = False
    return _fts_available


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            session_id TEXT,
            project_path TEXT,
            kind TEXT NOT NULL,
            tags TEXT,
            files_touched TEXT,
            summary TEXT NOT NULL,
            search_blob TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_project ON memory(project_path)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory(created_at DESC)"
    )

    if _fts5_supported():
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                summary, tags, files_touched, project_path, search_blob,
                content='memory', content_rowid='rowid',
                tokenize="unicode61 remove_diacritics 2 tokenchars '_./-'"
            )
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memory BEGIN
                INSERT INTO memory_fts(rowid, summary, tags, files_touched, project_path, search_blob)
                VALUES (new.rowid, new.summary, new.tags, new.files_touched,
                        new.project_path, new.search_blob);
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memory BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, summary, tags, files_touched, project_path, search_blob)
                VALUES ('delete', old.rowid, old.summary, old.tags, old.files_touched,
                        old.project_path, old.search_blob);
            END
            """
        )
    conn.commit()


def _get_conn() -> sqlite3.Connection:
    global _conn
    with _conn_lock:
        if _conn is None:
            MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(str(MEMORY_DB), check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _ensure_schema(_conn)
            _migrate_jsonl_once(_conn)
        return _conn


# ── Camel splitting ───────────────────────────────────────────────────────────

def _split_camel(text: str) -> str:
    """Expand camelCase tokens with a space-separated and underscore form.

    ``MyClassName`` → ``MyClassName My Class Name``.
    ``foo_bar`` → ``foo_bar foo bar``.
    Used only to enrich the FTS index/queries; the stored ``summary`` is unchanged.
    """
    if not text:
        return ""
    out: list[str] = [text]
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        spaced = _CAMEL_RE.sub(" ", tok)
        if "_" in tok:
            spaced = spaced.replace("_", " ")
        if spaced != tok:
            out.append(spaced)
    return " ".join(out)


def _build_search_blob(
    summary: str,
    tags: Iterable[str] | None,
    files_touched: Iterable[str] | None,
    project_path: str | None,
) -> str:
    parts: list[str] = [_split_camel(summary)]
    if tags:
        parts.append(_split_camel(" ".join(tags)))
    if files_touched:
        parts.append(_split_camel(" ".join(files_touched)))
    if project_path:
        parts.append(_split_camel(project_path))
    return "\n".join(p for p in parts if p)


# ── Project-path detection ────────────────────────────────────────────────────

_project_cache: dict[str, str] = {}


def current_project_path(start: str | os.PathLike[str] | None = None) -> str:
    """Return the git root containing ``start`` (default cwd), else ``start`` itself."""
    base = Path(start or os.getcwd()).resolve()
    cache_key = str(base)
    if cache_key in _project_cache:
        return _project_cache[cache_key]
    cur = base
    found = base
    while True:
        if (cur / ".git").exists():
            found = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    result = str(found)
    _project_cache[cache_key] = result
    return result


# ── JSONL migration ───────────────────────────────────────────────────────────

def _migrate_jsonl_once(conn: sqlite3.Connection) -> None:
    legacy = next((p for p in LEGACY_JSONL_PATHS if p.exists()), None)
    if legacy is None:
        return
    cur = conn.execute("SELECT count(*) FROM memory")
    if cur.fetchone()[0] > 0:
        try:
            legacy.rename(legacy.with_suffix(legacy.suffix + ".migrated"))
        except OSError:
            pass
        return

    rows: list[tuple] = []
    for raw in legacy.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        summary = entry.get("summary") or ""
        if not summary:
            continue
        rows.append(
            (
                entry.get("id") or uuid.uuid4().hex,
                entry.get("created_at") or datetime.now().isoformat(timespec="seconds"),
                entry.get("session_id"),
                entry.get("project_path"),
                entry.get("kind") or "session_summary",
                json.dumps(entry.get("tags") or []),
                json.dumps(entry.get("files_touched") or []),
                summary,
                _build_search_blob(
                    summary,
                    entry.get("tags"),
                    entry.get("files_touched"),
                    entry.get("project_path"),
                ),
            )
        )

    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO memory "
            "(id, created_at, session_id, project_path, kind, tags, files_touched, summary, search_blob) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    try:
        legacy.rename(legacy.with_suffix(legacy.suffix + ".migrated"))
    except OSError:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def append_entry(
    session_id: str,
    summary: str,
    kind: str = "session_summary",
    tags: Iterable[str] | None = None,
    files_touched: Iterable[str] | None = None,
    project_path: str | None = None,
) -> None:
    """Insert a new memory entry. ``tags``/``files_touched`` may be None."""
    summary = (summary or "").strip()
    if not summary:
        return
    tags_list = list(tags) if tags else []
    files_list = list(files_touched) if files_touched else []
    blob = _build_search_blob(summary, tags_list, files_list, project_path)

    conn = _get_conn()
    with _conn_lock:
        conn.execute(
            "INSERT INTO memory "
            "(id, created_at, session_id, project_path, kind, tags, files_touched, summary, search_blob) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex,
                datetime.now().isoformat(timespec="seconds"),
                session_id,
                project_path,
                kind,
                json.dumps(tags_list),
                json.dumps(files_list),
                summary,
                blob,
            ),
        )
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for key in ("tags", "files_touched"):
        raw = d.get(key)
        if raw:
            try:
                d[key] = json.loads(raw)
            except json.JSONDecodeError:
                d[key] = []
        else:
            d[key] = []
    return d


def load_recent(n: int = 5, project_path: str | None = None) -> list[dict]:
    """Return the most recent ``n`` memories, optionally project-scoped."""
    conn = _get_conn()
    if project_path:
        cur = conn.execute(
            "SELECT * FROM memory WHERE project_path = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (project_path, n),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM memory ORDER BY created_at DESC LIMIT ?", (n,)
        )
    return [_row_to_dict(r) for r in cur.fetchall()]


def _fts_query_for(query: str) -> str:
    """Build a permissive FTS5 MATCH query: each token is OR'd with prefix match."""
    expanded = _split_camel(query)
    tokens = re.findall(r"[\w./-]+", expanded)
    tokens = [t for t in tokens if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t.replace(chr(34), chr(34) * 2)}"*' for t in tokens)


def search(
    query: str,
    top_k: int = 5,
    project_path: str | None = None,
    scope: str = "project",
) -> list[dict]:
    """Return up to ``top_k`` memories ranked by relevance to ``query``.

    With ``scope="project"`` and a project_path, results are filtered to that
    project; if zero matches, falls back to global. With ``scope="all"`` the
    project_path filter is ignored.
    """
    query = (query or "").strip()
    if not query:
        return []
    conn = _get_conn()

    if _fts5_supported():
        match = _fts_query_for(query)
        if not match:
            return []
        if scope == "project" and project_path:
            cur = conn.execute(
                "SELECT m.* FROM memory_fts "
                "JOIN memory m ON m.rowid = memory_fts.rowid "
                "WHERE memory_fts MATCH ? AND m.project_path = ? "
                "ORDER BY bm25(memory_fts) LIMIT ?",
                (match, project_path, top_k),
            )
            rows = cur.fetchall()
            if rows:
                return [_row_to_dict(r) for r in rows]
        cur = conn.execute(
            "SELECT m.* FROM memory_fts "
            "JOIN memory m ON m.rowid = memory_fts.rowid "
            "WHERE memory_fts MATCH ? "
            "ORDER BY bm25(memory_fts) LIMIT ?",
            (match, top_k),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # FTS5-less fallback: LIKE scan over the search_blob.
    needle = f"%{query.lower()}%"
    if scope == "project" and project_path:
        cur = conn.execute(
            "SELECT * FROM memory WHERE project_path = ? AND lower(search_blob) LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (project_path, needle, top_k),
        )
        rows = cur.fetchall()
        if rows:
            return [_row_to_dict(r) for r in rows]
    cur = conn.execute(
        "SELECT * FROM memory WHERE lower(search_blob) LIKE ? "
        "ORDER BY created_at DESC LIMIT ?",
        (needle, top_k),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def format_for_prompt(
    entries: list[dict],
    max_chars: int = _PROMPT_TOTAL,
    per_entry: int = _PROMPT_PER_ENTRY,
) -> str:
    """Render memories as a "## Past Work" block to inject into the system prompt."""
    if not entries:
        return ""
    lines = ["## Past Work"]
    used = len(lines[0])
    for e in entries:
        date = (e.get("created_at") or "")[:10]
        kind = e.get("kind") or ""
        summary = (e.get("summary") or "").strip()
        if not summary:
            continue
        if len(summary) > per_entry:
            summary = summary[:per_entry].rstrip() + "..."
        proj = e.get("project_path") or ""
        proj_short = Path(proj).name if proj else ""
        header_bits = [b for b in (date, kind, proj_short) if b]
        header = " ".join(f"[{b}]" for b in header_bits)
        line = f"- {header} {summary}".rstrip()
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if len(lines) > 1 else ""
