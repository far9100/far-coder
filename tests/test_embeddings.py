"""Tests for the embedding / hybrid code search module."""

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def emb_module(tmp_path, monkeypatch):
    """Reload memory + embeddings against a temp DB and stub out Ollama embed()."""
    if "farcode.memory" in sys.modules:
        del sys.modules["farcode.memory"]
    import farcode.memory as memory_mod

    memory_mod.MEMORY_DB = tmp_path / "emb-mem.db"
    memory_mod.LEGACY_JSONL_PATHS = []
    memory_mod._conn = None
    memory_mod._fts_available = None
    memory_mod._project_cache.clear()

    if "farcode.embeddings" in sys.modules:
        del sys.modules["farcode.embeddings"]
    import farcode.embeddings as emb_mod

    monkeypatch.chdir(tmp_path)
    return emb_mod


def _stub_embed(monkeypatch, mod, vector_for: dict[str, list[float]]):
    """Make _embed_texts return canned vectors keyed by substring match."""
    default = [0.01] * 8

    def fake(texts):
        out = []
        for t in texts:
            chosen = default
            for substring, v in vector_for.items():
                if substring in t:
                    chosen = v
                    break
            out.append(list(chosen))
        return out

    monkeypatch.setattr(mod, "_embed_texts", fake)


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_python_chunks_extracts_top_level_defs(emb_module, tmp_path):
    f = tmp_path / "m.py"
    f.write_text(
        "def foo():\n    return 1\n\n"
        "class Bar:\n    def method(self):\n        return 2\n",
        encoding="utf-8",
    )
    chunks = emb_module._python_chunks(f)
    names = [c[0] for c in chunks]
    assert "foo" in names
    assert "Bar" in names


def test_python_chunks_skips_unparseable(emb_module, tmp_path):
    f = tmp_path / "broken.py"
    f.write_text("def foo(\n", encoding="utf-8")
    assert emb_module._python_chunks(f) == []


def test_python_chunks_truncates_huge_chunk(emb_module, tmp_path):
    f = tmp_path / "huge.py"
    body_lines = ["def big():"] + ["    x = 1  # padding line " * 5 for _ in range(2000)]
    f.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
    chunks = emb_module._python_chunks(f)
    assert chunks
    _, _, _, body = chunks[0]
    assert len(body) <= emb_module.MAX_CHUNK_CHARS + 50


# ── Cosine similarity ────────────────────────────────────────────────────────

def test_cosine_identical_vectors_is_one(emb_module):
    v = [1.0, 2.0, 3.0]
    assert abs(emb_module._cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero(emb_module):
    assert emb_module._cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_handles_zero_vectors(emb_module):
    assert emb_module._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_dimension_mismatch_returns_zero(emb_module):
    assert emb_module._cosine([1.0], [1.0, 1.0]) == 0.0


# ── Indexing ──────────────────────────────────────────────────────────────────

def test_index_project_indexes_chunks(emb_module, tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def alpha(): pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def beta(): pass\n", encoding="utf-8")
    _stub_embed(monkeypatch, emb_module, {"alpha": [1.0] * 8, "beta": [0.5] * 8})

    n = emb_module.index_project(str(tmp_path))
    assert n == 2

    conn = emb_module._conn()
    cur = conn.execute(
        "SELECT chunk_name FROM code_chunks WHERE project_path = ?",
        (str(tmp_path.resolve()),),
    )
    names = sorted(r[0] for r in cur.fetchall())
    assert names == ["alpha", "beta"]


def test_index_project_skips_unchanged_files(emb_module, tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def alpha(): pass\n", encoding="utf-8")
    _stub_embed(monkeypatch, emb_module, {"alpha": [1.0] * 8})

    first = emb_module.index_project(str(tmp_path))
    second = emb_module.index_project(str(tmp_path))
    assert first == 1
    assert second == 0  # already indexed at the same mtime


def test_index_project_force_reindexes(emb_module, tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def alpha(): pass\n", encoding="utf-8")
    _stub_embed(monkeypatch, emb_module, {"alpha": [1.0] * 8})

    emb_module.index_project(str(tmp_path))
    forced = emb_module.index_project(str(tmp_path), force=True)
    assert forced == 1


def test_index_project_returns_zero_if_embed_unavailable(emb_module, tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def alpha(): pass\n", encoding="utf-8")
    monkeypatch.setattr(emb_module, "_embed_texts", lambda texts: None)
    assert emb_module.index_project(str(tmp_path)) == 0


def test_index_project_skips_node_modules(emb_module, tmp_path, monkeypatch):
    (tmp_path / "real.py").write_text("def real(): pass\n", encoding="utf-8")
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    (nm / "fake.py").write_text("def fake(): pass\n", encoding="utf-8")
    _stub_embed(monkeypatch, emb_module, {})

    n = emb_module.index_project(str(tmp_path))
    assert n == 1
    conn = emb_module._conn()
    rows = conn.execute(
        "SELECT chunk_name FROM code_chunks WHERE project_path = ?",
        (str(tmp_path.resolve()),),
    ).fetchall()
    assert len(rows) == 1


# ── Search ────────────────────────────────────────────────────────────────────

def test_embed_search_ranks_by_cosine(emb_module, tmp_path, monkeypatch):
    (tmp_path / "auth.py").write_text(
        "def login_user(): pass\n", encoding="utf-8"
    )
    (tmp_path / "draw.py").write_text(
        "def render_pixel(): pass\n", encoding="utf-8"
    )
    # "login" is close to "auth" semantically; use distinct vectors
    _stub_embed(monkeypatch, emb_module, {
        "login_user": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "render_pixel": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    emb_module.index_project(str(tmp_path))

    # Query embedding closer to login_user
    monkeypatch.setattr(
        emb_module, "_embed_texts",
        lambda texts: [[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * len(texts),
    )
    results = emb_module.embed_search("how does authentication work", top_k=2,
                                      project_path=str(tmp_path))
    assert results
    assert results[0]["chunk_name"] == "login_user"


def test_embed_search_returns_empty_when_no_index(emb_module, tmp_path, monkeypatch):
    monkeypatch.setattr(emb_module, "_embed_texts", lambda texts: [[1.0] * 8])
    out = emb_module.embed_search("anything", project_path=str(tmp_path))
    assert out == []


def test_embed_search_empty_query(emb_module, tmp_path):
    out = emb_module.embed_search("", project_path=str(tmp_path))
    assert out == []


def test_embed_search_returns_empty_when_embed_fails(emb_module, tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def x(): pass\n", encoding="utf-8")
    _stub_embed(monkeypatch, emb_module, {"x": [1.0] * 8})
    emb_module.index_project(str(tmp_path))

    monkeypatch.setattr(emb_module, "_embed_texts", lambda texts: None)
    out = emb_module.embed_search("query", project_path=str(tmp_path))
    assert out == []


# ── Hybrid search (RRF) ──────────────────────────────────────────────────────

def test_hybrid_search_fuses_both_channels(emb_module, tmp_path, monkeypatch):
    """A chunk that ranks well in BOTH channels should beat one that ranks
    well in only one."""
    (tmp_path / "a.py").write_text("def parse_token(): pass\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def something_else(): pass\n", encoding="utf-8")

    _stub_embed(monkeypatch, emb_module, {
        "parse_token": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "something_else": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    emb_module.index_project(str(tmp_path))

    # Query embedding favors parse_token; query KEYWORD also matches "token".
    monkeypatch.setattr(
        emb_module, "_embed_texts",
        lambda texts: [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * len(texts),
    )
    results = emb_module.hybrid_search("token", top_k=2,
                                       project_path=str(tmp_path))
    assert results
    assert results[0]["chunk_name"] == "parse_token"


def test_hybrid_search_keyword_only_match_still_returned(emb_module, tmp_path, monkeypatch):
    """Even if embedding similarity is zero, keyword match should surface a chunk."""
    (tmp_path / "x.py").write_text("def widget_factory(): pass\n", encoding="utf-8")
    _stub_embed(monkeypatch, emb_module, {})  # all zeroes
    emb_module.index_project(str(tmp_path))

    monkeypatch.setattr(
        emb_module, "_embed_texts",
        lambda texts: [[0.5] * 8] * len(texts),
    )
    results = emb_module.hybrid_search("widget", top_k=5,
                                       project_path=str(tmp_path))
    assert any(r["chunk_name"] == "widget_factory" for r in results)


# ── format_results ────────────────────────────────────────────────────────────

def test_format_results_empty(emb_module):
    assert "No matching" in emb_module.format_results([])


def test_format_results_includes_file_lines_and_body(emb_module):
    out = emb_module.format_results([{
        "file_path": "src/foo.py",
        "chunk_name": "bar",
        "start_line": 10,
        "end_line": 20,
        "body": "def bar():\n    return 1\n",
        "score": 0.9,
    }])
    assert "src/foo.py:10-20" in out
    assert "bar" in out
    assert "def bar()" in out


def test_format_results_truncates_huge_body(emb_module):
    out = emb_module.format_results([{
        "file_path": "x.py",
        "chunk_name": "huge",
        "start_line": 1,
        "end_line": 2,
        "body": "x" * 5000,
        "score": 1.0,
    }])
    assert "[truncated]" in out


# ── recall_code tool wiring ──────────────────────────────────────────────────

def test_recall_code_tool_handles_missing_index(emb_module, tmp_path, monkeypatch):
    """When Ollama embed isn't available, the tool returns an error string
    rather than crashing the agent loop."""
    monkeypatch.setattr(emb_module, "_embed_texts", lambda texts: None)
    from farcode.tools import _recall_code

    out = _recall_code("anything")
    # Either "No matching" (empty index) or an error mentioning the model — both fine.
    assert isinstance(out, str)
    assert "Error" not in out or "ollama pull" in out or "unavailable" in out


def test_recall_code_via_execute_tool(emb_module, tmp_path, monkeypatch):
    (tmp_path / "auth.py").write_text("def login(): pass\n", encoding="utf-8")
    _stub_embed(monkeypatch, emb_module, {"login": [1.0] * 8})

    from farcode.tools import execute_tool

    out = execute_tool("recall_code", {"query": "login", "top_k": 3})
    assert "auth.py" in out or "No matching" in out
