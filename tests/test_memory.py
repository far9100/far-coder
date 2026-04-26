"""Tests for the SQLite + FTS5 memory store."""

import json
import sys
from pathlib import Path

import pytest


def _reset_memory_module(tmp_db: Path):
    """Reload farcode.memory with paths pointed at the temp directory."""
    if "farcode.memory" in sys.modules:
        del sys.modules["farcode.memory"]
    import farcode.memory as memory_mod

    memory_mod.MEMORY_DB = tmp_db
    memory_mod.LEGACY_JSONL_PATHS = []
    memory_mod._conn = None
    memory_mod._fts_available = None
    memory_mod._project_cache.clear()
    return memory_mod


@pytest.fixture
def memory(tmp_path, monkeypatch):
    db = tmp_path / "mem.db"
    return _reset_memory_module(db)


# ── append_entry / load_recent ────────────────────────────────────────────────

def test_append_and_load_recent(memory):
    memory.append_entry("s1", "fixed the auth bug", project_path="/repo/a")
    memory.append_entry("s2", "added dark mode", project_path="/repo/b")

    entries = memory.load_recent(5)
    assert len(entries) == 2
    summaries = {e["summary"] for e in entries}
    assert summaries == {"fixed the auth bug", "added dark mode"}


def test_load_recent_project_scope(memory):
    memory.append_entry("s1", "repo A work", project_path="/repo/a")
    memory.append_entry("s2", "repo B work", project_path="/repo/b")

    entries = memory.load_recent(5, project_path="/repo/a")
    assert len(entries) == 1
    assert entries[0]["summary"] == "repo A work"


def test_append_skips_empty_summary(memory):
    memory.append_entry("s1", "")
    memory.append_entry("s2", "   ")
    assert memory.load_recent(5) == []


def test_append_stores_tags_and_files(memory):
    memory.append_entry(
        "s1",
        "did the thing",
        kind="task",
        tags=["auth", "bugfix"],
        files_touched=["src/a.py", "src/b.py"],
        project_path="/repo",
    )
    [entry] = memory.load_recent(5)
    assert entry["tags"] == ["auth", "bugfix"]
    assert entry["files_touched"] == ["src/a.py", "src/b.py"]
    assert entry["kind"] == "task"


# ── search ────────────────────────────────────────────────────────────────────

def test_search_finds_keyword(memory):
    memory.append_entry("s1", "fixed authentication middleware", project_path="/repo")
    memory.append_entry("s2", "rewrote pagination", project_path="/repo")
    results = memory.search("authentication", top_k=5, project_path="/repo")
    assert len(results) == 1
    assert "authentication" in results[0]["summary"]


def test_search_camel_case_identifier(memory):
    memory.append_entry(
        "s1",
        "renamed MyAuthHandler to AuthService",
        project_path="/repo",
    )
    # plain lowercase 'auth' should still surface the camelCase identifier
    results = memory.search("auth", top_k=5, project_path="/repo")
    assert len(results) == 1


def test_search_file_path_match(memory):
    memory.append_entry(
        "s1",
        "tweaked the login flow",
        files_touched=["src/farcode/chat.py"],
        project_path="/repo",
    )
    results = memory.search("chat.py", top_k=5, project_path="/repo")
    assert len(results) == 1


def test_search_project_scope_falls_back_to_global(memory):
    memory.append_entry("s1", "rare unique-keyword work", project_path="/repo/other")
    # Searching from /repo/current — project-scoped finds nothing, fallback returns it.
    results = memory.search(
        "unique-keyword", top_k=5, project_path="/repo/current", scope="project"
    )
    assert len(results) == 1
    assert results[0]["project_path"] == "/repo/other"


def test_search_scope_all_ignores_project(memory):
    memory.append_entry("s1", "global tip", project_path="/repo/x")
    memory.append_entry("s2", "global tip again", project_path="/repo/y")
    results = memory.search("tip", top_k=5, project_path="/repo/x", scope="all")
    assert len(results) == 2


def test_search_empty_query(memory):
    memory.append_entry("s1", "anything", project_path="/repo")
    assert memory.search("", project_path="/repo") == []


# ── format_for_prompt ─────────────────────────────────────────────────────────

def test_format_for_prompt_basic(memory):
    memory.append_entry(
        "s1",
        "did the thing well",
        kind="task",
        project_path="/repo/myproject",
    )
    entries = memory.load_recent(5)
    out = memory.format_for_prompt(entries)
    assert out.startswith("## Past Work")
    assert "did the thing well" in out
    assert "myproject" in out


def test_format_for_prompt_empty():
    # No fixture needed — function is pure.
    import farcode.memory as memory_mod
    assert memory_mod.format_for_prompt([]) == ""


def test_format_for_prompt_respects_total_cap(memory):
    big = "x" * 1000
    for i in range(10):
        memory.append_entry(f"s{i}", f"{big}-{i}", project_path="/repo")
    entries = memory.load_recent(10)
    out = memory.format_for_prompt(entries, max_chars=300)
    # The header alone is short; cap should clamp the body.
    assert len(out) <= 600  # header + a few bullets, never close to 10*1000


# ── _split_camel ──────────────────────────────────────────────────────────────

def test_split_camel(memory):
    out = memory._split_camel("MyClassName")
    assert "MyClassName" in out
    assert "My Class Name" in out


def test_split_camel_underscore(memory):
    out = memory._split_camel("foo_bar_baz")
    assert "foo_bar_baz" in out
    assert "foo bar baz" in out


def test_split_camel_empty(memory):
    assert memory._split_camel("") == ""


# ── current_project_path ──────────────────────────────────────────────────────

def test_current_project_path_finds_git_root(memory, tmp_path):
    repo = tmp_path / "repo"
    sub = repo / "src" / "deep"
    sub.mkdir(parents=True)
    (repo / ".git").mkdir()
    result = memory.current_project_path(sub)
    assert Path(result).resolve() == repo.resolve()


def test_current_project_path_falls_back_to_input(memory, tmp_path):
    no_git = tmp_path / "elsewhere"
    no_git.mkdir()
    result = memory.current_project_path(no_git)
    # Without a .git anywhere upward, we just return the resolved input dir.
    assert Path(result).resolve() == no_git.resolve()


# ── JSONL migration ───────────────────────────────────────────────────────────

def test_migrate_jsonl_once(tmp_path, monkeypatch):
    # Build a legacy JSONL with two entries
    legacy = tmp_path / "old.jsonl"
    legacy.write_text(
        "\n".join(
            [
                json.dumps({"summary": "first lesson", "session_id": "x"}),
                json.dumps({"summary": "second lesson", "session_id": "y"}),
            ]
        ),
        encoding="utf-8",
    )

    # Reload module and point it at our tmp paths
    if "farcode.memory" in sys.modules:
        del sys.modules["farcode.memory"]
    import farcode.memory as memory_mod
    memory_mod.MEMORY_DB = tmp_path / "mem.db"
    memory_mod.LEGACY_JSONL_PATHS = [legacy]
    memory_mod._conn = None
    memory_mod._fts_available = None

    entries = memory_mod.load_recent(10)
    assert len(entries) == 2
    summaries = {e["summary"] for e in entries}
    assert summaries == {"first lesson", "second lesson"}

    # Legacy file should be renamed
    assert not legacy.exists()
    assert (tmp_path / "old.jsonl.migrated").exists()
