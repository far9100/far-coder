import sys

import pytest
from pathlib import Path

from farcode.tools import (
    _create_file,
    _edit_file,
    _list_directory,
    _read_file,
    _save_memory,
    _search_in_files,
    execute_tool,
)


# ── read_file ─────────────────────────────────────────────────────────────────

def test_read_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    assert _read_file(str(f)) == "hello world"


def test_read_file_not_found(tmp_path):
    result = _read_file(str(tmp_path / "missing.txt"))
    assert result.startswith("Error:")


def test_read_file_not_a_file(tmp_path):
    result = _read_file(str(tmp_path))
    assert result.startswith("Error:")


# ── edit_file ─────────────────────────────────────────────────────────────────

def test_edit_file(tmp_path):
    f = tmp_path / "edit.txt"
    f.write_text("foo bar baz", encoding="utf-8")
    result = _edit_file(str(f), "foo", "qux")
    assert result.startswith("OK:")
    assert f.read_text(encoding="utf-8") == "qux bar baz"


def test_edit_file_replaces_only_first(tmp_path):
    f = tmp_path / "dup.txt"
    f.write_text("a a a", encoding="utf-8")
    _edit_file(str(f), "a", "b")
    assert f.read_text(encoding="utf-8") == "b a a"


def test_edit_file_not_found(tmp_path):
    result = _edit_file(str(tmp_path / "missing.txt"), "x", "y")
    assert result.startswith("Error:")


def test_edit_file_old_str_not_found(tmp_path):
    f = tmp_path / "no_match.txt"
    f.write_text("hello", encoding="utf-8")
    result = _edit_file(str(f), "xyz", "abc")
    assert result.startswith("Error:")


# ── list_directory ────────────────────────────────────────────────────────────

def test_list_directory(tmp_path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    result = _list_directory(str(tmp_path))
    assert "a.py" in result
    assert "sub" in result


def test_list_directory_depth(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "file.txt").write_text("", encoding="utf-8")
    result = _list_directory(str(tmp_path), depth=1)
    assert "file.txt" not in result
    result = _list_directory(str(tmp_path), depth=4)
    assert "file.txt" in result


def test_list_directory_not_found(tmp_path):
    result = _list_directory(str(tmp_path / "missing"))
    assert result.startswith("Error:")


def test_list_directory_not_a_dir(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("", encoding="utf-8")
    result = _list_directory(str(f))
    assert result.startswith("Error:")


# ── search_in_files ───────────────────────────────────────────────────────────

def test_search_in_files_finds_match(tmp_path):
    (tmp_path / "a.py").write_text("def foo(): pass", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar(): pass", encoding="utf-8")
    result = _search_in_files("def foo", str(tmp_path), "*.py")
    assert "a.py" in result
    assert "b.py" not in result


def test_search_in_files_no_match(tmp_path):
    (tmp_path / "a.py").write_text("hello", encoding="utf-8")
    result = _search_in_files("xyz123", str(tmp_path))
    assert "No matches" in result


def test_search_in_files_invalid_regex(tmp_path):
    result = _search_in_files("[invalid", str(tmp_path))
    assert result.startswith("Error:")


def test_search_in_files_not_found():
    result = _search_in_files("foo", "/nonexistent/path/xyz")
    assert result.startswith("Error:")


# ── create_file ───────────────────────────────────────────────────────────────

def test_create_file(tmp_path):
    p = tmp_path / "new.txt"
    result = _create_file(str(p), "content here")
    assert result.startswith("OK:")
    assert p.read_text(encoding="utf-8") == "content here"


def test_create_file_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "new.txt"
    result = _create_file(str(p), "nested")
    assert result.startswith("OK:")
    assert p.exists()


def test_create_file_already_exists(tmp_path):
    p = tmp_path / "exists.txt"
    p.write_text("original", encoding="utf-8")
    result = _create_file(str(p), "new content")
    assert result.startswith("Error:")
    assert p.read_text(encoding="utf-8") == "original"


# ── read_file with offset/limit ───────────────────────────────────────────────

def test_read_file_with_offset(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")
    result = _read_file(str(f), offset=3, limit=2)
    assert "line3" in result
    assert "line4" in result
    assert "line5" not in result
    assert "line1" not in result


def test_read_file_limit_only(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")
    result = _read_file(str(f), limit=3)
    assert "line1" in result
    assert "line3" in result
    assert "line4" not in result


def test_read_file_offset_past_end(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("only_line", encoding="utf-8")
    result = _read_file(str(f), offset=100, limit=10)
    # Offset past the last line returns the slice header but no content; just don't crash.
    assert "Error" not in result


def test_read_file_full_no_offset(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("complete contents", encoding="utf-8")
    assert _read_file(str(f)) == "complete contents"


def test_read_file_truncates_huge_file(tmp_path):
    f = tmp_path / "big.txt"
    # 200 KB file — exceeds the 50 KB cap
    f.write_text("x" * 200_000, encoding="utf-8")
    result = _read_file(str(f))
    assert "[truncated" in result
    assert len(result) < 200_000


# ── save_memory tool ──────────────────────────────────────────────────────────

@pytest.fixture
def reset_memory(tmp_path, monkeypatch):
    """Point memory storage at a temp DB and clean module state."""
    if "farcode.memory" in sys.modules:
        del sys.modules["farcode.memory"]
    import farcode.memory as memory_mod
    memory_mod.MEMORY_DB = tmp_path / "tools-mem.db"
    memory_mod.LEGACY_JSONL_PATHS = []
    memory_mod._conn = None
    memory_mod._fts_available = None
    memory_mod._project_cache.clear()
    monkeypatch.chdir(tmp_path)
    return memory_mod


def test_save_memory_round_trip(reset_memory, tmp_path):
    result = _save_memory("renamed widget API to use snake_case")
    assert result.startswith("OK:")
    entries = reset_memory.load_recent(5)
    assert len(entries) == 1
    assert entries[0]["summary"] == "renamed widget API to use snake_case"
    assert entries[0]["kind"] == "task"


def test_save_memory_with_tags_and_files(reset_memory, tmp_path):
    _save_memory(
        "fixed flaky test",
        tags=["tests", "bugfix"],
        files_touched=["tests/test_a.py"],
    )
    entries = reset_memory.load_recent(5)
    assert entries[0]["tags"] == ["tests", "bugfix"]
    assert entries[0]["files_touched"] == ["tests/test_a.py"]


def test_save_memory_rejects_empty(reset_memory):
    result = _save_memory("")
    assert result.startswith("Error:")
    assert reset_memory.load_recent(5) == []


def test_execute_tool_dispatches_save_memory(reset_memory):
    result = execute_tool("save_memory", {"summary": "via execute_tool"})
    assert result.startswith("OK:")
    assert reset_memory.load_recent(5)[0]["summary"] == "via execute_tool"


def test_execute_tool_dispatches_recall_memory(reset_memory):
    _save_memory("the auth refactor is done")
    result = execute_tool("recall_memory", {"query": "auth", "scope": "all"})
    assert "auth refactor" in result
