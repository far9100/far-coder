import pytest
from pathlib import Path

from ai_coder.tools import (
    _create_file,
    _edit_file,
    _list_directory,
    _read_file,
    _search_in_files,
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
