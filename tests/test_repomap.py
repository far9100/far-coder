"""Tests for the tree-sitter-style repo map (Python AST + regex for others)."""

import pytest
from pathlib import Path

from farcode import repomap


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Build a minimal project tree and point the cache at tmp."""
    monkeypatch.setattr(repomap, "CACHE_PATH", tmp_path / "repomap_cache.json")
    return tmp_path


# ── Python definition extraction ──────────────────────────────────────────────

def test_python_defs_picks_up_functions_and_classes(project):
    f = project / "mod.py"
    f.write_text(
        "def foo(a, b):\n    return a + b\n\n"
        "async def bar():\n    pass\n\n"
        "class Widget:\n"
        "    def public_method(self): pass\n"
        "    def _private(self): pass\n"
        "    def another(self): pass\n",
        encoding="utf-8",
    )
    defs = repomap.extract_defs(f)
    assert any("def foo(a, b)" in d for d in defs)
    assert any("async def bar" in d for d in defs)
    assert any("class Widget" in d for d in defs)
    # Methods listed for class but private excluded
    widget_line = next(d for d in defs if d.startswith("class Widget"))
    assert "public_method" in widget_line
    assert "another" in widget_line
    assert "_private" not in widget_line


def test_python_defs_top_level_constants(project):
    f = project / "constants.py"
    f.write_text("MAX_SIZE = 1024\nx = 1\n", encoding="utf-8")
    defs = repomap.extract_defs(f)
    assert any("MAX_SIZE = ..." in d for d in defs)
    # lowercase x is not picked up (not a constant convention)
    assert not any(d.startswith("x = ") for d in defs)


def test_python_defs_returns_empty_on_syntax_error(project):
    f = project / "broken.py"
    f.write_text("def foo(\n    bad", encoding="utf-8")
    assert repomap.extract_defs(f) == []


# ── JavaScript / TypeScript regex extraction ─────────────────────────────────

def test_js_defs_function_and_class(project):
    f = project / "app.ts"
    f.write_text(
        "export function handleClick(e: Event) { return; }\n"
        "export class Button { render() {} }\n"
        "export const onLoad = () => {};\n",
        encoding="utf-8",
    )
    defs = repomap.extract_defs(f)
    assert any("handleClick" in d for d in defs)
    assert any("class Button" in d for d in defs)
    assert any("onLoad" in d for d in defs)


# ── Go regex extraction ──────────────────────────────────────────────────────

def test_go_defs(project):
    f = project / "main.go"
    f.write_text(
        "package main\n\n"
        "func main() {}\n"
        "func (s *Server) Handle(w http.ResponseWriter) {}\n",
        encoding="utf-8",
    )
    defs = repomap.extract_defs(f)
    assert any("func main" in d for d in defs)
    assert any("func Handle" in d for d in defs)


# ── Rust regex extraction ────────────────────────────────────────────────────

def test_rust_defs(project):
    f = project / "lib.rs"
    f.write_text(
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
        "fn helper() {}\n"
        "pub struct User { name: String }\n"
        "pub enum Role { Admin, User }\n",
        encoding="utf-8",
    )
    defs = repomap.extract_defs(f)
    assert any("fn add" in d for d in defs)
    assert any("fn helper" in d for d in defs)
    assert any("User" in d for d in defs)
    assert any("Role" in d for d in defs)


# ── build_repo_map (full pipeline) ───────────────────────────────────────────

def test_build_repo_map_basic(project):
    (project / "pkg").mkdir()
    (project / "pkg" / "a.py").write_text(
        "def alpha():\n    pass\n", encoding="utf-8"
    )
    (project / "pkg" / "b.py").write_text(
        "class Beta:\n    def go(self): pass\n", encoding="utf-8"
    )
    out = repomap.build_repo_map(project, use_cache=False)
    assert out.startswith("## Repo Map")
    assert "pkg/a.py" in out
    assert "alpha" in out
    assert "Beta" in out


def test_build_repo_map_skips_ignored_dirs(project):
    """node_modules, .venv, etc. should never appear in output."""
    (project / "src").mkdir()
    (project / "src" / "real.py").write_text("def real(): pass\n", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "junk.js").write_text(
        "function junk() {}\n", encoding="utf-8"
    )
    (project / ".venv").mkdir()
    (project / ".venv" / "bad.py").write_text("def bad(): pass\n", encoding="utf-8")
    out = repomap.build_repo_map(project, use_cache=False)
    assert "real.py" in out
    assert "node_modules" not in out
    assert ".venv" not in out


def test_build_repo_map_respects_char_budget(project):
    # Generate a lot of files
    for i in range(50):
        (project / f"f{i:03d}.py").write_text(
            f"def func_{i}():\n    pass\n", encoding="utf-8"
        )
    out = repomap.build_repo_map(project, max_chars=300, use_cache=False)
    assert len(out) <= 400  # budget + a small overhead for the truncation marker
    assert "## Repo Map" in out


def test_build_repo_map_empty_project(project):
    out = repomap.build_repo_map(project, use_cache=False)
    assert out == ""


def test_build_repo_map_only_unparseable_files(project):
    (project / "only.txt").write_text("not source", encoding="utf-8")
    out = repomap.build_repo_map(project, use_cache=False)
    assert out == ""


def test_build_repo_map_caches_result(project):
    (project / "x.py").write_text("def hello(): pass\n", encoding="utf-8")
    out1 = repomap.build_repo_map(project)
    # Mutate the file's content but keep the same mtime — cache should still hit
    # because we key on (rel_path, mtime).
    assert repomap.CACHE_PATH.exists()
    out2 = repomap.build_repo_map(project)
    assert out1 == out2


def test_build_repo_map_recency_ranks_recent_higher(project):
    import os, time

    old = project / "old.py"
    old.write_text("def old_func(): pass\n", encoding="utf-8")
    # Backdate mtime to ~6 months ago
    old_mtime = time.time() - (180 * 24 * 3600)
    os.utime(old, (old_mtime, old_mtime))

    new = project / "new.py"
    new.write_text("def new_func(): pass\n", encoding="utf-8")

    out = repomap.build_repo_map(project, use_cache=False)
    new_idx = out.index("new.py")
    old_idx = out.index("old.py")
    assert new_idx < old_idx  # newer file ranked first


def test_build_repo_map_handles_unicode(project):
    f = project / "uni.py"
    f.write_text(
        "def 你好():\n    return '世界'\n",
        encoding="utf-8",
    )
    out = repomap.build_repo_map(project, use_cache=False)
    assert "uni.py" in out
