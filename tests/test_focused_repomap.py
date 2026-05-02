"""Tests for build_focused_repo_map — targeted, seed-rooted repo map.

Verifies BFS expansion via imports, the bypass of MAX_FILES_FOR_INJECTION,
and that the cache key separates focused output from global output."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from farcode import repomap


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Each test gets a fresh on-disk cache so we never pollute the user's
    real ~/.farcode_repomap_cache.json."""
    monkeypatch.setattr(repomap, "CACHE_PATH", tmp_path / "cache.json")


def _mkfile(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _make_pkg(root: Path, files: dict[str, str]) -> None:
    """Lay out a project tree under ``root`` from {rel_path: content}."""
    for rel, content in files.items():
        _mkfile(root, rel, content)


# ── empty seeds ───────────────────────────────────────────────────────────────

def test_empty_seeds_returns_blank(tmp_path):
    _make_pkg(tmp_path, {"a.py": "def x(): pass\n"})
    assert repomap.build_focused_repo_map([], root=tmp_path) == ""


def test_only_whitespace_seeds_returns_blank(tmp_path):
    _make_pkg(tmp_path, {"a.py": "def x(): pass\n"})
    assert repomap.build_focused_repo_map([" ", ""], root=tmp_path) == ""


# ── seed-only (depth=0) ───────────────────────────────────────────────────────

def test_seed_alone_at_depth_zero(tmp_path):
    _make_pkg(tmp_path, {
        "src/lib.py": "class Collector:\n    def delete(self): pass\n",
        "src/other.py": "def unrelated(): pass\n",
    })
    out = repomap.build_focused_repo_map(["src/lib.py"], root=tmp_path, depth=0)
    assert "src/lib.py" in out
    assert "Collector" in out
    # depth=0 means no import expansion
    assert "src/other.py" not in out


# ── BFS expansion via imports ─────────────────────────────────────────────────

def test_bfs_follows_from_imports(tmp_path):
    """Seed imports `helper`; helper file should be pulled in."""
    _make_pkg(tmp_path, {
        "src/main.py": "from src.helper import do_thing\n\ndef run(): do_thing()\n",
        "src/helper.py": "def do_thing(): pass\n",
    })
    out = repomap.build_focused_repo_map(["src/main.py"], root=tmp_path, depth=2)
    assert "src/main.py" in out
    assert "src/helper.py" in out
    assert "do_thing" in out


def test_bfs_follows_bare_imports(tmp_path):
    _make_pkg(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "import pkg.b\n\ndef foo(): pass\n",
        "pkg/b.py": "def bar(): pass\n",
    })
    out = repomap.build_focused_repo_map(["pkg/a.py"], root=tmp_path, depth=2)
    assert "pkg/a.py" in out
    assert "pkg/b.py" in out


def test_bfs_depth_bounds_traversal(tmp_path):
    """A→B→C; depth=1 should pull A and B but stop before C.

    Files with no top-level defs are filtered from the rendered output, so
    we give every node at least one def and assert on visibility."""
    _make_pkg(tmp_path, {
        "a.py": "from b import x\n\ndef a_fn(): pass\n",
        "b.py": "from c import y\n\ndef x(): pass\n",
        "c.py": "def y(): pass\n",
    })
    out = repomap.build_focused_repo_map(["a.py"], root=tmp_path, depth=1)
    assert "a.py" in out
    assert "b.py" in out
    assert "c.py" not in out


def test_bypasses_max_files_for_injection_cutoff(tmp_path, monkeypatch):
    """Global build_repo_map disables injection above 100 files; focused map
    must still produce output. We force a tiny cutoff to make this concrete."""
    monkeypatch.setattr(repomap, "MAX_FILES_FOR_INJECTION", 2)
    files: dict[str, str] = {f"f{i}.py": f"def fn{i}(): pass\n" for i in range(50)}
    files["seed.py"] = "from f0 import fn0\n\ndef seed(): pass\n"
    _make_pkg(tmp_path, files)

    # Global map self-disables (>2 files)
    assert repomap.build_repo_map(root=tmp_path, use_cache=False) == ""

    # Focused map still produces output
    out = repomap.build_focused_repo_map(["seed.py"], root=tmp_path, depth=2)
    assert "seed.py" in out


# ── budget ────────────────────────────────────────────────────────────────────

def test_respects_max_chars(tmp_path):
    files: dict[str, str] = {}
    for i in range(20):
        files[f"f{i}.py"] = f"def function_with_long_name_{i}(): pass\n" * 5
    files["seed.py"] = "\n".join(f"from f{i} import function_with_long_name_{i}" for i in range(20))
    _make_pkg(tmp_path, files)

    out = repomap.build_focused_repo_map(["seed.py"], root=tmp_path, depth=2, max_chars=200)
    assert len(out) <= 250  # small slack for the truncation marker


# ── cache isolation ───────────────────────────────────────────────────────────

def test_cache_key_separates_seed_lists(tmp_path):
    """Two calls with different seeds must not share cache entries — otherwise
    instance N would see instance N-1's repo map."""
    _make_pkg(tmp_path, {
        "a.py": "def fn_a(): pass\n",
        "b.py": "def fn_b(): pass\n",
    })
    out_a = repomap.build_focused_repo_map(["a.py"], root=tmp_path)
    out_b = repomap.build_focused_repo_map(["b.py"], root=tmp_path)
    assert "fn_a" in out_a
    assert "fn_b" in out_b
    assert "fn_b" not in out_a
    assert "fn_a" not in out_b


def test_cache_hit_returns_same_output(tmp_path):
    _make_pkg(tmp_path, {"a.py": "def fn_a(): pass\n"})
    out1 = repomap.build_focused_repo_map(["a.py"], root=tmp_path)
    out2 = repomap.build_focused_repo_map(["a.py"], root=tmp_path)
    assert out1 == out2


# ── seed not in workspace ─────────────────────────────────────────────────────

def test_unknown_seed_filtered_out(tmp_path):
    _make_pkg(tmp_path, {"a.py": "def fn_a(): pass\n"})
    # Real seed + bogus seed; bogus is silently dropped
    out = repomap.build_focused_repo_map(["a.py", "does/not/exist.py"], root=tmp_path)
    assert "fn_a" in out
    assert "does/not/exist.py" not in out


def test_all_seeds_unknown_returns_blank(tmp_path):
    _make_pkg(tmp_path, {"a.py": "def fn_a(): pass\n"})
    assert repomap.build_focused_repo_map(["does/not/exist.py"], root=tmp_path) == ""


# ── client.py integration: env var triggers focused mode ─────────────────────

def test_env_var_triggers_focused_in_build_system_messages(tmp_path, monkeypatch):
    """build_system_messages should call build_focused_repo_map when
    FARCODE_REPOMAP_SEEDS is set in the environment."""
    _make_pkg(tmp_path, {
        "lib.py": "def my_special_function(): pass\n",
        "other.py": "def unrelated(): pass\n",
    })
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FARCODE_REPOMAP_SEEDS", "lib.py")
    monkeypatch.setenv("FARCODE_DISABLE_MEMORY", "1")
    monkeypatch.setenv("FARCODE_DISABLE_CODER_MD", "1")

    from farcode.client import build_system_messages
    msgs = build_system_messages(num_ctx=8192)
    content = msgs[0]["content"]
    assert "my_special_function" in content
    # Focused header should appear, not the global one
    assert "focused on suspected files" in content
