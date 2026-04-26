"""Tests for codebase facts auto-extraction."""

import json
import sys
from pathlib import Path

import pytest

from farcode import facts


@pytest.fixture
def reset_memory(tmp_path, monkeypatch):
    """Point the memory store at a temp DB so fact caching doesn't persist."""
    if "farcode.memory" in sys.modules:
        del sys.modules["farcode.memory"]
    import farcode.memory as memory_mod

    memory_mod.MEMORY_DB = tmp_path / "facts-mem.db"
    memory_mod.LEGACY_JSONL_PATHS = []
    memory_mod._conn = None
    memory_mod._fts_available = None
    memory_mod._project_cache.clear()
    return memory_mod


# ── Language detection ────────────────────────────────────────────────────────

def test_detect_primary_language_python(tmp_path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "c.js").write_text("", encoding="utf-8")
    assert facts._detect_primary_language(tmp_path) == "Python"


def test_detect_primary_language_typescript(tmp_path):
    (tmp_path / "a.ts").write_text("", encoding="utf-8")
    (tmp_path / "b.tsx").write_text("", encoding="utf-8")
    assert facts._detect_primary_language(tmp_path) == "TypeScript"


def test_detect_primary_language_skips_node_modules(tmp_path):
    (tmp_path / "real.py").write_text("", encoding="utf-8")
    nm = tmp_path / "node_modules"
    nm.mkdir()
    for i in range(20):
        (nm / f"junk{i}.js").write_text("", encoding="utf-8")
    # Even though there are way more JS files in node_modules, Python wins
    assert facts._detect_primary_language(tmp_path) == "Python"


def test_detect_primary_language_empty(tmp_path):
    assert facts._detect_primary_language(tmp_path) == ""


# ── Package manager detection ─────────────────────────────────────────────────

def test_detect_package_manager_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    desc, marker = facts._detect_package_manager(tmp_path)
    assert "pip" in desc.lower() or "poetry" in desc.lower() or "uv" in desc.lower()
    assert marker == "pyproject.toml"


def test_detect_package_manager_node(tmp_path):
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    desc, marker = facts._detect_package_manager(tmp_path)
    assert marker == "package.json"


def test_detect_package_manager_none(tmp_path):
    desc, marker = facts._detect_package_manager(tmp_path)
    assert desc == "" and marker == ""


# ── Test runner detection ─────────────────────────────────────────────────────

def test_detect_test_runner_pytest_via_marker_file(tmp_path):
    (tmp_path / "pytest.ini").write_text("", encoding="utf-8")
    assert facts._detect_test_runner(tmp_path) == "pytest"


def test_detect_test_runner_pytest_via_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\naddopts = '-q'\n",
        encoding="utf-8",
    )
    assert facts._detect_test_runner(tmp_path) == "pytest"


def test_detect_test_runner_jest_via_package_json(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"scripts": {"test": "jest --coverage"}}),
        encoding="utf-8",
    )
    assert facts._detect_test_runner(tmp_path) == "jest"


def test_detect_test_runner_vitest(tmp_path):
    (tmp_path / "vitest.config.ts").write_text("", encoding="utf-8")
    assert facts._detect_test_runner(tmp_path) == "vitest"


# ── Entry points ──────────────────────────────────────────────────────────────

def test_detect_entry_points_pyproject_scripts(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n\n'
        '[project.scripts]\nfarcode = "farcode.cli:app"\nother = "x:y"\n',
        encoding="utf-8",
    )
    eps = facts._detect_entry_points(tmp_path)
    names = " ".join(eps)
    assert "farcode" in names
    assert "other" in names


def test_detect_entry_points_package_json_main(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps({"main": "index.js", "bin": {"my-cli": "./bin/cli.js"}}),
        encoding="utf-8",
    )
    eps = facts._detect_entry_points(tmp_path)
    assert any("index.js" in e for e in eps)
    assert any("my-cli" in e for e in eps)


def test_detect_entry_points_cargo(tmp_path):
    (tmp_path / "Cargo.toml").write_text(
        '[package]\nname = "thing"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    eps = facts._detect_entry_points(tmp_path)
    assert any("thing" in e for e in eps)


# ── Notable dirs ──────────────────────────────────────────────────────────────

def test_detect_notable_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "vendor").mkdir()  # not in candidate list
    dirs = facts._detect_notable_dirs(tmp_path)
    assert "src" in dirs
    assert "tests" in dirs
    assert "vendor" not in dirs


# ── Format ────────────────────────────────────────────────────────────────────

def test_facts_format_includes_all_present_fields():
    f = facts.CodebaseFacts(
        primary_language="Python",
        package_manager="uv / pip / poetry",
        test_runner="pytest",
        entry_points=["farcode (CLI)"],
        notable_dirs=["src", "tests"],
        marker_files=["pyproject.toml"],
    )
    out = f.format()
    assert "## Project Facts" in out
    assert "Python" in out
    assert "pytest" in out
    assert "farcode (CLI)" in out
    assert "src" in out


def test_facts_format_empty_returns_empty_string():
    f = facts.CodebaseFacts(
        primary_language="",
        package_manager="",
        test_runner="",
        entry_points=[],
        notable_dirs=[],
        marker_files=[],
    )
    assert f.format() == ""


# ── extract_facts (full pipeline) ─────────────────────────────────────────────

def test_extract_facts_python_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n\n'
        '[project.scripts]\ndemo = "demo.cli:app"\n\n'
        '[tool.pytest.ini_options]\naddopts = "-q"\n',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "demo.py").write_text("def main(): pass\n", encoding="utf-8")

    f = facts.extract_facts(tmp_path)
    assert f.primary_language == "Python"
    assert f.package_manager
    assert f.test_runner == "pytest"
    assert any("demo" in e for e in f.entry_points)


# ── get_or_build_facts caching via memory ─────────────────────────────────────

def test_get_or_build_facts_caches(reset_memory, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("def x(): pass\n", encoding="utf-8")

    out1 = facts.get_or_build_facts(tmp_path)
    out2 = facts.get_or_build_facts(tmp_path)
    assert out1 == out2
    assert out1.startswith("## Project Facts")

    # The cached entry should be in memory
    entries = reset_memory.load_recent(5)
    facts_entries = [e for e in entries if e["kind"] == "codebase_facts"]
    assert len(facts_entries) == 1


def test_get_or_build_facts_returns_empty_for_blank_project(reset_memory, tmp_path):
    out = facts.get_or_build_facts(tmp_path)
    assert out == ""


def test_get_or_build_facts_re_extracts_on_marker_change(reset_memory, tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("def x(): pass\n", encoding="utf-8")

    # First extraction
    facts.get_or_build_facts(tmp_path)
    initial = reset_memory.load_recent(5)
    initial_facts = [e for e in initial if e["kind"] == "codebase_facts"][0]

    # Force the marker signature to differ — simulate pyproject.toml mtime change
    import os, time
    new_mtime = time.time() + 100
    os.utime(tmp_path / "pyproject.toml", (new_mtime, new_mtime))

    facts.get_or_build_facts(tmp_path)
    after = reset_memory.load_recent(5)
    facts_entries = [e for e in after if e["kind"] == "codebase_facts"]
    # A second entry was written
    assert len(facts_entries) == 2
