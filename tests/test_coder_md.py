import pytest

import farcode.coder_md as coder_md_mod
from farcode.coder_md import find_coder_md_files, load_coder_md


@pytest.fixture(autouse=True)
def no_global_coder_md(tmp_path, monkeypatch):
    """Default to a non-existent global path so tests don't depend on the user's home."""
    monkeypatch.setattr(coder_md_mod, "GLOBAL_PATH", tmp_path / "no-such-global.md")


# ── discovery ────────────────────────────────────────────────────────────────

def test_find_in_cwd(tmp_path):
    (tmp_path / ".git").mkdir()
    rules = tmp_path / "CODER.md"
    rules.write_text("be terse", encoding="utf-8")
    assert find_coder_md_files(tmp_path) == [rules]


def test_find_walks_up_to_git_root(tmp_path):
    (tmp_path / ".git").mkdir()
    root_rules = tmp_path / "CODER.md"
    root_rules.write_text("project rules", encoding="utf-8")
    sub = tmp_path / "pkg" / "deep"
    sub.mkdir(parents=True)
    sub_rules = sub / "CODER.md"
    sub_rules.write_text("subdir rules", encoding="utf-8")

    found = find_coder_md_files(sub)
    # Outermost (.git root) first, deepest (cwd) last so cwd has precedence.
    assert found == [root_rules, sub_rules]


def test_find_stops_at_git_boundary(tmp_path):
    (tmp_path / ".git").mkdir()
    inside = tmp_path / "a" / "b"
    inside.mkdir(parents=True)
    (tmp_path / "CODER.md").write_text("inside", encoding="utf-8")

    # A CODER.md *above* the git boundary must NOT be picked up.
    above = tmp_path.parent / "CODER.md"
    if not above.exists():  # avoid clobbering anything real
        above.write_text("outside", encoding="utf-8")
        try:
            found = find_coder_md_files(inside)
            assert all(p != above for p in found)
        finally:
            above.unlink()


def test_find_returns_empty_when_none(tmp_path):
    (tmp_path / ".git").mkdir()
    assert find_coder_md_files(tmp_path) == []


def test_global_file_is_included(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    project_rules = tmp_path / "CODER.md"
    project_rules.write_text("project", encoding="utf-8")

    global_path = tmp_path / "home" / ".coder.md"
    global_path.parent.mkdir()
    global_path.write_text("user-wide", encoding="utf-8")
    monkeypatch.setattr(coder_md_mod, "GLOBAL_PATH", global_path)

    found = find_coder_md_files(tmp_path)
    # Global comes first, project last (project overrides on append order).
    assert found == [global_path, project_rules]


# ── load_coder_md ────────────────────────────────────────────────────────────

def test_load_coder_md_empty(tmp_path):
    (tmp_path / ".git").mkdir()
    assert load_coder_md(tmp_path) == ""


def test_load_coder_md_includes_content_and_path(tmp_path):
    (tmp_path / ".git").mkdir()
    rules = tmp_path / "CODER.md"
    rules.write_text("use snake_case for everything", encoding="utf-8")

    out = load_coder_md(tmp_path)
    assert "use snake_case for everything" in out
    assert "## Project Rules" in out
    assert str(rules) in out


def test_load_coder_md_skips_empty_file(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "CODER.md").write_text("   \n  \n", encoding="utf-8")
    assert load_coder_md(tmp_path) == ""


def test_load_coder_md_truncates_oversized(tmp_path):
    (tmp_path / ".git").mkdir()
    big = "x" * (coder_md_mod._MAX_BYTES + 5000)
    (tmp_path / "CODER.md").write_text(big, encoding="utf-8")

    out = load_coder_md(tmp_path)
    assert "truncated" in out
    # Output must not contain the full payload
    assert len(out) < len(big) + 500


# ── build_system_messages integration ────────────────────────────────────────

def test_build_system_messages_includes_rules(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "CODER.md").write_text("PREFER FUNCTIONAL STYLE", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    from farcode.client import build_system_messages

    msgs = build_system_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"
    assert "PREFER FUNCTIONAL STYLE" in msgs[0]["content"]


def test_build_system_messages_no_rules(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)

    from farcode.client import build_system_messages

    msgs = build_system_messages()
    # No CODER.md => no "## Project Rules" section.
    assert "## Project Rules" not in msgs[0]["content"]
