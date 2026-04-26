import pytest

import farcode.sessions as sessions_mod
from farcode.sessions import (
    delete_session,
    load_last_session,
    load_sessions,
    new_session,
    save_session,
    search_sessions,
)


@pytest.fixture(autouse=True)
def isolated_sessions_dir(tmp_path, monkeypatch):
    """Redirect all session I/O to a temp directory for every test."""
    monkeypatch.setattr(sessions_mod, "SESSIONS_DIR", tmp_path / "sessions")


# ── new_session ───────────────────────────────────────────────────────────────

def test_new_session_defaults():
    s = new_session("qwen3.5:4b")
    assert s.model == "qwen3.5:4b"
    assert s.messages == []
    assert s.title == "(new)"
    assert s.id  # non-empty


def test_new_session_unique_ids():
    ids = {new_session("m").id for _ in range(10)}
    assert len(ids) == 10


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_and_load():
    s = new_session("qwen3.5:4b")
    s.messages = [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Hi!"},
    ]
    save_session(s)

    loaded = load_sessions()
    assert len(loaded) == 1
    assert loaded[0].id == s.id
    assert loaded[0].title == "Hello world"
    assert loaded[0].model == "qwen3.5:4b"


def test_title_derived_from_first_user_message():
    s = new_session("m")
    s.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Fix the auth bug"},
    ]
    save_session(s)
    assert load_sessions()[0].title == "Fix the auth bug"


def test_title_skips_file_injection_messages():
    s = new_session("m")
    s.messages = [
        {"role": "user", "content": "File `main.py`:\n```\ncode\n```"},
        {"role": "user", "content": "Explain this file"},
    ]
    save_session(s)
    assert load_sessions()[0].title == "Explain this file"


def test_title_truncated_at_60_chars():
    s = new_session("m")
    long_msg = "A" * 80
    s.messages = [{"role": "user", "content": long_msg}]
    save_session(s)
    assert load_sessions()[0].title.endswith("...")
    assert len(load_sessions()[0].title) == 63  # 60 + "..."


def test_turn_count():
    s = new_session("m")
    s.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    save_session(s)
    assert load_sessions()[0].turn_count == 2


def test_load_sessions_limit():
    for i in range(5):
        s = new_session("m")
        s.messages = [{"role": "user", "content": f"msg {i}"}]
        save_session(s)
    assert len(load_sessions(limit=3)) == 3


# ── load_last_session ─────────────────────────────────────────────────────────

def test_load_last_session_none_when_empty():
    assert load_last_session() is None


def test_load_last_session_returns_something():
    s = new_session("m")
    s.messages = [{"role": "user", "content": "hi"}]
    save_session(s)
    last = load_last_session()
    assert last is not None
    assert last.id == s.id


# ── delete_session ────────────────────────────────────────────────────────────

def test_delete_by_exact_id():
    s = new_session("m")
    s.messages = [{"role": "user", "content": "delete me"}]
    save_session(s)
    assert delete_session(s.id) is True
    assert load_sessions() == []


def test_delete_by_prefix():
    s = new_session("m")
    s.messages = [{"role": "user", "content": "prefix test"}]
    save_session(s)
    prefix = s.id[:8]
    assert delete_session(prefix) is True
    assert load_sessions() == []


def test_delete_not_found():
    assert delete_session("nonexistent_id") is False


def test_delete_ambiguous_prefix():
    for i in range(2):
        s = new_session("m")
        s.id = f"20240101_000000_aabb{i:02d}"
        s.messages = [{"role": "user", "content": f"msg {i}"}]
        save_session(s)
    assert delete_session("20240101") is False


# ── search_sessions ───────────────────────────────────────────────────────────

def test_search_sessions():
    for title in ["fix auth bug", "add dark mode", "auth refactor"]:
        s = new_session("m")
        s.messages = [{"role": "user", "content": title}]
        save_session(s)

    results = search_sessions("auth")
    titles = [r.title for r in results]
    assert "fix auth bug" in titles
    assert "auth refactor" in titles
    assert "add dark mode" not in titles


def test_search_sessions_case_insensitive():
    s = new_session("m")
    s.messages = [{"role": "user", "content": "Fix Auth Bug"}]
    save_session(s)
    assert len(search_sessions("auth")) == 1
