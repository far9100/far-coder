"""Tests for the in-session task tracking module and tools."""

import pytest

from farcode import tasks


@pytest.fixture(autouse=True)
def fresh_tasks():
    """Bind a fresh empty list before every test so module state never leaks."""
    tasks.bind([])
    yield
    tasks.bind([])


# ── tasks module ──────────────────────────────────────────────────────────────

def test_create_appends_pending_task():
    t = tasks.create("read the README")
    assert t["status"] == "pending"
    assert t["content"] == "read the README"
    assert t["id"]
    assert tasks.list_all() == [t]


def test_create_strips_whitespace():
    t = tasks.create("  trimmed  ")
    assert t["content"] == "trimmed"


def test_create_rejects_empty():
    with pytest.raises(ValueError):
        tasks.create("")
    with pytest.raises(ValueError):
        tasks.create("   ")


def test_update_changes_status():
    t = tasks.create("step")
    updated = tasks.update(t["id"], "in_progress")
    assert updated["status"] == "in_progress"
    assert tasks.list_all()[0]["status"] == "in_progress"


def test_update_rejects_invalid_status():
    t = tasks.create("step")
    with pytest.raises(ValueError):
        tasks.update(t["id"], "done")  # not in VALID_STATUSES


def test_update_rejects_unknown_id():
    with pytest.raises(KeyError):
        tasks.update("nonexistent", "completed")


def test_list_all_returns_copy_not_reference():
    tasks.create("a")
    snapshot = tasks.list_all()
    snapshot.clear()
    assert tasks.list_all()  # original list untouched


def test_bind_redirects_state():
    tasks.create("a")
    other: list[dict] = []
    tasks.bind(other)
    assert tasks.list_all() == []  # bound to empty list
    tasks.create("in other")
    assert other[0]["content"] == "in other"


def test_move_into_transfers_and_rebinds():
    tasks.create("a")
    tasks.create("b")
    target: list[dict] = []
    tasks.move_into(target)
    assert len(target) == 2
    # subsequent create lands in target
    tasks.create("c")
    assert len(target) == 3


def test_unique_ids_across_creates():
    ids = {tasks.create(f"step {i}")["id"] for i in range(20)}
    assert len(ids) == 20


def test_ids_are_sequential_t_prefixed():
    a = tasks.create("a")
    b = tasks.create("b")
    c = tasks.create("c")
    assert a["id"] == "t1"
    assert b["id"] == "t2"
    assert c["id"] == "t3"


def test_new_id_skips_legacy_hex_ids():
    """When a session has hex ids from older versions, new ids still start
    from t1 (or pick up where the latest tN left off if any tN exists)."""
    legacy_target: list[dict] = [
        {"id": "a3f2c1", "content": "old", "status": "pending", "created_at": ""},
        {"id": "b8e102", "content": "older", "status": "pending", "created_at": ""},
    ]
    tasks.bind(legacy_target)
    new = tasks.create("fresh")
    assert new["id"] == "t1"


def test_new_id_continues_after_existing_t_ids():
    """If the bound list already has t1, t5 (say from a partial restore),
    the next id should be t6 — strictly increasing, never colliding."""
    target: list[dict] = [
        {"id": "t1", "content": "x", "status": "pending", "created_at": ""},
        {"id": "t5", "content": "y", "status": "pending", "created_at": ""},
    ]
    tasks.bind(target)
    new = tasks.create("z")
    assert new["id"] == "t6"


# ── tool handlers ─────────────────────────────────────────────────────────────

def test_task_create_handler_returns_id_and_content():
    from farcode.tools import _task_create

    out = _task_create("plan the work")
    assert "Created task" in out
    assert "plan the work" in out
    assert tasks.list_all()[0]["content"] == "plan the work"


def test_task_update_handler_changes_status():
    from farcode.tools import _task_create, _task_update

    _task_create("step")
    tid = tasks.list_all()[0]["id"]
    out = _task_update(tid, "in_progress")
    assert "in_progress" in out
    assert tasks.list_all()[0]["status"] == "in_progress"


def test_task_update_handler_rejects_bad_status():
    from farcode.tools import execute_tool

    tasks.create("step")
    tid = tasks.list_all()[0]["id"]
    out = execute_tool("task_update", {"id": tid, "status": "done"})
    assert "Tool error" in out


def test_task_list_handler_renders_status_icons():
    from farcode.tools import _task_list

    a = tasks.create("first")
    b = tasks.create("second")
    tasks.update(a["id"], "completed")
    tasks.update(b["id"], "in_progress")

    out = _task_list()
    assert "[x]" in out  # completed icon
    assert "[>]" in out  # in_progress icon
    assert "first" in out
    assert "second" in out


def test_task_list_handler_empty():
    from farcode.tools import _task_list

    assert _task_list() == "No tasks."


def test_execute_tool_dispatches_task_tools():
    from farcode.tools import execute_tool

    out = execute_tool("task_create", {"content": "do thing"})
    assert "Created task" in out
    tid = tasks.list_all()[0]["id"]
    out2 = execute_tool("task_update", {"id": tid, "status": "completed"})
    assert "completed" in out2
    out3 = execute_tool("task_list", {})
    assert "do thing" in out3


# ── session round-trip ───────────────────────────────────────────────────────

def test_session_round_trip_preserves_tasks(tmp_path, monkeypatch):
    from farcode import sessions

    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path)

    s = sessions.new_session("test-model")
    s.tasks.append({
        "id": "t01",
        "content": "step one",
        "status": "in_progress",
        "created_at": "2026-04-29T00:00:00",
    })
    s.messages.append({"role": "user", "content": "hi"})
    sessions.save_session(s)

    loaded = sessions.load_sessions()
    assert loaded
    assert loaded[0].tasks
    assert loaded[0].tasks[0]["status"] == "in_progress"
    assert loaded[0].tasks[0]["content"] == "step one"


def test_session_load_tolerates_missing_tasks_field(tmp_path, monkeypatch):
    """Older session files written before this feature must still load."""
    import json
    from farcode import sessions

    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({
            "id": "legacy",
            "title": "old",
            "model": "test-model",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "messages": [{"role": "user", "content": "hi"}],
        }),
        encoding="utf-8",
    )

    loaded = sessions.load_sessions()
    assert loaded
    assert loaded[0].tasks == []


# ── auto-render after task tool calls ────────────────────────────────────────

def test_auto_render_fires_on_task_update(monkeypatch):
    """When _run_tools_parallel handles a task_update call, it should render
    the live task list once afterwards."""
    from farcode import chat
    from farcode.client import _SyntheticToolCall

    rendered: list[list[dict]] = []
    monkeypatch.setattr(chat, "print_task_list", lambda items: rendered.append(list(items)))
    monkeypatch.setattr(chat, "print_tool_call", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "print_info", lambda *a, **kw: None)

    tasks.create("step")
    tid = tasks.list_all()[0]["id"]
    tc = _SyntheticToolCall("task_update", {"id": tid, "status": "completed"})
    chat._run_tools_parallel([tc])

    assert len(rendered) == 1
    assert rendered[0][0]["status"] == "completed"


def test_auto_render_fires_on_task_create(monkeypatch):
    from farcode import chat
    from farcode.client import _SyntheticToolCall

    rendered: list[list[dict]] = []
    monkeypatch.setattr(chat, "print_task_list", lambda items: rendered.append(list(items)))
    monkeypatch.setattr(chat, "print_tool_call", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "print_info", lambda *a, **kw: None)

    tc = _SyntheticToolCall("task_create", {"content": "step one"})
    chat._run_tools_parallel([tc])

    assert len(rendered) == 1
    assert rendered[0][0]["content"] == "step one"


def test_auto_render_does_not_fire_on_unrelated_tool(monkeypatch):
    from farcode import chat
    from farcode.client import _SyntheticToolCall

    rendered: list[list[dict]] = []
    monkeypatch.setattr(chat, "print_task_list", lambda items: rendered.append(list(items)))
    monkeypatch.setattr(chat, "print_tool_call", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "print_info", lambda *a, **kw: None)

    tc = _SyntheticToolCall("task_list", {})  # task_list is read-only, not mutating
    chat._run_tools_parallel([tc])

    assert rendered == []


def test_auto_render_fires_once_per_batch(monkeypatch):
    """Multiple task_create calls in one batch should render only once."""
    from farcode import chat
    from farcode.client import _SyntheticToolCall

    rendered: list[list[dict]] = []
    monkeypatch.setattr(chat, "print_task_list", lambda items: rendered.append(list(items)))
    monkeypatch.setattr(chat, "print_tool_call", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "print_info", lambda *a, **kw: None)

    calls = [
        _SyntheticToolCall("task_create", {"content": "a"}),
        _SyntheticToolCall("task_create", {"content": "b"}),
        _SyntheticToolCall("task_create", {"content": "c"}),
    ]
    chat._run_tools_parallel(calls)

    assert len(rendered) == 1
    assert len(rendered[0]) == 3  # all three tasks visible in the single render


def test_session_load_filters_malformed_tasks(tmp_path, monkeypatch):
    import json
    from farcode import sessions

    monkeypatch.setattr(sessions, "SESSIONS_DIR", tmp_path)
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps({
            "id": "s",
            "title": "t",
            "model": "m",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T00:00:00",
            "messages": [],
            "tasks": [
                {"id": "ok", "content": "good", "status": "completed", "created_at": ""},
                {"id": 123, "content": "bad-id-type", "status": "pending"},   # dropped
                "not a dict",                                                  # dropped
                {"id": "x", "content": "weird-status", "status": "weird"},    # status normalized
            ],
        }),
        encoding="utf-8",
    )

    loaded = sessions.load_sessions()
    assert loaded
    kept = loaded[0].tasks
    assert len(kept) == 2
    assert kept[0]["content"] == "good"
    assert kept[1]["status"] == "pending"  # normalized from "weird"
