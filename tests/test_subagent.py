"""Tests for the read-only exploration subagent."""

import pytest

from farcode import subagent
from farcode.client import _SyntheticResponse, _SyntheticToolCall


def _resp(text: str = "", *, tool_calls=None) -> _SyntheticResponse:
    return _SyntheticResponse(
        content=text,
        tool_calls=tool_calls or [],
        eval_count=10,
        prompt_eval_count=20,
    )


def _tc(name: str, **arguments) -> _SyntheticToolCall:
    return _SyntheticToolCall(name, arguments)


@pytest.fixture
def fake_subagent_ollama(monkeypatch):
    """Install a scripted call_nonstream on subagent.call_nonstream."""

    class Fake:
        def __init__(self, responses):
            self.responses = list(responses)
            self.calls: list[dict] = []

        def __call__(self, messages, model, tools=None, num_ctx=0, num_predict=0):
            self.calls.append({
                "messages": list(messages),
                "model": model,
                "tools": tools,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            })
            if not self.responses:
                raise AssertionError(
                    "Subagent FakeOllama exhausted; script more responses."
                )
            return self.responses.pop(0)

    def _install(responses):
        f = Fake(responses)
        monkeypatch.setattr(subagent, "call_nonstream", f)
        return f

    return _install


# ── tool subset ──────────────────────────────────────────────────────────────

def test_subagent_tools_are_only_read_only():
    schemas = subagent.get_subagent_tools()
    names = {s["function"]["name"] for s in schemas}
    assert names == subagent.READ_ONLY_TOOL_NAMES
    # In particular, dangerous tools are not exposed:
    assert "write_file" not in names
    assert "edit_file" not in names
    assert "run_bash" not in names
    assert "save_memory" not in names
    assert "explore_subagent" not in names  # no recursion


def test_get_subagent_model_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("FARCODE_SUBAGENT_MODEL", "tiny:0.5b")
    assert subagent.get_subagent_model("parent:7b") == "tiny:0.5b"


def test_get_subagent_model_inherits_when_env_unset(monkeypatch):
    monkeypatch.delenv("FARCODE_SUBAGENT_MODEL", raising=False)
    assert subagent.get_subagent_model("parent:7b") == "parent:7b"


def test_get_subagent_model_inherits_when_env_blank(monkeypatch):
    monkeypatch.setenv("FARCODE_SUBAGENT_MODEL", "   ")
    assert subagent.get_subagent_model("parent:7b") == "parent:7b"


# ── parent model binding ─────────────────────────────────────────────────────

def test_bind_parent_model_round_trip():
    subagent.bind_parent_model("test-model:1b")
    assert subagent.get_parent_model() == "test-model:1b"
    subagent.bind_parent_model("")  # cleanup


# ── basic subagent flow ──────────────────────────────────────────────────────

def test_subagent_returns_text_when_no_tool_calls(fake_subagent_ollama):
    fake = fake_subagent_ollama([_resp("auth lives in src/auth.py")])
    text, calls = subagent.run_subagent(
        question="where is auth?",
        focus_area=None,
        parent_model="parent:7b",
    )
    assert "auth lives" in text
    assert calls == 0
    assert fake.calls[0]["model"] == "parent:7b"


def test_subagent_passes_only_readonly_tools(fake_subagent_ollama):
    fake = fake_subagent_ollama([_resp("ok")])
    subagent.run_subagent("q", None, "parent:7b")
    sent_tools = fake.calls[0]["tools"]
    sent_names = {s["function"]["name"] for s in sent_tools}
    assert sent_names == subagent.READ_ONLY_TOOL_NAMES


def test_subagent_loops_through_tool_calls(fake_subagent_ollama, monkeypatch, tmp_path):
    target = tmp_path / "x.py"
    target.write_text("hello", encoding="utf-8")

    fake = fake_subagent_ollama([
        _resp("reading file", tool_calls=[_tc("read_file", path=str(target))]),
        _resp("done — file says 'hello'"),
    ])
    text, calls = subagent.run_subagent(
        question="what does x.py contain?",
        focus_area=None,
        parent_model="parent:7b",
    )
    assert "hello" in text or "done" in text
    assert calls == 1
    # Second call's messages must include the tool result
    second = fake.calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in second)


def test_subagent_includes_focus_area_in_user_message(fake_subagent_ollama):
    fake = fake_subagent_ollama([_resp("ok")])
    subagent.run_subagent("trace flow", focus_area="src/auth", parent_model="p")
    user_msg = next(m for m in fake.calls[0]["messages"] if m["role"] == "user")
    assert "src/auth" in user_msg["content"]
    assert "trace flow" in user_msg["content"]


# ── safety: forbidden tools refused ─────────────────────────────────────────

def test_subagent_refuses_forbidden_tool_call(fake_subagent_ollama):
    """If the model somehow emits a write tool, the subagent must refuse."""
    fake = fake_subagent_ollama([
        _resp("trying to write", tool_calls=[_tc("write_file", path="x", content="bad")]),
        _resp("acknowledged refusal"),
    ])
    text, calls = subagent.run_subagent("test", None, "parent")
    # The second call's messages must contain a refusal in the tool result
    second = fake.calls[1]["messages"]
    tool_msgs = [m for m in second if m.get("role") == "tool"]
    assert tool_msgs
    assert "not allowed" in tool_msgs[0]["content"]
    # And no actual file was written (write_file would be called via execute_tool;
    # we route the refusal before that, so this is a structural check).


# ── depth cap ───────────────────────────────────────────────────────────────

def test_subagent_refuses_recursion(fake_subagent_ollama, monkeypatch):
    monkeypatch.setattr(subagent, "_set_depth", lambda v: setattr(subagent._depth_state, "depth", v))
    subagent._set_depth(1)  # simulate already inside a subagent
    try:
        with pytest.raises(RuntimeError):
            subagent.run_subagent("nested", None, "parent")
    finally:
        subagent._set_depth(0)


def test_subagent_resets_depth_after_run(fake_subagent_ollama):
    fake_subagent_ollama([_resp("done")])
    subagent.run_subagent("q", None, "parent")
    assert subagent._depth() == 0


def test_subagent_resets_depth_after_exception(fake_subagent_ollama, monkeypatch):
    """Even if call_nonstream raises, depth must be unwound (we return error string)."""
    def boom(*a, **kw):
        raise ValueError("simulated")
    monkeypatch.setattr(subagent, "call_nonstream", boom)
    text, calls = subagent.run_subagent("q", None, "parent")
    assert "Subagent error" in text
    assert subagent._depth() == 0


# ── iter cap ────────────────────────────────────────────────────────────────

def test_subagent_caps_tool_calls(fake_subagent_ollama, tmp_path):
    """When the model keeps emitting tool calls, the subagent stops at the cap."""
    target = tmp_path / "f.py"
    target.write_text("x", encoding="utf-8")

    # +2: initial cap calls + 1 attempted + 1 final response (capped path)
    responses = []
    for _ in range(subagent.SUBAGENT_MAX_ITERS + 2):
        responses.append(
            _resp("looking", tool_calls=[_tc("read_file", path=str(target))])
        )
    # Append a sentinel (should not be consumed because cap returns early)
    responses.append(_resp("never reached"))

    fake = fake_subagent_ollama(responses)
    text, calls = subagent.run_subagent("q", None, "parent")
    assert "cap" in text.lower()
    assert calls == subagent.SUBAGENT_MAX_ITERS


# ── tool integration ────────────────────────────────────────────────────────

def test_explore_subagent_tool_handler_invokes_subagent(monkeypatch):
    from farcode.tools import _explore_subagent

    captured = {}

    def fake_run(question, focus_area, parent_model, num_ctx=0, num_predict=0):
        captured["q"] = question
        captured["focus"] = focus_area
        captured["parent"] = parent_model
        return "fake summary", 3

    monkeypatch.setattr(subagent, "run_subagent", fake_run)
    subagent.bind_parent_model("parent:7b")
    out = _explore_subagent("trace auth", focus_area="src/auth")
    assert "fake summary" in out
    assert "3 tool call" in out
    assert captured["q"] == "trace auth"
    assert captured["focus"] == "src/auth"
    assert captured["parent"] == "parent:7b"


def test_explore_subagent_handler_errors_when_parent_unbound(monkeypatch):
    from farcode.tools import _explore_subagent

    subagent.bind_parent_model("")
    out = _explore_subagent("q")
    assert "Tool error" in out
    assert "parent model" in out


def test_explore_subagent_schema_listed_in_main_tools():
    from farcode.tools import TOOL_SCHEMAS

    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert "explore_subagent" in names


def test_explore_subagent_not_listed_in_subagent_tools():
    """The subagent must never see its own tool — it cannot recurse."""
    schemas = subagent.get_subagent_tools()
    names = {s["function"]["name"] for s in schemas}
    assert "explore_subagent" not in names


# ── UI hooks: subagent start / done lines ────────────────────────────────────

def test_explore_subagent_handler_prints_start_and_done(monkeypatch):
    from farcode.tools import _explore_subagent
    from farcode import ui

    captured: list[str] = []
    monkeypatch.setattr(ui, "print_info", lambda msg: captured.append(msg))
    monkeypatch.setattr(subagent, "run_subagent", lambda *a, **kw: ("the answer", 4))
    subagent.bind_parent_model("parent:7b")

    out = _explore_subagent("trace the auth flow", focus_area="src/auth")
    assert "the answer" in out

    starts = [m for m in captured if "exploring" in m]
    dones = [m for m in captured if "done" in m]
    assert len(starts) == 1
    assert "trace the auth flow" in starts[0]
    assert "src/auth" in starts[0]  # focus appended in start line
    assert len(dones) == 1
    assert "4 tool call" in dones[0]


def test_explore_subagent_handler_truncates_long_question_in_start_line(monkeypatch):
    from farcode.tools import _explore_subagent
    from farcode import ui

    captured: list[str] = []
    monkeypatch.setattr(ui, "print_info", lambda msg: captured.append(msg))
    monkeypatch.setattr(subagent, "run_subagent", lambda *a, **kw: ("ok", 0))
    subagent.bind_parent_model("p")

    long_q = "a" * 200
    _explore_subagent(long_q)
    starts = [m for m in captured if "exploring" in m]
    assert "..." in starts[0]
    assert len(starts[0]) < 200  # not the full 200-char question


# ── /explore slash command helper ───────────────────────────────────────────

def test_run_explore_invokes_subagent_with_current_model(monkeypatch):
    from farcode import chat

    captured = {}
    monkeypatch.setattr(chat._subagent, "run_subagent",
                        lambda q, focus_area, parent_model, num_ctx=0, num_predict=0:
                            (captured.setdefault("call", {"q": q, "model": parent_model}),
                             ("explored result", 5))[1])
    monkeypatch.setattr(chat, "print_info", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "print_error", lambda *a, **kw: None)
    monkeypatch.setattr(chat.console, "print", lambda *a, **kw: None)

    chat._run_explore("how does X work", model="big-model:7b", num_ctx=8192, num_predict=512)
    assert captured["call"]["q"] == "how does X work"
    assert captured["call"]["model"] == "big-model:7b"


def test_run_explore_empty_question_prints_usage(monkeypatch):
    from farcode import chat

    infos: list[str] = []
    monkeypatch.setattr(chat, "print_info", lambda msg: infos.append(msg))
    monkeypatch.setattr(chat, "print_error", lambda *a, **kw: None)

    called = {"n": 0}
    monkeypatch.setattr(chat._subagent, "run_subagent",
                        lambda *a, **kw: (called.update(n=called["n"] + 1), ("x", 0))[1])

    chat._run_explore("   ", model="m", num_ctx=8192, num_predict=512)
    assert any("Usage" in m for m in infos)
    assert called["n"] == 0  # subagent never invoked


def test_explore_command_is_in_registry():
    from farcode.commands import SLASH_COMMANDS
    names = [n.split()[0] for n, _ in SLASH_COMMANDS]
    assert "/explore" in names
