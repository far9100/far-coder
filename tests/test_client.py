"""Tests for client.py — system prompt + Ollama call wrappers."""

import json

import pytest
import ollama

from farcode import client


# ── _system_prompt ────────────────────────────────────────────────────────────

def test_system_prompt_mentions_workflow_and_editing():
    p = client._system_prompt()
    assert "Workflow" in p
    assert "edit_file" in p
    assert "replace_lines" in p
    assert "write_file" in p


def test_system_prompt_includes_ctx_size():
    p = client._system_prompt(num_ctx=32 * 1024)
    assert "32K" in p


def test_system_prompt_does_not_duplicate_full_tool_list():
    """We removed the "## Available Tools" duplication. Don't reintroduce it."""
    p = client._system_prompt()
    # The schema descriptions in tools.py are the authority — system prompt
    # should not redundantly enumerate every tool.
    assert "## Available Tools" not in p
    assert "search_in_files" not in p
    assert "list_directory" not in p


# ── Synthetic response object plumbing ────────────────────────────────────────

def test_synthetic_response_has_message_shape():
    sr = client._SyntheticResponse(
        content="hi",
        tool_calls=[client._SyntheticToolCall("read_file", {"path": "x"})],
        eval_count=10,
        prompt_eval_count=20,
    )
    assert sr.message.content == "hi"
    assert sr.message.tool_calls[0].function.name == "read_file"
    assert sr.message.tool_calls[0].function.arguments == {"path": "x"}
    assert sr.eval_count == 10
    assert sr.prompt_eval_count == 20


# ── Grammar-constrained retry ─────────────────────────────────────────────────

@pytest.fixture
def fake_tool_schemas():
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "edit",
                "parameters": {"type": "object"},
            },
        },
    ]


def test_grammar_schema_enumerates_tool_names(fake_tool_schemas):
    schema = client._grammar_schema(fake_tool_schemas)
    enum = schema["properties"]["tool_call"]["properties"]["name"]["enum"]
    assert set(enum) == {"read_file", "edit_file"}


def test_grammar_retry_parses_tool_call(monkeypatch, fake_tool_schemas):
    """When the model returns valid JSON {tool_call: {...}}, we synthesize a response."""

    class _Msg:
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class _Resp:
        def __init__(self, content):
            self.message = _Msg(content)
            self.eval_count = 7
            self.prompt_eval_count = 13

    def fake_chat(**kwargs):
        # Verify the format constraint was passed through
        assert "format" in kwargs
        assert kwargs["format"]["type"] == "object"
        return _Resp(json.dumps({
            "tool_call": {"name": "read_file", "arguments": {"path": "main.py"}}
        }))

    monkeypatch.setattr(client.ollama, "chat", fake_chat)
    out = client._grammar_constrained_tool_call(
        [{"role": "user", "content": "read main.py"}],
        model="m", tools=fake_tool_schemas, num_ctx=8192, num_predict=512,
    )
    assert out is not None
    assert len(out.message.tool_calls) == 1
    tc = out.message.tool_calls[0]
    assert tc.function.name == "read_file"
    assert tc.function.arguments == {"path": "main.py"}


def test_grammar_retry_parses_answer(monkeypatch, fake_tool_schemas):
    class _Msg:
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class _Resp:
        def __init__(self, content):
            self.message = _Msg(content)
            self.eval_count = 0
            self.prompt_eval_count = 0

    monkeypatch.setattr(
        client.ollama,
        "chat",
        lambda **kw: _Resp(json.dumps({"answer": "the file is empty"})),
    )
    out = client._grammar_constrained_tool_call(
        [{"role": "user", "content": "x"}], model="m",
        tools=fake_tool_schemas, num_ctx=8192, num_predict=512,
    )
    assert out is not None
    assert out.message.content == "the file is empty"
    assert out.message.tool_calls == []


def test_grammar_retry_returns_none_on_invalid_json(monkeypatch, fake_tool_schemas):
    class _Msg:
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class _Resp:
        def __init__(self, content):
            self.message = _Msg(content)

    monkeypatch.setattr(
        client.ollama, "chat",
        lambda **kw: _Resp("this is not json at all"),
    )
    out = client._grammar_constrained_tool_call(
        [{"role": "user", "content": "x"}], model="m",
        tools=fake_tool_schemas, num_ctx=8192, num_predict=512,
    )
    assert out is None


def test_grammar_retry_returns_none_on_unknown_tool(monkeypatch, fake_tool_schemas):
    class _Msg:
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class _Resp:
        def __init__(self, content):
            self.message = _Msg(content)

    monkeypatch.setattr(
        client.ollama, "chat",
        lambda **kw: _Resp(json.dumps({
            "tool_call": {"name": "made_up_tool", "arguments": {}}
        })),
    )
    out = client._grammar_constrained_tool_call(
        [{"role": "user", "content": "x"}], model="m",
        tools=fake_tool_schemas, num_ctx=8192, num_predict=512,
    )
    assert out is None


def test_grammar_retry_swallows_ollama_errors(monkeypatch, fake_tool_schemas):
    def bad_chat(**kw):
        raise ollama.ResponseError("boom", 500)

    monkeypatch.setattr(client.ollama, "chat", bad_chat)
    out = client._grammar_constrained_tool_call(
        [{"role": "user", "content": "x"}], model="m",
        tools=fake_tool_schemas, num_ctx=8192, num_predict=512,
    )
    assert out is None


# ── call_nonstream recovery ladder ───────────────────────────────────────────

def test_call_nonstream_uses_grammar_path_on_500(monkeypatch, fake_tool_schemas):
    """First call 500s; grammar retry succeeds; we get the synthetic response."""
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ollama.ResponseError("malformed tool call", 500)
        # Second call (grammar retry) — return a valid JSON answer
        class _M:
            content = json.dumps({"answer": "hello"})
            tool_calls = []
        class _R:
            message = _M()
            eval_count = 1
            prompt_eval_count = 2
        return _R()

    monkeypatch.setattr(client.ollama, "chat", fake_chat)
    out = client.call_nonstream(
        [{"role": "user", "content": "hi"}],
        model="m",
        tools=fake_tool_schemas,
    )
    assert out.message.content == "hello"
    assert len(calls) == 2
    # Second call should have included the grammar format
    assert "format" in calls[1]


def test_call_nonstream_falls_through_to_plain_when_grammar_also_fails(
    monkeypatch, fake_tool_schemas
):
    """First call 500s, grammar retry returns garbage, we fall back to plain (no tools)."""
    calls = []

    class _Msg:
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    class _Resp:
        def __init__(self, content):
            self.message = _Msg(content)
            self.eval_count = 0
            self.prompt_eval_count = 0

    def fake_chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ollama.ResponseError("malformed", 500)
        if len(calls) == 2:
            return _Resp("totally not json")
        # Third call — plain chat with no tools — succeed.
        return _Resp("plain reply")

    monkeypatch.setattr(client.ollama, "chat", fake_chat)
    out = client.call_nonstream(
        [{"role": "user", "content": "hi"}],
        model="m",
        tools=fake_tool_schemas,
    )
    assert out.message.content == "plain reply"
    assert len(calls) == 3
    # Third call should NOT have tools
    assert "tools" not in calls[2]


def test_call_nonstream_does_not_retry_on_non_500_errors(monkeypatch, fake_tool_schemas):
    def fake_chat(**kwargs):
        raise ollama.ResponseError("bad request", 400)

    monkeypatch.setattr(client.ollama, "chat", fake_chat)
    with pytest.raises(ollama.ResponseError):
        client.call_nonstream(
            [{"role": "user", "content": "x"}],
            model="m",
            tools=fake_tool_schemas,
        )


def test_call_nonstream_does_not_retry_when_no_tools(monkeypatch):
    """500 without tools is a real bug, not the malformed-tool-call quirk — re-raise."""
    def fake_chat(**kwargs):
        raise ollama.ResponseError("server error", 500)

    monkeypatch.setattr(client.ollama, "chat", fake_chat)
    with pytest.raises(ollama.ResponseError):
        client.call_nonstream(
            [{"role": "user", "content": "x"}], model="m",
        )


# ── build_system_messages includes new sections ──────────────────────────────

def test_build_system_messages_returns_a_single_system_msg(monkeypatch):
    monkeypatch.setattr(
        client, "_system_prompt", lambda num_ctx=0: "BASE"
    )
    # Stub all the optional sections so we don't depend on filesystem state
    monkeypatch.setitem(__import__("sys").modules, "farcode.coder_md",
                        type(__import__("sys"))("stub"))
    out = client.build_system_messages(num_ctx=8192)
    assert len(out) == 1
    assert out[0]["role"] == "system"
    assert "BASE" in out[0]["content"]
