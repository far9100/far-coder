"""End-to-end tests for ``chat._run_agent_turn`` driven by a scripted FakeOllama.

These tests exercise the full agent loop — tool dispatch, message-history
mutation, and turn termination — without touching the real Ollama server.
Per-component logic (token trimming, auto-compaction, grammar retry) lives
in test_chat.py and test_client.py; this file focuses on the loop's
turn-to-turn behaviour as a black box.
"""

from __future__ import annotations

from farcode import chat

from .conftest import make_response, make_tool_call


# ── Plain completion ──────────────────────────────────────────────────────────

def test_text_only_response_terminates_loop(fake_ollama):
    """No tool calls → assistant message appended and the loop returns."""
    fake = fake_ollama([make_response("the answer is 42")])

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "what is the answer?"},
    ]
    chat._run_agent_turn(msgs, model="m", num_ctx=8192, num_predict=512)

    assert fake.call_count == 1
    assert msgs[-1] == {"role": "assistant", "content": "the answer is 42"}


# ── Single tool then completion ───────────────────────────────────────────────

def test_one_tool_then_text(fake_ollama, monkeypatch):
    """Tool call → tool result appended → second call returns text → loop ends."""
    fake = fake_ollama([
        make_response("reading", tool_calls=[make_tool_call("read_file", path="x.py")]),
        make_response("file is empty"),
    ])
    monkeypatch.setattr(chat, "execute_tool", lambda name, args: "<empty>")

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "read x.py"},
    ]
    chat._run_agent_turn(msgs, model="m", num_ctx=8192, num_predict=512)

    assert fake.call_count == 2

    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]

    # Assistant message recorded the tool call
    assistant_with_call = msgs[2]
    assert assistant_with_call["tool_calls"][0]["function"]["name"] == "read_file"
    assert assistant_with_call["tool_calls"][0]["function"]["arguments"] == {"path": "x.py"}

    # Tool result was fed back
    assert msgs[3] == {"role": "tool", "content": "<empty>"}

    # Final text completion
    assert msgs[-1] == {"role": "assistant", "content": "file is empty"}


# ── Multi-turn tool chain ─────────────────────────────────────────────────────

def test_multi_tool_chain_reaches_completion(fake_ollama, monkeypatch):
    """read → edit → done. Each tool result must reach the model on the next call."""
    fake = fake_ollama([
        make_response("step 1", tool_calls=[make_tool_call("read_file", path="x.py")]),
        make_response("step 2", tool_calls=[make_tool_call(
            "edit_file", path="x.py", old_text="foo", new_text="bar",
        )]),
        make_response("done"),
    ])

    executed: list[tuple[str, dict]] = []

    def fake_execute(name, args):
        executed.append((name, args))
        return f"ok({name})"

    monkeypatch.setattr(chat, "execute_tool", fake_execute)

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "rename foo to bar"},
    ]
    chat._run_agent_turn(msgs, model="m", num_ctx=8192, num_predict=512)

    # Both tools fired in order
    assert [name for name, _ in executed] == ["read_file", "edit_file"]
    # Three model calls: 2 tool turns + 1 final
    assert fake.call_count == 3
    # Each subsequent model call saw the prior tool result
    assert any(m["role"] == "tool" and m["content"] == "ok(read_file)" for m in fake.calls[1]["messages"])
    assert any(m["role"] == "tool" and m["content"] == "ok(edit_file)" for m in fake.calls[2]["messages"])


# ── Tool error fed back to model ──────────────────────────────────────────────

def test_tool_error_string_reaches_next_turn(fake_ollama, monkeypatch):
    """When a tool returns an error string, the model sees it and can react."""
    fake = fake_ollama([
        make_response("trying", tool_calls=[make_tool_call("read_file", path="missing.py")]),
        make_response("file does not exist; aborting"),
    ])
    monkeypatch.setattr(
        chat, "execute_tool",
        lambda name, args: "Error: file not found: missing.py",
    )

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "read missing.py"},
    ]
    chat._run_agent_turn(msgs, model="m", num_ctx=8192, num_predict=512)

    # Second model call should have seen the error tool result
    second_call_msgs = fake.calls[1]["messages"]
    assert any(
        m["role"] == "tool" and "file not found" in m["content"]
        for m in second_call_msgs
    )
    assert msgs[-1]["content"].startswith("file does not exist")


# ── Auto-compaction triggers between turns ────────────────────────────────────

def test_compaction_runs_when_history_is_huge(fake_ollama, monkeypatch):
    """A long pre-stuffed history triggers _auto_compact at the start of the turn."""
    monkeypatch.setattr(
        chat, "_summarize_turns", lambda turns, model, num_ctx: "- prior context",
    )
    monkeypatch.setattr(chat, "append_entry", lambda **kw: None)

    fake = fake_ollama([make_response("acknowledged")])

    big = "x" * 5000
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(8):
        msgs.append({"role": "user", "content": f"q{i} {big}"})
        msgs.append({"role": "assistant", "content": f"a{i} {big}"})
    msgs.append({"role": "user", "content": "final question"})

    chat._run_agent_turn(msgs, model="m", num_ctx=4096, num_predict=512)

    # Compaction inserts a synthetic summary pair right after the system msg.
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"].startswith("[Summary of earlier turns]")
    assert msgs[2]["content"] == "Acknowledged."
    # Final assistant response from the model is the last message.
    assert msgs[-1] == {"role": "assistant", "content": "acknowledged"}


# ── FakeOllama exhaustion is loud ─────────────────────────────────────────────

def test_fake_ollama_raises_helpful_error_when_exhausted(fake_ollama, monkeypatch):
    """If the test under-scripts responses, the failure surfaces clearly.

    The agent loop's ``except Exception`` handler at chat.py:530 catches the
    AssertionError and routes it through ``print_error``. We verify the message
    is informative enough that the test author can fix their script.
    """
    fake_ollama([
        make_response("calling", tool_calls=[make_tool_call("read_file", path="x")]),
    ])
    monkeypatch.setattr(chat, "execute_tool", lambda name, args: "ok")

    errors: list[str] = []
    monkeypatch.setattr(chat, "print_error", lambda msg, *a, **kw: errors.append(msg))

    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    chat._run_agent_turn(msgs, model="m", num_ctx=8192, num_predict=512)

    assert errors, "expected the agent loop to surface the exhaustion error via print_error"
    assert any("exhausted" in e for e in errors), errors


# ── Tools were forwarded to the model ─────────────────────────────────────────

def test_tool_schemas_passed_to_model(fake_ollama):
    """The agent loop must forward the full TOOL_SCHEMAS list on every call."""
    fake = fake_ollama([make_response("ok")])

    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    chat._run_agent_turn(msgs, model="m", num_ctx=8192, num_predict=512)

    sent_tools = fake.calls[0]["tools"]
    assert sent_tools is not None
    sent_names = {t["function"]["name"] for t in sent_tools}
    # Sanity check a few core tools are on the wire.
    assert {"read_file", "edit_file", "write_file"}.issubset(sent_names)
