"""Tests for chat.py token accounting, trimming, and auto-compaction."""

from farcode import chat


# ── _est_tokens / _total_tokens ───────────────────────────────────────────────

def test_est_tokens_minimum_overhead():
    assert chat._est_tokens({"role": "user", "content": ""}) >= 4


def test_est_tokens_scales_with_length():
    short = chat._est_tokens({"role": "user", "content": "hello"})
    long = chat._est_tokens({"role": "user", "content": "hello" * 100})
    assert long > short


def test_total_tokens_sums():
    msgs = [
        {"role": "system", "content": "x" * 35},
        {"role": "user", "content": "y" * 35},
    ]
    expected = chat._est_tokens(msgs[0]) + chat._est_tokens(msgs[1])
    assert chat._total_tokens(msgs) == expected


def test_total_tokens_includes_tool_calls():
    plain = chat._total_tokens([{"role": "assistant", "content": "ok"}])
    with_tc = chat._total_tokens([
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "x"}}}],
        }
    ])
    assert with_tc > plain


# ── _trim_messages ────────────────────────────────────────────────────────────

def test_trim_passes_through_when_under_budget():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = chat._trim_messages(msgs, num_ctx=8192, num_predict=512)
    assert out == msgs


def test_trim_drops_oldest_pair_when_over_budget():
    big = "x" * 4000
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"q{i} {big}"})
        msgs.append({"role": "assistant", "content": f"a{i} {big}"})

    out = chat._trim_messages(msgs, num_ctx=4096, num_predict=512)
    # System always preserved
    assert out[0]["role"] == "system"
    # Some turns must have been dropped
    assert len(out) < len(msgs)
    # The most recent turn must still be there
    assert any(m.get("content", "").startswith("a5") for m in out)


def test_trim_does_not_mutate_original():
    big = "x" * 4000
    original = [{"role": "system", "content": "sys"}]
    for i in range(5):
        original.append({"role": "user", "content": f"q{i} {big}"})
        original.append({"role": "assistant", "content": f"a{i} {big}"})
    snapshot = list(original)
    chat._trim_messages(original, num_ctx=4096, num_predict=512)
    assert original == snapshot


# ── _snap_past_tool_replies ───────────────────────────────────────────────────

def test_snap_past_tool_replies_skips_tools():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "calling tool", "tool_calls": [{}]},
        {"role": "tool", "content": "result1"},
        {"role": "tool", "content": "result2"},
        {"role": "user", "content": "next"},
    ]
    # idx 2 points at first tool reply — should be moved past both tool messages.
    assert chat._snap_past_tool_replies(msgs, 2) == 4


def test_snap_past_tool_replies_no_tools():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert chat._snap_past_tool_replies(msgs, 1) == 1


# ── _auto_compact ─────────────────────────────────────────────────────────────

def test_auto_compact_no_op_when_under_threshold():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    out = chat._auto_compact(msgs, model="x", num_ctx=8192, num_predict=512)
    assert out is msgs


def test_auto_compact_preserves_system_and_recent_pairs(monkeypatch):
    # Stub the summarizer so we don't hit Ollama.
    monkeypatch.setattr(
        chat, "_summarize_turns", lambda turns, model, num_ctx: "- summary bullet"
    )
    monkeypatch.setattr(chat, "append_entry", lambda **kw: None)

    big = "x" * 5000
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(8):
        msgs.append({"role": "user", "content": f"q{i} {big}"})
        msgs.append({"role": "assistant", "content": f"a{i} {big}"})

    out = chat._auto_compact(msgs, model="x", num_ctx=4096, num_predict=512, force=True)
    # System preserved
    assert out[0]["role"] == "system"
    # The synthetic summary lives right after the system message
    assert out[1]["role"] == "user"
    assert out[1]["content"].startswith("[Summary of earlier turns]")
    assert out[2]["role"] == "assistant"
    assert out[2]["content"] == "Acknowledged."
    # Last 4 user→assistant pairs from the original survive (q4..q7)
    tail_users = [m for m in out[3:] if m.get("role") == "user"]
    assert any("q4" in m["content"] for m in tail_users)
    assert any("q7" in m["content"] for m in tail_users)


def test_auto_compact_keeps_tool_pair_intact(monkeypatch):
    monkeypatch.setattr(
        chat, "_summarize_turns", lambda turns, model, num_ctx: "- s"
    )
    monkeypatch.setattr(chat, "append_entry", lambda **kw: None)

    big = "x" * 5000
    # Build a long history where a tool_calls/tool pair sits at the boundary.
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"q{i} {big}"})
        msgs.append({"role": "assistant", "content": f"a{i}", "tool_calls": [{"function": {"name": "x", "arguments": {}}}]})
        msgs.append({"role": "tool", "content": f"r{i}"})

    out = chat._auto_compact(msgs, model="x", num_ctx=4096, num_predict=512, force=True)
    # No bare 'tool' message should appear without a preceding assistant w/ tool_calls.
    for i, m in enumerate(out):
        if m.get("role") == "tool":
            prev = out[i - 1] if i > 0 else None
            assert prev is not None
            assert prev.get("role") == "assistant"
            assert prev.get("tool_calls"), f"orphan tool reply at idx {i}: {m}"


def test_auto_compact_returns_original_when_summary_fails(monkeypatch):
    monkeypatch.setattr(
        chat, "_summarize_turns", lambda turns, model, num_ctx: ""
    )
    big = "x" * 5000
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"q{i} {big}"})
        msgs.append({"role": "assistant", "content": f"a{i} {big}"})
    out = chat._auto_compact(msgs, model="x", num_ctx=4096, num_predict=512, force=True)
    # On summary failure, return the original unchanged so the caller can fall
    # back to hard-trimming.
    assert out is msgs
