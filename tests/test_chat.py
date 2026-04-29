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


# ── _run_agent_turn tool-call cap ─────────────────────────────────────────────

class _StubFn:
    def __init__(self, name, args):
        self.name = name
        self.arguments = args


class _StubToolCall:
    def __init__(self, name, args):
        self.function = _StubFn(name, args)


class _StubMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _StubResponse:
    def __init__(self, content, tool_calls=None, eval_count=10, prompt_eval_count=20):
        self.message = _StubMessage(content, tool_calls)
        self.eval_count = eval_count
        self.prompt_eval_count = prompt_eval_count


def test_agent_turn_caps_tool_calls(monkeypatch):
    """When the model returns N>cap tools, only `cap` get executed and the
    rest get a single tool message noting they were dropped."""
    monkeypatch.setenv("FARCODE_MAX_TOOLS_PER_TURN", "1")

    # First call: model returns 3 tool calls. Second call: model returns no tools.
    responses = iter([
        _StubResponse(
            "calling tools",
            tool_calls=[
                _StubToolCall("read_file", {"path": "a.txt"}),
                _StubToolCall("read_file", {"path": "b.txt"}),
                _StubToolCall("read_file", {"path": "c.txt"}),
            ],
        ),
        _StubResponse("done", tool_calls=[]),
    ])

    monkeypatch.setattr(chat, "call_with_thinking", lambda fn, stats: fn())
    monkeypatch.setattr(chat, "call_nonstream", lambda *a, **kw: next(responses))
    monkeypatch.setattr(chat, "render_response", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "print_info", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "print_error", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "print_tool_call", lambda *a, **kw: None)

    executed = []

    def fake_execute(name, args):
        executed.append((name, args))
        return f"result for {args.get('path', '?')}"

    monkeypatch.setattr(chat, "execute_tool", fake_execute)

    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    chat._run_agent_turn(msgs, model="m", num_ctx=8192, num_predict=512)

    # Only one tool actually ran
    assert len(executed) == 1
    assert executed[0] == ("read_file", {"path": "a.txt"})

    # Assistant message records only the capped tool_calls (1)
    assistant_msgs = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistant_msgs[0]["tool_calls"]) == 1

    # A "dropped" notice was appended as a tool message
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert any("dropped" in (m.get("content") or "") for m in tool_msgs)


def test_agent_turn_no_cap_message_when_under_limit(monkeypatch):
    monkeypatch.setenv("FARCODE_MAX_TOOLS_PER_TURN", "5")

    responses = iter([
        _StubResponse(
            "x",
            tool_calls=[_StubToolCall("read_file", {"path": "a.txt"})],
        ),
        _StubResponse("done", tool_calls=[]),
    ])
    monkeypatch.setattr(chat, "call_with_thinking", lambda fn, stats: fn())
    monkeypatch.setattr(chat, "call_nonstream", lambda *a, **kw: next(responses))
    monkeypatch.setattr(chat, "render_response", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "print_info", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "print_tool_call", lambda *a, **kw: None)
    monkeypatch.setattr(chat, "execute_tool", lambda name, args: "ok")

    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    chat._run_agent_turn(msgs, model="m", num_ctx=8192, num_predict=512)

    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert not any("dropped" in (m.get("content") or "") for m in tool_msgs)


# ── _summary_model ────────────────────────────────────────────────────────────

def test_summary_model_falls_back_to_main_when_unset(monkeypatch):
    monkeypatch.delenv("FARCODE_SUMMARY_MODEL", raising=False)
    assert chat._summary_model("qwen3.5:4b") == "qwen3.5:4b"


def test_summary_model_uses_env_when_set(monkeypatch):
    monkeypatch.setenv("FARCODE_SUMMARY_MODEL", "qwen3:0.5b")
    assert chat._summary_model("qwen3.5:4b") == "qwen3:0.5b"


def test_summary_model_blank_env_falls_back(monkeypatch):
    monkeypatch.setenv("FARCODE_SUMMARY_MODEL", "   ")
    assert chat._summary_model("main") == "main"


def test_summarize_turns_uses_summary_model(monkeypatch):
    monkeypatch.setenv("FARCODE_SUMMARY_MODEL", "tiny-model")

    captured = {}

    class _M:
        def __init__(self, c): self.content = c
    class _R:
        def __init__(self, c): self.message = _M(c)

    def fake_call(msgs, model, tools=None, num_ctx=0, num_predict=0):
        captured["model"] = model
        return _R("- bullet")

    monkeypatch.setattr(chat, "call_nonstream", fake_call)

    out = chat._summarize_turns(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        model="qwen3.5:4b", num_ctx=8192,
    )
    assert out == "- bullet"
    assert captured["model"] == "tiny-model"


# ── _undo_last_write ──────────────────────────────────────────────────────────

def test_undo_last_write_restores_prior_content(tmp_path):
    from farcode.tools import _edit_file, clear_snapshots

    clear_snapshots()
    f = tmp_path / "x.txt"
    f.write_text("original", encoding="utf-8")
    _edit_file(str(f), "original", "modified")
    assert f.read_text(encoding="utf-8") == "modified"

    msg = chat._undo_last_write()
    assert "Undone" in msg
    assert f.read_text(encoding="utf-8") == "original"


def test_undo_last_write_deletes_newly_created_file(tmp_path):
    from farcode.tools import _create_file, clear_snapshots

    clear_snapshots()
    f = tmp_path / "fresh.txt"
    _create_file(str(f), "content")
    assert f.exists()

    msg = chat._undo_last_write()
    assert "deleted" in msg
    assert not f.exists()


def test_undo_last_write_empty_stack():
    from farcode.tools import clear_snapshots

    clear_snapshots()
    assert chat._undo_last_write() == "Nothing to undo."


def test_undo_pops_only_one_entry(tmp_path):
    from farcode.tools import _edit_file, clear_snapshots

    clear_snapshots()
    f = tmp_path / "y.txt"
    f.write_text("v1", encoding="utf-8")
    _edit_file(str(f), "v1", "v2")
    _edit_file(str(f), "v2", "v3")

    chat._undo_last_write()  # back to v2
    assert f.read_text(encoding="utf-8") == "v2"
    chat._undo_last_write()  # back to v1
    assert f.read_text(encoding="utf-8") == "v1"


# ── _show_git_diff ────────────────────────────────────────────────────────────

def test_show_git_diff_outside_a_repo_does_not_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = chat._show_git_diff()
    # Either "git diff returned N: not a git repository" or an empty result.
    assert isinstance(out, str)


# ── _reindex_code ─────────────────────────────────────────────────────────────

def test_reindex_code_with_no_files_returns_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Stub Ollama embed so no actual model call happens
    if "farcode.embeddings" in __import__("sys").modules:
        del __import__("sys").modules["farcode.embeddings"]
    from farcode import embeddings as emb
    monkeypatch.setattr(emb, "_embed_texts", lambda texts: [])

    out = chat._reindex_code()
    assert "Indexed 0" in out


# ── /help + slash command registry ───────────────────────────────────────────

def test_slash_commands_registry_contains_core_commands():
    from farcode.commands import SLASH_COMMANDS

    names = [name for name, _ in SLASH_COMMANDS]
    flat = " ".join(names)
    for required in ("/help", "/exit", "/clear", "/file", "/model",
                     "/compact", "/rules", "/undo", "/diff", "/reindex",
                     "/resume", "/tasks"):
        assert required in flat, f"missing {required} in registry"


def test_slash_commands_registry_descriptions_nonempty():
    from farcode.commands import SLASH_COMMANDS

    for name, desc in SLASH_COMMANDS:
        assert name and isinstance(name, str)
        assert desc and isinstance(desc, str)


def test_banner_line_includes_help_first():
    from farcode.commands import banner_line

    line = banner_line()
    assert line.startswith("/help")
    assert "/exit" in line
    assert "/tasks" in line


def test_slash_commands_grouped_into_sections():
    from farcode.commands import SLASH_COMMAND_SECTIONS

    section_names = [name for name, _ in SLASH_COMMAND_SECTIONS]
    assert "Discovery" in section_names
    assert "Files & code" in section_names
    assert "Session" in section_names
    assert "Config" in section_names

    # /help and /tasks live under Discovery
    discovery_items = next(items for name, items in SLASH_COMMAND_SECTIONS
                           if name == "Discovery")
    discovery_cmds = [c for c, _ in discovery_items]
    assert "/help" in discovery_cmds
    assert "/tasks" in discovery_cmds


def test_print_help_renders_section_headers(capsys):
    import io
    from farcode import ui
    from rich.console import Console

    buf = io.StringIO()
    test_console = Console(file=buf, width=120, force_terminal=False, legacy_windows=False)
    original = ui.console
    ui.console = test_console
    try:
        ui.print_help()
    finally:
        ui.console = original

    out = buf.getvalue()
    # Each section header must appear in the help output
    for section in ("Discovery", "Files & code", "Session", "Config"):
        assert section in out, f"section '{section}' missing from /help"


def test_print_help_renders_all_registry_entries(capsys):
    import io
    from farcode import ui
    from farcode.commands import SLASH_COMMANDS
    from rich.console import Console

    buf = io.StringIO()
    test_console = Console(file=buf, width=120, force_terminal=False, legacy_windows=False)
    original = ui.console
    ui.console = test_console
    try:
        ui.print_help()
    finally:
        ui.console = original

    out = buf.getvalue()
    for name, desc in SLASH_COMMANDS:
        # The first token of the name (handles "/exit  /quit" entries).
        first = name.split()[0]
        assert first in out, f"{first} missing from /help output"


# ── _expand_at_mentions / @completion ────────────────────────────────────────

def test_expand_at_mentions_inlines_existing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "hello.txt"
    f.write_text("greetings", encoding="utf-8")
    text, found, misses = chat._expand_at_mentions("look at @hello.txt please")
    assert "greetings" in text
    assert any(name.endswith("hello.txt") for name in found)
    assert misses == []


def test_expand_at_mentions_reports_miss(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text, found, misses = chat._expand_at_mentions("see @nope.txt for context")
    assert "@nope.txt" in text  # original token preserved
    assert found == []
    assert any("@nope.txt" in m and "not found" in m for m in misses)


def test_expand_at_mentions_directory_is_not_a_hit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "subdir").mkdir()
    text, found, misses = chat._expand_at_mentions("read @subdir entirely")
    assert found == []
    assert any("@subdir" in m for m in misses)


def test_expand_at_mentions_mixed_hits_and_misses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "good.txt").write_text("yes", encoding="utf-8")
    text, found, misses = chat._expand_at_mentions("@good.txt and @missing.txt")
    assert any(name.endswith("good.txt") for name in found)
    assert any("@missing.txt" in m for m in misses)
    assert "yes" in text
    assert "@missing.txt" in text  # miss token preserved


def test_expand_at_mentions_refuses_oversize_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    big = tmp_path / "huge.log"
    big.write_text("x" * (chat.MAX_INLINE_BYTES + 5000), encoding="utf-8")
    text, found, misses = chat._expand_at_mentions("look at @huge.log")
    assert found == []
    assert any("@huge.log" in m and "exceeds" in m for m in misses)
    assert "@huge.log" in text  # token preserved
    assert "xxxxx" not in text  # content NOT inlined


def test_expand_at_mentions_refuses_binary_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bin_file = tmp_path / "image.png"
    bin_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 100)
    text, found, misses = chat._expand_at_mentions("see @image.png")
    assert found == []
    assert any("@image.png" in m and "binary" in m for m in misses)


def test_expand_at_mentions_inlines_just_under_cap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "borderline.txt"
    f.write_text("y" * (chat.MAX_INLINE_BYTES - 100), encoding="utf-8")
    text, found, misses = chat._expand_at_mentions("@borderline.txt")
    assert any(name.endswith("borderline.txt") for name in found)
    assert misses == []


def test_inject_file_refuses_binary(tmp_path):
    bin_file = tmp_path / "blob.bin"
    bin_file.write_bytes(b"\x00\x01\x02\x03" * 100)
    msgs: list[dict] = []
    assert chat._inject_file(msgs, str(bin_file)) is False
    assert msgs == []


def test_inject_file_refuses_oversize(tmp_path):
    big = tmp_path / "big.log"
    big.write_text("x" * (chat.MAX_INLINE_BYTES + 1), encoding="utf-8")
    msgs: list[dict] = []
    assert chat._inject_file(msgs, str(big)) is False
    assert msgs == []


def test_classify_for_inline_returns_content_for_normal_file(tmp_path):
    f = tmp_path / "ok.txt"
    f.write_text("hello", encoding="utf-8")
    content, reason = chat._classify_for_inline(f)
    assert content == "hello"
    assert reason is None


def test_at_completer_suggests_files_in_cwd(tmp_path, monkeypatch):
    from prompt_toolkit.document import Document
    from farcode.completion import AtFileCompleter

    monkeypatch.chdir(tmp_path)
    (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("b", encoding="utf-8")
    (tmp_path / "src").mkdir()

    comp = AtFileCompleter()
    doc = Document("look at @al")
    suggestions = list(comp.get_completions(doc, complete_event=None))
    displays = [str(s.display) for s in suggestions]
    # Display strings are FormattedText objects; coerce via str()
    flat = " ".join(displays)
    assert "alpha.txt" in flat
    assert "beta.txt" not in flat  # filtered by 'al' prefix


def test_at_completer_no_suggestions_without_at(tmp_path, monkeypatch):
    from prompt_toolkit.document import Document
    from farcode.completion import AtFileCompleter

    monkeypatch.chdir(tmp_path)
    (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")

    comp = AtFileCompleter()
    doc = Document("just text al")
    suggestions = list(comp.get_completions(doc, complete_event=None))
    assert suggestions == []


def test_at_completer_lists_subdirectory(tmp_path, monkeypatch):
    from prompt_toolkit.document import Document
    from farcode.completion import AtFileCompleter

    monkeypatch.chdir(tmp_path)
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "main.py").write_text("x", encoding="utf-8")
    (sub / "util.py").write_text("y", encoding="utf-8")

    comp = AtFileCompleter()
    doc = Document("@src/")
    suggestions = list(comp.get_completions(doc, complete_event=None))
    flat = " ".join(str(s.display) for s in suggestions)
    assert "main.py" in flat
    assert "util.py" in flat


def test_at_completer_skips_hidden_unless_typed(tmp_path, monkeypatch):
    from prompt_toolkit.document import Document
    from farcode.completion import AtFileCompleter

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".secret").write_text("s", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("v", encoding="utf-8")

    comp = AtFileCompleter()
    plain_suggestions = list(comp.get_completions(Document("@"), None))
    plain_flat = " ".join(str(s.display) for s in plain_suggestions)
    assert "visible.txt" in plain_flat
    assert ".secret" not in plain_flat

    dot_suggestions = list(comp.get_completions(Document("@."), None))
    dot_flat = " ".join(str(s.display) for s in dot_suggestions)
    assert ".secret" in dot_flat


def test_print_welcome_uses_registry(capsys):
    import io
    from farcode import ui
    from rich.console import Console

    buf = io.StringIO()
    test_console = Console(file=buf, width=120, force_terminal=False, legacy_windows=False)
    original = ui.console
    ui.console = test_console
    try:
        ui.print_welcome("test-model")
    finally:
        ui.console = original

    out = buf.getvalue()
    assert "test-model" in out
    # Banner must include /help (the registry-driven part)
    assert "/help" in out
    assert "/tasks" in out
