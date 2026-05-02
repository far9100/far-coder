"""Stuck-detection and tool-budget tests for solve.py.

Drives FakeOllama with malicious / pathological scripts to verify the
sentinel-exception flow correctly classifies the exit_reason."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eval.swe_bench import solve

from .conftest import make_response, make_tool_call
from .test_swe_solve import _init_repo, patchable_chat  # noqa: F401  (fixture reuse)


def test_solve_aborts_on_repeated_identical_edit(
    fake_ollama, patchable_chat, tmp_path,
):
    """Model emits identical edit_file 4 times. After 2 actual executes and 1
    suppressed call, the 4th raises _SolveAbort; exit_reason should be 'stuck'."""
    repo = tmp_path / "repo"
    base = _init_repo(repo, {"app.py": 'x = "foo"\n'})

    same_call = lambda: make_tool_call(
        "edit_file", path="app.py", old_str='"foo"', new_str='"bar"',
    )
    fake_ollama([
        make_response("try 1", tool_calls=[same_call()]),
        make_response("try 2", tool_calls=[same_call()]),
        make_response("try 3", tool_calls=[same_call()]),
        make_response("try 4", tool_calls=[same_call()]),
        # Safety extras in case the loop ever asks one more time:
        make_response("try 5", tool_calls=[same_call()]),
        make_response("try 6", tool_calls=[same_call()]),
    ])

    instance = {
        "instance_id": "test__stuck-1",
        "repo": "test/stuck",
        "base_commit": base,
        "problem_statement": "x is wrong",
        "hints_text": "",
    }
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=100, timeout_s=30,
    )

    assert result.exit_reason == "stuck"
    # The first execute actually flipped foo→bar; the 2nd execute then failed
    # to find "foo" again (file already says "bar"). The 3rd was suppressed
    # synthetically. The 4th raised. Net: file is "bar", patch is non-empty.
    assert (repo / "app.py").read_text(encoding="utf-8") == 'x = "bar"\n'


def test_solve_aborts_when_tool_budget_exhausted(
    fake_ollama, patchable_chat, tmp_path,
):
    """With max_tools=2 and a script that wants 5 distinct tool calls, exit
    after the 3rd call raises and stops the loop."""
    repo = tmp_path / "repo"
    base = _init_repo(repo, {f"f{i}.py": "" for i in range(5)})

    fake_ollama([
        make_response(f"step {i}", tool_calls=[make_tool_call("read_file", path=f"f{i}.py")])
        for i in range(5)
    ] + [make_response("done")])

    instance = {
        "instance_id": "test__budget-1",
        "repo": "test/budget",
        "base_commit": base,
        "problem_statement": "look at all files",
        "hints_text": "",
    }
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=2, timeout_s=30,
    )

    assert result.exit_reason == "max_tools"
    assert result.tool_calls_used == 3  # the 3rd call is the one that tripped the cap


def test_read_loop_suppressed_at_threshold(fake_ollama, patchable_chat, tmp_path):
    """5 identical reads: the 5th returns the suppression message, the file
    is NOT read again that turn, and the model can keep going past it."""
    repo = tmp_path / "repo"
    base = _init_repo(repo, {"app.py": "x = 1\n"})

    # 4 identical reads, then a 5th identical read, then an edit, then done.
    same_read = lambda: make_tool_call("read_file", path="app.py")
    fake_ollama([
        make_response(f"r{i}", tool_calls=[same_read()]) for i in range(5)
    ] + [
        make_response("now editing",
                      tool_calls=[make_tool_call("edit_file",
                                                  path="app.py",
                                                  old_str="x = 1",
                                                  new_str="x = 2")]),
        make_response("done"),
    ])

    instance = {
        "instance_id": "test__readloop-1",
        "repo": "test/readloop",
        "base_commit": base,
        "problem_statement": "look",
        "hints_text": "",
    }
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=100, timeout_s=30,
    )

    # Model should have continued past the suppression and produced a patch
    assert result.exit_reason == "natural"
    assert "+x = 2" in result.model_patch


def test_all_read_nudge_appends_after_threshold(
    monkeypatch, fake_ollama, patchable_chat, tmp_path,
):
    """After ALL_READ_NUDGE_THRESHOLD pure-read calls without any mutate,
    the next read tool should return its content with a [harness nudge: ...]
    suffix appended."""
    # Lower the threshold so the test stays small and fast.
    monkeypatch.setattr(solve, "ALL_READ_NUDGE_THRESHOLD", 3)

    repo = tmp_path / "repo"
    files = {f"f{i}.py": f"x = {i}\n" for i in range(6)}
    base = _init_repo(repo, files)

    fake = fake_ollama([
        make_response(f"r{i}", tool_calls=[make_tool_call("read_file", path=f"f{i}.py")])
        for i in range(5)
    ] + [make_response("done")])

    instance = {
        "instance_id": "test__nudge-1",
        "repo": "test/nudge",
        "base_commit": base,
        "problem_statement": "look",
        "hints_text": "",
    }
    # Capture tool messages by looking at last fake_ollama call's messages
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=100, timeout_s=30,
    )
    # Either "natural" (model heeded the nudge and stopped) or "no_edits"
    # (stopped without editing) is fine — what matters is that the nudge
    # got injected into a tool result and exposed to the model.
    assert result.exit_reason in ("natural", "no_edits")
    seen_nudge = any(
        any(
            m.get("role") == "tool" and "harness nudge" in (m.get("content") or "")
            for m in call["messages"]
        )
        for call in fake.calls
    )
    assert seen_nudge, "expected [harness nudge: ...] in a tool result"


def test_solve_aborts_on_timeout(monkeypatch, fake_ollama, patchable_chat, tmp_path):
    """Force the wrapped call_nonstream / execute_tool to see a clock past the
    timeout — should set state.timed_out and raise."""
    repo = tmp_path / "repo"
    base = _init_repo(repo, {"app.py": "x = 1\n"})

    fake_ollama([
        make_response("step", tool_calls=[make_tool_call("read_file", path="app.py")]),
        make_response("done"),
    ])

    # Patch time.monotonic inside solve.py so the second call reads as past timeout
    real_mono = solve.time.monotonic
    calls = {"n": 0}

    def fake_mono():
        calls["n"] += 1
        # First call is the started_at capture (state init).
        # All subsequent reads jump 9999s into the future → timeout fires immediately.
        if calls["n"] == 1:
            return real_mono()
        return real_mono() + 9999.0

    monkeypatch.setattr(solve.time, "monotonic", fake_mono)

    instance = {
        "instance_id": "test__timeout-1",
        "repo": "test/timeout",
        "base_commit": base,
        "problem_statement": "anything",
        "hints_text": "",
    }
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=100, timeout_s=10,
    )

    assert result.exit_reason == "timeout"
