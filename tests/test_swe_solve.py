"""Unit tests for eval/swe_bench/solve.py — patch extraction, prompt
formatting, and end-to-end agent loop with a scripted FakeOllama.

These tests do NOT touch Ollama or any real network. They drive the same
FakeOllama fixture used by test_agent_loop.py."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from eval.swe_bench import solve

from .conftest import make_response, make_tool_call


# ── helpers ───────────────────────────────────────────────────────────────────

def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def _init_repo(repo: Path, files: dict[str, str]) -> str:
    """Build a tmp git repo with the given files, single commit. Returns the
    base_commit SHA."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], cwd=repo)
    _git(["config", "user.email", "test@example.com"], cwd=repo)
    _git(["config", "user.name", "test"], cwd=repo)
    _git(["config", "commit.gpgsign", "false"], cwd=repo)
    for rel, content in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    _git(["add", "-A"], cwd=repo)
    _git(["commit", "-q", "-m", "initial"], cwd=repo)
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return sha


@pytest.fixture
def patchable_chat(monkeypatch):
    """Snapshot every chat-module attribute that solve._install_solve_patches
    mutates, so monkeypatch restores them at teardown. fake_ollama also patches
    a few; the snapshots overlap, which is fine — last setattr wins, and the
    original (pre-test) values are what get restored."""
    from farcode import chat, memory
    for name in (
        "execute_tool", "call_nonstream", "print_tool_call",
        "print_info", "print_error", "render_response", "call_with_thinking",
    ):
        monkeypatch.setattr(chat, name, getattr(chat, name))
    monkeypatch.setattr(memory, "append_entry", memory.append_entry)
    yield


# ── _format_user_prompt ───────────────────────────────────────────────────────

def test_format_user_prompt_includes_problem_and_repo():
    prompt = solve._format_user_prompt({
        "repo": "astropy/astropy",
        "problem_statement": "Bug: division by zero in foo()",
        "hints_text": "see issue #1234",
    })
    assert "astropy/astropy" in prompt
    assert "division by zero" in prompt
    assert "see issue #1234" in prompt
    # Must explicitly forbid editing tests
    assert "NOT the test files" in prompt
    assert "Do NOT call run_bash to execute pytest" in prompt


def test_format_user_prompt_handles_missing_hints():
    prompt = solve._format_user_prompt({
        "repo": "x/y",
        "problem_statement": "broken",
        "hints_text": "",
    })
    assert "Hints:" not in prompt
    assert "broken" in prompt


def test_format_user_prompt_injects_suspect_section():
    """A non-empty suspect_section should appear before the Issue text so the
    model sees concrete file targets up-front."""
    prompt = solve._format_user_prompt(
        {
            "repo": "x/y",
            "problem_statement": "the bug is in foo()",
            "hints_text": "",
        },
        suspect_section="## Suspected locations\n- src/foo.py:10  (mentioned)",
    )
    assert "Suspected locations" in prompt
    # Suspect section appears before Issue: text
    assert prompt.index("Suspected locations") < prompt.index("Issue:")
    assert "src/foo.py:10" in prompt


def test_coerce_test_list_handles_list_input():
    assert solve._coerce_test_list(["a", "b", "a", "c"]) == ["a", "b", "c"]


def test_coerce_test_list_handles_json_string():
    assert solve._coerce_test_list('["x", "y"]') == ["x", "y"]


def test_coerce_test_list_handles_invalid_input():
    assert solve._coerce_test_list(None) == []
    assert solve._coerce_test_list("not json") == []
    assert solve._coerce_test_list(42) == []
    # Non-list JSON
    assert solve._coerce_test_list('{"foo":1}') == []


def test_format_test_contract_includes_fail_and_pass():
    contract = solve._format_test_contract({
        "instance_id": "x__y-1",
        "FAIL_TO_PASS": ["test_alpha", "test_beta"],
        "PASS_TO_PASS": ["test_gamma", "test_delta", "test_epsilon"],
    })
    assert "Test contract" in contract
    assert "test_alpha" in contract
    assert "test_beta" in contract
    # All 3 PASS_TO_PASS fit under the cap of 10
    assert "test_gamma" in contract
    assert "test_delta" in contract
    assert "test_epsilon" in contract


def test_format_test_contract_caps_pass_to_pass_at_sample():
    huge = [f"test_t_{i}" for i in range(50)]
    contract = solve._format_test_contract({
        "instance_id": "x__y-2",
        "FAIL_TO_PASS": ["test_target"],
        "PASS_TO_PASS": huge,
    })
    # Should mention sample of 10 from 50
    assert "sample of 10 from 50" in contract
    # Test names mentioned should be exactly 1 (FAIL) + 10 (PASS sample) = 11
    mentioned = sum(1 for line in contract.splitlines() if line.strip().startswith("- test_"))
    assert mentioned == 11


def test_format_test_contract_sampling_is_deterministic_per_instance():
    """Same instance_id → same sample, so A/B comparisons across runs are honest."""
    huge = [f"test_t_{i}" for i in range(50)]
    a = solve._format_test_contract({
        "instance_id": "x__y-3", "FAIL_TO_PASS": [], "PASS_TO_PASS": huge,
    })
    b = solve._format_test_contract({
        "instance_id": "x__y-3", "FAIL_TO_PASS": [], "PASS_TO_PASS": huge,
    })
    assert a == b
    # Different instance_id → different sample (with very high probability)
    c = solve._format_test_contract({
        "instance_id": "x__y-DIFFERENT", "FAIL_TO_PASS": [], "PASS_TO_PASS": huge,
    })
    assert a != c


def test_format_test_contract_returns_blank_when_both_missing():
    assert solve._format_test_contract({"instance_id": "x", "FAIL_TO_PASS": [], "PASS_TO_PASS": []}) == ""
    assert solve._format_test_contract({"instance_id": "x"}) == ""


def test_format_user_prompt_injects_test_contract_when_env_on(monkeypatch):
    monkeypatch.setenv("FARCODE_SWE_TEST_CONTRACT", "1")
    prompt = solve._format_user_prompt({
        "repo": "x/y",
        "problem_statement": "broken",
        "hints_text": "",
        "instance_id": "x__y-1",
        "FAIL_TO_PASS": ["test_a"],
        "PASS_TO_PASS": ["test_b"],
    })
    assert "Test contract" in prompt
    assert "test_a" in prompt
    # Order: contract before Issue text
    assert prompt.index("Test contract") < prompt.index("Issue:")


def test_format_user_prompt_skips_test_contract_when_env_off(monkeypatch):
    monkeypatch.delenv("FARCODE_SWE_TEST_CONTRACT", raising=False)
    prompt = solve._format_user_prompt({
        "repo": "x/y",
        "problem_statement": "broken",
        "hints_text": "",
        "instance_id": "x__y-1",
        "FAIL_TO_PASS": ["test_a"],
        "PASS_TO_PASS": ["test_b"],
    })
    assert "Test contract" not in prompt


def test_format_user_prompt_skips_empty_suspect_section():
    prompt = solve._format_user_prompt(
        {"repo": "x/y", "problem_statement": "broken", "hints_text": ""},
        suspect_section="",
    )
    assert "Suspected locations" not in prompt


# ── _extract_patch ────────────────────────────────────────────────────────────

def test_extract_patch_basic_change(tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo, {"app.py": 'def hello():\n    return "wrong"\n'})
    (repo / "app.py").write_text('def hello():\n    return "right"\n', encoding="utf-8")

    patch, dropped = solve._extract_patch(repo, base)

    assert patch  # non-empty
    assert "+++ b/app.py" in patch
    assert '+    return "right"' in patch
    assert '-    return "wrong"' in patch
    assert dropped == []
    # Repo path must be relative; no absolute prefix leaked
    assert str(repo) not in patch


def test_extract_patch_excludes_test_files(tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo, {
        "app.py": "x = 1\n",
        "tests/test_app.py": "def test_x(): assert True\n",
        "conftest.py": "# conftest\n",
    })
    # Modify all three: source, tests/, and conftest.py
    (repo / "app.py").write_text("x = 2\n", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text("def test_x(): assert False\n", encoding="utf-8")
    (repo / "conftest.py").write_text("# changed\n", encoding="utf-8")

    patch, dropped = solve._extract_patch(repo, base)

    assert "+++ b/app.py" in patch
    assert "test_app.py" not in patch
    assert "conftest.py" not in patch
    assert any("test_app.py" in d for d in dropped)
    assert any("conftest.py" in d for d in dropped)


def test_extract_patch_no_changes(tmp_path):
    repo = tmp_path / "repo"
    base = _init_repo(repo, {"app.py": "x = 1\n"})

    patch, dropped = solve._extract_patch(repo, base)

    assert patch.strip() == ""
    assert dropped == []


# ── solve_instance — natural exit ─────────────────────────────────────────────

def test_solve_instance_natural_exit_produces_valid_patch(
    fake_ollama, patchable_chat, tmp_path,
):
    """Model: read → edit → done. Result should be exit_reason=natural with a
    well-formed unified diff against base_commit."""
    repo = tmp_path / "repo"
    base = _init_repo(repo, {"app.py": 'def hello():\n    return "wrong"\n'})

    fake_ollama([
        make_response("reading", tool_calls=[make_tool_call("read_file", path="app.py")]),
        make_response("fixing", tool_calls=[make_tool_call(
            "edit_file", path="app.py", old_str='"wrong"', new_str='"right"',
        )]),
        make_response("done — flipped wrong→right"),
    ])

    instance = {
        "instance_id": "test__simple-1",
        "repo": "test/simple",
        "base_commit": base,
        "problem_statement": "hello() returns the wrong word",
        "hints_text": "",
    }
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=10, timeout_s=30,
    )

    assert result.instance_id == "test__simple-1"
    assert result.exit_reason == "natural"
    assert "+++ b/app.py" in result.model_patch
    assert '+    return "right"' in result.model_patch
    assert result.dropped_test_edits == []
    assert result.tool_calls_used >= 2  # read_file + edit_file


def test_solve_instance_drops_test_edits_from_patch(
    fake_ollama, patchable_chat, tmp_path,
):
    """If the model edits a test file despite the prompt, the diff should
    exclude it and the dropped path should be reported."""
    repo = tmp_path / "repo"
    base = _init_repo(repo, {
        "src/app.py": "x = 1\n",
        "tests/test_app.py": "def test_x(): assert True\n",
    })

    fake_ollama([
        make_response("editing source", tool_calls=[make_tool_call(
            "edit_file", path="src/app.py", old_str="x = 1", new_str="x = 2",
        )]),
        make_response("editing test", tool_calls=[make_tool_call(
            "edit_file", path="tests/test_app.py",
            old_str="assert True", new_str="assert x == 2",
        )]),
        make_response("done"),
    ])

    instance = {
        "instance_id": "test__drop-1",
        "repo": "test/drop",
        "base_commit": base,
        "problem_statement": "make x equal 2",
        "hints_text": "",
    }
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=10, timeout_s=30,
    )

    assert "src/app.py" in result.model_patch
    assert "test_app.py" not in result.model_patch
    assert any("test_app.py" in d for d in result.dropped_test_edits)


def test_solve_instance_runs_locator_when_env_set(
    fake_ollama, patchable_chat, monkeypatch, tmp_path,
):
    """With FARCODE_SWE_LOCATOR=1 the locator should run, inject a suspect
    section into the user prompt, and export FARCODE_REPOMAP_SEEDS."""
    monkeypatch.setenv("FARCODE_SWE_LOCATOR", "1")
    monkeypatch.setenv("FARCODE_DISABLE_MEMORY", "1")
    monkeypatch.setenv("FARCODE_DISABLE_CODER_MD", "1")
    monkeypatch.setenv("FARCODE_DISABLE_REPOMAP", "1")
    monkeypatch.delenv("FARCODE_REPOMAP_SEEDS", raising=False)

    repo = tmp_path / "repo"
    base = _init_repo(repo, {
        "src/special_module.py": "def special_function():\n    return 1\n",
    })

    fake = fake_ollama([make_response("done — no changes needed")])

    instance = {
        "instance_id": "test__loc-1",
        "repo": "test/loc",
        "base_commit": base,
        "problem_statement": (
            "The `special_function` in src/special_module.py is buggy."
        ),
        "hints_text": "",
    }
    solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=10, timeout_s=30,
    )

    # The user message that reached the model should carry the suspect
    # section header.
    last_user = next(
        (m for m in reversed(fake.calls[0]["messages"]) if m.get("role") == "user"),
        None,
    )
    assert last_user is not None
    assert "Suspected locations" in last_user["content"]
    assert "src/special_module.py" in last_user["content"]
    # And the env var should have been exported for downstream focused-repomap
    assert "src/special_module.py" in os.environ.get("FARCODE_REPOMAP_SEEDS", "")


def test_solve_instance_skips_locator_when_env_unset(
    fake_ollama, patchable_chat, monkeypatch, tmp_path,
):
    """Without FARCODE_SWE_LOCATOR=1, no suspect section should be injected
    even if the problem statement contains explicit anchors."""
    monkeypatch.delenv("FARCODE_SWE_LOCATOR", raising=False)
    monkeypatch.setenv("FARCODE_DISABLE_MEMORY", "1")
    monkeypatch.setenv("FARCODE_DISABLE_CODER_MD", "1")
    monkeypatch.setenv("FARCODE_DISABLE_REPOMAP", "1")

    repo = tmp_path / "repo"
    base = _init_repo(repo, {"src/foo.py": "x = 1\n"})

    fake = fake_ollama([make_response("done")])

    instance = {
        "instance_id": "test__noloc-1",
        "repo": "test/noloc",
        "base_commit": base,
        "problem_statement": "Bug in src/foo.py",
        "hints_text": "",
    }
    solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=10, timeout_s=30,
    )

    last_user = next(
        (m for m in reversed(fake.calls[0]["messages"]) if m.get("role") == "user"),
        None,
    )
    assert last_user is not None
    assert "Suspected locations" not in last_user["content"]


def test_syntax_gate_retries_then_succeeds(
    fake_ollama, patchable_chat, monkeypatch, tmp_path,
):
    """Model edits a file into a syntax error, says done; gate fires once,
    model fixes it on the bonus pass, exit_reason stays natural."""
    monkeypatch.setenv("FARCODE_SWE_SYNTAX_GATE", "1")
    repo = tmp_path / "repo"
    base = _init_repo(repo, {"app.py": "x = 1\n"})

    fake_ollama([
        # First pass: write a broken file then "done"
        make_response("editing", tool_calls=[make_tool_call(
            "write_file", path="app.py", content="def foo(:\n    pass\n",
        )]),
        make_response("done"),
        # Bonus pass triggered by gate: fix the syntax
        make_response("fixing", tool_calls=[make_tool_call(
            "write_file", path="app.py", content="def foo():\n    pass\n",
        )]),
        make_response("done"),
    ])

    instance = {
        "instance_id": "test__gate-ok",
        "repo": "test/gate",
        "base_commit": base,
        "problem_statement": "fix foo",
        "hints_text": "",
    }
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=20, timeout_s=30,
    )

    assert result.exit_reason == "natural"
    assert "+def foo():" in result.model_patch
    # Confirm the workspace file is the corrected version
    assert (repo / "app.py").read_text(encoding="utf-8") == "def foo():\n    pass\n"


def test_syntax_gate_marks_syntax_broken_when_unfixable(
    fake_ollama, patchable_chat, monkeypatch, tmp_path,
):
    """Model leaves the file broken even after the gate's bonus pass —
    exit_reason becomes syntax_broken so the runlog tells us why."""
    monkeypatch.setenv("FARCODE_SWE_SYNTAX_GATE", "1")
    repo = tmp_path / "repo"
    base = _init_repo(repo, {"app.py": "x = 1\n"})

    broken = lambda label: make_response(
        f"writing {label}", tool_calls=[make_tool_call(
            "write_file", path="app.py", content="def foo(:\n    pass\n",
        )],
    )

    fake_ollama([
        broken("v1"),
        make_response("first done"),
        # Bonus pass: still broken
        broken("v2"),
        make_response("second done"),
    ])

    instance = {
        "instance_id": "test__gate-bad",
        "repo": "test/gate",
        "base_commit": base,
        "problem_statement": "fix foo",
        "hints_text": "",
    }
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=20, timeout_s=30,
    )

    assert result.exit_reason == "syntax_broken"


def test_syntax_gate_skipped_when_env_unset(
    fake_ollama, patchable_chat, monkeypatch, tmp_path,
):
    """Without FARCODE_SWE_SYNTAX_GATE=1 the broken patch passes through."""
    monkeypatch.delenv("FARCODE_SWE_SYNTAX_GATE", raising=False)
    repo = tmp_path / "repo"
    base = _init_repo(repo, {"app.py": "x = 1\n"})

    fake_ollama([
        make_response("editing", tool_calls=[make_tool_call(
            "write_file", path="app.py", content="def foo(:\n    pass\n",
        )]),
        make_response("done"),
    ])

    instance = {
        "instance_id": "test__gate-off",
        "repo": "test/gate",
        "base_commit": base,
        "problem_statement": "x",
        "hints_text": "",
    }
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=20, timeout_s=30,
    )

    # exit reason stays natural — gate didn't run
    assert result.exit_reason == "natural"


def test_solve_instance_no_edits_classified_correctly(
    fake_ollama, patchable_chat, tmp_path,
):
    """Model finishes without making any edits → exit_reason=no_edits."""
    repo = tmp_path / "repo"
    base = _init_repo(repo, {"app.py": "x = 1\n"})

    fake_ollama([
        make_response("looked but didn't change anything"),
    ])

    instance = {
        "instance_id": "test__noedit-1",
        "repo": "test/noedit",
        "base_commit": base,
        "problem_statement": "review only",
        "hints_text": "",
    }
    result = solve.solve_instance(
        instance, repo,
        model="m", num_ctx=8192, num_predict=512,
        max_tools=10, timeout_s=30,
    )

    assert result.exit_reason == "no_edits"
    assert result.model_patch.strip() == ""
