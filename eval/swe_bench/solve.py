"""Single-instance SWE-bench driver.

Run as a subprocess from `run.py`:

    python -m eval.swe_bench.solve \\
        --instance-json <path> --workspace <path> \\
        --model qwen3.5:4b --output <path> \\
        [--num-ctx 65536] [--num-predict 4096] \\
        [--max-tools 60] [--timeout 900]

Any FARCODE_DISABLE_* env vars for ablations are set by the parent process
and read by farcode/{client,tools}.py via `_env.env_on`. solve.py just runs.

Output is a JSON file with fields:
    instance_id, model_patch, model_name, exit_reason, turns_used,
    tool_calls_used, elapsed_s, dropped_test_edits

`exit_reason` is one of:
    natural | stuck | max_tools | timeout | error:<type> | no_edits

Note on `_run_agent_turn` semantics: chat.py's `_run_agent_turn` is one full
agentic *exchange* — it loops internally calling the LLM and running tools
until the LLM stops emitting tool_calls. We only call it once (one user
prompt = the SWE-bench problem). Stuck/budget enforcement comes from raising
a sentinel exception inside the wrapped `chat.execute_tool`, which bubbles
up to `_run_agent_turn`'s generic `except Exception` (line 719) and ends
the turn cleanly. We then read state flags to classify the exit_reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

MUTATING_TOOLS = frozenset({"edit_file", "write_file", "replace_lines", "create_file"})

# Read-loop detection thresholds. Tuned conservatively: real solves on
# qwen3.5:4b legitimately re-read the same file 2-3 times. The 5th identical
# read is when "model is stuck checking the same content". Once suppressed
# we *don't* re-trigger on the 6th — the model is told once and we move on.
READ_LOOP_THRESHOLD = 5
# How many tool calls without a mutating call before nudging the model to
# either edit or stop. 25 is roughly half a typical 60-tool budget — enough
# room to investigate, not enough to grind to budget exhaustion all-read.
ALL_READ_NUDGE_THRESHOLD = 25

TEST_PATHSPECS = (
    ":(exclude)**/test_*.py",
    ":(exclude)**/tests/**",
    ":(exclude)**/*_test.py",
    ":(exclude)**/conftest.py",
    ":(exclude)test_*.py",
    ":(exclude)tests/**",
    ":(exclude)*_test.py",
    ":(exclude)conftest.py",
    # Lock files: never part of a SWE-bench fix. They can leak into the
    # workspace if a tool (e.g. `uv` running in the parent) accidentally
    # writes one. Excluding them keeps the diff focused on real source edits.
    ":(exclude)uv.lock",
    ":(exclude)poetry.lock",
    ":(exclude)Pipfile.lock",
    ":(exclude)package-lock.json",
    ":(exclude)yarn.lock",
)


class _SolveAbort(Exception):
    """Sentinel raised from the wrapped execute_tool to bail the inner loop.
    Caught by `_run_agent_turn`'s generic `except Exception` (chat.py:719)."""


@dataclass
class SolveResult:
    instance_id: str
    model_patch: str
    model_name: str
    exit_reason: str
    turns_used: int
    tool_calls_used: int
    elapsed_s: float
    dropped_test_edits: list = field(default_factory=list)


@dataclass
class _SolveState:
    max_tools: int
    timeout_s: int
    started_at: float
    recent_edits: list = field(default_factory=list)
    tool_calls_used: int = 0
    inner_turns: int = 0
    stuck: bool = False
    max_tools_reached: bool = False
    timed_out: bool = False
    # Read-loop detection: count identical (path, offset, limit) read_file
    # invocations. Once we suppress a read it stays suppressed (kept at the
    # threshold value) so we don't re-fire the same warning every turn.
    read_counts: dict = field(default_factory=dict)
    # Index of the last mutating tool call (relative to tool_calls_used).
    # 0 means "never mutated yet". Used by the all-read nudge.
    last_mutating_idx: int = 0
    nudge_fired: bool = False
    # Files mutated by the agent this run, used by the post-loop syntax gate
    # (see _check_syntax_gate). Tracked even when the tool fails — the gate
    # re-reads each file and only flags genuinely broken parses.
    modified_files: set = field(default_factory=set)
    # Set to True once the syntax gate has triggered a single bonus retry —
    # prevents an infinite syntax-fix loop. After one retry the patch is
    # submitted as-is and exit_reason becomes "syntax_broken".
    syntax_retry_used: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def _args_hash(args: dict) -> str:
    return hashlib.md5(
        json.dumps(args, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _consecutive_tail(seq: list, value) -> int:
    n = 0
    for item in reversed(seq):
        if item == value:
            n += 1
        else:
            break
    return n


def _build_wrapped_execute_tool(state: _SolveState, original):
    def wrapped(name: str, args: dict) -> str:
        # nudge_tail is set inside the lock and appended after we run the
        # underlying tool — keeps the slow original() call outside the lock.
        nudge_tail = ""
        with state.lock:
            if time.monotonic() - state.started_at > state.timeout_s:
                state.timed_out = True
                raise _SolveAbort("timeout")

            if name in MUTATING_TOOLS:
                key = (name, _args_hash(args))
                tail = _consecutive_tail(state.recent_edits, key)
                state.recent_edits.append(key)
                if tail >= 3:
                    state.stuck = True
                    raise _SolveAbort("stuck:duplicate_edits")
                if tail >= 2:
                    return (
                        "duplicate edit suppressed: this exact tool call has fired "
                        "3 times in a row. Try a different approach instead of "
                        "repeating it."
                    )

            # Read-loop detection: same exact read_file invocation many
            # times is a strong "model is going in circles" signal. We
            # suppress once at the threshold; further repeats just no-op
            # (the count keeps incrementing but we don't re-fire) so the
            # warning isn't spammed every turn.
            if name == "read_file":
                rk = (
                    str(args.get("path", "")),
                    int(args.get("offset") or 0),
                    int(args.get("limit") or 0),
                )
                count = state.read_counts.get(rk, 0) + 1
                state.read_counts[rk] = count
                if count == READ_LOOP_THRESHOLD:
                    state.tool_calls_used += 1
                    if state.tool_calls_used > state.max_tools:
                        state.max_tools_reached = True
                        raise _SolveAbort("max_tools")
                    return (
                        f"read-loop suppressed: this exact read has fired "
                        f"{READ_LOOP_THRESHOLD} times. You already have the file "
                        "content. Edit it now using edit_file or replace_lines."
                    )

            state.tool_calls_used += 1
            if state.tool_calls_used > state.max_tools:
                state.max_tools_reached = True
                raise _SolveAbort("max_tools")

            if name in MUTATING_TOOLS:
                state.last_mutating_idx = state.tool_calls_used
                # Track for the post-loop syntax gate. We add even when the
                # underlying call may fail — re-checking is cheap and we want
                # any path the agent _attempted_ to mutate to be in the set.
                p = args.get("path")
                if isinstance(p, str) and p:
                    state.modified_files.add(p)
            elif (
                not state.nudge_fired
                and state.tool_calls_used - state.last_mutating_idx
                    >= ALL_READ_NUDGE_THRESHOLD
            ):
                # All-read drift nudge: many tool calls without a single
                # mutating one. Fire ONCE; the message is appended to this
                # read's result so the model sees it before its next turn.
                # Wording deliberately gives no escape hatch — "stop and reply"
                # was found in M5 to make the model bail (e.g. django-14999
                # quit with 0 lines edited after the nudge fired).
                state.nudge_fired = True
                gap = state.tool_calls_used - state.last_mutating_idx
                nudge_tail = (
                    f"\n\n[harness nudge: {gap} tool calls since the last edit. "
                    "You have read enough. Edit a file now using edit_file or "
                    "replace_lines.]"
                )
        # original() runs outside the lock so a slow read doesn't serialize
        # subsequent harness work.
        result = original(name, args)
        return result + nudge_tail if nudge_tail else result
    return wrapped


def _build_wrapped_call_nonstream(state: _SolveState, original):
    def wrapped(*args, **kwargs):
        state.inner_turns += 1
        if time.monotonic() - state.started_at > state.timeout_s:
            state.timed_out = True
            raise _SolveAbort("timeout")
        return original(*args, **kwargs)
    return wrapped


def _install_solve_patches(state: _SolveState) -> None:
    from farcode import chat, memory, tasks
    from farcode.tools import set_bash_require_confirm

    set_bash_require_confirm(False)

    chat.execute_tool = _build_wrapped_execute_tool(state, chat.execute_tool)
    chat.call_nonstream = _build_wrapped_call_nonstream(state, chat.call_nonstream)
    chat.print_tool_call = lambda *a, **kw: None
    chat.print_info = lambda *a, **kw: None
    chat.print_error = lambda *a, **kw: None
    chat.render_response = lambda *a, **kw: None
    chat.call_with_thinking = lambda fn, stats: fn()

    memory.append_entry = lambda *a, **kw: None
    tasks.bind([])


_PASS_TO_PASS_SAMPLE_CAP = 10


def _coerce_test_list(raw) -> list[str]:
    """Accept FAIL_TO_PASS / PASS_TO_PASS as either a list[str] (HuggingFace
    `datasets` shape) or a JSON-encoded string (raw JSONL shape). Returns a
    de-duplicated list preserving original order; returns ``[]`` on any
    parse failure so the caller can degrade gracefully."""
    if not raw:
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str):
        try:
            items = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(items, list):
            return []
    else:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if isinstance(x, str) and x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _format_test_contract(instance: dict) -> str:
    """Build the ``## Test contract`` section from FAIL_TO_PASS / PASS_TO_PASS.

    For 4B models, naming the exact tests that must pass is one of the most
    information-dense hints we can give that doesn't leak the gold answer.
    e.g. seeing ``test_Mathematica_Max_Min`` makes it obvious that the fix
    needs Max/Min printers, not edits to a function-mapping dict.

    Returns ``""`` when no test names are available so the caller can omit
    the section entirely (an empty contract header is worse than no header).
    """
    f2p = _coerce_test_list(instance.get("FAIL_TO_PASS"))
    p2p = _coerce_test_list(instance.get("PASS_TO_PASS"))
    if not f2p and not p2p:
        return ""

    lines = ["## Test contract (these are test NAMES — do not read, edit, or run them)"]
    if f2p:
        lines.append("Your fix must make these tests pass:")
        for name in f2p[:20]:  # cap so a giant list doesn't blow the prompt
            lines.append(f"  - {name}")
        if len(f2p) > 20:
            lines.append(f"  ... (+{len(f2p) - 20} more)")
    if p2p:
        # Deterministic per-instance sampling so the same instance always sees
        # the same hint set across runs (lets us A/B without random-seed noise).
        import random as _random
        rng = _random.Random(instance.get("instance_id", ""))
        sample_n = min(_PASS_TO_PASS_SAMPLE_CAP, len(p2p))
        sample = rng.sample(p2p, sample_n) if sample_n < len(p2p) else list(p2p)
        lines.append(
            f"It must NOT break these previously-passing tests "
            f"(sample of {sample_n} from {len(p2p)} total):"
        )
        for name in sample:
            lines.append(f"  - {name}")
    return "\n".join(lines)


def _format_user_prompt(instance: dict, suspect_section: str = "") -> str:
    problem = instance["problem_statement"].strip()
    hints = (instance.get("hints_text") or "").strip()
    repo = instance.get("repo", "the repository")

    sections = [
        f"You are working in the {repo} repository at the current directory.",
        "There is a bug or feature request described below. Modify the source",
        "files (NOT the test files) to make the issue resolved.",
        "",
        "Rules:",
        "- Use read_file before editing. Make edits with edit_file / replace_lines / write_file.",
        "- Do NOT modify any file under tests/, conftest.py, or *_test.py — the harness",
        "  applies its own test patch separately.",
        "- Do NOT call run_bash to execute pytest. The harness will run the tests.",
        "- When you believe the issue is fixed, stop calling tools and reply with a",
        "  one-line summary of what you changed.",
    ]
    # The locator-emitted suspect section sits between rules and issue text so
    # the model sees concrete file targets before parsing the prose. Suppressed
    # entirely when locator returned no suspects (don't inject a "no
    # locations found" header — that's worse than nothing).
    if suspect_section:
        sections.extend(["", suspect_section])
    # Test contract: gated, sourced from instance dict's FAIL_TO_PASS /
    # PASS_TO_PASS. See _format_test_contract for the rationale.
    if os.environ.get("FARCODE_SWE_TEST_CONTRACT") == "1":
        contract = _format_test_contract(instance)
        if contract:
            sections.extend(["", contract])
    sections.extend(["", "Issue:", problem])
    if hints:
        sections.extend(["", "Hints:", hints])
    return "\n".join(sections)


def _extract_patch(workspace: Path, base_commit: str) -> tuple[str, list[str]]:
    subprocess.run(
        ["git", "-C", str(workspace), "add", "-A"],
        check=True, capture_output=True, text=True,
    )
    full_diff = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--cached", "--binary", base_commit],
        check=True, capture_output=True, text=True,
    ).stdout
    filtered_diff = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--cached", "--binary", base_commit,
         "--", *TEST_PATHSPECS],
        check=True, capture_output=True, text=True,
    ).stdout

    dropped: list[str] = []
    if full_diff != filtered_diff:
        full_files = set(_files_in_diff(full_diff))
        kept_files = set(_files_in_diff(filtered_diff))
        dropped = sorted(full_files - kept_files)

    subprocess.run(
        ["git", "-C", str(workspace), "reset"],
        check=True, capture_output=True, text=True,
    )
    return filtered_diff, dropped


def _files_in_diff(diff: str) -> list[str]:
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[len("+++ b/"):])
    return files


def _classify_exit(state: _SolveState, last_message: dict | None) -> str:
    if state.timed_out:
        return "timeout"
    if state.stuck:
        return "stuck"
    if state.max_tools_reached:
        return "max_tools"
    if last_message and last_message.get("role") == "assistant" and not last_message.get("tool_calls"):
        return "natural"
    return "unknown"


def _check_syntax_errors(workspace: Path, paths: list[str]) -> list[tuple[str, int, str]]:
    """Return ``[(rel_path, lineno, msg)]`` for every .py file in ``paths`` that
    no longer parses. Skips files that don't exist (model may have created
    then deleted), aren't .py, or are unreadable.

    Mirrors ``src/farcode/tools.py:_check_syntax`` but operates on the gate's
    set of paths in one batch and returns structured info for the prompt.
    """
    import ast
    errors: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for raw in paths:
        rel = raw.replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        if not rel.endswith(".py") and not rel.endswith(".pyi"):
            continue
        p = workspace / rel
        if not p.is_file():
            continue
        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            ast.parse(source, filename=str(p))
        except SyntaxError as e:
            errors.append((rel, e.lineno or 0, e.msg or "syntax error"))
    return errors


def _format_syntax_gate_message(errors: list[tuple[str, int, str]]) -> str:
    """Build the synthetic user message handed to the agent for one bonus
    fix-up pass. Specific enough to anchor the model on the broken hunk."""
    lines = [
        "Your patch contains syntax errors and will fail to apply. "
        "Re-read the affected file(s) and fix them now, then stop calling tools."
    ]
    for rel, lineno, msg in errors[:5]:  # cap so the prompt stays small
        lines.append(f"  - {rel}:{lineno}: {msg}")
    return "\n".join(lines)


def solve_instance(
    instance: dict,
    workspace: Path,
    *,
    model: str,
    num_ctx: int,
    num_predict: int,
    max_tools: int,
    timeout_s: int,
) -> SolveResult:
    instance_id = instance["instance_id"]
    base_commit = instance["base_commit"]

    original_cwd = os.getcwd()
    started = time.monotonic()
    state = _SolveState(
        max_tools=max_tools, timeout_s=timeout_s, started_at=started,
    )

    try:
        os.chdir(workspace)
        _install_solve_patches(state)

        from farcode.chat import _run_agent_turn
        from farcode.client import build_system_messages

        # Run the bug locator BEFORE chdir-affected farcode imports so it
        # reads the workspace fresh. Gated on FARCODE_SWE_LOCATOR so the
        # `full` ablation comparison stays clean.
        suspect_section = ""
        if os.environ.get("FARCODE_SWE_LOCATOR") == "1":
            from . import locator
            try:
                suspects = locator.locate(instance.get("problem_statement", ""), workspace)
            except Exception:
                suspects = []
            if suspects:
                suspect_section = locator.format_section(suspects)
                seeds = locator.seeds_for_repomap(suspects)
                if seeds:
                    # Picked up by farcode.client.build_system_messages → build_focused_repo_map.
                    os.environ["FARCODE_REPOMAP_SEEDS"] = seeds
        # Make the original problem statement available to the critique
        # subagent (which lives inside the farcode process and doesn't
        # otherwise see the instance dict).
        os.environ["FARCODE_SWE_PROBLEM"] = (instance.get("problem_statement") or "")[:8192]

        user_prompt = _format_user_prompt(instance, suspect_section=suspect_section)
        messages = build_system_messages(first_user_message=user_prompt, num_ctx=num_ctx)
        messages.append({"role": "user", "content": user_prompt})

        exit_reason = "unknown"
        try:
            _run_agent_turn(messages, model, num_ctx, num_predict)
        except _SolveAbort:
            pass
        except Exception as e:
            exit_reason = f"error:{type(e).__name__}"

        last = messages[-1] if messages else None
        if exit_reason == "unknown":
            exit_reason = _classify_exit(state, last)

        # Syntax gate: if the agent stopped naturally but left .py files
        # that don't parse, give it ONE bonus pass to fix them. This is the
        # cheapest fix for the post-M5 finding that 3 of 5 always-failing
        # instances submitted syntactically-broken patches that couldn't
        # even import. Bounded to one retry to avoid runaway.
        if (
            exit_reason == "natural"
            and os.environ.get("FARCODE_SWE_SYNTAX_GATE") == "1"
            and state.modified_files
            and not state.syntax_retry_used
        ):
            errors = _check_syntax_errors(workspace, sorted(state.modified_files))
            if errors:
                state.syntax_retry_used = True
                messages.append({
                    "role": "user",
                    "content": _format_syntax_gate_message(errors),
                })
                try:
                    _run_agent_turn(messages, model, num_ctx, num_predict)
                except _SolveAbort:
                    pass
                except Exception as e:
                    exit_reason = f"error:{type(e).__name__}"
                # Re-classify: if still broken after retry, mark distinctly
                # so we can attribute scoring failures to syntax in runlogs.
                last = messages[-1] if messages else None
                exit_reason = _classify_exit(state, last)
                still_broken = _check_syntax_errors(workspace, sorted(state.modified_files))
                if still_broken and exit_reason == "natural":
                    exit_reason = "syntax_broken"

        patch, dropped = _extract_patch(workspace, base_commit)
        if not patch.strip() and exit_reason == "natural":
            exit_reason = "no_edits"

        return SolveResult(
            instance_id=instance_id,
            model_patch=patch,
            model_name=model,
            exit_reason=exit_reason,
            turns_used=state.inner_turns,
            tool_calls_used=state.tool_calls_used,
            elapsed_s=time.monotonic() - started,
            dropped_test_edits=dropped,
        )
    finally:
        os.chdir(original_cwd)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve one SWE-bench instance with farcode.")
    p.add_argument("--instance-json", required=True, type=Path,
                   help="Path to a JSON file with one SWE-bench instance.")
    p.add_argument("--workspace", required=True, type=Path,
                   help="Path to the prepared git worktree (already at base_commit).")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the SolveResult JSON.")
    p.add_argument("--model", default="qwen3.5:4b")
    p.add_argument("--num-ctx", type=int, default=65536)
    p.add_argument("--num-predict", type=int, default=4096)
    p.add_argument("--max-tools", type=int, default=60)
    p.add_argument("--timeout", type=int, default=900)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    instance = json.loads(args.instance_json.read_text(encoding="utf-8"))
    result = solve_instance(
        instance, args.workspace,
        model=args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        max_tools=args.max_tools,
        timeout_s=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
