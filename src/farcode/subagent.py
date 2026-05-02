"""Read-only exploration sub-agent.

A subagent runs an isolated agent loop with a restricted tool set
(read_file, list_directory, search_in_files, recall_code, recall_memory).
Its purpose is to keep noisy investigation chatter out of the parent
agent's history while still surfacing the conclusions.

Constraints:
- Cannot call ``explore_subagent`` itself (depth cap = 1, enforced via
  thread-local).
- Cannot mutate state — write/edit/replace/create/run_bash/save_memory are
  not exposed in the subagent's schema and any attempt to call them is
  refused at execution time.
- Capped at ``SUBAGENT_MAX_ITERS`` tool calls before being forced to
  return.
"""

from __future__ import annotations

import os
import threading

from .client import (
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    build_subagent_system_message,
    call_nonstream,
)
from .tools import TOOL_SCHEMAS, execute_tool

READ_ONLY_TOOL_NAMES = frozenset({
    "read_file",
    "list_directory",
    "search_in_files",
    "recall_code",
    "recall_memory",
})

SUBAGENT_MAX_ITERS = 8

_parent_model: str = ""

# In-memory result cache. Key: (question_norm, focus_area_norm). Value:
# (final_text, tool_call_count). Cleared whenever the chat session restarts
# (chat.run_chat calls clear_cache() on /clear and /resume) so stale results
# never leak between unrelated investigations.
_result_cache: dict[tuple[str, str], tuple[str, int]] = {}


def clear_cache() -> None:
    _result_cache.clear()


def bind_parent_model(model: str) -> None:
    """Record the main agent's model so the subagent tool can inherit it."""
    global _parent_model
    _parent_model = model


def get_parent_model() -> str:
    return _parent_model


_depth_state = threading.local()


def _depth() -> int:
    return getattr(_depth_state, "depth", 0)


def _set_depth(value: int) -> None:
    _depth_state.depth = value


def get_subagent_tools() -> list[dict]:
    """Return TOOL_SCHEMAS filtered to the read-only subset, in original order."""
    return [
        s for s in TOOL_SCHEMAS
        if s.get("function", {}).get("name") in READ_ONLY_TOOL_NAMES
    ]


def get_subagent_model(parent_model: str) -> str:
    """Pick the subagent's model: env override, else inherit from parent."""
    return os.environ.get("FARCODE_SUBAGENT_MODEL", "").strip() or parent_model


def run_subagent(
    question: str,
    focus_area: str | None,
    parent_model: str,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> tuple[str, int]:
    """Run a subagent and return (final_text, num_tool_calls_made).

    Raises ``RuntimeError`` if invoked from inside another subagent
    (depth cap = 1).
    """
    if _depth() > 0:
        raise RuntimeError(
            "explore_subagent cannot be called from within a subagent (depth cap = 1)"
        )

    cache_key = (
        (question or "").strip().lower(),
        (focus_area or "").strip().lower(),
    )
    cached = _result_cache.get(cache_key)
    if cached is not None:
        text, n_calls = cached
        return f"{text}\n\n[cached]", n_calls

    tools = get_subagent_tools()
    model = get_subagent_model(parent_model)

    user_content = question.strip()
    if focus_area:
        user_content = f"Focus area: {focus_area.strip()}\n\nQuestion: {user_content}"

    messages: list[dict] = [
        build_subagent_system_message(num_ctx),
        {"role": "user", "content": user_content},
    ]

    tool_call_count = 0

    _set_depth(_depth() + 1)
    try:
        for _ in range(SUBAGENT_MAX_ITERS + 1):
            try:
                response = call_nonstream(
                    messages, model, tools=tools,
                    num_ctx=num_ctx, num_predict=num_predict,
                )
            except Exception as e:
                return f"Subagent error: {e}", tool_call_count

            content = (response.message.content or "").strip()
            calls = list(response.message.tool_calls or [])

            if not calls:
                final = content or "(subagent returned no answer)"
                _maybe_cache(cache_key, final, tool_call_count)
                return final, tool_call_count

            if tool_call_count >= SUBAGENT_MAX_ITERS:
                tail = "\n\n[subagent: tool-call cap reached]" if content else \
                       "(subagent: tool-call cap reached)"
                # Don't cache cap-reached results — they may be incomplete.
                return (content + tail).strip(), tool_call_count

            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": dict(tc.function.arguments),
                        }
                    }
                    for tc in calls
                ],
            })

            for tc in calls:
                name = tc.function.name
                if name not in READ_ONLY_TOOL_NAMES:
                    result = (
                        f"Tool '{name}' is not allowed in subagent context "
                        "(only read-only tools permitted)."
                    )
                else:
                    args = dict(tc.function.arguments)
                    result = execute_tool(name, args)
                tool_call_count += 1
                messages.append({"role": "tool", "content": result})

        return "(subagent: max iterations reached)", tool_call_count
    finally:
        _set_depth(_depth() - 1)


def _maybe_cache(key: tuple[str, str], text: str, n_calls: int) -> None:
    """Cache a successful subagent result. Errors are deliberately skipped."""
    if not text:
        return
    if text.startswith("Subagent error"):
        return
    _result_cache[key] = (text, n_calls)


# ── Patch critique subagent (SWE-bench mode) ──────────────────────────────────
#
# Goal: catch obvious problems before the patch is submitted — wrong location,
# missing the actual contract, or breaking nearby code. The subagent sees the
# diff and the modified files (read-only) and reports up to 3 risks. With a 4B
# model self-critiquing, we expect modest lift; gated behind FARCODE_SWE_CRITIQUE
# so it can be measured independently in the ablation matrix.

_CRITIQUE_SYSTEM = (
    "You are a code-review subagent. Read the patch (a unified diff) and the "
    "modified files, plus the original problem statement. Your only job is to "
    "list up to 3 SPECIFIC concerns about whether the patch will fix the "
    "issue and not break other tests.\n\n"
    "## Rules\n"
    "- Do NOT propose a new fix. Do NOT suggest refactors.\n"
    "- Each concern must cite a file path and a one-sentence reason.\n"
    "- If the patch looks correct, reply: 'No major concerns.'\n"
    "- Maximum 200 words. Be concrete, not generic.\n"
    "- Do not call any tools other than read_file and search_in_files."
)


def run_critique_subagent(
    diff: str,
    problem_statement: str,
    parent_model: str,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> tuple[str, int]:
    """Spawn a read-only subagent to review a patch. Returns (text, n_calls).

    The diff is truncated at 4 KB and the problem statement at 2 KB so the
    subagent's first prompt fits comfortably. Subsequent ``read_file`` calls
    let it pull in surrounding context for any file mentioned in the diff.
    """
    if _depth() > 0:
        return "(critique skipped: already in subagent)", 0

    diff_clip = diff[:4096] + ("\n[... diff truncated ...]" if len(diff) > 4096 else "")
    ps_clip = (problem_statement or "").strip()
    if len(ps_clip) > 2048:
        ps_clip = ps_clip[:2048] + "\n[... truncated ...]"

    user_content = (
        "## Problem\n"
        f"{ps_clip}\n\n"
        "## Proposed patch (unified diff)\n"
        f"```diff\n{diff_clip}\n```\n\n"
        "List up to 3 concerns or reply 'No major concerns.'"
    )

    tools = get_subagent_tools()
    model = get_subagent_model(parent_model)
    messages: list[dict] = [
        {"role": "system", "content": _CRITIQUE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    tool_call_count = 0

    _set_depth(_depth() + 1)
    try:
        # Tighter cap than the exploration subagent — critique should be a
        # few targeted reads, not a full investigation.
        for _ in range(5):
            try:
                response = call_nonstream(
                    messages, model, tools=tools,
                    num_ctx=num_ctx, num_predict=num_predict,
                )
            except Exception as e:
                return f"(critique error: {e})", tool_call_count

            content = (response.message.content or "").strip()
            calls = list(response.message.tool_calls or [])

            if not calls:
                return content or "No major concerns.", tool_call_count

            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": dict(tc.function.arguments),
                        }
                    }
                    for tc in calls
                ],
            })
            for tc in calls:
                name = tc.function.name
                if name not in READ_ONLY_TOOL_NAMES:
                    result = (
                        f"Tool '{name}' is not allowed in critique subagent."
                    )
                else:
                    args = dict(tc.function.arguments)
                    result = execute_tool(name, args)
                tool_call_count += 1
                messages.append({"role": "tool", "content": result})

        return "(critique: iteration cap reached)", tool_call_count
    finally:
        _set_depth(_depth() - 1)
