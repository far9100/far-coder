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
