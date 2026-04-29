import os
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import ollama

DEFAULT_NUM_CTX = int(os.environ.get("FARCODE_NUM_CTX", "65536"))
DEFAULT_NUM_PREDICT = int(os.environ.get("FARCODE_NUM_PREDICT", "4096"))


def _system_prompt(num_ctx: int = DEFAULT_NUM_CTX) -> str:
    ctx_k = max(1, num_ctx // 1024)
    return (
        "You are a coding assistant on Windows. Use the provided tools to read, "
        "edit, and run code in the user's project.\n\n"
        "## Workflow\n"
        "1. For multi-step tasks, write a one-line plan in plain text before calling tools.\n"
        "2. ALWAYS call `read_file` before editing. Never guess file contents.\n"
        "3. After each tool result, write a one-line summary — never echo the output.\n"
        "4. After completing a non-trivial sub-goal, call `save_memory`.\n\n"
        "## Editing\n"
        "Pick the smallest tool that fits the change:\n"
        "- `edit_file` — small targeted string replace (1–5 lines)\n"
        "- `replace_lines` — multi-line block change when you know the line range\n"
        "- `write_file` — full rewrite, or when other edits keep failing\n\n"
        "## Tasks\n"
        "For requests that take 3+ steps, call `task_create` once per step "
        "before starting work, then `task_update` to mark each step "
        "in_progress and completed as you progress.\n\n"
        "## Exploration\n"
        "When the user asks 'how does X work', 'where is Y', or 'trace this "
        "feature' and answering needs reading 5+ files, prefer "
        "`explore_subagent(question, focus_area?)` over reading the files "
        "directly — it isolates the investigation in a sub-loop and returns a "
        "concise summary, keeping this conversation's context clean.\n\n"
        "## Documentation lookup\n"
        "If `fetch_doc` is available (only when the user passed `--allow-web`), "
        "use it to look up package versions, summaries, and licenses on PyPI / "
        "npm / crates.io / pkg.go.dev rather than guessing. Do NOT call it if "
        "it returns 'disabled' — that means web access is off.\n\n"
        f"Context window is {ctx_k}K tokens. Keep replies short."
    )


SYSTEM_PROMPT = _system_prompt()


@dataclass
class StreamStats:
    start_time: float = field(default_factory=time.monotonic)
    token_count: int = 0
    input_tokens: int = 0
    ctx_size: int = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time


def _ollama_options(num_ctx: int, num_predict: int) -> dict:
    return {"num_ctx": num_ctx, "num_predict": num_predict}


def stream_chat(
    messages: list[dict],
    model: str,
    stats: StreamStats,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> Generator[str, None, None]:
    """Streaming chat — yields text chunks. Used by the `ask` command."""
    stats.ctx_size = num_ctx
    response = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options=_ollama_options(num_ctx, num_predict),
    )
    for chunk in response:
        thinking = getattr(chunk.message, "thinking", None)
        content = chunk.message.content
        if thinking:
            yield thinking
        if content:
            stats.token_count += 1
            yield content
        if chunk.done:
            if chunk.eval_count:
                stats.token_count = chunk.eval_count
            prompt_eval = getattr(chunk, "prompt_eval_count", None)
            if prompt_eval:
                stats.input_tokens = prompt_eval


def call_nonstream(
    messages: list[dict],
    model: str,
    tools: list[dict] | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> Any:
    """Non-streaming chat — returns the full ChatResponse. Used by the agent loop.

    Recovery ladder when the native tool-calling path 500s (a known Ollama bug
    with some smaller models that emit malformed tool XML/JSON):
      1. Retry once with ``format=<json_schema>`` constraining output to a
         single ``{name, arguments}`` payload. The grammar makes valid output
         the *only* possibility, which 4B models handle far more reliably.
      2. If that also 500s, fall back to plain-text chat with no tools.
    """
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "options": _ollama_options(num_ctx, num_predict),
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return ollama.chat(**kwargs)
    except ollama.ResponseError as e:
        if e.status_code != 500 or not tools:
            raise

        retry = _grammar_constrained_tool_call(
            messages, model, tools, num_ctx, num_predict
        )
        if retry is not None:
            return retry

        kwargs.pop("tools", None)
        return ollama.chat(**kwargs)


# ── Grammar-constrained tool-call retry ───────────────────────────────────────

class _SyntheticFunction:
    def __init__(self, name: str, arguments: dict) -> None:
        self.name = name
        self.arguments = arguments


class _SyntheticToolCall:
    def __init__(self, name: str, arguments: dict) -> None:
        self.function = _SyntheticFunction(name, arguments)


class _SyntheticMessage:
    def __init__(self, content: str, tool_calls: list) -> None:
        self.content = content
        self.tool_calls = tool_calls
        self.thinking = None


class _SyntheticResponse:
    """Mimics ``ollama.ChatResponse`` enough for the agent loop's needs."""

    def __init__(
        self,
        content: str,
        tool_calls: list,
        eval_count: int = 0,
        prompt_eval_count: int = 0,
    ) -> None:
        self.message = _SyntheticMessage(content, tool_calls)
        self.eval_count = eval_count
        self.prompt_eval_count = prompt_eval_count
        self.done = True


def _tool_names(tools: list[dict]) -> list[str]:
    return [t.get("function", {}).get("name") for t in tools if t.get("function")]


def _grammar_schema(tools: list[dict]) -> dict:
    """JSON schema constraining the model to one of:
       - {"tool_call": {"name": <enum>, "arguments": <object>}}
       - {"answer": <string>}
    """
    return {
        "type": "object",
        "properties": {
            "tool_call": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": _tool_names(tools)},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
            "answer": {"type": "string"},
        },
    }


def _grammar_prompt_suffix(tools: list[dict]) -> str:
    names = ", ".join(_tool_names(tools))
    return (
        "\n\n[The previous response was malformed. Reply with ONLY a JSON object "
        f'matching this shape: either {{"tool_call": {{"name": "<one of: {names}>", '
        '"arguments": {...}}} to call a tool, or {"answer": "..."} to give a '
        "final reply. No prose, no markdown.]"
    )


def _grammar_constrained_tool_call(
    messages: list[dict],
    model: str,
    tools: list[dict],
    num_ctx: int,
    num_predict: int,
) -> Any:
    """Retry the failed call with a JSON-schema constraint. Returns a
    ``_SyntheticResponse`` on success or ``None`` if the retry also failed."""
    import json as _json

    nudge = list(messages)
    if nudge and nudge[-1].get("role") != "system":
        nudge = nudge + [
            {"role": "user", "content": _grammar_prompt_suffix(tools).strip()}
        ]
    try:
        response = ollama.chat(
            model=model,
            messages=nudge,
            options=_ollama_options(num_ctx, num_predict),
            format=_grammar_schema(tools),
        )
    except Exception:
        return None

    content = (response.message.content or "").strip()
    if not content:
        return None
    try:
        parsed = _json.loads(content)
    except _json.JSONDecodeError:
        return None

    if isinstance(parsed, dict) and "tool_call" in parsed:
        tc = parsed["tool_call"] or {}
        name = tc.get("name")
        args = tc.get("arguments") or {}
        if name in _tool_names(tools) and isinstance(args, dict):
            return _SyntheticResponse(
                content="",
                tool_calls=[_SyntheticToolCall(name, args)],
                eval_count=getattr(response, "eval_count", 0) or 0,
                prompt_eval_count=getattr(response, "prompt_eval_count", 0) or 0,
            )
    if isinstance(parsed, dict) and "answer" in parsed:
        return _SyntheticResponse(
            content=str(parsed.get("answer", "")),
            tool_calls=[],
            eval_count=getattr(response, "eval_count", 0) or 0,
            prompt_eval_count=getattr(response, "prompt_eval_count", 0) or 0,
        )
    return None


def stream_agent_iter(
    messages: list[dict],
    model: str,
    stats: StreamStats,
    tools: list[dict] | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> tuple[Generator[str, None, None], dict]:
    """
    Streaming agent call with optional tools.
    Returns (text_chunk_generator, result_holder).
    After exhausting the generator, result_holder['response'] holds the final ChatResponse.
    """
    result: dict = {}
    stats.ctx_size = num_ctx

    def _gen() -> Generator[str, None, None]:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": _ollama_options(num_ctx, num_predict),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["think"] = False  # thinking mode breaks structured tool_calls in qwen3-class models
        content_buf = ""
        for chunk in ollama.chat(**kwargs):
            thinking = getattr(chunk.message, "thinking", None)
            content = chunk.message.content
            if thinking:
                yield thinking
            if content:
                content_buf += content
                stats.token_count += 1
                yield content
            if chunk.done:
                if chunk.eval_count:
                    stats.token_count = chunk.eval_count
                prompt_eval = getattr(chunk, "prompt_eval_count", None)
                if prompt_eval:
                    stats.input_tokens = prompt_eval
                result["response"] = chunk
                result["content"] = content_buf  # actual response only (excludes thinking)

    return _gen(), result


def build_system_messages(
    first_user_message: str | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> list[dict]:
    """Build the system message, with project context injected.

    Order (later items appear later in the prompt, where the model attends most):
      1. Base system prompt
      2. CODER.md project rules
      3. Codebase facts (language, package manager, test runner, entry points)
      4. Repo map (top-level definitions, ranked & token-budgeted)
      5. Past Work memory (FTS-matched on first user message, else recent)
    """
    parts: list[str] = [_system_prompt(num_ctx)]

    try:
        from .coder_md import load_coder_md
        rules = load_coder_md()
        if rules:
            parts.append(rules)
    except Exception:
        pass

    try:
        from .facts import get_or_build_facts
        facts = get_or_build_facts()
        if facts:
            parts.append(facts)
    except Exception:
        pass

    try:
        from .repomap import build_repo_map
        repo = build_repo_map()
        if repo:
            parts.append(repo)
    except Exception:
        pass

    try:
        from .memory import current_project_path, format_for_prompt, load_recent, search
        project = current_project_path()
        if first_user_message:
            entries = search(first_user_message, top_k=3, project_path=project, scope="project")
            if not entries:
                entries = load_recent(3, project_path=project)
        else:
            entries = load_recent(3, project_path=project)
        past = format_for_prompt(entries)
        if past:
            parts.append(past)
    except Exception:
        pass

    return [{"role": "system", "content": "\n\n".join(parts)}]


def build_subagent_system_message(num_ctx: int = DEFAULT_NUM_CTX) -> dict:
    """Build a focused system prompt for a read-only exploration subagent.

    Includes codebase facts and the repo map (lightweight project context)
    but deliberately omits CODER.md project rules and memory recall, so the
    subagent stays anchored on the immediate question.
    """
    ctx_k = max(1, num_ctx // 1024)
    parts: list[str] = [
        "You are a read-only exploration subagent. Use the provided tools "
        "(read_file, list_directory, search_in_files, recall_code, recall_memory) "
        "to investigate the user's question and return a concise factual summary.\n\n"
        "## Rules\n"
        "- Do not propose changes, refactors, or fixes — only describe what exists.\n"
        "- Cite exact file paths and line numbers where possible.\n"
        "- Stop once you have enough to answer; do not over-explore.\n"
        "- Keep the final answer under 600 words.\n\n"
        f"Context window is {ctx_k}K tokens."
    ]

    try:
        from .facts import get_or_build_facts
        facts = get_or_build_facts()
        if facts:
            parts.append(facts)
    except Exception:
        pass

    try:
        from .repomap import build_repo_map
        repo = build_repo_map()
        if repo:
            parts.append(repo)
    except Exception:
        pass

    return {"role": "system", "content": "\n\n".join(parts)}


def check_ollama(model: str, num_ctx: int = DEFAULT_NUM_CTX) -> None:
    try:
        models = ollama.list()
        names = [m.model for m in models.models]
        if not any(model in n for n in names):
            available = ", ".join(names) if names else "(none)"
            raise SystemExit(
                f"Model '{model}' not found in Ollama.\n"
                f"Available models: {available}\n"
                f"Pull it with:  ollama pull {model}"
            )
    except ollama.ResponseError as e:
        raise SystemExit(
            f"Ollama error: {e}\nMake sure Ollama is running:  ollama serve"
        ) from e
    except Exception as e:
        if "connect" in str(e).lower() or "connection" in str(e).lower():
            raise SystemExit(
                "Cannot connect to Ollama. Start it with:  ollama serve"
            ) from e
        raise

    if num_ctx > 8192 and not os.environ.get("OLLAMA_KV_CACHE_TYPE"):
        try:
            from .ui import print_info
            print_info(
                f"num_ctx={num_ctx}: KV cache may be large in fp16. "
                "Set OLLAMA_KV_CACHE_TYPE=q8_0 (or q4_0) before `ollama serve` to fit on smaller GPUs."
            )
        except Exception:
            pass
