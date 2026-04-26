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

    If the model returns a 500 error (malformed tool-call XML, a known Ollama
    bug with some smaller models), automatically retries without tools so the
    conversation continues with a plain-text response.
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
        if e.status_code == 500 and tools:
            kwargs.pop("tools", None)
            return ollama.chat(**kwargs)
        raise


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
    """Build the system message, with project-scoped memory injected.

    If `first_user_message` is provided, runs an FTS5 search against memory
    using its terms; otherwise falls back to the most recent project memories.
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
