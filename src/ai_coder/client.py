import time
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import ollama

SYSTEM_PROMPT = (
    "You are an expert AI coding assistant with tool access.\n\n"
    "## Available Tools\n"
    "- `read_file(path)`: Read any file's full content\n"
    "- `edit_file(path, old_str, new_str)`: Make a precise inline edit (replaces first occurrence)\n"
    "- `run_bash(command)`: Execute a shell command and get stdout+stderr\n"
    "- `list_directory(path, depth=2)`: Tree view of a directory (max depth 5)\n"
    "- `search_in_files(pattern, path='.', file_pattern='*')`: Regex search across files (capped at 200)\n"
    "- `create_file(path, content)`: Create a new file (fails if it already exists)\n\n"
    "## How to work\n"
    "- ALWAYS use tools when a task involves files — never guess at content\n"
    "- Chain multiple tool calls to complete tasks: read → plan → edit → verify\n"
    "- After editing, run tests or check output with run_bash to confirm correctness\n"
    "- Use markdown fenced code blocks for all code in your text responses\n"
    "- Cite line numbers when reviewing or explaining code"
)


@dataclass
class StreamStats:
    start_time: float = field(default_factory=time.monotonic)
    token_count: int = 0

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time


def stream_chat(
    messages: list[dict],
    model: str,
    stats: StreamStats,
) -> Generator[str, None, None]:
    """Streaming chat — yields text chunks. Used by the `ask` command."""
    response = ollama.chat(model=model, messages=messages, stream=True)
    for chunk in response:
        thinking = getattr(chunk.message, "thinking", None)
        content = chunk.message.content
        if thinking:
            yield thinking
        if content:
            stats.token_count += 1
            yield content
        if chunk.done and chunk.eval_count:
            stats.token_count = chunk.eval_count


def call_nonstream(
    messages: list[dict],
    model: str,
    tools: list[dict] | None = None,
) -> Any:
    """Non-streaming chat — returns the full ChatResponse. Used by the agent loop."""
    kwargs: dict = {"model": model, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    return ollama.chat(**kwargs)


def stream_agent_iter(
    messages: list[dict],
    model: str,
    stats: StreamStats,
    tools: list[dict] | None = None,
) -> tuple[Generator[str, None, None], dict]:
    """
    Streaming agent call with optional tools.
    Returns (text_chunk_generator, result_holder).
    After exhausting the generator, result_holder['response'] holds the final ChatResponse.
    """
    result: dict = {}

    def _gen() -> Generator[str, None, None]:
        kwargs: dict = {"model": model, "messages": messages, "stream": True}
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
                result["response"] = chunk
                result["content"] = content_buf  # actual response only (excludes thinking)

    return _gen(), result


def build_system_messages() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def check_ollama(model: str) -> None:
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
