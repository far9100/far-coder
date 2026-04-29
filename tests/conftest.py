"""Shared pytest fixtures for end-to-end agent-loop tests.

The chat module is normally driven by ``ollama.chat`` via ``call_nonstream``.
For tests we want to exercise the real ``_run_agent_turn`` loop without
hitting Ollama. ``fake_ollama`` installs a scripted stand-in that returns a
pre-recorded sequence of responses and captures every call's arguments for
later assertion.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import pytest

from farcode.client import _SyntheticResponse, _SyntheticToolCall


def make_response(
    text: str = "",
    *,
    tool_calls: list | None = None,
    eval_count: int = 10,
    prompt_eval_count: int = 20,
) -> _SyntheticResponse:
    """Build an Ollama-shaped response object for use in scripted scenarios."""
    return _SyntheticResponse(
        content=text,
        tool_calls=tool_calls or [],
        eval_count=eval_count,
        prompt_eval_count=prompt_eval_count,
    )


def make_tool_call(name: str, **arguments: Any) -> _SyntheticToolCall:
    """Build a tool-call object that mimics ``ollama.ChatResponse.message.tool_calls[i]``."""
    return _SyntheticToolCall(name, arguments)


class FakeOllama:
    """Scripted stand-in for ``call_nonstream``.

    Construct with a list of responses; each agent-loop call to the model pops
    one. Every call's keyword arguments are recorded in ``self.calls`` so tests
    can assert on the messages, tools, and options that were sent.
    """

    def __init__(self, responses: list[Any]) -> None:
        self.responses: list[Any] = list(responses)
        self.calls: list[dict] = []

    def __call__(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        num_ctx: int = 0,
        num_predict: int = 0,
    ) -> Any:
        self.calls.append({
            "messages": copy.deepcopy(messages),
            "model": model,
            "tools": tools,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        })
        if not self.responses:
            raise AssertionError(
                f"FakeOllama exhausted after {len(self.calls)} call(s). "
                "The agent loop made more calls than the test scripted; "
                "append more responses with .push() or shorten the scenario."
            )
        return self.responses.pop(0)

    def push(self, response: Any) -> None:
        """Append a response mid-test (useful when reacting to captured calls)."""
        self.responses.append(response)

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def remaining(self) -> int:
        return len(self.responses)


@pytest.fixture
def fake_ollama(monkeypatch) -> Callable[[list[Any]], FakeOllama]:
    """Factory fixture: ``fake = fake_ollama([resp1, resp2, ...])``.

    Installs the FakeOllama stand-in on ``chat.call_nonstream`` and silences
    the UI side-effects so tests run quietly. Returns the FakeOllama instance
    so tests can inspect ``.calls`` and ``.remaining``.

    Tests that need to stub tool execution should still patch
    ``chat.execute_tool`` themselves — that's intentionally left unscoped so
    each test can choose between real tools (with ``tmp_path``) and a stub.
    """
    from farcode import chat

    def _install(responses: list[Any]) -> FakeOllama:
        fake = FakeOllama(responses)
        monkeypatch.setattr(chat, "call_nonstream", fake)
        monkeypatch.setattr(chat, "call_with_thinking", lambda fn, stats: fn())
        monkeypatch.setattr(chat, "render_response", lambda *a, **kw: None)
        monkeypatch.setattr(chat, "print_info", lambda *a, **kw: None)
        monkeypatch.setattr(chat, "print_error", lambda *a, **kw: None)
        monkeypatch.setattr(chat, "print_tool_call", lambda *a, **kw: None)
        return fake

    return _install
