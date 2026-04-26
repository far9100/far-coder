import re

from typer.testing import CliRunner

from farcode.cli import app

runner = CliRunner()

# Rich/typer interleaves ANSI color codes into option names on TTY-like outputs
# (e.g. `\x1b[1;36m-\x1b[0m\x1b[1;36m-model\x1b[0m`), so the literal substring
# `--model` won't appear. Strip color before substring checks.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_main_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "ask" in out
    assert "chat" in out
    assert "review" in out
    assert "explain" in out


def test_ask_help():
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "question" in out.lower() or "--model" in out


def test_chat_help():
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--model" in _plain(result.output)


def test_review_help():
    result = runner.invoke(app, ["review", "--help"])
    assert result.exit_code == 0


def test_explain_help():
    result = runner.invoke(app, ["explain", "--help"])
    assert result.exit_code == 0


def test_sessions_help():
    result = runner.invoke(app, ["sessions", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    assert "list" in out
    assert "delete" in out
    assert "search" in out


def test_sessions_list_help():
    result = runner.invoke(app, ["sessions", "list", "--help"])
    assert result.exit_code == 0


def test_sessions_delete_help():
    result = runner.invoke(app, ["sessions", "delete", "--help"])
    assert result.exit_code == 0


def test_sessions_search_help():
    result = runner.invoke(app, ["sessions", "search", "--help"])
    assert result.exit_code == 0
