from typer.testing import CliRunner

from farcode.cli import app

runner = CliRunner()


def test_main_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ask" in result.output
    assert "chat" in result.output
    assert "review" in result.output
    assert "explain" in result.output


def test_ask_help():
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "question" in result.output.lower() or "--model" in result.output


def test_chat_help():
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_review_help():
    result = runner.invoke(app, ["review", "--help"])
    assert result.exit_code == 0


def test_explain_help():
    result = runner.invoke(app, ["explain", "--help"])
    assert result.exit_code == 0


def test_sessions_help():
    result = runner.invoke(app, ["sessions", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "delete" in result.output
    assert "search" in result.output


def test_sessions_list_help():
    result = runner.invoke(app, ["sessions", "list", "--help"])
    assert result.exit_code == 0


def test_sessions_delete_help():
    result = runner.invoke(app, ["sessions", "delete", "--help"])
    assert result.exit_code == 0


def test_sessions_search_help():
    result = runner.invoke(app, ["sessions", "search", "--help"])
    assert result.exit_code == 0
