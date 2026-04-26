from pathlib import Path
from typing import Annotated, Optional

import typer

from rich.table import Table

from .client import StreamStats, build_system_messages, check_ollama, stream_chat
from .ui import console, print_error, print_user_panel, stream_response

DEFAULT_MODEL = "qwen3.5:4b"

app = typer.Typer(
    name="ai-coder",
    help="AI coding assistant powered by Ollama.",
    add_completion=False,
    rich_markup_mode="rich",
)

sessions_app = typer.Typer(
    name="sessions",
    help="Manage saved chat sessions.",
    add_completion=False,
)
app.add_typer(sessions_app, name="sessions")


def _sessions_table(items: list) -> Table:
    table = Table(border_style="dim", show_lines=False, highlight=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Title")
    table.add_column("Model", style="yellow")
    table.add_column("Turns", justify="right", width=6)
    table.add_column("Updated", style="dim", width=16)
    for s in items:
        table.add_row(s.id, s.title, s.model, str(s.turn_count), s.updated_at[:16].replace("T", " "))
    return table


@sessions_app.command("list")
def sessions_list(
    n: Annotated[int, typer.Option("-n", help="Max sessions to show")] = 20,
) -> None:
    """List saved sessions."""
    from .sessions import load_sessions

    items = load_sessions(limit=n)
    if not items:
        console.print("[dim]No saved sessions.[/]")
        raise typer.Exit()
    console.print(_sessions_table(items))


@sessions_app.command("delete")
def sessions_delete(
    session_id: Annotated[str, typer.Argument(help="Session ID or unique prefix")],
) -> None:
    """Delete a session by ID or unique prefix."""
    from .sessions import delete_session

    if delete_session(session_id):
        console.print(f"[dim]Deleted:[/] {session_id}")
    else:
        print_error(f"Session '{session_id}' not found or prefix is ambiguous.")
        raise typer.Exit(1)


@sessions_app.command("search")
def sessions_search(
    query: Annotated[str, typer.Argument(help="Search term for session title")],
    n: Annotated[int, typer.Option("-n", help="Max results to show")] = 20,
) -> None:
    """Search sessions by title (case-insensitive)."""
    from .sessions import search_sessions

    items = search_sessions(query, limit=n)
    if not items:
        console.print(f"[dim]No sessions matching '{query}'.[/]")
        raise typer.Exit()
    console.print(_sessions_table(items))


def _read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print_error(f"File not found: {p}")
        raise typer.Exit(1)
    return p.read_text(encoding="utf-8", errors="replace")


def _run_ask(prompt: str, model: str) -> None:
    check_ollama(model)
    print_user_panel(prompt)
    messages = build_system_messages()
    messages.append({"role": "user", "content": prompt})
    stats = StreamStats()
    try:
        chunks = stream_chat(messages, model, stats)
        stream_response(chunks, stats)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/]")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question or instruction for the AI")],
    file: Annotated[
        Optional[str],
        typer.Option("-f", "--file", help="Include a file as context"),
    ] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Ollama model name")] = DEFAULT_MODEL,
) -> None:
    """Ask a single question. Use -f to attach a file as context."""
    if file:
        content = _read_file(file)
        prompt = f"File `{Path(file).name}`:\n\n```\n{content}\n```\n\n{question}"
    else:
        prompt = question
    _run_ask(prompt, model)


@app.command()
def chat(
    file: Annotated[
        Optional[str],
        typer.Option("-f", "--file", help="Pre-load a file into the conversation"),
    ] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Ollama model name")] = DEFAULT_MODEL,
    resume_last: Annotated[
        bool,
        typer.Option("-c", "--resume-last", help="Resume the most recent conversation"),
    ] = False,
    background: Annotated[
        bool,
        typer.Option("-b", "--background", help="Queue messages; type freely while the AI works"),
    ] = False,
    allow_bash: Annotated[
        bool,
        typer.Option("--allow-bash", help="Skip confirmation prompts for run_bash tool calls"),
    ] = False,
) -> None:
    """Start an interactive multi-turn chat session."""
    from .chat import run_chat
    from .sessions import load_last_session

    session = None
    if resume_last:
        session = load_last_session()
        if session is None:
            from .ui import print_info
            print_info("No saved sessions found — starting fresh.")

    if allow_bash:
        from .tools import set_bash_require_confirm
        set_bash_require_confirm(False)

    run_chat(model=model, file=file, resume_session=session, background=background)


@app.command()
def review(
    file: Annotated[str, typer.Argument(help="File to review")],
    model: Annotated[str, typer.Option("--model", "-m", help="Ollama model name")] = DEFAULT_MODEL,
) -> None:
    """Review a code file and provide feedback."""
    content = _read_file(file)
    name = Path(file).name
    prompt = (
        f"Please do a thorough code review of `{name}`. "
        "Point out bugs, security issues, performance problems, style issues, and suggestions for improvement.\n\n"
        f"```\n{content}\n```"
    )
    _run_ask(prompt, model)


@app.command()
def explain(
    file: Annotated[str, typer.Argument(help="File to explain")],
    model: Annotated[str, typer.Option("--model", "-m", help="Ollama model name")] = DEFAULT_MODEL,
) -> None:
    """Explain what a code file does."""
    content = _read_file(file)
    name = Path(file).name
    prompt = (
        f"Explain what `{name}` does. Describe the overall purpose, key functions/classes, "
        "and any important logic or patterns used.\n\n"
        f"```\n{content}\n```"
    )
    _run_ask(prompt, model)
