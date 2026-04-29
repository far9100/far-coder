from pathlib import Path
from typing import Annotated, Optional

import typer

from rich.table import Table

from .client import (
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    StreamStats,
    build_system_messages,
    check_ollama,
    stream_chat,
)
from .ui import console, print_error, print_user_panel, stream_response

DEFAULT_MODEL = "qwen3.5:4b"

app = typer.Typer(
    name="farcode",
    help="Local AI coding assistant powered by Ollama.",
    add_completion=False,
    rich_markup_mode="rich",
)

sessions_app = typer.Typer(
    name="sessions",
    help="Manage saved chat sessions.",
    add_completion=False,
)
app.add_typer(sessions_app, name="sessions")


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Local AI coding assistant powered by Ollama."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(
            chat,
            file=None,
            model=DEFAULT_MODEL,
            resume_last=False,
            background=False,
            allow_bash=False,
            allow_all=False,
            allow_web=False,
            num_ctx=DEFAULT_NUM_CTX,
            max_output=DEFAULT_NUM_PREDICT,
        )


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


def _run_ask(prompt: str, model: str, num_ctx: int, num_predict: int) -> None:
    check_ollama(model, num_ctx=num_ctx)
    print_user_panel(prompt)
    messages = build_system_messages(first_user_message=prompt, num_ctx=num_ctx)
    messages.append({"role": "user", "content": prompt})
    stats = StreamStats()
    try:
        chunks = stream_chat(messages, model, stats, num_ctx=num_ctx, num_predict=num_predict)
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
    num_ctx: Annotated[
        int,
        typer.Option("--ctx", help="Context window size in tokens (overrides FARCODE_NUM_CTX)"),
    ] = DEFAULT_NUM_CTX,
    max_output: Annotated[
        int,
        typer.Option("--max-output", help="Max output tokens reserved for the model's reply"),
    ] = DEFAULT_NUM_PREDICT,
) -> None:
    """Ask a single question. Use -f to attach a file as context."""
    if file:
        content = _read_file(file)
        prompt = f"File `{Path(file).name}`:\n\n```\n{content}\n```\n\n{question}"
    else:
        prompt = question
    _run_ask(prompt, model, num_ctx=num_ctx, num_predict=max_output)


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
    allow_all: Annotated[
        bool,
        typer.Option("--allow-all", help="Auto-approve ALL tool calls without any confirmation prompt"),
    ] = False,
    allow_web: Annotated[
        bool,
        typer.Option("--allow-web", help="Enable scoped fetch_doc tool (PyPI/npm/crates/pkg.go.dev only)"),
    ] = False,
    num_ctx: Annotated[
        int,
        typer.Option("--ctx", help="Context window size in tokens (overrides FARCODE_NUM_CTX)"),
    ] = DEFAULT_NUM_CTX,
    max_output: Annotated[
        int,
        typer.Option("--max-output", help="Max output tokens reserved for the model's reply"),
    ] = DEFAULT_NUM_PREDICT,
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

    if allow_all or allow_bash:
        from .tools import set_bash_require_confirm
        set_bash_require_confirm(False)

    if allow_web:
        from .tools import set_web_enabled
        set_web_enabled(True)

    run_chat(
        model=model,
        file=file,
        resume_session=session,
        background=background,
        num_ctx=num_ctx,
        num_predict=max_output,
    )


@app.command()
def review(
    file: Annotated[str, typer.Argument(help="File to review")],
    model: Annotated[str, typer.Option("--model", "-m", help="Ollama model name")] = DEFAULT_MODEL,
    num_ctx: Annotated[int, typer.Option("--ctx")] = DEFAULT_NUM_CTX,
    max_output: Annotated[int, typer.Option("--max-output")] = DEFAULT_NUM_PREDICT,
) -> None:
    """Review a code file and provide feedback."""
    content = _read_file(file)
    name = Path(file).name
    prompt = (
        f"Please do a thorough code review of `{name}`. "
        "Point out bugs, security issues, performance problems, style issues, and suggestions for improvement.\n\n"
        f"```\n{content}\n```"
    )
    _run_ask(prompt, model, num_ctx=num_ctx, num_predict=max_output)


@app.command()
def explain(
    file: Annotated[str, typer.Argument(help="File to explain")],
    model: Annotated[str, typer.Option("--model", "-m", help="Ollama model name")] = DEFAULT_MODEL,
    num_ctx: Annotated[int, typer.Option("--ctx")] = DEFAULT_NUM_CTX,
    max_output: Annotated[int, typer.Option("--max-output")] = DEFAULT_NUM_PREDICT,
) -> None:
    """Explain what a code file does."""
    content = _read_file(file)
    name = Path(file).name
    prompt = (
        f"Explain what `{name}` does. Describe the overall purpose, key functions/classes, "
        "and any important logic or patterns used.\n\n"
        f"```\n{content}\n```"
    )
    _run_ask(prompt, model, num_ctx=num_ctx, num_predict=max_output)
