import sys
import threading
from typing import Any, Callable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .client import StreamStats
from .commands import SLASH_COMMANDS, banner_line

# Reconfigure stdout/stderr to UTF-8 on Windows so Rich can emit Unicode
# spinners and box-drawing characters in any terminal.
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")

# force_terminal=True  → always use ANSI (not pipe/dumb-terminal mode)
# legacy_windows=False → use ANSI escape codes, not Win32 console API
# Both are needed for correct dynamic-width queries and clean resize behaviour
# on modern Windows terminals (Windows Terminal, VS Code, etc.).
console = Console(force_terminal=True, legacy_windows=False)

ERROR_STYLE = "bold red"
DIM_STYLE = "dim"


# ── Static helpers ────────────────────────────────────────────────────────────

def print_user_panel(text: str) -> None:
    console.print()
    console.print("[bold green]You[/]")
    console.print(text)
    console.print()


def print_error(msg: str) -> None:
    console.print(f"[{ERROR_STYLE}]Error:[/] {msg}")


def print_info(msg: str) -> None:
    console.print(f"[{DIM_STYLE}]{msg}[/]")


def print_welcome(model: str) -> None:
    console.print()
    console.print(
        f"[bold cyan]FarCode[/] — powered by [bold yellow]{model}[/]\n"
        "[dim]Enter[/] send  [dim]Alt+Enter[/] newline  "
        "[dim]↑↓[/] history  [dim]@file[/] attach  "
        "[dim]Shift+Tab[/] toggle auto-approve\n"
        f"[dim]{banner_line()}[/]"
    )
    console.print()


def print_help() -> None:
    """Render the full slash-command list with one-line descriptions."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="dim")
    for name, desc in SLASH_COMMANDS:
        table.add_row(name, desc)
    console.print()
    console.print("[bold]In-session commands[/]")
    console.print(table)
    console.print()


_TASK_ICON = {"pending": "○", "in_progress": "→", "completed": "✓"}
_TASK_STYLE = {"pending": "dim", "in_progress": "bold yellow", "completed": "bold green"}


def print_task_list(tasks: list[dict]) -> None:
    """Render the in-session task list as a Rich table."""
    if not tasks:
        console.print("[dim]No tasks.[/]")
        return
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(overflow="fold")
    for t in tasks:
        status = t.get("status", "pending")
        icon = _TASK_ICON.get(status, "?")
        style = _TASK_STYLE.get(status, "dim")
        content = t.get("content", "")
        if status == "completed":
            content = f"[strike dim]{content}[/]"
        table.add_row(f"[{style}]{icon}[/]", t.get("id", ""), content)
    console.print()
    console.print("[bold]Tasks[/]")
    console.print(table)
    console.print()


def print_tool_call(name: str, args: dict, result: str) -> None:
    arg_lines = "\n".join(
        f"  [dim]{k}:[/] {_truncate(str(v), 120)}" for k, v in args.items()
    )
    console.print()
    console.print(f"[bold yellow]Tool Call  > {name}[/]")
    if arg_lines:
        console.print(arg_lines)
    console.print(f"[bold green]Result:[/]\n{_truncate(result, 1000)}")
    console.print()


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"\n[dim]... ({len(s) - n} more chars)[/]"


def _ctx_indicator_markup(stats: StreamStats) -> str:
    """Return `  [color]ctx 28K/64K[/]` markup for the title line, or '' if unset."""
    if stats.ctx_size <= 0:
        return ""
    used_k = max(0, stats.input_tokens) // 1024
    total_k = max(1, stats.ctx_size // 1024)
    ratio = stats.input_tokens / stats.ctx_size if stats.ctx_size else 0.0
    if ratio >= 0.85:
        color = "bold red"
    elif ratio >= 0.60:
        color = "yellow"
    else:
        color = "green"
    return f"  [{color}]ctx {used_k}K/{total_k}K[/]"


def _ctx_indicator_text(stats: StreamStats) -> tuple[str, str] | None:
    """Same as _ctx_indicator_markup but for rich.Text; returns (text, style) or None."""
    if stats.ctx_size <= 0:
        return None
    used_k = max(0, stats.input_tokens) // 1024
    total_k = max(1, stats.ctx_size // 1024)
    ratio = stats.input_tokens / stats.ctx_size if stats.ctx_size else 0.0
    if ratio >= 0.85:
        style = "bold red"
    elif ratio >= 0.60:
        style = "yellow"
    else:
        style = "green"
    return (f"ctx {used_k}K/{total_k}K", style)


def render_response(content: str, stats: StreamStats) -> None:
    """Print a static assistant response at the *current* terminal width."""
    if not content.strip():
        return
    title = (
        f"[bold cyan]Assistant[/]  "
        f"[dim]{stats.elapsed:.1f}s | {stats.token_count} tok[/]"
        f"{_ctx_indicator_markup(stats)}"
    )
    console.print()
    console.print(title)
    console.print(Markdown(content))
    console.print()


# ── Live display ──────────────────────────────────────────────────────────────

class _LiveDisplay:
    """
    Rich renderable for Live sessions.

    Uses __rich_console__ so it receives a fresh ConsoleOptions (with the
    *current* max_width) on every auto-refresh tick.  This means the content
    is always laid out at the live terminal width, not a stale cached value.

    All Live contexts that use this class should be opened with
    transient=True.  That guarantees the animated content is erased cleanly
    when the session closes, and is immediately replaced by a static
    render_response() call at the correct width — so even if a mid-session
    resize leaves a cursor-positioning artefact, the final output is always
    clean.
    """

    def __init__(self, stats: StreamStats) -> None:
        self._stats = stats
        self._text = ""

    def append(self, chunk: str) -> None:
        self._text += chunk

    def __rich_console__(self, console_obj, options):
        elapsed = self._stats.elapsed
        tokens = self._stats.token_count

        if self._text:
            parts = [
                ("Assistant", "bold cyan"),
                ("  ", ""),
                (f"{elapsed:.1f}s | {tokens} tok", "dim"),
            ]
            ctx = _ctx_indicator_text(self._stats)
            if ctx:
                parts.append(("  ", ""))
                parts.append(ctx)
            title = Text.assemble(*parts)
            yield from console_obj.render(title, options)
            yield from console_obj.render(Markdown(self._text), options)
        else:
            renderable = Spinner(
                "dots",
                text=Text(f"Thinking...  {elapsed:.1f}s", style="dim"),
            )
            yield from console_obj.render(renderable, options)


# ── Public Live-based helpers ─────────────────────────────────────────────────

def stream_response(chunks, stats: StreamStats) -> str:
    """
    Stream text chunks into an animated live panel (transient), then print
    the finished response as a static panel at the current terminal width.
    """
    display = _LiveDisplay(stats)

    with Live(
        display,
        console=console,
        refresh_per_second=15,
        auto_refresh=True,
        transient=True,       # erased cleanly when streaming ends
    ):
        try:
            for chunk in chunks:
                display.append(chunk)
        except KeyboardInterrupt:
            pass

    # Static final render — always at the current (possibly resized) width
    render_response(display._text, stats)
    return display._text


def call_with_thinking(fn: Callable[[], Any], stats: StreamStats) -> Any:
    """
    Run fn() in a background thread while showing a live thinking spinner
    (transient).  Raises KeyboardInterrupt if Ctrl+C is pressed.
    """
    _result: list[Any] = [None]
    _exc: list[BaseException | None] = [None]
    _done = threading.Event()

    def _worker() -> None:
        try:
            _result[0] = fn()
        except BaseException as exc:
            _exc[0] = exc
        finally:
            _done.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    interrupted = False
    display = _LiveDisplay(stats)

    with Live(
        display,
        console=console,
        refresh_per_second=15,
        auto_refresh=True,
        transient=True,       # spinner erased when fn() returns
    ):
        try:
            _done.wait()
        except KeyboardInterrupt:
            interrupted = True

    t.join(timeout=2)

    if interrupted:
        raise KeyboardInterrupt
    if _exc[0] is not None:
        raise _exc[0]  # type: ignore[misc]
    return _result[0]
