import queue as _queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from .client import SYSTEM_PROMPT, StreamStats, build_system_messages, call_nonstream, check_ollama
from .sessions import Session, load_sessions, new_session, save_session
from .tools import TOOL_SCHEMAS, execute_tool
from .ui import (
    call_with_thinking,
    console,
    print_error,
    print_info,
    print_tool_call,
    print_welcome,
    render_response,
)

_PROMPT_STYLE = Style.from_dict({"prompt": "ansicyan bold"})
_HISTORY_FILE = Path.home() / ".ai_coder_history"
DEFAULT_MODEL = "qwen3.5:4b"


# ── Input session ─────────────────────────────────────────────────────────────

def _make_prompt_session() -> PromptSession:
    kb = KeyBindings()

    @kb.add("enter", eager=True)
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    return PromptSession(
        key_bindings=kb,
        style=_PROMPT_STYLE,
        multiline=True,
        history=FileHistory(str(_HISTORY_FILE)),
    )


# ── Session helpers ───────────────────────────────────────────────────────────

def _restore_session(session: Session, messages: list[dict]) -> None:
    """Load session messages into the active list, always refreshing the system prompt."""
    messages.clear()
    messages.extend(session.messages)
    # Keep the system prompt current even if it changed since the session was saved
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": SYSTEM_PROMPT}


def _pick_session() -> Session | None:
    """Show a numbered session table, let the user choose one."""
    from rich.table import Table

    sessions = load_sessions()
    if not sessions:
        print_info("No saved sessions found.")
        return None

    table = Table(
        title="Saved Sessions",
        border_style="dim",
        highlight=True,
        show_lines=False,
    )
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Updated", style="dim", width=16)
    table.add_column("Title")
    table.add_column("Model", style="yellow")
    table.add_column("Turns", justify="right", width=6)

    for i, s in enumerate(sessions, 1):
        dt = s.updated_at[:16].replace("T", " ")
        table.add_row(str(i), dt, s.title, s.model, str(s.turn_count))

    console.print(table)

    try:
        raw = console.input("[dim]Enter number (Enter to cancel):[/] ").strip()
        if not raw:
            return None
        idx = int(raw) - 1
        if 0 <= idx < len(sessions):
            return sessions[idx]
        print_error(f"No session #{raw}.")
        return None
    except ValueError:
        print_error("Please enter a number.")
        return None
    except (EOFError, KeyboardInterrupt):
        console.print("")
        return None


# ── File / @mention helpers ───────────────────────────────────────────────────

def _inject_file(messages: list[dict], path: str) -> bool:
    p = Path(path.strip())
    if not p.exists():
        print_error(f"File not found: {p}")
        return False
    content = p.read_text(encoding="utf-8", errors="replace")
    messages.append({"role": "user", "content": f"File `{p.name}`:\n\n```\n{content}\n```"})
    messages.append({"role": "assistant", "content": f"Got it, I've loaded `{p.name}`."})
    print_info(f"Loaded {p} ({len(content)} chars)")
    return True


def _expand_at_mentions(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def _replace(m: re.Match) -> str:
        p = Path(m.group(1))
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8", errors="replace")
            found.append(str(p))
            return f"\n\nFile `{p.name}`:\n```\n{content}\n```\n"
        return m.group(0)

    return re.sub(r"@(\S+)", _replace, text), found


# ── Parallel tool execution ───────────────────────────────────────────────────

def _run_tools_parallel(tool_calls: list) -> list[tuple[str, dict, str]]:
    """
    Execute all tool calls concurrently.
    Results are *displayed* in completion order (fastest first) but
    *returned* in the original call order for consistent message history.
    """
    if len(tool_calls) == 1:
        tc = tool_calls[0]
        name, args = tc.function.name, dict(tc.function.arguments)
        result = execute_tool(name, args)
        print_tool_call(name, args, result)
        return [(name, args, result)]

    print_info(f"Running [bold]{len(tool_calls)}[/] tools in parallel...")

    ordered: dict[int, tuple[str, dict, str]] = {}

    def _one(idx: int, tc) -> tuple[int, str, dict, str]:
        name = tc.function.name
        args = dict(tc.function.arguments)
        result = execute_tool(name, args)
        return idx, name, args, result

    with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as pool:
        futures = {pool.submit(_one, i, tc): i for i, tc in enumerate(tool_calls)}
        for fut in as_completed(futures):
            idx, name, args, result = fut.result()
            ordered[idx] = (name, args, result)
            print_tool_call(name, args, result)

    return [ordered[i] for i in range(len(tool_calls))]


# ── Agent loop ────────────────────────────────────────────────────────────────

def _run_agent_turn(messages: list[dict], model: str) -> None:
    """
    Execute one user turn as a full agentic loop.

    Uses non-streaming calls so that tool_calls are always reliably populated
    (Ollama's streaming API does not guarantee tool_calls in the final chunk).
    A spinner is shown while the model generates each response.
    """
    while True:
        stats = StreamStats()

        try:
            response = call_with_thinking(
                lambda: call_nonstream(messages, model, tools=TOOL_SCHEMAS),
                stats,
            )
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/]")
            return
        except Exception as e:
            print_error(f"Model error: {e}")
            return

        if hasattr(response, "eval_count") and response.eval_count:
            stats.token_count = response.eval_count

        content = response.message.content or ""
        tool_calls = list(response.message.tool_calls or [])

        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            render_response(content, stats)
            return

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
                for tc in tool_calls
            ],
        })

        tool_results = _run_tools_parallel(tool_calls)
        for _name, _args, result in tool_results:
            messages.append({"role": "tool", "content": result})


# ── Main chat loop ────────────────────────────────────────────────────────────

def run_chat(
    model: str = DEFAULT_MODEL,
    file: str | None = None,
    resume_session: Session | None = None,
    background: bool = False,
) -> None:
    check_ollama(model)
    current_model = model
    current_session: Session | None = None
    messages = build_system_messages()

    if resume_session is not None:
        _restore_session(resume_session, messages)
        current_model = resume_session.model
        current_session = resume_session

    print_welcome(current_model)

    if resume_session is not None:
        print_info(
            f"Resumed: [bold]{resume_session.title}[/]  "
            f"[dim]{resume_session.turn_count} turns · {resume_session.updated_at[:10]}[/]"
        )

    if background:
        print_info("Background mode — type freely while the AI works.")

    if file:
        _inject_file(messages, file)

    prompt_session = _make_prompt_session()

    # ── Background worker ─────────────────────────────────────────────────────
    _work_q: _queue.Queue = _queue.Queue()

    def _worker() -> None:
        nonlocal current_session
        while True:
            item = _work_q.get()
            if item is None:
                _work_q.task_done()
                return
            expanded, model_snap = item
            messages.append({"role": "user", "content": expanded})
            _run_agent_turn(messages, model_snap)
            if current_session is None:
                current_session = new_session(model_snap)
            current_session.messages = list(messages)
            current_session.model = model_snap
            save_session(current_session)
            _work_q.task_done()

    if background:
        threading.Thread(target=_worker, daemon=True).start()

    def _dispatch(expanded: str, model: str) -> None:
        if background:
            _work_q.put((expanded, model))
        else:
            messages.append({"role": "user", "content": expanded})
            _run_agent_turn(messages, model)

    def _wait_idle() -> None:
        _work_q.join()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            user_input = prompt_session.prompt(
                HTML("<ansicyan><b>You</b></ansicyan> <ansigreen>></ansigreen> "),
                style=_PROMPT_STYLE,
            ).strip()
        except KeyboardInterrupt:
            console.print("\n[dim]Ctrl+C — type /exit to quit.[/]")
            continue
        except EOFError:
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Slash commands ────────────────────────────────────────────────────

        if cmd in ("/exit", "/quit"):
            _wait_idle()
            break

        if cmd == "/clear":
            _wait_idle()
            if current_session is not None:
                save_session(current_session)
            messages = build_system_messages()
            current_session = None
            print_info("Conversation cleared. Starting new session.")
            continue

        if cmd.startswith("/file "):
            _wait_idle()
            _inject_file(messages, user_input[6:])
            continue

        if cmd.startswith("/model"):
            parts = user_input.split(None, 1)
            if len(parts) == 2:
                current_model = parts[1].strip()
                if current_session is not None:
                    current_session.model = current_model
                print_info(f"Model: [bold yellow]{current_model}[/]")
            else:
                print_info(f"Current model: [bold yellow]{current_model}[/]")
            continue

        if cmd == "/resume":
            _wait_idle()
            chosen = _pick_session()
            if chosen is not None:
                if current_session is not None:
                    current_session.messages = list(messages)
                    save_session(current_session)
                _restore_session(chosen, messages)
                current_model = chosen.model
                current_session = chosen
                print_info(
                    f"Resumed: [bold]{chosen.title}[/]  "
                    f"[dim]{chosen.turn_count} turns · {chosen.updated_at[:10]}[/]"
                )
            continue

        # ── Normal message ────────────────────────────────────────────────────

        expanded, found = _expand_at_mentions(user_input)
        if found:
            print_info(f"Attached: {', '.join(found)}")

        _dispatch(expanded, current_model)

        if not background:
            if current_session is None:
                current_session = new_session(current_model)
            current_session.messages = list(messages)
            current_session.model = current_model
            save_session(current_session)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    _wait_idle()
    if background:
        _work_q.put(None)
    print_info("Bye!")
