import math
import os
import queue as _queue
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from . import subagent as _subagent
from . import tasks as _tasks
from .completion import AtFileCompleter

from .client import (
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    StreamStats,
    build_system_messages,
    call_nonstream,
    check_ollama,
)
from .coder_md import find_coder_md_files, load_coder_md
from .memory import append_entry, current_project_path
from .sessions import Session, load_sessions, new_session, save_session
from .tools import (
    TOOL_SCHEMAS,
    bash_require_confirm,
    clear_snapshots,
    execute_tool,
    max_tools_per_turn,
    pop_snapshot,
    set_bash_require_confirm,
)
from .ui import (
    call_with_thinking,
    console,
    print_error,
    print_help,
    print_info,
    print_task_list,
    print_tool_call,
    print_welcome,
    render_response,
)

_PROMPT_STYLE = Style.from_dict({"prompt": "ansicyan bold"})
_HISTORY_FILE = Path.home() / ".farcode_history"
DEFAULT_MODEL = "qwen3.5:4b"

_CHARS_PER_TOKEN = 3.5
_HEADROOM_FRAC = 0.10
_AUTO_COMPACT_THRESHOLD = 0.80


def _summary_model(default_model: str) -> str:
    """Use FARCODE_SUMMARY_MODEL for compaction/session summary if set; else
    fall back to the chat's main model. A tiny model (e.g. qwen3:0.5b) is
    plenty for "turn 3000 chars into 4 bullets" and frees the main model from
    cold-cache reload between turns."""
    return os.environ.get("FARCODE_SUMMARY_MODEL", "").strip() or default_model


# ── Token accounting ─────────────────────────────────────────────────────────

def _est_tokens(msg: dict) -> int:
    """Heuristic token estimate for a single message. ~3.5 chars/token + overhead."""
    content = msg.get("content") or ""
    extra = msg.get("tool_calls")
    text = str(content) + (str(extra) if extra else "")
    return int(math.ceil(len(text) / _CHARS_PER_TOKEN)) + 4


def _total_tokens(msgs: list[dict]) -> int:
    return sum(_est_tokens(m) for m in msgs)


def _input_budget(num_ctx: int, num_predict: int) -> int:
    return max(1024, int((num_ctx - num_predict) * (1 - _HEADROOM_FRAC)))


def _trim_messages(messages: list[dict], num_ctx: int, num_predict: int) -> list[dict]:
    """Drop oldest user→assistant turn pairs when over budget. Never split a
    `tool_calls` → `tool` reply pair. The original list is never mutated.
    """
    budget = _input_budget(num_ctx, num_predict)
    if _total_tokens(messages) <= budget:
        return messages

    system = messages[:1]
    rest = list(messages[1:])

    while _total_tokens(system + rest) > budget and len(rest) >= 2:
        drop_end = 1
        for i in range(1, len(rest)):
            if rest[i].get("role") == "user":
                drop_end = i
                break
        else:
            drop_end = max(1, len(rest) - 2)
        rest = rest[drop_end:]

    return system + rest


# ── Auto-compaction ──────────────────────────────────────────────────────────

def _summarize_turns(turns: list[dict], model: str, num_ctx: int) -> str:
    """Summarize a slice of turns into a few bullets. Returns "" on failure."""
    lines: list[str] = []
    for m in turns:
        role = m.get("role")
        if role in ("user", "assistant") and m.get("content"):
            lines.append(f"{role.upper()}: {str(m['content'])[:400]}")
        elif role == "tool" and m.get("content"):
            lines.append(f"TOOL: {str(m['content'])[:200]}")
    if not lines:
        return ""
    conv = "\n".join(lines)
    if len(conv) > 6000:
        conv = conv[:6000] + "\n...[truncated]"
    summarize_msgs = [
        {
            "role": "system",
            "content": (
                "You are a concise summarizer. "
                "Reply with ONLY 3-6 bullet points (each starting with '- ') "
                "describing what the user asked, what was decided, and what files were touched. "
                "No intro, no headers."
            ),
        },
        {"role": "user", "content": f"Summarize these conversation turns:\n\n{conv}"},
    ]
    try:
        response = call_nonstream(
            summarize_msgs, _summary_model(model), tools=None,
            num_ctx=num_ctx, num_predict=512,
        )
        return (response.message.content or "").strip()
    except Exception:
        return ""


def _snap_past_tool_replies(messages: list[dict], idx: int) -> int:
    """Move idx forward past role='tool' entries to keep tool_calls/tool pairs intact."""
    while idx < len(messages) and messages[idx].get("role") == "tool":
        idx += 1
    return idx


def _auto_compact(
    messages: list[dict],
    model: str,
    num_ctx: int,
    num_predict: int,
    *,
    force: bool = False,
) -> list[dict]:
    """Summarize older turns when usage is high (or always, when force=True).

    Keeps system + last 4 user→assistant pairs verbatim; replaces the middle
    span with a synthetic user/assistant pair carrying the bullet summary.
    """
    budget = _input_budget(num_ctx, num_predict)
    if not force and _total_tokens(messages) < budget * _AUTO_COMPACT_THRESHOLD:
        return messages
    if len(messages) < 6:
        return messages

    system = messages[:1]
    rest = messages[1:]

    user_indices = [i for i, m in enumerate(rest) if m.get("role") == "user"]
    keep_pairs = 4
    if len(user_indices) <= keep_pairs:
        return messages

    cutoff = user_indices[-keep_pairs]
    cutoff = _snap_past_tool_replies(rest, cutoff)
    if cutoff <= 0 or cutoff >= len(rest):
        return messages

    older = rest[:cutoff]
    newer = rest[cutoff:]

    print_info("Auto-compacting earlier turns...")
    summary = _summarize_turns(older, model, num_ctx)
    if not summary:
        return messages

    try:
        append_entry(
            session_id="auto_compact",
            summary=summary,
            kind="task",
            project_path=current_project_path(),
        )
    except Exception:
        pass

    synthetic = [
        {"role": "user", "content": f"[Summary of earlier turns]\n{summary}"},
        {"role": "assistant", "content": "Acknowledged."},
    ]
    return system + synthetic + newer


# ── Input session ─────────────────────────────────────────────────────────────

def _make_prompt_session() -> PromptSession:
    kb = KeyBindings()

    @kb.add("enter", eager=True)
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("s-tab")
    def _toggle_auto_approve(event):
        set_bash_require_confirm(not bash_require_confirm())
        event.app.invalidate()

    def _bottom_toolbar():
        if bash_require_confirm():
            return HTML(
                '<style fg="ansiyellow">[shift+tab]</style> '
                'auto-approve: <style fg="ansired">OFF</style> '
                '<style fg="ansibrightblack">(tool calls require confirmation)</style>'
            )
        return HTML(
            '<style fg="ansiyellow">[shift+tab]</style> '
            'auto-approve: <style fg="ansigreen">ON</style> '
            '<style fg="ansibrightblack">(all tool calls auto-approved)</style>'
        )

    return PromptSession(
        key_bindings=kb,
        style=_PROMPT_STYLE,
        multiline=True,
        history=FileHistory(str(_HISTORY_FILE)),
        bottom_toolbar=_bottom_toolbar,
        completer=AtFileCompleter(),
        complete_while_typing=True,
    )


# ── Session helpers ───────────────────────────────────────────────────────────

def _restore_session(session: Session, messages: list[dict], num_ctx: int) -> None:
    """Load session messages into the active list, refreshing the system prompt.

    Rebuilds via build_system_messages() so any changes to CODER.md or stored
    memory since the session was saved are picked up on resume.
    """
    messages.clear()
    messages.extend(session.messages)
    fresh_system = build_system_messages(num_ctx=num_ctx)[0]
    if messages and messages[0].get("role") == "system":
        messages[0] = fresh_system
    else:
        messages.insert(0, fresh_system)


def _extract_files_touched(messages: list[dict]) -> list[str]:
    """Pull file paths out of tool_calls (read_file/edit_file/write_file/create_file)."""
    found: dict[str, None] = {}
    file_tools = {"read_file", "edit_file", "write_file", "create_file"}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            try:
                fn = tc.get("function") or {}
                if fn.get("name") in file_tools:
                    args = fn.get("arguments") or {}
                    path = args.get("path")
                    if isinstance(path, str) and path:
                        found.setdefault(path, None)
            except (AttributeError, TypeError):
                continue
    return list(found)


def _summarize_session(session: Session, model: str, num_ctx: int) -> None:
    try:
        real_turns = sum(
            1 for m in session.messages
            if m.get("role") == "user"
            and not str(m.get("content", "")).startswith("File `")
        )
        if real_turns == 0:
            return
        print_info("Summarizing session...")
        lines = []
        for m in session.messages:
            if m.get("role") in ("user", "assistant") and m.get("content"):
                lines.append(f"{m['role'].upper()}: {str(m['content'])[:400]}")
        conv = "\n".join(lines)
        if len(conv) > 6000:
            conv = conv[:6000] + "\n...[truncated]"
        summarize_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a concise summarizer. "
                    "Reply with ONLY 2-4 bullet points (each starting with '- ') "
                    "describing what was accomplished. No intro, no headers."
                ),
            },
            {"role": "user", "content": f"Summarize this coding session:\n\n{conv}"},
        ]
        response = call_nonstream(
            summarize_msgs, _summary_model(model), tools=None,
            num_ctx=num_ctx, num_predict=512,
        )
        summary = (response.message.content or "").strip()
        if summary:
            append_entry(
                session_id=session.id,
                summary=summary,
                kind="session_summary",
                project_path=current_project_path(),
                files_touched=_extract_files_touched(session.messages),
            )
    except Exception:
        pass


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


# ── Undo / diff / reindex ────────────────────────────────────────────────────

def _undo_last_write() -> str:
    """Restore the most recent file mutation. Pops the snapshot stack."""
    snap = pop_snapshot()
    if snap is None:
        return "Nothing to undo."
    path, prev = snap
    p = Path(path)
    try:
        if prev is None:
            # The snapshot represented a newly-created file — undo means delete.
            if p.exists():
                p.unlink()
            return f"Undone: deleted {path} (was newly created)."
        p.write_text(prev, encoding="utf-8")
        return f"Undone: restored {path} ({len(prev)} chars)."
    except OSError as e:
        return f"Undo failed for {path}: {e}"


def _show_git_diff() -> str:
    """Return `git diff` against HEAD as a string. Only meaningful if the cwd
    is a git repo. Falls back to a tracked-files-modified message otherwise."""
    if not shutil.which("git"):
        return "git not available."
    try:
        r = subprocess.run(
            ["git", "diff", "--no-color"],
            capture_output=True, text=True, timeout=10, cwd=os.getcwd(),
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return f"git diff failed: {e}"
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()
        return f"git diff returned {r.returncode}: {msg or '(no message)'}"
    out = r.stdout.rstrip()
    if not out:
        return "(no working-tree changes)"
    if len(out) > 8000:
        out = out[:8000] + "\n... [diff truncated; run `git diff` directly to see full output]"
    return out


def _reindex_code() -> str:
    """Force a full rebuild of the embedding index for the current project."""
    try:
        from .embeddings import index_project
    except ImportError as e:
        return f"reindex unavailable: {e}"
    try:
        n = index_project(force=True)
    except Exception as e:
        return (
            f"reindex error: {e}\n"
            "If this says 'model not found', run: ollama pull nomic-embed-text"
        )
    return f"Indexed {n} code chunk(s)."


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


def _expand_at_mentions(text: str) -> tuple[str, list[str], list[str]]:
    """Expand `@path` mentions in user input.

    Returns (expanded_text, hits, misses) where ``hits`` is a list of files
    that were inlined, and ``misses`` is a list of `@token` strings the user
    typed that did not resolve to a readable file. The original `@token`
    text is preserved in place for misses so the model still sees the
    user's literal request.
    """
    found: list[str] = []
    misses: list[str] = []

    def _replace(m: re.Match) -> str:
        token = m.group(1)
        p = Path(token)
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8", errors="replace")
            found.append(str(p))
            return f"\n\nFile `{p.name}`:\n```\n{content}\n```\n"
        misses.append(m.group(0))
        return m.group(0)

    return re.sub(r"@(\S+)", _replace, text), found, misses


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

def _run_agent_turn(
    messages: list[dict],
    model: str,
    num_ctx: int,
    num_predict: int,
) -> None:
    """Execute one user turn as a full agentic loop.

    Auto-compacts older turns when usage exceeds the threshold; falls back to
    hard trimming if compaction can't free enough space.
    """
    while True:
        compacted = _auto_compact(messages, model, num_ctx, num_predict)
        if compacted is not messages:
            messages.clear()
            messages.extend(compacted)

        stats = StreamStats()
        stats.ctx_size = num_ctx

        try:
            trimmed = _trim_messages(messages, num_ctx, num_predict)
            stats.input_tokens = _total_tokens(trimmed)
            response = call_with_thinking(
                lambda: call_nonstream(
                    trimmed, model, tools=TOOL_SCHEMAS,
                    num_ctx=num_ctx, num_predict=num_predict,
                ),
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
        prompt_eval = getattr(response, "prompt_eval_count", None)
        if prompt_eval:
            stats.input_tokens = prompt_eval

        content = response.message.content or ""
        tool_calls = list(response.message.tool_calls or [])

        if not tool_calls:
            messages.append({"role": "assistant", "content": content})
            render_response(content, stats)
            return

        cap = max_tools_per_turn()
        dropped = 0
        if len(tool_calls) > cap:
            dropped = len(tool_calls) - cap
            tool_calls = tool_calls[:cap]
            print_info(
                f"[dim]Dropped {dropped} extra tool call(s); cap is {cap}/turn "
                f"(set FARCODE_MAX_TOOLS_PER_TURN to change).[/]"
            )

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
        if dropped:
            messages.append({
                "role": "tool",
                "content": (
                    f"[note: {dropped} additional tool call(s) were dropped — "
                    "this agent runs one tool per turn by default. Issue follow-up "
                    "calls in the next turn.]"
                ),
            })


# ── Main chat loop ────────────────────────────────────────────────────────────

def run_chat(
    model: str = DEFAULT_MODEL,
    file: str | None = None,
    resume_session: Session | None = None,
    background: bool = False,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> None:
    check_ollama(model, num_ctx=num_ctx)
    clear_snapshots()  # /undo only spans the current session
    current_model = model
    current_session: Session | None = None
    messages = build_system_messages(num_ctx=num_ctx)
    first_user_seen = False
    _tasks.bind([])  # interim list until a Session is created (lazy)
    _subagent.bind_parent_model(current_model)

    if resume_session is not None:
        _restore_session(resume_session, messages, num_ctx)
        current_model = resume_session.model
        current_session = resume_session
        _tasks.bind(current_session.tasks)
        _subagent.bind_parent_model(current_model)
        first_user_seen = True
        compacted = _auto_compact(messages, current_model, num_ctx, num_predict)
        if compacted is not messages:
            messages.clear()
            messages.extend(compacted)

    print_welcome(current_model)

    for rules_path in find_coder_md_files():
        try:
            size = rules_path.stat().st_size
            print_info(f"Loaded rules: {rules_path} ({size} B)")
        except OSError:
            pass

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
            _run_agent_turn(messages, model_snap, num_ctx, num_predict)
            if current_session is None:
                current_session = new_session(model_snap)
                _tasks.move_into(current_session.tasks)
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
            _run_agent_turn(messages, model, num_ctx, num_predict)

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

        if cmd == "/help":
            print_help()
            continue

        if cmd == "/tasks":
            print_task_list(_tasks.list_all())
            continue

        if cmd in ("/exit", "/quit"):
            _wait_idle()
            if current_session is not None:
                _summarize_session(current_session, current_model, num_ctx)
            break

        if cmd == "/clear":
            _wait_idle()
            if current_session is not None:
                _summarize_session(current_session, current_model, num_ctx)
                save_session(current_session)
            messages = build_system_messages(num_ctx=num_ctx)
            current_session = None
            first_user_seen = False
            _tasks.bind([])  # fresh task list for the new session
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
                _subagent.bind_parent_model(current_model)
                print_info(f"Model: [bold yellow]{current_model}[/]")
            else:
                print_info(f"Current model: [bold yellow]{current_model}[/]")
            continue

        if cmd == "/compact":
            _wait_idle()
            before_n = len(messages)
            before_tok = _total_tokens(messages)
            compacted = _auto_compact(
                messages, current_model, num_ctx, num_predict, force=True
            )
            messages.clear()
            messages.extend(compacted)
            after_tok = _total_tokens(messages)
            print_info(
                f"Compacted: {before_n} → {len(messages)} msgs, "
                f"{before_tok} → {after_tok} tokens (est.)"
            )
            continue

        if cmd == "/rules":
            paths = find_coder_md_files()
            if not paths:
                print_info("No CODER.md found. Drop one in your project root or at ~/.coder.md.")
            else:
                rules = load_coder_md()
                for p in paths:
                    print_info(f"  · {p}")
                console.print()
                console.print(rules)
                console.print()
            continue

        if cmd == "/undo":
            _wait_idle()
            console.print(_undo_last_write())
            continue

        if cmd == "/diff":
            _wait_idle()
            diff = _show_git_diff()
            console.print(diff)
            continue

        if cmd == "/reindex":
            _wait_idle()
            print_info(_reindex_code())
            continue

        if cmd == "/resume":
            _wait_idle()
            chosen = _pick_session()
            if chosen is not None:
                if current_session is not None:
                    current_session.messages = list(messages)
                    save_session(current_session)
                _restore_session(chosen, messages, num_ctx)
                current_model = chosen.model
                current_session = chosen
                _tasks.bind(current_session.tasks)
                _subagent.bind_parent_model(current_model)
                first_user_seen = True
                compacted = _auto_compact(
                    messages, current_model, num_ctx, num_predict
                )
                if compacted is not messages:
                    messages.clear()
                    messages.extend(compacted)
                print_info(
                    f"Resumed: [bold]{chosen.title}[/]  "
                    f"[dim]{chosen.turn_count} turns · {chosen.updated_at[:10]}[/]"
                )
            continue

        # ── Normal message ────────────────────────────────────────────────────

        expanded, found, misses = _expand_at_mentions(user_input)
        if found:
            print_info(f"Attached: {', '.join(found)}")
        if misses:
            print_info(
                f"Did not resolve as file (left as text): {', '.join(misses)}"
            )

        # On the first real user message, swap in a query-aware system prompt
        # so memory recall surfaces project-relevant past work.
        if not first_user_seen and not user_input.startswith("/"):
            try:
                new_system = build_system_messages(
                    first_user_message=user_input, num_ctx=num_ctx
                )
                if new_system and messages and messages[0].get("role") == "system":
                    messages[0] = new_system[0]
            except Exception:
                pass
            first_user_seen = True

        _dispatch(expanded, current_model)

        if not background:
            if current_session is None:
                current_session = new_session(current_model)
                _tasks.move_into(current_session.tasks)
            current_session.messages = list(messages)
            current_session.model = current_model
            save_session(current_session)

    # ── Shutdown ──────────────────────────────────────────────────────────────
    _wait_idle()
    if background:
        _work_q.put(None)
    print_info("Bye!")
