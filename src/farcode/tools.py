import ast
import fnmatch
import json as _json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from rich.prompt import Confirm

# ── Bash sandboxing ───────────────────────────────────────────────────────────

_bash_require_confirm: bool = True


def set_bash_require_confirm(val: bool) -> None:
    """Call with False to skip confirmation prompts (e.g. --allow-bash flag)."""
    global _bash_require_confirm
    _bash_require_confirm = val


def bash_require_confirm() -> bool:
    """Return True when run_bash should prompt for confirmation."""
    return _bash_require_confirm


# ── Per-turn tool-call cap ────────────────────────────────────────────────────

def max_tools_per_turn() -> int:
    """Cap on tool calls executed in a single agent turn.

    Small models often emit one good call plus one malformed one. Defaulting
    to 1 forces serial execution, which dramatically improves reliability on
    qwen3-class 4B models. Set FARCODE_MAX_TOOLS_PER_TURN=0 (or any large
    number) to allow parallel calls again.
    """
    raw = os.environ.get("FARCODE_MAX_TOOLS_PER_TURN", "1")
    try:
        n = int(raw)
    except ValueError:
        return 1
    return n if n > 0 else 999


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file. Returns full file (capped at ~50 KB) or a 1-based "
                "line slice via offset/limit. ALWAYS read before editing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start reading from",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of lines to return",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace the first occurrence of old_str with new_str. Tries an "
                "exact match first, then falls back to whitespace-tolerant matching. "
                "If it still misses, use replace_lines or write_file instead of guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "old_str": {
                        "type": "string",
                        "description": "String to find — exact preferred, but whitespace mismatch is tolerated",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement string",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": (
                "Replace lines start_line through end_line (1-based, inclusive) with "
                "new_content. Easier than edit_file for multi-line changes when you "
                "have the exact line numbers from a recent read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {
                        "type": "integer",
                        "description": "First line to replace (1-based)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to replace, inclusive (1-based)",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Replacement text. Newline at end is added if missing.",
                    },
                },
                "required": ["path", "start_line", "end_line", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or overwrite a file with the given content. The safe fallback "
                "when edit_file or replace_lines fail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a PowerShell command on Windows and return stdout + stderr. "
                "PowerShell syntax: chain with `;` not `&&`; delete with "
                "`Remove-Item -Recurse -Force path` not `rm -rf`; rename with "
                "`Move-Item src dst`. Default timeout 30s."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "Show a tree view of a directory's contents. "
                "`depth` controls how many levels to recurse (max 5, default 2)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                    "depth": {
                        "type": "integer",
                        "description": "Max recursion depth (1–5, default 2)",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": (
                "Search for a regex pattern across files in a directory. "
                "Results capped at 200 matches. "
                "`file_pattern` is a glob (e.g. '*.py')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: '.')",
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Glob to filter files (default: '*')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file with given content. Fails if the file already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path for the new file"},
                    "content": {"type": "string", "description": "File content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Search past session memories by keyword (FTS5-ranked). "
                "Returns up to 5 matching summaries. Default scope is 'project' "
                "(only the current repo); pass scope='all' to search globally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search for in past session summaries",
                    },
                    "scope": {
                        "type": "string",
                        "description": "'project' (default) or 'all'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Record a one-sentence lesson from the current session so it survives "
                "into future sessions. Use after completing a non-trivial sub-goal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One- or two-sentence distilled lesson or decision",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of short tag strings",
                    },
                    "files_touched": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of file paths that this lesson relates to",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    handlers = {
        "read_file": _read_file,
        "write_file": _write_file,
        "edit_file": _edit_file,
        "replace_lines": _replace_lines,
        "run_bash": _run_bash,
        "list_directory": _list_directory,
        "search_in_files": _search_in_files,
        "create_file": _create_file,
        "recall_memory": _recall_memory,
        "save_memory": _save_memory,
    }
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        return handler(**arguments)
    except TypeError as e:
        return f"Tool call error (bad arguments): {e}"
    except Exception as e:
        return f"Tool error: {e}"


# ── Post-edit syntax check ────────────────────────────────────────────────────

def _check_syntax(path: str) -> str:
    """Return "" on success, " | <msg>" on failure. Quick, in-process when possible.

    Skips silently for unrecognized extensions and for tools that are missing.
    Output is suffixed onto the tool result so the model can self-correct.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if not p.is_file():
        return ""
    try:
        if ext == ".py":
            try:
                ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
            except SyntaxError as e:
                return f" | syntax error: line {e.lineno}: {e.msg}"
            return ""
        if ext == ".json":
            try:
                _json.loads(p.read_text(encoding="utf-8", errors="replace"))
            except _json.JSONDecodeError as e:
                return f" | json error: line {e.lineno}: {e.msg}"
            return ""
        if ext in (".js", ".mjs", ".cjs"):
            node = shutil.which("node")
            if not node:
                return ""
            r = subprocess.run(
                [node, "--check", str(p)],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                err = (r.stderr or r.stdout).strip().splitlines()
                msg = err[0] if err else "syntax error"
                return f" | js syntax error: {msg[:200]}"
            return ""
        if ext in (".ts", ".tsx"):
            tsc = shutil.which("tsc")
            if not tsc:
                return ""
            r = subprocess.run(
                [tsc, "--noEmit", "--allowJs", str(p)],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                err = (r.stdout or r.stderr).strip().splitlines()
                msg = err[0] if err else "type error"
                return f" | tsc error: {msg[:200]}"
            return ""
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return ""


# ── Snapshot hooks (for /undo, populated in chat session) ────────────────────

_snapshot_before: "list[tuple[str, str | None]]" = []
_snapshot_max = 20


def _snapshot_before_write(path: str, prev_content: str) -> None:
    """Record the file's pre-edit content so /undo can restore it."""
    _snapshot_before.append((str(Path(path).resolve()), prev_content))
    if len(_snapshot_before) > _snapshot_max:
        del _snapshot_before[: len(_snapshot_before) - _snapshot_max]


def _snapshot_after_write(path: str) -> None:
    """For new-file writes where there's no prior content; pre = None."""
    p = Path(path).resolve()
    # Only record if we don't already have a snapshot for this write
    if not _snapshot_before or _snapshot_before[-1][0] != str(p):
        _snapshot_before.append((str(p), None))
        if len(_snapshot_before) > _snapshot_max:
            del _snapshot_before[: len(_snapshot_before) - _snapshot_max]


def pop_snapshot() -> tuple[str, str | None] | None:
    """Pop the most recent snapshot; used by /undo."""
    if not _snapshot_before:
        return None
    return _snapshot_before.pop()


def clear_snapshots() -> None:
    """Wipe the snapshot stack, e.g. when a session starts."""
    _snapshot_before.clear()


def _read_file(path: str, offset: int | None = None, limit: int | None = None) -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    if not p.is_file():
        return f"Error: Not a file: {path}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Error: Permission denied: {path}"

    if offset is not None or limit is not None:
        lines = text.splitlines(keepends=True)
        start = max(0, (offset or 1) - 1)
        end = start + limit if limit and limit > 0 else len(lines)
        end = min(end, len(lines))
        sliced = "".join(lines[start:end])
        prefix = f"[lines {start + 1}-{end} of {len(lines)}]\n" if lines else ""
        return _truncate_output(prefix + sliced, cap=_MAX_FILE_READ_CHARS)
    return _truncate_output(text, cap=_MAX_FILE_READ_CHARS)


def _write_file(path: str, content: str) -> str:
    p = Path(path)
    action = "overwritten" if p.exists() else "created"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        _snapshot_after_write(path)
        return f"OK: {action} {path}{_check_syntax(path)}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except OSError as e:
        return f"Error: {e}"


def _fuzzy_locate(content: str, old_str: str) -> tuple[int, int] | None:
    """Locate old_str in content with whitespace tolerance.

    Tries: (1) exact substring; (2) line-stripped match — each line in old_str
    has its leading/trailing whitespace stripped, then we look for a sequence of
    consecutive lines in content that strip to the same thing.

    Returns (start_offset, end_offset) into the original content on success,
    or None if nothing matches. The returned span uses the original (un-stripped)
    text indices so the caller can splice content[start:end] cleanly.
    """
    idx = content.find(old_str)
    if idx >= 0:
        return idx, idx + len(old_str)

    target_lines = [ln.strip() for ln in old_str.splitlines()]
    target_lines = [ln for ln in target_lines if ln]
    if not target_lines:
        return None

    content_lines = content.splitlines(keepends=True)
    stripped = [ln.strip() for ln in content_lines]

    n = len(target_lines)
    for i in range(len(stripped) - n + 1):
        window = [s for s in stripped[i:i + n] if s]
        if window == target_lines:
            start_off = sum(len(content_lines[k]) for k in range(i))
            end_off = start_off + sum(len(content_lines[k]) for k in range(i, i + n))
            return start_off, end_off
    return None


def _edit_file(path: str, old_str: str, new_str: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    content = p.read_text(encoding="utf-8", errors="replace")

    span = _fuzzy_locate(content, old_str)
    if span is None:
        return (
            f"Error: old_str not found in {path}.\n"
            "Tried exact match and whitespace-tolerant match. Either re-read the "
            "file (it may have changed), or use replace_lines / write_file instead."
        )
    start, end = span
    note = ""
    if content[start:end] != old_str:
        note = " (matched with whitespace tolerance)"
    elif content.count(old_str) > 1:
        note = f" ({content.count(old_str)} occurrences, only first replaced)"

    new_content = content[:start] + new_str + content[end:]
    _snapshot_before_write(path, content)
    p.write_text(new_content, encoding="utf-8")
    _snapshot_after_write(path)
    return f"OK: edited {path}{note}{_check_syntax(path)}"


def _replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> str:
    """Replace lines [start_line, end_line] (1-based, inclusive) with new_content."""
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    try:
        start_line = int(start_line)
        end_line = int(end_line)
    except (TypeError, ValueError):
        return "Error: start_line and end_line must be integers"
    if start_line < 1 or end_line < start_line:
        return f"Error: invalid line range {start_line}-{end_line}"

    text = p.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    if start_line > len(lines):
        return (
            f"Error: start_line {start_line} is past end of file "
            f"({len(lines)} lines). Use write_file to append, or read_file first."
        )
    end_line = min(end_line, len(lines))

    new = new_content if new_content.endswith("\n") or end_line == len(lines) and not lines[-1].endswith("\n") else new_content + "\n"
    if not new_content:
        new = ""

    spliced = lines[:start_line - 1] + ([new] if new else []) + lines[end_line:]
    _snapshot_before_write(path, text)
    p.write_text("".join(spliced), encoding="utf-8")
    _snapshot_after_write(path)
    replaced = end_line - start_line + 1
    return (
        f"OK: replaced {replaced} line{'s' if replaced != 1 else ''} in {path} "
        f"(lines {start_line}-{end_line}){_check_syntax(path)}"
    )


def _run_bash(command: str, timeout: int = 30) -> str:
    if _bash_require_confirm:
        if not sys.stdin.isatty():
            return "run_bash denied: not running in an interactive terminal"
        if not Confirm.ask(f"[yellow]Allow bash command?[/]\n  {command}"):
            return "run_bash denied: user declined"
    try:
        if sys.platform == "win32":
            shell_exe = shutil.which("pwsh") or shutil.which("powershell")
            if shell_exe:
                result = subprocess.run(
                    [shell_exe, "-NoProfile", "-NonInteractive", "-Command", command],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            else:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True, timeout=timeout,
                )
        else:
            bash = shutil.which("bash")
            if bash:
                result = subprocess.run(
                    [bash, "-c", command],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            else:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True, timeout=timeout,
                )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"

    parts = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(result.stderr.rstrip())
    output = "\n".join(parts) if parts else "(no output)"
    if result.returncode != 0:
        output = f"[exit {result.returncode}]\n{output}"
    return _truncate_output(output)


def _list_directory(path: str, depth: int = 2) -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: Path not found: {path}"
    if not p.is_dir():
        return f"Error: Not a directory: {path}"
    depth = max(1, min(depth, 5))
    lines: list[str] = [str(p)]

    def _walk(directory: Path, current_depth: int, prefix: str) -> None:
        if current_depth > depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        except PermissionError:
            lines.append(f"{prefix}[permission denied]")
            return
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir() and current_depth < depth:
                extension = "    " if is_last else "│   "
                _walk(entry, current_depth + 1, prefix + extension)

    _walk(p, 1, "")
    return "\n".join(lines)


_SEARCH_CAP = 200
_MAX_OUTPUT_CHARS = 3000
_MAX_FILE_READ_CHARS = 50_000


def _truncate_output(text: str, cap: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [truncated — {len(text)} chars total, showing first {cap}]"


def _search_in_files(pattern: str, path: str = ".", file_pattern: str = "*") -> str:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex: {e}"
    root = Path(path)
    if not root.exists():
        return f"Error: Path not found: {path}"
    if not root.is_dir():
        return f"Error: Not a directory: {path}"
    results: list[str] = []
    capped = False
    for fp in sorted(root.rglob("*")):
        if not fp.is_file() or not fnmatch.fnmatch(fp.name, file_pattern):
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except PermissionError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                results.append(f"{fp}:{lineno}: {line.rstrip()}")
                if len(results) >= _SEARCH_CAP:
                    capped = True
                    break
        if capped:
            break
    if not results:
        return "No matches found."
    out = "\n".join(results)
    out = out + f"\n\n[capped at {_SEARCH_CAP} results]" if capped else out
    return _truncate_output(out)


def _create_file(path: str, content: str) -> str:
    p = Path(path)
    if p.exists():
        return f"Error: File already exists: {path}"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        _snapshot_after_write(path)
        return f"OK: created {path}{_check_syntax(path)}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except OSError as e:
        return f"Error: {e}"


def _recall_memory(query: str, scope: str = "project") -> str:
    try:
        from .memory import current_project_path, search
        project = current_project_path() if scope != "all" else None
        results = search(query, top_k=5, project_path=project, scope=scope)
    except Exception as e:
        return f"Memory search error: {e}"
    if not results:
        return "No matching memories found."
    lines = [f"Found {len(results)} matching memory/memories:\n"]
    for entry in results:
        date = (entry.get("created_at", "") or "")[:10]
        kind = entry.get("kind", "")
        proj = entry.get("project_path", "") or ""
        proj_short = Path(proj).name if proj else ""
        header = f"[{date}]"
        if kind:
            header += f" [{kind}]"
        if proj_short:
            header += f" ({proj_short})"
        lines.append(f"{header}\n{entry.get('summary', '')}\n")
    return "\n".join(lines)


def _save_memory(
    summary: str,
    tags: list[str] | None = None,
    files_touched: list[str] | None = None,
) -> str:
    if not summary or not summary.strip():
        return "Error: summary is required"
    try:
        from .memory import append_entry, current_project_path
        append_entry(
            session_id="tool_call",
            summary=summary.strip(),
            kind="task",
            tags=tags,
            files_touched=files_touched,
            project_path=current_project_path(),
        )
        return "OK: memory saved"
    except Exception as e:
        return f"Error saving memory: {e}"
