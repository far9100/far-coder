import fnmatch
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


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file's contents. By default returns the whole file (capped at "
                "~50 KB). Use offset and limit (1-based line numbers) to read a specific "
                "slice of a large file."
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
                "Edit a file by replacing an exact string with new content. "
                "old_str must match exactly (whitespace and indentation included). "
                "Only the first occurrence is replaced."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to edit"},
                    "old_str": {
                        "type": "string",
                        "description": "Exact string to find and replace",
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
            "name": "write_file",
            "description": (
                "Write content to a file, creating it if it does not exist or overwriting it if it does. "
                "Use this whenever you need to create or fully replace a file."
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
                "Use PowerShell syntax: chain with `;` not `&&`, use `Remove-Item -Recurse -Force` not `rm -rf`."
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
        return f"OK: {action} {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except OSError as e:
        return f"Error: {e}"


def _edit_file(path: str, old_str: str, new_str: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    content = p.read_text(encoding="utf-8", errors="replace")
    if old_str not in content:
        return (
            f"Error: old_str not found in {path}.\n"
            "Make sure whitespace and indentation match exactly."
        )
    count = content.count(old_str)
    p.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
    note = f" ({count} occurrences, only first replaced)" if count > 1 else ""
    return f"OK: edited {path}{note}"


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
        return f"OK: created {path}"
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
