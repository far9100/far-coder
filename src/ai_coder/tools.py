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


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
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
            "name": "run_bash",
            "description": (
                "Run a shell command and return stdout + stderr. "
                "Use for running tests, building, grepping, or any shell task."
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
]


def execute_tool(name: str, arguments: dict) -> str:
    handlers = {
        "read_file": _read_file,
        "edit_file": _edit_file,
        "run_bash": _run_bash,
        "list_directory": _list_directory,
        "search_in_files": _search_in_files,
        "create_file": _create_file,
    }
    handler = handlers.get(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        return handler(**{k: v for k, v in arguments.items() if k != "timeout" or name == "run_bash"})
    except TypeError as e:
        return f"Tool call error (bad arguments): {e}"
    except Exception as e:
        return f"Tool error: {e}"


def _read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    if not p.is_file():
        return f"Error: Not a file: {path}"
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return f"Error: Permission denied: {path}"


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
    return output


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
    return out + f"\n\n[capped at {_SEARCH_CAP} results]" if capped else out


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
