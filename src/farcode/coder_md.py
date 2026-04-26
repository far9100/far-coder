"""Project-level coding rules loaded from CODER.md files.

Discovery order (lowest precedence first, so later files override earlier ones
in the assembled prompt):

1. ``~/.coder.md``                           — user-wide preferences
2. ``CODER.md`` walking upward from cwd      — outermost ancestor first,
   project root last. Walking stops at the directory containing ``.git``
   (inclusive) or the filesystem root.
"""

from pathlib import Path

GLOBAL_PATH = Path.home() / ".coder.md"
FILENAME = "CODER.md"
_MAX_BYTES = 16 * 1024  # 16 KB per file — keeps a runaway rules file from eating context
_TRUNC_NOTICE = "\n\n[...truncated: CODER.md exceeded 16 KB...]"


def _walk_up(start: Path) -> list[Path]:
    """Return CODER.md files from outermost ancestor down to ``start``.

    Stops at the directory containing ``.git`` (inclusive) or the filesystem
    root. Outermost-first ordering means project-root rules are appended last
    in the prompt and therefore take precedence over any parent rules.
    """
    found: list[Path] = []
    current = start.resolve()
    while True:
        candidate = current / FILENAME
        if candidate.is_file():
            found.append(candidate)
        if (current / ".git").exists():
            break
        if current.parent == current:
            break
        current = current.parent
    return list(reversed(found))


def find_coder_md_files(start: Path | None = None) -> list[Path]:
    """Return all CODER.md files to load, in prompt-append order."""
    files: list[Path] = []
    if GLOBAL_PATH.is_file():
        files.append(GLOBAL_PATH)
    files.extend(_walk_up(start or Path.cwd()))
    return files


def _read_truncated(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) <= _MAX_BYTES:
        return raw.decode("utf-8", errors="replace")
    return raw[:_MAX_BYTES].decode("utf-8", errors="replace") + _TRUNC_NOTICE


def load_coder_md(start: Path | None = None) -> str:
    """Return formatted CODER.md content for injection into the system prompt.

    Empty string when no rules files exist.
    """
    files = find_coder_md_files(start)
    if not files:
        return ""
    blocks: list[str] = []
    for path in files:
        try:
            text = _read_truncated(path).strip()
        except OSError:
            continue
        if not text:
            continue
        blocks.append(f"## Project Rules ({path})\n\n{text}")
    return "\n\n".join(blocks)
