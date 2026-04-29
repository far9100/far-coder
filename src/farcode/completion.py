"""prompt_toolkit completer that suggests file paths after `@`.

Activated only when the token under the cursor starts with `@`. Suggests
files and directories relative to the current working directory, matching
the partial path the user has typed so far. Hidden entries (starting with
`.`) are skipped unless the user has explicitly typed a leading `.`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

_MAX_SUGGESTIONS = 30


class AtFileCompleter(Completer):
    """Suggest paths whenever the current word begins with '@'."""

    def get_completions(
        self, document: Document, complete_event
    ) -> Iterable[Completion]:
        word = document.get_word_before_cursor(WORD=True)
        if not word.startswith("@"):
            return
        partial = word[1:]  # strip the leading '@'

        directory, prefix = _split_dir_and_prefix(partial)
        try:
            entries = list(_iter_entries(directory))
        except OSError:
            return

        show_hidden = prefix.startswith(".")
        matches: list[tuple[str, bool]] = []
        for name, is_dir in entries:
            if not show_hidden and name.startswith("."):
                continue
            if not name.lower().startswith(prefix.lower()):
                continue
            matches.append((name, is_dir))

        matches.sort(key=lambda t: (not t[1], t[0].lower()))
        for name, is_dir in matches[:_MAX_SUGGESTIONS]:
            full = name + ("/" if is_dir else "")
            text = (str(directory / name) + ("/" if is_dir else "")) if directory != Path(".") else full
            text = text.replace(os.sep, "/")
            yield Completion(
                text,
                start_position=-len(partial),
                display=full,
                display_meta="dir" if is_dir else "file",
            )


def _split_dir_and_prefix(partial: str) -> tuple[Path, str]:
    """Split `src/far` -> (Path('src'), 'far'); `foo` -> (Path('.'), 'foo').

    Without a separator, the whole token is the prefix in cwd — including
    leading dots like `.gi` (Path() would otherwise discard them via .name).
    """
    if not partial:
        return Path("."), ""
    if partial.endswith("/") or partial.endswith(os.sep):
        return Path(partial), ""
    if "/" not in partial and os.sep not in partial:
        return Path("."), partial
    p = Path(partial)
    parent = p.parent
    return (parent if str(parent) != "" else Path(".")), p.name


def _iter_entries(directory: Path) -> Iterable[tuple[str, bool]]:
    base = directory if directory.is_absolute() else Path.cwd() / directory
    if not base.exists() or not base.is_dir():
        return
    with os.scandir(base) as it:
        for entry in it:
            yield entry.name, entry.is_dir()
