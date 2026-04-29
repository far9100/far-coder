"""Single source of truth for in-session slash commands.

Commands are organised into sections so ``/help`` can render a categorized
table; ``SLASH_COMMANDS`` keeps a flat view for the welcome banner and any
caller that doesn't care about grouping.
"""

SLASH_COMMAND_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Discovery",
        [
            ("/help",        "Show this list"),
            ("/tasks",       "Show in-session task list"),
            ("/rules",       "Show loaded CODER.md rules"),
        ],
    ),
    (
        "Files & code",
        [
            ("/file <path>", "Inject a file into context"),
            ("/diff",        "Show git diff against HEAD"),
            ("/undo",        "Restore the last file mutation"),
            ("/reindex",     "Rebuild code embeddings index"),
            ("/explore <q>", "Run a read-only subagent to investigate a question"),
        ],
    ),
    (
        "Session",
        [
            ("/clear",       "Wipe conversation, start new session"),
            ("/compact",     "Force-summarize older turns"),
            ("/resume",      "Pick a saved session to resume"),
            ("/exit  /quit", "Save summary and exit"),
        ],
    ),
    (
        "Config",
        [
            ("/model [name]", "Get/set the current model"),
        ],
    ),
]


# Flat view: every (name, description) tuple, section order preserved.
SLASH_COMMANDS: list[tuple[str, str]] = [
    item for _, items in SLASH_COMMAND_SECTIONS for item in items
]


def banner_line() -> str:
    """Compact one-line listing for the welcome banner."""
    return "  ".join(name for name, _ in SLASH_COMMANDS)
