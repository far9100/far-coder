"""Single source of truth for in-session slash commands.

The registry below drives both the welcome banner and the `/help` command
output, so any command added here is automatically advertised in both.
"""

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help",         "Show this list"),
    ("/exit  /quit",  "Save summary and exit"),
    ("/clear",        "Wipe conversation, start new session"),
    ("/file <path>",  "Inject a file into context"),
    ("/model [name]", "Get/set the current model"),
    ("/compact",      "Force-summarize older turns"),
    ("/rules",        "Show loaded CODER.md rules"),
    ("/undo",         "Restore the last file mutation"),
    ("/diff",         "Show git diff against HEAD"),
    ("/reindex",      "Rebuild code embeddings index"),
    ("/resume",       "Pick a saved session to resume"),
    ("/tasks",        "Show in-session task list"),
]


def banner_line() -> str:
    """Compact one-line listing for the welcome banner."""
    return "  ".join(name for name, _ in SLASH_COMMANDS)
