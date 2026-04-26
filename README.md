# farcode

A local AI coding assistant CLI that runs entirely on your machine via [Ollama](https://ollama.com). Chat with a local LLM, attach files, and let the AI read, edit, search, and create files in your project through an agentic tool loop.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- A pulled model, e.g. `ollama pull qwen3.5:4b`

## Installation

```bash
uv pip install -e .
```

## Commands

### `ask` — Single-turn question

```bash
farcode ask "What is a context manager?"
farcode ask "Review this function" -f src/main.py
farcode ask "Explain the bug" -f app.py --model llama3.2
```

### `chat` — Interactive multi-turn session

```bash
farcode chat                       # fresh session
farcode chat -c                    # resume last session
farcode chat -f src/main.py        # pre-load a file
farcode chat -b                    # background mode (type while AI works)
farcode chat --allow-bash          # skip confirmation on shell commands
farcode chat --allow-all           # auto-approve ALL tool calls, no prompts
```

**In-session commands:**

| Command | Description |
|---------|-------------|
| `/clear` | Start a new session |
| `/file <path>` | Inject a file into the conversation |
| `/model <name>` | Switch model mid-session |
| `/resume` | Pick a saved session to resume |
| `/rules` | Show the currently-loaded `CODER.md` rules |
| `/exit` or `/quit` | Exit chat |

### `review` — Code review

```bash
farcode review src/auth.py
```

### `explain` — Code explanation

```bash
farcode explain src/utils.py
```

### `sessions` — Manage saved sessions

```bash
farcode sessions list              # list recent sessions
farcode sessions list -n 50        # show up to 50 sessions
farcode sessions search "auth"     # search by title
farcode sessions delete <id>       # delete by full ID or unique prefix
```

Sessions are auto-saved to `~/.farcode_sessions/` after each turn.

## AI Tools

During `chat`, the AI can use these tools autonomously:

| Tool | Description |
|------|-------------|
| `read_file(path)` | Read a file's full content |
| `edit_file(path, old_str, new_str)` | Precise in-place string replacement |
| `run_bash(command)` | Run a shell command and get stdout+stderr |
| `list_directory(path, depth=2)` | Tree view of a directory (max depth 5) |
| `search_in_files(pattern, path, file_pattern)` | Regex search across files (cap: 200 results) |
| `create_file(path, content)` | Create a new file (fails if already exists) |

## Project rules: `CODER.md`

Drop a `CODER.md` file in your project root and farcode will append its contents to the system prompt for every `chat` and `ask` invocation. Use it for project-specific conventions the model should respect — coding style, framework choices, files to leave alone.

```markdown
# CODER.md

- Use type hints on all public functions.
- Prefer `pathlib.Path` over `os.path`.
- Never edit files under `vendor/`.
- Tests live in `tests/`, not next to source.
```

**Discovery order** (later files override earlier ones in the assembled prompt):

1. `~/.coder.md` — your personal rules, applied to every project
2. Each `CODER.md` walking up from the current directory to the repo root (the directory containing `.git`). Outermost ancestor first, project root last.

Each file is capped at 16 KB. Use `/rules` inside `chat` to inspect what was loaded.

## Safety: run_bash

By default, the AI will ask for your confirmation before running any shell command:

```
Allow bash command?
  pytest tests/
[y/N]
```

To skip these prompts in a trusted environment:

```bash
farcode chat --allow-bash   # skip confirmation for shell commands only
farcode chat --allow-all    # auto-approve ALL tool calls without any prompt
```

On **Windows**, `run_bash` uses PowerShell (`pwsh` or `powershell`) so that
Unix-style commands like `mv`, `ls`, and `cp` work out of the box.

> Never use `--allow-all` or `--allow-bash` on untrusted input or in automated pipelines.
