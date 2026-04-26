# ai-coder

An AI coding assistant CLI that runs entirely locally via [Ollama](https://ollama.com). Chat with a local LLM, attach files, and let the AI read, edit, search, and create files in your project through an agentic tool loop.

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
ai-coder ask "What is a context manager?"
ai-coder ask "Review this function" -f src/main.py
ai-coder ask "Explain the bug" -f app.py --model llama3.2
```

### `chat` — Interactive multi-turn session

```bash
ai-coder chat                       # fresh session
ai-coder chat -c                    # resume last session
ai-coder chat -f src/main.py        # pre-load a file
ai-coder chat -b                    # background mode (type while AI works)
ai-coder chat --allow-bash          # skip confirmation on shell commands
```

**In-session commands:**

| Command | Description |
|---------|-------------|
| `/clear` | Start a new session |
| `/file <path>` | Inject a file into the conversation |
| `/model <name>` | Switch model mid-session |
| `/resume` | Pick a saved session to resume |
| `/exit` or `/quit` | Exit chat |

### `review` — Code review

```bash
ai-coder review src/auth.py
```

### `explain` — Code explanation

```bash
ai-coder explain src/utils.py
```

### `sessions` — Manage saved sessions

```bash
ai-coder sessions list              # list recent sessions
ai-coder sessions list -n 50        # show up to 50 sessions
ai-coder sessions search "auth"     # search by title
ai-coder sessions delete <id>       # delete by full ID or unique prefix
```

Sessions are auto-saved to `~/.ai_coder_sessions/` after each turn.

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

## Safety: run_bash

By default, the AI will ask for your confirmation before running any shell command:

```
Allow bash command?
  pytest tests/
[y/N]
```

To skip these prompts in a trusted environment:

```bash
ai-coder chat --allow-bash
```

> Never use `--allow-bash` on untrusted input or in automated pipelines.
