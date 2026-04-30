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
farcode chat --allow-web           # enable scoped fetch_doc tool (PyPI/npm/crates/pkg.go.dev)
```

**In-session commands** — type `/help` at any time for the categorized list (Discovery / Files & code / Session / Config sections):

| Command | Description |
|---------|-------------|
| `/help` | List all slash commands, grouped by section |
| `/exit` or `/quit` | Save summary and exit |
| `/clear` | Wipe conversation, start a new session |
| `/file <path>` | Inject a file into the conversation |
| `/model [name]` | Get or set the current model |
| `/compact` | Force-summarize older turns to free context |
| `/rules` | Show the currently-loaded `CODER.md` rules |
| `/undo` | Revert the last file write made by the AI |
| `/diff` | Show the working-tree git diff |
| `/reindex` | Force-rebuild the code-chunk embedding index |
| `/resume` | Pick a saved session to resume |
| `/tasks` | Show the in-session task list (see [Tasks](#in-session-tasks)) |
| `/explore <q>` | Run a [read-only sub-agent](#read-only-sub-agents) on a question, without going through the main agent |

**Attaching files inline** — type `@path/to/file` anywhere in your message to inline its contents. The prompt offers tab completion after `@`. Files larger than 100 KB or detected as binary (null bytes in the first 8 KB) are skipped with a printed reason; misses also report a reason ("not found", "not a regular file", "binary file", etc.) instead of being silently dropped.

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
| `read_file(path, offset?, limit?)` | Read a file (or 1-based line slice). 50 KB cap. |
| `edit_file(path, old_str, new_str)` | String replace with exact + whitespace-tolerant matching |
| `replace_lines(path, start, end, new_content)` | Multi-line replace by line range |
| `write_file(path, content)` | Create or overwrite a file (safe fallback) |
| `create_file(path, content)` | Create a new file (fails if already exists) |
| `run_bash(command, timeout?)` | Run a PowerShell command (Windows) / bash (Unix) |
| `list_directory(path, depth=2)` | Tree view of a directory (max depth 5) |
| `search_in_files(pattern, path, file_pattern)` | Regex search across files (cap: 200 results) |
| `recall_code(query, top_k?)` | Semantic + keyword search over code chunks (needs `nomic-embed-text`) |
| `recall_memory(query, scope?)` | Search past session memories (BM25 over FTS5) |
| `save_memory(summary, tags?, files_touched?)` | Persist a lesson for future sessions |
| `task_create(content)` | Create a pending task in the in-session todo list |
| `task_update(id, status)` | Mark a task `pending` / `in_progress` / `completed` |
| `task_list()` | List all current tasks with status |
| `explore_subagent(question, focus_area?)` | Delegate a focused investigation to a read-only sub-agent |
| `fetch_doc(query, ecosystem?)` | Look up package metadata on PyPI/npm/crates.io/pkg.go.dev (requires `--allow-web`) |

After every file write, farcode runs a per-language **syntax check** (Python `ast`,
`json.loads`, optional `node --check` / `tsc --noEmit`) and appends the result to
the tool output, so the model sees errors and can self-correct on the next turn.

### In-session tasks

For requests that take 3+ steps, the model is encouraged to call `task_create`
up front and `task_update` as it progresses, giving you a visible plan. Type
`/tasks` to see the current list at any time:

```
○ a3f2c1: read auth.py
→ b8e102: extract login flow
✓ c0d1f4: write integration test
```

Tasks live on `Session.tasks` and are persisted to disk with the rest of the
session, so they survive `/resume`. Whenever the model calls `task_create` or
`task_update`, farcode auto-renders the full task list once at the end of that
tool batch so progress stays visible without typing `/tasks`.

### Read-only sub-agents

`explore_subagent` runs an isolated agent loop with its own message history
and only the read-only tools (`read_file`, `list_directory`, `search_in_files`,
`recall_code`, `recall_memory`). It returns a single text summary back to the
parent agent, so noisy investigation chatter never pollutes the main
conversation. Use it for "trace this feature across many files" or "how does X
work" questions.

Constraints (enforced):

- Cannot itself call `explore_subagent` (depth cap = 1).
- Cannot mutate state — write/edit/replace/create/run_bash/save_memory are
  refused at execution time.
- Capped at 8 tool calls per sub-agent run before being forced to return.
- Set `FARCODE_SUBAGENT_MODEL` to use a smaller/cheaper model for exploration
  while the main agent keeps its larger model.

When a sub-agent runs you'll see live `Subagent exploring: ...` and `Subagent
done: N tool call(s), X chars` lines so the wait is never silent. You can also
trigger one directly with `/explore <question>` (skips the main agent).

### Web access (opt-in)

`fetch_doc` is **disabled by default** to preserve the local-first stance. When
enabled with `--allow-web` (or `FARCODE_ALLOW_WEB=1`), it can fetch package
metadata from a hard-coded allowlist:

- **PyPI** — `pypi.org/pypi/<pkg>/json`
- **npm** — `registry.npmjs.org/<pkg>`
- **crates.io** — `crates.io/api/v1/crates/<pkg>`
- **pkg.go.dev** — `pkg.go.dev/<module>`

JSON registry responses are summarized to name/version/license/homepage; HTML
pages are stripped of script/style/nav/footer. All responses are capped at
8 KB. Package names are validated to reject path-traversal and shell
metacharacters. No general URL fetch is supported; no search-engine integration.

Successful responses are cached on disk at `~/.farcode_doc_cache.json` for 24 h
(LRU-pruned at 200 entries) so repeat lookups skip the network. Cached results
are tagged `[cached]` in the returned text. Errors (4xx / 5xx / network) are
never cached.

## Context the AI sees

On every chat turn, farcode auto-injects the following into the system prompt
(in this order, so the model attends most to the things at the bottom):

1. **Base system prompt** (~250 tokens — terse workflow rules)
2. **`CODER.md` project rules** (your conventions; see below)
3. **Project facts** — language, package manager, test runner, entry points
   (extracted once per project, cached in `~/.farcode_memory.db`)
4. **Repo map** — top-level definitions ranked by recency × import count, capped
   at ~1500 tokens. Cached in `~/.farcode_repomap_cache.json` keyed by project
   path; busts on file mtime change.
5. **Past Work** — up to 3 memory entries, FTS-matched against your first message
   (or most-recent if no match), project-scoped first then global.

## Reliability features for small (4B) models

Local 4B models are far less reliable at tool-calling than frontier models.
Farcode includes several mitigations:

- **`edit_file` falls back to whitespace-tolerant matching** when an exact
  string match fails. Catches mis-indented `old_str` from the model.
- **One tool call per turn by default** (`FARCODE_MAX_TOOLS_PER_TURN=1`).
  Small models often emit one good call plus one malformed one; serial
  execution dramatically improves success rate. Set to `0` for unlimited.
- **Grammar-constrained tool retry**: when Ollama 500s on a malformed
  tool call, farcode retries once with `format=<json_schema>` constraining
  output to a valid `{tool_call|answer}` shape, then falls back to plain
  chat only if that also fails.
- **Post-edit syntax check** lets the model see and fix its own mistakes.
- **`/undo`** restores files if a tool call did the wrong thing.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `FARCODE_NUM_CTX` | `65536` | Ollama context window in tokens |
| `FARCODE_NUM_PREDICT` | `4096` | Max output tokens reserved per turn |
| `FARCODE_MAX_TOOLS_PER_TURN` | `1` | Cap on tool calls per turn (0 = unlimited) |
| `FARCODE_SUMMARY_MODEL` | (unset) | Smaller model for compaction/summary |
| `FARCODE_SUBAGENT_MODEL` | (unset) | Smaller model for `explore_subagent` runs |
| `FARCODE_EMBED_MODEL` | `nomic-embed-text` | Embedding model for `recall_code` |
| `FARCODE_ALLOW_WEB` | `0` | Set to `1` to enable `fetch_doc` without the CLI flag |

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
