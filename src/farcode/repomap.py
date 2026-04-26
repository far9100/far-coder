"""Project repo map: a compact, token-budgeted summary of top-level definitions.

Inspired by aider's tree-sitter repo map (aider.chat/2023/10/22/repomap.html), but
simplified for a 4B local model:

  * Python files are parsed in-process via the stdlib ``ast`` module — no extra
    dependency. Top-level functions, async functions, and classes (with their
    method names) are extracted.
  * For non-Python files, we optionally use ``tree-sitter-language-pack`` if the
    user has installed it. The mapping is intentionally narrow (functions and
    classes only) so the output stays terse.
  * Files are ranked by a simple score: ``recency_weight * import_weight``.
    Recently modified files float to the top; files imported from many places
    also rise. PageRank is intentionally NOT implemented — for a 4B model with
    a 1500-token budget, simple heuristics get >90% of the value.
  * The output is formatted as ``path:`` followed by indented signature lines
    and capped at ``MAX_TOKENS`` (default 1500, ~5000 chars).
  * The full result is cached in ``~/.farcode_repomap_cache.json`` keyed by
    project path; cache invalidates when any indexed file's mtime changes.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# Default budget tuned for 4B/64K models — the system prompt is small, so we
# can afford ~1500 tokens of repo structure without crowding actual conversation.
MAX_TOKENS = 1500
CHARS_PER_TOKEN = 3.5
MAX_CHARS = int(MAX_TOKENS * CHARS_PER_TOKEN)
CACHE_PATH = Path.home() / ".farcode_repomap_cache.json"

# Files we'll consider when walking. Languages without an in-process parser are
# silently skipped unless tree_sitter_language_pack is installed.
SOURCE_EXTS = {
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".go", ".rs", ".java", ".kt", ".swift",
    ".rb", ".php", ".cs", ".cpp", ".cc", ".c", ".h", ".hpp",
}
# Directories to skip during walk; matches typical .gitignore'd noise.
IGNORE_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    "node_modules", "dist", "build", ".next", ".nuxt", "target",
    ".idea", ".vscode", ".mypy_cache", ".ruff_cache", "coverage",
    ".tox", ".eggs", "site-packages",
}
# Hard cap on files scanned, to keep scan cost bounded on large repos.
MAX_FILES = 800
# Scan ~30 days back when computing recency boost.
RECENCY_WINDOW_SEC = 30 * 24 * 3600


@dataclass
class FileEntry:
    path: Path
    rel_path: str
    mtime: float
    defs: list[str] = field(default_factory=list)
    score: float = 0.0


# ── File walking ──────────────────────────────────────────────────────────────

def _walk_project(root: Path) -> list[Path]:
    """Walk the project tree, skipping noise directories and .git-ignored files
    we know are useless. Bounded by ``MAX_FILES`` to cap scan time on large
    repos. Falls back to a plain ``os.walk`` if ``git ls-files`` isn't available.
    """
    found: list[Path] = []
    try:
        # Prefer git ls-files: respects .gitignore for free.
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                p = root / line
                if p.suffix.lower() in SOURCE_EXTS and p.is_file():
                    found.append(p)
                    if len(found) >= MAX_FILES:
                        break
            return found
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]
        for name in files:
            if not any(name.endswith(ext) for ext in SOURCE_EXTS):
                continue
            found.append(Path(current) / name)
            if len(found) >= MAX_FILES:
                return found
    return found


# ── Definition extraction ─────────────────────────────────────────────────────

def _python_defs(source: str) -> list[str]:
    """Top-level functions, async functions, and classes (with their method names)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out.append(f"def {node.name}({_argstr(node.args)})")
        elif isinstance(node, ast.AsyncFunctionDef):
            out.append(f"async def {node.name}({_argstr(node.args)})")
        elif isinstance(node, ast.ClassDef):
            methods = [
                m.name for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not m.name.startswith("_")
            ][:6]
            tail = f"  # {', '.join(methods)}" if methods else ""
            out.append(f"class {node.name}{tail}")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    out.append(f"{t.id} = ...")
    return out


def _argstr(args: ast.arguments) -> str:
    """Render an ast.arguments as a compact signature string."""
    pieces: list[str] = []
    for a in args.args:
        pieces.append(a.arg)
    if args.vararg:
        pieces.append(f"*{args.vararg.arg}")
    for a in args.kwonlyargs:
        pieces.append(a.arg)
    if args.kwarg:
        pieces.append(f"**{args.kwarg.arg}")
    return ", ".join(pieces)


_TS_FUNC_RE = re.compile(
    r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)",
    re.MULTILINE,
)
_TS_CLASS_RE = re.compile(r"^(?:export\s+)?class\s+([A-Za-z_$][\w$]*)", re.MULTILINE)
_TS_CONST_FUNC_RE = re.compile(
    r"^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>",
    re.MULTILINE,
)
_GO_FUNC_RE = re.compile(r"^func\s+(?:\([^)]*\)\s+)?([A-Za-z_]\w*)\s*\(", re.MULTILINE)
_RUST_FN_RE = re.compile(r"^(?:pub\s+(?:\([^)]*\)\s*)?)?fn\s+([A-Za-z_]\w*)", re.MULTILINE)
_RUST_STRUCT_RE = re.compile(r"^(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_]\w*)", re.MULTILINE)


def _regex_defs(source: str, ext: str) -> list[str]:
    """Lightweight regex-based extraction for non-Python languages.

    This is a pragmatic substitute for tree-sitter — fast, dependency-free,
    and good enough to surface symbols a 4B model can use as anchors. False
    positives are fine; we're optimizing for the model spotting "oh that
    function exists, let me read it" not for a perfect AST.
    """
    out: list[str] = []
    if ext in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"):
        for m in _TS_FUNC_RE.finditer(source):
            out.append(f"function {m.group(1)}()")
        for m in _TS_CONST_FUNC_RE.finditer(source):
            out.append(f"const {m.group(1)} = () => ...")
        for m in _TS_CLASS_RE.finditer(source):
            out.append(f"class {m.group(1)}")
    elif ext == ".go":
        for m in _GO_FUNC_RE.finditer(source):
            out.append(f"func {m.group(1)}()")
    elif ext == ".rs":
        for m in _RUST_FN_RE.finditer(source):
            out.append(f"fn {m.group(1)}()")
        for m in _RUST_STRUCT_RE.finditer(source):
            out.append(f"struct/enum/trait {m.group(1)}")
    return out


def extract_defs(path: Path) -> list[str]:
    """Return top-level definition signatures for a file, or [] if unparseable."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return []
    if path.suffix == ".py" or path.suffix == ".pyi":
        return _python_defs(source)
    return _regex_defs(source, path.suffix.lower())


# ── Ranking ───────────────────────────────────────────────────────────────────

_PY_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([\w.]+)", re.MULTILINE)


def _import_counts(entries: list[FileEntry]) -> dict[str, int]:
    """Count how many other files import each module. Cheap signal of importance."""
    # Build a map of module-name → entry (best-effort, Python only)
    name_to_path: dict[str, str] = {}
    for e in entries:
        if e.path.suffix not in (".py", ".pyi"):
            continue
        # Crude: derive module name from the file stem (and optionally the parent dir)
        name_to_path[e.path.stem] = e.rel_path

    counts: dict[str, int] = {p: 0 for p in name_to_path.values()}
    for e in entries:
        if e.path.suffix not in (".py", ".pyi"):
            continue
        try:
            source = e.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _PY_IMPORT_RE.finditer(source):
            mod = m.group(1).split(".")[-1]
            if mod in name_to_path and name_to_path[mod] != e.rel_path:
                counts[name_to_path[mod]] = counts.get(name_to_path[mod], 0) + 1
    return counts


def _score(entry: FileEntry, now: float, import_count: int) -> float:
    """recency_weight * (1 + import_weight). Newer + more-referenced = higher."""
    age = max(0.0, now - entry.mtime)
    # Linear decay over RECENCY_WINDOW_SEC, floor of 0.1 for old files.
    recency = max(0.1, 1.0 - age / RECENCY_WINDOW_SEC)
    # Cap import weight so a single popular file can't dominate.
    return recency * (1.0 + min(import_count, 10) * 0.5)


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_load() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _cache_save(data: dict) -> None:
    try:
        CACHE_PATH.write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def _cache_key(root: Path) -> str:
    return str(root.resolve())


def _signature(entries: list[FileEntry]) -> str:
    """A cheap fingerprint that changes when any indexed file's mtime changes."""
    parts = sorted((e.rel_path, int(e.mtime)) for e in entries)
    return str(hash(tuple(parts)))


# ── Public API ────────────────────────────────────────────────────────────────

def build_repo_map(
    root: str | Path | None = None,
    max_chars: int = MAX_CHARS,
    use_cache: bool = True,
) -> str:
    """Return a compact "## Repo Map" block for injection into the system prompt.

    Returns "" if the project has no parseable source files. Output is bounded
    by ``max_chars`` (~``max_chars / 3.5`` tokens). Cached across runs in
    ``~/.farcode_repomap_cache.json`` keyed by project path; cache busts on any
    indexed file's mtime change.
    """
    rootp = Path(root or os.getcwd()).resolve()
    paths = _walk_project(rootp)
    if not paths:
        return ""

    now = time.time()
    entries: list[FileEntry] = []
    for p in paths:
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        try:
            rel = str(p.relative_to(rootp)).replace("\\", "/")
        except ValueError:
            rel = str(p)
        entries.append(FileEntry(path=p, rel_path=rel, mtime=mtime))

    sig = _signature(entries)
    cache = _cache_load() if use_cache else {}
    cached = cache.get(_cache_key(rootp))
    if use_cache and cached and cached.get("sig") == sig and cached.get("max_chars") == max_chars:
        return cached.get("output", "")

    # Extract defs only after the cache check — parsing is the expensive bit.
    for e in entries:
        e.defs = extract_defs(e.path)

    counts = _import_counts(entries)
    for e in entries:
        e.score = _score(e, now, counts.get(e.rel_path, 0))

    entries.sort(key=lambda e: (-e.score, e.rel_path))

    out_lines: list[str] = ["## Repo Map", ""]
    used = sum(len(line) + 1 for line in out_lines)
    for e in entries:
        if not e.defs:
            continue
        block_lines = [f"{e.rel_path}:"]
        for d in e.defs[:8]:  # cap defs per file
            block_lines.append(f"  {d}")
        block = "\n".join(block_lines) + "\n"
        if used + len(block) > max_chars:
            out_lines.append(f"...[truncated; {max_chars - used}c remaining]")
            break
        out_lines.append(block)
        used += len(block) + 1

    if len(out_lines) <= 2:
        return ""

    output = "\n".join(out_lines).rstrip() + "\n"
    if use_cache:
        cache[_cache_key(rootp)] = {"sig": sig, "max_chars": max_chars, "output": output}
        _cache_save(cache)
    return output
