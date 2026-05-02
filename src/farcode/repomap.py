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
# Above this many parseable source files, skip injection entirely. At repo
# scale (Django, sympy, sklearn ~1k–30k files) the 1500-token cap forces such
# sparse coverage that a 4B model gets more confused by missing names than
# helped by the few anchors that fit. SWE-bench Lite ablations confirmed full
# mode lost django-11179 to bare specifically because of repomap dilution;
# disabling for large repos restores the bare-equivalent edit location.
MAX_FILES_FOR_INJECTION = 100
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
    if len(paths) > MAX_FILES_FOR_INJECTION:
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


# ── Focused repo map (SWE-bench mode) ─────────────────────────────────────────
#
# Problem the focused map exists to solve: ``build_repo_map`` skips injection
# entirely above ``MAX_FILES_FOR_INJECTION`` (100) parseable files, because at
# repo scale a 1500-token map of unrelated structure confuses a 4B model more
# than it helps (results-2026-05-01.md, django-11179 case study). But the
# SWE-bench Lite repos (django, sympy, sklearn, flask, pytest) are ALL well
# above that threshold — exactly the situation where some local context would
# matter most.
#
# When the harness has a list of suspect files (from ``locator.py``), we can
# build a much smaller, targeted map: BFS from the seeds via Python ``import``
# statements (depth-bounded), capturing top-level defs/classes for each
# visited file. Output uses the same format as ``build_repo_map`` so the
# system prompt stays consistent.

_FOCUSED_DEPTH_DEFAULT = 2
_FOCUSED_MAX_FILES = 30  # absolute cap on files visited via BFS

_PY_IMPORT_FROM_RE = re.compile(r"^\s*from\s+([\w.]+)\s+import\b", re.MULTILINE)
_PY_IMPORT_BARE_RE = re.compile(r"^\s*import\s+([\w.]+)", re.MULTILINE)


def _module_to_path(module: str, root: Path, py_files_set: set[str]) -> str | None:
    """Best-effort module → relative-path. Tries ``foo/bar.py``, then
    ``foo/bar/__init__.py``. Returns rel-path with forward slashes."""
    parts = module.split(".")
    candidates = [
        "/".join(parts) + ".py",
        "/".join(parts) + "/__init__.py",
        # Same-package relative imports often resolve as just the leaf
        parts[-1] + ".py",
    ]
    for c in candidates:
        if c in py_files_set:
            return c
    # Suffix match — e.g. seed lives in pkg/sub/x.py and imports "models.foo"
    # which resolves to pkg/models/foo.py somewhere in the repo. We pick the
    # shortest matching path to bias toward top-level packages.
    suffix = "/" + parts[-1] + ".py"
    matches = [f for f in py_files_set if f.endswith(suffix)]
    if matches:
        return min(matches, key=len)
    return None


def _expand_via_imports(
    seeds: list[str],
    root: Path,
    py_files: list[str],
    depth: int,
) -> list[str]:
    """BFS from ``seeds`` via Python import statements. Returns the union of
    seeds + reachable files (forward edges only — we don't compute reverse
    imports here because it'd require reading every .py file in the repo)."""
    py_set = set(py_files)
    visited: set[str] = set()
    frontier = [s for s in seeds if s in py_set]

    for _ in range(max(0, depth) + 1):
        if not frontier or len(visited) >= _FOCUSED_MAX_FILES:
            break
        next_frontier: list[str] = []
        for path in frontier:
            if path in visited:
                continue
            visited.add(path)
            if len(visited) >= _FOCUSED_MAX_FILES:
                break
            try:
                source = (root / path).read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            mods: list[str] = []
            mods.extend(m.group(1) for m in _PY_IMPORT_FROM_RE.finditer(source))
            mods.extend(m.group(1) for m in _PY_IMPORT_BARE_RE.finditer(source))
            for mod in mods:
                resolved = _module_to_path(mod, root, py_set)
                if resolved and resolved not in visited:
                    next_frontier.append(resolved)
        frontier = next_frontier

    # Preserve seed order, then append discovered files in BFS order.
    ordered: list[str] = []
    for s in seeds:
        if s in visited and s not in ordered:
            ordered.append(s)
    for f in visited:
        if f not in ordered:
            ordered.append(f)
    return ordered


def build_focused_repo_map(
    seed_paths: list[str],
    root: str | Path | None = None,
    max_chars: int = MAX_CHARS,
    depth: int = _FOCUSED_DEPTH_DEFAULT,
) -> str:
    """Targeted repo map rooted at ``seed_paths``, expanded via imports.

    Bypasses the ``MAX_FILES_FOR_INJECTION`` cutoff that ``build_repo_map``
    uses — the whole point of the focused map is to inject useful context
    on big repos where the global map self-disables.

    Cache key includes the sorted seed list so different SWE-bench instances
    don't collide.
    """
    if not seed_paths:
        return ""
    rootp = Path(root or os.getcwd()).resolve()
    paths = _walk_project(rootp)
    if not paths:
        return ""

    py_rel: list[str] = []
    for p in paths:
        if p.suffix not in (".py", ".pyi"):
            continue
        try:
            rel = str(p.relative_to(rootp)).replace("\\", "/")
        except ValueError:
            continue
        py_rel.append(rel)

    norm_seeds = [s.replace("\\", "/").strip() for s in seed_paths if s]
    expanded = _expand_via_imports(norm_seeds, rootp, py_rel, depth)
    if not expanded:
        return ""

    # Cache key includes seeds; reuses the same on-disk file as build_repo_map
    # but with a distinct top-level key to avoid collision.
    cache_key = f"{_cache_key(rootp)}|focused|{','.join(sorted(norm_seeds))}|d{depth}"
    cache = _cache_load()
    cached = cache.get(cache_key)
    sig_parts: list[tuple[str, int]] = []
    for rel in expanded:
        try:
            sig_parts.append((rel, int((rootp / rel).stat().st_mtime)))
        except OSError:
            continue
    sig = str(hash(tuple(sorted(sig_parts))))
    if cached and cached.get("sig") == sig and cached.get("max_chars") == max_chars:
        return cached.get("output", "")

    out_lines: list[str] = ["## Repo Map (focused on suspected files)", ""]
    used = sum(len(line) + 1 for line in out_lines)
    for rel in expanded:
        defs = extract_defs(rootp / rel)
        if not defs:
            continue
        block_lines = [f"{rel}:"]
        for d in defs[:8]:
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
    cache[cache_key] = {"sig": sig, "max_chars": max_chars, "output": output}
    _cache_save(cache)
    return output
