"""Bug locator: extract suspect file paths and symbols from a SWE-bench
problem statement. Pure heuristics — no LLM call.

The output is consumed by ``solve.py`` in two ways:

1. Prepended to the user prompt as a ``## Suspected locations`` block — a
   short ranked list giving the model a starting point so it stops
   wandering through unrelated files.
2. Exported to the farcode subprocess as ``FARCODE_REPOMAP_SEEDS=path1,path2``
   so the focused repomap can BFS through imports rooted at these files.

Per results-2026-05-01.md the dominant failure mode for `qwen3.5:4b` is
picking the wrong location, not failing at tool use. Even a noisy 5-suspect
list typically contains the right file, which is enough anchoring for the
4B model to converge on the bug site instead of producing 500-line patches
in unrelated structure.

Extractors:
  * tracebacks  — Python ``File "X", line N, in func`` frames (highest signal)
  * paths       — code-extension tokens that resolve to tracked files
  * symbols     — ``def foo`` / ``class Foo`` / `Foo.bar(...)` mentions, then
                  grepped for definitions in the workspace
  * code blocks — triple-backtick blocks scanned again for symbols/paths

If the problem statement is pure English with no explicit anchors, locate()
returns ``[]`` and the harness simply skips the ``## Suspected locations``
section — falling back to today's behavior.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ── Regex catalog ─────────────────────────────────────────────────────────────

# Path-like tokens. Restricted to common code/config extensions and
# conservative on leading chars (avoid catching URLs and jinja tags).
_PATH_RE = re.compile(
    r"(?:^|[\s`(\[<\"',])"
    r"([a-zA-Z][\w./_-]*?\.(?:py|pyx|pyi|js|jsx|mjs|ts|tsx|go|rs|java|kt|swift|"
    r"rb|php|cs|cpp|cc|c|h|hpp|sql|yml|yaml|toml|cfg|ini))"
    r"(?:[\s`)\]>:\"',]|$)",
    re.MULTILINE,
)

# Python traceback frames: ``File "X", line N, in func``
_PY_TRACE_RE = re.compile(r'File "([^"]+)", line (\d+), in (\w+)')

# `def foo` / `class Foo` / `async def foo` mentions
_DEF_RE = re.compile(r"\b(?:async\s+def|def|class)\s+([A-Za-z_]\w*)")
# Backtick-quoted symbols, optionally dotted, optionally with parens
_BACKTICK_SYMBOL_RE = re.compile(r"`([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\(?\)?`")
# ``ClassName.method(`` style call, where ClassName starts with uppercase
_DOTTED_CALL_RE = re.compile(r"\b([A-Z]\w*(?:\.[a-z_]\w*)+)\(")

# Triple-backtick code blocks (greedy across newlines)
_CODE_BLOCK_RE = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)

# Symbols that match the regexes but carry no localization signal.
_NOISE_SYMBOLS = frozenset({
    "True", "False", "None", "self", "cls", "args", "kwargs", "kwarg",
    "this", "that", "Error", "Exception", "Warning", "BaseException",
    "object", "list", "dict", "set", "tuple", "str", "int", "bool",
    "float", "bytes", "type", "len", "range", "open", "print", "input",
})


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Suspect:
    """One ranked suspected location.

    ``confidence`` is a float in [0, 1] used only for ranking — values are
    relative, not calibrated. ``reason`` is a short human-readable origin
    tag rendered in the prompt section so the model knows whether to trust
    the suspect (a traceback frame is far stronger than a backtick mention).
    """
    path: str
    line_range: tuple[int, int] | None  # 1-based inclusive, or None
    reason: str
    confidence: float

    def format(self) -> str:
        if self.line_range:
            lo, hi = self.line_range
            loc = f"{self.path}:{lo}" if lo == hi else f"{self.path}:{lo}-{hi}"
        else:
            loc = self.path
        return f"- {loc}  ({self.reason})"


# ── Extraction primitives ─────────────────────────────────────────────────────

def _normpath(p: str) -> str:
    return p.replace("\\", "/").strip().rstrip(".,)\"'>`]")


def extract_paths(text: str) -> list[str]:
    """Path-like tokens with a known code/config extension. De-duplicated,
    insertion-order-preserving."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _PATH_RE.finditer(text):
        p = _normpath(m.group(1))
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def extract_traceback_frames(text: str) -> list[tuple[str, int, str]]:
    """``[(path, line_num, func_name)]`` from Python ``File "..."`` frames."""
    return [
        (_normpath(m.group(1)), int(m.group(2)), m.group(3))
        for m in _PY_TRACE_RE.finditer(text)
    ]


def extract_symbols(text: str) -> list[str]:
    """Function / class identifiers loosely mentioned. Conservative — drops
    very short tokens and a small noise blocklist."""
    seen: set[str] = set()
    out: list[str] = []

    def _add(s: str) -> None:
        if not s or s in seen:
            return
        # Strip dotted prefix to get the leaf identifier — the leaf is what
        # we'll grep for. We still record the full thing for the prompt
        # but de-dup on the leaf to avoid Foo and Foo.bar both appearing.
        leaf = s.rsplit(".", 1)[-1]
        if len(leaf) < 3 or leaf in _NOISE_SYMBOLS:
            return
        seen.add(s)
        out.append(s)

    for m in _DEF_RE.finditer(text):
        _add(m.group(1))
    for m in _BACKTICK_SYMBOL_RE.finditer(text):
        _add(m.group(1))
    for m in _DOTTED_CALL_RE.finditer(text):
        _add(m.group(1))
    return out


def extract_code_blocks(text: str) -> list[str]:
    """Triple-backtick code-block contents. Useful for nested re-extraction."""
    return [m.group(1) for m in _CODE_BLOCK_RE.finditer(text)]


# ── Workspace probing ─────────────────────────────────────────────────────────

def _git_ls_files(workspace: Path) -> list[str]:
    """Tracked files in the workspace via git. Forward-slash relative paths.
    Returns ``[]`` on any error so the locator degrades to "no suspects"
    rather than crashing the harness."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(workspace), "ls-files"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def resolve_paths(suspects: list[str], all_files: list[str]) -> list[str]:
    """Map each suspect string to an actual workspace file. Suffix-matches
    (``models/deletion.py`` → ``django/db/models/deletion.py``); on tie,
    keeps the shortest path. Returns paths in input order, deduplicated."""
    resolved: list[str] = []
    seen: set[str] = set()
    file_set = set(all_files)
    for sp in suspects:
        match: str | None = None
        if sp in file_set:
            match = sp
        else:
            sfx = "/" + sp
            candidates = [f for f in all_files if f == sp or f.endswith(sfx)]
            if candidates:
                match = min(candidates, key=len)
        if match and match not in seen:
            seen.add(match)
            resolved.append(match)
    return resolved


def _run_grep_pattern(
    workspace: Path, pattern: str, max_hits: int,
) -> list[tuple[str, int]] | None:
    """Run ripgrep with a single pattern; return ``[(path, line)]`` capped at
    ``max_hits`` on success, or ``None`` if rg is unavailable so the caller
    can fall back to the per-file Python loop."""
    try:
        proc = subprocess.run(
            ["rg", "--no-heading", "-n", "-N", "--max-count", str(max_hits),
             "-g", "*.py", "-g", "*.pyi", pattern, "."],
            cwd=workspace, capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    if proc.returncode not in (0, 1):  # 0 = match, 1 = no match — both ok
        return None
    hits: list[tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 2:
            continue
        try:
            hits.append(
                (parts[0].lstrip("./").replace("\\", "/"), int(parts[1]))
            )
        except ValueError:
            continue
        if len(hits) >= max_hits:
            break
    return hits


def grep_symbol(
    workspace: Path,
    symbol: str,
    py_files: list[str],
    max_hits: int = 2,
) -> list[tuple[str, int]]:
    """Find Python files that *define* ``symbol`` (def/class). Returns
    ``[(path, line_num)]`` capped at ``max_hits``. Uses ripgrep when
    available, falls back to per-file Python regex.

    We deliberately match definitions, not references — references add
    noise and the model can navigate from a definition to its callers
    much more easily than the reverse.
    """
    leaf = symbol.rsplit(".", 1)[-1]
    if not leaf or leaf in _NOISE_SYMBOLS:
        return []

    # ripgrep is way faster on large repos (Django: ~5k .py files).
    pattern = rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(leaf)}\s*[\(:]"
    rg_hits = _run_grep_pattern(workspace, pattern, max_hits)
    if rg_hits:
        return rg_hits
    # If ripgrep returned [] explicitly (vs None for unavailable) we still
    # want to try the per-file fallback so test workspaces without rg work.
    if rg_hits is not None:
        # Try extension-point patterns BEFORE the per-file fallback so
        # printer / visitor methods get found cheaply at the rg layer.
        ext_hits = grep_extension_point(workspace, leaf, max_hits=max_hits)
        if ext_hits:
            return [(p, ln) for p, ln, _kind in ext_hits]

    # Pure-Python fallback. Bounded scan: searching all of django takes
    # ~1s, fine for a one-time call per instance.
    regex = re.compile(pattern, re.MULTILINE)
    hits = []
    for f in py_files:
        if len(hits) >= max_hits:
            break
        try:
            content = (workspace / f).read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        m = regex.search(content)
        if m:
            line_num = content[: m.start()].count("\n") + 1
            hits.append((f, line_num))
    if hits:
        return hits

    # Final fallback: extension-point patterns (printer, visitor) for
    # symbols that aren't defined themselves but extend a base class via
    # naming convention. See grep_extension_point for the rationale.
    ext_hits = grep_extension_point(workspace, leaf, max_hits=max_hits, py_files=py_files)
    return [(p, ln) for p, ln, _kind in ext_hits]


# Common "extension by naming convention" patterns. The model often mentions
# the *concept* (e.g. "Max in mathematica printer") not the actual method
# name (`_print_Max`); we reverse-engineer the method name. Keeping this as
# a small explicit list is a deliberate choice over inferring it — a 4B
# model misled by an over-eager pattern would be worse than no anchor at all.
EXTENSION_POINT_PATTERNS: list[tuple[str, str]] = [
    ("_print", "_print_{}"),    # SymPy printer methods
    ("visit",  "visit_{}"),     # ast.NodeVisitor / similar
    ("p_",     "p_{}"),         # ply parser productions
]


def grep_extension_point(
    workspace: Path,
    leaf: str,
    *,
    max_hits: int = 2,
    py_files: list[str] | None = None,
) -> list[tuple[str, int, str]]:
    """Find Python def-sites that extend a known base class by naming
    convention — e.g. ``_print_Max`` for a SymPy mathematica printer when
    the issue mentions the symbol ``Max``. Returns ``[(path, line, kind)]``
    where ``kind`` is the prefix (``_print``, ``visit``, ...).

    This is the lever for the M5 sympy-15345 regression: the locator pointed
    at the right *file* but the model picked the wrong section (a function-
    map dict). Surfacing line numbers of the relevant ``_print_<symbol>``
    methods gives the model a section-level anchor.
    """
    if not leaf or leaf in _NOISE_SYMBOLS:
        return []
    out: list[tuple[str, int, str]] = []
    for kind, fmt in EXTENSION_POINT_PATTERNS:
        method = fmt.format(leaf)
        pattern = rf"^\s*(?:async\s+)?def\s+{re.escape(method)}\s*\("
        rg_hits = _run_grep_pattern(workspace, pattern, max_hits)
        if rg_hits is None and py_files is not None:
            # Per-file fallback when ripgrep unavailable. Bounded by py_files.
            regex = re.compile(pattern, re.MULTILINE)
            rg_hits = []
            for f in py_files:
                if len(rg_hits) >= max_hits:
                    break
                try:
                    content = (workspace / f).read_text(
                        encoding="utf-8", errors="replace"
                    )
                except (OSError, UnicodeDecodeError):
                    continue
                m = regex.search(content)
                if m:
                    line_num = content[: m.start()].count("\n") + 1
                    rg_hits.append((f, line_num))
        if rg_hits:
            for path, line_num in rg_hits:
                out.append((path, line_num, kind))
                if len(out) >= max_hits:
                    return out
    return out


# ── Public API ────────────────────────────────────────────────────────────────

MAX_SUSPECTS = 5
SECTION_BUDGET_CHARS = 400


def locate(problem_statement: str, workspace: Path) -> list[Suspect]:
    """Run all extractors against the problem statement and return a ranked
    list of suspects (≤ MAX_SUSPECTS). Confidence ordering:

      * 1.0 — Python traceback frame (we know exact path + line)
      * 0.8 — explicit file path mentioned and present in workspace
      * 0.6 — symbol that grep found defined in a workspace file

    Code blocks inside the problem statement are scanned in a second pass
    for paths/tracebacks so a snippet like::

        Traceback (most recent call last):
          File "django/db/models/deletion.py", line 277, in delete
        ...

    is picked up even when it sits inside triple-backtick-fenced text.
    """
    if not problem_statement:
        return []

    all_files = _git_ls_files(workspace)
    if not all_files:
        return []
    py_files = [f for f in all_files if f.endswith(".py") or f.endswith(".pyi")]

    # Code blocks often contain the most actionable signal — scan them too.
    expanded = problem_statement + "\n" + "\n".join(extract_code_blocks(problem_statement))

    suspects: list[Suspect] = []
    seen_paths: set[str] = set()

    def _add(s: Suspect) -> None:
        if s.path in seen_paths:
            return
        seen_paths.add(s.path)
        suspects.append(s)

    # 1) Tracebacks (highest confidence — exact line is known)
    for raw_path, line, func in extract_traceback_frames(expanded):
        resolved = resolve_paths([raw_path], all_files)
        if not resolved:
            continue
        _add(Suspect(
            path=resolved[0],
            line_range=(max(1, line - 5), line + 5),
            reason=f"traceback frame, {func}()",
            confidence=1.0,
        ))

    # 2) Path mentions that resolve to real files
    if len(suspects) < MAX_SUSPECTS:
        for path_token in extract_paths(expanded):
            if len(suspects) >= MAX_SUSPECTS:
                break
            resolved = resolve_paths([path_token], all_files)
            if not resolved:
                continue
            _add(Suspect(
                path=resolved[0],
                line_range=None,
                reason="mentioned by name in issue",
                confidence=0.8,
            ))

    # 3) Symbol mentions → grep for definitions
    if len(suspects) < MAX_SUSPECTS:
        for sym in extract_symbols(expanded):
            if len(suspects) >= MAX_SUSPECTS:
                break
            leaf = sym.rsplit(".", 1)[-1]
            for path, line_num in grep_symbol(workspace, sym, py_files, max_hits=2):
                if path in seen_paths:
                    continue
                _add(Suspect(
                    path=path,
                    line_range=(max(1, line_num - 2), line_num + 10),
                    reason=f"defines '{leaf}'",
                    confidence=0.6,
                ))
                if len(suspects) >= MAX_SUSPECTS:
                    break
            # 3b) Extension-point patterns (printer/visitor methods named after
            # the symbol). Targets the M5 sympy-15345 regression where the model
            # found the right file but edited the wrong section. We add these as
            # ADDITIONAL suspects (not a fallback) — both the symbol's def-site
            # AND its `_print_<symbol>` method may matter, and surfacing the
            # method line number is the actionable hint.
            if len(suspects) >= MAX_SUSPECTS:
                break
            for path, line_num, kind in grep_extension_point(
                workspace, leaf, max_hits=2, py_files=py_files,
            ):
                if path in seen_paths:
                    continue
                method = "_print_" + leaf if kind == "_print" else (
                    "visit_" + leaf if kind == "visit" else f"{kind}_{leaf}"
                )
                _add(Suspect(
                    path=path,
                    line_range=(max(1, line_num - 2), line_num + 10),
                    reason=f"defines {method} (extension point)",
                    confidence=0.7,  # higher than plain symbol — much more specific
                ))
                if len(suspects) >= MAX_SUSPECTS:
                    break

    suspects.sort(key=lambda s: -s.confidence)
    return suspects[:MAX_SUSPECTS]


def format_section(suspects: list[Suspect]) -> str:
    """Render suspects as a Markdown section. Empty string when no
    suspects — caller should skip injection entirely so the model isn't
    handed a misleading "no locations found" header."""
    if not suspects:
        return ""
    header = "## Suspected locations (from issue text — verify before editing)"
    body = "\n".join(s.format() for s in suspects)
    section = header + "\n" + body
    if len(section) > SECTION_BUDGET_CHARS:
        section = section[: SECTION_BUDGET_CHARS - 4].rstrip() + "\n..."
    return section


def seeds_for_repomap(suspects: list[Suspect]) -> str:
    """Comma-separated paths suitable for ``FARCODE_REPOMAP_SEEDS``. Capped
    at the top-3 suspects to keep the focused repomap small."""
    if not suspects:
        return ""
    return ",".join(s.path for s in suspects[:3])
