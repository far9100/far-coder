"""One-shot codebase fact extraction.

The first time farcode runs in a project, scan for obvious metadata: language,
package manager, test runner, entry points. Store the result as a memory entry
with kind="codebase_facts" so future sessions can recall it without re-scanning.
Inject the formatted facts into the system prompt under "## Project Facts".

Re-extraction policy: if the project's marker files (pyproject.toml,
package.json, ...) have changed mtime since the cached facts were written, we
re-scan. Otherwise we trust the cache.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Files whose presence implies a particular package manager / runtime.
_MARKERS: dict[str, list[tuple[str, str]]] = {
    "python": [
        ("pyproject.toml", "uv / pip / poetry"),
        ("setup.py", "setuptools"),
        ("requirements.txt", "pip"),
    ],
    "node": [
        ("package.json", "npm / pnpm / yarn"),
        ("pnpm-lock.yaml", "pnpm"),
        ("yarn.lock", "yarn"),
    ],
    "rust": [("Cargo.toml", "cargo")],
    "go": [("go.mod", "go modules")],
    "ruby": [("Gemfile", "bundler")],
    "java": [("pom.xml", "maven"), ("build.gradle", "gradle")],
}

_TEST_MARKERS = [
    ("pytest.ini", "pytest"),
    ("conftest.py", "pytest"),
    ("jest.config.js", "jest"),
    ("jest.config.ts", "jest"),
    ("vitest.config.ts", "vitest"),
    ("vitest.config.js", "vitest"),
    ("karma.conf.js", "karma"),
    ("phpunit.xml", "phpunit"),
    ("RSpec", "rspec"),
    ("go.sum", "go test"),
]

_LANG_BY_EXT = {
    ".py": "Python",
    ".js": "JavaScript", ".jsx": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".go": "Go", ".rs": "Rust", ".java": "Java",
    ".rb": "Ruby", ".php": "PHP", ".cs": "C#",
    ".c": "C", ".cpp": "C++", ".cc": "C++", ".h": "C/C++",
    ".swift": "Swift", ".kt": "Kotlin",
}

# Skip these dirs when sampling extensions.
_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "dist", "build", "target",
    "__pycache__", ".pytest_cache", ".idea", ".vscode",
}


@dataclass
class CodebaseFacts:
    primary_language: str
    package_manager: str
    test_runner: str
    entry_points: list[str]
    notable_dirs: list[str]
    marker_files: list[str]

    def format(self) -> str:
        bits: list[str] = ["## Project Facts"]
        if self.primary_language:
            bits.append(f"- Language: {self.primary_language}")
        if self.package_manager:
            bits.append(f"- Package manager: {self.package_manager}")
        if self.test_runner:
            bits.append(f"- Test runner: {self.test_runner}")
        if self.entry_points:
            bits.append(f"- Entry points: {', '.join(self.entry_points)}")
        if self.notable_dirs:
            bits.append(f"- Source dirs: {', '.join(self.notable_dirs)}")
        return "\n".join(bits) if len(bits) > 1 else ""


# ── Extraction ────────────────────────────────────────────────────────────────

def _detect_primary_language(root: Path) -> str:
    """Return the dominant language by file count under root (excluding noise dirs)."""
    counts: Counter = Counter()
    scanned = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in files:
            ext = Path(name).suffix.lower()
            lang = _LANG_BY_EXT.get(ext)
            if lang:
                counts[lang] += 1
                scanned += 1
                if scanned > 2000:
                    break
        if scanned > 2000:
            break
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _detect_package_manager(root: Path) -> tuple[str, str]:
    """Return (description, marker_file_name) of the first match, else ('', '')."""
    for _lang, markers in _MARKERS.items():
        for filename, desc in markers:
            if (root / filename).is_file():
                return desc, filename
    return "", ""


def _detect_test_runner(root: Path) -> str:
    for filename, runner in _TEST_MARKERS:
        if (root / filename).is_file():
            return runner
        if (root / "tests" / filename).is_file():
            return runner
    # pyproject.toml [tool.pytest.ini_options]
    pp = root / "pyproject.toml"
    if pp.is_file():
        try:
            text = pp.read_text(encoding="utf-8", errors="replace")
            if "[tool.pytest" in text:
                return "pytest"
        except OSError:
            pass
    # package.json scripts.test
    pj = root / "package.json"
    if pj.is_file():
        try:
            data = json.loads(pj.read_text(encoding="utf-8", errors="replace"))
            scripts = data.get("scripts") or {}
            test_cmd = scripts.get("test") or ""
            for runner in ("vitest", "jest", "mocha", "playwright", "cypress"):
                if runner in test_cmd:
                    return runner
        except (json.JSONDecodeError, OSError):
            pass
    return ""


_PYPROJECT_SCRIPTS_RE = re.compile(
    r"\[project\.scripts\][^\[]*", re.DOTALL,
)


def _detect_entry_points(root: Path) -> list[str]:
    out: list[str] = []
    pp = root / "pyproject.toml"
    if pp.is_file():
        try:
            text = pp.read_text(encoding="utf-8", errors="replace")
            m = _PYPROJECT_SCRIPTS_RE.search(text)
            if m:
                for line in m.group(0).splitlines():
                    line = line.strip()
                    if "=" in line and not line.startswith("["):
                        name = line.split("=", 1)[0].strip().strip('"').strip("'")
                        if name and not name.startswith("#"):
                            out.append(f"{name} (CLI)")
        except OSError:
            pass
    pj = root / "package.json"
    if pj.is_file():
        try:
            data = json.loads(pj.read_text(encoding="utf-8", errors="replace"))
            main = data.get("main")
            if main:
                out.append(f"{main} (main)")
            if "bin" in data:
                bin_data = data["bin"]
                if isinstance(bin_data, dict):
                    for k in list(bin_data)[:3]:
                        out.append(f"{k} (CLI)")
                elif isinstance(bin_data, str):
                    out.append(f"{Path(bin_data).name} (CLI)")
        except (json.JSONDecodeError, OSError):
            pass
    cargo = root / "Cargo.toml"
    if cargo.is_file():
        try:
            text = cargo.read_text(encoding="utf-8", errors="replace")
            m = re.search(r'^\s*name\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if m:
                out.append(f"{m.group(1)} (binary)")
        except OSError:
            pass
    return out[:5]


def _detect_notable_dirs(root: Path) -> list[str]:
    """Top-level dirs that look like source/test homes."""
    candidates = ("src", "lib", "app", "server", "client", "tests", "test", "spec")
    out: list[str] = []
    for name in candidates:
        if (root / name).is_dir():
            out.append(name)
    return out


def _marker_signature(root: Path) -> str:
    """Cheap fingerprint: hash of (filename, mtime) for known marker files."""
    parts: list[tuple[str, int]] = []
    all_markers = [m[0] for ms in _MARKERS.values() for m in ms]
    for name in all_markers + ["package.json", "pyproject.toml"]:
        p = root / name
        if p.is_file():
            try:
                parts.append((name, int(p.stat().st_mtime)))
            except OSError:
                pass
    return str(hash(tuple(sorted(parts))))


def extract_facts(root: str | Path | None = None) -> CodebaseFacts:
    """Run the full extraction synchronously. Cheap (~50ms on a typical repo)."""
    rootp = Path(root or os.getcwd()).resolve()
    pkg, marker = _detect_package_manager(rootp)
    return CodebaseFacts(
        primary_language=_detect_primary_language(rootp),
        package_manager=pkg,
        test_runner=_detect_test_runner(rootp),
        entry_points=_detect_entry_points(rootp),
        notable_dirs=_detect_notable_dirs(rootp),
        marker_files=[marker] if marker else [],
    )


# ── Cache via memory store ────────────────────────────────────────────────────

_FACTS_KIND = "codebase_facts"


def get_or_build_facts(root: str | Path | None = None) -> str:
    """Return formatted facts block, building & caching it on first call.

    Caches the result as a memory entry with ``kind="codebase_facts"`` so the
    facts persist across sessions. Re-extracts when project marker files change.
    """
    rootp = Path(root or os.getcwd()).resolve()
    sig = _marker_signature(rootp)
    try:
        from .memory import _get_conn
        conn = _get_conn()
        cur = conn.execute(
            "SELECT summary FROM memory WHERE kind = ? AND project_path = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (_FACTS_KIND, str(rootp)),
        )
        row = cur.fetchone()
        if row:
            cached = row["summary"] if hasattr(row, "keys") else row[0]
            # Cached entries store "<sig>\n<facts_text>"; bust on sig mismatch.
            if "\n" in cached:
                head, body = cached.split("\n", 1)
                if head == sig:
                    return body
    except Exception:
        # Memory store unavailable — fall through and compute fresh.
        pass

    facts = extract_facts(rootp)
    text = facts.format()
    if not text:
        return ""

    try:
        from .memory import append_entry
        append_entry(
            session_id="facts_extract",
            summary=f"{sig}\n{text}",
            kind=_FACTS_KIND,
            project_path=str(rootp),
        )
    except Exception:
        pass
    return text
