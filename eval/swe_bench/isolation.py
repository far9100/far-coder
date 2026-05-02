"""Repo isolation via bare clone + per-instance git worktree.

Strategy: keep one bare clone per repo (so the second instance from the same
repo is essentially free), and check out a worktree per instance pinned to
the SWE-bench `base_commit`. Worktrees are removed after each instance.

We also reset farcode's path-keyed caches (`~/.farcode_repomap_cache.json`,
`~/.farcode_doc_cache.json`) between instances, just to be safe — the cache
is cheap to rebuild and removing it eliminates any cross-instance leakage.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CACHE_ROOT = Path.home() / ".farcode_swe_cache"
REPOS_DIR = CACHE_ROOT / "repos"
WORKTREES_DIR = CACHE_ROOT / "worktrees"


def _slug(repo: str) -> str:
    """Turn 'astropy/astropy' into 'astropy__astropy' for filesystem use."""
    return repo.replace("/", "__")


def _bare_path(repo: str) -> Path:
    return REPOS_DIR / f"{_slug(repo)}.git"


def _ensure_bare_clone(repo: str) -> Path:
    bare = _bare_path(repo)
    if bare.exists():
        return bare
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    subprocess.run(
        ["git", "clone", "--bare", url, str(bare)],
        check=True, capture_output=True, text=True,
    )
    return bare


def _ensure_commit(bare: Path, base_commit: str) -> None:
    """Make sure base_commit is fetchable. Bare clone fetches all branches by
    default but PRs / detached commits may need an explicit fetch."""
    have = subprocess.run(
        ["git", "-C", str(bare), "cat-file", "-e", f"{base_commit}^{{commit}}"],
        capture_output=True,
    )
    if have.returncode == 0:
        return
    subprocess.run(
        ["git", "-C", str(bare), "fetch", "origin", base_commit],
        check=True, capture_output=True, text=True,
    )


def prepare_workspace(instance: dict) -> Path:
    """Build a fresh worktree pinned at instance['base_commit']. Returns the
    worktree path. Caller is responsible for cleanup_workspace()."""
    repo: str = instance["repo"]
    base_commit: str = instance["base_commit"]
    instance_id: str = instance["instance_id"]

    bare = _ensure_bare_clone(repo)
    _ensure_commit(bare, base_commit)

    worktree = WORKTREES_DIR / instance_id
    if worktree.exists():
        cleanup_workspace(worktree)
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "-C", str(bare), "worktree", "add", "--detach", str(worktree), base_commit],
        check=True, capture_output=True, text=True,
    )
    reset_caches()
    return worktree


def cleanup_workspace(worktree: Path) -> None:
    if not worktree.exists():
        return
    bare = None
    for d in REPOS_DIR.glob("*.git") if REPOS_DIR.exists() else []:
        bare = d
        try:
            subprocess.run(
                ["git", "-C", str(d), "worktree", "remove", "--force", str(worktree)],
                capture_output=True, text=True,
            )
            if not worktree.exists():
                return
        except Exception:
            continue
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)
        if bare is not None:
            subprocess.run(
                ["git", "-C", str(bare), "worktree", "prune"],
                capture_output=True, text=True,
            )


def reset_caches() -> None:
    """Drop farcode's path-keyed caches. Cheap to rebuild; prevents any
    cross-instance leakage even though the cache is keyed by abs path."""
    for name in (".farcode_repomap_cache.json", ".farcode_doc_cache.json"):
        p = Path.home() / name
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def cache_disk_usage() -> int:
    """Total bytes under ~/.farcode_swe_cache. For monitoring."""
    if not CACHE_ROOT.exists():
        return 0
    total = 0
    for root, _, files in os.walk(CACHE_ROOT):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total
