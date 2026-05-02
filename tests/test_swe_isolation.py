"""Unit tests for eval/swe_bench/isolation.py — bare-clone + worktree round trip.

We use a local file:// URL as the "remote" so no network is hit. The cache
root is redirected to tmp_path to keep tests hermetic."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eval.swe_bench import isolation


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def fake_remote(tmp_path):
    """Build a real git repo to act as the 'remote', plus return its file:// URL
    and a base_commit SHA."""
    remote = tmp_path / "remote"
    remote.mkdir()
    _git(["init", "-q", "-b", "main"], cwd=remote)
    _git(["config", "user.email", "t@t"], cwd=remote)
    _git(["config", "user.name", "t"], cwd=remote)
    _git(["config", "commit.gpgsign", "false"], cwd=remote)
    (remote / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "-A"], cwd=remote)
    _git(["commit", "-q", "-m", "init"], cwd=remote)
    sha = _git(["rev-parse", "HEAD"], cwd=remote)
    url = remote.as_uri()
    return remote, url, sha


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect isolation's cache root to tmp_path so tests don't touch
    ~/.farcode_swe_cache."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(isolation, "CACHE_ROOT", cache)
    monkeypatch.setattr(isolation, "REPOS_DIR", cache / "repos")
    monkeypatch.setattr(isolation, "WORKTREES_DIR", cache / "worktrees")
    return cache


def _make_instance(url: str, sha: str, instance_id: str) -> dict:
    """isolation expects 'repo' as 'owner/name' but we patch _ensure_bare_clone
    to use the file:// URL."""
    return {
        "repo": "fake/remote",
        "base_commit": sha,
        "instance_id": instance_id,
    }


def test_prepare_and_cleanup_round_trip(monkeypatch, fake_remote, isolated_cache):
    remote, url, sha = fake_remote
    cache = isolated_cache

    # Patch the URL-builder to point at our local file:// remote.
    original_ensure = isolation._ensure_bare_clone

    def fake_ensure(repo: str) -> Path:
        bare = isolation._bare_path(repo)
        if bare.exists():
            return bare
        isolation.REPOS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--bare", url, str(bare)],
            check=True, capture_output=True, text=True,
        )
        return bare

    monkeypatch.setattr(isolation, "_ensure_bare_clone", fake_ensure)

    instance = _make_instance(url, sha, "fake__remote-1")

    workspace = isolation.prepare_workspace(instance)
    try:
        assert workspace.exists()
        assert (workspace / "README.md").read_text(encoding="utf-8") == "hello\n"
        # HEAD is detached at base_commit
        head = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        assert head == sha
    finally:
        isolation.cleanup_workspace(workspace)

    assert not workspace.exists()
    # Bare repo survives so the next instance can reuse it
    assert isolation._bare_path("fake/remote").exists()


def test_prepare_workspace_replaces_existing(monkeypatch, fake_remote, isolated_cache):
    """If a worktree already exists at the target path, prepare_workspace must
    rebuild it cleanly rather than fail."""
    remote, url, sha = fake_remote

    def fake_ensure(repo: str) -> Path:
        bare = isolation._bare_path(repo)
        if bare.exists():
            return bare
        isolation.REPOS_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--bare", url, str(bare)],
            check=True, capture_output=True, text=True,
        )
        return bare

    monkeypatch.setattr(isolation, "_ensure_bare_clone", fake_ensure)

    instance = _make_instance(url, sha, "fake__remote-2")

    ws1 = isolation.prepare_workspace(instance)
    # Pollute it with a stray file
    (ws1 / "stray.txt").write_text("garbage", encoding="utf-8")

    # Re-prepare without explicit cleanup — must wipe and rebuild
    ws2 = isolation.prepare_workspace(instance)
    try:
        assert ws2 == ws1
        assert not (ws2 / "stray.txt").exists()
        assert (ws2 / "README.md").exists()
    finally:
        isolation.cleanup_workspace(ws2)


def test_reset_caches_unlinks_existing(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".farcode_repomap_cache.json").write_text("{}", encoding="utf-8")
    (fake_home / ".farcode_doc_cache.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    isolation.reset_caches()

    assert not (fake_home / ".farcode_repomap_cache.json").exists()
    assert not (fake_home / ".farcode_doc_cache.json").exists()


def test_reset_caches_silent_when_missing(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    isolation.reset_caches()  # must not raise
