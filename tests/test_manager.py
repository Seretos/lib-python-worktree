"""Integration tests for the W2 core manager.

These exercise real ``git worktree`` operations against a temporary repo, as
required by the planning comment's Verifikation section.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterator

import pytest

from lib_python_worktree.core import manager as manager_module
from lib_python_worktree.core.manager import (
    BranchAlreadyCheckedOutError,
    BranchNotFoundError,
    DuplicateWorktreeError,
    GitCommandError,
    GitTimeoutError,
    ManagerConfig,
    WorktreeError,
    WorktreeManager,
    WorktreeNotFoundError,
    _run_git,
)
from lib_python_worktree.core.state import InMemoryStateStore


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def temp_repo(tmp_path: Path) -> Iterator[Path]:
    repo = tmp_path / "src-repo"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "-A", cwd=repo)
    _git("commit", "-q", "-m", "init", cwd=repo)
    _git("branch", "feature/alpha", cwd=repo)
    yield repo


@pytest.fixture
def manager(tmp_path: Path) -> WorktreeManager:
    store_root = tmp_path / "store"
    return WorktreeManager(
        config=ManagerConfig(store_root=store_root),
        state=InMemoryStateStore(),
    )


@pytest.mark.requires_git
def test_create_list_remove_roundtrip(manager: WorktreeManager, temp_repo: Path):
    rec = manager.create(str(temp_repo), "feature/alpha")
    assert rec.id.startswith("src-repo-feature-alpha-")
    assert rec.branch == "feature/alpha"
    assert Path(rec.path).exists()
    assert Path(rec.path).is_dir()

    listed = manager.list()
    assert len(listed) == 1
    assert listed[0].id == rec.id

    removed = manager.remove(rec.id)
    assert removed.id == rec.id
    assert not Path(rec.path).exists()
    assert manager.list() == []


@pytest.mark.requires_git
def test_create_unknown_branch_without_base(
    manager: WorktreeManager, temp_repo: Path
):
    with pytest.raises(BranchNotFoundError):
        manager.create(str(temp_repo), "feature/does-not-exist")


@pytest.mark.requires_git
def test_create_unknown_branch_with_base(
    manager: WorktreeManager, temp_repo: Path
):
    rec = manager.create(str(temp_repo), "feature/new", base="main")
    assert rec.branch == "feature/new"
    assert Path(rec.path).exists()
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=rec.path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "feature/new"


@pytest.mark.requires_git
def test_duplicate_create_same_branch_fails(
    manager: WorktreeManager, temp_repo: Path
):
    manager.create(str(temp_repo), "feature/alpha")
    with pytest.raises(DuplicateWorktreeError):
        manager.create(str(temp_repo), "feature/alpha")


def test_remove_unknown_id_fails(manager: WorktreeManager):
    with pytest.raises(WorktreeNotFoundError):
        manager.remove("nope-nope-12345678")


def test_store_root_from_env(tmp_path: Path, monkeypatch):
    target = tmp_path / "custom-store"
    monkeypatch.setenv("WORKTREE_STORE_ROOT", str(target))
    cfg = ManagerConfig.from_env()
    assert cfg.store_root == target.resolve()


def test_store_root_default(monkeypatch):
    monkeypatch.delenv("WORKTREE_STORE_ROOT", raising=False)
    cfg = ManagerConfig.from_env()
    assert cfg.store_root.name == "agent-worktree-store"
    assert cfg.store_root.is_absolute()


@pytest.mark.requires_git
def test_worktree_paths_under_store_root(
    manager: WorktreeManager, temp_repo: Path, tmp_path: Path
):
    rec = manager.create(str(temp_repo), "feature/alpha")
    # store_root / repo_slug / id
    assert Path(rec.path).parent.parent == (tmp_path / "store").resolve()
    assert Path(rec.path).parent.name == "src-repo"


# ---- Ticket #19: _run_git timeout + stdin handling ----


@pytest.mark.requires_git
def test_run_git_smoke_version_completes_quickly():
    """Sanity check: ``git --version`` finishes well under 1 s with the new
    Popen-based plumbing. Catches pipe/handle plumbing regressions on every
    platform (Linux, Windows, packaged exe).
    """

    import time as _time

    start = _time.monotonic()
    proc = _run_git(["--version"])
    elapsed = _time.monotonic() - start
    assert proc.returncode == 0
    assert proc.stdout.startswith("git version")
    assert elapsed < 1.0, f"_run_git(['--version']) took {elapsed:.2f}s"


def test_run_git_raises_timeout_when_subprocess_hangs(monkeypatch):
    """Simulate a hanging git via a fake Popen, confirm GitTimeoutError fires
    and the process gets killed (rather than the call blocking forever).
    """

    killed = {"value": False}

    class _HangingPopen:
        def __init__(self, *args, **kwargs):
            self.returncode = None

        def communicate(self, timeout=None):
            # Always pretend the child is still running.
            raise subprocess.TimeoutExpired(cmd=["git", "hang"], timeout=timeout)

        def kill(self):
            killed["value"] = True
            self.returncode = -9

    monkeypatch.setattr(manager_module.subprocess, "Popen", _HangingPopen)

    with pytest.raises(GitTimeoutError) as excinfo:
        _run_git(["status"], timeout=0.05)

    assert killed["value"] is True
    assert excinfo.value.command == ["git", "status"]
    assert excinfo.value.elapsed >= 0.0


def test_run_git_timeout_respects_env_override(monkeypatch):
    """``WORKTREE_GIT_TIMEOUT_SEC`` overrides the built-in 30 s default
    when no explicit timeout kwarg is passed.
    """

    captured = {"timeout": None}

    class _CapturingPopen:
        def __init__(self, *args, **kwargs):
            self.returncode = 0

        def communicate(self, timeout=None):
            captured["timeout"] = timeout
            return ("", "")

        def kill(self):  # pragma: no cover - not reached in this test
            pass

    monkeypatch.setenv("WORKTREE_GIT_TIMEOUT_SEC", "7.5")
    monkeypatch.setattr(manager_module.subprocess, "Popen", _CapturingPopen)

    _run_git(["--version"])
    assert captured["timeout"] == 7.5


def test_run_git_closes_stdin(monkeypatch):
    """Regression guard: ``stdin=DEVNULL`` must always be passed so the spawned
    git can never inherit the MCP client's stdin pipe (the Windows hang root
    cause).
    """

    captured_kwargs: dict = {}

    class _RecordingPopen:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            self.returncode = 0

        def communicate(self, timeout=None):
            return ("", "")

        def kill(self):  # pragma: no cover - not reached in this test
            pass

    monkeypatch.setattr(manager_module.subprocess, "Popen", _RecordingPopen)
    _run_git(["--version"])
    assert captured_kwargs.get("stdin") is subprocess.DEVNULL


# ---- Ticket #18: structured error for "branch already checked out elsewhere" ----


@pytest.mark.requires_git
def test_create_branch_already_checked_out_elsewhere(
    manager: WorktreeManager, temp_repo: Path, tmp_path: Path
):
    """Creating a worktree for a branch that is already checked out in
    another worktree (tracked by a different state store, so the in-memory
    duplicate-check shortcut at manager.py:133 doesn't fire) must surface as
    a structured ``BranchAlreadyCheckedOutError`` with branch + path attrs.
    """

    # First state store creates worktree A for feature/alpha.
    first = manager.create(str(temp_repo), "feature/alpha")
    assert Path(first.path).exists()

    # Fresh manager + fresh state store simulates a second client session
    # that doesn't know about worktree A yet -- now the duplicate-check at
    # manager.py:133 falls through and we reach the actual `git worktree add`.
    other = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store2"),
        state=InMemoryStateStore(),
    )

    with pytest.raises(BranchAlreadyCheckedOutError) as excinfo:
        other.create(str(temp_repo), "feature/alpha")

    err = excinfo.value
    assert err.branch == "feature/alpha"
    assert Path(err.path).resolve() == Path(first.path).resolve()
    # Existing dir -> not prunable.
    assert err.prunable is False
    # Message contract matches the format used by tools/worktree.py callers.
    msg = str(err)
    assert "branch_already_checked_out" in msg
    assert "'feature/alpha'" in msg
    assert "git worktree prune" in msg


@pytest.mark.requires_git
def test_already_checked_out_reports_prunable_after_dir_removed(
    manager: WorktreeManager, temp_repo: Path, tmp_path: Path
):
    """If the worktree directory is gone but git still has the registration,
    the structured error must report ``prunable is True`` so the caller can
    suggest ``git worktree prune``.
    """

    import shutil

    first = manager.create(str(temp_repo), "feature/alpha")
    # Wipe the worktree dir behind git's back so its registration goes stale.
    shutil.rmtree(first.path)

    other = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store2"),
        state=InMemoryStateStore(),
    )

    with pytest.raises(BranchAlreadyCheckedOutError) as excinfo:
        other.create(str(temp_repo), "feature/alpha")

    err = excinfo.value
    assert err.branch == "feature/alpha"
    assert err.prunable is True
    assert "prunable=True" in str(err)


# ---- New real-git tests using shared conftest fixtures ----


@pytest.mark.requires_git
def test_force_remove_worktree_with_uncommitted_changes(
    manager_factory, git_repo: Path
):
    """Regression: remove(force=False) must refuse when the worktree has
    uncommitted changes; remove(force=True) must succeed and delete the path.
    """
    mgr = manager_factory()
    rec = mgr.create(str(git_repo), "feature/dirty", base="main")
    wt_path = Path(rec.path)

    # Write an uncommitted file inside the worktree.
    (wt_path / "dirty.txt").write_text("not committed\n", encoding="utf-8")

    # Non-forced removal must fail with a GitCommandError (git refuses).
    with pytest.raises(GitCommandError):
        mgr.remove(rec.id, force=False)

    # Forced removal must succeed and the directory must be gone.
    removed = mgr.remove(rec.id, force=True)
    assert removed.id == rec.id
    assert not wt_path.exists()
    assert mgr.list() == []


@pytest.mark.requires_git
def test_validate_repo_with_non_git_directory(manager_factory, tmp_path: Path):
    """Creating a worktree against a directory that has no .git must raise
    WorktreeError (specifically the "Not a git repository" path).
    """
    not_a_repo = tmp_path / "plain-dir"
    not_a_repo.mkdir()
    mgr = manager_factory()
    with pytest.raises(WorktreeError):
        mgr.create(str(not_a_repo), "some-branch")


@pytest.mark.requires_git
def test_create_with_empty_branch_raises(manager_factory, git_repo: Path):
    """Passing an empty (or whitespace-only) branch name must raise WorktreeError.
    (_validate_repo runs git rev-parse first; the branch guard fires after that.)
    """
    mgr = manager_factory()
    with pytest.raises(WorktreeError, match="branch must be a non-empty string"):
        mgr.create(str(git_repo), "")
