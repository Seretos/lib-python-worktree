"""Integration tests for the W2 core manager.

These exercise real ``git worktree`` operations against a temporary repo, as
required by the planning comment's Verifikation section.

Fixtures ``git_repo``, ``manager``, and ``manager_factory`` come from conftest.py.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from lib_python_worktree.core import manager as manager_module
from lib_python_worktree.core.manager import (
    BranchAlreadyCheckedOutError,
    BranchNotFoundError,
    DuplicateWorktreeError,
    GitTimeoutError,
    ManagerConfig,
    WorktreeManager,
    WorktreeNotFoundError,
    _run_git,
)
from lib_python_worktree.core.state import InMemoryStateStore


# ---------------------------------------------------------------------------
# Tests that touch real git
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_create_list_remove_roundtrip(manager: WorktreeManager, git_repo: Path):
    rec = manager.create(str(git_repo), "feature/alpha")
    assert rec.id.startswith("repo-feature-alpha-")
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
    manager: WorktreeManager, git_repo: Path
):
    with pytest.raises(BranchNotFoundError):
        manager.create(str(git_repo), "feature/does-not-exist")


@pytest.mark.requires_git
def test_create_unknown_branch_with_base(
    manager: WorktreeManager, git_repo: Path
):
    rec = manager.create(str(git_repo), "feature/new", base="main")
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
    manager: WorktreeManager, git_repo: Path
):
    manager.create(str(git_repo), "feature/alpha")
    with pytest.raises(DuplicateWorktreeError):
        manager.create(str(git_repo), "feature/alpha")


@pytest.mark.requires_git
def test_remove_unknown_id_fails(manager: WorktreeManager):
    with pytest.raises(WorktreeNotFoundError):
        manager.remove("nope-nope-12345678")


@pytest.mark.requires_git
def test_worktree_paths_under_store_root(
    manager: WorktreeManager, git_repo: Path, tmp_path: Path
):
    rec = manager.create(str(git_repo), "feature/alpha")
    # The manager fixture uses store-1 inside tmp_path.
    wt_path = Path(rec.path)
    # store_root / repo_slug / id
    assert wt_path.parent.name == "repo"
    assert wt_path.parent.parent.name.startswith("store-")


@pytest.mark.requires_git
def test_run_git_smoke_version_completes_quickly(skip_if_no_git):  # noqa: ARG001
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


@pytest.mark.requires_git
def test_create_branch_already_checked_out_elsewhere(
    manager_factory: Callable[..., WorktreeManager],
    git_repo: Path,
    tmp_path: Path,
):
    """Creating a worktree for a branch that is already checked out in
    another worktree (tracked by a different state store, so the in-memory
    duplicate-check shortcut at manager.py:133 doesn't fire) must surface as
    a structured ``BranchAlreadyCheckedOutError`` with branch + path attrs.
    """

    first_manager = manager_factory()
    first = first_manager.create(str(git_repo), "feature/alpha")
    assert Path(first.path).exists()

    # Fresh manager + fresh state store simulates a second client session
    # that doesn't know about worktree A yet -- now the duplicate-check
    # falls through and we reach the actual `git worktree add`.
    other = manager_factory()

    with pytest.raises(BranchAlreadyCheckedOutError) as excinfo:
        other.create(str(git_repo), "feature/alpha")

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
    manager_factory: Callable[..., WorktreeManager],
    git_repo: Path,
    tmp_path: Path,
):
    """If the worktree directory is gone but git still has the registration,
    the structured error must report ``prunable is True`` so the caller can
    suggest ``git worktree prune``.
    """

    import shutil

    first_manager = manager_factory()
    first = first_manager.create(str(git_repo), "feature/alpha")
    # Wipe the worktree dir behind git's back so its registration goes stale.
    shutil.rmtree(first.path)

    other = manager_factory()

    with pytest.raises(BranchAlreadyCheckedOutError) as excinfo:
        other.create(str(git_repo), "feature/alpha")

    err = excinfo.value
    assert err.branch == "feature/alpha"
    assert err.prunable is True
    assert "prunable=True" in str(err)


@pytest.mark.requires_git
def test_remove_with_force_flag(
    manager: WorktreeManager, git_repo: Path
):
    """Removing a worktree with force=True must succeed even when the
    worktree has uncommitted changes (covers the _teardown --force branch).
    """
    rec = manager.create(str(git_repo), "feature/alpha")
    wt_path = Path(rec.path)
    assert wt_path.exists()

    # Dirty the worktree with an untracked file so git would normally refuse.
    (wt_path / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    # force=True must not raise and must remove the directory.
    removed = manager.remove(rec.id, force=True)
    assert removed.id == rec.id
    assert not wt_path.exists()
    assert manager.list() == []


# ---------------------------------------------------------------------------
# Pure-environment tests — no git binary needed, no marker
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Monkeypatch-only tests — no real git, no marker
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ticket #1: worktree_remove must delete the branch it created
# ---------------------------------------------------------------------------


def test_worktree_record_branch_created_by_us_default_false():
    """Pure dataclass test: WorktreeRecord.branch_created_by_us defaults to False."""
    rec = WorktreeRecord(id="x", repo_root="/r", branch="b", path="/p")
    assert rec.branch_created_by_us is False


@pytest.mark.requires_git
def test_remove_deletes_branch_created_by_worktree_create(
    manager: WorktreeManager, git_repo: Path
):
    """Regression: branch created by 'git worktree add -b' must be deleted on remove."""
    rec = manager.create(str(git_repo), "feature/new", base="main")
    assert rec.branch_created_by_us is True

    manager.remove(rec.id)

    branch_check = subprocess.run(
        ["git", "rev-parse", "--verify", "refs/heads/feature/new"],
        cwd=git_repo,
        capture_output=True,
    )
    assert branch_check.returncode != 0, (
        "branch created by worktree_create must be deleted after worktree_remove"
    )


@pytest.mark.requires_git
def test_remove_does_not_delete_preexisting_branch(
    manager: WorktreeManager, git_repo: Path
):
    """Reuse path: a branch that pre-existed before create must survive remove."""
    # feature/alpha is created by the git_repo fixture (pre-existing).
    rec = manager.create(str(git_repo), "feature/alpha")
    assert rec.branch_created_by_us is False

    manager.remove(rec.id)

    branch_check = subprocess.run(
        ["git", "rev-parse", "--verify", "refs/heads/feature/alpha"],
        cwd=git_repo,
        capture_output=True,
    )
    assert branch_check.returncode == 0, "pre-existing branch must survive remove"


@pytest.mark.requires_git
def test_remove_force_deletes_branch_with_unmerged_commits(
    manager: WorktreeManager, git_repo: Path
):
    """force=True must use 'git branch -D' to delete a branch with unmerged commits."""
    rec = manager.create(str(git_repo), "feature/unmerged", base="main")
    assert rec.branch_created_by_us is True

    # Commit something inside the worktree so the branch has unmerged commits.
    wt_path = Path(rec.path)
    (wt_path / "new_file.txt").write_text("unmerged\n", encoding="utf-8")
    _git("add", "-A", cwd=wt_path)
    _git("commit", "-q", "-m", "unmerged commit", cwd=wt_path)

    manager.remove(rec.id, force=True)

    assert not wt_path.exists(), "worktree path must be gone after remove"

    branch_check = subprocess.run(
        ["git", "rev-parse", "--verify", "refs/heads/feature/unmerged"],
        cwd=git_repo,
        capture_output=True,
    )
    assert branch_check.returncode != 0, (
        "branch with unmerged commits must be deleted when force=True"
    )


@pytest.mark.requires_git
def test_remove_tolerates_already_deleted_branch(
    manager: WorktreeManager, git_repo: Path
):
    """remove must be idempotent when the owned branch was already deleted."""
    rec = manager.create(str(git_repo), "feature/gone", base="main")
    assert rec.branch_created_by_us is True

    # ``git branch -D`` refuses a branch checked out in another worktree, so
    # detach the worktree HEAD first, then delete the branch for real.
    _git("checkout", "--detach", cwd=Path(rec.path))
    _git("branch", "-D", "feature/gone", cwd=git_repo)

    # manager.remove must not raise even though the branch is already gone.
    removed = manager.remove(rec.id)
    assert removed.id == rec.id
    assert manager.list() == []


@pytest.mark.requires_git
def test_remove_unmerged_branch_without_force_cleans_state_and_raises(
    manager: WorktreeManager, git_repo: Path
):
    """force=False on an unmerged owned branch raises, but the state record is
    still cleaned up (the worktree dir was already removed)."""
    rec = manager.create(str(git_repo), "feature/leak-test", base="main")
    assert rec.branch_created_by_us is True

    # Add an unmerged commit so ``git branch -d`` will refuse.
    wt_path = Path(rec.path)
    (wt_path / "leak_file.txt").write_text("unmerged\n", encoding="utf-8")
    _git("add", "-A", cwd=wt_path)
    _git("commit", "-q", "-m", "unmerged commit", cwd=wt_path)

    with pytest.raises(GitCommandError):
        manager.remove(rec.id, force=False)

    # The state record must be gone despite the exception (no stale entry).
    remaining_ids = [r.id for r in manager.list()]
    assert rec.id not in remaining_ids, (
        "State record must be removed even when branch-delete raises"
    )
