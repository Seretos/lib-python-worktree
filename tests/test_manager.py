"""Integration tests for the W2 core manager.

These exercise real ``git worktree`` operations against a temporary repo, as
required by the planning comment's Verifikation section.

Fixtures ``git_repo``, ``manager``, and ``manager_factory`` come from conftest.py.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

import pytest

from lib_python_worktree.core import manager as manager_module
from lib_python_worktree.core import _git_utils as git_utils_module
from lib_python_worktree.core.manager import (
    BranchAlreadyCheckedOutError,
    BranchNotFoundError,
    DirtyWorktreeError,
    DuplicateWorktreeError,
    GitTimeoutError,
    ManagerConfig,
    WorktreeError,
    WorktreeManager,
    GitCommandError,
    WorktreeNotFoundError,
    _is_path_prunable,
    _run_git,
)
from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord
from lib_python_worktree.core.yaml_store import YamlStateStore


def _git(*args: str, cwd: Path) -> None:
    """Run a git command that must succeed (used to set up worktree state)."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


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
    assert removed.status == "removed"
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

    monkeypatch.setattr(git_utils_module.subprocess, "Popen", _HangingPopen)

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
    monkeypatch.setattr(git_utils_module.subprocess, "Popen", _CapturingPopen)

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

    monkeypatch.setattr(git_utils_module.subprocess, "Popen", _RecordingPopen)
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


# ---------------------------------------------------------------------------
# Ticket #10: _is_path_prunable crash when proc.stdout is None
# ---------------------------------------------------------------------------

def test_is_path_prunable_returns_none_when_stdout_is_none(monkeypatch):
    """Regression: _is_path_prunable must not raise AttributeError when
    _run_git returns a CompletedProcess with stdout=None."""
    monkeypatch.setattr(
        manager_module,
        "_run_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=None, stderr=""
        ),
    )
    result = _is_path_prunable(Path("/fake/repo"), "/some/path")
    assert result is None


def test_is_path_prunable_empty_stdout_returns_none(monkeypatch):
    """_is_path_prunable with empty stdout must return None without raising."""
    monkeypatch.setattr(
        manager_module,
        "_run_git",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ),
    )
    result = _is_path_prunable(Path("/fake/repo"), "/some/path")
    assert result is None


def test_is_path_prunable_swallows_git_timeout_error(monkeypatch):
    """Regression: a GitTimeoutError raised during the _is_path_prunable probe
    must be swallowed (returns None) and must NOT escape create() as a bare
    GitTimeoutError replacing the intended BranchAlreadyCheckedOutError.

    This required GitTimeoutError to be a WorktreeError subclass so that the
    `except WorktreeError: return None` guard in _is_path_prunable catches it.
    """
    def _raise_timeout(*args, **kwargs):
        raise GitTimeoutError(["git", "worktree", "list", "--porcelain"], 30.0)

    monkeypatch.setattr(manager_module, "_run_git", _raise_timeout)

    result = _is_path_prunable(Path("/fake/repo"), "/some/path")
    assert result is None, (
        "GitTimeoutError inside _is_path_prunable must be swallowed (return None)"
    )

    # Also confirm the class hierarchy is intact.
    assert issubclass(GitTimeoutError, WorktreeError), (
        "GitTimeoutError must be a WorktreeError subclass"
    )


# ---------------------------------------------------------------------------
# Ticket #10: already-checked-out on an out-of-band worktree raises structured error
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_create_raises_structured_error_for_out_of_band_worktree(
    manager_factory: Callable,
    git_repo: Path,
    tmp_path: Path,
):
    """Creating a worktree for a branch that is checked out out-of-band
    (not via the manager) must raise BranchAlreadyCheckedOutError, not
    a bare AttributeError. Regression for stdout=None crash path."""
    # Create the worktree out-of-band via subprocess so no manager knows about it.
    oot_path = tmp_path / "out-of-band-wt"
    subprocess.run(
        ["git", "worktree", "add", str(oot_path), "feature/alpha"],
        cwd=git_repo,
        check=True,
        capture_output=True,
    )

    # A fresh manager with an empty store — won't hit the duplicate-check shortcut.
    fresh_manager = manager_factory()
    with pytest.raises(BranchAlreadyCheckedOutError) as excinfo:
        fresh_manager.create(str(git_repo), "feature/alpha")

    err = excinfo.value
    assert err.branch == "feature/alpha"
    assert "branch_already_checked_out" in str(err)

    # Clean up the out-of-band worktree.
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(oot_path)],
        cwd=git_repo,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# Ticket #10: WorktreeManager.adopt()
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_manager_adopt_discovers_out_of_band_worktree(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    """adopt() must import a worktree that was created out-of-band (not via the
    manager) into the store with status='adopted' and branch_created_by_us=False."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    # Create the worktree out-of-band via subprocess.
    oot_path = tmp_path / "oot-wt"
    subprocess.run(
        ["git", "worktree", "add", str(oot_path), "feature/alpha"],
        cwd=git_repo,
        check=True,
        capture_output=True,
    )

    try:
        report = mgr.adopt(str(git_repo))

        assert len(report.adopted) == 1
        records = mgr.list()
        assert len(records) == 1
        rec = records[0]
        assert rec.status == "adopted"
        assert rec.branch_created_by_us is False
        assert rec.branch == "feature/alpha"
        assert rec.ports == {}
        assert rec.pids == {}
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(oot_path)],
            cwd=git_repo,
            capture_output=True,
        )


@pytest.mark.requires_git
def test_manager_adopt_idempotent(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    """Calling adopt() twice must not raise and must not duplicate records."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    oot_path = tmp_path / "oot-wt-idem"
    subprocess.run(
        ["git", "worktree", "add", str(oot_path), "feature/alpha"],
        cwd=git_repo,
        check=True,
        capture_output=True,
    )

    try:
        report1 = mgr.adopt(str(git_repo))
        report2 = mgr.adopt(str(git_repo))

        assert len(report1.adopted) == 1
        assert len(report2.adopted) == 0  # second call: nothing new to import
        assert len(mgr.list()) == 1
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(oot_path)],
            cwd=git_repo,
            capture_output=True,
        )


def test_manager_adopt_raises_for_non_yaml_store(tmp_path: Path):
    """adopt() must raise WorktreeError when the store is not a YamlStateStore."""
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )
    with pytest.raises(WorktreeError, match="YamlStateStore"):
        mgr.adopt("/any/path")


# ---------------------------------------------------------------------------
# Ticket #10: WorktreeManager.prune()
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_manager_prune_smoke(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    """prune() must remove the stale git registration after a worktree dir is
    wiped externally.  Verifies both no-raise AND that the stale entry is gone
    from ``git worktree list --porcelain`` afterwards.

    Uses ``--expire=now`` internally so the 3-month gc.worktreePruneExpire grace
    period does not prevent immediate removal of a freshly-deleted directory.
    """
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    # Create via manager then wipe dir without informing git, leaving a stale reg.
    rec = mgr.create(str(git_repo), "feature/alpha")
    stale_path = rec.path
    shutil.rmtree(stale_path)

    # Normalise path separators for comparison: git uses forward slashes on
    # all platforms in --porcelain output, but rec.path may use backslashes on
    # Windows.
    stale_path_fwd = stale_path.replace("\\", "/")

    # Confirm the stale registration exists BEFORE prune.
    before = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=git_repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert stale_path_fwd in before.stdout, (
        "stale worktree path must appear in git worktree list before prune"
    )

    # prune() must succeed and clear the stale entry immediately.
    mgr.prune(str(git_repo))

    after = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=git_repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert stale_path_fwd not in after.stdout, (
        "stale worktree path must be absent from git worktree list after prune"
    )


@pytest.mark.requires_git
def test_manager_prune_raises_git_command_error_v2(
    tmp_path: Path, git_repo: Path, skip_if_no_git, monkeypatch  # noqa: ARG001
):
    """prune() raises GitCommandError when git worktree prune returns non-zero."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    original_run_git = manager_module._run_git

    def _fake_run_git(args, cwd=None, **kwargs):
        if args[:2] == ["worktree", "prune"]:
            return subprocess.CompletedProcess(
                args=["git", "worktree", "prune", "--expire=now"],
                returncode=1,
                stdout="",
                stderr="fatal: simulated prune failure",
            )
        return original_run_git(args, cwd=cwd, **kwargs)

    monkeypatch.setattr(manager_module, "_run_git", _fake_run_git)

    with pytest.raises(GitCommandError, match="simulated prune failure"):
        mgr.prune(str(git_repo))


# ---------------------------------------------------------------------------
# Ticket #2: remove() contract — status="removed" and DirtyWorktreeError
# ---------------------------------------------------------------------------


@pytest.mark.requires_git
def test_remove_success_return_status(
    manager: WorktreeManager, git_repo: Path
):
    """remove() must return a record with status='removed' and the correct id."""
    rec = manager.create(str(git_repo), "feature/alpha")
    result = manager.remove(rec.id)
    assert result.status == "removed"
    assert result.id == rec.id


@pytest.mark.requires_git
def test_remove_dirty_no_force_raises_dirty_error(
    manager: WorktreeManager, git_repo: Path
):
    """remove(force=False) on a dirty worktree must raise DirtyWorktreeError
    with 'force=True' in the message and no raw '--force' git flag exposed."""
    rec = manager.create(str(git_repo), "feature/alpha")
    wt_path = Path(rec.path)

    # Dirty the worktree so git refuses without --force.
    (wt_path / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(DirtyWorktreeError) as excinfo:
        manager.remove(rec.id, force=False)

    msg = str(excinfo.value)
    assert "force=True" in msg
    assert "--force" not in msg


def test_dirty_worktree_error_message_no_git_internals(monkeypatch):
    """DirtyWorktreeError message must contain 'force=True' and must not
    contain '--force' or '128' (no git internals leaked).

    Uses a fake _run_git that returns returncode=128 with a realistic git
    stderr so the test does not need a real git binary.
    """
    from lib_python_worktree.core.state import WorktreeRecord

    fake_record = WorktreeRecord(
        id="test-wt-deadbeef",
        repo_root="/fake/repo",
        branch="feature/test",
        path="/fake/repo-store/test-wt-deadbeef",
    )

    real_git_stderr = (
        "fatal: '/fake/repo-store/test-wt-deadbeef' contains modified or "
        "untracked files, use --force to delete it"
    )

    def _fake_run_git(args, cwd=None, **kwargs):
        if args[:2] == ["worktree", "remove"]:
            return subprocess.CompletedProcess(
                args=["git", *args],
                returncode=128,
                stdout="",
                stderr=real_git_stderr,
            )
        # Any other git call (e.g. lifecycle stop) returns success.
        return subprocess.CompletedProcess(
            args=["git", *args], returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(manager_module, "_run_git", _fake_run_git)

    mgr = WorktreeManager(
        config=ManagerConfig(store_root=Path("/fake/store")),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )

    with pytest.raises(DirtyWorktreeError) as excinfo:
        mgr._teardown(fake_record, force=False)

    msg = str(excinfo.value)
    assert "force=True" in msg
    assert "--force" not in msg
    assert "128" not in msg
