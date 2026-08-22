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
    UnknownVariantError,
    VariantResolutionError,
    WorktreeError,
    WorktreeManager,
    GitCommandError,
    WorktreeNotFoundError,
    _is_path_prunable,
    _run_git,
)
from lib_python_worktree.core.state import (
    STOP_REASON_SURVIVORS,
    InMemoryStateStore,
    StopDetail,
    WorktreeRecord,
)
from lib_python_worktree.core.yaml_store import YamlStateStore, _pid_alive


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


def _make_run_git_spy(monkeypatch: pytest.MonkeyPatch) -> "list[list[str]]":
    """Monkeypatch ``manager_module._run_git`` with a recording wrapper that
    delegates to the real implementation, so a test can assert on exactly
    which git subcommands ``create()`` issued (e.g. that a fetch or a
    ``rev-parse --abbrev-ref HEAD`` call did/didn't happen)."""
    calls: "list[list[str]]" = []
    real_run_git = manager_module._run_git

    def _spy(args, cwd=None, **kwargs):
        calls.append(list(args))
        return real_run_git(args, cwd=cwd, **kwargs)

    monkeypatch.setattr(manager_module, "_run_git", _spy)
    return calls


# ---------------------------------------------------------------------------
# Ticket #93: create() defaults an omitted `base` to the branch currently
# checked out at the main clone, instead of always raising BranchNotFoundError.
# ---------------------------------------------------------------------------


@pytest.mark.requires_git
def test_create_new_branch_without_base_defaults_to_current_branch(
    manager: WorktreeManager, git_repo: Path
):
    """git_repo's HEAD is on `main` -- a new branch created without `base`
    must start from `main`, not raise."""
    rec = manager.create(str(git_repo), "feature/does-not-exist")
    assert rec.branch == "feature/does-not-exist"
    assert Path(rec.path).exists()
    assert rec.branch_created_by_us is True

    wt_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=rec.path,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    main_head = subprocess.run(
        ["git", "rev-parse", "main"], cwd=git_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert wt_head == main_head


@pytest.mark.requires_git
def test_create_existing_branch_without_base_unaffected(
    manager: WorktreeManager, git_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    """When `branch` already exists, the new default-base resolution path
    must not run at all -- no `rev-parse --abbrev-ref HEAD` call issued."""
    calls = _make_run_git_spy(monkeypatch)

    rec = manager.create(str(git_repo), "feature/alpha")
    assert rec.branch == "feature/alpha"

    assert ["rev-parse", "--abbrev-ref", "HEAD"] not in calls


@pytest.mark.requires_git
def test_create_without_base_raises_when_head_detached(
    manager: WorktreeManager, git_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    _git("checkout", "--detach", "HEAD", cwd=git_repo)
    calls = _make_run_git_spy(monkeypatch)

    with pytest.raises(BranchNotFoundError, match="current branch"):
        manager.create(str(git_repo), "feature/detached")

    assert not any(c[:2] == ["worktree", "add"] for c in calls)


@pytest.mark.requires_git
def test_create_without_base_raises_when_head_unborn(
    manager: WorktreeManager, tmp_path: Path, skip_if_no_git  # noqa: ARG001
):
    empty_repo = tmp_path / "empty-repo"
    empty_repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=empty_repo)
    _git("config", "user.email", "test@example.com", cwd=empty_repo)
    _git("config", "user.name", "Test", cwd=empty_repo)

    with pytest.raises(BranchNotFoundError, match="current branch"):
        manager.create(str(empty_repo), "feature/unborn")


@pytest.mark.requires_git
def test_create_without_base_propagates_unexpected_current_branch_failure(
    manager: WorktreeManager, git_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    """A `git rev-parse --abbrev-ref HEAD` failure that is neither the
    unborn-HEAD case (exit 128) nor the detached-HEAD case (exit 0, stdout
    "HEAD") is a genuine repo problem (corrupt .git/HEAD, broken refs,
    permission error, ...), not an expected "no default base available"
    case. It must propagate as GitCommandError -- NOT get folded into
    BranchNotFoundError / "detached or unborn HEAD"."""
    real_run_git = manager_module._run_git

    def _fake_run_git(args, cwd=None, **kwargs):
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(
                args=["git", *args],
                returncode=13,
                stdout="",
                stderr="fatal: unable to read HEAD: Permission denied",
            )
        return real_run_git(args, cwd=cwd, **kwargs)

    monkeypatch.setattr(manager_module, "_run_git", _fake_run_git)

    with pytest.raises(GitCommandError) as excinfo:
        manager.create(str(git_repo), "feature/other-failure")

    assert excinfo.value.returncode == 13
    assert "Permission denied" in excinfo.value.stderr
    # Must not have been misclassified as the "no default base" case.
    assert not isinstance(excinfo.value, BranchNotFoundError)


@pytest.mark.requires_git
def test_create_without_base_propagates_128_with_non_unborn_stderr(
    manager: WorktreeManager, git_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    """Round-2 reviewer finding (codex, [blocking]): exit code 128 alone is
    not a precise-enough signal for "unborn HEAD" -- git also returns 128
    for other failures (e.g. corrupt refs, "not a git repository"). A 128
    exit whose stderr does NOT contain the unborn-HEAD signature ("unknown
    revision or path not in the working tree") must propagate as
    GitCommandError, not be silently folded into the "" / unborn-HEAD
    bucket."""
    real_run_git = manager_module._run_git

    def _fake_run_git(args, cwd=None, **kwargs):
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return subprocess.CompletedProcess(
                args=["git", *args],
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository (or any parent up to mount point /)",
            )
        return real_run_git(args, cwd=cwd, **kwargs)

    monkeypatch.setattr(manager_module, "_run_git", _fake_run_git)

    with pytest.raises(GitCommandError) as excinfo:
        manager.create(str(git_repo), "feature/128-not-unborn")

    assert excinfo.value.returncode == 128
    assert "not a git repository" in excinfo.value.stderr
    # Must not have been misclassified as the "no default base" case.
    assert not isinstance(excinfo.value, BranchNotFoundError)


@pytest.mark.requires_git
def test_create_without_base_skips_fetch_and_uses_local_ref(
    manager: WorktreeManager, git_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    """git_repo has no `origin` remote. Defaulting `base` (fetch left at its
    default True) must not attempt any network fetch and must branch from
    the local `main` ref."""
    calls = _make_run_git_spy(monkeypatch)

    rec = manager.create(str(git_repo), "feature/offline-default")

    assert not any(c and c[0] == "fetch" for c in calls)
    assert not any("origin/main" in c for c in calls)

    wt_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=rec.path,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    main_head = subprocess.run(
        ["git", "rev-parse", "main"], cwd=git_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert wt_head == main_head


@pytest.mark.requires_git
def test_create_without_base_with_fetch_false_is_equivalent(
    manager: WorktreeManager, git_repo: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _make_run_git_spy(monkeypatch)

    rec = manager.create(str(git_repo), "feature/offline-default-2", fetch=False)

    assert not any(c and c[0] == "fetch" for c in calls)
    assert not any("origin/main" in c for c in calls)

    wt_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=rec.path,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    main_head = subprocess.run(
        ["git", "rev-parse", "main"], cwd=git_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert wt_head == main_head


@pytest.mark.requires_git
def test_create_without_base_from_linked_worktree_uses_main_clone_branch(
    manager: WorktreeManager, git_repo: Path, linked_worktree: Path
):
    """linked_worktree is checked out on `feature/alpha`; git_repo (the main
    clone) is on `main`. Calling create() with repo_root pointing INSIDE the
    linked worktree must still default `base` to the main clone's branch
    (`main`), not the linked worktree's (`feature/alpha`)."""
    # Advance `main` so its tip differs from `feature/alpha`'s -- otherwise
    # the SHAs coincide and the test can't discriminate between the two.
    (git_repo / "extra.txt").write_text("second commit\n", encoding="utf-8")
    _git("add", "-A", cwd=git_repo)
    _git("commit", "-q", "-m", "second commit on main", cwd=git_repo)

    rec = manager.create(str(linked_worktree), "feature/from-linked")

    assert rec.repo_root == git_repo.resolve().as_posix()

    wt_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=rec.path,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    main_head = subprocess.run(
        ["git", "rev-parse", "main"], cwd=git_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    alpha_head = subprocess.run(
        ["git", "rev-parse", "feature/alpha"], cwd=git_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert wt_head == main_head
    assert wt_head != alpha_head


@pytest.mark.requires_git
def test_create_unknown_branch_with_base(
    manager: WorktreeManager, git_repo: Path
):
    rec = manager.create(str(git_repo), "feature/new", base="main", fetch=False)
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
# Ticket #102: GitTimeoutError network-vs-local classification
# ---------------------------------------------------------------------------


def test_run_git_timeout_on_fetch_flags_network_operation(monkeypatch):
    """R1: a timed-out network git command (fetch) is flagged .network=True,
    and the message names it as a network operation for retry-worthiness.
    """

    class _HangingPopen:
        def __init__(self, *args, **kwargs):
            self.returncode = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd=["git", "fetch"], timeout=timeout)

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(git_utils_module.subprocess, "Popen", _HangingPopen)

    with pytest.raises(GitTimeoutError) as excinfo:
        _run_git(["fetch", "origin", "main"], timeout=0.05)

    err = excinfo.value
    assert err.network is True
    assert err.command == ["git", "fetch", "origin", "main"]
    assert "network operation" in str(err)


@pytest.mark.parametrize(
    "command",
    [
        ["git", "-c", "protocol.version=2", "fetch", "origin", "main"],
        ["git", "--no-pager", "fetch", "origin", "main"],
        ["fetch", "origin", "main"],  # no leading "git" token
    ],
)
def test_git_subcommand_classifies_fetch_variants_as_network(command):
    from lib_python_worktree.core._exceptions import _git_subcommand

    assert _git_subcommand(command) == "fetch"


@pytest.mark.parametrize(
    "command, expected",
    [
        (["git", "-C", "repo", "fetch", "origin"], "fetch"),
        (["git", "--git-dir", "repo", "fetch", "origin"], "fetch"),
        (["git", "--work-tree", "wt", "fetch", "origin"], "fetch"),
        (["git", "--namespace", "ns", "push", "origin"], "push"),
        (["git", "--super-prefix", "prefix/", "pull", "origin"], "pull"),
        (
            ["git", "--config-env", "http.extraHeader=AUTH", "clone", "url"],
            "clone",
        ),
    ],
)
def test_git_subcommand_skips_space_separated_global_option_values(command, expected):
    """Reviewer finding #1: global options that take a SEPARATE (space-
    delimited) value token -- not just ``-c``/``-C`` -- must have that value
    token skipped rather than mistaken for the subcommand. Without this,
    e.g. ``git --git-dir repo fetch origin`` misclassifies as subcommand
    ``"repo"`` instead of ``"fetch"``, silently defeating the network
    signal for a genuine timeout on such an invocation.
    """
    from lib_python_worktree.core._exceptions import _git_subcommand

    assert _git_subcommand(command) == expected


@pytest.mark.parametrize(
    "command, expected",
    [
        (["git", "--git-dir=repo", "fetch", "origin"], "fetch"),
        (["git", "--work-tree=wt", "fetch", "origin"], "fetch"),
        (["git", "--namespace=ns", "push", "origin"], "push"),
        (["git", "--super-prefix=prefix/", "pull", "origin"], "pull"),
        (
            ["git", "--config-env=http.extraHeader=AUTH", "clone", "url"],
            "clone",
        ),
    ],
)
def test_git_subcommand_equals_joined_global_options_do_not_consume_next_token(
    command, expected
):
    """The ``=``-joined form of a value-taking global option (e.g.
    ``--git-dir=/path``) is a single token and must NOT consume the
    following token as a value -- only the space-separated form does.
    """
    from lib_python_worktree.core._exceptions import _git_subcommand

    assert _git_subcommand(command) == expected


@pytest.mark.parametrize(
    "command, expected_subcommand, expected_network",
    [
        # Bare `--exec-path` takes no separate value per real git behaviour
        # (verified against git 2.53.0.windows.1: `git --exec-path rev-parse
        # --is-bare-repository` prints only the exec path and never runs
        # `rev-parse`), so it must stay OUT of `_takes_value` -- the next
        # bare token is the subcommand, not a consumed value.
        (["git", "--exec-path", "rev-parse"], "rev-parse", False),
        (["git", "--exec-path=/some/path", "fetch", "origin"], "fetch", True),
    ],
)
def test_git_subcommand_exec_path_does_not_take_a_separate_value(
    command, expected_subcommand, expected_network
):
    from lib_python_worktree.core._exceptions import _git_subcommand

    assert _git_subcommand(command) == expected_subcommand
    err = GitTimeoutError(command, 1.0)
    assert err.network is expected_network


@pytest.mark.parametrize("command", [["git"], []])
def test_git_subcommand_handles_degenerate_commands_without_raising(command):
    from lib_python_worktree.core._exceptions import _git_subcommand

    assert _git_subcommand(command) is None
    err = GitTimeoutError(command, 1.0)
    assert err.network is False


def test_run_git_timeout_on_local_command_is_not_network(monkeypatch):
    """R2: a timed-out local git command (worktree list) is NOT flagged as
    network, and the message text is unchanged (exact-match regression guard).
    """

    class _HangingPopen:
        def __init__(self, *args, **kwargs):
            self.returncode = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd=["git", "worktree"], timeout=timeout)

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(git_utils_module.subprocess, "Popen", _HangingPopen)

    with pytest.raises(GitTimeoutError) as excinfo:
        _run_git(["worktree", "list", "--porcelain"], timeout=0.05)

    err = excinfo.value
    assert err.network is False
    assert str(err) == (
        f"git command timed out after {err.elapsed:.1f}s: "
        f"git worktree list --porcelain"
    )


@pytest.mark.parametrize(
    "command",
    [
        ["git", "remote", "-v"],
        ["git", "submodule", "status"],
    ],
)
def test_git_subcommand_remote_and_submodule_are_not_network(command):
    from lib_python_worktree.core._exceptions import _git_subcommand, _NETWORK_SUBCOMMANDS

    assert _git_subcommand(command) not in _NETWORK_SUBCOMMANDS


def test_git_timeout_error_construction_is_backwards_compatible():
    """R3: positional construction still works, still subclasses
    WorktreeError, and the ``network`` kwarg override wins over derivation.
    """

    err = GitTimeoutError(["git", "rev-parse"], 30.0)
    assert isinstance(err, WorktreeError)
    assert err.command == ["git", "rev-parse"]
    assert err.elapsed == 30.0
    assert err.network is False

    # Explicit override wins: force a normally-local command to be flagged...
    forced_true = GitTimeoutError(["git", "rev-parse"], 30.0, network=True)
    assert forced_true.network is True

    # ...and suppress it on a normally-network command.
    forced_false = GitTimeoutError(["git", "fetch", "origin"], 30.0, network=False)
    assert forced_false.network is False


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
    rec = manager.create(str(git_repo), "feature/new", base="main", fetch=False)
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
    rec = manager.create(str(git_repo), "feature/unmerged", base="main", fetch=False)
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
    rec = manager.create(str(git_repo), "feature/gone", base="main", fetch=False)
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
    rec = manager.create(str(git_repo), "feature/leak-test", base="main", fetch=False)
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
        # Ticket #105: adopt() never runs the setup: hook, so the adopted
        # record's setup_outcome must stay None -- distinct from
        # status="skipped", which only create() ever sets.
        assert rec.setup_outcome is None
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(oot_path)],
            cwd=git_repo,
            capture_output=True,
        )


@pytest.mark.requires_git
def test_manager_list_repo_synthesised_untracked_entry_setup_outcome_is_none(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    """Ticket #105: an untracked linked worktree synthesised by list_repo()
    (no persisted WorktreeRecord yet, tracked=False) must have
    setup_outcome is None -- it never went through create()'s setup: hook."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    oot_path = tmp_path / "oot-wt-listrepo"
    subprocess.run(
        ["git", "worktree", "add", str(oot_path), "feature/alpha"],
        cwd=git_repo,
        check=True,
        capture_output=True,
    )

    try:
        listing = mgr.list_repo(str(git_repo))
        # Untracked entries include both the synthesised primary (never
        # start()-ed) and the synthesised linked worktree created above --
        # every one of them must have setup_outcome is None, since none of
        # them ever went through create()'s setup: hook.
        untracked = [e for e in listing.entries if not e.tracked]
        assert len(untracked) >= 1, "expected at least one synthesised untracked entry"
        assert all(e.record.setup_outcome is None for e in untracked)
        linked = [e for e in untracked if e.record.backing == "worktree"]
        assert len(linked) == 1, "expected exactly one synthesised untracked linked worktree"
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


# ---------------------------------------------------------------------------
# Ticket #100: an untracked `.seretos/` convenience copy must not force
# force=True on a plain create() -> remove() cycle with zero real edits.
# ---------------------------------------------------------------------------


@pytest.mark.requires_git
def test_remove_untracked_contract_copy_no_force_succeeds(
    manager: WorktreeManager, git_repo: Path
):
    """Driving test (ticket #100): remove(force=False) must succeed on a
    checkout whose ONLY dirt is the untracked `.seretos/` copy the
    agent-worktree plugin drops into every new checkout after create()
    returns. Before the fix this always raised DirtyWorktreeError."""
    rec = manager.create(str(git_repo), "feature/alpha")
    wt_path = Path(rec.path)

    # Simulate the plugin's post-create convenience copy: an untracked
    # `.seretos/worktree-setup.yml` inside the checkout, with zero other
    # edits anywhere.
    contract_dir = wt_path / ".seretos"
    contract_dir.mkdir()
    (contract_dir / "worktree-setup.yml").write_text(
        "version: 1\nisolation: none\n", encoding="utf-8"
    )

    result = manager.remove(rec.id, force=False)

    assert result.status == "removed"
    assert not wt_path.exists()
    assert manager.state.get(rec.id) is None


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


# ---------------------------------------------------------------------------
# Ticket #23: WorktreeRecord paths must use forward slashes on all platforms
# ---------------------------------------------------------------------------


@pytest.mark.requires_git
def test_create_record_paths_use_forward_slashes(
    manager: WorktreeManager, git_repo: Path
):
    """Regression: create() must store repo_root and path with forward slashes.

    On Windows, Path.resolve() returns backslash-separated strings by default.
    The fix uses Path.as_posix() so the stored strings are always forward-slash,
    making them safe for cross-platform consumers and equality checks.
    """
    rec = manager.create(str(git_repo), "feature/alpha")
    assert "\\" not in rec.repo_root, (
        f"repo_root must use forward slashes, got: {rec.repo_root!r}"
    )
    assert "\\" not in rec.path, (
        f"path must use forward slashes, got: {rec.path!r}"
    )


@pytest.mark.requires_git
def test_find_by_branch_matches_after_create_forward_slash_input(
    manager: WorktreeManager, git_repo: Path
):
    """Regression: find_by_branch must find the record created by create().

    Before the fix, create() stored repo_root as a native-backslash string on
    Windows but the duplicate-check also used a backslash string, so it matched
    incidentally.  After the fix both sides use as_posix(), so a caller passing
    the forward-slash key should also find the record.
    """
    rec = manager.create(str(git_repo), "feature/alpha")
    # The key must match the stored forward-slash value.
    forward_slash_root = Path(str(git_repo)).resolve().as_posix()
    found = manager.state.find_by_branch(forward_slash_root, "feature/alpha")
    assert found is not None, (
        "find_by_branch must return the record when queried with a forward-slash key"
    )
    assert found.id == rec.id


# ---------------------------------------------------------------------------
# Ticket #25: WorktreeManager.start reads cmd from contract; stop runs steps
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, call, patch  # noqa: E402 – after sys-level imports

from lib_python_worktree.contract.schema import Step, WorktreeContract  # noqa: E402
from lib_python_worktree.contract.loader import ContractError, ContractValidationError  # noqa: E402
from lib_python_worktree.setup.runner import _resolve_shell  # noqa: E402

# NOTE: `_build_step_command` (ticket #109) is imported locally inside each
# test function that needs it below, rather than at module level here -- it
# is a brand-new helper introduced by this fix, and importing it at module
# level would make the *entire* test_manager.py module fail to collect
# against pre-fix code (masking every other test's genuine RED/GREEN result).


def _make_mgr_in_memory(tmp_path: Path) -> WorktreeManager:
    return WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )


def _make_wt_record(wt_id: str = "wt-abc12345", **kwargs) -> "WorktreeRecord":
    defaults = dict(
        id=wt_id,
        repo_root="/fake/repo",
        branch="feature/x",
        path="/fake/store/wt-abc12345",
    )
    defaults.update(kwargs)
    return WorktreeRecord(**defaults)


# ---------------------------------------------------------------------------
# TestManagerListReconciles -- ticket #110
# ---------------------------------------------------------------------------

class TestManagerListReconciles:
    """WorktreeManager.list() must refresh liveness via reconcile() for a
    YamlStateStore-backed manager (ticket #110), exactly like __init__
    already does -- otherwise a caller that only ever calls list() (never
    re-instantiating the manager) keeps seeing a stale dead pid/status
    indefinitely. Gated on the same reconcile_on_init flag so tests/callers
    that opt out at construction time keep that opt-out on every list()
    call too, and InMemoryStateStore-backed managers (reconcile() only
    accepts a YamlStateStore) are unaffected either way.
    """

    def test_list_reconciles_dead_pid_for_yaml_state_store(self, tmp_path: Path):
        """Driving test: a record with a dead pid, written directly to a
        YamlStateStore (bypassing start()), must have that pid cleared and
        status normalized to "stopped" by the time list() returns -- proving
        list() actually ran reconcile(), not just read the store as-is."""
        state_dir = tmp_path / "state"
        existing_path = tmp_path / "wt-reconcile-list"
        existing_path.mkdir()
        store = YamlStateStore(state_dir=state_dir)
        mgr = WorktreeManager(
            config=ManagerConfig(store_root=tmp_path / "store"),
            state=store,
            reconcile_on_init=True,
        )

        if _pid_alive(99999997):
            pytest.skip("PID 99999997 is alive on this machine — skipping")
        dead_record = WorktreeRecord(
            id="wt-reconcile-list",
            repo_root="/fake/repo",
            branch="feature/x",
            path=str(existing_path),
            status="running",
            pids={"main": 99999997},
        )
        store.add(dead_record)

        listed = mgr.list()

        assert len(listed) == 1
        assert listed[0].pids == {}
        assert listed[0].status == "stopped"

    def test_list_does_not_reconcile_when_reconcile_on_init_false(self, tmp_path: Path):
        """Edge case: a manager constructed with reconcile_on_init=False
        must keep that opt-out on list() too -- a dead pid must NOT be
        cleared."""
        state_dir = tmp_path / "state"
        existing_path = tmp_path / "wt-no-reconcile-list"
        existing_path.mkdir()
        store = YamlStateStore(state_dir=state_dir)
        mgr = WorktreeManager(
            config=ManagerConfig(store_root=tmp_path / "store"),
            state=store,
            reconcile_on_init=False,
        )

        if _pid_alive(99999996):
            pytest.skip("PID 99999996 is alive on this machine — skipping")
        dead_record = WorktreeRecord(
            id="wt-no-reconcile-list",
            repo_root="/fake/repo",
            branch="feature/x",
            path=str(existing_path),
            status="running",
            pids={"main": 99999996},
        )
        store.add(dead_record)

        listed = mgr.list()

        assert len(listed) == 1
        assert listed[0].pids == {"main": 99999996}
        assert listed[0].status == "running"

    def test_list_with_in_memory_state_store_unaffected(self, tmp_path: Path):
        """Edge case: an InMemoryStateStore-backed manager's list() must not
        raise and must not attempt reconcile() (which only accepts a
        YamlStateStore) -- record contents pass through unchanged."""
        mgr = _make_mgr_in_memory(tmp_path)  # reconcile_on_init=False
        record = _make_wt_record(pids={"main": 99999995}, status="running")
        mgr.state.add(record)

        listed = mgr.list()  # must not raise

        assert len(listed) == 1
        assert listed[0].pids == {"main": 99999995}
        assert listed[0].status == "running"

    @pytest.mark.requires_git
    def test_list_repo_reconciles_dead_pid_for_yaml_state_store(
        self, tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG002
    ):
        """Driving test (review round 2, blocking finding 1): mirrors
        test_list_reconciles_dead_pid_for_yaml_state_store above, but through
        list_repo() -- the repo-scoped listing used to call
        self.state.list() directly, bypassing the same reconcile gate
        list() uses, so a caller polling only list_repo() kept seeing a
        stale dead pid / stale "running" status indefinitely. A tracked
        primary-checkout record with a dead pid, written directly to a
        YamlStateStore (bypassing start()), must have that pid cleared and
        status normalized to "stopped" by the time list_repo() returns."""
        from lib_python_worktree.core.checkout import primary_id_for

        state_dir = tmp_path / "state"
        store = YamlStateStore(state_dir=state_dir)
        mgr = WorktreeManager(
            config=ManagerConfig(store_root=tmp_path / "store"),
            state=store,
            reconcile_on_init=True,
        )

        if _pid_alive(99999993):
            pytest.skip("PID 99999993 is alive on this machine — skipping")
        dead_record = WorktreeRecord(
            id=primary_id_for(git_repo),
            repo_root=str(git_repo.resolve().as_posix()),
            branch=None,
            path=str(git_repo),
            backing="primary",
            status="running",
            pids={"main": 99999993},
        )
        store.add(dead_record)

        listing = mgr.list_repo(str(git_repo))

        primary_entries = [
            e for e in listing.entries if e.record.id == primary_id_for(git_repo)
        ]
        assert len(primary_entries) == 1
        assert primary_entries[0].record.pids == {}
        assert primary_entries[0].record.status == "stopped"


def test_manager_start_reads_cmd_from_contract(tmp_path: Path):
    """start() builds the cmd from contract.start[0] and passes it to _lifecycle_start."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    expected_shell = _resolve_shell(None)  # platform-appropriate prefix
    expected_cmd = _build_step_command(expected_shell, "python server.py")

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)

    call_kwargs = mock_start.call_args
    assert call_kwargs.args[0] == record.id
    assert call_kwargs.args[1] == expected_cmd
    assert call_kwargs.kwargs["store"] is mgr.state
    assert call_kwargs.kwargs["role"] == "main"
    assert call_kwargs.kwargs["cwd"] == record.path
    # env must be a dict (built by _build_worktree_env, not None)
    built_env = call_kwargs.kwargs["env"]
    assert isinstance(built_env, dict)
    assert built_env["WORKTREE_ID"] == record.id
    assert built_env["WORKTREE_PATH"] == record.path
    assert built_env["WORKTREE_BRANCH"] == record.branch


def test_manager_start_defaults_cwd_to_worktree_path(tmp_path: Path):
    """Ticket #81: a cwd=None start defaults to record.path, not the caller's cwd."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id, cwd=None)

    call_kwargs = mock_start.call_args
    assert call_kwargs.kwargs["cwd"] == record.path


def test_manager_start_explicit_cwd_passed_through(tmp_path: Path):
    """An explicit cwd reaches _lifecycle_start unchanged, not overridden by record.path."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id, cwd="/some/dir")

    call_kwargs = mock_start.call_args
    assert call_kwargs.kwargs["cwd"] == "/some/dir"


def test_manager_start_injects_worktree_env_vars(tmp_path: Path):
    """start() injects WORKTREE_ID/PATH/BRANCH and WORKTREE_PORT_* into the child env."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(ports={"web": 8080, "grpc": 50051})
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)

    built_env = mock_start.call_args.kwargs["env"]
    assert isinstance(built_env, dict)
    assert built_env["WORKTREE_ID"] == record.id
    assert built_env["WORKTREE_PATH"] == record.path
    assert built_env["WORKTREE_BRANCH"] == record.branch
    assert built_env["WORKTREE_PORT_WEB"] == "8080"
    assert built_env["WORKTREE_PORT_GRPC"] == "50051"


def test_manager_start_injects_env_empty_ports(tmp_path: Path):
    """start() with ports={} injects identity vars but no WORKTREE_PORT_* keys."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(ports={})
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)

    built_env = mock_start.call_args.kwargs["env"]
    assert isinstance(built_env, dict)
    assert built_env["WORKTREE_ID"] == record.id
    # no port keys injected
    port_keys = [k for k in built_env if k.startswith("WORKTREE_PORT_")]
    assert port_keys == []


def test_manager_start_injects_env_uppercase_slot(tmp_path: Path):
    """start() normalises slot names to upper-case for WORKTREE_PORT_* keys."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(ports={"Web": 9000})
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)

    built_env = mock_start.call_args.kwargs["env"]
    assert "WORKTREE_PORT_WEB" in built_env
    assert built_env["WORKTREE_PORT_WEB"] == "9000"
    assert "WORKTREE_PORT_Web" not in built_env


def test_manager_start_caller_env_overrides_worktree_vars(tmp_path: Path):
    """Caller-supplied env wins on collision; worktree vars and os.environ are still present for non-colliding keys."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(ports={"web": 3000})
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )

    # MY_VAR is a brand-new key; WORKTREE_ID collides with an injected worktree var.
    caller_env = {"MY_VAR": "1", "WORKTREE_ID": "overridden-id"}

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id, env=caller_env)

    built_env = mock_start.call_args.kwargs["env"]
    # caller-supplied brand-new key is present
    assert built_env["MY_VAR"] == "1"
    # caller wins on collision: its value must overwrite the injected worktree var
    assert built_env["WORKTREE_ID"] == "overridden-id"
    # non-colliding worktree var still present
    assert built_env["WORKTREE_PORT_WEB"] == "3000"
    # inherited os.environ key (PATH always exists on every platform)
    assert "PATH" in built_env


def test_manager_start_no_caller_env_inherits_os_environ(tmp_path: Path):
    """When caller passes env=None, os.environ keys (e.g. PATH) are inherited."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)  # env=None by default

    built_env = mock_start.call_args.kwargs["env"]
    assert "PATH" in built_env
    assert built_env["WORKTREE_ID"] == record.id


def test_manager_start_empty_contract_is_noop_ready(tmp_path: Path):
    """start() with no configured start: step is a no-op that marks ready (ticket #41).

    No process is spawned, no WorktreeError is raised, _lifecycle_start is not
    called, the worktree gains status="ready" and records no PID, and the
    persisted record reflects the same.
    """
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(version=1, isolation="full", start=[])

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        result = mgr.start(record.id)

    mock_start.assert_not_called()
    assert result.status == "ready"
    assert result.pids == {}
    assert mgr.state.get(record.id).status == "ready"
    # Ticket #104: no process was spawned, so no variant is recorded either.
    assert result.variants == {}


def test_manager_start_empty_contract_clears_leftover_teardown_ran(tmp_path: Path):
    """Ticket #126: the no-op "ready" start path never reaches
    _lifecycle_start, so it needs its own reset of a leftover
    teardown_ran=True marker (e.g. left by a prior remove() attempt) --
    a restarted environment is a new logical lifecycle and must earn a
    fresh teardown."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(teardown_ran=True)
    mgr.state.add(record)

    fake_contract = WorktreeContract(version=1, isolation="full", start=[])

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        result = mgr.start(record.id)

    mock_start.assert_not_called()
    assert result.status == "ready"
    assert result.teardown_ran is False
    assert mgr.state.get(record.id).teardown_ran is False


def test_manager_start_no_contract_is_noop_ready(tmp_path: Path):
    """start() with a missing contract (implicit isolation:none) is a no-op ready start.

    Exercises the real loader path: no .seretos/worktree-setup.yml exists at
    repo_root, so load() yields an empty start: list and start() must not raise.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root))
    mgr.state.add(record)

    with patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start:
        result = mgr.start(record.id)

    mock_start.assert_not_called()
    assert result.status == "ready"
    assert result.pids == {}


# ---------------------------------------------------------------------------
# Ticket #100: start() flags a checkout-local contract copy it never reads.
# ---------------------------------------------------------------------------


def test_start_flags_shadowed_checkout_contract(tmp_path: Path, caplog):
    """Driving test (ticket #100): a checkout-local `.seretos/worktree-
    setup.yml` that would yield a DIFFERENT contract than the repo-root one
    actually read must be flagged on the returned record's
    `shadowed_contract`, and the identical message must be logged at
    WARNING. Before the fix, `WorktreeRecord` had no such attribute at all."""
    import logging as _logging

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    # Repo root has no start: step -> no-op ready path.
    _write_contract(repo_root, "version: 1\nisolation: none\n")
    # Checkout-local copy DOES have a start: step -> different contract.
    _write_contract(
        checkout,
        "version: 1\nisolation: full\nstart:\n  - run: npm run dev\n",
    )

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), path=str(checkout))
    mgr.state.add(record)

    caplog.set_level(_logging.WARNING, logger="lib_python_worktree.core.manager")

    result = mgr.start(record.id)

    assert result.status == "ready"
    assert result.pids == {}
    assert result.shadowed_contract is not None
    shadow = result.shadowed_contract
    assert shadow.reason == "differs"
    assert Path(shadow.path) == (checkout / ".seretos" / "worktree-setup.yml")
    assert Path(shadow.used_path) == (repo_root / ".seretos" / "worktree-setup.yml")
    assert any(rec.message == shadow.message for rec in caplog.records), (
        "the logged WARNING message must be identical to shadow.message"
    )


def test_start_no_shadow_when_checkout_contract_identical(tmp_path: Path):
    """A checkout-local copy that parses to the SAME contract as the one
    actually used must not be flagged -- since the plugin copies `.seretos/`
    into every checkout, an unconditional flag would fire on every single
    start() and be pure noise."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    contract_text = "version: 1\nisolation: none\n"
    _write_contract(repo_root, contract_text)
    _write_contract(checkout, contract_text)

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), path=str(checkout))
    mgr.state.add(record)

    result = mgr.start(record.id)

    assert result.shadowed_contract is None


def test_start_no_shadow_when_no_checkout_contract(tmp_path: Path):
    """No checkout-local `.seretos/` copy at all -> no flag."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: none\n")

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), path=str(checkout))
    mgr.state.add(record)

    result = mgr.start(record.id)

    assert result.shadowed_contract is None


def test_start_shadow_reason_unreadable_on_malformed_checkout_contract(
    tmp_path: Path,
):
    """A checkout-local copy that exists but fails to parse/validate must be
    flagged with reason="unreadable" -- and, unlike a malformed REPO-ROOT
    contract (which raises through start(), see the ticket #70 tests below),
    start() itself must still return normally."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: none\n")
    _write_contract(checkout, "version: 1\n  bad: indent: here\n")

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), path=str(checkout))
    mgr.state.add(record)

    result = mgr.start(record.id)

    assert result.status == "ready"
    assert result.shadowed_contract is not None
    assert result.shadowed_contract.reason == "unreadable"


def test_start_shadow_reason_unreadable_on_dangling_symlink_checkout_contract(
    tmp_path: Path,
):
    """A checkout-local `.seretos/worktree-setup.yml` that is a broken/
    dangling symlink must be flagged with reason="unreadable", not silently
    treated as "no checkout-local contract at all". `Path.exists()` follows
    symlinks and would report False for a dangling link -- indistinguishable
    from "nothing there" -- which is why the implementation must use
    `os.path.lexists()` for the presence check instead. start() must still
    return normally without raising.

    Symlink creation on Windows normally requires elevation/Developer Mode,
    so this is skipped rather than failed on an unprivileged runner.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: none\n")

    shadow_dir = checkout / ".seretos"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    shadow_file = shadow_dir / "worktree-setup.yml"
    try:
        os.symlink(shadow_dir / "does-not-exist.yml", shadow_file)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this machine")

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), path=str(checkout))
    mgr.state.add(record)

    result = mgr.start(record.id)

    assert result.status == "ready"
    assert result.shadowed_contract is not None
    assert result.shadowed_contract.reason == "unreadable"


def test_start_flags_shadowed_contract_on_real_spawn_exit_path(tmp_path: Path):
    """Blocking-finding-2 follow-up (ticket #100): the shadowed-contract flag
    must also survive the REAL-SPAWN exit path (repo-root contract HAS a
    `start:` step, so start() calls `_lifecycle_start(...)` rather than
    taking the no-op `status="ready"` shortcut).

    `_lifecycle_start` hands back a freshly store-loaded record that never
    carries the transient `shadowed_contract` field (see the comment at the
    `result.shadowed_contract = shadowed_contract` assignment in
    `manager.start()`), so start() must assign it explicitly onto whatever
    `_lifecycle_start` returns. Every OTHER shadowed_contract test in this
    file uses a contract with no `start:` step, so they only ever exercise
    the no-op branch's own (separate) assignment -- this is the only test
    that pins the real-spawn branch's assignment specifically. Uses a
    distinct `spawned_record` object (not the same instance as `record`,
    and with `shadowed_contract` left at its default) so the assertion can
    only pass via the explicit assignment onto the object `_lifecycle_start`
    returns, not by accident via shared identity with `record`."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    # Repo root HAS a start: step -> real-spawn path (not the no-op "ready" one).
    _write_contract(repo_root, "version: 1\nisolation: full\nstart:\n  - run: npm run dev\n")
    # Checkout-local copy differs -> shadowed_contract must be populated.
    _write_contract(
        checkout,
        "version: 1\nisolation: none\n",
    )

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), path=str(checkout))
    mgr.state.add(record)

    spawned_record = _make_wt_record(
        wt_id=record.id,
        repo_root=str(repo_root),
        path=str(checkout),
        status="running",
        pids={"main": 4242},
    )
    assert spawned_record.shadowed_contract is None
    assert spawned_record is not record

    with patch(
        "lib_python_worktree.core.manager._lifecycle_start",
        return_value=spawned_record,
    ) as mock_start:
        result = mgr.start(record.id)

    mock_start.assert_called_once()
    assert result is spawned_record
    assert result.status == "running"
    assert result.shadowed_contract is not None
    shadow = result.shadowed_contract
    assert shadow.reason == "differs"
    assert Path(shadow.path) == (checkout / ".seretos" / "worktree-setup.yml")
    assert Path(shadow.used_path) == (repo_root / ".seretos" / "worktree-setup.yml")


# ---------------------------------------------------------------------------
# Ticket #70 (Befund 1b/1c): malformed/invalid contracts must raise loudly
# through the FULL start() call path, not be swallowed into a silent "ready".
#
# These deliberately duplicate a little of test_contract.py's loader-level
# coverage but assert at the manager.start() boundary, proving the error
# propagates unguarded all the way out of the public API — not just out of
# the loader function in isolation.
# ---------------------------------------------------------------------------


def _write_contract(repo_root: Path, text: str) -> None:
    p = repo_root / ".seretos" / "worktree-setup.yml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_manager_start_contract_start_as_string_list_raises_validation_error(
    tmp_path: Path,
):
    """start: as a plain list of strings (not step mappings) raises
    ContractValidationError through start(), not a silent no-op ready."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nstart:\n  - npm run start\n  - npm run build\n",
    )
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root))
    mgr.state.add(record)

    with pytest.raises(ContractValidationError):
        mgr.start(record.id)


def test_manager_start_contract_start_as_name_keyed_dict_raises_validation_error(
    tmp_path: Path,
):
    """start: as a name-keyed/role-nested mapping (instead of a list) raises
    ContractValidationError through start()."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nstart:\n  default:\n    run: npm run start\n",
    )
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root))
    mgr.state.add(record)

    with pytest.raises(ContractValidationError):
        mgr.start(record.id)


def test_manager_start_contract_step_with_command_field_raises_validation_error(
    tmp_path: Path,
):
    """A start: step shaped {name, command} (wrong field name, missing
    required run:) raises ContractValidationError through start()."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nstart:\n  - name: default\n    command: npm run start\n",
    )
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root))
    mgr.state.add(record)

    with pytest.raises(ContractValidationError):
        mgr.start(record.id)


def test_manager_start_contract_invalid_yaml_raises_contract_error(tmp_path: Path):
    """Syntactically unparseable YAML in worktree-setup.yml raises
    ContractError through start() (not ContractValidationError — parse
    failure, not schema failure)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(repo_root, "version: 1\n  bad: indent: here\n")
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root))
    mgr.state.add(record)

    with pytest.raises(ContractError) as exc_info:
        mgr.start(record.id)

    # Must be ContractError but NOT the validation subclass.
    assert not isinstance(exc_info.value, ContractValidationError)


def test_manager_start_named_variant_selected(tmp_path: Path):
    """start(variant="headless") picks the step with name="headless" from a multi-step contract."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    headless_step = Step(run="python server.py --headless", name="headless")
    gui_step = Step(run="python server.py --gui", name="gui")
    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[headless_step, gui_step],
    )
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    expected_shell = _resolve_shell(None)
    expected_cmd = _build_step_command(expected_shell, "python server.py --headless")

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id, variant="headless")

    call_kwargs = mock_start.call_args
    assert call_kwargs.args[1] == expected_cmd


def test_manager_start_forwards_variant_to_lifecycle_start(tmp_path: Path):
    """Ticket #104: start(variant="headless") forwards variant="headless"
    to the delegated process_lifecycle.start call, so it can be recorded
    under record.variants[role]."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    headless_step = Step(run="python server.py --headless", name="headless")
    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[headless_step],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id, variant="headless")

    call_kwargs = mock_start.call_args
    assert call_kwargs.kwargs["variant"] == "headless"


def test_manager_start_backward_compat_single_unnamed_step(tmp_path: Path):
    """start() with no variant still works when there is exactly one unnamed step."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    expected_shell = _resolve_shell(None)
    expected_cmd = _build_step_command(expected_shell, "python server.py")

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)

    call_kwargs = mock_start.call_args
    assert call_kwargs.args[1] == expected_cmd


def test_manager_start_named_default_step_selected(tmp_path: Path):
    """start() with variant="default" picks the step explicitly named "default"."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    default_step = Step(run="python server.py", name="default")
    other_step = Step(run="python server.py --debug", name="debug")
    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[default_step, other_step],
    )
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    expected_shell = _resolve_shell(None)
    expected_cmd = _build_step_command(expected_shell, "python server.py")

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id, variant="default")

    call_kwargs = mock_start.call_args
    assert call_kwargs.args[1] == expected_cmd


def test_manager_start_named_default_wins_over_unnamed(tmp_path: Path):
    """Explicit name="default" step wins over a co-existing unnamed step.

    When the contract has both an unnamed step AND a step explicitly named
    "default", calling start() (which defaults to variant="default") must
    invoke _lifecycle_start with the *named* default step's command, not the
    unnamed step's command.
    """
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    unnamed_step = Step(run="python unnamed.py")         # no name
    default_step = Step(run="python default.py", name="default")
    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[unnamed_step, default_step],
    )
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    expected_shell = _resolve_shell(None)
    expected_cmd = _build_step_command(expected_shell, "python default.py")

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)  # variant="default" by default

    call_kwargs = mock_start.call_args
    assert call_kwargs.args[1] == expected_cmd, (
        "Expected the explicitly named 'default' step to win over the unnamed step"
    )


def test_manager_start_uses_encoded_command_argv(tmp_path: Path, monkeypatch):
    """Ticket #109: manager.start() funnels its argv assembly through the
    same _build_step_command helper as SetupRunner._invoke, so a `start:`
    step gets the identical -EncodedCommand fix as a `setup:` step on
    Windows -- not a hand-rolled `[*_resolve_shell(...), step.run]`
    concatenation that would still be vulnerable to the quoting bug.
    """
    import lib_python_worktree.setup.runner as _runner_module  # noqa: PLC0415

    monkeypatch.setattr(_runner_module.sys, "platform", "win32")
    # `sys` is a process-wide singleton, so the patch above also makes
    # _env_utils._get_user_profile_env() (called by manager._build_worktree_env
    # while assembling the args to the mocked _lifecycle_start below) believe
    # it is on Windows and try a real `import winreg`, which does not exist
    # on non-Windows CI runners. Stub it out -- this test only asserts on the
    # argv shape, not on environment-variable sourcing.
    monkeypatch.setattr(
        "lib_python_worktree.core.manager._get_user_profile_env",
        lambda: dict(os.environ),
    )

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )
    expected_cmd = _runner_module._build_step_command(_resolve_shell(None), "python server.py")
    assert expected_cmd[:4] == [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
    ]

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)

    call_kwargs = mock_start.call_args
    assert call_kwargs.args[1] == expected_cmd


def test_manager_start_unknown_variant_raises_worktree_error(tmp_path: Path):
    """start(variant="nonexistent") raises WorktreeError listing available names."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="cmd1", name="headless"), Step(run="cmd2", name="gui")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        pytest.raises(WorktreeError) as exc_info,
    ):
        mgr.start(record.id, variant="nonexistent")

    error_msg = str(exc_info.value)
    assert "nonexistent" in error_msg
    assert "headless" in error_msg
    assert "gui" in error_msg


def test_manager_start_unknown_variant_is_valueerror(tmp_path: Path):
    """Ticket #70: start(variant="nonexistent") must ALSO be catchable as a
    plain ValueError, matching the start() docstring's documented contract.

    Before the fix, the unknown-variant branch raised a bare WorktreeError
    (RuntimeError-based), silently breaking any caller that followed the
    docstring and wrapped the call in `except ValueError`.
    """
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="cmd1", name="headless"), Step(run="cmd2", name="gui")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        pytest.raises(ValueError) as exc_info,
    ):
        mgr.start(record.id, variant="nonexistent")

    error_msg = str(exc_info.value)
    assert "nonexistent" in error_msg
    assert "headless" in error_msg
    assert "gui" in error_msg

    # And it must still be a WorktreeError subclass (so existing callers
    # catching WorktreeError keep working unchanged).
    assert issubclass(UnknownVariantError, WorktreeError)
    assert issubclass(UnknownVariantError, ValueError)
    assert isinstance(exc_info.value, UnknownVariantError)


def test_manager_start_multi_unnamed_steps_unknown_default_raises(tmp_path: Path):
    """start() with variant="default" and multiple unnamed steps raises WorktreeError."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="cmd1"), Step(run="cmd2")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        pytest.raises(WorktreeError) as exc_info,
    ):
        mgr.start(record.id)

    assert "default" in str(exc_info.value)


@pytest.mark.parametrize("step_name", ["main", "web", "dev server"])
def test_manager_start_default_resolves_lone_named_step(tmp_path: Path, step_name: str):
    """Ticket #112: bare start() (and start(variant="default")) must resolve
    a contract whose sole start: step carries a name, not just a lone
    unnamed one. Before the fix this raised UnknownVariantError on the very
    first call."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    named_step = Step(run="python server.py", name=step_name)
    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[named_step],
    )
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    expected_shell = _resolve_shell(None)
    expected_cmd = _build_step_command(expected_shell, "python server.py")

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)

    call_kwargs = mock_start.call_args
    assert call_kwargs.args[1] == expected_cmd

    # Also covers variant="default" passed explicitly on the same contract.
    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id, variant="default")

    call_kwargs = mock_start.call_args
    assert call_kwargs.args[1] == expected_cmd


def test_manager_start_default_lone_named_step_records_step_name(tmp_path: Path):
    """Ticket #112: when the fallback resolves a lone named step, the
    variant recorded for the role must be the step's own name ("main"),
    not the literal "default"."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py", name="main")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)

    call_kwargs = mock_start.call_args
    assert call_kwargs.kwargs["variant"] == "main"


def test_manager_start_lone_unnamed_step_still_records_default(tmp_path: Path):
    """Back-compat: a lone unnamed step still records variant="default"
    for the role (unchanged by the ticket #112 fallback broadening)."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)

    call_kwargs = mock_start.call_args
    assert call_kwargs.kwargs["variant"] == "default"


def test_manager_start_default_prefers_lone_unnamed_over_named_sibling(tmp_path: Path):
    """Load-bearing: rule 1 (lone unnamed step) must still win over rule 2
    (lone step total) by ordering. A two-step contract with one unnamed and
    one named step resolves via the pre-existing unnamed-step rule, not the
    new single-step-total rule."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    unnamed_step = Step(run="python a.py")
    gui_step = Step(run="python b.py", name="gui")
    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[unnamed_step, gui_step],
    )
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    expected_shell = _resolve_shell(None)
    expected_cmd = _build_step_command(expected_shell, "python a.py")

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_start") as mock_start,
    ):
        mock_start.return_value = record
        mgr.start(record.id)

    call_kwargs = mock_start.call_args
    assert call_kwargs.args[1] == expected_cmd
    assert call_kwargs.kwargs["variant"] == "default"


def test_manager_start_default_two_named_steps_still_raises(tmp_path: Path):
    """Multi-step contracts (two named steps) keep raising UnknownVariantError
    unchanged -- the ticket #112 fallback only applies when there is exactly
    one start: step total."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="cmd1", name="headless"), Step(run="cmd2", name="gui")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        pytest.raises(UnknownVariantError) as exc_info,
    ):
        mgr.start(record.id)

    assert exc_info.value.available == ["headless", "gui"]


def test_manager_start_default_two_unnamed_steps_still_raises(tmp_path: Path):
    """Multi-step contracts (two unnamed steps) keep raising
    UnknownVariantError unchanged -- exactly one step total is required for
    the new fallback rule to apply."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="cmd1"), Step(run="cmd2")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        pytest.raises(UnknownVariantError) as exc_info,
    ):
        mgr.start(record.id)

    # Ticket #131: two unnamed steps means no fallback tier is reachable
    # either -- `available` genuinely stays empty -- but the message must
    # no longer leave a bare `available: []` unexplained; it appends a
    # remediation clause pointing at the actual fix (distinct `name:`
    # values).
    assert exc_info.value.available == []
    assert "no start: step is addressable" in str(exc_info.value)
    assert "name:" in str(exc_info.value)


def test_manager_start_unknown_variant_message_omits_remediation_when_available_nonempty(
    tmp_path: Path,
):
    """The remediation clause added for ticket #131's empty-list case must
    NOT appear when `available` is non-empty -- it is guarded, not
    unconditional."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="cmd1", name="headless"), Step(run="cmd2", name="gui")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        pytest.raises(UnknownVariantError) as exc_info,
    ):
        mgr.start(record.id, variant="nonexistent")

    assert exc_info.value.available == ["headless", "gui"]
    assert "no start: step is addressable" not in str(exc_info.value)


def test_manager_start_unknown_variant_on_lone_named_step_still_raises(tmp_path: Path):
    """A lone named step ("main") does NOT make every variant name valid --
    requesting an unrelated variant still raises UnknownVariantError. Ticket
    #131: `available` now also surfaces the implicit "default" fallback --
    reachable here via the ticket #112 single-step-total tier -- alongside
    the step's own name, since `variant="default"` would also resolve this
    contract even though the specific "nope" request didn't."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py", name="main")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        pytest.raises(UnknownVariantError) as exc_info,
    ):
        mgr.start(record.id, variant="nope")

    assert exc_info.value.available == ["main", "default"]


def test_manager_start_unknown_variant_lone_unnamed_step_surfaces_default(
    tmp_path: Path,
):
    """Ticket #131 driving test (the reported defect): a contract with a
    single unnamed start: step -- which environment_start() with no variant
    argument resolves successfully via the tier-2 fallback -- must not
    report available=[] on an unrelated failed variant request. The
    reachable implicit "default" fallback is surfaced instead of an
    empty/misleading list."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        pytest.raises(UnknownVariantError) as exc_info,
    ):
        mgr.start(record.id, variant="nope")

    assert exc_info.value.available == ["default"]
    assert "default" in str(exc_info.value)


def test_manager_start_unknown_variant_mixed_unnamed_and_named_surfaces_default(
    tmp_path: Path,
):
    """Ticket #131: an unnamed step plus a named sibling -- the unnamed step
    is what "default" resolves to (tier 2 wins over tier 3) -- surfaces both
    the named sibling's own name and the implicit "default"."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="a"), Step(run="b", name="gui")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        pytest.raises(UnknownVariantError) as exc_info,
    ):
        mgr.start(record.id, variant="nope")

    assert exc_info.value.available == ["gui", "default"]


def test_manager_start_unknown_variant_explicit_default_name_not_duplicated(
    tmp_path: Path,
):
    """Ticket #131: a step explicitly named "default" must appear in
    `available` exactly once, even though it is both the tier-1 exact-match
    name AND independently reachable via the tier-3 single-step-total
    fallback."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        start=[Step(run="python server.py", name="default")],
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        pytest.raises(UnknownVariantError) as exc_info,
    ):
        mgr.start(record.id, variant="nope")

    assert exc_info.value.available == ["default"]
    assert str(exc_info.value).count("default") == 1  # once in available=[...], no dup


def test_manager_start_unknown_worktree_raises_not_found(tmp_path: Path):
    """start() raises WorktreeNotFoundError when the worktree id is not in the store."""
    mgr = _make_mgr_in_memory(tmp_path)

    with pytest.raises(WorktreeNotFoundError):
        mgr.start("nonexistent-id-12345678")


def test_manager_start_cmd_argument_removed(tmp_path: Path):
    """start() no longer accepts a positional cmd argument; passing one raises TypeError."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    with pytest.raises(TypeError):
        mgr.start(record.id, ["python", "x"])  # type: ignore[call-arg]


def test_manager_stop_unknown_worktree_raises_not_found(tmp_path: Path):
    """stop() raises WorktreeNotFoundError when the worktree id is not in the store."""
    mgr = _make_mgr_in_memory(tmp_path)

    with pytest.raises(WorktreeNotFoundError):
        mgr.stop("nonexistent-id-12345678")


def test_manager_stop_runs_stop_steps_before_sigterm(tmp_path: Path):
    """stop() runs contract.stop steps via SetupRunner before calling _lifecycle_stop."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(pids={"main": 12345})
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        stop=[Step(run="docker compose stop")],
    )

    call_order: list = []

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.side_effect = lambda **kw: call_order.append("runner.run")

    mock_lifecycle_stop = MagicMock()
    mock_lifecycle_stop.return_value = record
    mock_lifecycle_stop.side_effect = lambda *a, **kw: (
        call_order.append("_lifecycle_stop"),
        record,
    )[1]

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner_instance),
        patch("lib_python_worktree.core.manager._lifecycle_stop", mock_lifecycle_stop),
    ):
        mgr.stop(record.id)

    assert call_order == ["runner.run", "_lifecycle_stop"], (
        f"Expected runner.run before _lifecycle_stop, got: {call_order}"
    )


def test_manager_stop_without_stop_steps_skips_setup_runner(tmp_path: Path):
    """stop() does not invoke SetupRunner when contract.stop is empty."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(pids={"main": 12345})
    mgr.state.add(record)

    fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

    mock_runner_instance = MagicMock()

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner_instance),
        patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
    ):
        mock_lc_stop.return_value = record
        mgr.stop(record.id)

    mock_runner_instance.run.assert_not_called()
    mock_lc_stop.assert_called_once()


def test_manager_stop_steps_failure_does_not_block_sigterm(tmp_path: Path):
    """stop() calls _lifecycle_stop even when SetupRunner.run raises."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(pids={"main": 12345})
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        stop=[Step(run="will fail")],
    )

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.side_effect = RuntimeError("step exploded")

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner_instance),
        patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
    ):
        mock_lc_stop.return_value = record
        mgr.stop(record.id)  # must not raise

    mock_lc_stop.assert_called_once_with(
        record.id,
        store=mgr.state,
        role="main",
        timeout=10.0,
        kill_orphans=False,
    )


def test_manager_stop_without_pid_is_noop(tmp_path: Path):
    """stop() on a worktree with no recorded PID is a graceful no-op (ticket #41).

    Symmetric with the no-op "ready" start: there is nothing to signal, so
    _lifecycle_stop (which would raise ProcessNotRunningError) is not called,
    no error is raised, and the worktree is marked "stopped".
    """
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(status="ready")  # no pids
    mgr.state.add(record)

    fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
    ):
        result = mgr.stop(record.id)  # must not raise

    mock_lc_stop.assert_not_called()
    assert result.status == "stopped"
    assert mgr.state.get(record.id).status == "stopped"


def test_manager_stop_without_pid_sets_stop_attempt_no_process_recorded(
    tmp_path: Path,
):
    """Ticket #110: the no-op branch (no pid recorded for the resolved
    role) must set stop_attempt=StopAttempt(outcome="no_process_recorded")
    -- process_lifecycle.stop() (and therefore every other StopAttempt
    outcome) is never even reached for this role."""
    from lib_python_worktree.core.state import STOP_ATTEMPT_NO_PROCESS_RECORDED

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(status="ready")  # no pids
    mgr.state.add(record)

    fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
    ):
        result = mgr.stop(record.id)

    mock_lc_stop.assert_not_called()
    assert result.stop_attempt is not None
    assert result.stop_attempt.outcome == STOP_ATTEMPT_NO_PROCESS_RECORDED
    assert result.stop_attempt.role == "main"
    assert result.stop_attempt.tracked_pid is None
    stored = mgr.state.get(record.id)
    assert stored.stop_attempt is not None
    assert stored.stop_attempt.outcome == STOP_ATTEMPT_NO_PROCESS_RECORDED


def test_manager_stop_without_pid_clears_stale_killed_pids(tmp_path: Path):
    """Driving test (review round 2, blocking finding 2): the no-op branch
    (no pid recorded for the resolved role) must also refresh
    record.killed_pids to [] -- mirroring process_lifecycle.stop(), which
    unconditionally recomputes killed_pids on every path, including to an
    empty list. Before this fix, a record left with stale killed_pids from
    an earlier real stop()/start() call kept reporting those stale pids
    even though this call reports "no process recorded; nothing to stop"."""
    from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

    mgr = _make_mgr_in_memory(tmp_path)
    stale_killed = [
        KilledProcessInfo(pid=4242, name="stale.exe", source="tracked"),
    ]
    record = _make_wt_record(status="ready", killed_pids=stale_killed)  # no pids
    mgr.state.add(record)

    fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
    ):
        result = mgr.stop(record.id)

    mock_lc_stop.assert_not_called()
    assert result.killed_pids == []
    stored = mgr.state.get(record.id)
    assert stored.killed_pids == []


# ---------------------------------------------------------------------------
# Ticket #128: stop_hook_outcome -- contract/isolation diagnostics distinguish
# an "isolation: none, nothing to stop by design" no-op from an ordinary
# "role never started" no-op, both of which set the same
# stop_attempt.outcome == STOP_ATTEMPT_NO_PROCESS_RECORDED.
# ---------------------------------------------------------------------------


def _write_contract(repo_root: Path, text: str) -> Path:
    """Write a real on-disk `.seretos/worktree-setup.yml` under *repo_root*."""
    contract_dir = repo_root / ".seretos"
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract_dir / "worktree-setup.yml"
    contract_path.write_text(text, encoding="utf-8")
    return contract_path


def test_manager_stop_isolation_none_no_op_is_distinguishable(tmp_path: Path):
    """Ticket #128: a real on-disk `isolation: none` contract must set
    stop_hook_outcome.no_op_reason == "isolation_none" on the no-op path --
    distinct from an ordinary never-started role under a full-isolation
    contract, which sets "no_process_recorded". _load_contract is
    deliberately NOT patched here (unlike the older #110 no-op tests) so
    that contract_path.exists() and the parsed contract can never diverge."""
    from lib_python_worktree.core.state import STOP_NO_OP_ISOLATION_NONE, STOP_NO_OP_NO_PROCESS_RECORDED

    mgr = _make_mgr_in_memory(tmp_path)

    none_repo = tmp_path / "none-repo"
    none_repo.mkdir()
    _write_contract(none_repo, "version: 1\nisolation: none\n")
    none_record = _make_wt_record(
        wt_id="wt-isolation-none", repo_root=str(none_repo), status="ready"
    )
    mgr.state.add(none_record)

    none_result = mgr.stop(none_record.id)

    assert none_result.stop_hook_outcome is not None
    assert none_result.stop_hook_outcome.no_op_reason == STOP_NO_OP_ISOLATION_NONE

    full_repo = tmp_path / "full-repo"
    full_repo.mkdir()
    _write_contract(full_repo, "version: 1\nisolation: full\nstop: []\n")
    full_record = _make_wt_record(
        wt_id="wt-isolation-full", repo_root=str(full_repo), status="ready"
    )
    mgr.state.add(full_record)

    full_result = mgr.stop(full_record.id)

    assert full_result.stop_hook_outcome is not None
    assert full_result.stop_hook_outcome.no_op_reason == STOP_NO_OP_NO_PROCESS_RECORDED


def test_manager_stop_reports_contract_diagnostics(tmp_path: Path):
    """Ticket #128: stop_hook_outcome carries contract_found/contract_path/
    contract_isolation from a real on-disk isolation: full contract with a
    stop: step. SetupRunner is patched so the (fake) step never actually
    spawns a subprocess."""
    repo_root = tmp_path / "diag-repo"
    repo_root.mkdir()
    contract_path = _write_contract(
        repo_root,
        "version: 1\nisolation: full\nstop:\n  - run: echo bye\n",
    )

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(
        wt_id="wt-diag", repo_root=str(repo_root), status="ready"
    )
    mgr.state.add(record)

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = SetupResult(worktree_id=record.id)

    with patch(
        "lib_python_worktree.setup.runner.SetupRunner",
        return_value=mock_runner_instance,
    ):
        result = mgr.stop(record.id)

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.contract_found is True
    assert result.stop_hook_outcome.contract_path == contract_path.as_posix()
    assert result.stop_hook_outcome.contract_isolation == "full"


def test_manager_stop_reports_stop_steps_run(tmp_path: Path):
    """Ticket #128: stop_hook_outcome.steps_run reflects the number of
    stop: steps SetupRunner.run() actually ran, with status "completed"."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(status="ready")
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        stop=[Step(run="one"), Step(run="two")],
    )

    fake_result = SetupResult(
        worktree_id=record.id,
        steps=[
            SetupStepResult(index=0, name="one", returncode=0, log_path=tmp_path / "a.log"),
            SetupStepResult(index=1, name="two", returncode=0, log_path=tmp_path / "b.log"),
        ],
    )

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = fake_result

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner_instance),
    ):
        result = mgr.stop(record.id)

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.steps_run == 2
    assert result.stop_hook_outcome.status == "completed"


def test_manager_stop_reports_skipped_when_no_stop_steps(tmp_path: Path):
    """Sibling of the steps_run test above: an empty stop: list must report
    status="skipped" and steps_run == 0."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(status="ready")
    mgr.state.add(record)

    fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

    with patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract):
        result = mgr.stop(record.id)

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.steps_run == 0
    assert result.stop_hook_outcome.status == "skipped"


def test_manager_stop_reports_failed_when_setup_runner_raises(tmp_path: Path):
    """Sibling of the steps_run test above: SetupRunner.run() raising
    SetupFailedError must report status="failed", steps_run == 0, and
    message == str(exc) -- and stop() must still complete normally (the
    no-op branch must still be reached, exactly as before this ticket)."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(status="ready")  # no pids -> no-op branch
    mgr.state.add(record)

    fake_contract = WorktreeContract(
        version=1, isolation="full", stop=[Step(run="will fail")]
    )
    exc = SetupFailedError(
        worktree_id=record.id,
        step_index=0,
        step_name="will fail",
        log_path=tmp_path / "fail.log",
        returncode=1,
    )

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.side_effect = exc

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner_instance),
    ):
        result = mgr.stop(record.id)  # must not raise

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.status == "failed"
    assert result.stop_hook_outcome.steps_run == 0
    assert result.stop_hook_outcome.message == str(exc)
    # The no-op branch was still reached -- stop_attempt was still set.
    assert result.stop_attempt is not None
    assert result.stop_attempt.outcome == "no_process_recorded"


def test_manager_stop_with_pid_still_reports_contract_diagnostics(tmp_path: Path):
    """Ticket #128: on the delegated (has-a-pid) path, stop_hook_outcome
    must be assigned onto the object _lifecycle_stop returns -- NOT onto
    the pre-call `record` reference, which may be stale for
    YamlStateStore. Simulated here by having the patched _lifecycle_stop
    return a different, freshly-constructed WorktreeRecord object."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(pids={"main": 12345})
    mgr.state.add(record)

    fresh_record = _make_wt_record(pids={"main": 12345})
    assert fresh_record is not record

    fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.core.manager._lifecycle_stop", return_value=fresh_record) as mock_lc_stop,
    ):
        result = mgr.stop(record.id)

    mock_lc_stop.assert_called_once()
    assert result is fresh_record
    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.no_op_reason is None


def test_manager_stop_no_contract_file_reports_implicit_isolation_none(tmp_path: Path):
    """No `.seretos/worktree-setup.yml` at all -> contract_found is False,
    but the loader's implicit isolation: none contract still makes this
    no-op distinguishable as "isolation_none"."""
    from lib_python_worktree.core.state import STOP_NO_OP_ISOLATION_NONE

    repo_root = tmp_path / "no-contract-repo"
    repo_root.mkdir()

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(
        wt_id="wt-no-contract", repo_root=str(repo_root), status="ready"
    )
    mgr.state.add(record)

    result = mgr.stop(record.id)

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.contract_found is False
    assert result.stop_hook_outcome.no_op_reason == STOP_NO_OP_ISOLATION_NONE


def test_manager_stop_unparseable_contract_reports_failed_no_isolation(tmp_path: Path):
    """A `.seretos/worktree-setup.yml` that exists but fails to parse must
    report contract_found is True, contract_isolation is None,
    status="failed", and stop() must not raise."""
    repo_root = tmp_path / "bad-contract-repo"
    repo_root.mkdir()
    _write_contract(repo_root, "{not: valid: yaml: [")

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(
        wt_id="wt-bad-contract", repo_root=str(repo_root), status="ready"
    )
    mgr.state.add(record)

    result = mgr.stop(record.id)  # must not raise

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.contract_found is True
    assert result.stop_hook_outcome.contract_isolation is None
    assert result.stop_hook_outcome.status == "failed"


def test_manager_stop_nonexistent_repo_root_reports_contract_not_found(tmp_path: Path):
    """record.repo_root pointing at a directory that doesn't exist on disk
    at all -> contract_found is False, and stop() must not raise."""
    missing_repo_root = tmp_path / "does-not-exist-at-all"

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(
        wt_id="wt-missing-repo-root", repo_root=str(missing_repo_root), status="ready"
    )
    mgr.state.add(record)

    result = mgr.stop(record.id)  # must not raise

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.contract_found is False


# ---------------------------------------------------------------------------
# TestTicket130ForceRemoveSurfacesStopHook -- ticket #130
#
# WorktreeManager.remove() (via _teardown()) must surface the outcome of
# the contract's stop: hook onto the returned record's stop_hook_outcome --
# previously always None on the teardown path (discarded by a bare
# `except Exception: pass`), including on a force=True removal of a still-
# running environment, which is the ticket's headline regression.
# ---------------------------------------------------------------------------


def _write_teardown_contract(repo_root: Path, text: str) -> Path:
    """Write a real on-disk `.seretos/worktree-setup.yml` under *repo_root*."""
    contract_dir = repo_root / ".seretos"
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract_dir / "worktree-setup.yml"
    contract_path.write_text(text, encoding="utf-8")
    return contract_path


@pytest.mark.requires_git
def test_remove_force_true_surfaces_completed_stop_hook_outcome(
    manager: WorktreeManager, git_repo: Path
):
    """Driving test (ticket #130): remove(force=True) on an environment with
    a stop: contract must return a record whose stop_hook_outcome reports
    the hook's own completion -- previously this was always None on the
    force-remove path."""
    contract_path = _write_teardown_contract(
        git_repo, "version: 1\nisolation: full\nstop:\n  - run: echo bye\n"
    )
    rec = manager.create(str(git_repo), "feature/alpha")

    result = manager.remove(rec.id, force=True)

    assert result.status == "removed"
    outcome = result.stop_hook_outcome
    assert outcome is not None
    assert outcome.status == "completed"
    assert outcome.steps_run == 1
    assert outcome.message == "stop: completed 1 step(s)"
    assert outcome.contract_found is True
    assert outcome.contract_path == contract_path.as_posix()
    assert outcome.contract_isolation == "full"
    assert outcome.no_op_reason is None
    # Ticket #130 scope decision: stop_attempt is intentionally NOT
    # recomputed by _teardown() -- StopAttempt is single-valued but Step 1
    # stops every role in a loop, so there is no non-arbitrary single
    # attempt to report. This is a documented scope boundary, not a gap.
    assert result.stop_attempt is None


@pytest.mark.requires_git
def test_remove_force_false_also_surfaces_stop_hook_outcome(
    manager: WorktreeManager, git_repo: Path
):
    """Sibling of the driving test: force=False must populate
    stop_hook_outcome exactly the same way -- Step 1b runs on both
    force=True and force=False."""
    _write_teardown_contract(
        git_repo, "version: 1\nisolation: full\nstop:\n  - run: echo bye\n"
    )
    rec = manager.create(str(git_repo), "feature/alpha")

    result = manager.remove(rec.id, force=False)

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.status == "completed"


@pytest.mark.requires_git
def test_remove_untracked_target_also_surfaces_stop_hook_outcome(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """The untracked (ticket #88) removal path -- addressed only via
    checkout_path, synthesised record never written to the state store --
    must also carry stop_hook_outcome on the returned record."""
    _write_teardown_contract(
        git_repo, "version: 1\nisolation: full\nstop:\n  - run: echo bye\n"
    )

    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    result = mgr.remove(checkout_path=str(linked_worktree), force=True)

    assert result.status == "removed"
    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.status == "completed"


@pytest.mark.requires_git
def test_remove_empty_stop_contract_reports_skipped(
    manager: WorktreeManager, git_repo: Path
):
    """A contract with isolation: full but no stop: steps -> status
    "skipped", steps_run == 0."""
    _write_teardown_contract(git_repo, "version: 1\nisolation: full\n")
    rec = manager.create(str(git_repo), "feature/alpha")

    result = manager.remove(rec.id, force=True)

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.status == "skipped"
    assert result.stop_hook_outcome.steps_run == 0


@pytest.mark.requires_git
def test_remove_no_contract_file_reports_not_found_and_implicit_isolation_none(
    manager: WorktreeManager, git_repo: Path
):
    """No `.seretos/worktree-setup.yml` at all -> contract_found is False,
    but the loader's implicit isolation: none contract still sets
    contract_isolation == "none"."""
    rec = manager.create(str(git_repo), "feature/alpha")

    result = manager.remove(rec.id, force=True)

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.contract_found is False
    assert result.stop_hook_outcome.contract_isolation == "none"


@pytest.mark.requires_git
def test_remove_unparseable_contract_reports_failed(
    manager: WorktreeManager, git_repo: Path
):
    """A `.seretos/worktree-setup.yml` that exists but fails to parse ->
    contract_found is True, contract_isolation is None, status="failed",
    and remove() must not raise from the hook failure itself.

    The broken contract is written AFTER create() (which also loads the
    contract for its own setup: validation and would fail outright on
    unparseable YAML) so only remove()'s own teardown-time load sees it."""
    rec = manager.create(str(git_repo), "feature/alpha")
    _write_teardown_contract(git_repo, "{not: valid: yaml: [")

    result = manager.remove(rec.id, force=True)  # must not raise

    assert result.stop_hook_outcome is not None
    assert result.stop_hook_outcome.contract_found is True
    assert result.stop_hook_outcome.contract_isolation is None
    assert result.stop_hook_outcome.status == "failed"



class TestManagerStopStickyStatus:
    """R6 (ticket #95): the no-pid-for-role no-op path in ``WorktreeManager.stop``
    must not clobber a sticky honest status.

    Root cause (finding 6): ``manager.py``'s no-op branch set
    ``record.status = "stopped"`` unconditionally whenever ``record.pids`` was
    empty, unlike ``process_lifecycle.stop()``'s own guard (which excludes
    ``"stop_incomplete"``/``"orphaned"``). A worktree already marked
    ``"stop_incomplete"`` by an earlier ``stop()`` call for a different role
    (or ``"orphaned"`` by ``reconcile()``) could have that honest status
    silently overwritten back to ``"stopped"`` just by calling ``stop()``
    again on a role with no recorded PID.
    """

    def test_no_pid_for_role_does_not_clobber_stop_incomplete(self, tmp_path: Path):
        """Driving test: a record already marked "stop_incomplete" with no
        recorded pids must keep that status through a no-op stop() call."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(status="stop_incomplete")  # no pids
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            result = mgr.stop(record.id)  # must not raise

        mock_lc_stop.assert_not_called()
        assert result.status == "stop_incomplete"
        assert mgr.state.get(record.id).status == "stop_incomplete"

    def test_no_pid_for_role_does_not_clobber_orphaned(self, tmp_path: Path):
        """Same guard for the "orphaned" status set by reconcile()."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(status="orphaned")  # no pids
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            result = mgr.stop(record.id)

        mock_lc_stop.assert_not_called()
        assert result.status == "orphaned"
        assert mgr.state.get(record.id).status == "orphaned"

    def test_no_pid_for_role_running_still_becomes_stopped(self, tmp_path: Path):
        """Non-sticky statuses (e.g. "running") are still normalized to
        "stopped" by the no-op path -- the guard only protects the sticky
        statuses, it must not turn the no-op branch into a total no-op."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(status="running")  # no pids
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            result = mgr.stop(record.id)

        mock_lc_stop.assert_not_called()
        assert result.status == "stopped"
        assert mgr.state.get(record.id).status == "stopped"

    def test_no_pid_for_role_still_runs_contract_stop_steps(self, tmp_path: Path):
        """The sticky-status guard must not skip running contract stop: steps
        -- only the status assignment changes."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(status="stop_incomplete")  # no pids
        mgr.state.add(record)

        fake_contract = WorktreeContract(
            version=1, isolation="full", stop=[Step(run="echo done")],
        )
        mock_runner_instance = MagicMock()

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            result = mgr.stop(record.id)

        mock_runner_instance.run.assert_called_once()
        mock_lc_stop.assert_not_called()
        assert result.status == "stop_incomplete"

    def test_no_pid_for_role_clears_stop_detail_when_marking_stopped(self, tmp_path: Path):
        """Ticket #99 (B2 edge case): mirrors
        test_no_pid_for_role_running_still_becomes_stopped -- when the no-op
        branch actually transitions a non-sticky status to "stopped", any
        stale stop_detail left over on the record must be cleared too."""
        mgr = _make_mgr_in_memory(tmp_path)
        stale_detail = StopDetail(
            reason=STOP_REASON_SURVIVORS, message="stale", role="main",
        )
        record = _make_wt_record(status="running", stop_detail=stale_detail)  # no pids
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            result = mgr.stop(record.id)

        mock_lc_stop.assert_not_called()
        assert result.status == "stopped"
        assert result.stop_detail is None
        assert mgr.state.get(record.id).stop_detail is None

    def test_no_pid_for_role_preserves_stop_detail_when_sticky(self, tmp_path: Path):
        """Ticket #99 (B2 edge case): mirrors
        test_no_pid_for_role_does_not_clobber_stop_incomplete -- the sticky
        guard preserving status must also preserve the stop_detail attached
        to it; the no-op branch must not touch stop_detail when it does not
        touch status."""
        mgr = _make_mgr_in_memory(tmp_path)
        sticky_detail = StopDetail(
            reason=STOP_REASON_SURVIVORS, message="sticky", role="main",
        )
        record = _make_wt_record(
            status="stop_incomplete", stop_detail=sticky_detail,
        )  # no pids
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            result = mgr.stop(record.id)

        mock_lc_stop.assert_not_called()
        assert result.status == "stop_incomplete"
        assert result.stop_detail == sticky_detail
        assert mgr.state.get(record.id).stop_detail == sticky_detail


# ---------------------------------------------------------------------------
# TestManagerStopByVariant -- ticket #104 (B3)
# ---------------------------------------------------------------------------

class TestManagerStopByVariant:
    """B3 (ticket #104): stop(variant=...) resolves to the role that
    started it via record.variants, symmetric with start(variant=...)."""

    def test_manager_stop_by_variant_resolves_role(self, tmp_path: Path):
        """Driving test: stop(variant="web") delegates to _lifecycle_stop
        with role="web" resolved from record.variants."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(
            pids={"main": 111, "web": 222},
            variants={"main": "default", "web": "web"},
        )
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            mock_lc_stop.return_value = record
            mgr.stop(record.id, variant="web")

        mock_lc_stop.assert_called_once_with(
            record.id,
            store=mgr.state,
            role="web",
            timeout=10.0,
            kill_orphans=False,
        )

    def test_manager_stop_without_variant_unchanged(self, tmp_path: Path):
        """Bare stop(id) still targets 'main' when variant is not given."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(
            pids={"main": 111, "web": 222},
            variants={"main": "default", "web": "web"},
        )
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            mock_lc_stop.return_value = record
            mgr.stop(record.id)

        mock_lc_stop.assert_called_once_with(
            record.id,
            store=mgr.state,
            role="main",
            timeout=10.0,
            kill_orphans=False,
        )

    def test_manager_stop_role_none_defaults_to_main(self, tmp_path: Path):
        """Explicit role=None is the same as omitting it entirely -- the
        sentinel default is not itself a no-op."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(pids={"main": 111})
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            mock_lc_stop.return_value = record
            mgr.stop(record.id, role=None)

        mock_lc_stop.assert_called_once_with(
            record.id,
            store=mgr.state,
            role="main",
            timeout=10.0,
            kill_orphans=False,
        )

    def test_manager_stop_role_and_variant_agree(self, tmp_path: Path):
        """role="web", variant="web" where variants={"web": "web"} resolves
        and delegates with role="web" (the pair agrees; no error)."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(pids={"web": 222}, variants={"web": "web"})
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            mock_lc_stop.return_value = record
            mgr.stop(record.id, role="web", variant="web")

        mock_lc_stop.assert_called_once_with(
            record.id,
            store=mgr.state,
            role="web",
            timeout=10.0,
            kill_orphans=False,
        )

    def test_manager_stop_by_variant_after_reload(self, tmp_path: Path):
        """End-to-end through YamlStateStore: a record with a persisted
        variants mapping survives a fresh WorktreeManager instance
        (simulating a host restart), and stop(variant=...) still resolves
        correctly."""
        state_dir = tmp_path / "state"
        store1 = YamlStateStore(state_dir=state_dir)
        record = _make_wt_record(pids={"web": 222}, variants={"web": "web"})
        store1.add(record)

        mgr2 = WorktreeManager(
            config=ManagerConfig(store_root=tmp_path / "store"),
            state=YamlStateStore(state_dir=state_dir),
            reconcile_on_init=False,
        )

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])
        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            mock_lc_stop.return_value = record
            mgr2.stop(record.id, variant="web")

        mock_lc_stop.assert_called_once_with(
            record.id,
            store=mgr2.state,
            role="web",
            timeout=10.0,
            kill_orphans=False,
        )

    def test_manager_stop_by_variant_after_default_resolved_named_step(self, tmp_path: Path):
        """Ticket #112: a record left the way start() now leaves it after
        resolving a lone named step via the default fallback (pids and
        variants keyed by the step's own name) is stoppable by that same
        name via stop(variant=...) without raising."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(
            pids={"main": 111},
            variants={"main": "main"},
        )
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            mock_lc_stop.return_value = record
            mgr.stop(record.id, variant="main")

        mock_lc_stop.assert_called_once_with(
            record.id,
            store=mgr.state,
            role="main",
            timeout=10.0,
            kill_orphans=False,
        )

    def test_manager_stop_by_default_variant_fails_after_named_step_fallback(
        self, tmp_path: Path
    ):
        """Ticket #112 (blocking review finding): a record left the way
        start() leaves it after resolving a lone named step via the default
        fallback (variants keyed by the step's own name, e.g. "main", not
        the literal "default") is NOT stoppable via
        stop(variant="default") -- the exact literal the caller originally
        passed to start(). This is a deliberate, documented asymmetry (see
        the start() docstring and README's role-vs-variant section): only
        stop(variant="main") resolves it."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(
            pids={"main": 111},
            variants={"main": "main"},
        )
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            with pytest.raises(VariantResolutionError) as exc_info:
                mgr.stop(record.id, variant="default")

        mock_lc_stop.assert_not_called()
        err = exc_info.value
        assert err.roles == []


# ---------------------------------------------------------------------------
# TestManagerStopVariantResolutionErrors -- ticket #104 (B4)
# ---------------------------------------------------------------------------

class TestManagerStopVariantResolutionErrors:
    """B4 (ticket #104): an unknown, ambiguous, or role-disagreeing variant
    raises a clear VariantResolutionError, and nothing is attempted (no
    contract stop: steps, no _lifecycle_stop call) before that error is
    raised."""

    def test_manager_stop_ambiguous_variant_raises(self, tmp_path: Path):
        """Driving test: two roles both mapped to variant="web" -- stop()
        raises VariantResolutionError naming both candidates, and nothing
        was attempted (no _lifecycle_stop call, no contract stop: steps)."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(
            pids={"web1": 111, "web2": 222},
            variants={"web1": "web", "web2": "web"},
        )
        mgr.state.add(record)

        fake_contract = WorktreeContract(
            version=1, isolation="full", stop=[Step(run="echo stopping")],
        )
        mock_runner_instance = MagicMock()

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            with pytest.raises(VariantResolutionError) as exc_info:
                mgr.stop(record.id, variant="web")

        mock_lc_stop.assert_not_called()
        mock_runner_instance.run.assert_not_called()
        err = exc_info.value
        assert err.roles == ["web1", "web2"]
        assert err.requested_role is None

    def test_manager_stop_unknown_variant_raises(self, tmp_path: Path):
        """Zero match: no currently-running role was started with this
        variant. The message points at role= and names running roles."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(pids={"main": 111}, variants={"main": "default"})
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            with pytest.raises(VariantResolutionError) as exc_info:
                mgr.stop(record.id, variant="nope")

        mock_lc_stop.assert_not_called()
        err = exc_info.value
        assert err.roles == []
        message = str(err)
        assert "role=" in message
        assert "main" in message

    def test_manager_stop_legacy_record_zero_match_names_running_roles(self, tmp_path: Path):
        """A record with a live pid but no variants entry (started before
        this ticket shipped) hits the zero-match branch and names the
        running role, while stop() and stop(role="main") still work
        unchanged."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(pids={"main": 111})  # variants == {}
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            with pytest.raises(VariantResolutionError) as exc_info:
                mgr.stop(record.id, variant="default")

        mock_lc_stop.assert_not_called()
        assert "main" in str(exc_info.value)

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop2,
        ):
            mock_lc_stop2.return_value = record
            mgr.stop(record.id)

        mock_lc_stop2.assert_called_once_with(
            record.id,
            store=mgr.state,
            role="main",
            timeout=10.0,
            kill_orphans=False,
        )

    def test_manager_stop_role_conflicts_with_variant_raises(self, tmp_path: Path):
        """An explicit role= that disagrees with the role variant= resolves
        to raises rather than silently picking one."""
        mgr = _make_mgr_in_memory(tmp_path)
        record = _make_wt_record(
            pids={"main": 111, "web": 222},
            variants={"web": "web"},
        )
        mgr.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        with (
            patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
            patch("lib_python_worktree.core.manager._lifecycle_stop") as mock_lc_stop,
        ):
            with pytest.raises(VariantResolutionError) as exc_info:
                mgr.stop(record.id, role="main", variant="web")

        mock_lc_stop.assert_not_called()
        err = exc_info.value
        assert err.requested_role == "main"
        assert err.roles == ["web"]

    def test_variant_resolution_error_is_worktree_error_and_valueerror(self):
        assert issubclass(VariantResolutionError, WorktreeError)
        assert issubclass(VariantResolutionError, ValueError)

    def test_variant_resolution_error_importable_and_exported(self):
        import lib_python_worktree as top

        assert top.VariantResolutionError is VariantResolutionError
        assert "VariantResolutionError" in top.__all__
        assert "VariantResolutionError" in manager_module.__all__


# ---------------------------------------------------------------------------
# Ticket #27: Consistent error contract — InvalidRepoError / InvalidBranchError
# ---------------------------------------------------------------------------

from lib_python_worktree.core.manager import InvalidBranchError, InvalidRepoError  # noqa: E402


def test_invalid_repo_error_is_worktree_error_subclass():
    """InvalidRepoError must be a WorktreeError subclass so existing
    ``except WorktreeError`` catch-alls continue to work."""
    assert issubclass(InvalidRepoError, WorktreeError), (
        "InvalidRepoError must be a subclass of WorktreeError"
    )


def test_invalid_branch_error_is_worktree_error_subclass():
    """InvalidBranchError must be a WorktreeError subclass."""
    assert issubclass(InvalidBranchError, WorktreeError), (
        "InvalidBranchError must be a subclass of WorktreeError"
    )


def test_create_invalid_repo_root_empty_string(manager: WorktreeManager):
    """create() with an empty repo_root must raise InvalidRepoError, not
    the bare WorktreeError base class."""
    with pytest.raises(InvalidRepoError):
        manager.create("", "feature/x")


def test_create_invalid_repo_root_nonexistent(manager: WorktreeManager):
    """create() with a non-existent path must raise InvalidRepoError."""
    with pytest.raises(InvalidRepoError):
        manager.create("/no/such/path/xyzzy_does_not_exist", "feature/x")


@pytest.mark.requires_git
def test_create_invalid_repo_root_not_a_git_repo(
    manager: WorktreeManager, tmp_path: Path
):
    """create() given a real directory that is NOT a git repo must raise
    InvalidRepoError (git rev-parse non-zero path)."""
    plain_dir = tmp_path / "not_a_git_repo"
    plain_dir.mkdir()
    with pytest.raises(InvalidRepoError):
        manager.create(str(plain_dir), "feature/x")


@pytest.mark.requires_git
def test_create_invalid_branch_empty_string(
    manager: WorktreeManager, git_repo: Path
):
    """create() with an empty branch string must raise InvalidBranchError,
    not the bare WorktreeError base class."""
    with pytest.raises(InvalidBranchError):
        manager.create(str(git_repo), "")


@pytest.mark.requires_git
def test_create_invalid_branch_whitespace_only(
    manager: WorktreeManager, git_repo: Path
):
    """create() with a whitespace-only branch string must raise
    InvalidBranchError (branch.strip() is empty)."""
    with pytest.raises(InvalidBranchError):
        manager.create(str(git_repo), "   ")


def test_invalid_repo_error_attributes():
    """InvalidRepoError exposes repo_root and reason as attributes."""
    err = InvalidRepoError("/some/path", "path does not exist")
    assert err.repo_root == "/some/path"
    assert err.reason == "path does not exist"
    assert "/some/path" in str(err)
    assert "path does not exist" in str(err)


def test_invalid_repo_error_public_import():
    """InvalidRepoError must be importable from the public package surface."""
    from lib_python_worktree import InvalidRepoError as PublicInvalidRepoError  # noqa: PLC0415
    assert PublicInvalidRepoError is InvalidRepoError


def test_invalid_branch_error_public_import():
    """InvalidBranchError must be importable from the public package surface."""
    from lib_python_worktree import InvalidBranchError as PublicInvalidBranchError  # noqa: PLC0415
    assert PublicInvalidBranchError is InvalidBranchError


# ---------------------------------------------------------------------------
# Ticket #39: plugin registry seeding
# ---------------------------------------------------------------------------

import json  # noqa: E402


@pytest.mark.requires_git
def test_create_clones_plugin_registry_entry_without_cli(
    manager_factory: Callable[..., WorktreeManager],
    git_repo: Path,
    tmp_path: Path,
):
    """create() registers enabledPlugins via registry clone even when the
    claude CLI is unavailable (ticket #64).

    ``seed_plugin_registry`` is no longer wired from ``manager.py`` at all --
    ``install_enabled_plugins`` is now self-sufficient. A fake registry is
    set up with a *structurally valid* entry (real on-disk
    ``.claude-plugin/plugin.json``) for an unrelated project path; after
    create() the registry must contain a clone of that entry with
    projectPath set to the new worktree's native-OS path, and installPath /
    version preserved from the original -- all without the claude CLI ever
    being resolvable.
    """
    # Build a fake config_dir with a structurally-valid plugin registry entry.
    config_dir = tmp_path / "fake_claude"
    plugins_dir = config_dir / "plugins"
    plugins_dir.mkdir(parents=True)

    # A real on-disk install so `_is_structurally_valid` accepts it as a
    # clone source.
    install_dir = tmp_path / "cache" / "my-plugin"
    manifest_dir = install_dir / ".claude-plugin"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "plugin.json").write_text("{}", encoding="utf-8")
    install_path = str(install_dir)

    registry_path = plugins_dir / "installed_plugins.json"
    original_entry = {
        "scope": "project",
        "projectPath": "/some/unrelated/project",
        "installPath": install_path,
        "version": "3.1.4",
    }
    # Write Schema v2 registry (the real format used by Claude Code).
    registry_path.write_text(
        json.dumps({"version": 2, "plugins": {"my-plugin@marketplace": [original_entry]}}),
        encoding="utf-8",
    )

    # Declare an enabledPlugins entry so install_enabled_plugins() has
    # something to act on.
    claude_dir = git_repo / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.json").write_text(
        json.dumps({"enabledPlugins": {"my-plugin@marketplace": True}}),
        encoding="utf-8",
    )

    mgr = manager_factory()
    mgr._plugin_install_config_dir = config_dir
    # Force the claude CLI to appear unavailable -- clone-first must still work.
    mgr._plugin_install_which = lambda *_a, **_kw: None

    rec = mgr.create(str(git_repo), "feature/alpha")

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    assert data["version"] == 2
    plugin_list = data["plugins"]["my-plugin@marketplace"]

    # The original must still be there.
    src = [e for e in plugin_list if e.get("projectPath") == "/some/unrelated/project"]
    assert len(src) == 1

    # A clone for the worktree must have been added.
    expected_dest = str(Path(rec.path))
    cloned = [e for e in plugin_list if e.get("projectPath") == expected_dest]
    assert len(cloned) == 1, (
        f"Expected exactly one cloned registry entry with projectPath={expected_dest!r}, "
        f"got {cloned!r}"
    )
    assert cloned[0]["installPath"] == install_path
    assert cloned[0]["version"] == "3.1.4"
    assert cloned[0]["scope"] == "project"


# ---------------------------------------------------------------------------
# Ticket #62: install_enabled_plugins() as the primary mechanism, with
# seed_plugin_registry() as fallback
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_create_uses_cli_install_and_skips_seed_fallback_when_available(
    manager_factory: Callable[..., WorktreeManager],
    git_repo: Path,
    tmp_path: Path,
    monkeypatch,
):
    """When the claude CLI is 'available' (fake which resolves), create()
    installs enabledPlugins via the CLI and never falls back to
    seed_plugin_registry()."""
    monkeypatch.setenv("WORKTREE_LOG_ROOT", str(tmp_path / "logs"))

    claude_dir = git_repo / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.json").write_text(
        json.dumps({"enabledPlugins": {"my-plugin@marketplace": True}}),
        encoding="utf-8",
    )

    import types  # noqa: PLC0415

    calls: list = []

    def _fake_which(name):  # noqa: ANN001
        return "C:/fake/claude.exe" if name == "claude" else None

    def _fake_runner(cmd, *, cwd, timeout):  # noqa: ANN001
        calls.append({"cmd": cmd, "cwd": cwd, "timeout": timeout})
        return types.SimpleNamespace(returncode=0, stdout="ok", stderr="")

    import lib_python_worktree.core.plugin_seed as _ps_module  # noqa: PLC0415
    seed_calls: list = []
    monkeypatch.setattr(
        _ps_module, "seed_plugin_registry",
        lambda *a, **kw: seed_calls.append((a, kw)),
    )

    mgr = manager_factory()
    mgr._plugin_install_which = _fake_which
    mgr._plugin_install_runner = _fake_runner
    # Isolated config_dir so the idempotency check never consults a real
    # ambient ~/.claude/plugins/installed_plugins.json on the host machine.
    mgr._plugin_install_config_dir = tmp_path / "claude_install_config"

    rec = mgr.create(str(git_repo), "feature/alpha")

    assert len(calls) == 1
    assert calls[0]["cmd"] == [
        "C:/fake/claude.exe", "plugin", "install", "my-plugin@marketplace", "--scope", "project",
    ]
    assert calls[0]["cwd"] == rec.path
    assert seed_calls == [], "seed_plugin_registry must not be called when the CLI is available"


@pytest.mark.requires_git
def test_create_never_calls_seed_when_cli_unavailable(
    manager_factory: Callable[..., WorktreeManager],
    git_repo: Path,
    tmp_path: Path,
    monkeypatch,
):
    """seed_plugin_registry is retired from manager.py (ticket #64): even when
    install_enabled_plugins() reports claude_unavailable=True (and there is no
    valid clone source, so the key ends up failed), create() must never call
    it -- and must still return successfully (best-effort)."""
    monkeypatch.setenv("WORKTREE_LOG_ROOT", str(tmp_path / "logs"))

    claude_dir = git_repo / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.json").write_text(
        json.dumps({"enabledPlugins": {"my-plugin@marketplace": True}}),
        encoding="utf-8",
    )

    import lib_python_worktree.core.plugin_seed as _ps_module  # noqa: PLC0415
    seed_calls: list = []
    monkeypatch.setattr(
        _ps_module, "seed_plugin_registry",
        lambda *a, **kw: seed_calls.append((a, kw)),
    )

    mgr = manager_factory()
    mgr._plugin_install_which = lambda *_a, **_kw: None  # simulate CLI absent
    # Isolated, empty config_dir -- no clone source exists anywhere, so the
    # key ends up in `failed`; create() must still succeed (best-effort).
    mgr._plugin_install_config_dir = tmp_path / "claude_install_config"

    rec = mgr.create(str(git_repo), "feature/alpha")

    assert rec is not None
    assert seed_calls == [], "seed_plugin_registry must never be called (retired in #64)"


# ---------------------------------------------------------------------------
# Ticket #46: WorktreeManager.run_seed_postprocess()
# ---------------------------------------------------------------------------

from lib_python_worktree.setup.runner import (  # noqa: E402
    SetupFailedError,
    SetupResult,
    SetupStepResult,
)


def test_manager_run_seed_postprocess_unknown_worktree_raises_not_found(tmp_path: Path):
    """run_seed_postprocess() raises WorktreeNotFoundError when the id is not in the store."""
    mgr = _make_mgr_in_memory(tmp_path)

    with pytest.raises(WorktreeNotFoundError):
        mgr.run_seed_postprocess("nonexistent-id-12345678")


def test_manager_run_seed_postprocess_empty_contract_is_noop(tmp_path: Path):
    """run_seed_postprocess() returns an empty SetupResult and never calls SetupRunner when seed_postprocess is empty."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    fake_contract = WorktreeContract(version=1, isolation="full", seed_postprocess=[])

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.setup.runner.SetupRunner") as mock_runner_cls,
    ):
        result = mgr.run_seed_postprocess(record.id)

    mock_runner_cls.assert_not_called()
    assert isinstance(result, SetupResult)
    assert result.worktree_id == record.id
    assert result.steps == []
    assert result.ok is True


def test_manager_run_seed_postprocess_delegates_to_setup_runner(tmp_path: Path):
    """run_seed_postprocess() calls SetupRunner.run with setup=seed_postprocess steps and correct kwargs."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(ports={"db": 5432})
    mgr.state.add(record)

    step = Step(run="patch-hostnames.sh", name="patch-hostnames")
    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        seed_postprocess=[step],
    )

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = SetupResult(worktree_id=record.id)

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner_instance),
    ):
        result = mgr.run_seed_postprocess(record.id)

    mock_runner_instance.run.assert_called_once_with(
        setup=fake_contract.seed_postprocess,
        worktree_id=record.id,
        worktree_path=Path(record.path),
        branch=record.branch,
        port_mapping=record.ports,
    )
    assert isinstance(result, SetupResult)


def test_manager_run_seed_postprocess_port_mapping_forwarded(tmp_path: Path):
    """run_seed_postprocess() forwards the worktree's port_mapping to SetupRunner.run."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(ports={"web": 8080, "grpc": 50051})
    mgr.state.add(record)

    step = Step(run="patch-ports.sh")
    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        seed_postprocess=[step],
    )

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = SetupResult(worktree_id=record.id)

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner_instance),
    ):
        mgr.run_seed_postprocess(record.id)

    call_kwargs = mock_runner_instance.run.call_args.kwargs
    assert call_kwargs["port_mapping"] == {"web": 8080, "grpc": 50051}


def test_manager_run_seed_postprocess_setup_failed_error_propagates(tmp_path: Path):
    """run_seed_postprocess() does NOT swallow SetupFailedError — it propagates."""
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record()
    mgr.state.add(record)

    step = Step(run="will-fail.sh")
    fake_contract = WorktreeContract(
        version=1,
        isolation="full",
        seed_postprocess=[step],
    )

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.side_effect = SetupFailedError(
        worktree_id=record.id,
        step_index=0,
        step_name="will-fail.sh",
        log_path=Path("/fake/log.log"),
        returncode=1,
    )

    with (
        patch("lib_python_worktree.core.manager._load_contract", return_value=fake_contract),
        patch("lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner_instance),
        pytest.raises(SetupFailedError),
    ):
        mgr.run_seed_postprocess(record.id)


def test_create_tolerates_seed_failure(monkeypatch):
    """create() returns normally even if seed_plugin_registry were to raise.

    seed_plugin_registry is retired from manager.py's create() as of ticket
    #64 (superseded by install_enabled_plugins' clone-first mechanism), so
    this monkeypatch is now inert -- it is kept to document that create()
    never depends on that module, and would still tolerate a failure there
    if some caller ever re-wired it. Uses InMemoryStateStore (no git, no
    filesystem worktree).
    """
    import lib_python_worktree.core.plugin_seed as _ps_module  # noqa: PLC0415

    monkeypatch.setattr(_ps_module, "seed_plugin_registry", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("simulated seed failure")))

    # Build a manager that uses InMemoryStateStore (no git → create() will
    # fail at _validate_repo before it ever reaches the seed call, so we need
    # to patch _validate_repo too).
    store = InMemoryStateStore()
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=Path("/nonexistent/store")),
        state=store,
        reconcile_on_init=False,
    )

    # Patch _validate_repo to bypass git.
    fake_repo = Path("/fake/repo")

    def _fake_validate(repo_root):  # noqa: ANN001
        return fake_repo

    monkeypatch.setattr(mgr, "_validate_repo", _fake_validate)

    # Also patch _branch_exists to return True.
    monkeypatch.setattr(mgr, "_branch_exists", lambda *_a: True)

    # Patch _run_git so that git worktree add succeeds (exit 0, no output).
    import lib_python_worktree.core.manager as _mgr_module  # noqa: PLC0415
    import types  # noqa: PLC0415

    fake_proc = types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(_mgr_module, "_run_git", lambda *_a, **_kw: fake_proc)

    # Patch the store_root / target_path creation; mkdir is called on
    # target_path.parent — patch Path.mkdir to be a no-op.
    original_mkdir = Path.mkdir

    def _noop_mkdir(self, **kwargs):  # noqa: ANN001
        pass  # don't actually create dirs

    monkeypatch.setattr(Path, "mkdir", _noop_mkdir)

    rec = mgr.create("/fake/repo", "feature/alpha")

    assert rec.branch == "feature/alpha"
    assert rec.repo_root == fake_repo.as_posix()

    # Restore mkdir for other tests.
    monkeypatch.setattr(Path, "mkdir", original_mkdir)


# ---------------------------------------------------------------------------
# Ticket #49: _build_worktree_env uses _get_user_profile_env as its base
# ---------------------------------------------------------------------------

from lib_python_worktree.core.manager import _build_worktree_env  # noqa: E402


def test_build_worktree_env_uses_get_user_profile_env_as_base(tmp_path: Path):
    """_build_worktree_env() starts from _get_user_profile_env(), not raw os.environ.

    Patching _get_user_profile_env to return a sentinel dict proves the function
    is called as the base rather than dict(os.environ).
    """
    record = _make_wt_record()

    sentinel_env = {"SENTINEL_FROM_PROFILE": "1"}

    with patch("lib_python_worktree.core.manager._get_user_profile_env", return_value=dict(sentinel_env)):
        result = _build_worktree_env(record, None)

    assert "SENTINEL_FROM_PROFILE" in result, (
        "_get_user_profile_env() sentinel key must appear in the built env"
    )
    assert result["SENTINEL_FROM_PROFILE"] == "1"


def test_build_worktree_env_worktree_vars_overlay_base(tmp_path: Path):
    """Worktree identity vars (WORKTREE_ID etc.) overlay the base env."""
    record = _make_wt_record(id="my-wt-deadbeef")

    # Base that already contains WORKTREE_ID with the wrong value.
    fake_base = {"WORKTREE_ID": "stale-id", "OTHER": "preserved"}

    with patch("lib_python_worktree.core.manager._get_user_profile_env", return_value=dict(fake_base)):
        result = _build_worktree_env(record, None)

    assert result["WORKTREE_ID"] == record.id, (
        "Worktree identity var must override the base environment"
    )
    assert result["OTHER"] == "preserved", "Non-colliding base keys must survive"


def test_build_worktree_env_caller_env_overlays_last(tmp_path: Path):
    """caller_env is applied last and wins over both base and worktree vars."""
    record = _make_wt_record(id="wt-abc12345")

    fake_base = {"X": "base"}

    caller_env = {"X": "caller"}

    with patch("lib_python_worktree.core.manager._get_user_profile_env", return_value=dict(fake_base)):
        result = _build_worktree_env(record, caller_env)

    assert result["X"] == "caller", (
        "caller_env must win (applied last) over the base and worktree vars"
    )


# ---------------------------------------------------------------------------
# Ticket #59: create() must branch from origin/<base>, not stale local <base>
# ---------------------------------------------------------------------------


@pytest.mark.requires_git
def test_create_fetch_false_uses_local_ref(manager: WorktreeManager, git_repo: Path):
    """fetch=False (offline path) branches from the local ref and must succeed
    even when there is no origin remote."""
    rec = manager.create(str(git_repo), "feature/offline", base="main", fetch=False)
    assert rec.branch == "feature/offline"
    assert Path(rec.path).exists()
    # Confirm the new worktree is on the expected branch.
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=rec.path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "feature/offline"


@pytest.mark.requires_git
def test_create_fetch_true_raises_git_command_error_when_no_origin(
    manager: WorktreeManager, git_repo: Path
):
    """With the default fetch=True, create() must raise GitCommandError when
    the repo has no origin remote (fetch fails).

    This guards that the default path does NOT silently fall back to the
    local ref — it must surface the network/remote failure instead.
    """
    with pytest.raises(GitCommandError):
        manager.create(str(git_repo), "feature/no-origin", base="main")


@pytest.mark.requires_git
def test_create_fetch_true_branches_from_origin_not_stale_local(
    tmp_path: Path, skip_if_no_git  # noqa: ARG001
):
    """Golden path: fetch=True branches the new worktree from origin/main's tip,
    not from a stale local main.

    Setup:
    1. Create an upstream bare repo with one commit.
    2. Clone it so that local main == upstream commit 1.
    3. Add a second commit directly to upstream (local main is now stale).
    4. Call create() with fetch=True (the default).
    5. Assert the new worktree's HEAD matches the upstream tip (commit 2),
       not the stale local main (commit 1).
    """
    # ---- Setup: upstream bare repo with initial commit ----
    upstream = tmp_path / "upstream.git"
    upstream.mkdir()
    _git("init", "--bare", "-b", "main", cwd=upstream)
    _git("config", "user.email", "test@example.com", cwd=upstream)
    _git("config", "user.name", "Test", cwd=upstream)

    # We need a non-bare working copy to create commits and push to upstream.
    staging = tmp_path / "staging"
    staging.mkdir()
    _git("init", "-b", "main", cwd=staging)
    _git("config", "user.email", "test@example.com", cwd=staging)
    _git("config", "user.name", "Test", cwd=staging)
    (staging / "README.md").write_text("commit 1\n", encoding="utf-8")
    _git("add", "-A", cwd=staging)
    _git("commit", "-q", "-m", "commit 1", cwd=staging)
    _git("remote", "add", "origin", str(upstream), cwd=staging)
    _git("push", "-u", "origin", "main", cwd=staging)

    # ---- Clone: local main is at commit 1 ----
    local = tmp_path / "local"
    subprocess.run(
        ["git", "clone", str(upstream), str(local)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=local,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=local,
        check=True,
        capture_output=True,
    )

    # Record commit 1 SHA (local main, which will become stale).
    local_head_before = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=local,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # ---- Advance upstream: add commit 2 without updating local main ----
    (staging / "extra.txt").write_text("commit 2\n", encoding="utf-8")
    _git("add", "-A", cwd=staging)
    _git("commit", "-q", "-m", "commit 2", cwd=staging)
    _git("push", "origin", "main", cwd=staging)

    # Record origin's tip (commit 2 SHA).
    origin_head = subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=local,
        # origin/main is NOT updated yet — fetch hasn't happened.
        # We need the upstream's HEAD directly.
        capture_output=True,
        text=True,
    )
    # Get the upstream tip by checking staging's HEAD (which is commit 2).
    upstream_tip = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=staging,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Confirm local is stale: local main != upstream tip.
    assert local_head_before != upstream_tip, (
        "Test setup error: local main must be behind upstream"
    )

    # ---- Call create() with default fetch=True ----
    store_root = tmp_path / "store"
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=store_root),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )

    rec = mgr.create(str(local), "feature/from-origin", base="main")
    assert rec.branch == "feature/from-origin"
    assert Path(rec.path).exists()

    # ---- Assert: new worktree's HEAD == upstream tip (commit 2), not stale local ----
    wt_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=rec.path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert wt_head == upstream_tip, (
        f"New worktree must start from origin/main (commit 2: {upstream_tip[:8]}), "
        f"not stale local main (commit 1: {local_head_before[:8]}). "
        f"Got: {wt_head[:8]}"
    )
    assert wt_head != local_head_before, (
        "New worktree must NOT start from the stale local main"
    )


# ---------------------------------------------------------------------------
# R2 (ticket #84, B2) -- _validate_repo() resolves the main clone from
# anywhere inside the repo, including from within a linked worktree.
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_create_from_linked_worktree_path_targets_main_clone(
    manager: WorktreeManager, git_repo: Path, linked_worktree: Path
):
    """R2 driving test: create() called with repo_root pointing INSIDE a
    linked worktree must target the main clone, not that linked worktree."""
    # fetch=False: the git_repo fixture has no `origin` remote configured
    # (same reason every other base= test in this file passes fetch=False).
    record = manager.create(str(linked_worktree), "feature/beta", base="main", fetch=False)

    assert record.repo_root == git_repo.resolve().as_posix()

    # The new worktree must appear in `git worktree list --porcelain` when
    # run from the main clone (proving it was actually created off the main
    # clone's registry, not the linked worktree's).
    listing = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=git_repo, capture_output=True, text=True, check=True,
    ).stdout
    assert record.path in listing.replace("\\", "/") or Path(record.path).name in listing


@pytest.mark.requires_git
def test_validate_repo_from_main_clone_subdir_targets_main_clone(
    manager: WorktreeManager, git_repo: Path
):
    subdir = git_repo / "sub"
    subdir.mkdir()
    resolved = manager._validate_repo(str(subdir))
    assert resolved == git_repo.resolve()


@pytest.mark.requires_git
def test_adopt_from_linked_worktree_path_targets_main_clone(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """adopt() called with a linked-worktree path still adopts against the
    main clone (extends the existing adopt() coverage above)."""
    state_dir = tmp_path / "state-r2-adopt"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store-r2-adopt"),
        state=store,
        reconcile_on_init=False,
    )

    oot_path = tmp_path / "oot-wt-r2"
    subprocess.run(
        ["git", "worktree", "add", str(oot_path), "-b", "feature/r2-oot", "main"],
        cwd=git_repo, check=True, capture_output=True,
    )
    try:
        report = mgr.adopt(str(linked_worktree))
        # linked_worktree (feature/alpha) itself is also untracked in this
        # fresh store, so BOTH it and oot_path (feature/r2-oot) get adopted.
        # The point under test is repo_root resolution, not adoption count.
        assert len(report.adopted) == 2
        for rec in mgr.list():
            assert rec.repo_root == git_repo.resolve().as_posix()
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(oot_path)],
            cwd=git_repo, capture_output=True,
        )


def test_validate_repo_non_repo_path_still_raises_invalid_repo_error(tmp_path: Path):
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )
    plain_dir = tmp_path / "not-a-repo"
    plain_dir.mkdir()
    with pytest.raises(InvalidRepoError):
        mgr._validate_repo(str(plain_dir))
