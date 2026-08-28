"""Tests for the updated _teardown method in WorktreeManager (ticket #8).

Verifies that:
- _teardown calls stop for each tracked PID.
- _teardown skips stop when no PIDs are recorded.
- _teardown runs contract teardown: steps via SetupRunner.
- _teardown skips teardown steps when the contract is missing.

Uses InMemoryStateStore and mocks; no real git required for the _teardown
unit tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from lib_python_worktree.core.manager import WorktreeManager, ManagerConfig
from lib_python_worktree.core.process_lifecycle import (
    KilledProcessInfo,
    ProcessNotRunningError,
    _PartialList,
)
from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_path: Path) -> WorktreeManager:
    """Create a WorktreeManager with an InMemoryStateStore (no git needed)."""
    return WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )


def _make_record(wt_id: str = "wt-td", **kwargs) -> WorktreeRecord:
    defaults = dict(
        id=wt_id,
        repo_root="/fake/repo",
        branch="feature/x",
        path="/fake/store/wt-td",
    )
    defaults.update(kwargs)
    return WorktreeRecord(**defaults)


@pytest.fixture(autouse=True)
def _no_blocking_processes_by_default():
    """Default-patch ``_find_blocking_processes`` to return ``[]`` for every
    test in this module (ticket #76).

    ``_teardown``'s Windows-only pre-flight check (Step 2b) calls
    ``_find_blocking_processes(record.path, os.getpid())`` before running
    ``git worktree remove``. These tests run on a real Windows dev machine,
    so ``sys.platform`` is genuinely ``"win32"`` even in tests that never
    patch ``manager.sys`` -- leaving this unpatched would exercise the real
    (slow, psutil-based) scan against fake paths in almost every test in
    this module. Tests that specifically exercise the new pre-flight
    behaviour override this via their own nested ``patch(...)`` for the
    duration of their ``with`` block (an inner ``patch`` on the same target
    always wins for its scope and unwinds back to this outer patch on exit).
    """
    with patch(
        "lib_python_worktree.core.teardown._find_blocking_processes",
        return_value=[],
    ):
        yield


@pytest.fixture(autouse=True)
def _no_real_preflight_settle_sleep():
    """Default-patch ``time.sleep`` (as seen through ``manager.time``) to a
    no-op for every test in this module (ticket #117).

    Gate A's confirmed-blocker pre-flight check now sleeps
    ``_PREFLIGHT_SETTLE_SLEEP`` seconds before re-scanning a foreign
    (non-owned) blocker. Left unpatched, that would cost ~1s of real wall
    time across every foreign-blocker pre-flight test in this module. Tests
    that specifically assert on sleep/retry behaviour override this via
    their own nested ``patch("lib_python_worktree.core.teardown.time")``
    (patching the whole module reference, not just ``.sleep``) for the
    duration of their ``with`` block -- an inner patch on the same target
    always wins for its scope and unwinds back to this outer patch on exit.
    """
    with patch("lib_python_worktree.core.teardown.time.sleep"):
        yield


@pytest.fixture(autouse=True)
def _target_present_by_default():
    """Default-patch ``teardown._target_is_absent`` (ticket #135) so that
    BOTH ``target_absent`` probe sites (``remove()``'s own probe and
    ``_teardown()``'s, both of which now call the shared
    ``teardown._target_is_absent(record)`` seam) read the checkout as
    PRESENT for every test in this module, while every OTHER
    ``os.path.exists`` call (the long-path fallback and the Final guard in
    ``_teardown``, which still call ``os.path.exists`` directly, not through
    this seam -- see ticket #135's R4) keeps evaluating against the real,
    always-absent fake path used throughout this module.

    Replaces the former ~120-line stack-walking version of this fixture
    (ticket #127 fix-pass 2), which had to distinguish the two
    ``target_absent = not os.path.exists(record.path)`` call sites from
    every other ``os.path.exists``/``Path.exists()`` call by walking the
    stack to the direct caller's source line. Ticket #135 replaced both
    call sites with a single named seam (``teardown._target_is_absent``),
    so that stack-walking is no longer needed: patching the seam directly
    naturally leaves every other filesystem check alone.

    Tests that specifically exercise ``target_absent`` behaviour
    (``TestTeardownAbsentTarget``) override this via their own nested
    ``patch("lib_python_worktree.core.teardown._target_is_absent", ...)``
    for the duration of their ``with`` block -- an inner ``patch`` on the
    same target always wins for its scope and unwinds back to this outer
    patch on exit.
    """
    with patch(
        "lib_python_worktree.core.teardown._target_is_absent",
        return_value=False,
    ):
        yield


def _run_teardown_with_mocked_git(
    manager: WorktreeManager,
    record: WorktreeRecord,
    *,
    force: bool = False,
    lifecycle_module=None,
) -> None:
    """Call _teardown with git subprocess mocked to succeed."""
    with patch(
        "lib_python_worktree.core.teardown._run_git"
    ) as mock_git:
        mock_git.return_value = MagicMock(returncode=0, stderr="")
        manager._teardown(record, force=force, _lifecycle_module=lifecycle_module)


# ---------------------------------------------------------------------------
# test_teardown_calls_stop_when_pids_present
# ---------------------------------------------------------------------------

class TestTeardownCallsStop:
    def test_teardown_calls_stop_when_pids_present(self, tmp_path):
        """_teardown invokes lifecycle.stop for each PID in record.pids."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-stop-test", pids={"main": 12345, "worker": 67890})
        manager.state.add(record)

        stop_calls = []

        mock_lifecycle = MagicMock()
        mock_lifecycle.stop.side_effect = lambda wt_id, store, role: (
            stop_calls.append((wt_id, role))
        )

        _run_teardown_with_mocked_git(
            manager, record, lifecycle_module=mock_lifecycle
        )

        # stop must be called for each role
        called_roles = {role for (_, role) in stop_calls}
        assert "main" in called_roles
        assert "worker" in called_roles

    def test_teardown_skips_stop_when_no_pids(self, tmp_path):
        """_teardown does not call lifecycle.stop when pids is empty."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-no-pids")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        _run_teardown_with_mocked_git(
            manager, record, lifecycle_module=mock_lifecycle
        )

        mock_lifecycle.stop.assert_not_called()

    def test_teardown_swallows_process_not_running(self, tmp_path):
        """_teardown swallows ProcessNotRunningError from lifecycle.stop."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-swallow", pids={"main": 99999999})
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        mock_lifecycle.stop.side_effect = ProcessNotRunningError("wt-swallow", "main")

        # Must not raise
        _run_teardown_with_mocked_git(
            manager, record, lifecycle_module=mock_lifecycle
        )


# ---------------------------------------------------------------------------
# test_teardown_runs_contract_teardown_steps
# ---------------------------------------------------------------------------

class TestTeardownContractSteps:
    def test_teardown_runs_contract_teardown_steps(self, tmp_path):
        """_teardown runs teardown: steps from the contract via SetupRunner."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-contract")
        manager.state.add(record)

        # Build a fake contract with one teardown step.
        from lib_python_worktree.contract.schema import Step, WorktreeContract
        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo "bye"', name="goodbye")],
        )

        runner_calls = []

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        mock_lifecycle = MagicMock()
        mock_lifecycle.stop.side_effect = ProcessNotRunningError("wt-contract", "main")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git"
            ) as mock_git,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            manager._teardown(
                record, force=False, _lifecycle_module=mock_lifecycle
            )

        assert len(runner_calls) == 1
        kw = runner_calls[0]
        assert kw["worktree_id"] == "wt-contract"
        assert kw["setup"] == fake_contract.teardown

    def test_teardown_skips_teardown_steps_on_missing_contract(self, tmp_path):
        """_teardown continues without error when no contract file exists."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-no-contract")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        # _load_contract for a missing file returns isolation:none contract —
        # so teardown list is empty; SetupRunner.run should NOT be called.
        mock_runner_instance = MagicMock()

        with (
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git"
            ) as mock_git,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            # No patch on _load_contract — let the real loader handle a
            # non-existent path (returns isolation:none with empty teardown).
            manager._teardown(
                record, force=False, _lifecycle_module=mock_lifecycle
            )

        mock_runner_instance.run.assert_not_called()

    def test_teardown_skips_steps_on_contract_load_error(self, tmp_path):
        """_teardown continues when _load_contract raises unexpectedly."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-contract-err")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        mock_runner_instance = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                side_effect=RuntimeError("disk error"),
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git"
            ) as mock_git,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            manager._teardown(
                record, force=False, _lifecycle_module=mock_lifecycle
            )

        mock_runner_instance.run.assert_not_called()

    def test_teardown_continues_when_teardown_step_fails(self, tmp_path):
        """_teardown proceeds to git-remove even when a teardown step raises."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-step-fail")
        manager.state.add(record)

        from lib_python_worktree.contract.schema import Step, WorktreeContract
        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='exit 1', name="will-fail")],
        )

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = RuntimeError("step failed")

        mock_lifecycle = MagicMock()

        git_calls = []

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git"
            ) as mock_git,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_git.side_effect = lambda *a, **kw: (
                git_calls.append(a), MagicMock(returncode=0, stderr="")
            )[1]
            manager._teardown(
                record, force=False, _lifecycle_module=mock_lifecycle
            )

        # git worktree remove must still be called despite the step failure.
        assert any(
            "worktree" in str(args) and "remove" in str(args)
            for args in git_calls
        ), "git worktree remove must be called even when teardown step fails"


# ---------------------------------------------------------------------------
# TestTeardownForceExit128 -- ticket #5 / #11 regression tests
# ---------------------------------------------------------------------------

class TestTeardownForceExit128:
    """Verify the exit-128 fallback path in _teardown(force=True).

    When 'git worktree remove --force' exits 128 (the .git link is already
    gone), _teardown must NOT raise; instead it falls back to shutil.rmtree +
    git worktree prune, then continues to port release.
    """

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _make_mock_git_128():
        """Return a _run_git mock whose first call returns returncode=128."""
        def _side_effect(args, cwd=None, **kwargs):
            # The first call is 'worktree remove --force <path>' → exit 128.
            # Subsequent calls (worktree prune) succeed.
            if "remove" in args:
                return MagicMock(returncode=128, stderr="fatal: not a git repo")
            return MagicMock(returncode=0, stderr="")
        return _side_effect

    # --- tests -------------------------------------------------------------

    def test_force_exit128_does_not_raise(self, tmp_path):
        """Regression: exit 128 with force=True must not raise GitCommandError."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-128")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=self._make_mock_git_128(),
        ), patch("lib_python_worktree.core.teardown.shutil"):
            # Should complete without raising.
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)

    def test_force_exit128_calls_rmtree(self, tmp_path):
        """shutil.rmtree must be called with record.path and ignore_errors=True."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-rmtree", path="/fake/store/wt-rmtree")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=self._make_mock_git_128(),
        ), patch(
            "lib_python_worktree.core.teardown.shutil"
        ) as mock_shutil:
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)

        mock_shutil.rmtree.assert_called_once_with(
            record.path, ignore_errors=True
        )

    def test_force_exit128_calls_worktree_prune(self, tmp_path):
        """git worktree prune must be called on the repo root after rmtree."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-prune", repo_root="/fake/repo")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        git_calls = []

        def _tracking_git(args, cwd=None, **kwargs):
            git_calls.append((list(args), cwd))
            if "remove" in args:
                return MagicMock(returncode=128, stderr="fatal: not a git repo")
            return MagicMock(returncode=0, stderr="")

        with patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=_tracking_git,
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)

        prune_calls = [
            (args, cwd) for (args, cwd) in git_calls
            if args[:2] == ["worktree", "prune"]
        ]
        assert prune_calls, "git worktree prune was not called"
        _, prune_cwd = prune_calls[0]
        assert prune_cwd == Path(record.repo_root), (
            f"prune cwd should be Path(record.repo_root)={Path(record.repo_root)!r}, "
            f"got {prune_cwd!r}"
        )

    def test_force_exit128_releases_ports(self, tmp_path):
        """Ports must be released even when the exit-128 fallback path is taken."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-ports-128")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        # Replace the real (no-op) allocator with a spy.
        mock_allocator = MagicMock()
        manager._allocator = mock_allocator

        with patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=self._make_mock_git_128(),
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)

        mock_allocator.release.assert_called_once_with(record.id)

    def test_force_exit128_state_removed_after_remove(self, tmp_path):
        """Full remove() must leave state.list() empty after exit-128 fallback."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-state-128",
            branch_created_by_us=False,  # skip branch-delete step
        )
        manager.state.add(record)

        # Verify the record is present before removal.
        assert len(manager.state.list()) == 1

        mock_lifecycle = MagicMock()

        with patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=self._make_mock_git_128(),
        ), patch(
            "lib_python_worktree.core.teardown.shutil"
        ), patch.object(
            manager, "_teardown", wraps=lambda rec, force, **kw: (
                # Call the real _teardown but inject the mock lifecycle.
                WorktreeManager._teardown(
                    manager, rec, force=force, _lifecycle_module=mock_lifecycle
                )
            )
        ):
            manager.remove(record.id, force=True)

        assert manager.state.list() == [], "state should be empty after remove()"

    def test_non128_error_still_raises(self, tmp_path):
        """exit 1 with force=True must still raise GitCommandError."""
        from lib_python_worktree.core.manager import GitCommandError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-non128")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _git_exit1(args, cwd=None, **kwargs):
            if "remove" in args:
                return MagicMock(returncode=1, stderr="fatal: other error")
            return MagicMock(returncode=0, stderr="")

        with pytest.raises(GitCommandError), patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=_git_exit1,
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)

    def test_exit128_without_force_raises_dirty_error(self, tmp_path):
        """exit 128 with force=False must raise DirtyWorktreeError (not a bare
        GitCommandError), and the message must contain 'force=True' but not
        '--force' or the raw exit code '128'."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-dirty-no-force")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _git_exit128(args, cwd=None, **kwargs):
            return MagicMock(
                returncode=128,
                stderr=(
                    "fatal: 'some/path' contains modified or untracked files,"
                    " use --force to delete it"
                ),
            )

        with pytest.raises(DirtyWorktreeError) as excinfo, patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=_git_exit128,
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        msg = str(excinfo.value)
        assert "force=True" in msg
        assert "--force" not in msg
        assert "128" not in msg

    def test_exit128_without_force_non_dirty_stderr_raises_git_error(self, tmp_path):
        """exit 128 with force=False but a non-dirty stderr (e.g. 'not a git
        repo') must raise GitCommandError, not DirtyWorktreeError, so the
        caller sees the real failure reason."""
        from lib_python_worktree.core.manager import GitCommandError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-128-non-dirty")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _git_not_a_repo(args, cwd=None, **kwargs):
            return MagicMock(returncode=128, stderr="fatal: not a git repo")

        with pytest.raises(GitCommandError), patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=_git_not_a_repo,
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)


# ---------------------------------------------------------------------------
# TestContractCopyDirtExemption -- ticket #100
# ---------------------------------------------------------------------------

_DIRTY_STDERR = (
    "fatal: 'some/path' contains modified or untracked files, use "
    "--force to delete it"
)


def _is_plain_remove_call(args: list) -> bool:
    return args[:2] == ["worktree", "remove"] and "--force" not in args


def _is_force_remove_call(args: list) -> bool:
    return args[:2] == ["worktree", "remove"] and "--force" in args


def _is_status_call(args: list) -> bool:
    return args[:3] == ["status", "--porcelain", "-z"]


class TestContractCopyDirtExemption:
    """Ticket #100: an untracked-only `.seretos/` convenience copy must not
    force `force=True` on a plain `remove()`. Several of these cases already
    passed before the fix (they pin the guard so a future change cannot
    silently widen the exemption to real dirt)."""

    def test_only_seretos_dir_untracked_retry_succeeds_no_raise(self, tmp_path):
        """Benign dirt (only `.seretos/`, untracked): the plain remove is
        refused, the status probe shows only `?? .seretos/`, the retry with
        --force succeeds, and _teardown does not raise."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-benign-dirt")
        manager.state.add(record)

        calls = []

        def _side_effect(args, cwd=None, **kwargs):
            calls.append(list(args))
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? .seretos/\0", stderr="")
            if _is_force_remove_call(args):
                return MagicMock(returncode=0, stderr="")
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()
        mock_allocator = MagicMock()
        manager._allocator = mock_allocator

        with patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert any(_is_force_remove_call(c) for c in calls), (
            "the benign-dirt path must retry with --force"
        )
        mock_allocator.release.assert_called_once_with(record.id)

    def test_untracked_file_outside_seretos_still_raises_dirty_error(self, tmp_path):
        """An untracked file outside `.seretos/` is real dirt -- must still
        raise DirtyWorktreeError, not auto-escalate."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-real-dirt")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? scratch.txt\0", stderr="")
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with pytest.raises(DirtyWorktreeError), patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_mixed_seretos_and_other_untracked_still_raises_dirty_error(self, tmp_path):
        """`.seretos/` plus another untracked file is still real dirt."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-mixed-dirt")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(
                    returncode=0, stdout="?? .seretos/\0?? scratch.txt\0", stderr=""
                )
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with pytest.raises(DirtyWorktreeError), patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_modified_tracked_file_under_seretos_still_raises_dirty_error(
        self, tmp_path
    ):
        """A MODIFIED (tracked) file under `.seretos/` is real work, not the
        untracked convenience copy -- the exemption is untracked-only."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-modified-seretos")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(
                    returncode=0,
                    stdout=" M .seretos/worktree-setup.yml\0",
                    stderr="",
                )
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with pytest.raises(DirtyWorktreeError), patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_empty_status_output_still_raises_dirty_error(self, tmp_path):
        """An empty `git status` result is inconsistent with git having just
        refused the remove -- never auto-force in that case."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-empty-status")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with pytest.raises(DirtyWorktreeError), patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_status_nonzero_exit_still_raises_dirty_error(self, tmp_path):
        """A non-zero `git status` exit must not be treated as benign."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-status-nonzero")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=1, stdout="", stderr="fatal: boom")
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with pytest.raises(DirtyWorktreeError), patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_status_non_str_stdout_still_raises_dirty_error(self, tmp_path):
        """Defensive: a test (or a real caller) that patches `_run_git` with
        a blanket `MagicMock(...)` leaves `.stdout` as a `MagicMock`, not a
        `str`. This must never be misread as "no output" / benign -- it must
        still fall through to DirtyWorktreeError. This is the exact shape
        `test_exit128_without_force_raises_dirty_error` (above) and
        `test_dirty_worktree_error_message_no_git_internals`
        (test_manager.py) already use, pinned here explicitly."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-non-str-stdout")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            # A blanket MagicMock (no explicit stdout=) for every call,
            # including the dirty-refusal itself: `.stdout` on the returned
            # MagicMock is itself a MagicMock, not a str.
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with pytest.raises(DirtyWorktreeError), patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_status_git_timeout_still_raises_dirty_error(self, tmp_path):
        """A `GitTimeoutError` from the status probe must not block removal
        with an unrelated exception -- it degrades to the ordinary
        DirtyWorktreeError path."""
        from lib_python_worktree.core.manager import DirtyWorktreeError, GitTimeoutError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-status-timeout")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                raise GitTimeoutError(["git", *args], 30.0)
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with pytest.raises(DirtyWorktreeError), patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_benign_dirt_retry_phantom_state_no_raise(self, tmp_path):
        """Benign dirt, but the retry itself reports 'is not a working tree'
        (phantom-state, ticket #51) -- treated as already-gone, no raise."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-benign-phantom")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? .seretos/\0", stderr="")
            if _is_force_remove_call(args):
                return MagicMock(
                    returncode=128,
                    stderr=f"fatal: '{record.path}' is not a working tree",
                )
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            # Must not raise.
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_benign_dirt_retry_other_failure_raises_git_command_error(self, tmp_path):
        """Benign dirt, but the --force retry fails for an unrelated reason
        -- surfaced as GitCommandError, not silently swallowed."""
        from lib_python_worktree.core.manager import GitCommandError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-benign-retry-fails")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? .seretos/\0", stderr="")
            if _is_force_remove_call(args):
                return MagicMock(returncode=1, stderr="fatal: other error")
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with pytest.raises(GitCommandError), patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_benign_dirt_retry_lock_signal_kill_and_retry_succeeds_no_raise(
        self, tmp_path
    ):
        """Blocking-finding-1 follow-up: the `.seretos/`-exemption --force
        retry itself hits a genuine directory lock (e.g. an AV scanner
        holding a handle on a file under `.seretos/` exactly when the
        forced delete runs). With kill_blocking_processes=True this must be
        routed through the SAME kill-and-retry remedy as the primary
        removal attempt -- not a bare GitCommandError -- and succeed once
        the second force-retry (post-kill) reports rc=0."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record("wt-benign-retry-lock-kill", path="/fake/store/wt-lock")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=4242, name="node.exe", cmdline=["node"])]
        force_calls = {"n": 0}

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? .seretos/\0", stderr="")
            if _is_force_remove_call(args):
                force_calls["n"] += 1
                if force_calls["n"] == 1:
                    # The exemption's own --force retry hits the lock.
                    return MagicMock(returncode=255, stderr="Permission denied")
                # Post-kill retry succeeds.
                return MagicMock(returncode=0, stderr="")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch("lib_python_worktree.core.teardown.shutil"),
        ):
            mock_sys.platform = "win32"
            # Must not raise.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_called_once_with(record.path)
        assert record.killed_pids == fake_killed
        assert force_calls["n"] == 2

    def test_benign_dirt_retry_lock_signal_flag_off_raises_dir_locked(
        self, tmp_path
    ):
        """Same lock-on-retry scenario, but kill_blocking_processes=False
        (the default): must raise WorktreeDirLockedError naming the
        kill_blocking_processes=True remedy -- NOT a bare GitCommandError
        leaking git internals -- and no kill may be attempted."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-benign-retry-lock-flagoff")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? .seretos/\0", stderr="")
            if _is_force_remove_call(args):
                return MagicMock(returncode=255, stderr="Permission denied")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch("lib_python_worktree.core.teardown.shutil"),
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_not_called()
        err = excinfo.value
        assert err.kill_attempted is False
        assert err.killed == []
        assert "kill_blocking_processes=True" in str(err)

    def test_benign_dirt_retry_lock_signal_still_locked_raises_dir_locked(
        self, tmp_path
    ):
        """Same lock-on-retry scenario with kill_blocking_processes=True,
        but the directory stays locked through every post-kill retry:
        WorktreeDirLockedError must still be raised (naming the killed
        processes) rather than a bare GitCommandError."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-benign-retry-lock-stilllocked", path="/fake/store/wt-still-locked"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=9999, name="claude", cmdline=["claude"])]

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? .seretos/\0", stderr="")
            if _is_force_remove_call(args):
                return MagicMock(returncode=255, stderr="Permission denied")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch("lib_python_worktree.core.teardown.time"),
            patch("lib_python_worktree.core.teardown.shutil"),
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=True,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_called_once_with(record.path)
        err = excinfo.value
        assert err.killed == fake_killed
        assert record.killed_pids == fake_killed

    def test_benign_dirt_discard_logs_warning_naming_paths(self, tmp_path, caplog):
        """Codex correctness follow-up (ticket #100): the auto-escalation
        must not be silent -- _teardown must emit a WARNING naming the
        actual untracked path(s) it is about to discard, not just the fact
        that an escalation happened. Uses an untracked `.seretos/notes.txt`
        -- something OTHER than the plugin's copied `worktree-setup.yml` --
        since that is precisely the scenario the finding is about: the
        exemption covers any untracked content under `.seretos/`, not
        merely the contract copy."""
        import logging as _logging

        manager = _make_manager(tmp_path)
        record = _make_record("wt-warn-discard")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(
                    returncode=0, stdout="?? .seretos/notes.txt\0", stderr=""
                )
            if _is_force_remove_call(args):
                return MagicMock(returncode=0, stderr="")
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()
        mock_allocator = MagicMock()
        manager._allocator = mock_allocator

        caplog.set_level(_logging.WARNING, logger="lib_python_worktree.core.manager")

        with patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert any(".seretos/notes.txt" in r.message for r in warnings), (
            "expected a WARNING naming the discarded path '.seretos/notes.txt', "
            f"got: {[r.message for r in warnings]}"
        )

    def test_lock_signal_with_dirty_phrase_raises_dir_locked_not_exemption(
        self, tmp_path
    ):
        """Lock-signal precedence (pre-existing #72 ordering; pinned here
        explicitly per the plan's checklist for this class): a stderr that
        is BOTH a lock signal (Windows 'Permission denied') AND contains the
        dirty-tree phrase must still raise WorktreeDirLockedError (not the
        `.seretos/`-only exemption path) as the PRIMARY classification.

        Ticket #103: unlike before, the dirt probe now DOES run on this
        path (folded into `_resolve_lock_or_raise` so a genuinely combined
        lock+dirt condition can be reported in one shot) -- but this mock
        returns rc=128 for every call, including the status probe, so the
        probe is inconclusive and no dirt is ever claimed. The result stays
        a plain single-condition WorktreeDirLockedError, not the combined
        WorktreeRemovalBlockedError."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-lock-and-dirty")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        calls = []

        def _side_effect(args, cwd=None, **kwargs):
            calls.append(list(args))
            return MagicMock(
                returncode=128,
                stderr=(
                    "fatal: 'some/path' contains modified or untracked files,"
                    " use --force to delete it (Permission denied)"
                ),
            )

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert type(excinfo.value) is WorktreeDirLockedError, (
            "an inconclusive dirt probe (rc=128 on the status call too) "
            "must never escalate to the combined WorktreeRemovalBlockedError"
        )
        assert any(_is_status_call(c) for c in calls), (
            "ticket #103: the dirt probe must now run on the lock-signal "
            "path so a genuinely combined lock+dirt condition can be "
            "detected -- it just stays inconclusive here"
        )

    def test_seretos_backup_sibling_not_exempted_still_raises_dirty_error(
        self, tmp_path
    ):
        """Prefix-match correctness: a `.seretos-backup/` untracked
        directory is a `.seretos`-prefixed SIBLING, not `.seretos` itself or
        a path under it -- it must NOT be treated as exempt."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-seretos-backup-sibling")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(
                    returncode=0, stdout="?? .seretos-backup/\0", stderr=""
                )
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with pytest.raises(DirtyWorktreeError), patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_literal_backslash_in_path_not_normalised_still_raises_dirty_error(
        self, tmp_path
    ):
        """`git status --porcelain -z` always reports paths with `/` as the
        separator (git's index format mandates it, on every OS including
        Windows) -- a literal backslash in an entry is part of the filename
        itself, not a Windows-style separator that needs normalising. An
        untracked file literally named `.seretos\\notes.txt` sitting at the
        repo root (entry `?? .seretos\\notes.txt`) is NOT under `.seretos/`
        and must not be misclassified as the benign exempt copy."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-literal-backslash")
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(
                    returncode=0, stdout="?? .seretos\\notes.txt\0", stderr=""
                )
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with pytest.raises(DirtyWorktreeError), patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_happy_path_issues_no_status_call(self, tmp_path):
        """When `git worktree remove` succeeds outright (exit 0), the dirt
        classifier must never run -- no `git status` call at all."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-happy-no-status")
        manager.state.add(record)

        calls = []

        def _side_effect(args, cwd=None, **kwargs):
            calls.append(list(args))
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ), patch("lib_python_worktree.core.teardown.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert not any(_is_status_call(c) for c in calls), (
            f"git status must not be called on the happy path, got: {calls}"
        )


# ---------------------------------------------------------------------------
# TestKillBlockingProcesses* -- ticket #29
# ---------------------------------------------------------------------------

class TestKillBlockingProcessesWindows:
    """Windows path: rc=255 + 'Permission denied' triggers kill+retry."""

    def test_kill_and_retry_succeeds_no_raise(self, tmp_path):
        """First git call returns 255/'Permission denied'; second returns 0.
        kill helper called once; no exception raised; record.killed_pids set."""
        import sys
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record("wt-win-kill", path="/fake/store/wt-win-kill")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=1234, name="node.exe", cmdline=["node"])]

        call_count = {"n": 0}

        def _git_side_effect(args, cwd=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MagicMock(returncode=255, stderr="Permission denied")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_called_once_with(record.path)
        assert record.killed_pids == fake_killed
        assert call_count["n"] == 2

    def test_flag_off_lock_signal_raises_dir_locked_with_remedy(self, tmp_path):
        """Ticket #72 (Befund 2): with kill_blocking_processes=False (default),
        rc=255/'Permission denied' (a lock signal) must raise
        WorktreeDirLockedError naming the kill_blocking_processes=True remedy
        — not a raw GitCommandError — and the kill helper must never be
        called (no kill is attempted when the flag is off)."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-win-flagoff")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _git_perm_denied(args, cwd=None, **kwargs):
            return MagicMock(returncode=255, stderr="Permission denied")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_perm_denied),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_not_called()
        err = excinfo.value
        assert err.kill_attempted is False
        assert err.killed == []
        msg = str(err)
        assert "kill_blocking_processes=True" in msg
        # No raw git stderr/path/exit-code leakage in the remedy message.
        assert "Permission denied" not in msg
        assert record.path not in msg
        assert "255" not in msg

    def test_still_locked_after_retry_raises_dir_locked_error(self, tmp_path):
        """Both git calls fail; WorktreeDirLockedError raised with killed list."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record("wt-win-locked", path="/fake/store/wt-win-locked")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=5678, name="claude", cmdline=["claude", "--bg"])]

        def _git_always_fail(args, cwd=None, **kwargs):
            return MagicMock(returncode=255, stderr="Permission denied")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_always_fail),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch("lib_python_worktree.core.teardown.time"),
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as exc_info:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=True,
                    _lifecycle_module=mock_lifecycle,
                )

        err = exc_info.value
        assert err.worktree_id == "wt-win-locked"
        assert err.killed == fake_killed
        assert record.killed_pids == fake_killed


class TestKillBlockingProcessesPosix:
    """POSIX path: 'locked' in stderr with flag triggers kill+retry."""

    def test_posix_locked_stderr_kill_and_retry_succeeds(self, tmp_path):
        """POSIX: stderr containing 'locked' with kill_blocking_processes=True
        triggers kill+retry and succeeds on the second call."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record("wt-posix-kill", path="/fake/store/wt-posix-kill")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=9999, name="codex-broker", cmdline=["codex"])]

        call_count = {"n": 0}

        def _git_side_effect(args, cwd=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MagicMock(returncode=128, stderr="error: unable to lock worktree")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "linux"
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_called_once_with(record.path)
        assert record.killed_pids == fake_killed
        assert call_count["n"] == 2

    def test_posix_locked_stderr_case_insensitive(self, tmp_path):
        """POSIX: 'Locked' (capital L) — i.e. the 'lock' substring is present
        case-insensitively — in stderr also triggers kill+retry."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record("wt-posix-lock-ci", path="/fake/store/wt-posix-lock-ci")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=8888, name="sh", cmdline=["sh"])]

        call_count = {"n": 0}

        def _git_side_effect(args, cwd=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MagicMock(returncode=1, stderr="fatal: worktree is Locked")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "linux"
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_called_once_with(record.path)
        assert call_count["n"] == 2

    def test_posix_still_locked_raises_dir_locked_error(self, tmp_path):
        """POSIX: both calls fail with lock stderr; WorktreeDirLockedError raised."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record("wt-posix-locked")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=7777, name="sh", cmdline=["sh"])]

        def _git_always_fail(args, cwd=None, **kwargs):
            return MagicMock(returncode=128, stderr="error: cannot lock worktree")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_always_fail),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch("lib_python_worktree.core.teardown.time"),
        ):
            mock_sys.platform = "linux"
            with pytest.raises(WorktreeDirLockedError) as exc_info:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=True,
                    _lifecycle_module=mock_lifecycle,
                )

        assert exc_info.value.worktree_id == record.id
        assert exc_info.value.killed == fake_killed

    def test_posix_non_lock_stderr_raises_git_command_error(self, tmp_path):
        """POSIX: flag on but stderr has NO 'locked' pattern → GitCommandError,
        kill helper never called.  Covers broken-repo / network-FS error paths."""
        from lib_python_worktree.core.manager import GitCommandError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-posix-non-lock")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _git_broken_repo(args, cwd=None, **kwargs):
            return MagicMock(returncode=128, stderr="fatal: not a git repository")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_broken_repo),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "linux"
            with pytest.raises(GitCommandError):
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=True,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_not_called()


class TestTicket72LockVsDirtyClassification:
    """Regression tests for ticket #72: lock-signal detection must run BEFORE
    the dirty-tree check (Befund 1), and a genuine lock signal must raise a
    clean WorktreeDirLockedError instead of leaking GitCommandError when
    kill_blocking_processes=False, for BOTH force=True and force=False
    (Befund 2)."""

    def test_befund1_lock_and_dirty_stderr_raises_dir_locked_not_dirty(self, tmp_path):
        """Windows, force=False, stderr containing BOTH a Win32 lock indicator
        ('Permission denied') AND the dirty-tree phrase ('contains modified or
        untracked files'), kill_blocking_processes=False.

        Pre-fix: the dirty-tree substring check ran before any lock check, so
        this stderr was misclassified as DirtyWorktreeError.
        Post-fix: the lock-signal check runs first, so this must raise
        WorktreeDirLockedError (kill_attempted=False), not DirtyWorktreeError.
        """
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-befund1")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _git_lock_and_dirty(args, cwd=None, **kwargs):
            return MagicMock(
                returncode=128,
                stderr=(
                    "fatal: 'some/path' contains modified or untracked files,"
                    " use --force to delete it (Permission denied)"
                ),
            )

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_lock_and_dirty),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_not_called()
        assert excinfo.value.kill_attempted is False
        assert excinfo.value.killed == []

    def test_befund2_force_true_flag_off_lock_signal_raises_dir_locked(self, tmp_path):
        """Windows, force=True, rc=255 'Permission denied', kill_blocking_processes=False.

        Pre-fix: this genuine lock signal fell through to the else branch's
        GitCommandError, leaking git's raw stderr.
        Post-fix: must raise WorktreeDirLockedError (kill_attempted=False);
        _kill_blocking_processes must never be called; no GitCommandError.
        """
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-befund2")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _git_perm_denied(args, cwd=None, **kwargs):
            return MagicMock(returncode=255, stderr="Permission denied")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_perm_denied),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_not_called()
        assert excinfo.value.kill_attempted is False
        assert excinfo.value.killed == []

    def test_befund2_force_true_flag_off_invalid_argument_raises_dir_locked(self, tmp_path):
        """Same as above but with 'Invalid argument' instead of 'Permission
        denied' — both Win32 lock strings must be covered."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-befund2-invarg")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _git_invalid_arg(args, cwd=None, **kwargs):
            return MagicMock(returncode=255, stderr="Invalid argument")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_invalid_arg),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_not_called()
        assert excinfo.value.kill_attempted is False

    @pytest.mark.parametrize("force_flag", [True, False])
    def test_posix_flag_off_lock_stderr_raises_dir_locked_with_remedy(
        self, tmp_path, force_flag
    ):
        """POSIX variant of the remedy branch: 'lock' in stderr (case
        insensitive), kill_blocking_processes=False, either force value ->
        WorktreeDirLockedError(kill_attempted=False)."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record(f"wt-posix-remedy-{force_flag}")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _git_locked(args, cwd=None, **kwargs):
            return MagicMock(returncode=1, stderr="fatal: worktree is Locked")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_locked),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "linux"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=force_flag,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_not_called()
        assert excinfo.value.kill_attempted is False
        assert excinfo.value.killed == []

    def test_force_true_kill_true_still_locked_after_retry_raises_dir_locked(
        self, tmp_path
    ):
        """Negative control: force=True + kill_blocking_processes=True, both
        git calls fail with a lock signal -> WorktreeDirLockedError with
        kill_attempted defaulting True and a non-empty killed list (the
        unified lock branch applies for force=True exactly as it does for
        force=False)."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record("wt-force-kill-locked", path="/fake/store/wt-force-kill-locked")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=4242, name="node.exe", cmdline=["node"])]

        def _git_always_fail(args, cwd=None, **kwargs):
            return MagicMock(returncode=255, stderr="Permission denied")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_always_fail),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch("lib_python_worktree.core.teardown.time"),
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=True,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_called_once_with(record.path)
        err = excinfo.value
        assert err.kill_attempted is True
        assert err.killed == fake_killed
        assert err.killed != []


class TestWorktreeDirLockedErrorMessages:
    """Direct unit tests for WorktreeDirLockedError's two message phrasings
    (ticket #72)."""

    def test_kill_attempted_true_message_mentions_killed_count(self):
        from lib_python_worktree.core._exceptions import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        killed = [
            KilledProcessInfo(pid=1, name="a.exe", cmdline=["a"]),
            KilledProcessInfo(pid=2, name="b.exe", cmdline=["b"]),
        ]
        err = WorktreeDirLockedError("wt-1", killed=killed, kill_attempted=True)

        assert err.kill_attempted is True
        assert err.killed == killed
        msg = str(err)
        assert "after killing 2" in msg
        assert "wt-1" in msg

    def test_kill_attempted_false_message_mentions_remedy_no_raw_details(self):
        from lib_python_worktree.core._exceptions import WorktreeDirLockedError

        err = WorktreeDirLockedError("wt-2", killed=[], kill_attempted=False)

        assert err.kill_attempted is False
        assert err.killed == []
        msg = str(err)
        assert "kill_blocking_processes=True" in msg
        assert "wt-2" in msg
        # No raw stderr, path, or exit-code leakage in the remedy message.
        assert "Permission denied" not in msg
        assert "Invalid argument" not in msg
        assert "255" not in msg
        assert "128" not in msg

    def test_kill_attempted_defaults_to_true(self):
        """Default kill_attempted=True preserves the pre-#72 message wording
        for any existing call site that doesn't pass the new kwarg."""
        from lib_python_worktree.core._exceptions import WorktreeDirLockedError

        err = WorktreeDirLockedError("wt-3", killed=[])

        assert err.kill_attempted is True
        assert "after killing 0" in str(err)


class TestKillBlockingFlagOff:
    """When kill_blocking_processes=False (the default), behaviour is unchanged."""

    def test_flag_off_rc1_still_raises_git_command_error(self, tmp_path):
        """Default (flag=False): rc=1 raises GitCommandError, kill not called."""
        from lib_python_worktree.core.manager import GitCommandError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-flagoff-posix")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _git_rc1(args, cwd=None, **kwargs):
            return MagicMock(returncode=1, stderr="some error")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_rc1),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill,
        ):
            with pytest.raises(GitCommandError):
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_not_called()

    def test_remove_default_flag_off(self, tmp_path):
        """remove() default call (no kill_blocking_processes) on exit
        255/'Permission denied' — a lock signal — raises WorktreeDirLockedError
        with the kill_blocking_processes=True remedy (ticket #72, Befund 2),
        not a raw GitCommandError, confirming remove()'s default plumbing
        reaches the same classification as _teardown() directly."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-remove-default", branch_created_by_us=False)
        manager.state.add(record)

        def _git_perm_denied(args, cwd=None, **kwargs):
            return MagicMock(returncode=255, stderr="Permission denied")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_perm_denied),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager.remove(record.id)

        assert excinfo.value.kill_attempted is False


class TestKillBlockingRecordKilledPids:
    """Verify record.killed_pids is populated and returned by remove()."""

    def test_remove_returns_record_with_killed_pids(self, tmp_path):
        """remove(kill_blocking_processes=True) returns record.killed_pids on success."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-ret-killed",
            path="/fake/store/wt-ret-killed",
            branch_created_by_us=False,
        )
        manager.state.add(record)

        fake_killed = [KilledProcessInfo(pid=1111, name="node", cmdline=["node", "server.js"])]

        call_count = {"n": 0}

        def _git_side_effect(args, cwd=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1 and "remove" in args:
                return MagicMock(returncode=255, stderr="Permission denied")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=[],
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            removed = manager.remove(record.id, kill_blocking_processes=True)

        assert removed.killed_pids == fake_killed
        # Fix-pass 2 (ticket #127 blocking finding): the checkout here is
        # actually present (this record's directory was never externally
        # deleted), so Gate A's Windows pre-flight blocking-process scan
        # must genuinely run -- not be silently skipped because
        # ``_teardown``'s own ``target_absent`` probe was miscomputed as
        # ``True`` by a buggy test fixture. This pins that regression: it
        # failed to hold before the fixture's double-probe bug was fixed
        # (Gate A was skipped even though this test is semantically about a
        # present checkout).
        # Ticket #140: the new all-platform orphan-scan phase (index 6) also
        # calls _find_blocking_processes on this same present target, AFTER
        # Gate A -- so the total is now 2 calls, not 1. Both calls use the
        # same (record.path, <our own pid>) shape.
        assert mock_find.call_count == 2
        mock_find.assert_any_call(record.path, ANY)

    def test_yaml_store_remove_returns_killed_pids(self, tmp_path):
        """Regression for blocking #2: YamlStateStore.remove() returns a freshly
        deserialized object; killed_pids must be explicitly copied onto it so
        the caller sees the list even when using the file-backed store."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo
        from lib_python_worktree.core.yaml_store import YamlStateStore

        yaml_store = YamlStateStore(state_dir=tmp_path / "state")
        manager = WorktreeManager(
            config=ManagerConfig(store_root=tmp_path / "store"),
            state=yaml_store,
            reconcile_on_init=False,
        )

        record = _make_record(
            "wt-yaml-killed",
            path="/fake/store/wt-yaml-killed",
            branch_created_by_us=False,
        )
        yaml_store.add(record)

        fake_killed = [KilledProcessInfo(pid=2222, name="node", cmdline=["node"])]

        call_count = {"n": 0}

        def _git_side_effect(args, cwd=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1 and "remove" in args:
                return MagicMock(returncode=255, stderr="Permission denied")
            return MagicMock(returncode=0, stderr="")

        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=[],
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            removed = manager.remove(record.id, kill_blocking_processes=True)

        # The critical assertion: YamlStateStore deserializes a fresh object,
        # so without the explicit copy in remove() this would be [].
        assert removed.killed_pids == fake_killed, (
            "killed_pids must survive YamlStateStore round-trip via remove()"
        )
        # Fix-pass 2 (ticket #127 blocking finding): same regression pin as
        # test_remove_returns_record_with_killed_pids above -- this
        # record's checkout is present, so Gate A's pre-flight
        # blocking-process scan must genuinely run rather than being
        # silently skipped by a miscomputed ``target_absent``.
        # Ticket #140: plus the new orphan-scan phase's own call -- see the
        # sibling test's comment above for the full rationale.
        assert mock_find.call_count == 2
        mock_find.assert_any_call(record.path, ANY)


# ---------------------------------------------------------------------------
# TestTeardownContractStopSteps -- ticket #31 gap 1
# ---------------------------------------------------------------------------

class TestTeardownContractStopSteps:
    """Verify that contract stop: steps are run inside _teardown before
    kill_blocking_processes, and that failures are swallowed."""

    def test_stop_steps_run_before_teardown_and_before_kill(self, tmp_path):
        """Regression #31: when a contract has stop: steps, SetupRunner.run is
        called with setup=contract.stop before _kill_blocking_processes.

        Sequence verified:
          1. SetupRunner.run(setup=contract.stop, ...)
          2. git worktree remove  → returns 255/'Permission denied'
          3. _kill_blocking_processes
          4. git worktree remove  → returns 0
        """
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-stop-steps",
            path="/fake/store/wt-stop-steps",
            ports={"web": 30001},
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            stop=[Step(run='echo stop', name="stop-svc")],
        )

        call_order: list[str] = []

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = lambda **kw: call_order.append("stop_runner")

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=5050, name="daemon", cmdline=["daemon"])]

        call_count = {"n": 0}

        def _git_side_effect(args, cwd=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MagicMock(returncode=255, stderr="Permission denied")
            return MagicMock(returncode=0, stderr="")

        def _mock_kill(path):
            call_order.append("kill")
            return fake_killed

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                side_effect=_mock_kill,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        # stop_runner must appear before kill
        assert "stop_runner" in call_order
        assert "kill" in call_order
        assert call_order.index("stop_runner") < call_order.index("kill"), (
            "contract stop: steps must run before _kill_blocking_processes"
        )
        # Verify port_mapping is forwarded
        runner_kw = mock_runner_instance.run.call_args_list[0][1]
        assert runner_kw["worktree_id"] == "wt-stop-steps"
        assert runner_kw["setup"] == fake_contract.stop
        assert runner_kw["port_mapping"] == {"web": 30001}

    def test_stop_steps_swallow_runner_exception(self, tmp_path):
        """SetupFailedError from SetupRunner.run must not propagate out of _teardown."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract
        from lib_python_worktree.setup.runner import SetupFailedError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-stop-swallow")
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            stop=[Step(run='exit 1', name="boom")],
        )

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = SetupFailedError(
            worktree_id="wt-stop-swallow",
            step_index=0,
            step_name="boom",
            log_path=Path("/tmp/fake.log"),
            returncode=1,
        )

        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            # Must not raise despite SetupFailedError from stop runner
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_stop_steps_skipped_when_no_stop_field(self, tmp_path):
        """When contract.stop is empty, SetupRunner is never constructed for stop:."""
        from lib_python_worktree.contract.schema import WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record("wt-no-stop-field")
        manager.state.add(record)

        # Contract with no stop steps
        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        runner_stop_calls: list = []

        mock_runner_instance = MagicMock()
        # Track whether run is ever called with a stop setup
        original_run = mock_runner_instance.run
        mock_runner_instance.run.side_effect = lambda **kw: runner_stop_calls.append(kw["setup"])

        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        # No call should have been made with setup=[] (stop field empty)
        for setup_arg in runner_stop_calls:
            assert setup_arg != [], (
                "SetupRunner.run must not be called with empty stop: list"
            )


# ---------------------------------------------------------------------------
# TestTicket130TeardownStopHookOutcome -- ticket #130
#
# Behavioural requirement: _teardown() must compute a StopHookOutcome for
# its own best-effort Step 1b (contract stop: hook) run, exactly like
# WorktreeManager.stop() already does for its own delegated/no-op paths,
# and assign it onto record.stop_hook_outcome BEFORE any later gate
# (Gate A / Gate B) can raise. Before this ticket, Step 1b's outcome was
# discarded via a bare `except Exception: pass`, so a force-removed,
# still-running environment always reported stop_hook_outcome: null.
# ---------------------------------------------------------------------------

class TestTicket130TeardownStopHookOutcome:
    def test_completed_stop_hook_reports_full_diagnostics(self, tmp_path, caplog):
        """Driving test: a contract with stop: steps that run successfully
        must leave record.stop_hook_outcome.status == "completed", with
        steps_run/message/contract diagnostics matching stop()'s own
        vocabulary, and no_op_reason always None on the teardown path."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract
        from lib_python_worktree.setup.runner import SetupResult, SetupStepResult

        manager = _make_manager(tmp_path)
        repo_root = tmp_path / "repo-130-completed"
        repo_root.mkdir()
        # A real on-disk contract file so contract_found's independent
        # filesystem probe (contract_path.exists()) agrees with the mocked
        # _load_contract() below -- the two must never diverge in this test.
        contract_dir = repo_root / ".seretos"
        contract_dir.mkdir()
        (contract_dir / "worktree-setup.yml").write_text(
            "version: 1\nisolation: full\nstop:\n  - run: one\n  - run: two\n",
            encoding="utf-8",
        )
        record = _make_record(
            "wt-130-completed",
            path="/fake/store/wt-130-completed",
            repo_root=str(repo_root),
        )
        manager.state.add(record)

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
        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)

        outcome = record.stop_hook_outcome
        assert outcome is not None
        assert outcome.status == "completed"
        assert outcome.steps_run == 2
        assert outcome.message == "stop: completed 2 step(s)"
        assert outcome.contract_found is True
        expected_path = (repo_root / ".seretos" / "worktree-setup.yml").as_posix()
        assert outcome.contract_path == expected_path
        assert outcome.contract_isolation == "full"
        assert outcome.no_op_reason is None
        # Ticket #130 scope decision: stop_attempt is intentionally NOT
        # recomputed by _teardown() -- StopAttempt is single-valued but
        # Step 1 above stops every role in a loop, so there is no
        # non-arbitrary single attempt to report. This is a documented
        # scope boundary, not a gap.
        assert record.stop_attempt is None

    def test_completed_stop_hook_reports_diagnostics_on_non_force_removal_too(
        self, tmp_path
    ):
        """Sibling of the driving test: force=False must populate
        stop_hook_outcome exactly the same way -- Step 1b runs on both
        force=True and force=False."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract
        from lib_python_worktree.setup.runner import SetupResult, SetupStepResult

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-130-completed-noforce", path="/fake/store/wt-130-completed-noforce"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1, isolation="full", stop=[Step(run="one")]
        )
        fake_result = SetupResult(
            worktree_id=record.id,
            steps=[SetupStepResult(index=0, name="one", returncode=0, log_path=tmp_path / "a.log")],
        )

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = fake_result
        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert record.stop_hook_outcome is not None
        assert record.stop_hook_outcome.status == "completed"
        assert record.stop_hook_outcome.steps_run == 1

    def test_empty_stop_steps_reports_skipped(self, tmp_path):
        """Contract loads fine but has no stop: steps -> status="skipped",
        steps_run == 0, matching stop()'s own "no stop: steps" message."""
        from lib_python_worktree.contract.schema import WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record("wt-130-skipped", path="/fake/store/wt-130-skipped")
        manager.state.add(record)

        fake_contract = WorktreeContract(version=1, isolation="full", stop=[])

        mock_lifecycle = MagicMock()
        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        outcome = record.stop_hook_outcome
        assert outcome is not None
        assert outcome.status == "skipped"
        assert outcome.steps_run == 0
        assert outcome.message == "no stop: steps in contract"

    def test_missing_contract_file_reports_not_found_and_implicit_isolation_none(
        self, tmp_path
    ):
        """No .seretos/worktree-setup.yml at all on a real repo_root ->
        contract_found is False, but the loader's implicit isolation: none
        contract still makes contract_isolation == "none" (real _load_contract,
        not mocked, so contract_found and the parsed isolation can never
        diverge -- mirrors test_manager.py's #128 no-contract-file test)."""
        manager = _make_manager(tmp_path)
        repo_root = tmp_path / "repo-130-no-contract"
        repo_root.mkdir()
        record = _make_record(
            "wt-130-no-contract",
            path="/fake/store/wt-130-no-contract",
            repo_root=str(repo_root),
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        with patch("lib_python_worktree.core.teardown._run_git") as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        outcome = record.stop_hook_outcome
        assert outcome is not None
        assert outcome.contract_found is False
        assert outcome.contract_isolation == "none"
        assert outcome.status == "skipped"

    def test_contract_exists_check_oserror_reports_not_found(self, tmp_path):
        """contract_path.exists() raising OSError on the independent
        contract_found probe -> contract_found is False, and _teardown()
        must not raise from that probe itself. _run_git is mocked (no real
        git subprocess spawned), so patching Path.exists here cannot
        interfere with subprocess.Popen the way it would in a real-git
        integration test."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        repo_root = tmp_path / "repo-130-exists-oserror"
        repo_root.mkdir()
        record = _make_record(
            "wt-130-exists-oserror",
            path="/fake/store/wt-130-exists-oserror",
            repo_root=str(repo_root),
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1, isolation="full", stop=[Step(run="one")]
        )
        contract_file = repo_root / ".seretos" / "worktree-setup.yml"
        real_exists = Path.exists

        def _flaky_exists(self, *a, **kw):
            if self == contract_file:
                raise OSError("boom")
            return real_exists(self, *a, **kw)

        mock_lifecycle = MagicMock()
        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("pathlib.Path.exists", new=_flaky_exists),
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            # Must not raise despite the contract_found probe raising OSError.
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert record.stop_hook_outcome is not None
        assert record.stop_hook_outcome.contract_found is False

    def test_load_contract_raising_reports_failed(self, tmp_path, caplog):
        """_load_contract() raising (unparseable contract) -> status="failed",
        steps_run == 0, message == str(exc), and _teardown() must still
        complete normally (no exception propagates from the hook itself).
        Also asserts the new warning names the worktree id and the failure
        message."""
        import logging

        manager = _make_manager(tmp_path)
        record = _make_record("wt-130-load-fail", path="/fake/store/wt-130-load-fail")
        manager.state.add(record)

        exc = ValueError("contract is not valid yaml")
        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._load_contract", side_effect=exc),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            caplog.at_level(logging.WARNING, logger="lib_python_worktree.core.manager"),
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            # Must not raise despite the contract failing to load.
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        outcome = record.stop_hook_outcome
        assert outcome is not None
        assert outcome.status == "failed"
        assert outcome.steps_run == 0
        assert outcome.message == str(exc)
        assert outcome.contract_isolation is None

        assert any(
            record.id in rec.message and str(exc) in rec.message
            for rec in caplog.records
        ), "expected a warning naming the worktree id and the failure message"

    def test_setup_runner_raising_reports_failed_and_warns(self, tmp_path, caplog):
        """Sibling of test_stop_steps_swallow_runner_exception in
        TestTeardownContractStopSteps: SetupRunner.run() raising must still
        report status="failed"/steps_run == 0/message == str(exc), AND
        (new for #130) emit a warning naming the worktree id and message."""
        import logging
        from lib_python_worktree.contract.schema import Step, WorktreeContract
        from lib_python_worktree.setup.runner import SetupFailedError

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-130-runner-fail", path="/fake/store/wt-130-runner-fail"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1, isolation="full", stop=[Step(run="exit 1", name="boom")]
        )
        exc = SetupFailedError(
            worktree_id=record.id,
            step_index=0,
            step_name="boom",
            log_path=tmp_path / "fail.log",
            returncode=1,
        )

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = exc
        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            caplog.at_level(logging.WARNING, logger="lib_python_worktree.core.manager"),
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        outcome = record.stop_hook_outcome
        assert outcome is not None
        assert outcome.status == "failed"
        assert outcome.steps_run == 0
        assert outcome.message == str(exc)

        assert any(
            record.id in rec.message and str(exc) in rec.message
            for rec in caplog.records
        ), "expected a warning naming the worktree id and the failure message"

    def test_hook_failure_does_not_mask_later_gate_error(self, tmp_path):
        """When Step 1b's hook fails AND a later gate subsequently raises
        WorktreeDirLockedError, the raised exception must be unchanged
        (same type as before this ticket's change), but
        record.stop_hook_outcome must still be populated on the
        in-place-mutated record -- proving the outcome assignment happens
        before that gate, not only on a clean-exit path. Reuses the exact
        lock-signal-then-permission-denied recipe from
        test_benign_dirt_retry_lock_signal_flag_off_raises_dir_locked
        (proven to reach the Final guard's WorktreeDirLockedError), adding
        a contract stop: step whose runner raises."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-130-gate-after-hook-fail", path="/fake/store/wt-130-gate-after-hook-fail"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1, isolation="full", stop=[Step(run="boom")]
        )

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = RuntimeError("stop hook exploded")

        mock_lifecycle = MagicMock()

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? .seretos/\0", stderr="")
            if _is_force_remove_call(args):
                return MagicMock(returncode=255, stderr="Permission denied")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch("lib_python_worktree.core.teardown.shutil"),
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError):
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_not_called()

        # The hook failure must still be recorded despite the later raise.
        assert record.stop_hook_outcome is not None
        assert record.stop_hook_outcome.status == "failed"
        assert record.stop_hook_outcome.message == "stop hook exploded"


# ---------------------------------------------------------------------------
# TestKillBlockingProcessesWindowsInvalidArg -- ticket #31 gap 2
# ---------------------------------------------------------------------------

class TestKillBlockingProcessesWindowsInvalidArg:
    """Windows path: rc=255 + 'Invalid argument' also triggers kill+retry."""

    def test_invalid_argument_triggers_kill_and_retry(self, tmp_path):
        """Regression #31 gap 2: 'Invalid argument' must trigger the same
        kill-and-retry path as 'Permission denied' on Windows."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record("wt-win-invarg", path="/fake/store/wt-win-invarg")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=7070, name="unity.exe", cmdline=["unity"])]

        call_count = {"n": 0}

        def _git_side_effect(args, cwd=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MagicMock(returncode=255, stderr="Invalid argument")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            # Must not raise — currently fails because "Invalid argument" was
            # not in the heuristic before this fix.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_called_once_with(record.path)
        assert record.killed_pids == fake_killed
        assert call_count["n"] == 2

    def test_permission_denied_still_triggers_kill(self, tmp_path):
        """'Permission denied' must still trigger kill+retry after the heuristic
        change (regression guard)."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record("wt-win-perm-guard", path="/fake/store/wt-win-perm-guard")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=8080, name="code.exe", cmdline=["code"])]

        call_count = {"n": 0}

        def _git_side_effect(args, cwd=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return MagicMock(returncode=255, stderr="Permission denied")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_called_once_with(record.path)
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# TestLongPathFallback -- ticket #31 gap 3
# ---------------------------------------------------------------------------

class TestLongPathFallback:
    """Verify the long-path post-delete fallback in _teardown."""

    def test_directory_still_exists_after_git_remove_triggers_longpath_deletion(
        self, tmp_path
    ):
        """Regression #31 gap 3: when git worktree remove returns 0 but the
        directory still exists on win32, shutil.rmtree is called with the
        extended-length path prefix."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-longpath-win",
            path="C:\\fake\\store\\wt-longpath-win",
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        rmtree_calls: list = []

        # In Python 3.14, pathlib.Path.exists() calls os.path.exists() directly.
        # _teardown calls _load_contract twice (for stop: and teardown: steps),
        # each of which calls Path.exists() on the contract file — those are
        # filtered out below since they don't match record.path.
        # Ticket #135: the target_absent probe now goes through the separately
        # mockable teardown._target_is_absent seam, patched by the autouse
        # _target_present_by_default fixture (return_value=False, i.e. present)
        # — it no longer calls os.path.exists at all. So this mock only needs
        # to cover the two remaining real os.path.exists(record.path) calls:
        # call #1 is the long-path-fallback check at line 897 (True, dir still
        # present after git remove), call #2 is the final guard after rmtree
        # succeeds (False).
        _path_calls = {"n": 0}

        def _mock_exists(path):
            if str(path) == record.path:
                _path_calls["n"] += 1
                return _path_calls["n"] <= 1  # True for the first call only
            # Contract file / other paths → not present.
            return False

        def _mock_rmtree(path, **kwargs):
            rmtree_calls.append(path)

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.os.path.exists", side_effect=_mock_exists),
            patch("lib_python_worktree.core.teardown.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        # The extended-length path must have been passed to shutil.rmtree
        extended = "\\\\\\\\" + "?\\\\" + os.path.abspath(record.path)
        # Accept any call that starts with \\?\
        extended_calls = [c for c in rmtree_calls if c.startswith("\\\\?\\")]
        assert extended_calls, (
            f"Expected shutil.rmtree call with \\\\?\\ prefix, got: {rmtree_calls}"
        )

    def test_longpath_fallback_skipped_on_posix(self, tmp_path):
        r"""On non-Windows, the \\?\-prefixed rmtree variant must never be called."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-longpath-posix", path="/fake/store/wt-longpath-posix")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        rmtree_calls: list = []

        # See explanation in test_directory_still_exists_after_git_remove_triggers_longpath_deletion.
        # Ticket #135: the target_absent probe now goes through the separately
        # mockable teardown._target_is_absent seam (patched by the autouse
        # _target_present_by_default fixture) and no longer touches
        # os.path.exists. This mock only needs to cover the remaining two real
        # os.path.exists(record.path) calls: call #1 is the long-path-fallback
        # check (True), and False for all other paths and all subsequent
        # checks of record.path (the final guard).
        _path_calls = {"n": 0}

        def _mock_exists(path):
            if str(path) == record.path:
                _path_calls["n"] += 1
                return _path_calls["n"] <= 1
            return False

        def _mock_rmtree(path, **kwargs):
            rmtree_calls.append(path)

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.os.path.exists", side_effect=_mock_exists),
            patch("lib_python_worktree.core.teardown.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "linux"
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        # Must not use \\?\ prefix on POSIX
        extended_calls = [c for c in rmtree_calls if c.startswith("\\\\?\\")]
        assert not extended_calls, (
            f"\\\\?\\ prefix must not be used on POSIX, got: {rmtree_calls}"
        )
        # But a plain rmtree must still be called
        assert rmtree_calls, "shutil.rmtree must be called on POSIX fallback"

    def test_robocopy_fallback_used_when_first_rmtree_fails(self, tmp_path):
        """When the extended-path shutil.rmtree raises OSError, robocopy is
        attempted as the second fallback."""
        import subprocess as subprocess_module

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-robocopy",
            path="C:\\fake\\store\\wt-robocopy",
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        robocopy_calls: list = []

        def _mock_rmtree(path, **kwargs):
            if path.startswith("\\\\?\\"):
                raise OSError("path too long")
            # Second call (after robocopy) succeeds silently

        def _mock_subprocess_run(cmd, **kwargs):
            robocopy_calls.append(cmd)
            return MagicMock(returncode=1)  # robocopy exits 1 on success-with-copies

        # See explanation in test_directory_still_exists_after_git_remove_triggers_longpath_deletion.
        # Ticket #135: the target_absent probe now goes through the separately
        # mockable teardown._target_is_absent seam (patched by the autouse
        # _target_present_by_default fixture) and no longer touches
        # os.path.exists. This mock only needs to cover the remaining two real
        # os.path.exists(record.path) calls: call #1 is the long-path-fallback
        # check (True), and False for all other paths and all subsequent
        # checks of record.path (the final guard).
        _path_calls = {"n": 0}

        def _mock_exists(path):
            if str(path) == record.path:
                _path_calls["n"] += 1
                return _path_calls["n"] <= 1
            return False

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.os.path.exists", side_effect=_mock_exists),
            patch("lib_python_worktree.core.teardown.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.teardown.subprocess.run", side_effect=_mock_subprocess_run),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            # Must not raise even if robocopy path is taken
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert robocopy_calls, "robocopy must be called when extended-path rmtree fails"
        assert robocopy_calls[0][0] == "robocopy", (
            f"first element of robocopy cmd must be 'robocopy', got {robocopy_calls[0]}"
        )
        assert record.path in robocopy_calls[0], (
            "record.path must be in robocopy args"
        )

    def test_longpath_fallback_no_rmtree_when_dir_gone(self, tmp_path):
        """When git worktree remove succeeds and the directory is gone,
        no fallback rmtree is called."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-no-fallback", path="/fake/store/wt-no-fallback")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        rmtree_calls: list = []

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.os.path.exists", return_value=False),
            patch("lib_python_worktree.core.teardown.shutil.rmtree", side_effect=lambda *a, **kw: rmtree_calls.append(a)),
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert not rmtree_calls, "shutil.rmtree must not be called when dir is already gone"


# ---------------------------------------------------------------------------
# TestKillRetryLoop -- ticket #51 (post-kill bounded retry loop)
# ---------------------------------------------------------------------------

class TestKillRetryLoop:
    """Verify the bounded post-kill retry loop introduced for ticket #51."""

    def test_kill_retry_loop_succeeds_after_multiple_attempts(self, tmp_path):
        """Kill fires once; the first 4 post-kill retries return 255/'Permission
        denied'; the 5th retry returns 0.  Assert: no exception raised; total
        _run_git calls == 6 (1 initial + 5 retries); time.sleep called 4 times;
        record.killed_pids is set."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record("wt-retry-loop", path="/fake/store/wt-retry-loop")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=4321, name="node.exe", cmdline=["node"])]

        call_count = {"n": 0}

        def _git_side_effect(args, cwd=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Initial call — returns lock signal (triggers kill + retry loop).
                return MagicMock(returncode=255, stderr="Permission denied")
            elif call_count["n"] <= 5:
                # Post-kill retries 1–4 still fail.
                return MagicMock(returncode=255, stderr="Permission denied")
            else:
                # 5th post-kill retry (6th total call) succeeds.
                return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.time") as mock_time,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            # Must not raise.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_called_once_with(record.path)
        assert record.killed_pids == fake_killed
        # 1 initial + 5 post-kill retry calls.
        assert call_count["n"] == 6, (
            f"Expected 6 total _run_git calls, got {call_count['n']}"
        )
        # sleep called between retries: 4 times (not after the last successful attempt).
        assert mock_time.sleep.call_count == 4, (
            f"Expected 4 time.sleep calls, got {mock_time.sleep.call_count}"
        )

    def test_kill_retry_phantom_state_mid_loop(self, tmp_path):
        """Combined path (finding 3): lock-signal on initial call → kill →
        a retry returns exit 128 'is not a working tree' → no
        WorktreeDirLockedError raised; phantom-state cleanup (rmtree +
        worktree prune) runs; teardown completes (ports released).

        Sequence:
          call 1: returncode=255 / 'Permission denied'  (triggers kill + loop)
          call 2 (retry 1): returncode=128 / 'is not a working tree'
            → phantom cleanup fires; loop exits; no raise.
        """
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-mid-phantom", path="/fake/store/wt-mid-phantom")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=9876, name="code.exe", cmdline=["code"])]
        mock_allocator = MagicMock()
        manager._allocator = mock_allocator

        call_count = {"n": 0}

        def _git_side_effect(args, cwd=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Initial attempt — lock signal.
                return MagicMock(returncode=255, stderr="Permission denied")
            if "remove" in args:
                # First retry — git has now deregistered the worktree.
                return MagicMock(
                    returncode=128,
                    stderr="fatal: '/fake/store/wt-mid-phantom' is not a working tree",
                )
            # worktree prune (and any other git call) succeeds.
            return MagicMock(returncode=0, stderr="")

        rmtree_calls: list = []

        def _mock_rmtree(path, **kwargs):
            rmtree_calls.append((path, kwargs))

        git_calls: list = []

        def _tracking_git(args, cwd=None, **kwargs):
            git_calls.append(list(args))
            return _git_side_effect(args, cwd=cwd, **kwargs)

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_tracking_git),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.teardown.time"),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            # Must NOT raise WorktreeDirLockedError.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        # kill was called once.
        mock_kill.assert_called_once_with(record.path)
        assert record.killed_pids == fake_killed

        # shutil.rmtree called with record.path and ignore_errors=True
        # (phantom-state cleanup).
        assert any(
            c[0] == record.path and c[1].get("ignore_errors") is True
            for c in rmtree_calls
        ), f"Expected rmtree({record.path!r}, ignore_errors=True), got {rmtree_calls}"

        # git worktree prune was called on the repo root.
        prune_calls = [a for a in git_calls if a[:2] == ["worktree", "prune"]]
        assert prune_calls, "git worktree prune must be called during phantom cleanup"

        # Port allocator must still release the worktree id.
        mock_allocator.release.assert_called_once_with(record.id)


# ---------------------------------------------------------------------------
# TestTeardownAlreadyDeregistered -- ticket #51 (phantom-state fix)
# ---------------------------------------------------------------------------

class TestTeardownAlreadyDeregistered:
    """Regression tests for the phantom-state scenario described in ticket #51.

    When git has already deregistered a worktree (it returns exit 128 with
    'is not a working tree' in stderr), _teardown must NOT raise; instead it
    must clean up the leftover directory, prune stale git metadata, and release
    ports — so the caller can complete the removal cycle.
    """

    def test_already_deregistered_force_false_does_not_raise(self, tmp_path):
        """git exits 128 + 'is not a working tree', force=False: no exception;
        shutil.rmtree called once with (record.path, ignore_errors=True);
        git worktree prune called; port allocator .release called with record.id."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-phantom", path="/fake/store/wt-phantom")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        mock_allocator = MagicMock()
        manager._allocator = mock_allocator

        git_calls: list = []

        def _git_side_effect(args, cwd=None, **kwargs):
            git_calls.append(list(args))
            if "remove" in args:
                return MagicMock(
                    returncode=128,
                    stderr="fatal: '/fake/store/wt-phantom' is not a working tree",
                )
            # worktree prune succeeds.
            return MagicMock(returncode=0, stderr="")

        rmtree_calls: list = []

        def _mock_rmtree(path, **kwargs):
            rmtree_calls.append((path, kwargs))

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch("lib_python_worktree.core.teardown.shutil.rmtree", side_effect=_mock_rmtree),
        ):
            # Must not raise.
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        # shutil.rmtree called once with record.path and ignore_errors=True.
        assert len(rmtree_calls) == 1, (
            f"Expected 1 shutil.rmtree call, got {len(rmtree_calls)}: {rmtree_calls}"
        )
        assert rmtree_calls[0][0] == record.path, (
            f"rmtree path mismatch: {rmtree_calls[0][0]!r} != {record.path!r}"
        )
        assert rmtree_calls[0][1].get("ignore_errors") is True, (
            "rmtree must be called with ignore_errors=True"
        )

        # git worktree prune called on repo root.
        prune_calls = [a for a in git_calls if a[:2] == ["worktree", "prune"]]
        assert prune_calls, "git worktree prune must be called"

        # Port allocator must release the worktree id.
        mock_allocator.release.assert_called_once_with(record.id)

    def test_already_deregistered_second_remove_completes(self, tmp_path):
        """Regression for ticket #51: full manager.remove(record.id, force=False)
        where teardown receives 'is not a working tree' from git.
        After remove() returns, manager.state.list() must be empty."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-phantom-state",
            branch_created_by_us=False,  # skip branch-delete step
        )
        manager.state.add(record)

        # Verify the record is tracked before removal.
        assert len(manager.state.list()) == 1

        mock_lifecycle = MagicMock()

        def _git_side_effect(args, cwd=None, **kwargs):
            if "remove" in args:
                return MagicMock(
                    returncode=128,
                    stderr="fatal: '{}' is not a working tree".format(record.path),
                )
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_git_side_effect),
            patch("lib_python_worktree.core.teardown.shutil"),
            patch.object(
                manager, "_teardown", wraps=lambda rec, force, **kw: (
                    WorktreeManager._teardown(
                        manager, rec, force=force, _lifecycle_module=mock_lifecycle
                    )
                )
            ),
        ):
            manager.remove(record.id, force=False)

        # The critical regression assertion: state must be empty after remove().
        assert manager.state.list() == [], (
            "state must be empty after remove() on a phantom (already-deregistered) worktree"
        )


# ---------------------------------------------------------------------------
# TestTeardownAbsentTarget -- ticket #127 (orphaned-record cleanup without force)
# ---------------------------------------------------------------------------

class TestTeardownAbsentTarget:
    """Ticket #127: a tracked, ``status="orphaned"`` record whose checkout
    directory was deleted externally must be removable via
    ``remove(worktree_id=...)`` WITHOUT ``force``, without leaking its port
    reservations, and without an unmerged owned branch turning an otherwise
    successful cleanup into a raised exception.

    Root cause: for such a record, ``git worktree remove`` (no ``--force``)
    exits 128 with stderr matching neither the phantom-state phrase ("is not
    a working tree") nor a dirty-tree phrase -- so it used to fall through
    to a bare ``GitCommandError``, and the caller was forced to retry with
    ``force=True`` just to clean up a directory that was already gone.
    """

    def test_absent_target_exit128_without_force_does_not_raise(self, tmp_path):
        """Driving test: exit 128 + non-matching stderr + force=False, but
        the checkout directory is genuinely absent -- _teardown must not
        raise; it must fall back to shutil.rmtree + git worktree prune, then
        release ports, exactly like the force=True fallback does today."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-absent-128", path="/fake/store/wt-absent-128")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        mock_allocator = MagicMock()
        manager._allocator = mock_allocator

        git_calls: list = []
        rmtree_calls: list = []

        def _side_effect(args, cwd=None, **kwargs):
            git_calls.append(list(args))
            if args[:2] == ["worktree", "remove"]:
                # Neither the phantom-state phrase nor a dirty-tree phrase --
                # the exact shape a `prunable` porcelain block with an
                # absent checkout directory produces.
                return MagicMock(returncode=128, stderr="fatal: unable to stat")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_side_effect),
            patch("lib_python_worktree.core.teardown._target_is_absent", return_value=True),
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=lambda *a, **kw: rmtree_calls.append((a, kw)),
            ),
        ):
            # Must not raise.
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert any(
            a == (record.path,) and kw.get("ignore_errors") is True
            for a, kw in rmtree_calls
        ), f"expected rmtree(record.path, ignore_errors=True), got {rmtree_calls}"
        prune_calls = [a for a in git_calls if a[:2] == ["worktree", "prune"]]
        assert prune_calls, "git worktree prune must be called"
        mock_allocator.release.assert_called_once_with(record.id)

    def test_absent_target_exit128_without_force_full_remove_returns_removed(
        self, tmp_path
    ):
        """Full remove(worktree_id=..., force=False) on an orphaned record:
        the state record must be deleted and the returned record must carry
        status="removed", without ever needing force=True."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-absent-full-remove",
            path="/fake/store/wt-absent-full-remove",
            branch_created_by_us=False,
        )
        manager.state.add(record)
        assert len(manager.state.list()) == 1

        mock_allocator = MagicMock()
        manager._allocator = mock_allocator

        def _side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                return MagicMock(returncode=128, stderr="fatal: unable to stat")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_side_effect),
            patch("lib_python_worktree.core.teardown._target_is_absent", return_value=True),
            patch("lib_python_worktree.core.teardown.shutil.rmtree"),
        ):
            removed = manager.remove(record.id, force=False)

        assert removed.status == "removed"
        assert manager.state.list() == [], "state must be empty after remove()"
        mock_allocator.release.assert_called_once_with(record.id)

    def test_present_target_exit128_without_force_still_raises_git_error(
        self, tmp_path
    ):
        """Over-widening guard: the SAME exit 128 + non-matching stderr, but
        the checkout directory is actually present -- must still raise
        GitCommandError exactly as before ticket #127. This pins the
        existing test_exit128_without_force_non_dirty_stderr_raises_git_error
        behaviour so the widened fallback can never accidentally swallow a
        real, unexplained git failure on a checkout that still exists."""
        from lib_python_worktree.core.manager import GitCommandError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-present-128", path="/fake/store/wt-present-128")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                return MagicMock(returncode=128, stderr="fatal: unable to stat")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_side_effect),
            patch("lib_python_worktree.core.teardown._target_is_absent", return_value=False),
        ):
            with pytest.raises(GitCommandError):
                manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_absent_target_skips_gate_a_preflight_scan(self, tmp_path):
        """Gate A short-circuit: on win32, with the checkout absent, the
        Windows blocking-process pre-flight scan must never run -- there is
        nothing on disk that could hold a file handle."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-absent-gatea", path="/fake/store/wt-absent-gatea")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes"
            ) as mock_find,
            patch("lib_python_worktree.core.teardown._target_is_absent", return_value=True),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        mock_find.assert_not_called()

    def test_present_target_still_runs_gate_a_preflight_scan(self, tmp_path):
        """Companion to the short-circuit test: when the checkout IS
        present, Gate A's pre-flight scan still runs exactly as before."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-present-gatea", path="/fake/store/wt-present-gatea")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=[],
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        # Ticket #140: the new all-platform orphan-scan phase (index 6) also
        # calls _find_blocking_processes on this same present target, AFTER
        # Gate A -- so the total is now 2 calls, not 1.
        assert mock_find.call_count == 2
        mock_find.assert_any_call(record.path, os.getpid())

    def test_absent_target_phantom_state_stderr_still_routes_to_phantom_cleanup(
        self, tmp_path
    ):
        """Ordering guard: 'is not a working tree' stderr must still route
        through _phantom_state_cleanup() first, regardless of target_absent
        -- the widened exit-128 fallback branch must never shadow the
        earlier phantom-state branch (ticket #51)."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-absent-phantom", path="/fake/store/wt-absent-phantom")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        mock_allocator = MagicMock()
        manager._allocator = mock_allocator

        rmtree_calls: list = []

        def _side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                return MagicMock(
                    returncode=128,
                    stderr=f"fatal: '{record.path}' is not a working tree",
                )
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_side_effect),
            patch("lib_python_worktree.core.teardown._target_is_absent", return_value=True),
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=lambda *a, **kw: rmtree_calls.append((a, kw)),
            ),
        ):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        # Exactly one rmtree call from _phantom_state_cleanup -- the widened
        # elif branch below it must never also fire for the same attempt.
        assert len(rmtree_calls) == 1, rmtree_calls
        mock_allocator.release.assert_called_once_with(record.id)

    def test_primary_checkout_refused_even_with_absent_target(self, tmp_path):
        """Primary-checkout refusal (ticket #84) still wins over the new
        absent-target path -- force=True never bypasses it, and neither does
        an absent target."""
        from lib_python_worktree.core.manager import PrimaryCheckoutError

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-primary-absent",
            path="/fake/repo",
            repo_root="/fake/repo",
            backing="primary",
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with patch("lib_python_worktree.core.teardown._target_is_absent", return_value=True):
            with pytest.raises(PrimaryCheckoutError):
                manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_absent_target_unmerged_owned_branch_warns_and_still_returns_removed(
        self, tmp_path, caplog
    ):
        """Driving test: an owned branch that is unmerged must not turn an
        otherwise-successful orphaned-record removal into a raised
        exception. remove() must still return status="removed", the state
        record must be gone, and a WARNING naming the branch must be
        logged."""
        import logging as _logging

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-absent-unmerged",
            path="/fake/store/wt-absent-unmerged",
            branch="feature/unmerged",
            branch_created_by_us=True,
        )
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                return MagicMock(returncode=128, stderr="fatal: unable to stat")
            if args[:1] == ["rev-parse"]:
                return MagicMock(returncode=0, stderr="")  # branch exists
            if args[:2] == ["branch", "-d"]:
                return MagicMock(
                    returncode=1,
                    stderr="error: branch 'feature/unmerged' is not fully merged",
                )
            return MagicMock(returncode=0, stderr="")

        caplog.set_level(_logging.WARNING, logger="lib_python_worktree.core.manager")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_side_effect),
            patch(
                "lib_python_worktree.core.manager._run_git",
                side_effect=_side_effect,
            ),  # ticket #135: _delete_owned_branch still calls
                # manager._run_git directly, so it needs its own patch
                # alongside the teardown-side one above.
            patch("lib_python_worktree.core.teardown._target_is_absent", return_value=True),
            patch("lib_python_worktree.core.teardown.shutil.rmtree"),
        ):
            removed = manager.remove(record.id, force=False)

        assert removed.status == "removed"
        assert manager.state.list() == []
        warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert any("feature/unmerged" in r.message for r in warnings), (
            f"expected a WARNING naming the unmerged branch, got: "
            f"{[r.message for r in warnings]}"
        )

    def test_present_target_unmerged_owned_branch_still_raises(self, tmp_path):
        """Over-widening guard: the SAME unmerged-branch refusal, but the
        checkout was actually present (a normal, non-orphaned removal) --
        must still raise GitCommandError exactly as before ticket #127."""
        from lib_python_worktree.core.manager import GitCommandError

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-present-unmerged",
            path="/fake/store/wt-present-unmerged",
            branch="feature/unmerged2",
            branch_created_by_us=True,
        )
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                return MagicMock(returncode=0, stderr="")
            if args[:1] == ["rev-parse"]:
                return MagicMock(returncode=0, stderr="")  # branch exists
            if args[:2] == ["branch", "-d"]:
                return MagicMock(
                    returncode=1,
                    stderr="error: branch 'feature/unmerged2' is not fully merged",
                )
            return MagicMock(returncode=0, stderr="")

        # No os.path.exists patch here: the module default fixture forces
        # BOTH remove()'s own target_absent probe AND _teardown()'s own
        # target_absent probe to read True -- present -- while every other
        # call (long-path fallback, Final guard) falls through to the real
        # (always-False, since this is a fake path) filesystem, exactly
        # like the pre-existing happy-path tests in this file (ticket #127
        # fix-pass 2: fixed a prior double-probe bug where only the first
        # of these two probes was forced, silently leaving _teardown()'s
        # own target_absent computed as True here despite this test being
        # about a present checkout).
        #
        # This is pinned concretely, not just asserted by comment: Gate A's
        # Windows pre-flight only runs when _teardown() computes
        # target_absent=False (`sys.platform == "win32" and not
        # target_absent`), so asserting _find_blocking_processes was
        # actually called proves _teardown() genuinely saw the checkout as
        # present.
        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_side_effect),
            patch(
                "lib_python_worktree.core.manager._run_git",
                side_effect=_side_effect,
            ),  # ticket #135: _delete_owned_branch still calls
                # manager._run_git directly, so it needs its own patch
                # alongside the teardown-side one above.
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=[],
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(GitCommandError):
                manager.remove(record.id, force=False)

        # Ticket #140: the new all-platform orphan-scan phase (index 6) also
        # calls _find_blocking_processes on this same present target, AFTER
        # Gate A -- so the total is now 2 calls, not 1.
        assert mock_find.call_count == 2
        mock_find.assert_any_call(record.path, ANY)

    def test_absent_target_branch_not_created_by_us_no_branch_call(self, tmp_path):
        """branch_created_by_us=False: no git branch/rev-parse call is ever
        made, even on the orphaned-record path."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-absent-notowned",
            path="/fake/store/wt-absent-notowned",
            branch="feature/notowned",
            branch_created_by_us=False,
        )
        manager.state.add(record)

        branch_calls: list = []

        def _side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                return MagicMock(returncode=128, stderr="fatal: unable to stat")
            if args and args[0] in ("branch", "rev-parse"):
                branch_calls.append(list(args))
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_side_effect),
            patch("lib_python_worktree.core.teardown._target_is_absent", return_value=True),
            patch("lib_python_worktree.core.teardown.shutil.rmtree"),
        ):
            removed = manager.remove(record.id, force=False)

        assert removed.status == "removed"
        assert branch_calls == [], f"expected no branch-related git calls, got {branch_calls}"

    def test_absent_target_branch_already_gone_no_warning(self, tmp_path, caplog):
        """Branch already gone (git rev-parse --verify fails): remove()
        completes with status="removed" and no warning is logged -- there is
        nothing to warn about."""
        import logging as _logging

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-absent-branchgone",
            path="/fake/store/wt-absent-branchgone",
            branch="feature/gone",
            branch_created_by_us=True,
        )
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                return MagicMock(returncode=128, stderr="fatal: unable to stat")
            if args[:1] == ["rev-parse"]:
                return MagicMock(returncode=1, stderr="")  # branch already gone
            return MagicMock(returncode=0, stderr="")

        caplog.set_level(_logging.WARNING, logger="lib_python_worktree.core.manager")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_side_effect),
            patch(
                "lib_python_worktree.core.manager._run_git",
                side_effect=_side_effect,
            ),  # ticket #135: _delete_owned_branch still calls
                # manager._run_git directly, so it needs its own patch
                # alongside the teardown-side one above.
            patch("lib_python_worktree.core.teardown._target_is_absent", return_value=True),
            patch("lib_python_worktree.core.teardown.shutil.rmtree"),
        ):
            removed = manager.remove(record.id, force=False)

        assert removed.status == "removed"
        warnings = [r for r in caplog.records if r.levelno == _logging.WARNING]
        assert not any("feature/gone" in r.message for r in warnings), (
            f"expected no warning, got: {[r.message for r in warnings]}"
        )

    def test_absent_target_non_unmerged_branch_delete_error_still_raises(
        self, tmp_path
    ):
        """Narrow-scope guard: on the orphaned-record path (target_absent
        =True), a `git branch -d` failure that is NOT the unmerged-branch
        refusal (e.g. the branch is checked out elsewhere, or lockfile
        contention) must still raise GitCommandError. #127's tolerance is
        specifically for an unmerged owned branch -- it must never widen
        into swallowing an arbitrary git error just because the checkout
        directory happened to be absent."""
        from lib_python_worktree.core.manager import GitCommandError

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-absent-checkedout",
            path="/fake/store/wt-absent-checkedout",
            branch="feature/checkedout",
            branch_created_by_us=True,
        )
        manager.state.add(record)

        def _side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                return MagicMock(returncode=128, stderr="fatal: unable to stat")
            if args[:1] == ["rev-parse"]:
                return MagicMock(returncode=0, stderr="")  # branch exists
            if args[:2] == ["branch", "-d"]:
                return MagicMock(
                    returncode=1,
                    stderr=(
                        "error: Cannot delete branch 'feature/checkedout' "
                        "checked out at '/some/other/worktree'"
                    ),
                )
            return MagicMock(returncode=0, stderr="")

        with (
            patch("lib_python_worktree.core.teardown._run_git", side_effect=_side_effect),
            patch(
                "lib_python_worktree.core.manager._run_git",
                side_effect=_side_effect,
            ),  # ticket #135: _delete_owned_branch still calls
                # manager._run_git directly, so it needs its own patch
                # alongside the teardown-side one above.
            patch("lib_python_worktree.core.teardown._target_is_absent", return_value=True),
            patch("lib_python_worktree.core.teardown.shutil.rmtree"),
        ):
            with pytest.raises(GitCommandError):
                manager.remove(record.id, force=False)


# ---------------------------------------------------------------------------
# TestLongPathFallbackLockedGuard -- ticket #57 regression tests
# ---------------------------------------------------------------------------

class TestLongPathFallbackLockedGuard:
    """Regression tests for ticket #57: the final guard that raises
    WorktreeDirLockedError when the directory is still present after all
    deletion attempts.

    Root cause: _teardown's long-path fallback block swallowed OSError
    silently and fell through to port release + status 'removed' regardless
    of whether the directory was actually deleted.
    """

    def test_directory_gone_after_fallback_succeeds(self, tmp_path):
        """When git exits 0 and os.path.exists returns False (directory gone),
        _teardown must complete without raising and port release must be called."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-gone-after-fallback", path="/fake/store/wt-gone")
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        mock_allocator = MagicMock()
        manager._allocator = mock_allocator

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.os.path.exists", return_value=False),
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            # Must not raise.
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        # Port release must still be called when directory is gone.
        mock_allocator.release.assert_called_once_with(record.id)

    def test_directory_still_exists_after_fallback_raises(self, tmp_path):
        """Regression #57: when git exits 0 but the directory still exists
        (long-path fallback failed silently), _teardown must raise
        WorktreeDirLockedError instead of returning a false 'removed' status."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record("wt-still-locked", path="/fake/store/wt-still-locked")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown.os.path.exists",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=OSError("path too long / locked"),
            ),
            patch("lib_python_worktree.core.teardown.subprocess.run", return_value=MagicMock(returncode=1)),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as exc_info:
                manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        err = exc_info.value
        assert err.worktree_id == "wt-still-locked", (
            f"WorktreeDirLockedError must carry the worktree id, got {err.worktree_id!r}"
        )


# ---------------------------------------------------------------------------
# TestWindowsPreflightBlockingCheck -- ticket #76
# ---------------------------------------------------------------------------

class TestWindowsPreflightBlockingCheck:
    """Ticket #76: on Windows, `git worktree remove --force` can return exit
    0 while still leaving a content-less locked directory behind (an open
    file handle blocks only the final directory removal, not the individual
    file unlinks). That used to fall past the lock-detection branch (which
    only triggers on `returncode != 0`) into the Final guard, which raised
    WorktreeDirLockedError without ever having attempted a kill -- and by
    then the destructive removal had already run, so no later retry (even
    with kill_blocking_processes=True) could ever recover.

    _teardown now runs a Windows-only pre-flight blocking-process check
    BEFORE the destructive `git worktree remove` call so an errored removal
    has no destructive side effect, and threads a real `kill_attempted` flag
    into the Final guard so its message correctly distinguishes "kill
    attempted, 0 found" from "kill not attempted"."""

    def test_preflight_flag_off_raises_before_git_remove(self, tmp_path):
        """Windows, kill_blocking_processes=False, pre-flight detects a
        blocker: _teardown must raise WorktreeDirLockedError with
        kill_attempted=False and killed=[] WITHOUT ever invoking `git
        worktree remove` or _kill_blocking_processes -- the destructive git
        call must never run on this attempt."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-preflight-flagoff", path="/fake/store/wt-preflight-flagoff"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_blockers = [KilledProcessInfo(pid=4444, name="node.exe", cmdline=["node"])]

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=fake_blockers,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes"
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_git.assert_not_called()
        mock_kill.assert_not_called()
        err = excinfo.value
        assert err.kill_attempted is False
        assert err.killed == []

    def test_preflight_no_blockers_falls_through_to_normal_removal(self, tmp_path):
        """Windows, pre-flight finds nothing: the normal git-remove path runs
        unchanged -- exit 0, no raise (retrospective coverage: this is the
        pre-existing happy path, unaffected by the new pre-flight check)."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-preflight-clean", path="/fake/store/wt-preflight-clean"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=[],
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            # Must not raise.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        # Ticket #140: the new all-platform orphan-scan phase (index 6) also
        # calls _find_blocking_processes on this same present target, AFTER
        # Gate A -- so the total is now 2 calls, not 1. Both find nothing,
        # so no kill call is made and record.orphan_scan stays None.
        assert mock_find.call_count == 2
        mock_find.assert_any_call(record.path, os.getpid())
        mock_git.assert_called_once()

    def test_preflight_flag_on_kills_before_git_remove_and_succeeds(self, tmp_path):
        """Windows, kill_blocking_processes=True, pre-flight detects a
        blocker: _kill_blocking_processes must be called BEFORE `git
        worktree remove` runs, and removal then proceeds normally (no
        raise), with record.killed_pids populated from the pre-flight
        kill."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-preflight-flagon", path="/fake/store/wt-preflight-flagon"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_blockers = [KilledProcessInfo(pid=5555, name="node.exe", cmdline=["node"])]

        call_order: List[str] = []

        def _mock_kill(path):
            call_order.append("kill")
            return fake_blockers

        def _git_side_effect(args, cwd=None, **kwargs):
            call_order.append("git_remove")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=fake_blockers,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                side_effect=_mock_kill,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            # Must not raise.
            manager._teardown(
                record,
                force=True,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        # Ticket #140: the new all-platform orphan-scan phase (index 6) runs
        # AFTER Gate A but still BEFORE `git worktree remove` -- it sees the
        # same mocked blocker via its own fresh _find_blocking_processes
        # call, and (kill_blocking_processes=True, >=1 hit) makes its own
        # reused _kill_blocking_processes call too. So a second "kill" now
        # legitimately lands before "git_remove"; what this test still pins
        # is that NEITHER kill happens after git_remove.
        assert call_order == ["kill", "kill", "git_remove"], (
            f"expected pre-flight kill(s) before git remove, got {call_order}"
        )
        assert record.killed_pids == fake_blockers

    def test_final_guard_kill_attempted_false_when_no_kill_ever_ran(self, tmp_path):
        """Ticket #76 root cause: git exits 0 (no lock signal, so the
        post-removal kill/retry branch never runs) but the directory still
        persists after all deletion attempts (long-path fallback also
        fails), and no kill was ever attempted (pre-flight found nothing,
        flag off). The Final guard must raise WorktreeDirLockedError with
        kill_attempted=False so the message correctly says "not attempted"
        instead of the pre-fix bug's false "after killing 0 blocking
        process(es)"."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-finalguard-nokill", path="/fake/store/wt-finalguard-nokill"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.teardown.os.path.exists",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=OSError("still locked"),
            ),
            patch(
                "lib_python_worktree.core.teardown.subprocess.run",
                return_value=MagicMock(returncode=1),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        err = excinfo.value
        assert err.kill_attempted is False, (
            "no kill was ever attempted (pre-flight clean, no post-removal "
            "lock signal) -- kill_attempted must be False, not the buggy "
            "default True"
        )
        msg = str(err)
        assert "kill_blocking_processes=True" in msg
        assert "after killing 0" not in msg

    def test_post_removal_kill_attempted_true_message_unchanged(self, tmp_path):
        """Regression: when a kill *is* actually attempted via the existing
        post-removal lock-retry branch (git returncode != 0 with a lock
        signal), the exhausted-retry message must still say "after killing
        N blocking process(es)" -- threading kill_attempted through the
        Final guard must not regress this pre-existing, correct phrasing
        for the case where a kill really was attempted."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-postkill-msg", path="/fake/store/wt-postkill-msg"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=6666, name="node.exe", cmdline=["node"])]

        def _git_always_fail(args, cwd=None, **kwargs):
            return MagicMock(returncode=255, stderr="Permission denied")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_always_fail,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch("lib_python_worktree.core.teardown.time"),
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=True,
                    _lifecycle_module=mock_lifecycle,
                )

        err = excinfo.value
        assert err.kill_attempted is True
        assert "after killing 1 blocking process(es)" in str(err)

    def test_preflight_skipped_entirely_on_posix(self, tmp_path):
        """POSIX: even if _find_blocking_processes were to report a blocker,
        the Windows-only Gate A pre-flight must never run on non-Windows
        platforms -- POSIX unlinks files even under an open handle, so a
        naive pre-flight there would incorrectly block/kill on a merely-
        cwd'd process. `git worktree remove` must proceed as the sole
        removal mechanism (no WorktreeDirLockedError, no refusal).

        Ticket #140: _find_blocking_processes is no longer uncalled on
        POSIX overall -- the new all-platform, warn-only orphan-scan phase
        (index 6) now calls it too, exactly once, AFTER Gate A's own
        (still-skipped) win32-only slot. What this test still pins is
        Gate A's specific absence: no kill (flag is off) and no refusal --
        the mocked blocker is only ever warn-reported by the new phase, not
        acted on."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-preflight-posix", path="/fake/store/wt-preflight-posix"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_blockers = [KilledProcessInfo(pid=7777, name="python", cmdline=["python"])]

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=fake_blockers,
            ) as mock_find,
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes"
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "linux"
            # Must not raise.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        mock_find.assert_called_once_with(record.path, os.getpid())
        mock_kill.assert_not_called()
        mock_git.assert_called_once()

    def test_degraded_scan_alone_does_not_block_removal(self, tmp_path):
        """Ticket #117 (Q2 Option A -- inverts the pre-#117 escalation
        pinned by the old test_preflight_degraded_partial_raises_instead_of_
        silent_removal): when _find_blocking_processes' Pass 2 degrades
        (e.g. psutil's open_files() raised an OS-wide RuntimeError on
        Windows -- see TestDiscoveryCompleteness::
        test_open_files_runtime_error_degrades_instead_of_raising in
        test_process_lifecycle.py), it returns an empty, degraded
        _PartialList (complete=False, skipped_passes=("open_files:degraded",)).

        Under the pre-#117 behaviour this alone escalated to
        WorktreeDirLockedError -- "no information" was treated the same as
        "confirmed blocker", making removal impossible for a never-started
        or cleanly-stopped environment on a host where the OS-wide handle
        query happens to fail. Ticket #117 makes this a logged, tolerated
        condition instead: the degraded tag carries no pid to confirm, so it
        can never contribute to Gate A's owned/foreign confirmation on its
        own -- removal must proceed, with a warning naming the degraded
        pass so the condition is observable rather than silent."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-preflight-degraded", path="/fake/store/wt-preflight-degraded"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        degraded = _PartialList(
            [], complete=False, skipped_passes=("open_files:degraded",)
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=degraded,
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            # Must NOT raise -- degraded alone is never a hard block.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        # Gate A itself only issues one scan: an empty (owned+foreign both
        # empty) result never enters the settle-window rescan. Ticket #140:
        # the new orphan-scan phase (index 6) makes its own additional call
        # after Gate A, so the module-wide total is 2, not 1.
        assert mock_find.call_count == 2
        mock_find.assert_any_call(record.path, os.getpid())
        # The destructive `git worktree remove` call did run.
        assert any(
            (call_args.args[0] if call_args.args else call_args.kwargs.get("args"))[
                :2
            ]
            == ["worktree", "remove"]
            for call_args in mock_git.call_args_list
        ), "git worktree remove must run once a degraded-alone scan is tolerated"

    def test_degraded_scan_alone_logs_warning(self, tmp_path, caplog):
        """Companion assertion (split out for a clean caplog fixture): the
        degraded tag is still observable via a WARNING log naming
        'open_files:degraded', even though it no longer blocks removal."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-preflight-degraded-log", path="/fake/store/wt-preflight-degraded-log"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        degraded = _PartialList(
            [], complete=False, skipped_passes=("open_files:degraded",)
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=degraded,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            caplog.at_level("WARNING", logger="lib_python_worktree.core.manager"),
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert any(
            "open_files:degraded" in rec.message for rec in caplog.records
        ), "expected a warning naming open_files:degraded"

    def test_degraded_scan_with_kill_flag_does_not_attempt_kill(self, tmp_path):
        """Rewrite of the old test_preflight_degraded_partial_kill_flag_on_
        attempts_kill_and_proceeds: since a degraded-alone scan no longer
        enters Gate A's confirmed-blocker branch at all (ticket #117),
        kill_blocking_processes=True must NOT attempt a kill -- there is
        nothing confirmed to kill. Removal still proceeds straight through
        to `git worktree remove`."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-preflight-degraded-killon",
            path="/fake/store/wt-preflight-degraded-killon",
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        degraded = _PartialList(
            [], complete=False, skipped_passes=("open_files:degraded",)
        )

        call_order: List[str] = []

        def _mock_kill(path):
            call_order.append("kill")
            return []

        def _git_side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                call_order.append("git_remove")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=degraded,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                side_effect=_mock_kill,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            # Must not raise.
            manager._teardown(
                record,
                force=True,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_not_called()
        assert call_order == ["git_remove"], (
            f"expected git remove with no kill attempt for a degraded-alone "
            f"scan, got {call_order}"
        )

    def test_degraded_plus_confirmed_blocker_still_raises(self, tmp_path):
        """Additional coverage (ticket #117): the degraded tag never
        *suppresses* a real confirmation -- if the SAME scan also carries a
        pid that is confirmed on the settle-window rescan, Gate A must still
        raise, exactly as it would for a non-degraded confirmed blocker."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-preflight-degraded-plus-blocker",
            path="/fake/store/wt-preflight-degraded-plus-blocker",
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        degraded_with_blocker = _PartialList(
            [KilledProcessInfo(pid=3131, name="node.exe", cmdline=["node"])],
            complete=False,
            skipped_passes=("open_files:degraded",),
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=degraded_with_blocker,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert excinfo.value.kill_attempted is False
        mock_git.assert_not_called()

    def test_preflight_ordinary_truncation_falls_through_to_normal_removal(
        self, tmp_path
    ):
        """Ticket #107 fix cycle (review finding 2, follow-up): the
        completeness check added above must be scoped to the new
        ``"open_files:degraded"`` tag specifically, NOT to
        ``_PartialList.complete`` in general. Pass 1c/Pass 2 can also come
        back with ``complete=False`` for the pre-existing, previously
        tolerated ``"handle_scan:truncated"``/``"open_files:truncated"``
        reasons (D5/D7) -- the scan simply ran out of its own time budget
        partway through a real, large process/handle table, which is
        expected and common on a busy dev host, not "never got a real
        look". A pre-#107 truthiness-only check already accepted a
        truncated-but-empty result and fell through to the normal removal
        path; escalating *that* to WorktreeDirLockedError as well would
        make `remove()` unconditionally fail under exactly the ambient-load
        conditions ticket #107 is about, which is the opposite of this
        ticket's intent. Only the degraded tag should escalate; ordinary
        truncation must still fall through unchanged."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-preflight-truncated", path="/fake/store/wt-preflight-truncated"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        truncated = _PartialList(
            [], complete=False, skipped_passes=("open_files:truncated",)
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=truncated,
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            # Must not raise -- ordinary truncation is tolerated, unlike
            # the degraded case above.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        # Gate A itself only issues one scan for an empty result. Ticket
        # #140: the new orphan-scan phase (index 6) makes its own additional
        # call after Gate A, so the module-wide total is 2, not 1.
        assert mock_find.call_count == 2
        mock_find.assert_any_call(record.path, os.getpid())
        mock_git.assert_called_once()


# ---------------------------------------------------------------------------
# TestTicket117PreflightFalsePositive -- ticket #117, behavioural
# requirement 1: a transient foreign blocker must not permanently block
# removal (the reported false-positive bug).
# ---------------------------------------------------------------------------

class TestTicket117PreflightFalsePositive:
    def test_transient_foreign_blocker_settles_and_removal_proceeds(
        self, tmp_path
    ):
        """Driving/regression test: a foreign process (not one of ours) is
        found on the first pre-flight scan but is gone by the confirming
        re-scan -- Gate A must NOT refuse removal. Before ticket #117, ANY
        non-empty scan result refused immediately (WorktreeDirLockedError,
        killed=[], kill_attempted=False) with no settle window at all."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-transient", path="/fake/store/wt-117-transient"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        transient = [KilledProcessInfo(pid=4444, name="claude", cmdline=["claude"])]

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                # Ticket #140: a THIRD entry for the new orphan-scan phase's
                # own post-Gate-A call (index 6) -- an empty result, so it
                # stays a clean, silent scan and does not itself add a
                # "scan:failed" marker (which a StopIteration from an
                # exhausted 2-item side_effect list would otherwise trigger,
                # since the phase's `except Exception` also catches that).
                side_effect=[transient, [], []],
            ) as mock_find,
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes"
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            # Must NOT raise.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert mock_find.call_count == 3
        mock_kill.assert_not_called()
        assert any(
            (call_args.args[0] if call_args.args else call_args.kwargs.get("args"))[
                :2
            ]
            == ["worktree", "remove"]
            for call_args in mock_git.call_args_list
        ), "git worktree remove must run once the transient blocker settles"

    def test_persistent_foreign_blocker_still_raises(self, tmp_path):
        """A foreign process present on BOTH the initial scan and the
        confirming re-scan is a genuine, confirmed blocker -- ticket #76's
        protection is fully retained. kill_blocking_processes=False must
        still refuse before the destructive git call."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-persistent", path="/fake/store/wt-117-persistent"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        persistent = [KilledProcessInfo(pid=4444, name="node.exe", cmdline=["node"])]

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=persistent,
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert excinfo.value.kill_attempted is False
        assert excinfo.value.killed == []
        assert mock_find.call_count == 2
        mock_git.assert_not_called()

    def test_persistent_foreign_blocker_with_kill_flag_clears_and_proceeds(
        self, tmp_path
    ):
        """Same persistent, confirmed blocker as above, but with
        kill_blocking_processes=True: the kill remedy is attempted and
        removal proceeds -- AC #2, a kill is only ever attempted against a
        confirmed blocker."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-persistent-kill", path="/fake/store/wt-117-persistent-kill"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        persistent = [KilledProcessInfo(pid=5555, name="node.exe", cmdline=["node"])]

        call_order: List[str] = []

        def _mock_kill(path):
            call_order.append("kill")
            return persistent

        def _git_side_effect(args, cwd=None, **kwargs):
            call_order.append("git_remove")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=persistent,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                side_effect=_mock_kill,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            manager._teardown(
                record,
                force=True,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        # Ticket #140: the new orphan-scan phase (index 6) runs after Gate A
        # but still before `git worktree remove`; it sees the same mocked
        # persistent blocker via its own fresh find call and (kill flag on,
        # >=1 hit) makes its own reused kill call too -- a second "kill"
        # legitimately lands before "git_remove".
        assert call_order == ["kill", "kill", "git_remove"]
        assert record.killed_pids == persistent

    def test_settle_warning_names_pid_and_process_name(self, tmp_path, caplog):
        """The confirmation warning names the offending pid and process
        name, satisfying AC #4's 'surfacing WHO holds the lock'."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-named", path="/fake/store/wt-117-named"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        persistent = [
            KilledProcessInfo(pid=9191, name="MsMpEng.exe", cmdline=["MsMpEng.exe"])
        ]

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=persistent,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            caplog.at_level("WARNING", logger="lib_python_worktree.core.manager"),
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError):
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert any(
            "9191" in rec.message and "MsMpEng.exe" in rec.message
            for rec in caplog.records
        ), "expected a warning naming pid=9191 and MsMpEng.exe"

    def test_settle_loop_is_bounded(self, tmp_path):
        """The settle window is bounded: at most _PREFLIGHT_SETTLE_RETRIES
        sleep calls and at most 2 total scan calls, even for a persistent
        blocker."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.teardown import _PREFLIGHT_SETTLE_RETRIES
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-bounded", path="/fake/store/wt-117-bounded"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        persistent = [KilledProcessInfo(pid=6262, name="node.exe", cmdline=["node"])]

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=persistent,
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.time") as mock_time,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError):
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert mock_find.call_count <= 2
        assert mock_time.sleep.call_count <= _PREFLIGHT_SETTLE_RETRIES

    def test_plain_list_without_skipped_passes_attr_works(self, tmp_path):
        """A plain `list` return (no `.skipped_passes` attribute, e.g. an
        older/simpler test double) must not raise AttributeError -- the
        degraded check uses `getattr(..., "skipped_passes", ())`."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-plainlist", path="/fake/store/wt-117-plainlist"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=[],
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            # Must not raise.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

    def test_posix_never_scans(self, tmp_path):
        """Retrospective coverage: Gate A's Windows-only guard is unaffected
        by the Gate A rewrite -- Gate A itself never calls
        _find_blocking_processes on POSIX (existing coverage, re-asserted
        here for the new gate).

        Ticket #140: _find_blocking_processes is no longer uncalled on
        POSIX overall -- the new all-platform, warn-only orphan-scan phase
        (index 6) now calls it too, exactly once, independently of Gate A's
        own (still win32-only) slot."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-117-posix", path="/fake/store/wt-117-posix")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=[],
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "linux"
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        mock_find.assert_called_once_with(record.path, os.getpid())


# ---------------------------------------------------------------------------
# TestTicket117OwnedBlocker -- ticket #117, behavioural requirement 2: an
# owned blocker (a pid this environment itself tracked) is confirmed
# without a settle delay, and still hard-blocks.
# ---------------------------------------------------------------------------

class TestTicket117OwnedBlocker:
    def test_owned_blocker_confirmed_even_when_co_occurring_foreign_settles(
        self, tmp_path
    ):
        """Driving test: a single scan returns BOTH an owned blocker (a pid
        this environment tracked in record.pids) and a foreign one that
        turns out to be transient (gone by the confirming re-scan). Gate A
        must still raise -- the owned pid is trusted immediately, with no
        need for it to also survive a confirming re-scan -- AND the
        re-scan must still have happened (because the co-occurring foreign
        hit needs settling), so `_find_blocking_processes` is called
        exactly twice.

        This is the genuinely RED-worthy assertion for requirement 2: a
        single *owned-only* blocker (no co-occurring foreign hit) happens
        to produce the SAME observable outcome under the pre-#117 code
        (which also raises immediately with exactly one scan and no sleep,
        since it never distinguished owned from foreign at all) -- see
        test_single_owned_blocker_needs_no_settle_delay below, which is
        legitimate additional coverage but does not itself demonstrate a
        behavioural difference. The owned/foreign PARTITION machinery this
        ticket adds only becomes observable once a foreign pid is also in
        play: pre-#117 code would raise after exactly ONE scan call
        (compound truthiness check, no settle window at all); ticket #117
        code must still raise, but only after TWO scan calls (the
        mandatory settle re-scan for the foreign hit)."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-owned-mixed",
            path="/fake/store/wt-117-owned-mixed",
            pids={"main": 4444},
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        owned_hit = KilledProcessInfo(pid=4444, name="daemon", cmdline=["daemon"])
        foreign_hit = KilledProcessInfo(pid=5555, name="claude", cmdline=["claude"])

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                side_effect=[[owned_hit, foreign_hit], [owned_hit]],
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert excinfo.value.kill_attempted is False
        assert mock_find.call_count == 2, (
            "the co-occurring foreign hit must still trigger a confirming "
            "settle re-scan even though the owned hit alone already "
            "guarantees the refusal"
        )
        mock_git.assert_not_called()

    def test_single_owned_blocker_needs_no_settle_delay(self, tmp_path):
        """Additional coverage: a scan containing ONLY an owned blocker (no
        co-occurring foreign hit) is confirmed with a single scan call and
        no sleep. Disclosed honestly: this exact assertion set ALSO holds
        under the pre-#117 code (which never had a settle window for
        anything, owned or foreign, so a lone non-empty scan result always
        raised immediately after one call) -- it does not by itself
        demonstrate the new owned/foreign partition; the mixed-blocker test
        above does. Kept here as a pin on the no-unnecessary-delay
        guarantee for the common single-owned-daemon case."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-owned",
            path="/fake/store/wt-117-owned",
            pids={"main": 4444},
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        owned_hit = [KilledProcessInfo(pid=4444, name="daemon", cmdline=["daemon"])]

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=owned_hit,
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.time") as mock_time,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert excinfo.value.kill_attempted is False
        mock_find.assert_called_once_with(record.path, os.getpid())
        mock_time.sleep.assert_not_called()

    def test_owned_blocker_kill_flag_kills_and_proceeds(self, tmp_path):
        """Extends the existing owned-blocker coverage: with
        kill_blocking_processes=True, the owned blocker is killed (no
        settle delay) and removal proceeds."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-owned-kill",
            path="/fake/store/wt-117-owned-kill",
            pids={"main": 7373},
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        owned_hit = [KilledProcessInfo(pid=7373, name="daemon", cmdline=["daemon"])]

        call_order: List[str] = []

        def _mock_kill(path):
            call_order.append("kill")
            return owned_hit

        def _git_side_effect(args, cwd=None, **kwargs):
            call_order.append("git_remove")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=owned_hit,
            ) as mock_find,
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                side_effect=_mock_kill,
            ),
            patch("lib_python_worktree.core.teardown.time") as mock_time,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            manager._teardown(
                record,
                force=True,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        # Ticket #140: the new orphan-scan phase (index 6) runs after Gate A
        # but still before `git worktree remove`; it sees the same mocked
        # owned hit via its own fresh find call and (kill flag on, >=1 hit)
        # makes its own reused kill call too -- a second "kill" legitimately
        # lands before "git_remove", and _find_blocking_processes is now
        # called twice module-wide (Gate A's own immediate-confirm call,
        # still with no settle-window rescan since the hit is owned, plus
        # the new phase's call).
        assert call_order == ["kill", "kill", "git_remove"]
        assert record.killed_pids == owned_hit
        assert mock_find.call_count == 2
        mock_find.assert_any_call(record.path, os.getpid())
        mock_time.sleep.assert_not_called()


# ---------------------------------------------------------------------------
# TestTicket117FixCycleDegradedConfirmation -- fix-loop round (blocking
# findings 1 & 2): a DEGRADED rescan must never be read as evidence that a
# previously observed blocker cleared. The ticket's core de-escalation (a
# degraded scan with NO prior hit does not block removal -- see
# TestWindowsPreflightBlockingCheck::test_degraded_scan_alone_does_not_
# block_removal) must remain intact; these tests cover only the narrower
# case where a real hit was already observed and the follow-up scan (either
# Gate A's confirming settle rescan, or _kill_blocking_processes' own
# internal rescan) comes back blind instead of clean.
# ---------------------------------------------------------------------------

class TestTicket117FixCycleDegradedConfirmation:
    def test_degraded_confirming_rescan_does_not_clear_real_blocker(
        self, tmp_path
    ):
        """Driving test for blocking finding 1: the settle-window
        CONFIRMING re-scan comes back DEGRADED (open_files:degraded, empty
        contents) instead of clean, after the first scan found a genuine
        foreign hit. A blind rescan is not evidence the foreign blocker
        went away -- Gate A must NOT intersect the degraded rescan's empty
        pid set with the first scan's hit and conclude "settled, proceed".
        Before this fix, intersecting an empty `_rescan_pids` (the
        degraded _PartialList's empty contents) with the first-scan
        foreign pid always produced an empty `_foreign`, silently clearing
        a real, unconfirmed blocker and falling through to the destructive
        `git worktree remove` call -- reproducing the exact locked-
        directory failure ticket #117 exists to prevent."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-degraded-confirm",
            path="/fake/store/wt-117-degraded-confirm",
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        foreign_hit = [
            KilledProcessInfo(pid=8181, name="node.exe", cmdline=["node"])
        ]
        degraded_rescan = _PartialList(
            [], complete=False, skipped_passes=("open_files:degraded",)
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                side_effect=[foreign_hit, degraded_rescan],
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert excinfo.value.kill_attempted is False
        assert mock_find.call_count == 2, (
            "the degraded confirming rescan must still have been "
            "attempted once within the bounded settle window"
        )
        mock_git.assert_not_called()

    def test_degraded_scan_alone_still_does_not_block_with_new_logic(
        self, tmp_path
    ):
        """Non-regression companion for AC #1: a degraded scan with NO
        prior hit at all must still proceed without refusal -- the
        finding-1 fix only concerns a degraded rescan that follows a real
        first-scan hit; it must not resurrect the pre-#117 blanket
        degraded-always-blocks behaviour for the no-hit case."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-degraded-nohit-fixcycle",
            path="/fake/store/wt-117-degraded-nohit-fixcycle",
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        degraded = _PartialList(
            [], complete=False, skipped_passes=("open_files:degraded",)
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=degraded,
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            # Must NOT raise.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        # No prior hit -> `_foreign` is empty -> the settle loop (and thus
        # any confirming rescan) is never entered at all -- Gate A itself
        # issues only one scan. Ticket #140: the new orphan-scan phase
        # (index 6) makes its own additional call after Gate A, so the
        # module-wide total is 2, not 1.
        assert mock_find.call_count == 2
        mock_find.assert_any_call(record.path, os.getpid())
        mock_git.assert_called_once()

    def test_kill_blocking_processes_degraded_result_does_not_clear_confirmed_blocker(
        self, tmp_path
    ):
        """Driving test for blocking finding 2: `_kill_blocking_processes`
        returns an empty, DEGRADED `_PartialList` (its own internal rescan
        hit open_files:degraded and never actually looked for anything to
        kill) rather than a genuinely empty list. `if not killed:` alone
        cannot distinguish "killed nothing because the confirmed blocker
        was already gone" from "the kill's own scan went blind" -- a
        `_PartialList` is falsy-empty in both cases; only
        `skipped_passes` on the returned value tells them apart. Before
        this fix, the pre-#117 guard for exactly this case had been
        dropped with no replacement, so a still-live, twice-confirmed
        blocker fell through silently to the destructive `git worktree
        remove` call."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-kill-degraded", path="/fake/store/wt-117-kill-degraded"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        persistent = [
            KilledProcessInfo(pid=9292, name="node.exe", cmdline=["node"])
        ]
        degraded_kill_result = _PartialList(
            [], complete=False, skipped_passes=("open_files:degraded",)
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=persistent,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=degraded_kill_result,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=True,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_called_once()
        assert excinfo.value.kill_attempted is True
        assert excinfo.value.killed == []
        mock_git.assert_not_called()

    def test_kill_blocking_processes_empty_non_degraded_still_proceeds(
        self, tmp_path
    ):
        """Non-regression companion: when `_kill_blocking_processes`
        returns a genuinely empty, NON-degraded result (the confirmed
        blocker exited in the interval between confirmation and the kill
        call), the existing 'warn and proceed' behaviour must be
        unaffected by the finding-2 fix."""
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-kill-empty-clean",
            path="/fake/store/wt-117-kill-empty-clean",
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        persistent = [
            KilledProcessInfo(pid=9393, name="node.exe", cmdline=["node"])
        ]

        call_order: List[str] = []

        def _mock_kill(path):
            call_order.append("kill")
            return []

        def _git_side_effect(args, cwd=None, **kwargs):
            call_order.append("git_remove")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=persistent,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                side_effect=_mock_kill,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            # Must not raise.
            manager._teardown(
                record,
                force=True,
                kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        # Ticket #140: the new orphan-scan phase (index 6) runs after Gate A
        # but still before `git worktree remove`; it sees the same mocked
        # persistent blocker via its own fresh find call and (kill flag on,
        # >=1 hit) makes its own reused (also empty-result) kill call too --
        # a second "kill" legitimately lands before "git_remove".
        assert call_order == ["kill", "kill", "git_remove"]


# ---------------------------------------------------------------------------
# TestTicket117TeardownRunsAtMostOnce -- ticket #117, behavioural
# requirement 4 (AC #3): contract teardown: steps run at most once per
# logical removal, only past both gates.
# ---------------------------------------------------------------------------

class TestTicket117TeardownRunsAtMostOnce:
    def test_preflight_block_does_not_run_teardown_steps(self, tmp_path):
        """Driving test: a twice-confirmed persistent blocker refuses
        removal (Gate A) BEFORE the contract's teardown: steps ever run --
        only the stop: steps (Step 1b) may have run. Before ticket #117,
        teardown: ran unconditionally before the pre-flight gate, so a
        blocked attempt had already executed it once."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.contract.schema import Step, WorktreeContract
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-teardown-once", path="/fake/store/wt-117-teardown-once"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            stop=[Step(run='echo stop', name="stop-svc")],
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        runner_calls: List[dict] = []
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        mock_lifecycle = MagicMock()
        persistent = [KilledProcessInfo(pid=8181, name="node.exe", cmdline=["node"])]

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=persistent,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError):
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        setups_run = [kw["setup"] for kw in runner_calls]
        assert fake_contract.stop in setups_run, "stop: steps must still run"
        assert fake_contract.teardown not in setups_run, (
            "teardown: steps must NOT run when Gate A refuses the removal"
        )

    def test_dirty_tree_raises_before_teardown_steps(self, tmp_path):
        """Gate B: real (non-.seretos) uncommitted dirt refuses removal
        BEFORE the teardown: steps run, when the contract has teardown:
        steps to protect."""
        from lib_python_worktree.core.manager import DirtyWorktreeError
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-dirty-gate", path="/fake/store/wt-117-dirty-gate"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        mock_runner_instance = MagicMock()
        runner_calls: List[dict] = []
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        mock_lifecycle = MagicMock()

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? real.txt\0", stderr="")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(DirtyWorktreeError):
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert runner_calls == [], "teardown: steps must not run before Gate B"

    def test_no_teardown_steps_means_no_extra_git_status(self, tmp_path):
        """When the contract has no teardown: steps, Gate B must not call
        _dirt_probe() at all -- zero extra `git status` calls, preserving
        the pre-#117 happy-path call-count guarantee."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-no-teardown-status", path="/fake/store/wt-117-no-teardown-status"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            stop=[Step(run='echo stop', name="stop-svc")],
        )

        mock_runner_instance = MagicMock()
        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert not any(_is_status_call(c.args[0]) for c in mock_git.call_args_list), (
            "no git status call expected when there are no teardown: steps"
        )
        mock_git.assert_called_once()

    def test_teardown_step_dirt_is_not_masked_by_stale_pre_teardown_cache(
        self, tmp_path
    ):
        """Ticket #117 fix cycle (blocking finding): pre-teardown dirt is
        ONLY the benign, exempt `.seretos/` copy (so Gate B correctly does
        not refuse and lets teardown: run), but the teardown: step itself
        leaves behind REAL, non-exempt dirt (e.g. a log a teardown script
        forgot to clean up). The post-teardown `.seretos/`-exemption
        classifier must re-probe `git status` rather than reuse Gate B's
        memoised pre-teardown snapshot -- reusing it would make the
        classifier believe all dirt is still the benign `.seretos/` copy
        and silently auto-force-discard the real dirt instead of raising
        DirtyWorktreeError."""
        from lib_python_worktree.core.manager import DirtyWorktreeError
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-stale-cache", path="/fake/store/wt-117-stale-cache"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        teardown_ran = [False]
        runner_calls: List[dict] = []

        def _run_teardown(**kw):
            runner_calls.append(kw)
            teardown_ran[0] = True

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = _run_teardown

        mock_lifecycle = MagicMock()
        status_call_count = [0]

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                status_call_count[0] += 1
                if not teardown_ran[0]:
                    # Pre-teardown: only the benign, exempt .seretos/ copy.
                    return MagicMock(
                        returncode=0, stdout="?? .seretos/notes.txt\0", stderr=""
                    )
                # Post-teardown: the teardown step left real, non-exempt
                # dirt behind (the .seretos/ copy is gone from this
                # re-probe -- the point is the REAL dirt must be seen, not
                # masked by whatever the stale snapshot held).
                return MagicMock(
                    returncode=0, stdout="?? real_leftover.log\0", stderr=""
                )
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(DirtyWorktreeError):
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert len(runner_calls) == 1, "teardown: steps must have run exactly once"
        assert status_call_count[0] == 2, (
            "expected exactly two git status calls: Gate B's pre-teardown "
            "probe (memoised for Gate B's own use), and a fresh "
            "post-teardown re-probe by the .seretos/-exemption classifier "
            "-- reusing Gate B's stale snapshot instead of re-probing is "
            "exactly the bug this test guards against"
        )

    def test_only_seretos_dirt_still_reaches_git_and_auto_forces(self, tmp_path):
        """Ticket #100 exemption preserved: when the ONLY dirt is benign
        untracked content under .seretos/, Gate B must NOT refuse -- the
        dirt probe used by Gate B (_real_dirt_paths) already filters exempt
        untracked entries, so removal reaches the git call and the existing
        auto-force retry logic further down handles the rest."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-seretos-only", path="/fake/store/wt-117-seretos-only"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        mock_runner_instance = MagicMock()
        runner_calls: List[dict] = []
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        mock_lifecycle = MagicMock()

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                return MagicMock(
                    returncode=0, stdout="?? .seretos/notes.txt\0", stderr=""
                )
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            # Must not raise.
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert len(runner_calls) == 1
        assert runner_calls[0]["setup"] == fake_contract.teardown

    def test_force_true_skips_early_dirt_probe(self, tmp_path):
        """force=True: Gate B's condition requires `not force`, so it never
        even calls _dirt_probe() -- real dirt cannot block a forced
        removal, matching the existing force=True/no-status-call
        guarantee."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-force-skip", path="/fake/store/wt-117-force-skip"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        mock_runner_instance = MagicMock()
        runner_calls: List[dict] = []
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        mock_lifecycle = MagicMock()
        calls: List[list] = []

        def _side_effect(args, cwd=None, **kwargs):
            calls.append(list(args))
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=True, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert not any(_is_status_call(c) for c in calls)
        assert len(runner_calls) == 1

    def test_contract_loaded_once_per_teardown(self, tmp_path):
        """The consolidated single contract load (ticket #117) means
        _load_contract is called exactly once per _teardown() invocation,
        not twice (once for stop:, once for teardown:) as before."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-load-once", path="/fake/store/wt-117-load-once"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            stop=[Step(run='echo stop', name="stop-svc")],
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        mock_runner_instance = MagicMock()
        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ) as mock_load,
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        mock_load.assert_called_once()

    def test_dirt_snapshot_taken_before_teardown_steps(self, tmp_path):
        """Shared-call-site note (ticket #117): Gate B primes the memoised
        status cache BEFORE the teardown: steps run, so a teardown step's
        own generated untracked file cannot retroactively taint the dirt
        snapshot this same removal attempt already judged. Uses a
        conclusive (non-empty, exempt-only) status result so the pre-
        existing memoisation actually short-circuits the second call site
        further down (an inconclusive/empty result is a documented
        `_status_entries` special case -- see its docstring -- that is
        deliberately out of this ticket's scope). The status call must
        happen exactly once, and before the runner call that executes
        teardown:."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-117-snapshot-order", path="/fake/store/wt-117-snapshot-order"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        call_order: List[str] = []

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = lambda **kw: call_order.append(
            "teardown_runner"
        )

        mock_lifecycle = MagicMock()

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                call_order.append("git_status")
                return MagicMock(
                    returncode=0, stdout="?? .seretos/notes.txt\0", stderr=""
                )
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert call_order == ["git_status", "teardown_runner"], (
            f"expected dirt snapshot before teardown: steps, got {call_order}"
        )
        assert call_order.count("git_status") == 1


# ---------------------------------------------------------------------------
# TestTicket126TeardownNotReRunOnRetry -- ticket #126
# ---------------------------------------------------------------------------

class TestTicket126TeardownNotReRunOnRetry:
    """Ticket #126: a SECOND reachable path to the #117 symptom -- Gate B
    passes (pre-teardown dirt is clean/exempt), teardown: steps run, but
    the subsequent `git worktree remove` fails with a POST-teardown
    DirtyWorktreeError (e.g. because the teardown step itself wrote a
    file). Following that error's own suggested remedy, the caller retries
    with force=True, which bypasses Gate B entirely -- without the
    persisted `teardown_ran` marker, that retry would re-run teardown: a
    second time."""

    def test_forced_retry_after_post_teardown_dirty_failure_does_not_rerun_teardown(
        self, tmp_path
    ):
        """Driving test: force=False raises DirtyWorktreeError after
        teardown: has already run once; a subsequent force=True retry on
        the SAME record must not run teardown: again."""
        from lib_python_worktree.core.manager import DirtyWorktreeError
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-126-forced-retry", path="/fake/store/wt-126-forced-retry"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        teardown_ran = [False]
        runner_calls: List[dict] = []

        def _run_teardown(**kw):
            runner_calls.append(kw)
            teardown_ran[0] = True

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = _run_teardown

        mock_lifecycle = MagicMock()

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                if not teardown_ran[0]:
                    # Pre-teardown: clean.
                    return MagicMock(returncode=0, stdout="", stderr="")
                # Post-teardown: the teardown step left real, non-exempt
                # dirt behind.
                return MagicMock(
                    returncode=0, stdout="?? real_leftover.log\0", stderr=""
                )
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_force_remove_call(args):
                return MagicMock(returncode=0, stderr="")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(DirtyWorktreeError):
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

            # Caller follows the error's own suggested remedy.
            manager._teardown(
                record,
                force=True,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        setups_run = [kw["setup"] for kw in runner_calls]
        assert setups_run.count(fake_contract.teardown) == 1, (
            f"teardown: steps must run exactly once across both calls, "
            f"got {setups_run}"
        )
        assert record.teardown_ran is True

    def test_retry_after_gate_a_lock_runs_teardown_on_first_successful_attempt(
        self, tmp_path
    ):
        """A retry after a Gate A WorktreeDirLockedError (teardown: never
        ran on the blocked attempt) must still run teardown: exactly once
        -- on the first attempt that actually clears the block. Proves the
        `teardown_ran` marker does not over-suppress steps that never ran
        in the first place."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.contract.schema import Step, WorktreeContract
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-126-lock-retry", path="/fake/store/wt-126-lock-retry"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        runner_calls: List[dict] = []
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        mock_lifecycle = MagicMock()
        persistent = [KilledProcessInfo(pid=9191, name="node.exe", cmdline=["node"])]

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=persistent,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError):
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert runner_calls == [], "teardown: must not run while Gate A is blocked"

        # Blocker clears; this retry is the FIRST successful attempt.
        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=[],
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        setups_run = [kw["setup"] for kw in runner_calls]
        assert setups_run.count(fake_contract.teardown) == 1, (
            f"teardown: must run exactly once, on the first successful "
            f"attempt, got {setups_run}"
        )

    def test_force_true_first_call_still_runs_teardown_once(self, tmp_path):
        """Additional coverage: a force=True call with no prior attempt
        (teardown_ran defaults to False) still runs teardown: exactly
        once -- the new guard must not suppress the very first run."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-126-force-first", path="/fake/store/wt-126-force-first"
        )
        manager.state.add(record)
        assert record.teardown_ran is False

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        mock_runner_instance = MagicMock()
        runner_calls: List[dict] = []
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=True, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert len(runner_calls) == 1
        assert record.teardown_ran is True

    def test_no_teardown_steps_never_sets_marker_and_no_extra_status_call(
        self, tmp_path
    ):
        """Additional coverage: protects the existing zero-extra-`git
        status`-calls invariant (see
        test_no_teardown_steps_means_no_extra_git_status above) AND
        confirms the new marker is only ever set when teardown: steps
        actually exist and run."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-126-no-teardown", path="/fake/store/wt-126-no-teardown"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            stop=[Step(run='echo stop', name="stop-svc")],
        )

        mock_runner_instance = MagicMock()
        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert not any(_is_status_call(c.args[0]) for c in mock_git.call_args_list)
        assert record.teardown_ran is False

    def test_untracked_target_persist_swallows_keyerror(self, tmp_path):
        """Ticket #88 style: a synthesised record never added to the state
        store (mirrors the untracked-removal-target path). The guarded
        ``self.state.update(record)`` persist added by this ticket must
        swallow InMemoryStateStore's KeyError rather than let it
        propagate, and must not create a spurious store entry."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-126-untracked", path="/fake/store/wt-126-untracked"
        )
        # Deliberately NOT added to manager.state.

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        mock_runner_instance = MagicMock()
        runner_calls: List[dict] = []
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=True, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert len(runner_calls) == 1
        assert record.teardown_ran is True, "marker still set on the in-memory object"
        assert manager.state.list() == [], "untracked path must never write to the store"

    def test_partial_teardown_failure_leaves_marker_false_and_retries_run_again(
        self, tmp_path
    ):
        """Driving test (fix-cycle blocking finding): `SetupRunner.run()`
        raises partway through a multi-step `teardown:` sequence -- e.g. it
        fails on step 2 of 3. `_teardown()` must (a) swallow the exception
        per ticket #117's "teardown failure must not block remove" policy,
        (b) leave `record.teardown_ran` at False rather than force it True
        (the steps never actually completed), and (c) let a *subsequent*
        `_teardown()` call on the SAME record attempt the teardown steps
        again -- the retry must get a genuine chance to finish the cleanup
        work (e.g. releasing a lock, stopping a service) that the first
        attempt never got to."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-126-partial-failure", path="/fake/store/wt-126-partial-failure"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[
                Step(run='echo one', name="step-1"),
                Step(run='echo two', name="step-2"),
                Step(run='echo three', name="step-3"),
            ],
        )

        runner_calls: List[dict] = []

        def _raise_partway(**kw):
            # Simulates the real SetupRunner having executed step 1
            # successfully, then raising while attempting step 2 -- step 3
            # never runs either. The mock only models the *outcome*
            # (`run()` raises) since `run()` is what `_teardown()` calls
            # and swallows.
            runner_calls.append(kw)
            raise RuntimeError("step-2 failed")

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = _raise_partway

        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"

            # First attempt: teardown: raises partway through. Per #117's
            # policy this must not propagate out of _teardown() and must
            # not block the (mocked) git worktree remove.
            manager._teardown(
                record, force=True, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert len(runner_calls) == 1, "first attempt must have tried to run teardown:"
        assert record.teardown_ran is False, (
            "a partially-failed teardown: run must NOT set the at-most-once "
            "marker -- steps 2/3 never completed, so a retry must still be "
            "allowed to run the full sequence again"
        )

        # Second attempt (retry): this time teardown: completes without
        # raising.
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=True, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert len(runner_calls) == 2, (
            "the retry must attempt teardown: again -- it must not be "
            "skipped just because a prior attempt failed partway through"
        )
        assert record.teardown_ran is True, (
            "once teardown: actually completes without raising, the "
            "at-most-once marker must be set"
        )

    def test_marker_persist_oserror_does_not_block_git_worktree_remove(
        self, tmp_path
    ):
        """Driving test (review round 2, blocking finding): teardown: steps
        complete successfully, but persisting the `teardown_ran` marker via
        `self.state.update(record)` raises `OSError` (e.g. disk full,
        transient state.yaml lock contention, permission hiccup). This must
        NOT propagate out of `_teardown()` and must NOT prevent Step 4
        (`git worktree remove`) from running -- the teardown steps already
        completed; the only failure is a bookkeeping write, and ticket
        #117's policy (a failure that isn't the removal itself must never
        block the removal) applies here too."""
        from lib_python_worktree.contract.schema import Step, WorktreeContract

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-126-persist-oserror", path="/fake/store/wt-126-persist-oserror"
        )
        manager.state.add(record)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            teardown=[Step(run='echo bye', name="teardown-svc")],
        )

        mock_runner_instance = MagicMock()
        runner_calls: List[dict] = []
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch.object(
                manager.state, "update", side_effect=OSError("disk full")
            ),
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            # Must not raise: the OSError from the marker-persist write must
            # be swallowed, exactly like the existing KeyError swallow.
            manager._teardown(
                record, force=True, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert len(runner_calls) == 1, "teardown: steps must have run"
        assert record.teardown_ran is True, (
            "the in-memory record must still reflect completion even though "
            "persisting it failed"
        )
        assert any(
            _is_force_remove_call(c.args[0]) or _is_plain_remove_call(c.args[0])
            for c in mock_git.call_args_list
        ), (
            "Step 4 (git worktree remove) must still run despite the "
            "marker-persist OSError -- the persistence failure must not "
            "block the actual removal"
        )


# ---------------------------------------------------------------------------
# TestRobocopyTimeoutBounding -- ticket #78
# ---------------------------------------------------------------------------

class TestRobocopyTimeoutBounding:
    """Ticket #78: the Windows long-path robocopy empty-mirror fallback in
    _teardown used to call `subprocess.run(["robocopy", ...])` with no
    timeout at all. Robocopy's own defaults (/R:1000000 /W:30) retry a
    locked directory for ~347 days, wedging the single synchronous MCP
    request thread and hanging the whole server -- while the destructive
    git worktree remove had *already* completed. This class verifies the
    fix bounds that call two independent ways (explicit timeout= kwarg,
    plus robocopy's own /R:1 /W:1 fast-fail flags), that a timeout still
    surfaces the existing clean WorktreeDirLockedError contract rather than
    leaking subprocess.TimeoutExpired, that the timeout is env-configurable,
    and that the Step 2b preflight boundary from ticket #76/#77 is
    unaffected by this change."""

    def test_robocopy_fallback_is_time_bounded(self, tmp_path):
        """Driving test: the subprocess.run call for robocopy must carry a
        finite numeric timeout= kwarg and robocopy's own fast-fail retry
        flags (/R:1 /W:1) -- so a locked directory can never wedge the
        request thread for robocopy's default ~347-day retry window."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-robocopy-bounded", path="C:\\fake\\store\\wt-robocopy-bounded"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        robocopy_calls: list = []
        robocopy_kwargs: list = []

        def _mock_rmtree(path, **kwargs):
            if path.startswith("\\\\?\\"):
                raise OSError("path too long")
            # Second call (after robocopy) succeeds silently.

        def _mock_subprocess_run(cmd, **kwargs):
            robocopy_calls.append(cmd)
            robocopy_kwargs.append(kwargs)
            return MagicMock(returncode=1)  # robocopy exits 1 on success-with-copies

        # See explanation in
        # test_directory_still_exists_after_git_remove_triggers_longpath_deletion.
        # Ticket #135: the target_absent probe now goes through the
        # separately mockable teardown._target_is_absent seam (patched by
        # the autouse _target_present_by_default fixture) and no longer
        # touches os.path.exists. This mock only needs to cover the
        # remaining real os.path.exists(record.path) call: the
        # long-path-fallback check (True).
        _path_calls = {"n": 0}

        def _mock_exists(path):
            if str(path) == record.path:
                _path_calls["n"] += 1
                return _path_calls["n"] <= 1
            return False

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch("lib_python_worktree.core.teardown.os.path.exists", side_effect=_mock_exists),
            patch("lib_python_worktree.core.teardown.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.teardown.subprocess.run", side_effect=_mock_subprocess_run),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert robocopy_calls, "robocopy must be called when extended-path rmtree fails"
        kwargs = robocopy_kwargs[0]
        assert "timeout" in kwargs, (
            f"robocopy subprocess.run must pass an explicit timeout kwarg, got kwargs={kwargs}"
        )
        assert isinstance(kwargs["timeout"], (int, float)) and kwargs["timeout"] > 0, (
            f"robocopy timeout must be a finite positive number, got {kwargs['timeout']!r}"
        )
        args = robocopy_calls[0]
        assert "/R:1" in args, f"robocopy args must include fast-fail /R:1, got {args}"
        assert "/W:1" in args, f"robocopy args must include fast-fail /W:1, got {args}"

    def test_robocopy_timeout_raises_dir_locked(self, tmp_path):
        """When robocopy itself times out (subprocess.TimeoutExpired), with
        the leftover directory still present, _teardown must not leak the
        raw TimeoutExpired -- the existing Final guard must still raise
        WorktreeDirLockedError, carrying the record's id."""
        import subprocess as subprocess_module

        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-robocopy-timeout", path="C:\\fake\\store\\wt-robocopy-timeout"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _mock_rmtree(path, **kwargs):
            if path.startswith("\\\\?\\"):
                raise OSError("path too long")

        def _mock_subprocess_run(cmd, **kwargs):
            raise subprocess_module.TimeoutExpired(cmd=cmd, timeout=kwargs.get("timeout"))

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            # Leftover dir stays present throughout (robocopy never got the
            # chance to mirror it empty), so both the long-path guard and
            # the Final guard see it as still there. Contract-file loads
            # are wrapped in broad try/except in _teardown so a "True" for
            # those paths too is harmless (matches the existing
            # test_directory_still_exists_after_fallback_raises pattern).
            patch("lib_python_worktree.core.teardown.os.path.exists", return_value=True),
            patch("lib_python_worktree.core.teardown.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.teardown.subprocess.run", side_effect=_mock_subprocess_run),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as exc_info:
                manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert exc_info.value.worktree_id == "wt-robocopy-timeout"

    def test_robocopy_timeout_env_override(self, tmp_path, monkeypatch):
        """WORKTREE_ROBOCOPY_TIMEOUT_SEC overrides the built-in 30s default
        when set to a float string; an empty string disables the timeout
        entirely (resolves to timeout=None), mirroring WORKTREE_GIT_TIMEOUT_SEC."""
        manager = _make_manager(tmp_path)

        mock_lifecycle = MagicMock()
        captured = {"timeout": "unset"}

        def _mock_rmtree(path, **kwargs):
            if path.startswith("\\\\?\\"):
                raise OSError("path too long")

        def _mock_subprocess_run(cmd, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return MagicMock(returncode=1)

        def _make_exists_for(target_path):
            # Ticket #135: the target_absent probe now goes through the
            # separately mockable teardown._target_is_absent seam (patched
            # by the autouse _target_present_by_default fixture) and no
            # longer touches os.path.exists. This mock only needs to cover
            # the remaining real os.path.exists(target_path) call: the
            # long-path-fallback check (True) -- see the explanation in
            # test_directory_still_exists_after_git_remove_triggers_longpath_deletion.
            calls = {"n": 0}

            def _mock_exists(path):
                if str(path) == target_path:
                    calls["n"] += 1
                    return calls["n"] <= 1
                return False

            return _mock_exists

        # --- Override to a specific value ---
        record_override = _make_record(
            "wt-robocopy-env-override", path="C:\\fake\\store\\wt-robocopy-env-override"
        )
        manager.state.add(record_override)
        monkeypatch.setenv("WORKTREE_ROBOCOPY_TIMEOUT_SEC", "12.5")

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown.os.path.exists",
                side_effect=_make_exists_for(record_override.path),
            ),
            patch("lib_python_worktree.core.teardown.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.teardown.subprocess.run", side_effect=_mock_subprocess_run),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(record_override, force=False, _lifecycle_module=mock_lifecycle)

        assert captured["timeout"] == 12.5

        # --- Empty string disables the timeout entirely ---
        record_disabled = _make_record(
            "wt-robocopy-env-disabled", path="C:\\fake\\store\\wt-robocopy-env-disabled"
        )
        manager.state.add(record_disabled)
        monkeypatch.setenv("WORKTREE_ROBOCOPY_TIMEOUT_SEC", "")

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown.os.path.exists",
                side_effect=_make_exists_for(record_disabled.path),
            ),
            patch("lib_python_worktree.core.teardown.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.teardown.subprocess.run", side_effect=_mock_subprocess_run),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            manager._teardown(record_disabled, force=False, _lifecycle_module=mock_lifecycle)

        assert captured["timeout"] is None

    def test_no_regression_preflight_nokill_raises_before_git(self, tmp_path):
        """Regression guard (pins PR #77's Step 2b preflight boundary): with
        kill_blocking_processes=False and a detected blocker, _teardown must
        still raise WorktreeDirLockedError at the preflight, BEFORE either
        `git worktree remove` or the robocopy fallback ever runs. This fix
        must not shift that boundary. Expected to already pass unmodified --
        it is a regression guard, not new behavior."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-preflight-nokill-regression",
            path="/fake/store/wt-preflight-nokill-regression",
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_blockers = [KilledProcessInfo(pid=9999, name="node.exe", cmdline=["node"])]

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=fake_blockers,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes"
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.subprocess.run") as mock_robocopy,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_git.assert_not_called()
        mock_kill.assert_not_called()
        mock_robocopy.assert_not_called()
        err = excinfo.value
        assert err.kill_attempted is False
        assert err.killed == []
        assert err.worktree_id == "wt-preflight-nokill-regression"


# ---------------------------------------------------------------------------
# TestCombinedBlockingConditions -- ticket #103
# ---------------------------------------------------------------------------

class TestCombinedBlockingConditions:
    """Ticket #103: when TWO OR MORE conditions are simultaneously blocking
    a removal -- an OS-level directory lock AND real (non-`.seretos/`)
    uncommitted/untracked changes -- _teardown must report every currently-
    blocking condition and the flag needed to clear each in a SINGLE raise
    (`WorktreeRemovalBlockedError`), so one informed retry (force=True AND
    kill_blocking_processes=True) suffices instead of up to three
    round-trips. A single blocking condition must keep raising the existing
    single-condition exception unchanged."""

    @pytest.mark.parametrize("kill_blocking_processes", [False, True])
    def test_preflight_lock_and_real_dirt_reports_both_conditions(
        self, tmp_path, kill_blocking_processes
    ):
        """Windows preflight (Step 2b) finds a blocker, force=False, and a
        real (non-`.seretos/`) dirt probe finds real dirt: ONE
        WorktreeRemovalBlockedError names both conditions and both
        remedies, with NO kill attempted and NO destructive `git worktree
        remove` call -- regardless of kill_blocking_processes, since the
        removal cannot succeed on this attempt either way (git would still
        refuse for being dirty even after a successful kill)."""
        from lib_python_worktree.core.manager import WorktreeRemovalBlockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-combined-preflight", path="/fake/store/wt-combined-preflight"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_blockers = [KilledProcessInfo(pid=8888, name="node.exe", cmdline=["node"])]
        calls = []

        def _side_effect(args, cwd=None, **kwargs):
            calls.append(list(args))
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? real.txt\0", stderr="")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=fake_blockers,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes"
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeRemovalBlockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=kill_blocking_processes,
                    _lifecycle_module=mock_lifecycle,
                )

        err = excinfo.value
        msg = str(err)
        assert "kill_blocking_processes=True" in msg
        assert "force=True" in msg
        assert err.dirty_paths == ["real.txt"]
        assert err.kill_attempted is False
        assert err.killed == []
        mock_kill.assert_not_called()
        assert not any(
            _is_plain_remove_call(c) or _is_force_remove_call(c) for c in calls
        ), "no destructive git worktree remove call may run for a removal that cannot succeed"

    def test_preflight_lock_force_true_unaffected_no_status_call(self, tmp_path):
        """force=True: real dirt cannot block a forced removal, so the dirt
        probe must never run at all -- unchanged single-condition
        WorktreeDirLockedError, zero `git status` calls."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-combined-force-true", path="/fake/store/wt-combined-force-true"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_blockers = [KilledProcessInfo(pid=7654, name="node.exe", cmdline=["node"])]
        calls = []

        def _side_effect(args, cwd=None, **kwargs):
            calls.append(list(args))
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=fake_blockers,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=True,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert type(excinfo.value) is WorktreeDirLockedError
        assert not any(_is_status_call(c) for c in calls)

    def test_preflight_lock_inconclusive_status_unaffected(self, tmp_path):
        """Windows preflight finds a blocker, force=False, but the status
        probe returns an empty/inconclusive result (per the established
        convention this is NEVER read as "confirmed clean", but it must
        equally never be read as dirty): still the unchanged single-
        condition WorktreeDirLockedError -- the message and type must not
        regress just because the probe now runs."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-combined-clean", path="/fake/store/wt-combined-clean"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_blockers = [KilledProcessInfo(pid=1212, name="node.exe", cmdline=["node"])]

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="", stderr="")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=fake_blockers,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        err = excinfo.value
        assert type(err) is WorktreeDirLockedError
        assert err.kill_attempted is False
        assert "kill_blocking_processes=True" in str(err)

    def test_lock_signal_flag_off_with_real_dirt_reports_both(self, tmp_path):
        """Windows: primary `git worktree remove` fails with a lock signal
        (kill_blocking_processes=False), and a real dirt probe finds real
        dirt: `_resolve_lock_or_raise`'s kill_attempted=False raise site
        also escalates to WorktreeRemovalBlockedError, naming both
        remedies."""
        from lib_python_worktree.core.manager import WorktreeRemovalBlockedError

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-combined-lock-flagoff", path="/fake/store/wt-combined-lock-flagoff"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? scratch.txt\0", stderr="")
            return MagicMock(returncode=255, stderr="Permission denied")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeRemovalBlockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        err = excinfo.value
        assert err.kill_attempted is False
        assert err.dirty_paths == ["scratch.txt"]
        msg = str(err)
        assert "kill_blocking_processes=True" in msg
        assert "force=True" in msg

    def test_posix_lock_stderr_with_real_dirt_reports_both(self, tmp_path):
        """POSIX sibling of the Windows case above: primary `git worktree
        remove` fails with a lock signal (kill_blocking_processes=False),
        and the status probe finds real dirt: same
        WorktreeRemovalBlockedError escalation. POSIX has no Step 2b
        preflight, so this exercises `_resolve_lock_or_raise`'s
        kill_attempted=False site directly."""
        from lib_python_worktree.core.manager import WorktreeRemovalBlockedError

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-combined-lock-posix", path="/fake/store/wt-combined-lock-posix"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? scratch.txt\0", stderr="")
            return MagicMock(returncode=128, stderr="error: unable to lock worktree")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "linux"
            with pytest.raises(WorktreeRemovalBlockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        err = excinfo.value
        assert err.kill_attempted is False
        assert err.dirty_paths == ["scratch.txt"]

    def test_still_locked_after_retry_with_real_dirt_reports_both(self, tmp_path):
        """kill_blocking_processes=True: kill+retry exhausts every attempt
        (still locked), and a real dirt probe finds real dirt: the
        exhausted-retry site (kill_attempted=True) ALSO escalates to
        WorktreeRemovalBlockedError, preserving the "after killing N"
        phrasing inside the combined message."""
        from lib_python_worktree.core.manager import WorktreeRemovalBlockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-combined-stilllocked", path="/fake/store/wt-combined-stilllocked"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_killed = [KilledProcessInfo(pid=3333, name="claude", cmdline=["claude"])]

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? scratch.txt\0", stderr="")
            return MagicMock(returncode=255, stderr="Permission denied")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch("lib_python_worktree.core.teardown.time"),
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeRemovalBlockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=True,
                    _lifecycle_module=mock_lifecycle,
                )

        mock_kill.assert_called_once_with(record.path)
        err = excinfo.value
        assert err.kill_attempted is True
        assert err.killed == fake_killed
        assert err.dirty_paths == ["scratch.txt"]
        assert "after killing 1 blocking process(es)" in str(err)
        assert "force=True" in str(err)

    def test_seretos_retry_lock_signal_type_is_plain_not_combined(self, tmp_path):
        """The `.seretos/`-exemption auto-force retry site (ticket #100)
        routes through the SAME `_resolve_lock_or_raise` helper. When the
        retry itself hits a lock signal but ALL dirt is the benign
        `.seretos/` copy (the only way this branch is even reached), the
        raise must stay the PLAIN WorktreeDirLockedError -- exact type, not
        merely isinstance (which would also accept the combined subclass)
        -- and must not name force=True."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-combined-seretos-retry", path="/fake/store/wt-combined-seretos-retry"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        def _side_effect(args, cwd=None, **kwargs):
            if _is_plain_remove_call(args):
                return MagicMock(returncode=128, stderr=_DIRTY_STDERR)
            if _is_status_call(args):
                return MagicMock(returncode=0, stdout="?? .seretos/\0", stderr="")
            if _is_force_remove_call(args):
                return MagicMock(returncode=255, stderr="Permission denied")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch("lib_python_worktree.core.teardown.shutil"),
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        err = excinfo.value
        assert type(err) is WorktreeDirLockedError
        assert "force=True" not in str(err)

    def test_lock_with_seretos_only_dirt_reports_lock_only(self, tmp_path):
        """A `.seretos/`-only untracked tree (ticket #100's benign
        exemption) alongside a Step-2b-preflight-detected lock must NOT be
        reported as a blocking dirty condition -- plain
        WorktreeDirLockedError (exact type), not the combined
        WorktreeRemovalBlockedError, and no force=True in the message."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-combined-seretos-only", path="/fake/store/wt-combined-seretos-only"
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_blockers = [KilledProcessInfo(pid=2121, name="node.exe", cmdline=["node"])]

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                return MagicMock(
                    returncode=0,
                    stdout="?? .seretos/worktree-setup.yml\0",
                    stderr="",
                )
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=fake_blockers,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        err = excinfo.value
        assert type(err) is WorktreeDirLockedError
        assert "force=True" not in str(err)

    @pytest.mark.parametrize(
        "status_side_effect",
        ["rc_nonzero", "git_timeout", "non_str_stdout", "empty_stdout"],
    )
    def test_inconclusive_dirt_probe_never_escalates_lock_only(
        self, tmp_path, status_side_effect
    ):
        """A failed/inconclusive `git status` probe (non-zero exit, a
        GitTimeoutError, a non-str stdout, or an empty result) must never
        be misread as real dirt -- it must degrade to the single-condition
        WorktreeDirLockedError (exact type), never fabricate the combined
        WorktreeRemovalBlockedError. Mirrors the defensive matrix already
        pinned for the `.seretos/`-exemption classifier
        (TestContractCopyDirtExemption, lines 663-760)."""
        from lib_python_worktree.core.manager import GitTimeoutError, WorktreeDirLockedError
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        manager = _make_manager(tmp_path)
        record = _make_record(
            f"wt-combined-inconclusive-{status_side_effect}",
            path=f"/fake/store/wt-combined-inconclusive-{status_side_effect}",
        )
        manager.state.add(record)

        mock_lifecycle = MagicMock()
        fake_blockers = [KilledProcessInfo(pid=4321, name="node.exe", cmdline=["node"])]

        def _side_effect(args, cwd=None, **kwargs):
            if _is_status_call(args):
                if status_side_effect == "rc_nonzero":
                    return MagicMock(returncode=1, stdout="", stderr="fatal: boom")
                if status_side_effect == "git_timeout":
                    raise GitTimeoutError(["git", *args], 30.0)
                if status_side_effect == "non_str_stdout":
                    return MagicMock(returncode=0, stderr="")  # .stdout stays a MagicMock
                return MagicMock(returncode=0, stdout="", stderr="")  # empty_stdout
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=fake_blockers,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        err = excinfo.value
        assert type(err) is WorktreeDirLockedError
        assert "force=True" not in str(err)

    def test_combined_error_is_catchable_as_both_legacy_types(self):
        """`isinstance` checks against BOTH legacy single-condition types
        succeed, and all four structured attributes (worktree_id, killed,
        kill_attempted, dirty_paths) are populated -- verifying the
        hand-written __init__ really set both parents' attributes without
        relying on a cooperative super() chain."""
        from lib_python_worktree.core.manager import (
            DirtyWorktreeError,
            WorktreeDirLockedError,
            WorktreeRemovalBlockedError,
        )
        from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

        killed = [KilledProcessInfo(pid=42, name="node.exe", cmdline=["node"])]
        err = WorktreeRemovalBlockedError(
            "wt-combined-isinstance",
            killed=killed,
            kill_attempted=True,
            dirty_paths=["a.txt", "b.txt"],
        )

        assert isinstance(err, WorktreeDirLockedError)
        assert isinstance(err, DirtyWorktreeError)
        assert err.worktree_id == "wt-combined-isinstance"
        assert err.killed == killed
        assert err.kill_attempted is True
        assert err.dirty_paths == ["a.txt", "b.txt"]

    def test_combined_message_leaks_no_git_internals(self):
        """Mirrors test_manager.py's DirtyWorktreeError leak-check plus the
        Q3 contract: no path from dirty_paths appears in the free-text
        message -- a caller who wants the actual paths must read the
        structured `dirty_paths` attribute instead."""
        from lib_python_worktree.core.manager import WorktreeRemovalBlockedError

        err = WorktreeRemovalBlockedError(
            "wt-combined-nogitleaks",
            killed=[],
            kill_attempted=False,
            dirty_paths=["secret/path/to/file.txt"],
        )
        msg = str(err)

        assert "--force" not in msg
        assert "128" not in msg
        assert "secret/path/to/file.txt" not in msg

    def test_worktree_removal_blocked_error_exported(self):
        """WorktreeRemovalBlockedError is importable from the top-level
        package and listed in __all__."""
        import lib_python_worktree

        assert hasattr(lib_python_worktree, "WorktreeRemovalBlockedError")
        assert "WorktreeRemovalBlockedError" in lib_python_worktree.__all__


# ---------------------------------------------------------------------------
# #154 (item 8 of plan.md, unchanged by the human override) -- `staged` on
# DirtyWorktreeError / WorktreeDirLockedError / WorktreeRemovalBlockedError,
# and `blockers` on WorktreeDirLockedError / WorktreeRemovalBlockedError.
# ---------------------------------------------------------------------------

class TestStagedAndBlockersAttributes:
    def test_dirty_worktree_error_staged_default_false_message_unchanged(self):
        """staged= must default to False and not perturb today's message
        (R9's byte-identical requirement, extended to DirtyWorktreeError)."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        err = DirtyWorktreeError("wt-staged-default")
        assert err.staged is False
        assert str(err) == (
            "worktree 'wt-staged-default' has uncommitted changes. "
            "Pass force=True to remove it anyway."
        )

    def test_dirty_worktree_error_staged_true_names_marker_not_path(self):
        """#154 item 17 / Decision 1: the dirt-probe undo-failure leg raises
        DirtyWorktreeError(staged=True) -- the message must name the id and
        the `.removing` marker suffix, and must NOT contain a filesystem
        path (per _exceptions.py's documented no-path-leak contract)."""
        from lib_python_worktree.core.manager import DirtyWorktreeError

        err = DirtyWorktreeError("wt-staged-true", staged=True)
        assert err.staged is True
        msg = str(err)
        assert "wt-staged-true" in msg
        assert ".removing" in msg
        assert "/fake/" not in msg and "\fake\\" not in msg

    def test_worktree_dir_locked_error_gains_staged_and_blockers(self):
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        blockers = [
            KilledProcessInfo(pid=123, name="", cmdline=[], source="tracked")
        ]
        err = WorktreeDirLockedError(
            "wt-locked-staged",
            killed=[],
            kill_attempted=False,
            staged=True,
            blockers=blockers,
        )
        assert err.staged is True
        assert err.blockers == blockers
        assert ".removing" in str(err)

    def test_worktree_dir_locked_error_blockers_defaults_to_empty_list_never_none(self):
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        err = WorktreeDirLockedError("wt-locked-default", killed=[])
        assert err.staged is False
        assert err.blockers == []

    def test_worktree_dir_locked_error_staged_false_message_byte_identical_to_v0312(self):
        """R9: the non-staged phrasing, selected by kill_attempted exactly as
        today, must stay byte-identical."""
        from lib_python_worktree.core.manager import WorktreeDirLockedError

        killed = [KilledProcessInfo(pid=1, name="a.exe", cmdline=["a"])]
        err_true = WorktreeDirLockedError("wt-v0312-a", killed=killed, kill_attempted=True)
        assert str(err_true) == (
            "worktree 'wt-v0312-a' directory is still locked after killing"
            " 1 blocking process(es)."
        )
        err_false = WorktreeDirLockedError("wt-v0312-b", killed=[], kill_attempted=False)
        assert str(err_false) == (
            "worktree 'wt-v0312-b' directory is locked by another process. "
            "Pass kill_blocking_processes=True to kill the blocking process(es) "
            "and retry."
        )
        assert err_true.staged is False
        assert err_false.staged is False

    def test_worktree_removal_blocked_error_gains_staged_and_blockers(self):
        """WorktreeRemovalBlockedError bypasses both parents' __init__ and
        must set staged/blockers explicitly like its other attributes, or
        `except WorktreeDirLockedError as e: e.blockers` raises
        AttributeError on this branch."""
        from lib_python_worktree.core.manager import WorktreeRemovalBlockedError

        blockers = [KilledProcessInfo(pid=9, name="", cmdline=[], source="orphan_scan")]
        err = WorktreeRemovalBlockedError(
            "wt-combined-staged",
            killed=[],
            kill_attempted=False,
            dirty_paths=["a.txt"],
            staged=True,
            blockers=blockers,
        )
        assert err.staged is True
        assert err.blockers == blockers

    def test_worktree_removal_blocked_error_staged_and_blockers_default(self):
        from lib_python_worktree.core.manager import WorktreeRemovalBlockedError

        err = WorktreeRemovalBlockedError("wt-combined-default", killed=[])
        assert err.staged is False
        assert err.blockers == []
