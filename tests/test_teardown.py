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
from unittest.mock import MagicMock, call, patch

import pytest

from lib_python_worktree.core.manager import WorktreeManager, ManagerConfig
from lib_python_worktree.core.process_lifecycle import (
    ProcessNotRunningError,
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


def _run_teardown_with_mocked_git(
    manager: WorktreeManager,
    record: WorktreeRecord,
    *,
    force: bool = False,
    lifecycle_module=None,
) -> None:
    """Call _teardown with git subprocess mocked to succeed."""
    with patch(
        "lib_python_worktree.core.manager._run_git"
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
                "lib_python_worktree.core.manager._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.manager._run_git"
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
                "lib_python_worktree.core.manager._run_git"
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
                "lib_python_worktree.core.manager._load_contract",
                side_effect=RuntimeError("disk error"),
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.manager._run_git"
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
                "lib_python_worktree.core.manager._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.manager._run_git"
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
            "lib_python_worktree.core.manager._run_git",
            side_effect=self._make_mock_git_128(),
        ), patch("lib_python_worktree.core.manager.shutil"):
            # Should complete without raising.
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)

    def test_force_exit128_calls_rmtree(self, tmp_path):
        """shutil.rmtree must be called with record.path and ignore_errors=True."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-rmtree", path="/fake/store/wt-rmtree")
        manager.state.add(record)

        mock_lifecycle = MagicMock()

        with patch(
            "lib_python_worktree.core.manager._run_git",
            side_effect=self._make_mock_git_128(),
        ), patch(
            "lib_python_worktree.core.manager.shutil"
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
            "lib_python_worktree.core.manager._run_git",
            side_effect=_tracking_git,
        ), patch("lib_python_worktree.core.manager.shutil"):
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
            "lib_python_worktree.core.manager._run_git",
            side_effect=self._make_mock_git_128(),
        ), patch("lib_python_worktree.core.manager.shutil"):
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
            "lib_python_worktree.core.manager._run_git",
            side_effect=self._make_mock_git_128(),
        ), patch(
            "lib_python_worktree.core.manager.shutil"
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
            "lib_python_worktree.core.manager._run_git",
            side_effect=_git_exit1,
        ), patch("lib_python_worktree.core.manager.shutil"):
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
            "lib_python_worktree.core.manager._run_git",
            side_effect=_git_exit128,
        ), patch("lib_python_worktree.core.manager.shutil"):
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
            "lib_python_worktree.core.manager._run_git",
            side_effect=_git_not_a_repo,
        ), patch("lib_python_worktree.core.manager.shutil"):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)


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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_perm_denied),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_always_fail),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
            patch("lib_python_worktree.core.manager.time"),
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_always_fail),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
            patch("lib_python_worktree.core.manager.time"),
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_broken_repo),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_lock_and_dirty),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_perm_denied),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_invalid_arg),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_locked),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_always_fail),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
            patch("lib_python_worktree.core.manager.time"),
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_rc1),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_perm_denied),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            removed = manager.remove(record.id, kill_blocking_processes=True)

        assert removed.killed_pids == fake_killed

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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            removed = manager.remove(record.id, kill_blocking_processes=True)

        # The critical assertion: YamlStateStore deserializes a fresh object,
        # so without the explicit copy in remove() this would be [].
        assert removed.killed_pids == fake_killed, (
            "killed_pids must survive YamlStateStore round-trip via remove()"
        )


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
                "lib_python_worktree.core.manager._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                side_effect=_mock_kill,
            ),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
                "lib_python_worktree.core.manager._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.manager._run_git") as mock_git,
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
                "lib_python_worktree.core.manager._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.manager._run_git") as mock_git,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        # No call should have been made with setup=[] (stop field empty)
        for setup_arg in runner_stop_calls:
            assert setup_arg != [], (
                "SetupRunner.run must not be called with empty stop: list"
            )


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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
        # each of which calls Path.exists() on the contract file — resulting in
        # 2 calls before the long-path guard at line 897.
        # We must return False for those contract-file checks and True only for
        # the actual worktree-path check at line 897.
        # Strategy: return False for any path that is NOT record.path; return
        # True for record.path on the first check (line 897) and False on the
        # second (line 920, the final guard after rmtree succeeds).
        _path_calls = {"n": 0}

        def _mock_exists(path):
            if str(path) == record.path:
                _path_calls["n"] += 1
                return _path_calls["n"] == 1  # True first, False thereafter
            # Contract file / other paths → not present.
            return False

        def _mock_rmtree(path, **kwargs):
            rmtree_calls.append(path)

        with (
            patch("lib_python_worktree.core.manager._run_git") as mock_git,
            patch("lib_python_worktree.core.manager.os.path.exists", side_effect=_mock_exists),
            patch("lib_python_worktree.core.manager.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
        # Returns True only for the first check of record.path (line 897),
        # and False for all other paths and all subsequent checks of record.path.
        _path_calls = {"n": 0}

        def _mock_exists(path):
            if str(path) == record.path:
                _path_calls["n"] += 1
                return _path_calls["n"] == 1
            return False

        def _mock_rmtree(path, **kwargs):
            rmtree_calls.append(path)

        with (
            patch("lib_python_worktree.core.manager._run_git") as mock_git,
            patch("lib_python_worktree.core.manager.os.path.exists", side_effect=_mock_exists),
            patch("lib_python_worktree.core.manager.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
        # Returns True only for the first check of record.path (line 897),
        # and False for all other paths and all subsequent checks of record.path.
        _path_calls = {"n": 0}

        def _mock_exists(path):
            if str(path) == record.path:
                _path_calls["n"] += 1
                return _path_calls["n"] == 1
            return False

        with (
            patch("lib_python_worktree.core.manager._run_git") as mock_git,
            patch("lib_python_worktree.core.manager.os.path.exists", side_effect=_mock_exists),
            patch("lib_python_worktree.core.manager.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.manager.subprocess.run", side_effect=_mock_subprocess_run),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git") as mock_git,
            patch("lib_python_worktree.core.manager.os.path.exists", return_value=False),
            patch("lib_python_worktree.core.manager.shutil.rmtree", side_effect=lambda *a, **kw: rmtree_calls.append(a)),
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.time") as mock_time,
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_tracking_git),
            patch(
                "lib_python_worktree.core.manager._kill_blocking_processes",
                return_value=fake_killed,
            ) as mock_kill,
            patch("lib_python_worktree.core.manager.shutil.rmtree", side_effect=_mock_rmtree),
            patch("lib_python_worktree.core.manager.time"),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch("lib_python_worktree.core.manager.shutil.rmtree", side_effect=_mock_rmtree),
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
            patch("lib_python_worktree.core.manager._run_git", side_effect=_git_side_effect),
            patch("lib_python_worktree.core.manager.shutil"),
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
            patch("lib_python_worktree.core.manager._run_git") as mock_git,
            patch("lib_python_worktree.core.manager.os.path.exists", return_value=False),
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
            patch("lib_python_worktree.core.manager._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.manager.os.path.exists",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.manager.shutil.rmtree",
                side_effect=OSError("path too long / locked"),
            ),
            patch("lib_python_worktree.core.manager.subprocess.run", return_value=MagicMock(returncode=1)),
            patch("lib_python_worktree.core.manager.sys") as mock_sys,
        ):
            mock_git.return_value = MagicMock(returncode=0, stderr="")
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as exc_info:
                manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        err = exc_info.value
        assert err.worktree_id == "wt-still-locked", (
            f"WorktreeDirLockedError must carry the worktree id, got {err.worktree_id!r}"
        )
