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
