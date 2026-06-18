"""Tests for setup: contract steps executed by WorktreeManager.create() (ticket #55).

Verifies that:
- create() runs setup: steps from the contract via SetupRunner after the state
  record is persisted.
- create() skips SetupRunner entirely when the contract has no setup: steps.
- create() marks the record status="setup_failed" and re-raises SetupFailedError
  when a setup step fails, leaving the worktree, ports, and state intact.

Uses InMemoryStateStore and mocks; no real git required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lib_python_worktree.contract.schema import Step, WorktreeContract
from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord
from lib_python_worktree.setup.runner import SetupFailedError


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


def _fake_git_success(args, cwd=None, **kwargs):
    """Stub for _run_git that always returns returncode=0."""
    return MagicMock(returncode=0, stderr="", stdout="")


def _fake_validate_repo(repo_root: str) -> Path:
    """Stub for _validate_repo that returns a predictable Path."""
    return Path("/fake/repo")


# ---------------------------------------------------------------------------
# TestCreateRunsSetupSteps
# ---------------------------------------------------------------------------

class TestCreateRunsSetupSteps:
    """create() must invoke SetupRunner.run() when the contract has setup: steps."""

    def test_create_runs_setup_steps(self, tmp_path):
        """Regression #55: SetupRunner.run is called with the expected kwargs."""
        manager = _make_manager(tmp_path)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            setup=[Step(run='echo "hello"', name="greet")],
        )

        runner_calls = []
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = lambda **kw: runner_calls.append(kw)

        with (
            patch(
                "lib_python_worktree.core.manager._run_git",
                side_effect=_fake_git_success,
            ),
            patch.object(manager, "_validate_repo", return_value=Path("/fake/repo")),
            patch.object(manager, "_branch_exists", return_value=True),
            patch(
                "lib_python_worktree.core.manager._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
        ):
            record = manager.create("/fake/repo", "feature/setup-test")

        # SetupRunner.run must have been called exactly once.
        assert len(runner_calls) == 1, (
            f"Expected SetupRunner.run called once, got {len(runner_calls)}"
        )
        kw = runner_calls[0]
        assert kw["worktree_id"] == record.id
        assert kw["setup"] == fake_contract.setup
        assert kw["worktree_path"] == Path(record.path)
        assert kw["branch"] == record.branch
        assert kw["port_mapping"] == record.ports


# ---------------------------------------------------------------------------
# TestCreateSkipsSetupWhenNoSteps
# ---------------------------------------------------------------------------

class TestCreateSkipsSetupWhenNoSteps:
    """create() must not invoke SetupRunner when setup: is empty."""

    def test_create_skips_setup_when_no_steps(self, tmp_path):
        """Regression #55: SetupRunner.run is never called for an empty setup list."""
        manager = _make_manager(tmp_path)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            setup=[],  # no setup steps
        )

        mock_runner_instance = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.manager._run_git",
                side_effect=_fake_git_success,
            ),
            patch.object(manager, "_validate_repo", return_value=Path("/fake/repo")),
            patch.object(manager, "_branch_exists", return_value=True),
            patch(
                "lib_python_worktree.core.manager._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
        ):
            manager.create("/fake/repo", "feature/no-setup")

        mock_runner_instance.run.assert_not_called()


# ---------------------------------------------------------------------------
# TestCreateSetupFailureMarksSetupFailed
# ---------------------------------------------------------------------------

class TestCreateSetupFailureMarksSetupFailed:
    """create() must set record.status='setup_failed' and re-raise on step failure."""

    def test_create_setup_failure_marks_setup_failed_and_reraises(self, tmp_path):
        """Regression #55: on SetupFailedError, status is persisted and exception propagates."""
        manager = _make_manager(tmp_path)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            setup=[Step(run='exit 1', name="failing-step")],
        )

        fake_log_path = Path("/tmp/fake-setup.log")
        # We cannot know worktree_id in advance; we capture it from state.
        captured_id: list[str] = []

        def _raise_setup_failed(**kw):
            captured_id.append(kw["worktree_id"])
            raise SetupFailedError(
                worktree_id=kw["worktree_id"],
                step_index=0,
                step_name="failing-step",
                log_path=fake_log_path,
                returncode=1,
            )

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = _raise_setup_failed

        with (
            patch(
                "lib_python_worktree.core.manager._run_git",
                side_effect=_fake_git_success,
            ),
            patch.object(manager, "_validate_repo", return_value=Path("/fake/repo")),
            patch.object(manager, "_branch_exists", return_value=True),
            patch(
                "lib_python_worktree.core.manager._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
        ):
            with pytest.raises(SetupFailedError):
                manager.create("/fake/repo", "feature/setup-fail")

        # The record must still exist in state (worktree left intact).
        assert len(captured_id) == 1
        worktree_id = captured_id[0]
        persisted = manager.state.get(worktree_id)
        assert persisted is not None, (
            "State record must not be removed on setup failure — worktree left intact."
        )
        assert persisted.status == "setup_failed", (
            f"Expected status='setup_failed', got {persisted.status!r}"
        )
