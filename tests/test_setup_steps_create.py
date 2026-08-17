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
from lib_python_worktree.core.state import (
    SETUP_STATUS_COMPLETED,
    SETUP_STATUS_FAILED,
    SETUP_STATUS_SKIPPED,
    InMemoryStateStore,
    WorktreeRecord,
)
from lib_python_worktree.setup.runner import SetupFailedError, SetupResult, SetupStepResult


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

        def _record_call(**kw):
            runner_calls.append(kw)
            return SetupResult(worktree_id=kw["worktree_id"], steps=[], aborted_at=None)

        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = _record_call

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


# ---------------------------------------------------------------------------
# TestCreateRecordsSetupOutcome (ticket #105)
# ---------------------------------------------------------------------------

class TestCreateRecordsSetupOutcome:
    """create() must record a persisted, first-class ``setup_outcome``
    verdict on the ``setup:`` hook -- independent of ``status``.
    """

    def test_create_records_setup_completed(self, tmp_path):
        """Every setup: step succeeding must persist setup_outcome.status
        == 'completed' with the right steps_run, leaving status untouched.
        """
        manager = _make_manager(tmp_path)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            setup=[
                Step(run='echo "a"', name="a"),
                Step(run='echo "b"', name="b"),
            ],
        )
        fake_result = SetupResult(
            worktree_id="ignored",
            steps=[
                SetupStepResult(0, "a", 0, Path("/tmp/a.log")),
                SetupStepResult(1, "b", 0, Path("/tmp/b.log")),
            ],
            aborted_at=None,
        )
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = fake_result

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
            record = manager.create("/fake/repo", "feature/setup-completed")

        persisted = manager.state.get(record.id)
        assert persisted is not None
        assert persisted.setup_outcome is not None, (
            "setup_outcome must be persisted (not just on the returned object)"
        )
        assert persisted.setup_outcome.status == SETUP_STATUS_COMPLETED
        assert persisted.setup_outcome.steps_run == 2
        assert persisted.setup_outcome.completed_at is not None
        # status is unrelated bookkeeping and must stay "created".
        assert persisted.status == "created"

    def test_create_records_setup_skipped_when_no_steps(self, tmp_path):
        """A contract with no setup: steps must persist setup_outcome.status
        == 'skipped' with steps_run == 0, and a bare WorktreeRecord never
        passed through create() must have setup_outcome is None -- proving
        "skipped" and None are distinct.
        """
        manager = _make_manager(tmp_path)

        fake_contract = WorktreeContract(version=1, isolation="full", setup=[])
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
            record = manager.create("/fake/repo", "feature/setup-skipped")

        mock_runner_instance.run.assert_not_called()
        persisted = manager.state.get(record.id)
        assert persisted is not None
        assert persisted.setup_outcome is not None
        assert persisted.setup_outcome.status == SETUP_STATUS_SKIPPED
        assert persisted.setup_outcome.steps_run == 0

        # A record never passed through create() has setup_outcome is None.
        bare = WorktreeRecord(id="x", repo_root="/r", branch="b", path="/p")
        assert bare.setup_outcome is None

    @pytest.mark.parametrize(
        "contract_fixture",
        ["missing_file", "empty_file", "explicit_empty_setup"],
    )
    def test_create_records_setup_skipped_for_all_no_op_contract_shapes(
        self, tmp_path, contract_fixture
    ):
        """All three "nothing to run" contract shapes -- no contract file,
        an empty contract file, and an explicit `setup: []` -- must reach
        setup_outcome.status == 'skipped' via the REAL contract loader (not
        a mock), since create() loads the contract from repo_path itself.
        """
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        seretos_dir = repo_root / ".seretos"

        if contract_fixture == "missing_file":
            pass  # no .seretos directory at all
        elif contract_fixture == "empty_file":
            seretos_dir.mkdir()
            (seretos_dir / "worktree-setup.yml").write_text("", encoding="utf-8")
        elif contract_fixture == "explicit_empty_setup":
            seretos_dir.mkdir()
            (seretos_dir / "worktree-setup.yml").write_text(
                "version: 1\nisolation: full\nsetup: []\n", encoding="utf-8"
            )

        manager = _make_manager(tmp_path)
        mock_runner_instance = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.manager._run_git",
                side_effect=_fake_git_success,
            ),
            patch.object(manager, "_validate_repo", return_value=repo_root),
            patch.object(manager, "_branch_exists", return_value=True),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
        ):
            record = manager.create(str(repo_root), f"feature/{contract_fixture}")

        mock_runner_instance.run.assert_not_called()
        persisted = manager.state.get(record.id)
        assert persisted.setup_outcome is not None
        assert persisted.setup_outcome.status == SETUP_STATUS_SKIPPED
        assert persisted.setup_outcome.steps_run == 0

    def test_create_records_setup_failed_with_detail(self, tmp_path):
        """A SetupFailedError must persist setup_outcome.status == 'failed'
        with the failure detail matching the raised error, and status must
        still be 'setup_failed'.
        """
        manager = _make_manager(tmp_path)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            setup=[Step(run="exit 1", name="failing-step")],
        )
        fake_log_path = Path("/tmp/fake-setup.log")
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
            with pytest.raises(SetupFailedError) as excinfo:
                manager.create("/fake/repo", "feature/setup-failed-detail")

        worktree_id = captured_id[0]
        persisted = manager.state.get(worktree_id)
        assert persisted.status == "setup_failed"
        assert persisted.setup_outcome is not None
        outcome = persisted.setup_outcome
        assert outcome.status == SETUP_STATUS_FAILED
        assert outcome.message == str(excinfo.value)
        assert outcome.failed_step_index == 0
        assert outcome.failed_step_name == "failing-step"
        assert outcome.log_path == fake_log_path.as_posix()
        assert outcome.returncode == 1
        assert outcome.timed_out is False

    def test_create_records_setup_failed_timeout(self, tmp_path):
        """A SetupFailedError carrying a timeout must persist timed_out=True."""
        manager = _make_manager(tmp_path)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            setup=[Step(run="sleep 100", name="slow-step")],
        )
        fake_log_path = Path("/tmp/fake-timeout.log")
        captured_id: list[str] = []

        def _raise_setup_failed(**kw):
            captured_id.append(kw["worktree_id"])
            raise SetupFailedError(
                worktree_id=kw["worktree_id"],
                step_index=0,
                step_name="slow-step",
                log_path=fake_log_path,
                returncode=-1,
                timeout=5.0,
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
                manager.create("/fake/repo", "feature/setup-failed-timeout")

        worktree_id = captured_id[0]
        persisted = manager.state.get(worktree_id)
        assert persisted.setup_outcome.timed_out is True

    def test_create_records_setup_failed_non_setup_failed_exception(self, tmp_path):
        """A non-SetupFailedError exception must still yield
        setup_outcome.status == 'failed' with step-detail fields left at
        their None defaults, and must propagate unchanged.
        """
        manager = _make_manager(tmp_path)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            setup=[Step(run="whatever", name="whatever-step")],
        )
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.side_effect = RuntimeError("boom")

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
            with pytest.raises(RuntimeError, match="boom"):
                manager.create("/fake/repo", "feature/setup-failed-runtimeerror")

        # Find the record created just before the raise (only one record
        # tracked by this manager instance).
        all_records = manager.state.list()
        assert len(all_records) == 1
        persisted = all_records[0]
        assert persisted.status == "setup_failed"
        assert persisted.setup_outcome is not None
        outcome = persisted.setup_outcome
        assert outcome.status == SETUP_STATUS_FAILED
        assert outcome.message == "RuntimeError: boom"
        assert outcome.failed_step_index is None
        assert outcome.failed_step_name is None
        assert outcome.log_path is None
        assert outcome.returncode is None
        assert outcome.timed_out is False

    def test_create_records_setup_failed_multi_step_index(self, tmp_path):
        """failed_step_index must point at the actual failing step in a
        multi-step contract, not always 0.
        """
        manager = _make_manager(tmp_path)

        fake_contract = WorktreeContract(
            version=1,
            isolation="full",
            setup=[
                Step(run="echo a", name="a"),
                Step(run="echo b", name="b"),
                Step(run="exit 1", name="c-fails"),
            ],
        )
        fake_log_path = Path("/tmp/fake-multi.log")
        captured_id: list[str] = []

        def _raise_setup_failed(**kw):
            captured_id.append(kw["worktree_id"])
            raise SetupFailedError(
                worktree_id=kw["worktree_id"],
                step_index=2,
                step_name="c-fails",
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
                manager.create("/fake/repo", "feature/setup-failed-multi-step")

        worktree_id = captured_id[0]
        persisted = manager.state.get(worktree_id)
        assert persisted.setup_outcome.failed_step_index == 2
        assert persisted.setup_outcome.failed_step_name == "c-fails"
