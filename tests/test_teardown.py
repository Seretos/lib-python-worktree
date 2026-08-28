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


class TestTeardownContractStopSteps:
    """Verify that contract stop: steps are run inside _teardown before
    kill_blocking_processes, and that failures are swallowed."""

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
