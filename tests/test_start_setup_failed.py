"""Tests for ticket #118: start() must not silently proceed past a
`setup_failed` environment.

`create()` sets `record.status = "setup_failed"` when a contract's `setup:`
step(s) fail, leaving the checkout half-provisioned but otherwise intact.
Before this fix, `start()` would silently overwrite that status to
`"running"` (a real spawn) or `"ready"` (the no-op path), erasing the only
top-level evidence that setup never completed -- the nested, easy-to-miss
`record.setup_outcome.status == "failed"` was all that remained.

This module deliberately does NOT append to `tests/test_manager.py` --
several other branches touch that 2600+-line file concurrently. All tests
here build a `WorktreeRecord(status="setup_failed")` directly in an
in-memory store and patch `lib_python_worktree.core.manager._lifecycle_start`
-- no real process spawn, no git required. Helpers mirror
`tests/test_primary_environment.py` / `tests/test_manager.py`'s
`_make_mgr_in_memory` / `_make_wt_record` pattern, kept module-private here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from lib_python_worktree.core.manager import (
    ManagerConfig,
    SetupIncompleteError,
    WorktreeError,
    WorktreeManager,
)
from lib_python_worktree.core.state import (
    InMemoryStateStore,
    SetupOutcome,
    WorktreeRecord,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mgr_in_memory(tmp_path: Path) -> WorktreeManager:
    return WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )


def _make_wt_record(wt_id: str = "wt-abc12345", **kwargs) -> WorktreeRecord:
    defaults = dict(
        id=wt_id,
        repo_root="/fake/repo",
        branch="feature/x",
        path="/fake/store/wt-abc12345",
    )
    defaults.update(kwargs)
    return WorktreeRecord(**defaults)


def _write_contract(repo_root: Path, text: str) -> None:
    p = repo_root / ".seretos" / "worktree-setup.yml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# R1 -- start() refuses when the contract HAS a start: step
# ---------------------------------------------------------------------------


def test_start_refuses_setup_failed_environment_with_start_step(tmp_path: Path):
    """Driving test: a record with status="setup_failed" whose contract has
    a real start: step must be refused with SetupIncompleteError -- the
    message names the worktree id and "setup_failed" -- and the mocked
    _lifecycle_start must never be called (no silent proceed-to-spawn)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: full\nstart:\n  - run: npm run dev\n")

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), status="setup_failed")
    mgr.state.add(record)

    with patch("lib_python_worktree.core.manager._lifecycle_start") as mock_lifecycle_start:
        with pytest.raises(SetupIncompleteError) as excinfo:
            mgr.start(record.id)

    assert record.id in str(excinfo.value)
    assert "setup_failed" in str(excinfo.value)
    mock_lifecycle_start.assert_not_called()


def test_start_refusal_is_resolution_independent(tmp_path: Path):
    """Edge case: the gate fires regardless of HOW the target was resolved
    -- patching _resolve_target itself to hand back the setup_failed record
    proves the gate is not accidentally coupled to worktree_id-only
    addressing (it must also refuse checkout_path-based resolution)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: full\nstart:\n  - run: npm run dev\n")

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), status="setup_failed")

    with patch.object(mgr, "_resolve_target", return_value=record):
        with patch(
            "lib_python_worktree.core.manager._lifecycle_start"
        ) as mock_lifecycle_start:
            with pytest.raises(SetupIncompleteError):
                mgr.start(checkout_path="/fake/store/wt-abc12345")

    mock_lifecycle_start.assert_not_called()


# ---------------------------------------------------------------------------
# R2 -- start() refuses on the no-op "ready" path too (no start: step)
# ---------------------------------------------------------------------------


def test_start_refuses_setup_failed_on_no_op_ready_path(tmp_path: Path):
    """Driving test: a contract with NO start: step (the no-op "ready"
    branch) must also be refused -- before the fix, this branch would set
    record.status = "ready", laundering the setup_failed status even
    without any process spawn involved."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: none\n")

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), status="setup_failed")
    mgr.state.add(record)

    with pytest.raises(SetupIncompleteError):
        mgr.start(record.id)

    stored = mgr.state.get(record.id)
    assert stored.status == "setup_failed"


def test_start_refuses_setup_failed_with_missing_contract(tmp_path: Path):
    """Edge case: no .seretos/worktree-setup.yml at all (missing contract,
    same as isolation: none) must also be refused."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    # No contract file written at all.

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), status="setup_failed")
    mgr.state.add(record)

    with pytest.raises(SetupIncompleteError):
        mgr.start(record.id)

    stored = mgr.state.get(record.id)
    assert stored.status == "setup_failed"


# ---------------------------------------------------------------------------
# R3 -- a refused start() performs NO state writes (side-effect free)
# ---------------------------------------------------------------------------


def test_start_refusal_performs_no_state_writes(tmp_path: Path):
    """Driving test: a contract declaring a ports: slot must NOT have that
    slot allocated when start() is refused -- the gate runs before port
    allocation, the no-op ready write, and any spawn, so a refused start()
    leaves ports/status/pids completely untouched."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nports:\n  - name: web\nstart:\n  - run: npm run dev\n",
    )

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), status="setup_failed")
    mgr.state.add(record)

    with pytest.raises(SetupIncompleteError):
        mgr.start(record.id)

    stored = mgr.state.get(record.id)
    assert stored.ports == {}
    assert stored.status == "setup_failed"
    assert stored.pids == {}


def test_refused_start_leaves_setup_outcome_untouched(tmp_path: Path):
    """Edge case: an executable form of the state.py invariant that only
    create()'s setup: hook block ever assigns record.setup_outcome --
    start()'s new gate must not read OR write it. Seed a setup_outcome
    verdict and confirm it is bit-for-bit unchanged after refusal."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: full\nstart:\n  - run: npm run dev\n")

    outcome = SetupOutcome(
        status="failed",
        message="step 1 (install) failed: exit 1",
        completed_at="2026-08-19T00:00:00+00:00",
        failed_step_index=0,
        failed_step_name="install",
        log_path="/fake/store/wt-abc12345/.setup-logs/install.log",
        returncode=1,
        timed_out=False,
    )
    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(
        repo_root=str(repo_root), status="setup_failed", setup_outcome=outcome
    )
    mgr.state.add(record)

    with pytest.raises(SetupIncompleteError):
        mgr.start(record.id)

    stored = mgr.state.get(record.id)
    assert stored.setup_outcome == outcome


# ---------------------------------------------------------------------------
# R4 -- allow_setup_failed=True overrides the gate and warns
# ---------------------------------------------------------------------------


def test_start_allow_setup_failed_override_starts_and_warns(tmp_path: Path, caplog):
    """Driving test: allow_setup_failed=True must skip the gate, proceed to
    the real spawn, and log a WARNING naming the worktree as having a
    setup_failed status that was overridden."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: full\nstart:\n  - run: npm run dev\n")

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), status="setup_failed")
    mgr.state.add(record)

    spawned_record = _make_wt_record(
        repo_root=str(repo_root), status="running", pids={"main": 4242}
    )

    caplog.set_level(logging.WARNING, logger="lib_python_worktree.core.manager")

    with patch(
        "lib_python_worktree.core.manager._lifecycle_start",
        return_value=spawned_record,
    ) as mock_lifecycle_start:
        result = mgr.start(record.id, allow_setup_failed=True)

    mock_lifecycle_start.assert_called_once()
    assert result.status == "running"
    assert any(
        "setup_failed" in rec.message for rec in caplog.records
    ), "allow_setup_failed=True must log a WARNING mentioning setup_failed"


def test_start_allow_setup_failed_override_on_no_op_ready_path(tmp_path: Path, caplog):
    """Edge case: the override also works on the no-op ready path (no
    start: step configured) -- returns status="ready" and still warns."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: none\n")

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), status="setup_failed")
    mgr.state.add(record)

    caplog.set_level(logging.WARNING, logger="lib_python_worktree.core.manager")

    result = mgr.start(record.id, allow_setup_failed=True)

    assert result.status == "ready"
    assert any(
        "setup_failed" in rec.message for rec in caplog.records
    ), "allow_setup_failed=True must log a WARNING mentioning setup_failed"


# ---------------------------------------------------------------------------
# R5 -- regression armour: every OTHER status must reach _lifecycle_start
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    ["created", "ready", "stopped", "stop_incomplete", "orphaned", "exited"],
)
def test_start_not_refused_for_other_statuses(tmp_path: Path, status: str):
    """Regression armour: the gate must only fire for status ==
    "setup_failed" -- every other known status must reach the mocked
    _lifecycle_start unimpeded. Passes before and after this fix."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: full\nstart:\n  - run: npm run dev\n")

    mgr = _make_mgr_in_memory(tmp_path)
    record = _make_wt_record(repo_root=str(repo_root), status=status)
    mgr.state.add(record)

    spawned_record = _make_wt_record(
        repo_root=str(repo_root), status="running", pids={"main": 4242}
    )

    with patch(
        "lib_python_worktree.core.manager._lifecycle_start",
        return_value=spawned_record,
    ) as mock_lifecycle_start:
        result = mgr.start(record.id)

    mock_lifecycle_start.assert_called_once()
    assert result.status == "running"


def test_setup_incomplete_error_is_worktree_error_and_exported():
    """Edge case: SetupIncompleteError must subclass WorktreeError (so
    existing broad `except WorktreeError` handlers still catch it) and must
    be exported from manager.__all__ specifically -- deliberately NOT
    checking lib_python_worktree.__all__, since ticket #118 explicitly does
    not touch lib_python_worktree/__init__.py."""
    from lib_python_worktree.core import manager as manager_module

    assert issubclass(SetupIncompleteError, WorktreeError)
    assert "SetupIncompleteError" in manager_module.__all__
