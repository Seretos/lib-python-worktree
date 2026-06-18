"""Tests for the port allocator (W4).

All tests use ``tmp_path`` or mocks — no test ever touches ``~/.agent-worktree``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from lib_python_worktree.core.port_allocator import (
    PortAllocationError,
    PortAllocator,
    _NoOpPortAllocator,
)
from lib_python_worktree.core.yaml_store import _PortsFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_allocator(tmp_path: Path, port_range: tuple = (30000, 30099)) -> PortAllocator:
    """Build a PortAllocator backed by a fresh ports.yaml in tmp_path."""
    ports_path = tmp_path / "ports.yaml"
    pf = _PortsFile(ports_path)
    return PortAllocator(pf, port_range=port_range)


def _make_ports_file(tmp_path: Path) -> _PortsFile:
    return _PortsFile(tmp_path / "ports.yaml")


# ---------------------------------------------------------------------------
# Basic allocation tests
# ---------------------------------------------------------------------------

def test_allocate_returns_nonempty_mapping(tmp_path: Path):
    """Primary acceptance test: allocate returns a non-empty mapping."""
    allocator = _make_allocator(tmp_path)
    result = allocator.allocate(["web", "db"], "wt-abc123")
    assert isinstance(result, dict)
    assert set(result.keys()) == {"web", "db"}
    assert all(30000 <= port <= 30099 for port in result.values())


def test_allocate_no_duplicate_within_call(tmp_path: Path):
    """No two slots within a single allocate call share a port."""
    allocator = _make_allocator(tmp_path, port_range=(30000, 30099))
    result = allocator.allocate(["a", "b", "c", "d", "e"], "wt-dup")
    ports = list(result.values())
    assert len(ports) == len(set(ports)), "duplicate ports within one call"


def test_allocate_no_reuse_across_calls(tmp_path: Path):
    """Ports allocated in a first call are not reused in a second call."""
    allocator = _make_allocator(tmp_path, port_range=(30000, 30099))
    first = allocator.allocate(["web"], "wt-first")
    second = allocator.allocate(["web"], "wt-second")
    assert first["web"] != second["web"], "same port reused across two calls"


def test_allocate_skips_already_recorded_port(tmp_path: Path):
    """Ports already in ports.yaml (from another worktree) are skipped."""
    pf = _make_ports_file(tmp_path)
    # Pre-populate port 30000 for another worktree.
    pf.set_all({"wt-other:web": 30000})

    allocator = PortAllocator(pf, port_range=(30000, 30001))
    # Only 30001 is free.
    result = allocator.allocate(["api"], "wt-new")
    assert result["api"] == 30001


def test_allocate_skips_port_in_use(tmp_path: Path):
    """Ports that _port_in_use returns True for are skipped."""
    allocator = _make_allocator(tmp_path, port_range=(30000, 30001))
    # Mock _port_in_use to report 30000 as in-use.
    with patch(
        "lib_python_worktree.core.port_allocator._port_in_use",
        side_effect=lambda p: p == 30000,
    ):
        result = allocator.allocate(["svc"], "wt-busy")
    assert result["svc"] == 30001


def test_allocate_raises_when_range_exhausted(tmp_path: Path):
    """PortAllocationError is raised when every port in the range is taken."""
    pf = _make_ports_file(tmp_path)
    # Occupy the single port in the range.
    pf.set_all({"wt-other:api": 30000})

    allocator = PortAllocator(pf, port_range=(30000, 30000))
    with pytest.raises(PortAllocationError, match="No free port"):
        allocator.allocate(["api"], "wt-fail")


def test_allocate_raises_when_range_exhausted_os_busy(tmp_path: Path):
    """PortAllocationError raised when the only port in range is OS-busy."""
    allocator = _make_allocator(tmp_path, port_range=(30000, 30000))
    with patch(
        "lib_python_worktree.core.port_allocator._port_in_use",
        return_value=True,
    ):
        with pytest.raises(PortAllocationError):
            allocator.allocate(["svc"], "wt-os-busy")


def test_allocate_empty_slot_list(tmp_path: Path):
    """Empty slot list returns {} without touching ports.yaml."""
    pf = _make_ports_file(tmp_path)
    allocator = PortAllocator(pf, port_range=(30000, 30099))
    result = allocator.allocate([], "wt-empty")
    assert result == {}
    # ports.yaml should NOT have been created.
    assert not (tmp_path / "ports.yaml").exists()


def test_port_range_boundary(tmp_path: Path):
    """Range (30000, 30001) — two distinct slots get exactly those two ports."""
    allocator = _make_allocator(tmp_path, port_range=(30000, 30001))
    result = allocator.allocate(["x", "y"], "wt-boundary")
    assert set(result.values()) == {30000, 30001}


# ---------------------------------------------------------------------------
# Release tests
# ---------------------------------------------------------------------------

def test_release_removes_entries_for_worktree(tmp_path: Path):
    """release() removes all entries for the given worktree."""
    allocator = _make_allocator(tmp_path, port_range=(30000, 30099))
    allocator.allocate(["web", "db"], "wt-release")

    allocator.release("wt-release")

    pf = _make_ports_file(tmp_path)
    remaining = pf.get_all()
    assert not any(k.startswith("wt-release:") for k in remaining)


def test_release_does_not_touch_other_worktrees(tmp_path: Path):
    """release() leaves entries for other worktrees intact."""
    allocator = _make_allocator(tmp_path, port_range=(30000, 30099))
    allocator.allocate(["web"], "wt-keep")
    allocator.allocate(["api"], "wt-remove")

    allocator.release("wt-remove")

    pf = _make_ports_file(tmp_path)
    remaining = pf.get_all()
    assert any(k.startswith("wt-keep:") for k in remaining)
    assert not any(k.startswith("wt-remove:") for k in remaining)


def test_release_idempotent(tmp_path: Path):
    """Calling release() twice on the same worktree id is a no-op."""
    allocator = _make_allocator(tmp_path, port_range=(30000, 30099))
    allocator.allocate(["web"], "wt-idem")

    allocator.release("wt-idem")
    # Second release must not raise.
    allocator.release("wt-idem")

    pf = _make_ports_file(tmp_path)
    remaining = pf.get_all()
    assert not any(k.startswith("wt-idem:") for k in remaining)


def test_release_on_empty_ports_file_is_safe(tmp_path: Path):
    """release() when no ports.yaml exists must not raise."""
    allocator = _make_allocator(tmp_path)
    allocator.release("wt-never-allocated")  # must not raise


# ---------------------------------------------------------------------------
# Concurrency test
# ---------------------------------------------------------------------------

def test_allocate_concurrent_no_duplicate(tmp_path: Path):
    """Two threads allocating from a small range must not produce duplicates."""
    pf = _make_ports_file(tmp_path)
    # Range of 20 ports; each thread takes 1 → plenty of room, but small enough
    # that without the lock they could collide.
    allocator = PortAllocator(pf, port_range=(30000, 30019))

    results: List[Dict[str, int]] = []
    errors: List[Exception] = []

    def _do_alloc(wt_id: str) -> None:
        try:
            r = allocator.allocate(["svc"], wt_id)
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=_do_alloc, args=(f"wt-thread-{i}",))
        for i in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread(s) raised: {errors}"
    all_ports = [r["svc"] for r in results]
    assert len(all_ports) == len(set(all_ports)), f"Duplicate ports: {all_ports}"


# ---------------------------------------------------------------------------
# Manager integration tests (mock-based, no git)
# ---------------------------------------------------------------------------

def test_manager_create_populates_record_ports(tmp_path: Path):
    """WorktreeManager.create populates record.ports via the allocator."""
    from unittest.mock import MagicMock, patch

    from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
    from lib_python_worktree.core.state import InMemoryStateStore

    store = InMemoryStateStore()
    cfg = ManagerConfig(store_root=tmp_path / "store")
    mgr = WorktreeManager(config=cfg, state=store, reconcile_on_init=False)

    # Replace the no-op allocator with a mock that returns a fixed mapping.
    mock_allocator = MagicMock()
    mock_allocator.allocate.return_value = {"web": 31000, "db": 31001}
    mock_allocator.release.return_value = None
    mgr._allocator = mock_allocator

    # Mock out all git interaction AND the contract loader.
    with (
        patch(
            "lib_python_worktree.core.manager._run_git",
            return_value=MagicMock(returncode=0, stdout="/fake/repo\n", stderr=""),
        ),
        patch(
            "lib_python_worktree.core.manager._load_contract",
        ) as mock_load,
    ):
        # Contract with two port slots (no setup steps so SetupRunner is skipped).
        from lib_python_worktree.contract.schema import PortSlot, WorktreeContract
        mock_contract = MagicMock(spec=WorktreeContract)
        mock_contract.ports = [PortSlot(name="web"), PortSlot(name="db")]
        mock_contract.setup = []  # no setup steps — avoids SetupRunner invocation
        mock_load.return_value = mock_contract

        # Patch _validate_repo to return the tmp_path (it's "a git repo").
        with patch.object(
            mgr, "_validate_repo", return_value=tmp_path / "fake_repo"
        ):
            # _branch_exists → False means we need a base, so patch to True.
            with patch.object(mgr, "_branch_exists", return_value=True):
                # The git worktree add call is already mocked via _run_git.
                # But target_path.parent.mkdir may fail; use a real tmp dir.
                (tmp_path / "store" / "fake-repo").mkdir(parents=True, exist_ok=True)
                with patch(
                    "lib_python_worktree.core.manager.Path.mkdir",
                ):
                    record = mgr.create(str(tmp_path / "fake_repo"), "feature/x")

    assert record.ports == {"web": 31000, "db": 31001}
    mock_allocator.allocate.assert_called_once_with(["web", "db"], record.id)


def test_manager_remove_calls_release(tmp_path: Path):
    """WorktreeManager._teardown calls allocator.release with the worktree id."""
    from unittest.mock import MagicMock, patch

    from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
    from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord

    store = InMemoryStateStore()
    cfg = ManagerConfig(store_root=tmp_path / "store")
    mgr = WorktreeManager(config=cfg, state=store, reconcile_on_init=False)

    mock_allocator = MagicMock()
    mock_allocator.release.return_value = None
    mgr._allocator = mock_allocator

    # Pre-populate state with a record so remove() finds it.
    record = WorktreeRecord(
        id="wt-spy-abc123",
        repo_root=str(tmp_path),
        branch="feature/spy",
        path=str(tmp_path / "wt"),
    )
    store.add(record)

    with patch(
        "lib_python_worktree.core.manager._run_git",
        return_value=MagicMock(returncode=0, stdout="", stderr=""),
    ):
        with patch.object(mgr, "_delete_owned_branch"):
            mgr._teardown(record, force=False)

    mock_allocator.release.assert_called_once_with("wt-spy-abc123")


# ---------------------------------------------------------------------------
# ManagerConfig port_range env tests
# ---------------------------------------------------------------------------

def test_manager_config_port_range_default(monkeypatch):
    """ManagerConfig.from_env defaults port_range to (30000, 40000)."""
    from lib_python_worktree.core.manager import ManagerConfig
    monkeypatch.delenv("WORKTREE_PORT_RANGE", raising=False)
    cfg = ManagerConfig.from_env()
    assert cfg.port_range == (30000, 40000)


def test_manager_config_port_range_from_env(monkeypatch):
    """ManagerConfig.from_env reads WORKTREE_PORT_RANGE correctly."""
    monkeypatch.setenv("WORKTREE_PORT_RANGE", "20000-25000")
    from lib_python_worktree.core.manager import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert cfg.port_range == (20000, 25000)


def test_manager_config_port_range_invalid_env_falls_back(monkeypatch):
    """ManagerConfig.from_env falls back to default for malformed env value."""
    monkeypatch.setenv("WORKTREE_PORT_RANGE", "not-a-range")
    from lib_python_worktree.core.manager import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert cfg.port_range == (30000, 40000)


# ---------------------------------------------------------------------------
# _NoOpPortAllocator
# ---------------------------------------------------------------------------

def test_no_op_allocator_returns_empty():
    noop = _NoOpPortAllocator()
    assert noop.allocate(["x", "y"], "wt-noop") == {}


def test_no_op_allocator_release_is_silent():
    noop = _NoOpPortAllocator()
    noop.release("wt-noop")  # must not raise


# ---------------------------------------------------------------------------
# Public API export check
# ---------------------------------------------------------------------------

def test_exports_from_package():
    """PortAllocator and PortAllocationError are importable from the package."""
    from lib_python_worktree import PortAllocationError, PortAllocator  # noqa: F401


# ---------------------------------------------------------------------------
# Regression: create() rollback on allocation failure (blocking finding #1)
# ---------------------------------------------------------------------------

def test_create_rolls_back_worktree_on_allocation_failure(tmp_path: Path):
    """Regression: if allocate() raises mid-create, the git worktree is removed.

    Asserts that when PortAllocationError is raised during create():
    - git worktree remove --force is called (no dangling checkout).
    - No state record is added for the failed worktree.
    - The PortAllocationError propagates to the caller.
    """
    from unittest.mock import MagicMock, call, patch

    from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
    from lib_python_worktree.core.state import InMemoryStateStore

    store = InMemoryStateStore()
    cfg = ManagerConfig(store_root=tmp_path / "store")
    mgr = WorktreeManager(config=cfg, state=store, reconcile_on_init=False)

    # Replace the no-op allocator with one that always raises.
    mock_allocator = MagicMock()
    mock_allocator.allocate.side_effect = PortAllocationError("No free port")
    mgr._allocator = mock_allocator

    git_calls: list = []

    def _fake_git(args, cwd=None, **kwargs):
        git_calls.append(args)
        return MagicMock(returncode=0, stdout="/fake/repo\n", stderr="")

    from lib_python_worktree.contract.schema import PortSlot, WorktreeContract

    fake_repo = tmp_path / "fake_repo"
    (cfg.store_root / "fake-repo").mkdir(parents=True, exist_ok=True)

    with (
        patch("lib_python_worktree.core.manager._run_git", side_effect=_fake_git),
        patch(
            "lib_python_worktree.core.manager._load_contract",
        ) as mock_load,
        patch.object(mgr, "_validate_repo", return_value=fake_repo),
        patch.object(mgr, "_branch_exists", return_value=True),
        patch("lib_python_worktree.core.manager.Path.mkdir"),
    ):
        mock_contract = MagicMock(spec=WorktreeContract)
        mock_contract.ports = [PortSlot(name="web")]
        mock_load.return_value = mock_contract

        with pytest.raises(PortAllocationError):
            mgr.create(str(fake_repo), "feature/alloc-fail")

    # allocate() was called (and raised).
    mock_allocator.allocate.assert_called_once()

    # A rollback git call with "worktree", "remove", "--force" must have been
    # issued after the allocation failure.
    rollback_calls = [c for c in git_calls if "remove" in c and "--force" in c]
    assert rollback_calls, (
        "Expected 'git worktree remove --force <path>' rollback call; "
        f"got git_calls={git_calls}"
    )

    # No state record should have been persisted.
    assert store.list() == [], "State record was not rolled back"


# ---------------------------------------------------------------------------
# Regression: _teardown() keeps ports when git remove fails (blocking #2)
# ---------------------------------------------------------------------------

def test_teardown_keeps_ports_when_git_remove_fails(tmp_path: Path):
    """Regression: if git worktree remove fails, ports must NOT be freed.

    Asserts that when _run_git raises GitCommandError inside _teardown():
    - allocator.release() is NOT called.
    - The state record is left untouched (remove() re-raises before state ops).
    - GitCommandError propagates to the caller.
    """
    from unittest.mock import MagicMock, patch

    from lib_python_worktree.core.manager import (
        GitCommandError,
        ManagerConfig,
        WorktreeManager,
    )
    from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord

    store = InMemoryStateStore()
    cfg = ManagerConfig(store_root=tmp_path / "store")
    mgr = WorktreeManager(config=cfg, state=store, reconcile_on_init=False)

    mock_allocator = MagicMock()
    mock_allocator.release.return_value = None
    mgr._allocator = mock_allocator

    # Pre-populate a record with allocated ports.
    record = WorktreeRecord(
        id="wt-keep-ports-abc",
        repo_root=str(tmp_path),
        branch="feature/keep-ports",
        path=str(tmp_path / "wt"),
        ports={"web": 31500},
    )
    store.add(record)

    # Make git worktree remove fail.
    def _fail_git(args, cwd=None, **kwargs):
        if "remove" in args:
            return MagicMock(returncode=1, stdout="", stderr="error: cannot remove")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("lib_python_worktree.core.manager._run_git", side_effect=_fail_git):
        with pytest.raises(GitCommandError):
            mgr._teardown(record, force=False)

    # Ports must NOT have been released — git remove failed, worktree still alive.
    mock_allocator.release.assert_not_called()

    # State record must still be present (teardown raised before state.remove).
    assert store.get("wt-keep-ports-abc") is not None, (
        "State record was removed despite git worktree remove failing"
    )


# ---------------------------------------------------------------------------
# Regression: allocate() succeeds then state.add() fails → release() called
# (blocking fix #1 — port leak when state.add raises after allocate)
# ---------------------------------------------------------------------------

def test_create_releases_ports_when_state_add_fails(tmp_path: Path):
    """Regression: if allocate() succeeds but state.add() then raises,
    allocator.release() must be called in the rollback block so the ports
    are freed and not permanently leaked.

    Asserts:
    - allocator.allocate() is called and succeeds.
    - state.add() raises, triggering rollback.
    - allocator.release() IS called with the same worktree_id.
    - git worktree remove --force IS issued (dangling checkout cleaned up).
    - No state record remains (add() failed before completion).
    - The original exception from state.add() propagates to the caller.
    """
    from unittest.mock import MagicMock, patch

    from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
    from lib_python_worktree.core.state import InMemoryStateStore

    store = InMemoryStateStore()
    cfg = ManagerConfig(store_root=tmp_path / "store")
    mgr = WorktreeManager(config=cfg, state=store, reconcile_on_init=False)

    # Allocator that succeeds — returns a port mapping.
    mock_allocator = MagicMock()
    mock_allocator.allocate.return_value = {"web": 31000}
    mock_allocator.release.return_value = None
    mgr._allocator = mock_allocator

    # Make state.add() blow up after allocate() has already committed ports.
    state_add_error = RuntimeError("disk full")
    original_add = store.add

    def _boom_add(record):
        raise state_add_error

    store.add = _boom_add  # type: ignore[method-assign]

    git_calls: list = []

    def _fake_git(args, cwd=None, **kwargs):
        git_calls.append(list(args))
        return MagicMock(returncode=0, stdout="/fake/repo\n", stderr="")

    from lib_python_worktree.contract.schema import PortSlot, WorktreeContract

    fake_repo = tmp_path / "fake_repo"
    (cfg.store_root / "fake-repo").mkdir(parents=True, exist_ok=True)

    with (
        patch("lib_python_worktree.core.manager._run_git", side_effect=_fake_git),
        patch("lib_python_worktree.core.manager._load_contract") as mock_load,
        patch.object(mgr, "_validate_repo", return_value=fake_repo),
        patch.object(mgr, "_branch_exists", return_value=True),
        patch("lib_python_worktree.core.manager.Path.mkdir"),
    ):
        mock_contract = MagicMock(spec=WorktreeContract)
        mock_contract.ports = [PortSlot(name="web")]
        mock_load.return_value = mock_contract

        with pytest.raises(RuntimeError, match="disk full"):
            mgr.create(str(fake_repo), "feature/state-fail")

    # allocate() succeeded.
    mock_allocator.allocate.assert_called_once()

    # release() MUST have been called during rollback.
    mock_allocator.release.assert_called_once()
    release_call_id = mock_allocator.release.call_args[0][0]
    # The same worktree_id that was passed to allocate() must go to release().
    allocate_call_id = mock_allocator.allocate.call_args[0][1]
    assert release_call_id == allocate_call_id, (
        f"release called with id={release_call_id!r} but allocate used id={allocate_call_id!r}"
    )

    # git worktree remove --force must have been issued in rollback.
    rollback_calls = [c for c in git_calls if "remove" in c and "--force" in c]
    assert rollback_calls, (
        f"Expected 'git worktree remove --force' rollback; got git_calls={git_calls}"
    )


# ---------------------------------------------------------------------------
# Regression: _load_contract() raising ContractError mid-create → worktree
# rollback triggered (blocking fix #2)
# ---------------------------------------------------------------------------

def test_create_rolls_back_worktree_on_contract_error(tmp_path: Path):
    """Regression: ContractError from _load_contract() after git worktree add
    must trigger the same rollback as any other mid-create failure.

    Before fix #2 the _load_contract() call sat OUTSIDE the try/except, so a
    ContractError left the freshly-created git worktree dangling with no cleanup.

    Asserts:
    - _load_contract() raises ContractError (simulates malformed YAML).
    - git worktree remove --force IS issued (the dangling worktree is cleaned up).
    - allocator.release() is called (consistent best-effort rollback).
    - No state record was persisted.
    - ContractError propagates to the caller.
    """
    from unittest.mock import MagicMock, patch

    from lib_python_worktree.contract.loader import ContractError
    from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
    from lib_python_worktree.core.state import InMemoryStateStore

    store = InMemoryStateStore()
    cfg = ManagerConfig(store_root=tmp_path / "store")
    mgr = WorktreeManager(config=cfg, state=store, reconcile_on_init=False)

    mock_allocator = MagicMock()
    mock_allocator.allocate.return_value = {}
    mock_allocator.release.return_value = None
    mgr._allocator = mock_allocator

    git_calls: list = []

    def _fake_git(args, cwd=None, **kwargs):
        git_calls.append(list(args))
        return MagicMock(returncode=0, stdout="/fake/repo\n", stderr="")

    fake_repo = tmp_path / "fake_repo"
    (cfg.store_root / "fake-repo").mkdir(parents=True, exist_ok=True)

    with (
        patch("lib_python_worktree.core.manager._run_git", side_effect=_fake_git),
        patch(
            "lib_python_worktree.core.manager._load_contract",
            side_effect=ContractError("YAML parse error: bad syntax"),
        ),
        patch.object(mgr, "_validate_repo", return_value=fake_repo),
        patch.object(mgr, "_branch_exists", return_value=True),
        patch("lib_python_worktree.core.manager.Path.mkdir"),
    ):
        with pytest.raises(ContractError, match="YAML parse error"):
            mgr.create(str(fake_repo), "feature/bad-contract")

    # git worktree remove --force must have been issued to clean up the dangling
    # worktree that was created before _load_contract() raised.
    rollback_calls = [c for c in git_calls if "remove" in c and "--force" in c]
    assert rollback_calls, (
        f"Expected 'git worktree remove --force' rollback after ContractError; "
        f"got git_calls={git_calls}"
    )

    # allocator.release() must have been called in rollback (best-effort).
    mock_allocator.release.assert_called_once()

    # No state record should have been persisted.
    assert store.list() == [], (
        "State record leaked after ContractError during create()"
    )
