"""Full test suite for the persistent state store (W7).

Covers:
- Regression: records survive a store reload (the primary ticket bug).
- CRUD / Protocol parity with InMemoryStateStore.
- Schema version fields in state.yaml and ports.yaml.
- pids field on WorktreeRecord.
- reconcile(): orphaned paths, dead PIDs, live PIDs (unchanged), freed ports,
  logging of inconsistencies.
- Concurrent-access lock safety (single-instance and two-instance).
- Edge cases: empty dir, missing file, atomic write (no corruption on error).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest
import yaml

from lib_python_worktree.core.state import WorktreeRecord
from lib_python_worktree.core.yaml_store import (
    ReconcileReport,
    YamlStateStore,
    _pid_alive,
    reconcile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary directory for a YamlStateStore."""
    return tmp_path / "state"


@pytest.fixture
def yaml_store(state_dir: Path) -> YamlStateStore:
    """Return a YamlStateStore backed by a fresh temp directory."""
    return YamlStateStore(state_dir=state_dir)


def _make_record(
    id: str = "rec-001",
    repo_root: str = "/repos/myrepo",
    branch: str = "main",
    path: str = "/store/myrepo/rec-001",
    status: str = "created",
    ports: dict | None = None,
    pids: dict | None = None,
    branch_created_by_us: bool = False,
) -> WorktreeRecord:
    return WorktreeRecord(
        id=id,
        repo_root=repo_root,
        branch=branch,
        path=path,
        status=status,
        ports=ports or {},
        pids=pids or {},
        branch_created_by_us=branch_created_by_us,
    )


# ---------------------------------------------------------------------------
# Regression: records survive a store reload
# ---------------------------------------------------------------------------

def test_records_survive_store_reload(state_dir: Path):
    """Primary regression: a record added to one YamlStateStore instance must
    be visible from a fresh instance pointing at the same directory."""
    store1 = YamlStateStore(state_dir=state_dir)
    rec = _make_record(id="wt-001", branch="feature/x")
    store1.add(rec)

    # Create a brand-new instance — this simulates an MCP restart.
    store2 = YamlStateStore(state_dir=state_dir)
    retrieved = store2.get("wt-001")
    assert retrieved is not None
    assert retrieved.id == "wt-001"
    assert retrieved.branch == "feature/x"


# ---------------------------------------------------------------------------
# CRUD / Protocol tests
# ---------------------------------------------------------------------------

def test_add_get_roundtrip_yaml(yaml_store: YamlStateStore):
    rec = _make_record()
    yaml_store.add(rec)
    retrieved = yaml_store.get("rec-001")
    assert retrieved is not None
    assert retrieved.id == rec.id
    assert retrieved.branch == rec.branch
    assert retrieved.path == rec.path


def test_get_missing_returns_none_yaml(yaml_store: YamlStateStore):
    assert yaml_store.get("does-not-exist") is None


def test_remove_existing_yaml(yaml_store: YamlStateStore):
    rec = _make_record()
    yaml_store.add(rec)
    removed = yaml_store.remove("rec-001")
    assert removed is not None
    assert removed.id == "rec-001"
    assert yaml_store.get("rec-001") is None


def test_remove_missing_returns_none_yaml(yaml_store: YamlStateStore):
    assert yaml_store.remove("no-such-id") is None


def test_list_yaml(yaml_store: YamlStateStore):
    yaml_store.add(_make_record(id="r1", path="/store/myrepo/r1"))
    yaml_store.add(_make_record(id="r2", branch="feature/x", path="/store/myrepo/r2"))
    listed = yaml_store.list()
    assert len(listed) == 2
    ids = {r.id for r in listed}
    assert ids == {"r1", "r2"}


def test_find_by_branch_yaml(yaml_store: YamlStateStore):
    rec = _make_record(repo_root="/repos/myrepo", branch="feature/beta")
    yaml_store.add(rec)
    found = yaml_store.find_by_branch("/repos/myrepo", "feature/beta")
    assert found is not None
    assert found.id == rec.id
    assert yaml_store.find_by_branch("/repos/myrepo", "other") is None


def test_add_duplicate_raises_yaml(yaml_store: YamlStateStore):
    yaml_store.add(_make_record(id="dup"))
    with pytest.raises(ValueError, match="dup"):
        yaml_store.add(_make_record(id="dup"))


def test_update_yaml(yaml_store: YamlStateStore):
    rec = _make_record()
    yaml_store.add(rec)
    updated = _make_record(status="stopped")
    yaml_store.update(updated)
    retrieved = yaml_store.get("rec-001")
    assert retrieved is not None
    assert retrieved.status == "stopped"


def test_update_missing_raises_yaml(yaml_store: YamlStateStore):
    rec = _make_record(id="ghost")
    with pytest.raises(KeyError):
        yaml_store.update(rec)


# ---------------------------------------------------------------------------
# Schema version fields
# ---------------------------------------------------------------------------

def test_state_yaml_has_version_field(yaml_store: YamlStateStore, state_dir: Path):
    yaml_store.add(_make_record())
    state_path = state_dir / "state.yaml"
    assert state_path.exists()
    with open(state_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert data.get("version") == 1


def test_ports_yaml_has_version_field(yaml_store: YamlStateStore, state_dir: Path):
    yaml_store._ports.ensure_file()
    ports_path = state_dir / "ports.yaml"
    assert ports_path.exists()
    with open(ports_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert data.get("version") == 1


# ---------------------------------------------------------------------------
# pids field
# ---------------------------------------------------------------------------

def test_pids_field_default_empty():
    rec = _make_record()
    assert rec.pids == {}


def test_worktree_record_with_pids_roundtrip(yaml_store: YamlStateStore):
    rec = _make_record(pids={"server": 1234, "worker": 5678})
    yaml_store.add(rec)
    retrieved = yaml_store.get("rec-001")
    assert retrieved is not None
    assert retrieved.pids == {"server": 1234, "worker": 5678}


# ---------------------------------------------------------------------------
# reconcile(): orphaned path
# ---------------------------------------------------------------------------

def test_reconcile_orphaned_path(state_dir: Path, tmp_path: Path):
    """A worktree whose path does not exist should be marked 'orphaned'."""
    store = YamlStateStore(state_dir=state_dir)
    non_existent = str(tmp_path / "gone" / "wt-001")
    rec = _make_record(id="wt-001", path=non_existent)
    store.add(rec)

    report = reconcile(store)

    assert "wt-001" in report.orphaned
    updated = store.get("wt-001")
    assert updated is not None
    assert updated.status == "orphaned"


# ---------------------------------------------------------------------------
# reconcile(): dead PID
# ---------------------------------------------------------------------------

def test_reconcile_dead_pid(state_dir: Path, tmp_path: Path):
    """A record with a PID that is not alive should have that PID removed and
    status set to 'stopped'."""
    store = YamlStateStore(state_dir=state_dir)
    wt_path = tmp_path / "wt-002"
    wt_path.mkdir()
    dead_pid = 99999999  # extremely unlikely to be alive

    rec = _make_record(id="wt-002", path=str(wt_path), pids={"server": dead_pid})
    store.add(rec)

    assert not _pid_alive(dead_pid), "test assumption: pid 99999999 must not be alive"

    report = reconcile(store)

    assert "wt-002" in report.stopped
    updated = store.get("wt-002")
    assert updated is not None
    assert updated.status == "stopped"
    assert "server" not in updated.pids


# ---------------------------------------------------------------------------
# reconcile(): live PID unchanged
# ---------------------------------------------------------------------------

def test_reconcile_live_pid_unchanged(state_dir: Path, tmp_path: Path):
    """A record with a live PID and an existing path should not be modified."""
    store = YamlStateStore(state_dir=state_dir)
    wt_path = tmp_path / "wt-live"
    wt_path.mkdir()
    live_pid = os.getpid()

    rec = _make_record(id="wt-live", path=str(wt_path), pids={"self": live_pid})
    store.add(rec)

    report = reconcile(store)

    assert "wt-live" not in report.orphaned
    assert "wt-live" not in report.stopped
    updated = store.get("wt-live")
    assert updated is not None
    assert updated.pids == {"self": live_pid}
    assert updated.status == "created"


# ---------------------------------------------------------------------------
# reconcile(): freed port
# ---------------------------------------------------------------------------

def test_reconcile_freed_port(state_dir: Path, tmp_path: Path):
    """A port allocation that is not in use should be freed from ports.yaml."""
    store = YamlStateStore(state_dir=state_dir)
    # Use a high port number extremely unlikely to be in use.
    unused_port = 19999
    # Write a port allocation directly.
    store._ports._save({"myservice": unused_port})

    # Add a dummy worktree with an existing path (so it doesn't become orphaned)
    # and no pids (so no surviving PID is associated with the port).
    wt_path = tmp_path / "wt-port"
    wt_path.mkdir()
    store.add(_make_record(id="wt-port", path=str(wt_path)))

    # Verify the port is not actually in use (best-effort).
    import socket as _socket
    try:
        with _socket.create_connection(("127.0.0.1", unused_port), timeout=0.1):
            pytest.skip("Port 19999 is unexpectedly in use on this machine")
    except OSError:
        pass

    report = reconcile(store)

    assert "myservice" in report.freed_ports
    remaining = store._ports.get_all()
    assert "myservice" not in remaining


# ---------------------------------------------------------------------------
# reconcile(): port retained when a surviving PID exists
# ---------------------------------------------------------------------------

def test_reconcile_port_retained_when_surviving_pid(state_dir: Path, tmp_path: Path):
    """A non-listening port must NOT be freed when a live PID is still tracked.

    The worktree process may not have bound the port yet (race between startup
    and reconcile). As long as its PID is alive, the allocation is kept.
    """
    store = YamlStateStore(state_dir=state_dir)
    # Use a high port number extremely unlikely to be in use.
    unused_port = 19998
    store._ports._save({"myservice": unused_port})

    # Add a worktree with an existing path and a live PID (this process).
    wt_path = tmp_path / "wt-live-port"
    wt_path.mkdir()
    live_pid = os.getpid()
    store.add(_make_record(id="wt-live-port", path=str(wt_path), pids={"server": live_pid}))

    # Verify the port is not actually in use (best-effort).
    import socket as _socket
    try:
        with _socket.create_connection(("127.0.0.1", unused_port), timeout=0.1):
            pytest.skip("Port 19998 is unexpectedly in use on this machine")
    except OSError:
        pass

    report = reconcile(store)

    # Port must NOT have been freed because a live PID is still tracked.
    assert "myservice" not in report.freed_ports
    remaining = store._ports.get_all()
    assert "myservice" in remaining
    assert remaining["myservice"] == unused_port


# ---------------------------------------------------------------------------
# reconcile(): logging of inconsistencies
# ---------------------------------------------------------------------------

def test_reconcile_logs_inconsistency(state_dir: Path, tmp_path: Path, caplog):
    """Reconcile must log at WARNING level for each inconsistency."""
    store = YamlStateStore(state_dir=state_dir)
    non_existent = str(tmp_path / "gone" / "wt-log")
    rec = _make_record(id="wt-log", path=non_existent)
    store.add(rec)

    with caplog.at_level(logging.WARNING, logger="lib_python_worktree.core.yaml_store"):
        reconcile(store)

    assert any("orphaned" in record.message or "wt-log" in record.message
               for record in caplog.records)


# ---------------------------------------------------------------------------
# Concurrency: single instance, multiple threads
# ---------------------------------------------------------------------------

def test_concurrent_add_same_instance_no_data_loss(state_dir: Path):
    """Multiple threads adding records to the same store instance must not lose
    any record (all records survive)."""
    store = YamlStateStore(state_dir=state_dir)
    n = 20
    errors: list[Exception] = []

    def _add(i: int) -> None:
        try:
            store.add(_make_record(
                id=f"wt-{i:03d}",
                path=f"/store/repo/wt-{i:03d}",
                branch=f"branch-{i}",
            ))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_add, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent add: {errors}"
    listed = store.list()
    assert len(listed) == n, f"Expected {n} records, got {len(listed)}"


# ---------------------------------------------------------------------------
# Concurrency: two independent instances
# ---------------------------------------------------------------------------

def test_concurrent_add_two_instances_no_data_loss(state_dir: Path):
    """Two independent YamlStateStore instances writing concurrently must not
    lose each other's records."""
    store_a = YamlStateStore(state_dir=state_dir)
    store_b = YamlStateStore(state_dir=state_dir)
    n = 10
    errors: list[Exception] = []

    def _add_via(store: YamlStateStore, prefix: str, count: int) -> None:
        for i in range(count):
            try:
                store.add(_make_record(
                    id=f"{prefix}-{i:03d}",
                    path=f"/store/repo/{prefix}-{i:03d}",
                    branch=f"{prefix}-branch-{i}",
                ))
            except Exception as exc:
                errors.append(exc)

    ta = threading.Thread(target=_add_via, args=(store_a, "a", n))
    tb = threading.Thread(target=_add_via, args=(store_b, "b", n))
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    assert not errors, f"Errors during concurrent add: {errors}"
    # Read back via a third instance to see the final state.
    store_c = YamlStateStore(state_dir=state_dir)
    listed = store_c.list()
    ids = {r.id for r in listed}
    for i in range(n):
        assert f"a-{i:03d}" in ids, f"Missing record a-{i:03d}"
        assert f"b-{i:03d}" in ids, f"Missing record b-{i:03d}"


# ---------------------------------------------------------------------------
# Concurrency: lock blocks concurrent write
# ---------------------------------------------------------------------------

def test_lock_blocks_concurrent_write(state_dir: Path):
    """Acquiring the lock in one thread should prevent a concurrent write from
    the second thread from interleaving inside the critical section."""
    import portalocker

    store = YamlStateStore(state_dir=state_dir)
    state_path = state_dir / "state.yaml"
    # Make sure state.yaml exists (lock file is based on its path)
    store._save_state({})

    # Use LOCK_EX|LOCK_NB so portalocker polls with a timeout (same flags used
    # by YamlStateStore internally).
    _flags = portalocker.LOCK_EX | portalocker.LOCK_NB

    lock_file = str(state_path) + ".lock"
    inside_critical = threading.Event()
    results: list[str] = []

    def _hold_lock() -> None:
        with portalocker.Lock(lock_file, timeout=10, flags=_flags):
            inside_critical.set()
            # Give the second thread time to try (and poll) acquiring
            time.sleep(0.15)
            results.append("first_released")

    def _try_lock() -> None:
        inside_critical.wait()  # wait until first thread holds the lock
        # This should block (polling) until the first thread releases.
        with portalocker.Lock(lock_file, timeout=5, flags=_flags):
            results.append("second_acquired")

    t1 = threading.Thread(target=_hold_lock)
    t2 = threading.Thread(target=_try_lock)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # The second thread must have acquired the lock AFTER the first released it.
    assert results == ["first_released", "second_acquired"], (
        f"Unexpected ordering: {results}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_state_dir_list_returns_empty(state_dir: Path):
    """A freshly created store with no state.yaml should return an empty list."""
    store = YamlStateStore(state_dir=state_dir)
    assert store.list() == []


def test_missing_state_yaml_after_construction(state_dir: Path):
    """state.yaml must NOT be created merely by constructing a YamlStateStore."""
    store = YamlStateStore(state_dir=state_dir)
    state_path = state_dir / "state.yaml"
    # The directory is created, but the file should be absent until first write.
    assert state_dir.exists()
    assert not state_path.exists()


def test_state_yaml_not_corrupted_on_exception_during_write(
    state_dir: Path,
):
    """If os.replace raises (simulating a crash mid-write), the existing
    state.yaml must remain intact and uncorrupted.

    The atomic write pattern (temp file + os.replace) means the original file
    is only replaced once the new content is safely in the temp file. If
    os.replace itself fails, the original file is untouched.
    """
    store = YamlStateStore(state_dir=state_dir)
    rec = _make_record(id="wt-safe", path="/store/myrepo/wt-safe")
    store.add(rec)

    state_path = state_dir / "state.yaml"
    original_text = state_path.read_text(encoding="utf-8")

    # Simulate os.replace failing (e.g., cross-device link / disk full).
    original_replace = os.replace

    call_count = [0]

    def _failing_replace(src: str, dst: str) -> None:
        call_count[0] += 1
        raise OSError("simulated disk full during replace")

    with patch("lib_python_worktree.core.yaml_store.os.replace", side_effect=_failing_replace):
        try:
            store.add(_make_record(id="wt-new", path="/store/myrepo/wt-new"))
        except OSError:
            pass

    assert call_count[0] >= 1, "os.replace was never called"

    # The file should still contain the original valid YAML.
    current_text = state_path.read_text(encoding="utf-8")
    assert current_text == original_text, (
        "state.yaml was corrupted by a failed write"
    )
    data = yaml.safe_load(current_text)
    assert data is not None
    assert "worktrees" in data
