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
import sys
import threading
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
import yaml

import subprocess

from lib_python_worktree.core.state import (
    SETUP_STATUS_COMPLETED,
    SETUP_STATUS_FAILED,
    STOP_REASON_SURVIVORS,
    SetupOutcome,
    StopDetail,
    WorktreeRecord,
    _STOP_DETAIL_MAX_PIDS,
)
from lib_python_worktree.core.process_lifecycle import start as _lifecycle_start
from lib_python_worktree.core.process_lifecycle import stop as _lifecycle_stop
from lib_python_worktree.core.process_lifecycle import _force_kill
from lib_python_worktree.core.yaml_store import (
    AdoptReport,
    ReconcileReport,
    YamlStateStore,
    _pid_alive,
    _pid_alive_windows,
    adopt,
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


def test_find_by_branch_normalizes_backslash_paths_on_load(state_dir: Path):
    """Regression (#23): records written with Windows backslash repo_root/path
    must be found by a forward-slash key after a YAML round-trip.

    Before the _record_from_dict fix, a state.yaml produced by a pre-fix build
    on Windows stored backslash strings.  After re-loading, find_by_branch was
    called with a forward-slash key (from repo_path.as_posix()) that never
    matched the stored backslash string, silently suppressing DuplicateWorktreeError
    and making adopt()'s idempotency checks fail.

    This test must FAIL without the Path(...).as_posix() normalisation in
    _record_from_dict and PASS with it.
    """
    store = YamlStateStore(state_dir=state_dir)

    # Simulate a record that was persisted by a pre-fix Windows build.
    # We write it with raw backslash strings directly, bypassing add() so that
    # the normalization in _record_from_dict is what we're testing on the
    # *read* path, not just what add() received.
    backslash_record = _make_record(
        id="wt-backslash",
        repo_root=r"C:\repos\myrepo",
        branch="feature/x",
        path=r"C:\store\wt-001",
    )
    store.add(backslash_record)

    # Reload from a fresh instance so _record_from_dict runs on the persisted data.
    store2 = YamlStateStore(state_dir=state_dir)

    # Forward-slash key — this is what create() and adopt() pass after the fix.
    found = store2.find_by_branch("C:/repos/myrepo", "feature/x")
    assert found is not None, (
        "find_by_branch with a forward-slash key must find a record that was "
        "stored with backslash paths — _record_from_dict must normalize on load"
    )
    assert found.id == "wt-backslash"
    # The loaded record's fields must also be forward-slash.
    assert found.repo_root == "C:/repos/myrepo"
    assert found.path == "C:/store/wt-001"


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


def test_worktree_record_with_returncode_roundtrip(
    yaml_store: YamlStateStore, tmp_path: Path
):
    """Ticket #81 (reviewer finding): a real YamlStateStore add/get cycle must
    round-trip a non-default ``returncode`` value.

    Unlike ``InMemoryStateStore`` (whose ``.update()``/``.add()`` just
    re-store the same object reference and therefore give zero protection
    against a serialization bug), this goes through the real
    ``_record_to_dict``/``_record_from_dict`` YAML (de)serialization path.
    """
    rec = _make_record(id="rec-returncode")
    rec.returncode = 3
    yaml_store.add(rec)

    retrieved = yaml_store.get("rec-returncode")
    assert retrieved is not None
    assert retrieved.returncode == 3


def test_worktree_record_start_log_paths_roundtrip(
    yaml_store: YamlStateStore, state_dir: Path, tmp_path: Path
):
    """Ticket #119: a real YamlStateStore add/get cycle must round-trip a
    two-role ``start_log_paths`` map byte-for-byte, and the serialised
    state.yaml must carry the new ``start_log_paths`` key -- never the
    removed scalar ``start_log_path`` key.
    """
    main_log = str(tmp_path / "start-main.log")
    ui_log = str(tmp_path / "start-ui.log")
    rec = _make_record(id="rec-start-logs")
    rec.start_log_paths = {"main": main_log, "ui": ui_log}
    yaml_store.add(rec)

    retrieved = yaml_store.get("rec-start-logs")
    assert retrieved is not None
    assert retrieved.start_log_paths == {"main": main_log, "ui": ui_log}

    state_path = state_dir / "state.yaml"
    with open(state_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    on_disk = data["worktrees"]["rec-start-logs"]
    assert on_disk["start_log_paths"] == {"main": main_log, "ui": ui_log}
    assert "start_log_path" not in on_disk


# ---------------------------------------------------------------------------
# start_log_paths: legacy scalar load + corrupt-value tolerance (ticket #119)
# ---------------------------------------------------------------------------

def _write_raw_state(state_dir: Path, worktree_dict: dict) -> None:
    """Hand-write a minimal state.yaml with one record, bypassing
    _record_to_dict entirely, so these tests exercise _record_from_dict's
    tolerance of a legacy/corrupt on-disk shape rather than round-tripping
    through the engine's own (well-behaved) writer."""
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.yaml"
    with open(state_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {"version": 1, "worktrees": {"rec-legacy": worktree_dict}}, fh
        )


def _base_worktree_dict(**overrides) -> dict:
    base = {
        "id": "rec-legacy",
        "repo_root": "/repos/myrepo",
        "branch": "main",
        "path": "/store/myrepo/rec-legacy",
        "status": "created",
    }
    base.update(overrides)
    return base


def test_start_log_paths_legacy_scalar_only_loads_to_empty_dict(state_dir: Path):
    """A hand-written state.yaml carrying only the removed scalar
    ``start_log_path`` key (no ``start_log_paths`` key at all) must load
    cleanly with an empty map -- the legacy scalar is deliberately not
    migrated."""
    _write_raw_state(
        state_dir,
        _base_worktree_dict(start_log_path="/logs/start-main.log"),
    )
    store = YamlStateStore(state_dir=state_dir)
    rec = store.get("rec-legacy")
    assert rec is not None
    assert rec.start_log_paths == {}


def test_start_log_paths_missing_key_loads_to_empty_dict(state_dir: Path):
    """Neither ``start_log_paths`` nor the legacy scalar present -> {}."""
    _write_raw_state(state_dir, _base_worktree_dict())
    store = YamlStateStore(state_dir=state_dir)
    rec = store.get("rec-legacy")
    assert rec is not None
    assert rec.start_log_paths == {}


def test_start_log_paths_null_loads_to_empty_dict(state_dir: Path):
    _write_raw_state(state_dir, _base_worktree_dict(start_log_paths=None))
    store = YamlStateStore(state_dir=state_dir)
    rec = store.get("rec-legacy")
    assert rec is not None
    assert rec.start_log_paths == {}


def test_start_log_paths_non_dict_string_loads_to_empty_dict(state_dir: Path):
    """A corrupt (hand-edited) value that is a bare string, not a mapping,
    must not raise -- it degrades to {}."""
    _write_raw_state(
        state_dir, _base_worktree_dict(start_log_paths="a string")
    )
    store = YamlStateStore(state_dir=state_dir)
    rec = store.get("rec-legacy")
    assert rec is not None
    assert rec.start_log_paths == {}


def test_start_log_paths_drops_non_string_entries_keeps_valid(state_dir: Path):
    """A dict containing a non-string value drops just that entry while
    keeping the other, valid entries."""
    _write_raw_state(
        state_dir,
        _base_worktree_dict(
            start_log_paths={"main": "/logs/start-main.log", "ui": 42}
        ),
    )
    store = YamlStateStore(state_dir=state_dir)
    rec = store.get("rec-legacy")
    assert rec is not None
    assert rec.start_log_paths == {"main": "/logs/start-main.log"}


def test_start_log_paths_drops_non_string_key_keeps_valid(state_dir: Path):
    """A dict containing a non-string key (e.g. a bare YAML integer) drops
    just that entry while keeping the other, valid entries."""
    _write_raw_state(
        state_dir,
        _base_worktree_dict(
            start_log_paths={"main": "/logs/start-main.log", 123: "/logs/x.log"}
        ),
    )
    store = YamlStateStore(state_dir=state_dir)
    rec = store.get("rec-legacy")
    assert rec is not None
    assert rec.start_log_paths == {"main": "/logs/start-main.log"}


# ---------------------------------------------------------------------------
# Ticket #100: shadowed_contract is transient -- never persisted to
# state.yaml, mirroring the killed_pids precedent (test_process_lifecycle.py
# ::test_stop_populates_killed_pids_with_tree_and_orphans).
# ---------------------------------------------------------------------------


def test_shadowed_contract_not_serialised_to_dict():
    """`_record_to_dict` must never write a `shadowed_contract` key."""
    from lib_python_worktree.core.state import ShadowedContract
    from lib_python_worktree.core.yaml_store import _record_to_dict

    rec = _make_record(id="rec-shadow")
    rec.shadowed_contract = ShadowedContract(
        path="/checkout/.seretos/worktree-setup.yml",
        used_path="/repo/.seretos/worktree-setup.yml",
        reason="differs",
        message="start(): checkout-local contract differs",
    )

    assert "shadowed_contract" not in _record_to_dict(rec)


def test_record_from_dict_legacy_dict_has_no_shadowed_contract_key():
    """A legacy dict with no `shadowed_contract` key at all (any state.yaml
    written before this field existed) must deserialise with `None`, not
    raise."""
    from lib_python_worktree.core.yaml_store import _record_from_dict

    legacy = {
        "id": "rec-legacy",
        "repo_root": "/repos/myrepo",
        "branch": "main",
        "path": "/store/myrepo/rec-legacy",
    }

    rec = _record_from_dict(legacy)

    assert rec.shadowed_contract is None


def test_shadowed_contract_dropped_on_yaml_roundtrip(yaml_store: YamlStateStore):
    """A real YamlStateStore add/get cycle must drop `shadowed_contract`
    rather than persisting a stale copy -- it is a live observation
    recomputed on every start(), never a stored verdict."""
    from lib_python_worktree.core.state import ShadowedContract

    rec = _make_record(id="rec-shadow-roundtrip")
    rec.shadowed_contract = ShadowedContract(
        path="/checkout/.seretos/worktree-setup.yml",
        used_path="/repo/.seretos/worktree-setup.yml",
        reason="unreadable",
        message="start(): checkout-local contract could not be read",
    )
    yaml_store.add(rec)

    retrieved = yaml_store.get("rec-shadow-roundtrip")

    assert retrieved is not None
    assert retrieved.shadowed_contract is None


# ---------------------------------------------------------------------------
# Ticket #110: stop_attempt is transient -- never persisted to state.yaml,
# mirroring the killed_pids/shadowed_contract precedent above.
# ---------------------------------------------------------------------------


def test_stop_attempt_not_serialised_to_dict():
    """`_record_to_dict` must never write a `stop_attempt` key."""
    from lib_python_worktree.core.state import STOP_ATTEMPT_KILLED, StopAttempt
    from lib_python_worktree.core.yaml_store import _record_to_dict

    rec = _make_record(id="rec-stop-attempt")
    rec.stop_attempt = StopAttempt(
        outcome=STOP_ATTEMPT_KILLED,
        message="stop(): killed pid 123",
        tracked_pid=123,
        tracked_pid_alive=True,
    )

    assert "stop_attempt" not in _record_to_dict(rec)


def test_record_from_dict_legacy_dict_has_no_stop_attempt_key():
    """A legacy dict with no `stop_attempt` key at all (any state.yaml
    written before this field existed) must deserialise with `None`, not
    raise."""
    from lib_python_worktree.core.yaml_store import _record_from_dict

    legacy = {
        "id": "rec-legacy-stop-attempt",
        "repo_root": "/repos/myrepo",
        "branch": "main",
        "path": "/store/myrepo/rec-legacy-stop-attempt",
    }

    rec = _record_from_dict(legacy)

    assert rec.stop_attempt is None


def test_stop_attempt_dropped_on_yaml_roundtrip(yaml_store: YamlStateStore):
    """A real YamlStateStore add/get cycle must drop `stop_attempt` rather
    than persisting a stale copy -- it describes a single stop() call's
    attempt, not a stored verdict."""
    from lib_python_worktree.core.state import STOP_ATTEMPT_KILLED, StopAttempt

    rec = _make_record(id="rec-stop-attempt-roundtrip")
    rec.stop_attempt = StopAttempt(
        outcome=STOP_ATTEMPT_KILLED,
        message="stop(): killed pid 456",
        tracked_pid=456,
        tracked_pid_alive=True,
    )
    yaml_store.add(rec)

    retrieved = yaml_store.get("rec-stop-attempt-roundtrip")

    assert retrieved is not None
    assert retrieved.stop_attempt is None


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


def test_reconcile_dead_pid_pops_variant_entry(state_dir: Path, tmp_path: Path):
    """Ticket #104: a dead role discovered by reconcile() must also lose its
    ``variants`` entry, keeping ``set(variants) <= set(pids)`` intact."""
    store = YamlStateStore(state_dir=state_dir)
    wt_path = tmp_path / "wt-variant-dead"
    wt_path.mkdir()
    dead_pid = 99999999  # extremely unlikely to be alive

    rec = _make_record(id="wt-variant-dead", path=str(wt_path), pids={"server": dead_pid})
    rec.variants = {"server": "web"}
    store.add(rec)

    assert not _pid_alive(dead_pid), "test assumption: pid 99999999 must not be alive"

    reconcile(store)

    updated = store.get("wt-variant-dead")
    assert updated is not None
    assert "server" not in updated.variants


# ---------------------------------------------------------------------------
# reconcile(): "stop_incomplete" status must survive a dead-role reconcile
# (ticket #87 follow-up, finding B2)
# ---------------------------------------------------------------------------

def test_reconcile_preserves_stop_incomplete_status(state_dir: Path, tmp_path: Path):
    """A record already marked 'stop_incomplete' (process_lifecycle.stop()'s
    honest "something I tried to kill may still be alive" status) must not be
    silently flipped back to 'stopped' just because reconcile() also finds a
    *different*, unrelated dead role on the same record.

    Regression scenario: a multi-role worktree where stop(role="main") could
    not confirm a leaked grandchild had died and set status="stop_incomplete",
    while record.pids still holds a live "worker" role. When that worker
    later dies for an unrelated reason and reconcile() runs, the dead-role
    branch must not discard the stop_incomplete guarantee."""
    store = YamlStateStore(state_dir=state_dir)
    wt_path = tmp_path / "wt-stop-incomplete"
    wt_path.mkdir()
    dead_pid = 99999999  # extremely unlikely to be alive

    rec = _make_record(
        id="wt-stop-incomplete",
        path=str(wt_path),
        status="stop_incomplete",
        pids={"worker": dead_pid},
    )
    store.add(rec)

    assert not _pid_alive(dead_pid), "test assumption: pid 99999999 must not be alive"

    report = reconcile(store)

    assert "wt-stop-incomplete" in report.stopped
    updated = store.get("wt-stop-incomplete")
    assert updated is not None
    assert "worker" not in updated.pids
    assert updated.status == "stop_incomplete", (
        "reconcile() must not overwrite an existing 'stop_incomplete' status "
        "back to 'stopped' when it also clears an unrelated dead role"
    )


# ---------------------------------------------------------------------------
# reconcile(): stop_detail is cleared in lockstep with status transitions
# (ticket #99, B2 edge cases)
# ---------------------------------------------------------------------------

def test_reconcile_orphaned_clears_stop_detail(state_dir: Path, tmp_path: Path):
    """reconcile()'s orphaned-path branch must clear any stale stop_detail
    in lockstep with the status transition to "orphaned"."""
    store = YamlStateStore(state_dir=state_dir)
    non_existent = str(tmp_path / "gone" / "wt-orphan-detail")
    rec = _make_record(id="wt-orphan-detail", path=non_existent)
    rec.stop_detail = StopDetail(
        reason=STOP_REASON_SURVIVORS, message="stale", role="main",
    )
    store.add(rec)

    report = reconcile(store)

    assert "wt-orphan-detail" in report.orphaned
    updated = store.get("wt-orphan-detail")
    assert updated is not None
    assert updated.status == "orphaned"
    assert updated.stop_detail is None


def test_reconcile_dead_pid_stopped_clears_stop_detail(state_dir: Path, tmp_path: Path):
    """reconcile()'s dead-PID branch, when it actually transitions status to
    "stopped", must clear any stale stop_detail left on the record."""
    store = YamlStateStore(state_dir=state_dir)
    wt_path = tmp_path / "wt-dead-detail"
    wt_path.mkdir()
    dead_pid = 99999999  # extremely unlikely to be alive

    rec = _make_record(id="wt-dead-detail", path=str(wt_path), pids={"server": dead_pid})
    rec.stop_detail = StopDetail(
        reason=STOP_REASON_SURVIVORS, message="stale", role="server",
    )
    store.add(rec)

    assert not _pid_alive(dead_pid), "test assumption: pid 99999999 must not be alive"

    report = reconcile(store)

    assert "wt-dead-detail" in report.stopped
    updated = store.get("wt-dead-detail")
    assert updated is not None
    assert updated.status == "stopped"
    assert updated.stop_detail is None


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
# reconcile(): port retained when its owner record still exists
# ---------------------------------------------------------------------------

def test_reconcile_port_retained_when_owner_record_exists(state_dir: Path, tmp_path: Path):
    """A non-listening port must NOT be freed when its owning record still
    exists in state.yaml -- regardless of whether any PID is alive.

    The worktree process may not have bound the port yet (race between
    startup and reconcile), or the environment may simply be stopped. As long
    as the owning record is still tracked, the allocation is kept.
    """
    store = YamlStateStore(state_dir=state_dir)
    # Use a high port number extremely unlikely to be in use.
    unused_port = 19998
    store._ports._save({"wt-live-port:myservice": unused_port})

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

    # Port must NOT have been freed because its owning record still exists.
    assert "wt-live-port:myservice" not in report.freed_ports
    remaining = store._ports.get_all()
    assert "wt-live-port:myservice" in remaining
    assert remaining["wt-live-port:myservice"] == unused_port


def test_reconcile_keeps_port_of_surviving_record_with_no_pids(state_dir: Path, tmp_path: Path):
    """R1 driving test: a stopped-but-tracked record's port must survive
    reconcile() even when it has no PIDs at all (not just a dead one).

    Under the old global-``surviving_pids`` heuristic this port was wiped
    because *no* record anywhere had a live PID -- even though the owning
    record ``wt-a`` was still perfectly tracked in state.yaml.
    """
    store = YamlStateStore(state_dir=state_dir)
    wt_path = tmp_path / "wt-a"
    wt_path.mkdir()
    store.add(_make_record(id="wt-a", path=str(wt_path), ports={"web": 19997}))
    store._ports._save({"wt-a:web": 19997})

    report = reconcile(store)

    assert "wt-a:web" not in report.freed_ports
    remaining = store._ports.get_all()
    assert remaining.get("wt-a:web") == 19997


def test_reconcile_frees_port_of_vanished_record(state_dir: Path, tmp_path: Path):
    """A port whose owner record no longer exists in state.yaml is freed."""
    store = YamlStateStore(state_dir=state_dir)
    # No record for "wt-gone" is ever added to the store.
    store._ports._save({"wt-gone:web": 19996})

    report = reconcile(store)

    assert "wt-gone:web" in report.freed_ports
    remaining = store._ports.get_all()
    assert "wt-gone:web" not in remaining


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


# ---------------------------------------------------------------------------
# Ticket #10: adopt() unit tests (monkeypatch _run_git in yaml_store)
# ---------------------------------------------------------------------------

# Helpers for constructing fake porcelain output
# Real git format (example):
#   worktree /path/to/main
#   HEAD abc1234
#   branch refs/heads/main
#
#   worktree /path/to/wt
#   HEAD def5678
#   branch refs/heads/feature/x
#

def _porcelain(*blocks: list[str]) -> str:
    """Join porcelain blocks, each block being a list of lines."""
    return "\n".join("\n".join(block) for block in blocks) + "\n"


def _main_block(path: str = "/repos/myrepo") -> list[str]:
    return [f"worktree {path}", "HEAD abc1234abc1234", "branch refs/heads/main", ""]


def _wt_block(path: str, branch: str = "feature/x") -> list[str]:
    return [f"worktree {path}", "HEAD def5678def5678", f"branch refs/heads/{branch}", ""]


def _detached_block(path: str = "/store/wt-detached") -> list[str]:
    return [f"worktree {path}", "HEAD aaa1234aaa1234", "detached", ""]


def _prunable_block(path: str, branch: str = "feature/stale") -> list[str]:
    """A block that has a branch but is marked prunable (directory deleted)."""
    return [
        f"worktree {path}",
        "HEAD bbb5678bbb5678",
        f"branch refs/heads/{branch}",
        "prunable gitdir file points to non-existent location",
        "",
    ]


import lib_python_worktree.core.yaml_store as yaml_store_module


@pytest.fixture
def ys(state_dir: Path) -> YamlStateStore:
    return YamlStateStore(state_dir=state_dir)


def _fake_run_git_ok(output: str):
    """Return a _run_git patcher that yields 'output' with returncode=0."""
    def _patched(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=["git", "worktree", "list", "--porcelain"],
            returncode=0,
            stdout=output,
            stderr="",
        )
    return _patched


def test_adopt_imports_out_of_band_worktree(state_dir: Path, monkeypatch):
    """adopt() must import one extra worktree found by git as status='adopted'."""
    repo_path = Path("/repos/myrepo")
    extra_path = "/store/myrepo/wt-001"
    output = _porcelain(_main_block(str(repo_path)), _wt_block(extra_path, "feature/x"))

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, repo_path)

    assert len(report.adopted) == 1
    records = store.list()
    assert len(records) == 1
    rec = records[0]
    assert rec.status == "adopted"
    assert rec.branch_created_by_us is False
    assert rec.branch == "feature/x"
    assert rec.ports == {}
    assert rec.pids == {}


def test_adopt_idempotent_same_path(state_dir: Path, monkeypatch):
    """adopt() must skip a worktree whose path is already in the store."""
    repo_path = Path("/repos/myrepo")
    extra_path = "/store/myrepo/wt-001"
    output = _porcelain(_main_block(str(repo_path)), _wt_block(extra_path, "feature/x"))

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    # Pre-add a record with the same path — keys must be forward-slash to match
    # what adopt() now stores via Path.as_posix().
    store.add(_make_record(
        id="pre-existing",
        repo_root=repo_path.resolve().as_posix(),
        branch="feature/x",
        path=Path(extra_path).resolve().as_posix(),
    ))

    report = adopt(store, repo_path)
    assert report.adopted == []
    assert len(store.list()) == 1  # no duplicate


def test_adopt_idempotent_same_branch(state_dir: Path, monkeypatch):
    """adopt() must skip a worktree whose (repo_root, branch) pair is already tracked."""
    repo_path = Path("/repos/myrepo")
    extra_path = "/store/myrepo/wt-001"
    output = _porcelain(_main_block(str(repo_path)), _wt_block(extra_path, "feature/x"))

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    # Same branch, different path — idempotent by branch key.
    # repo_root must be forward-slash to match what adopt() now stores via as_posix().
    store.add(_make_record(
        id="pre-existing",
        repo_root=repo_path.resolve().as_posix(),
        branch="feature/x",
        path="/some/other/path",
    ))

    report = adopt(store, repo_path)
    assert report.adopted == []


def test_adopt_skips_main_worktree_block(state_dir: Path, monkeypatch):
    """adopt() must skip the first block (the main worktree)."""
    repo_path = Path("/repos/myrepo")
    output = _porcelain(_main_block(str(repo_path)))

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, repo_path)

    assert report.adopted == []
    assert store.list() == []


def test_adopt_skips_detached_head_block(state_dir: Path, monkeypatch):
    """adopt() must skip detached-HEAD blocks and count them in skipped_detached."""
    repo_path = Path("/repos/myrepo")
    output = _porcelain(
        _main_block(str(repo_path)),
        _detached_block("/store/myrepo/wt-detached"),
    )

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, repo_path)

    assert report.adopted == []
    assert report.skipped_detached == 1


def test_adopt_git_failure_returns_empty_report(state_dir: Path, monkeypatch):
    """adopt() must return an empty AdoptReport when git returns non-zero, not raise."""
    def _fail(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=["git", "worktree", "list", "--porcelain"],
            returncode=1,
            stdout="",
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr(yaml_store_module, "_run_git", _fail)

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, Path("/not/a/repo"))

    assert report.adopted == []
    assert report.skipped_detached == 0


def test_adopt_git_raises_worktree_error(state_dir: Path, monkeypatch):
    """adopt() must return an empty AdoptReport when _run_git raises, not propagate."""
    def _raise(*args, **kwargs):
        raise OSError("simulated git execution error")

    monkeypatch.setattr(yaml_store_module, "_run_git", _raise)

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, Path("/not/a/repo"))

    assert report.adopted == []


def test_adopt_report_contains_adopted_ids(state_dir: Path, monkeypatch):
    """report.adopted must contain the id string of the newly-imported record."""
    repo_path = Path("/repos/myrepo")
    extra_path = "/store/myrepo/wt-x"
    output = _porcelain(_main_block(str(repo_path)), _wt_block(extra_path, "feature/y"))

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, repo_path)

    assert len(report.adopted) == 1
    adopted_id = report.adopted[0]
    assert isinstance(adopted_id, str)
    assert len(adopted_id) > 0
    # The record must be findable by id.
    rec = store.get(adopted_id)
    assert rec is not None
    assert rec.branch == "feature/y"


def test_adopt_zero_extra_worktrees(state_dir: Path, monkeypatch):
    """adopt() with only the main worktree block must return empty report."""
    repo_path = Path("/repos/myrepo")
    output = _porcelain(_main_block(str(repo_path)))

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, repo_path)

    assert report.adopted == []
    assert report.skipped_detached == 0


# ---------------------------------------------------------------------------
# Ticket #10 blocking A — skip by path match, not block order
# ---------------------------------------------------------------------------

def test_adopt_skips_repo_path_when_not_first_block(
    state_dir: Path, tmp_path: Path, monkeypatch
):
    """adopt() must skip BOTH the primary checkout (blocks[0]) AND the passed-in
    repo_path when repo_path is a linked worktree (not blocks[0]).

    Scenario: adopt() is called from a linked worktree.
      primary = <tmp>/primary  (blocks[0] — always the primary checkout)
      linked  = <tmp>/linked   (blocks[1] — this is repo_path passed to adopt())
      extra   = <tmp>/extra    (blocks[2] — an unrelated linked worktree)

    Expected:
      - primary is NOT adopted (it is the real repo dir; adopting it would be
        catastrophic since remove(force=True) would shutil.rmtree it).
      - linked is NOT adopted (it is repo_path itself, the caller's worktree).
      - extra IS adopted (it is a distinct, legitimate linked worktree).

    Uses real tmp_path subdirectories so Path.resolve() is unambiguous on Windows.
    """
    primary_path = tmp_path / "primary"
    linked_path = tmp_path / "linked"   # the repo_path we pass to adopt()
    extra_path = tmp_path / "extra"

    # Porcelain: primary block first (as git always emits), then linked, then extra.
    output = _porcelain(
        _main_block(str(primary_path)),                    # blocks[0] — primary
        _wt_block(str(linked_path), "feature/linked"),     # blocks[1] — repo_path
        _wt_block(str(extra_path), "feature/extra"),       # blocks[2] — adopt this
    )

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, linked_path)

    adopted_paths = {store.get(wid).path for wid in report.adopted}

    # Only 'extra' should have been adopted.
    assert len(report.adopted) == 1, (
        f"expected 1 adoption, got {len(report.adopted)}: {report.adopted}"
    )
    assert extra_path.resolve().as_posix() in adopted_paths, "extra must be adopted"

    # Primary checkout must NOT be adopted (it is the repo itself).
    primary_path_fwd = primary_path.resolve().as_posix()
    assert primary_path_fwd not in adopted_paths, (
        "primary checkout must NOT be adopted"
    )

    # Linked (repo_path) must NOT be adopted.
    linked_path_fwd = linked_path.resolve().as_posix()
    assert linked_path_fwd not in adopted_paths, (
        "linked (repo_path) must NOT be adopted"
    )


# ---------------------------------------------------------------------------
# Ticket #10 blocking B — prunable blocks are skipped, not adopted
# ---------------------------------------------------------------------------

def test_adopt_skips_prunable_block(state_dir: Path, tmp_path: Path, monkeypatch):
    """adopt() must NOT import a worktree whose block contains a 'prunable' line.

    A prunable worktree has had its directory deleted; it does not exist on-disk
    and should be cleaned up via prune(), not recorded as adopted.

    Uses tmp_path for the repo so path resolution is consistent on Windows.
    The prunable path is a fake string — it doesn't need to exist.
    """
    repo_path = tmp_path / "myrepo"
    stale_path = str(tmp_path / "wt-stale")
    output = _porcelain(
        _main_block(str(repo_path)),
        _prunable_block(stale_path, "feature/stale"),
    )

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, repo_path)

    assert report.adopted == []
    assert report.skipped_prunable == 1
    assert store.list() == []


def test_adopt_prunable_and_valid_block_together(
    state_dir: Path, tmp_path: Path, monkeypatch
):
    """A mix of prunable and valid blocks: only the valid one is adopted."""
    repo_path = tmp_path / "myrepo"
    stale_path = str(tmp_path / "wt-stale")
    good_path = str(tmp_path / "wt-good")
    output = _porcelain(
        _main_block(str(repo_path)),
        _prunable_block(stale_path, "feature/stale"),
        _wt_block(good_path, "feature/good"),
    )

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, repo_path)

    assert len(report.adopted) == 1
    assert report.skipped_prunable == 1
    records = store.list()
    assert len(records) == 1
    assert records[0].branch == "feature/good"


# ---------------------------------------------------------------------------
# Ticket #23: adopt() must store paths as forward slashes on all platforms
# ---------------------------------------------------------------------------


def test_adopt_record_paths_use_forward_slashes(
    state_dir: Path, tmp_path: Path, monkeypatch
):
    """Regression: adopt() must store repo_root and path with forward slashes.

    On Windows, Path.resolve() returns backslash-separated strings by default.
    The fix uses Path.as_posix() so the stored strings are always forward-slash,
    making them safe for cross-platform consumers and equality checks.

    Uses monkeypatched _run_git so no real git binary is required.
    """
    repo_path = tmp_path / "myrepo"
    wt_path = tmp_path / "wt-fwdslash"

    # Build fake porcelain output using the real tmp_path values so that
    # Path(wt_path_raw).resolve() returns an absolute Path that as_posix()
    # will normalise to forward slashes.
    output = _porcelain(
        _main_block(str(repo_path)),
        _wt_block(str(wt_path), "feature/fwdslash"),
    )

    monkeypatch.setattr(yaml_store_module, "_run_git", _fake_run_git_ok(output))

    store = YamlStateStore(state_dir=state_dir)
    report = adopt(store, repo_path)

    assert len(report.adopted) == 1, (
        f"expected 1 adoption, got {len(report.adopted)}"
    )
    rec = store.get(report.adopted[0])
    assert rec is not None

    assert "\\" not in rec.repo_root, (
        f"repo_root must use forward slashes, got: {rec.repo_root!r}"
    )
    assert "\\" not in rec.path, (
        f"path must use forward slashes, got: {rec.path!r}"
    )


# ---------------------------------------------------------------------------
# R4 (ticket #84) -- `backing` round-trips and legacy state.yaml still
# deserialises
# ---------------------------------------------------------------------------

def test_backing_round_trips_and_defaults_for_legacy_entries(state_dir: Path):
    """R4 driving test: a legacy state.yaml entry with no `backing` key (and
    `branch: null`) still deserialises, defaulting backing to "worktree";
    an explicit primary entry round-trips backing="primary"."""
    state_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        "version": 1,
        "worktrees": {
            "legacy-wt": {
                "id": "legacy-wt",
                "repo_root": "/repos/myrepo",
                "branch": None,
                "path": "/store/myrepo/legacy-wt",
                "status": "created",
                "ports": {},
                "pids": {},
                "branch_created_by_us": False,
                # no "backing" key at all -- simulates a pre-#84 state.yaml.
            },
        },
    }
    (state_dir / "state.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    store = YamlStateStore(state_dir=state_dir)
    legacy = store.get("legacy-wt")
    assert legacy is not None
    assert legacy.backing == "worktree"
    assert legacy.branch is None

    primary_rec = _make_record(id="primary-1", branch=None, path="/repos/myrepo")
    primary_rec.backing = "primary"
    store.add(primary_rec)

    reloaded_store = YamlStateStore(state_dir=state_dir)
    reloaded = reloaded_store.get("primary-1")
    assert reloaded is not None
    assert reloaded.backing == "primary"
    assert reloaded.branch is None


def test_find_by_branch_skips_primary_records_yaml(state_dir: Path):
    """A primary record must never shadow a create() duplicate-branch check
    via find_by_branch() (ticket #84, R4 edge case).

    Deliberately gives the primary record a matching ``branch="main"`` (even
    though a real primary always stores ``branch=None``) so the assertion
    proves the explicit ``backing == "primary"`` skip is doing the work --
    not merely that the branch values happen not to match.
    """
    store = YamlStateStore(state_dir=state_dir)
    primary_rec = _make_record(
        id="primary-shadow", repo_root="/repos/myrepo", branch="main",
        path="/repos/myrepo",
    )
    primary_rec.backing = "primary"
    store.add(primary_rec)

    assert store.find_by_branch("/repos/myrepo", "main") is None


# ---------------------------------------------------------------------------
# TestPidAliveWindowsAccessDenied -- ticket #95, R4
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform != "win32",
    reason="_pid_alive_windows exercises win32-only ctypes APIs "
    "(ctypes.get_last_error/set_last_error, kernel32 bindings)",
)
class TestPidAliveWindowsAccessDenied:
    """R4 (ticket #95): ``_pid_alive_windows`` must not read ACCESS_DENIED as
    dead.

    Root cause (finding 4): the original implementation opened with only
    ``PROCESS_QUERY_INFORMATION`` and returned ``False`` on *any*
    ``OpenProcess`` failure -- including ``ERROR_ACCESS_DENIED``, which
    actually proves the PID still exists (the OS returns
    ``ERROR_INVALID_PARAMETER`` for a PID that is genuinely gone). A live
    survivor process that ``stop()`` cannot open (e.g. spawned by another
    user/elevated context) therefore silently read as already dead, masking
    a real leaked process from the survivor re-probe.
    """

    @staticmethod
    def _get_exit_code_side_effect(value: int):
        def _side_effect(handle, ref):
            ref._obj.value = value
            return 1

        return _side_effect

    def test_access_denied_reads_as_alive(self):
        """Driving test: both OpenProcess attempts fail and GetLastError()
        reports ERROR_ACCESS_DENIED (5) -- the PID must read as alive, not
        dead."""
        kernel32 = MagicMock()
        kernel32.OpenProcess.return_value = 0
        with (
            patch(
                "lib_python_worktree.core.yaml_store._kernel32_lasterror",
                return_value=kernel32,
            ),
            patch("ctypes.get_last_error", return_value=5),
            patch("ctypes.set_last_error"),
        ):
            assert _pid_alive_windows(1234) is True

    def test_invalid_parameter_reads_as_dead(self):
        kernel32 = MagicMock()
        kernel32.OpenProcess.return_value = 0
        with (
            patch(
                "lib_python_worktree.core.yaml_store._kernel32_lasterror",
                return_value=kernel32,
            ),
            patch("ctypes.get_last_error", return_value=87),
            patch("ctypes.set_last_error"),
        ):
            assert _pid_alive_windows(999999) is False

    def test_unknown_error_code_reads_as_dead(self):
        kernel32 = MagicMock()
        kernel32.OpenProcess.return_value = 0
        with (
            patch(
                "lib_python_worktree.core.yaml_store._kernel32_lasterror",
                return_value=kernel32,
            ),
            patch("ctypes.get_last_error", return_value=1234),
            patch("ctypes.set_last_error"),
        ):
            assert _pid_alive_windows(555) is False

    def test_limited_info_tried_before_falling_back(self):
        """PROCESS_QUERY_LIMITED_INFORMATION (0x1000) is tried first; only
        falls back to the classic PROCESS_QUERY_INFORMATION (0x0400) if that
        fails."""
        kernel32 = MagicMock()
        calls = []

        def _open_process(desired_access, inherit, pid):
            calls.append(desired_access)
            return 0

        kernel32.OpenProcess.side_effect = _open_process
        with (
            patch(
                "lib_python_worktree.core.yaml_store._kernel32_lasterror",
                return_value=kernel32,
            ),
            patch("ctypes.get_last_error", return_value=87),
            patch("ctypes.set_last_error"),
        ):
            _pid_alive_windows(111)
        assert calls == [0x1000, 0x0400]

    def test_still_active_unchanged(self):
        """Happy path unchanged by the hardening: a successfully opened
        handle with a STILL_ACTIVE exit code reports alive."""
        kernel32 = MagicMock()
        kernel32.OpenProcess.return_value = 42
        kernel32.GetExitCodeProcess.side_effect = self._get_exit_code_side_effect(259)
        with patch(
            "lib_python_worktree.core.yaml_store._kernel32_lasterror",
            return_value=kernel32,
        ):
            assert _pid_alive_windows(42) is True

    def test_exited_process_reads_as_dead(self):
        kernel32 = MagicMock()
        kernel32.OpenProcess.return_value = 42
        kernel32.GetExitCodeProcess.side_effect = self._get_exit_code_side_effect(0)
        with patch(
            "lib_python_worktree.core.yaml_store._kernel32_lasterror",
            return_value=kernel32,
        ):
            assert _pid_alive_windows(42) is False

    def test_real_own_pid_is_alive(self):
        """End-to-end sanity check against a real Windows process: our own
        PID must read as alive."""
        assert _pid_alive_windows(os.getpid()) is True


# ---------------------------------------------------------------------------
# TestJobNameRoundTrip -- ticket #95, R5
# ---------------------------------------------------------------------------

class TestJobNameRoundTrip:
    """R5 (ticket #95): ``WorktreeRecord.job_names`` (per-role mapping,
    reviewer fix cycle -- was originally a single record-wide scalar
    ``job_name``) persists through ``state.yaml``, and a pre-fix record with
    no ``job_names``/``job_name`` key at all still deserialises (defaulting
    to ``{}``)."""

    def test_job_name_round_trips(self, state_dir: Path):
        """Driving test: a record with real per-role job names round-trips
        through a fresh YamlStateStore load."""
        store = YamlStateStore(state_dir=state_dir)
        record = _make_record(id="wt-job-name")
        record.job_names = {
            "main": "Local\\worktree-deadbeefcafebabe",
            "worker": "Local\\worktree-feedfacecafebeef",
        }
        store.add(record)

        reloaded_store = YamlStateStore(state_dir=state_dir)
        reloaded = reloaded_store.get("wt-job-name")
        assert reloaded is not None
        assert reloaded.job_names == {
            "main": "Local\\worktree-deadbeefcafebabe",
            "worker": "Local\\worktree-feedfacecafebeef",
        }

    def test_legacy_record_without_job_name_key_defaults_to_none(self, state_dir: Path):
        """A pre-fix state.yaml entry with no `job_names` (nor the older
        scalar `job_name`) key at all must still deserialise, defaulting
        job_names to an empty dict."""
        state_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "version": 1,
            "worktrees": {
                "legacy-wt-job": {
                    "id": "legacy-wt-job",
                    "repo_root": "/repos/myrepo",
                    "branch": "main",
                    "path": "/store/myrepo/legacy-wt-job",
                    "status": "created",
                    "ports": {},
                    "pids": {},
                    "branch_created_by_us": False,
                    "backing": "worktree",
                    # no "job_names" (nor "job_name") key at all -- simulates
                    # a pre-fix state.yaml.
                },
            },
        }
        (state_dir / "state.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

        store = YamlStateStore(state_dir=state_dir)
        legacy = store.get("legacy-wt-job")
        assert legacy is not None
        assert legacy.job_names == {}


# ---------------------------------------------------------------------------
# TestVariantsRoundTrip -- ticket #104
# ---------------------------------------------------------------------------

class TestVariantsRoundTrip:
    """B1 (ticket #104): ``WorktreeRecord.variants`` (per-role mapping of
    role -> the variant that started it) persists through ``state.yaml``,
    and a pre-fix record with no ``variants`` key at all still deserialises
    (defaulting to ``{}``)."""

    def test_variants_round_trip(self, state_dir: Path):
        """Driving test: a record with a real per-role variants mapping
        round-trips through a fresh YamlStateStore load."""
        store = YamlStateStore(state_dir=state_dir)
        record = _make_record(id="wt-variants")
        record.variants = {"main": "default", "web": "web"}
        store.add(record)

        reloaded_store = YamlStateStore(state_dir=state_dir)
        reloaded = reloaded_store.get("wt-variants")
        assert reloaded is not None
        assert reloaded.variants == {"main": "default", "web": "web"}

    def test_legacy_record_without_variants_key_defaults_to_empty(self, state_dir: Path):
        """A pre-fix state.yaml entry with no `variants` key at all must
        still deserialise, defaulting variants to an empty dict."""
        state_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "version": 1,
            "worktrees": {
                "legacy-wt-variants": {
                    "id": "legacy-wt-variants",
                    "repo_root": "/repos/myrepo",
                    "branch": "main",
                    "path": "/store/myrepo/legacy-wt-variants",
                    "status": "created",
                    "ports": {},
                    "pids": {},
                    "branch_created_by_us": False,
                    "backing": "worktree",
                    # no "variants" key at all -- simulates a pre-fix
                    # state.yaml.
                },
            },
        }
        (state_dir / "state.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

        store = YamlStateStore(state_dir=state_dir)
        legacy = store.get("legacy-wt-variants")
        assert legacy is not None
        assert legacy.variants == {}


# ---------------------------------------------------------------------------
# TestStopDetailPersistence -- ticket #99
# ---------------------------------------------------------------------------

class TestStopDetailPersistence:
    """B3 (ticket #99): ``stop_detail`` survives a host restart (persisted
    through ``state.yaml``), a legacy record with no ``stop_detail`` key
    still loads, and an unrecognised key inside it is silently ignored."""

    def test_stop_detail_round_trips(self, state_dir: Path):
        """Driving test: a record with a real StopDetail round-trips through
        a fresh YamlStateStore load."""
        store = YamlStateStore(state_dir=state_dir)
        record = _make_record(id="wt-stop-detail", status="stop_incomplete")
        record.stop_detail = StopDetail(
            reason=STOP_REASON_SURVIVORS,
            message=(
                "stop(worktree_id=wt-stop-detail, role=main): process(es) "
                "survived termination: [31337]"
            ),
            role="main",
            survivor_pids=(31337,),
            survivor_count=1,
            kill_orphans_may_help=True,
        )
        store.add(record)

        reloaded_store = YamlStateStore(state_dir=state_dir)
        reloaded = reloaded_store.get("wt-stop-detail")
        assert reloaded is not None
        assert reloaded.stop_detail == record.stop_detail

    def test_legacy_record_without_stop_detail_key_defaults_to_none(self, state_dir: Path):
        """A pre-#99 state.yaml entry with no `stop_detail` key at all must
        still deserialise, defaulting stop_detail to None."""
        state_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "version": 1,
            "worktrees": {
                "legacy-wt-stop": {
                    "id": "legacy-wt-stop",
                    "repo_root": "/repos/myrepo",
                    "branch": "main",
                    "path": "/store/myrepo/legacy-wt-stop",
                    "status": "stop_incomplete",
                    "ports": {},
                    "pids": {},
                    "branch_created_by_us": False,
                    "backing": "worktree",
                    "job_names": {},
                    # no "stop_detail" key at all -- simulates a pre-#99
                    # state.yaml.
                },
            },
        }
        (state_dir / "state.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

        store = YamlStateStore(state_dir=state_dir)
        legacy = store.get("legacy-wt-stop")
        assert legacy is not None
        assert legacy.stop_detail is None

    def test_unknown_stop_detail_key_is_ignored(self, state_dir: Path):
        """Forward compat: a state.yaml written by a future engine version
        may carry extra keys inside stop_detail this version does not know
        about -- they must be silently ignored, not raise."""
        state_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "version": 1,
            "worktrees": {
                "future-wt-stop": {
                    "id": "future-wt-stop",
                    "repo_root": "/repos/myrepo",
                    "branch": "main",
                    "path": "/store/myrepo/future-wt-stop",
                    "status": "stop_incomplete",
                    "ports": {},
                    "pids": {},
                    "branch_created_by_us": False,
                    "backing": "worktree",
                    "job_names": {},
                    "stop_detail": {
                        "reason": "survivors",
                        "message": "future msg",
                        "role": "main",
                        "survivor_pids": [123],
                        "survivor_count": 1,
                        "truncated_at": None,
                        "skipped_passes": [],
                        "kill_orphans_may_help": True,
                        "future_field_this_version_predates": "surprise",
                    },
                },
            },
        }
        (state_dir / "state.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

        store = YamlStateStore(state_dir=state_dir)
        future = store.get("future-wt-stop")
        assert future is not None
        assert future.stop_detail is not None
        assert future.stop_detail.reason == "survivors"
        assert future.stop_detail.survivor_pids == (123,)

    def test_survivor_pids_capped_in_persisted_detail(self, state_dir: Path):
        """A StopDetail carrying more than _STOP_DETAIL_MAX_PIDS survivor
        pids (e.g. constructed by a future engine version) is capped on the
        way to disk (and again on the way back); survivor_count still holds
        the true total."""
        store = YamlStateStore(state_dir=state_dir)
        many_pids = tuple(range(1000, 1100))  # 100 pids, well over the cap
        record = _make_record(id="wt-many-survivors", status="stop_incomplete")
        record.stop_detail = StopDetail(
            reason=STOP_REASON_SURVIVORS,
            message="many survivors",
            role="main",
            survivor_pids=many_pids,
            survivor_count=len(many_pids),
        )
        store.add(record)

        reloaded_store = YamlStateStore(state_dir=state_dir)
        reloaded = reloaded_store.get("wt-many-survivors")
        assert reloaded is not None
        assert reloaded.stop_detail is not None
        assert len(reloaded.stop_detail.survivor_pids) == _STOP_DETAIL_MAX_PIDS
        assert reloaded.stop_detail.survivor_count == 100


# ---------------------------------------------------------------------------
# TestSetupOutcomePersistence -- ticket #105
# ---------------------------------------------------------------------------

class TestSetupOutcomePersistence:
    """``setup_outcome`` persists through ``state.yaml`` exactly like
    ``stop_detail``: a legacy record with no ``setup_outcome`` key still
    loads (defaulting to ``None``), and an unrecognised status / extra key
    inside it is silently forward-compatible."""

    def test_setup_outcome_round_trips(self, state_dir: Path):
        """Driving test: a record with a fully-populated SetupOutcome
        round-trips through a fresh YamlStateStore load, field-by-field,
        including timed_out and the forward-slash log_path."""
        store = YamlStateStore(state_dir=state_dir)
        record = _make_record(id="wt-setup-outcome")
        record.setup_outcome = SetupOutcome(
            status=SETUP_STATUS_FAILED,
            message=(
                "setup step 0 ('build') for worktree 'wt-setup-outcome' "
                "failed with exit code 1. See log: C:/logs/0-build.log"
            ),
            completed_at="2026-08-17T12:00:00+00:00",
            steps_run=0,
            failed_step_index=0,
            failed_step_name="build",
            log_path="C:/logs/setup/wt-setup-outcome/0-build.log",
            returncode=1,
            timed_out=True,
        )
        store.add(record)

        reloaded_store = YamlStateStore(state_dir=state_dir)
        reloaded = reloaded_store.get("wt-setup-outcome")
        assert reloaded is not None
        assert reloaded.setup_outcome == record.setup_outcome
        assert reloaded.setup_outcome.log_path == "C:/logs/setup/wt-setup-outcome/0-build.log"
        assert reloaded.setup_outcome.timed_out is True

    def test_legacy_record_without_setup_outcome_key_defaults_to_none(self, state_dir: Path):
        """A pre-#105 state.yaml entry with no `setup_outcome` key at all
        must still deserialise, defaulting setup_outcome to None."""
        state_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "version": 1,
            "worktrees": {
                "legacy-wt-setup": {
                    "id": "legacy-wt-setup",
                    "repo_root": "/repos/myrepo",
                    "branch": "main",
                    "path": "/store/myrepo/legacy-wt-setup",
                    "status": "created",
                    "ports": {},
                    "pids": {},
                    "branch_created_by_us": False,
                    "backing": "worktree",
                    "job_names": {},
                    # no "setup_outcome" key at all -- simulates a pre-#105
                    # state.yaml.
                },
            },
        }
        (state_dir / "state.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

        store = YamlStateStore(state_dir=state_dir)
        legacy = store.get("legacy-wt-setup")
        assert legacy is not None
        assert legacy.setup_outcome is None

    def test_unknown_setup_outcome_status_and_extra_key_preserved(self, state_dir: Path):
        """Forward compat: a state.yaml written by a future engine version
        may carry an unrecognised `status` value and extra keys inside
        `setup_outcome` this version does not know about -- both must be
        handled without raising; the extra key is silently ignored and the
        unrecognised status is preserved verbatim."""
        state_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "version": 1,
            "worktrees": {
                "future-wt-setup": {
                    "id": "future-wt-setup",
                    "repo_root": "/repos/myrepo",
                    "branch": "main",
                    "path": "/store/myrepo/future-wt-setup",
                    "status": "created",
                    "ports": {},
                    "pids": {},
                    "branch_created_by_us": False,
                    "backing": "worktree",
                    "job_names": {},
                    "setup_outcome": {
                        "status": "partially_completed",
                        "message": "future msg",
                        "completed_at": "2099-01-01T00:00:00+00:00",
                        "steps_run": 3,
                        "future_field_this_version_predates": "surprise",
                    },
                },
            },
        }
        (state_dir / "state.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

        store = YamlStateStore(state_dir=state_dir)
        future = store.get("future-wt-setup")
        assert future is not None
        assert future.setup_outcome is not None
        assert future.setup_outcome.status == "partially_completed"
        assert future.setup_outcome.steps_run == 3

    @pytest.mark.parametrize("raw_value", [None, {}])
    def test_setup_outcome_null_and_empty_dict_deserialize_to_none(
        self, state_dir: Path, raw_value
    ):
        """`setup_outcome: null` and `setup_outcome: {}` both deserialise to
        None, mirroring `stop_detail`'s falsy-dict handling."""
        state_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "version": 1,
            "worktrees": {
                "wt-falsy-setup": {
                    "id": "wt-falsy-setup",
                    "repo_root": "/repos/myrepo",
                    "branch": "main",
                    "path": "/store/myrepo/wt-falsy-setup",
                    "status": "created",
                    "ports": {},
                    "pids": {},
                    "branch_created_by_us": False,
                    "backing": "worktree",
                    "job_names": {},
                    "setup_outcome": raw_value,
                },
            },
        }
        (state_dir / "state.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

        store = YamlStateStore(state_dir=state_dir)
        rec = store.get("wt-falsy-setup")
        assert rec is not None
        assert rec.setup_outcome is None


# ---------------------------------------------------------------------------
# TestTeardownRanPersistence -- ticket #126
# ---------------------------------------------------------------------------

class TestTeardownRanPersistence:
    """``teardown_ran`` persists through ``state.yaml`` as an unconditional
    scalar (mirrors ``branch_created_by_us``, not the ``Optional`` style of
    ``stop_detail``/``setup_outcome``): always present in the serialised
    dict, and a legacy record with no ``teardown_ran`` key deserialises to
    ``False``."""

    def test_teardown_ran_round_trips_true(self, state_dir: Path):
        """Driving test: a record with teardown_ran=True round-trips
        through a fresh YamlStateStore load."""
        store = YamlStateStore(state_dir=state_dir)
        record = _make_record(id="wt-teardown-ran")
        record.teardown_ran = True
        store.add(record)

        reloaded_store = YamlStateStore(state_dir=state_dir)
        reloaded = reloaded_store.get("wt-teardown-ran")
        assert reloaded is not None
        assert reloaded.teardown_ran is True

    def test_teardown_ran_round_trips_false(self, state_dir: Path):
        """The default False value also round-trips (not merely omitted)."""
        store = YamlStateStore(state_dir=state_dir)
        record = _make_record(id="wt-teardown-not-ran")
        assert record.teardown_ran is False
        store.add(record)

        reloaded_store = YamlStateStore(state_dir=state_dir)
        reloaded = reloaded_store.get("wt-teardown-not-ran")
        assert reloaded is not None
        assert reloaded.teardown_ran is False

    def test_legacy_record_without_teardown_ran_key_defaults_to_false(
        self, state_dir: Path
    ):
        """A pre-#126 state.yaml entry with no `teardown_ran` key at all
        must still deserialise, defaulting teardown_ran to False."""
        state_dir.mkdir(parents=True, exist_ok=True)
        raw = {
            "version": 1,
            "worktrees": {
                "legacy-wt-teardown": {
                    "id": "legacy-wt-teardown",
                    "repo_root": "/repos/myrepo",
                    "branch": "main",
                    "path": "/store/myrepo/legacy-wt-teardown",
                    "status": "created",
                    "ports": {},
                    "pids": {},
                    "branch_created_by_us": False,
                    "backing": "worktree",
                    "job_names": {},
                    # no "teardown_ran" key at all -- simulates a pre-#126
                    # state.yaml.
                },
            },
        }
        (state_dir / "state.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

        store = YamlStateStore(state_dir=state_dir)
        legacy = store.get("legacy-wt-teardown")
        assert legacy is not None
        assert legacy.teardown_ran is False

    def test_serialised_dict_always_includes_teardown_ran_key(self, state_dir: Path):
        """Unconditional-scalar convention check: the key is present in the
        serialised dict even for the (unconditional) False default -- unlike
        the Optional ``stop_detail``/``setup_outcome`` fields, there is no
        "field never reached" state to represent."""
        store = YamlStateStore(state_dir=state_dir)
        record = _make_record(id="wt-teardown-key-present")
        store.add(record)

        raw = yaml.safe_load((state_dir / "state.yaml").read_text(encoding="utf-8"))
        stored = raw["worktrees"]["wt-teardown-key-present"]
        assert "teardown_ran" in stored
        assert stored["teardown_ran"] is False


# ---------------------------------------------------------------------------
# TestSetupOutcomeSurvivesLifecycle -- ticket #105
# ---------------------------------------------------------------------------

class TestSetupOutcomeSurvivesLifecycle:
    """``setup_outcome`` must be left untouched by ``start()``/``stop()`` --
    only ``create()``'s ``setup:`` hook block ever assigns it. Uses
    ``YamlStateStore`` (not ``InMemoryStateStore``) specifically because an
    in-memory store keeps a record by reference and would hide a missing-
    serialisation regression."""

    def test_setup_outcome_survives_start_stop_cycle(
        self, state_dir: Path, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setenv("WORKTREE_LOG_ROOT", str(tmp_path / "logs"))
        store = YamlStateStore(state_dir=state_dir)
        record = _make_record(id="wt-setup-lifecycle")
        record.setup_outcome = SetupOutcome(
            status=SETUP_STATUS_COMPLETED,
            message="setup: completed 1 step(s)",
            completed_at="2026-08-17T12:00:00+00:00",
            steps_run=1,
        )
        store.add(record)
        original_outcome = record.setup_outcome

        started = _lifecycle_start(
            "wt-setup-lifecycle",
            [sys.executable, "-c", "import time; time.sleep(60)"],
            store=store,
        )
        assert started.status == "running"
        assert started.setup_outcome == original_outcome
        pid = started.pids["main"]

        try:
            stopped = _lifecycle_stop("wt-setup-lifecycle", store=store)
        finally:
            try:
                if _pid_alive(pid):
                    _force_kill(pid)
            except Exception:  # noqa: BLE001
                pass

        assert stopped.status == "stopped"
        assert stopped.setup_outcome == original_outcome, (
            "setup_outcome must be byte-identical before/after the "
            "start/stop cycle -- status changed, setup_outcome must not."
        )

        # Reload from a fresh store instance -- this is the assertion that
        # actually catches a missing-serialisation regression (an
        # InMemoryStateStore-backed check would hide it).
        reloaded_store = YamlStateStore(state_dir=state_dir)
        reloaded = reloaded_store.get("wt-setup-lifecycle")
        assert reloaded is not None
        assert reloaded.status == "stopped"
        assert reloaded.setup_outcome == original_outcome
