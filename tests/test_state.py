"""Unit tests for InMemoryStateStore (core/state.py).

Pure in-memory tests — no git binary required, no marker.
"""

from __future__ import annotations

import pytest

from lib_python_worktree.core.state import (
    SETUP_STATUS_COMPLETED,
    SETUP_STATUSES,
    STOP_ATTEMPT_ALREADY_EXITED,
    STOP_ATTEMPT_KILLED,
    STOP_ATTEMPT_NO_PROCESS_RECORDED,
    STOP_ATTEMPT_OUTCOMES,
    STOP_ATTEMPT_TRACKED_PID_MISSING,
    STOP_ATTEMPT_UNCONFIRMED_ALIVE,
    STOP_NO_OP_ISOLATION_NONE,
    STOP_NO_OP_NO_PROCESS_RECORDED,
    STOP_NO_OP_REASONS,
    InMemoryStateStore,
    StopAttempt,
    StopHookOutcome,
    WorktreeRecord,
)


def _make_record(
    id: str = "rec-001",
    repo_root: str = "/repos/myrepo",
    branch: str = "main",
    path: str = "/store/myrepo/rec-001",
) -> WorktreeRecord:
    return WorktreeRecord(id=id, repo_root=repo_root, branch=branch, path=path)


# ---------------------------------------------------------------------------
# WorktreeRecord default fields
# ---------------------------------------------------------------------------

def test_worktree_record_default_status():
    rec = _make_record()
    assert rec.status == "created"


def test_worktree_record_default_ports():
    rec = _make_record()
    assert rec.ports == {}


def test_worktree_record_default_setup_outcome_is_none():
    """Ticket #105: a freshly constructed WorktreeRecord (never passed
    through create()) has setup_outcome is None -- distinct from
    status="skipped", which only create() ever sets."""
    rec = _make_record()
    assert rec.setup_outcome is None


def test_worktree_record_default_variants():
    """Ticket #104: a freshly constructed WorktreeRecord has variants == {},
    and two records do not share the same dict instance (default_factory,
    not a mutable default)."""
    rec = _make_record()
    assert rec.variants == {}

    rec2 = _make_record(id="rec-002")
    rec.variants["main"] = "web"
    assert rec2.variants == {}


def test_worktree_record_default_start_variants_is_empty_list():
    """Ticket #146 (R1): a freshly constructed WorktreeRecord has
    start_variants == [] -- non-Optional, defaulting to an empty list, never
    None -- and two independently constructed records do not share the same
    list instance (default_factory, not a mutable default). Mirrors
    test_worktree_record_default_variants immediately above."""
    rec = _make_record()
    assert rec.start_variants == []
    assert rec.start_variants is not None

    rec2 = _make_record(id="rec-002")
    rec.start_variants.append("stale")
    assert rec2.start_variants == []


# ---------------------------------------------------------------------------
# add + get roundtrip
# ---------------------------------------------------------------------------

def test_add_get_roundtrip():
    store = InMemoryStateStore()
    rec = _make_record()
    store.add(rec)
    retrieved = store.get("rec-001")
    assert retrieved is rec


def test_get_missing_returns_none():
    store = InMemoryStateStore()
    assert store.get("does-not-exist") is None


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------

def test_remove_existing_returns_record_then_get_none():
    store = InMemoryStateStore()
    rec = _make_record()
    store.add(rec)
    removed = store.remove("rec-001")
    assert removed is rec
    assert store.get("rec-001") is None


def test_remove_missing_returns_none():
    store = InMemoryStateStore()
    result = store.remove("no-such-id")
    assert result is None


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def test_list_empty():
    store = InMemoryStateStore()
    assert store.list() == []


def test_list_populated():
    store = InMemoryStateStore()
    r1 = _make_record(id="r1")
    r2 = _make_record(id="r2", branch="feature/x", path="/store/myrepo/r2")
    store.add(r1)
    store.add(r2)
    listed = store.list()
    assert len(listed) == 2
    ids = {r.id for r in listed}
    assert ids == {"r1", "r2"}


def test_list_after_remove():
    store = InMemoryStateStore()
    r1 = _make_record(id="r1")
    r2 = _make_record(id="r2", branch="feature/x", path="/store/myrepo/r2")
    store.add(r1)
    store.add(r2)
    store.remove("r1")
    listed = store.list()
    assert len(listed) == 1
    assert listed[0].id == "r2"


# ---------------------------------------------------------------------------
# add duplicate id raises ValueError
# ---------------------------------------------------------------------------

def test_add_duplicate_id_raises():
    store = InMemoryStateStore()
    store.add(_make_record(id="dup"))
    with pytest.raises(ValueError, match="dup"):
        store.add(_make_record(id="dup"))


# ---------------------------------------------------------------------------
# find_by_branch
# ---------------------------------------------------------------------------

def test_find_by_branch_match():
    store = InMemoryStateStore()
    rec = _make_record(repo_root="/repos/myrepo", branch="feature/beta")
    store.add(rec)
    found = store.find_by_branch("/repos/myrepo", "feature/beta")
    assert found is rec


def test_find_by_branch_no_match_wrong_branch():
    store = InMemoryStateStore()
    store.add(_make_record(repo_root="/repos/myrepo", branch="main"))
    assert store.find_by_branch("/repos/myrepo", "feature/other") is None


def test_find_by_branch_no_match_wrong_repo_root():
    store = InMemoryStateStore()
    store.add(_make_record(repo_root="/repos/myrepo", branch="main"))
    assert store.find_by_branch("/repos/other", "main") is None


def test_find_by_branch_empty_store_returns_none():
    store = InMemoryStateStore()
    assert store.find_by_branch("/repos/myrepo", "main") is None


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

def test_update_replaces_record():
    store = InMemoryStateStore()
    rec = _make_record(id="upd-001")
    store.add(rec)
    updated = WorktreeRecord(
        id="upd-001",
        repo_root="/repos/myrepo",
        branch="main",
        path="/store/myrepo/upd-001",
        status="stopped",
    )
    store.update(updated)
    retrieved = store.get("upd-001")
    assert retrieved is updated
    assert retrieved.status == "stopped"


def test_update_missing_raises_key_error():
    store = InMemoryStateStore()
    rec = _make_record(id="ghost")
    with pytest.raises(KeyError):
        store.update(rec)


# ---------------------------------------------------------------------------
# backing field (ticket #84)
# ---------------------------------------------------------------------------

def test_worktree_record_default_backing_is_worktree():
    rec = _make_record()
    assert rec.backing == "worktree"


# ---------------------------------------------------------------------------
# StopAttempt / STOP_ATTEMPT_OUTCOMES (ticket #110)
# ---------------------------------------------------------------------------

def test_worktree_record_default_stop_attempt_is_none():
    """A freshly constructed WorktreeRecord (never passed through stop())
    has stop_attempt is None -- transient, like killed_pids."""
    rec = _make_record()
    assert rec.stop_attempt is None


def test_stop_attempt_outcomes_vocabulary_membership():
    """STOP_ATTEMPT_OUTCOMES names exactly the five outcome tags used by
    process_lifecycle.stop() and WorktreeManager.stop()'s no-op branch --
    ticket #157 fix cycle (R1, blocking) added "unconfirmed_alive" for a
    tracked pid that is alive at entry but whose identity could not be
    confirmed as ours, distinct from "already_exited" (which would be
    actively false for that case)."""
    assert set(STOP_ATTEMPT_OUTCOMES) == {
        STOP_ATTEMPT_KILLED,
        STOP_ATTEMPT_ALREADY_EXITED,
        STOP_ATTEMPT_TRACKED_PID_MISSING,
        STOP_ATTEMPT_NO_PROCESS_RECORDED,
        STOP_ATTEMPT_UNCONFIRMED_ALIVE,
    }


def test_stop_attempt_defaults():
    """StopAttempt's optional fields default to None/False, mirroring
    StopDetail's default-field convention."""
    attempt = StopAttempt(outcome=STOP_ATTEMPT_KILLED, message="killed pid 123")
    assert attempt.role is None
    assert attempt.tracked_pid is None
    assert attempt.tracked_pid_alive is False
    assert attempt.kill_orphans_may_help is False


# ---------------------------------------------------------------------------
# StopHookOutcome / STOP_NO_OP_REASONS (ticket #128)
# ---------------------------------------------------------------------------

def test_worktree_record_default_stop_hook_outcome_is_none():
    """A freshly constructed WorktreeRecord (never passed through stop())
    has stop_hook_outcome is None -- transient, like stop_attempt."""
    rec = _make_record()
    assert rec.stop_hook_outcome is None


def test_stop_no_op_reasons_vocabulary_membership():
    """STOP_NO_OP_REASONS names exactly the two no_op_reason tags used by
    WorktreeManager.stop()'s no-op branch."""
    assert set(STOP_NO_OP_REASONS) == {
        STOP_NO_OP_ISOLATION_NONE,
        STOP_NO_OP_NO_PROCESS_RECORDED,
    }


def test_setup_statuses_vocabulary_still_membership_complete():
    """StopHookOutcome.status reuses SETUP_STATUSES rather than minting a
    new vocabulary -- this just re-confirms membership from the StopHookOutcome
    side of that reuse."""
    assert SETUP_STATUS_COMPLETED in SETUP_STATUSES


def test_stop_hook_outcome_defaults():
    """StopHookOutcome's optional fields default to falsy/None, mirroring
    SetupOutcome's default-field convention."""
    outcome = StopHookOutcome(status=SETUP_STATUS_COMPLETED)
    assert outcome.message == ""
    assert outcome.steps_run == 0
    assert outcome.contract_found is False
    assert outcome.contract_path is None
    assert outcome.contract_isolation is None
    assert outcome.no_op_reason is None


def test_find_by_branch_skips_primary_records():
    """A primary record must never shadow a create() duplicate-branch check
    via find_by_branch() (ticket #84, R4 edge case).

    Deliberately gives the primary record a matching ``branch="main"`` (even
    though a real primary always stores ``branch=None``) so the assertion
    proves the explicit ``backing == "primary"`` skip is doing the work --
    not merely that the branch values happen not to match.
    """
    store = InMemoryStateStore()
    primary_rec = WorktreeRecord(
        id="primary-shadow",
        repo_root="/repos/myrepo",
        branch="main",
        path="/repos/myrepo",
        backing="primary",
    )
    store.add(primary_rec)
    assert store.find_by_branch("/repos/myrepo", "main") is None
