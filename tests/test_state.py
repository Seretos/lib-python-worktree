"""Unit tests for InMemoryStateStore (core/state.py).

Pure in-memory tests — no git binary required, no marker.
"""

from __future__ import annotations

import pytest

from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord


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
