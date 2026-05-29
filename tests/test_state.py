"""Tests for the in-memory state store (W2 core/state.py).

Verifies the real InMemoryStateStore API: add / get / remove / list /
find_by_branch, plus WorktreeRecord field defaults.
"""

from __future__ import annotations

import pytest

from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord


def _make_record(
    wt_id: str = "repo-branch-abc12345",
    repo_root: str = "/repo",
    branch: str = "main",
    path: str = "/store/repo/repo-branch-abc12345",
) -> WorktreeRecord:
    return WorktreeRecord(id=wt_id, repo_root=repo_root, branch=branch, path=path)


# ---- WorktreeRecord defaults ----


def test_worktree_record_default_status():
    rec = _make_record()
    assert rec.status == "created"


def test_worktree_record_default_ports_empty():
    rec = _make_record()
    assert rec.ports == {}


def test_worktree_record_fields():
    rec = _make_record(
        wt_id="myid", repo_root="/r", branch="feat", path="/p"
    )
    assert rec.id == "myid"
    assert rec.repo_root == "/r"
    assert rec.branch == "feat"
    assert rec.path == "/p"


# ---- InMemoryStateStore.add ----


def test_add_and_get():
    store = InMemoryStateStore()
    rec = _make_record()
    store.add(rec)
    assert store.get(rec.id) is rec


def test_add_duplicate_raises():
    store = InMemoryStateStore()
    rec = _make_record()
    store.add(rec)
    with pytest.raises(ValueError, match="already tracked"):
        store.add(rec)


# ---- InMemoryStateStore.get ----


def test_get_nonexistent_returns_none():
    store = InMemoryStateStore()
    assert store.get("no-such-id") is None


# ---- InMemoryStateStore.remove ----


def test_remove_existing_returns_record():
    store = InMemoryStateStore()
    rec = _make_record()
    store.add(rec)
    removed = store.remove(rec.id)
    assert removed is rec


def test_remove_existing_no_longer_listed():
    store = InMemoryStateStore()
    rec = _make_record()
    store.add(rec)
    store.remove(rec.id)
    assert store.list() == []


def test_remove_nonexistent_returns_none():
    store = InMemoryStateStore()
    result = store.remove("never-added")
    assert result is None


# ---- InMemoryStateStore.list ----


def test_list_empty():
    store = InMemoryStateStore()
    assert store.list() == []


def test_list_multiple_records():
    store = InMemoryStateStore()
    r1 = _make_record(wt_id="id-1", branch="feature/one")
    r2 = _make_record(wt_id="id-2", branch="feature/two")
    store.add(r1)
    store.add(r2)
    listed = store.list()
    assert len(listed) == 2
    assert {r.id for r in listed} == {"id-1", "id-2"}


def test_list_returns_copy_not_internal():
    """Mutating the returned list must not affect the store."""
    store = InMemoryStateStore()
    rec = _make_record()
    store.add(rec)
    first = store.list()
    first.clear()
    assert len(store.list()) == 1


# ---- InMemoryStateStore.find_by_branch ----


def test_find_by_branch_existing():
    store = InMemoryStateStore()
    rec = _make_record(wt_id="x", repo_root="/repo", branch="feat")
    store.add(rec)
    found = store.find_by_branch("/repo", "feat")
    assert found is rec


def test_find_by_branch_wrong_repo_returns_none():
    store = InMemoryStateStore()
    rec = _make_record(wt_id="x", repo_root="/repo-a", branch="feat")
    store.add(rec)
    assert store.find_by_branch("/repo-b", "feat") is None


def test_find_by_branch_wrong_branch_returns_none():
    store = InMemoryStateStore()
    rec = _make_record(wt_id="x", repo_root="/repo", branch="main")
    store.add(rec)
    assert store.find_by_branch("/repo", "other") is None


def test_find_by_branch_after_remove():
    store = InMemoryStateStore()
    rec = _make_record(wt_id="x", repo_root="/repo", branch="feat")
    store.add(rec)
    store.remove("x")
    assert store.find_by_branch("/repo", "feat") is None
