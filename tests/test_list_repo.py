"""Tests for the repo-scoped listing (ticket #84, R10):
``WorktreeManager.list_repo()`` and the pure ``checkout.list_repo()`` it
wraps.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lib_python_worktree.core._exceptions import InvalidRepoError
from lib_python_worktree.core.checkout import list_repo as _list_repo
from lib_python_worktree.core.checkout import primary_id_for
from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord
from lib_python_worktree.core.yaml_store import YamlStateStore


# ---------------------------------------------------------------------------
# R10 -- list_repo() returns the repo-scoped listing from any path inside
# the repo
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
@pytest.mark.parametrize(
    "path_fn",
    [
        lambda git_repo, linked_wt: git_repo,
        lambda git_repo, linked_wt: git_repo / "sub",
        lambda git_repo, linked_wt: linked_wt,
    ],
    ids=["main-root", "main-subdir", "linked-wt"],
)
def test_list_repo_marks_containing_entry(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, path_fn, skip_if_no_git  # noqa: ARG001
):
    """R10 driving test: list_repo() from any path inside the repo returns
    the same entry set, with exactly one entry (the containing one) marked
    is_current -- and it never writes to state.yaml."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )
    rec = WorktreeRecord(
        id="tracked-linked-r10",
        repo_root=git_repo.resolve().as_posix(),
        branch="feature/alpha",
        path=linked_worktree.resolve().as_posix(),
        backing="worktree",
    )
    store.add(rec)

    state_bytes_before = (state_dir / "state.yaml").read_bytes()

    query_path = path_fn(git_repo, linked_worktree)
    if not query_path.exists():
        query_path.mkdir()

    listing = mgr.list_repo(str(query_path))

    assert listing.repo_root == git_repo.resolve().as_posix()
    ids = {e.record.id for e in listing.entries}
    assert ids == {primary_id_for(git_repo), "tracked-linked-r10"}

    current = [e for e in listing.entries if e.is_current]
    assert len(current) == 1

    # Primary entry is synthesised, untracked; linked entry is tracked.
    primary_entries = [e for e in listing.entries if e.record.backing == "primary"]
    assert len(primary_entries) == 1
    assert primary_entries[0].tracked is False
    assert primary_entries[0].record.id == primary_id_for(git_repo)

    linked_entries = [e for e in listing.entries if e.record.backing == "worktree"]
    assert len(linked_entries) == 1
    assert linked_entries[0].tracked is True

    # Entries are primary-first.
    assert listing.entries[0].record.backing == "primary"

    assert (state_dir / "state.yaml").read_bytes() == state_bytes_before


@pytest.mark.requires_git
def test_list_repo_zero_linked_worktrees_single_primary_entry(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    store = YamlStateStore(state_dir=tmp_path / "state")
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )
    listing = mgr.list_repo(str(git_repo))
    assert len(listing.entries) == 1
    assert listing.entries[0].record.backing == "primary"
    assert listing.entries[0].is_current is True


@pytest.mark.requires_git
def test_list_repo_excludes_worktrees_of_a_different_repo(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    other_repo = tmp_path / "other-repo"
    other_repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=other_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=other_repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=other_repo, check=True, capture_output=True)
    (other_repo / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=other_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=other_repo, check=True, capture_output=True)

    store = YamlStateStore(state_dir=tmp_path / "state")
    store.add(WorktreeRecord(
        id="other-repo-record",
        repo_root=other_repo.resolve().as_posix(),
        branch="main",
        path=other_repo.resolve().as_posix(),
        backing="primary",
    ))
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    listing = mgr.list_repo(str(git_repo))
    ids = {e.record.id for e in listing.entries}
    assert "other-repo-record" not in ids


@pytest.mark.requires_git
def test_list_repo_after_start_primary_is_tracked_with_same_id(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    store = YamlStateStore(state_dir=tmp_path / "state")
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )
    before_id = mgr.list_repo(str(git_repo)).entries[0].record.id

    started = mgr.start(checkout_path=str(git_repo))

    after_listing = mgr.list_repo(str(git_repo))
    primary_entry = next(e for e in after_listing.entries if e.record.backing == "primary")
    assert primary_entry.record.id == before_id == started.id
    assert primary_entry.tracked is True


def test_manager_list_still_flat_store_wide(tmp_path: Path):
    """WorktreeManager.list() stays unchanged: flat, store-wide, no
    classification."""
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )
    rec = WorktreeRecord(
        id="flat-1", repo_root="/fake/repo", branch="main", path="/fake/repo/wt",
    )
    mgr.state.add(rec)
    assert mgr.list() == [rec]


# ---------------------------------------------------------------------------
# Fix cycle 2, blocking finding 1 -- an untracked linked worktree must
# appear in the listing (tracked=False), not be silently dropped.
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_list_repo_untracked_linked_worktree_appears_tracked_false(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Driving test: a linked worktree that is on disk (per `git worktree
    list --porcelain`) but has no persisted WorktreeRecord must still appear
    in list_repo()'s entries, marked tracked=False -- not be dropped."""
    # No records at all passed in: both the primary and the linked worktree
    # are untracked.
    listing = _list_repo(git_repo, [])

    linked_entries = [e for e in listing.entries if e.record.backing == "worktree"]
    assert len(linked_entries) == 1
    assert linked_entries[0].tracked is False
    # No deterministic id exists for a linked worktree -- pinned contract:
    # the empty string, never mistaken for a real tracked id.
    assert linked_entries[0].record.id == ""
    assert linked_entries[0].record.path == linked_worktree.resolve().as_posix()
    assert linked_entries[0].record.branch == "feature/alpha"
    assert linked_entries[0].record.repo_root == git_repo.resolve().as_posix()


@pytest.mark.requires_git
def test_list_repo_tracked_and_untracked_linked_worktrees_mixed(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """When one linked worktree is tracked (has a record) and another is not,
    list_repo() must report exactly the right tracked/untracked split -- the
    untracked one must not be dropped just because a tracked one exists."""
    other_wt = tmp_path / "other-linked-wt"
    subprocess.run(
        ["git", "worktree", "add", str(other_wt), "-b", "feature/other"],
        cwd=git_repo, check=True, capture_output=True,
    )
    try:
        records = [
            WorktreeRecord(
                id="tracked-other",
                repo_root=git_repo.resolve().as_posix(),
                branch="feature/other",
                path=other_wt.resolve().as_posix(),
                backing="worktree",
            )
        ]
        listing = _list_repo(git_repo, records)

        linked_entries = {
            e.record.path: e for e in listing.entries if e.record.backing == "worktree"
        }
        assert len(linked_entries) == 2

        tracked_entry = linked_entries[other_wt.resolve().as_posix()]
        assert tracked_entry.tracked is True
        assert tracked_entry.record.id == "tracked-other"

        untracked_entry = linked_entries[linked_worktree.resolve().as_posix()]
        assert untracked_entry.tracked is False
        assert untracked_entry.record.id == ""
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(other_wt)],
            cwd=git_repo, capture_output=True,
        )


# ---------------------------------------------------------------------------
# Fix cycle 2, blocking finding 2 -- a `git worktree list --porcelain`
# failure must raise, not silently report zero environments.
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_list_repo_git_worktree_list_failure_raises_invalid_repo_error(
    git_repo: Path, monkeypatch
):
    """Driving test: when the `git worktree list --porcelain` call itself
    fails (non-zero returncode), list_repo() must raise InvalidRepoError
    rather than defaulting to an empty listing."""
    import lib_python_worktree.core.checkout as checkout_module
    from lib_python_worktree.core._git_utils import _run_git as real_run_git

    def _flaky_run_git(args, cwd=None, **kwargs):
        if args[:2] == ["worktree", "list"]:
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=128, stdout="", stderr="fatal: boom"
            )
        return real_run_git(args, cwd=cwd, **kwargs)

    monkeypatch.setattr(checkout_module, "_run_git", _flaky_run_git)

    with pytest.raises(InvalidRepoError):
        _list_repo(git_repo, [])


# ---------------------------------------------------------------------------
# Pure function unit tests: checkout.list_repo(path, records) -- no manager
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_pure_list_repo_function_no_manager(git_repo: Path):
    records = [
        WorktreeRecord(
            id="hand-built-primary",
            repo_root=git_repo.resolve().as_posix(),
            branch=None,
            path=git_repo.resolve().as_posix(),
            backing="primary",
        )
    ]
    listing = _list_repo(git_repo, records)
    assert listing.repo_root == git_repo.resolve().as_posix()
    assert len(listing.entries) == 1
    assert listing.entries[0].tracked is True
    assert listing.entries[0].record.id == "hand-built-primary"
