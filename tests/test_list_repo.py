"""Tests for the repo-scoped listing (ticket #84, R10):
``WorktreeManager.list_repo()`` and the pure ``checkout.list_repo()`` it
wraps.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

import pytest

from lib_python_worktree.core._exceptions import InvalidRepoError
from lib_python_worktree.core.checkout import EnvironmentEntry
from lib_python_worktree.core.checkout import list_repo as _list_repo
from lib_python_worktree.core.checkout import primary_id_for, untracked_id_for
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
    # Ticket #88: a deterministic, location-hashed id -- not "" -- so the
    # entry environment_list displays is addressable via
    # remove(checkout_path=...).
    assert linked_entries[0].record.id == untracked_id_for(linked_worktree)
    assert linked_entries[0].record.id
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
        assert untracked_entry.record.id == untracked_id_for(linked_worktree)
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


# ---------------------------------------------------------------------------
# Ticket #85 -- tracked-entry branch freshness
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_list_repo_tracked_linked_worktree_branch_reflects_live_checkout(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Driving test: a tracked linked worktree's branch reflects git's live
    view (per `git worktree list --porcelain`), not the value last persisted
    to state.yaml -- e.g. after a manual `git checkout` inside the
    worktree."""
    records = [
        WorktreeRecord(
            id="tracked-linked-85",
            repo_root=git_repo.resolve().as_posix(),
            branch="feature/alpha",
            path=linked_worktree.resolve().as_posix(),
            backing="worktree",
        )
    ]
    subprocess.run(
        ["git", "checkout", "-b", "feature/switched"],
        cwd=linked_worktree, check=True, capture_output=True,
    )

    listing = _list_repo(git_repo, records)

    linked_entry = next(e for e in listing.entries if e.record.backing == "worktree")
    assert linked_entry.tracked is True
    assert linked_entry.record.branch == "feature/switched"


@pytest.mark.requires_git
def test_list_repo_tracked_linked_worktree_branch_unchanged_when_in_sync(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Edge case: when the stored branch already matches git's live view,
    list_repo() must not needlessly copy the record -- entry.record stays
    identical (by identity) to the record object passed in."""
    records = [
        WorktreeRecord(
            id="tracked-linked-85-sync",
            repo_root=git_repo.resolve().as_posix(),
            branch="feature/alpha",
            path=linked_worktree.resolve().as_posix(),
            backing="worktree",
        )
    ]

    listing = _list_repo(git_repo, records)

    linked_entry = next(e for e in listing.entries if e.record.backing == "worktree")
    assert linked_entry.record.branch == "feature/alpha"
    assert linked_entry.record is records[0]


@pytest.mark.requires_git
def test_list_repo_branch_refresh_does_not_mutate_caller_record_or_store(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Driving test: the in-memory branch refresh must never be persisted --
    state.yaml stays byte-identical and a fresh store.list() still reports
    the stale stored branch; only the returned RepoListing carries the live
    value. Against the pure function directly, the caller's own record
    object must also be left untouched (dataclasses.replace(), never
    in-place mutation)."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )
    rec = WorktreeRecord(
        id="tracked-linked-85-persist",
        repo_root=git_repo.resolve().as_posix(),
        branch="feature/alpha",
        path=linked_worktree.resolve().as_posix(),
        backing="worktree",
    )
    store.add(rec)

    state_bytes_before = (state_dir / "state.yaml").read_bytes()

    subprocess.run(
        ["git", "checkout", "-b", "feature/switched"],
        cwd=linked_worktree, check=True, capture_output=True,
    )

    listing = mgr.list_repo(str(git_repo))
    linked_entry = next(e for e in listing.entries if e.record.backing == "worktree")
    assert linked_entry.record.branch == "feature/switched"

    assert (state_dir / "state.yaml").read_bytes() == state_bytes_before
    reloaded = next(r for r in store.list() if r.id == rec.id)
    assert reloaded.branch == "feature/alpha"

    # Against the pure function with a plain records list: the caller's own
    # record object is untouched, and the returned entry is a different
    # object (dataclasses.replace(), not in-place mutation).
    plain_records = [
        WorktreeRecord(
            id="tracked-linked-85-persist",
            repo_root=git_repo.resolve().as_posix(),
            branch="feature/alpha",
            path=linked_worktree.resolve().as_posix(),
            backing="worktree",
        )
    ]
    pure_listing = _list_repo(git_repo, plain_records)
    pure_entry = next(e for e in pure_listing.entries if e.record.backing == "worktree")
    assert plain_records[0].branch == "feature/alpha"
    assert pure_entry.record is not plain_records[0]


@pytest.mark.requires_git
def test_list_repo_branch_refresh_does_not_mutate_in_memory_store_record(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Edge case: InMemoryStateStore.list() hands out its live record
    objects, not copies -- list_repo() must use dataclasses.replace(), never
    in-place mutation, or this would silently corrupt the in-memory store."""
    store = InMemoryStateStore()
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )
    rec = WorktreeRecord(
        id="tracked-linked-85-inmem",
        repo_root=git_repo.resolve().as_posix(),
        branch="feature/alpha",
        path=linked_worktree.resolve().as_posix(),
        backing="worktree",
    )
    store.add(rec)

    subprocess.run(
        ["git", "checkout", "-b", "feature/switched"],
        cwd=linked_worktree, check=True, capture_output=True,
    )

    listing = mgr.list_repo(str(git_repo))
    linked_entry = next(e for e in listing.entries if e.record.backing == "worktree")
    assert linked_entry.record.branch == "feature/switched"

    # The store's own record object must be untouched.
    assert store.get("tracked-linked-85-inmem").branch == "feature/alpha"
    assert rec.branch == "feature/alpha"


@pytest.mark.requires_git
def test_list_repo_tracked_linked_worktree_detached_head_keeps_stored_branch(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Guard test: a detached-HEAD linked worktree's porcelain block has no
    `branch` line, so the refresh must keep the stored (last-known) branch
    rather than blanking it to None."""
    records = [
        WorktreeRecord(
            id="tracked-linked-85-detached",
            repo_root=git_repo.resolve().as_posix(),
            branch="feature/alpha",
            path=linked_worktree.resolve().as_posix(),
            backing="worktree",
        )
    ]
    subprocess.run(
        ["git", "checkout", "--detach"],
        cwd=linked_worktree, check=True, capture_output=True,
    )

    listing = _list_repo(git_repo, records)

    linked_entry = next(e for e in listing.entries if e.record.backing == "worktree")
    assert linked_entry.tracked is True
    assert linked_entry.record.branch == "feature/alpha"
    assert linked_entry.record.branch is not None


@pytest.mark.requires_git
def test_list_repo_tracked_primary_branch_stays_none_after_start(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    """Guard test: a tracked primary entry's branch must stay None (#84
    decision) even though its porcelain block reports the live branch (e.g.
    main) -- the refresh applies only to linked worktrees, never the
    primary."""
    store = YamlStateStore(state_dir=tmp_path / "state")
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )
    mgr.start(checkout_path=str(git_repo))

    listing = mgr.list_repo(str(git_repo))
    primary_entry = next(e for e in listing.entries if e.record.backing == "primary")
    assert primary_entry.tracked is True
    assert primary_entry.record.branch is None


# ---------------------------------------------------------------------------
# Ticket #101 -- a worktree deregistered outside the MCP must stay
# discoverable as a tracked, orphaned entry
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_list_repo_deregistered_tracked_worktree_surfaces_as_orphan_entry(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Ticket #101 driving test (behavioural requirement 1): a worktree
    created through the MCP and then deregistered directly in the shell
    (`git worktree remove --force <path>`, bypassing this tool) must stay
    visible in list_repo() -- not vanish -- as a tracked, orphaned entry."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    rec = WorktreeRecord(
        id="tracked-linked-101",
        repo_root=git_repo.resolve().as_posix(),
        branch="feature/alpha",
        path=linked_worktree.resolve().as_posix(),
        backing="worktree",
    )
    store.add(rec)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    subprocess.run(
        ["git", "worktree", "remove", "--force", str(linked_worktree)],
        cwd=git_repo, check=True, capture_output=True,
    )

    listing = mgr.list_repo(str(git_repo))

    assert len(listing.entries) == 2
    orphan = next(e for e in listing.entries if e.record.id == "tracked-linked-101")
    assert orphan.tracked is True
    assert orphan.record.status == "orphaned"
    assert orphan.record.path == linked_worktree.resolve().as_posix()
    assert orphan.record.branch == "feature/alpha"
    assert orphan.is_current is False


@pytest.mark.requires_git
def test_list_repo_prunable_block_with_tracked_record_surfaces_once(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Edge case (also currently RED): deleting the worktree directory
    directly (leaving a `prunable` porcelain block rather than a fully
    deregistered one) must not create a duplicate entry -- exactly one
    orphaned entry for that record, not two."""
    records = [
        WorktreeRecord(
            id="tracked-linked-101-prunable",
            repo_root=git_repo.resolve().as_posix(),
            branch="feature/alpha",
            path=linked_worktree.resolve().as_posix(),
            backing="worktree",
        )
    ]
    shutil.rmtree(linked_worktree, ignore_errors=True)

    listing = _list_repo(git_repo, records)

    matches = [e for e in listing.entries if e.record.id == "tracked-linked-101-prunable"]
    assert len(matches) == 1
    assert matches[0].tracked is True
    assert matches[0].record.status == "orphaned"


@pytest.mark.requires_git
def test_list_repo_deregistered_worktree_with_directory_still_present_is_orphaned(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Ticket #101 blocking-finding regression test: a worktree whose git
    registration is gone but whose checkout *directory is still present on
    disk* must still surface as `tracked=True` with `record.status ==
    "orphaned"` -- not only when the directory itself was deleted.

    Deregisters by deleting the worktree's `.git/worktrees/<id>`
    administrative directory directly and running `git worktree prune`,
    rather than `git worktree remove --force` (which would delete the
    checkout directory too) or `shutil.rmtree` (which produces a
    `prunable` block, not a fully-vanished one) -- so the directory
    genuinely stays on disk while the porcelain block disappears.

    Before the fix, the second-pass override in `list_repo()` only applied
    `status="orphaned"` when `Path(rec.path).exists()` was False, so this
    case kept whatever status was last persisted (typically "created") and
    was invisible to the documented recovery predicate
    `e.tracked and e.record.status == "orphaned"`.
    """
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    rec = WorktreeRecord(
        id="tracked-linked-101-dir-present",
        repo_root=git_repo.resolve().as_posix(),
        branch="feature/alpha",
        path=linked_worktree.resolve().as_posix(),
        backing="worktree",
        status="created",
    )
    store.add(rec)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    admin_dir = git_repo / ".git" / "worktrees" / linked_worktree.name
    assert admin_dir.is_dir()
    shutil.rmtree(admin_dir)
    subprocess.run(
        ["git", "worktree", "prune", "-v"],
        cwd=git_repo, check=True, capture_output=True,
    )

    # Sanity: the checkout directory is still on disk, but git's porcelain
    # view no longer has a block for it at all (not even `prunable`).
    assert linked_worktree.exists()
    proc = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=git_repo, check=True, capture_output=True, text=True,
    )
    assert linked_worktree.name not in proc.stdout

    listing = mgr.list_repo(str(git_repo))

    orphan = next(
        e for e in listing.entries if e.record.id == "tracked-linked-101-dir-present"
    )
    assert orphan.tracked is True
    assert orphan.record.status == "orphaned"

    # Selected by the README's documented recovery predicate.
    selected = [e for e in listing.entries if e.tracked and e.record.status == "orphaned"]
    assert orphan in selected

    # The recovery flow's remove() step must also succeed for this
    # deregistration path, where the checkout directory is still present.
    # `git worktree remove` fails with "is not a working tree" (exit 128)
    # because the admin dir is gone; `_phantom_state_cleanup()` (ticket #51)
    # handles that by rmtree-ing the leftover directory, pruning stale git
    # metadata, then releasing ports and deleting the state record.
    removed = mgr.remove(orphan.record.id, force=True)
    assert removed.status == "removed"
    assert not linked_worktree.exists()
    assert mgr.state.get("tracked-linked-101-dir-present") is None


@pytest.mark.requires_git
def test_orphan_recovery_flow_end_to_end(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    """Ticket #101 driving test (behavioural requirement 2, the e2e-test
    label): the documented orphan-recovery flow -- list_repo(), select the
    entry via the README's D3 predicate (`e.tracked and e.record.status ==
    "orphaned"`), then remove(id, force=True) -- works end to end for a
    worktree deregistered outside the tool, across a *fresh* WorktreeManager
    construction (reconcile_on_init=True, the default) so reconcile() has
    already flipped the persisted status to "orphaned" before list_repo()
    ever runs."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
    )
    rec = mgr.create(str(git_repo), "feature/e2e-101", base="main", fetch=False)
    wt_path = Path(rec.path)

    subprocess.run(
        ["git", "worktree", "remove", "--force", str(wt_path)],
        cwd=git_repo, check=True, capture_output=True,
    )

    fresh_mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=YamlStateStore(state_dir=state_dir),
    )

    listing = fresh_mgr.list_repo(str(git_repo))
    orphan = next(
        e for e in listing.entries
        if e.tracked and e.record.status == "orphaned"
    )
    assert orphan.record.id == rec.id

    removed = fresh_mgr.remove(orphan.record.id, force=True)
    assert removed.status == "removed"

    after_listing = fresh_mgr.list_repo(str(git_repo))
    assert all(e.record.id != rec.id for e in after_listing.entries)
    assert fresh_mgr.state.get(rec.id) is None


@pytest.mark.requires_git
def test_list_repo_orphan_entries_do_not_mutate_store_or_caller_record(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Guard (mirrors the existing #85 branch-refresh purity test):
    surfacing a deregistered worktree as an in-memory "orphaned" status must
    never write back to the store, and must never mutate the caller's own
    record object."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    rec = WorktreeRecord(
        id="tracked-linked-101-purity",
        repo_root=git_repo.resolve().as_posix(),
        branch="feature/alpha",
        path=linked_worktree.resolve().as_posix(),
        backing="worktree",
    )
    store.add(rec)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    subprocess.run(
        ["git", "worktree", "remove", "--force", str(linked_worktree)],
        cwd=git_repo, check=True, capture_output=True,
    )

    state_bytes_before = (state_dir / "state.yaml").read_bytes()

    listing = mgr.list_repo(str(git_repo))
    orphan = next(e for e in listing.entries if e.record.id == "tracked-linked-101-purity")
    assert orphan.record.status == "orphaned"

    assert (state_dir / "state.yaml").read_bytes() == state_bytes_before
    reloaded = store.get("tracked-linked-101-purity")
    assert reloaded.status == "created"

    # Against InMemoryStateStore (hands out live objects, not copies): the
    # store's own record must stay untouched, and the returned entry must be
    # a different object.
    mem_store = InMemoryStateStore()
    mem_rec = WorktreeRecord(
        id="tracked-linked-101-purity-mem",
        repo_root=git_repo.resolve().as_posix(),
        branch="feature/alpha",
        path=linked_worktree.resolve().as_posix(),
        backing="worktree",
    )
    mem_store.add(mem_rec)
    pure_listing = _list_repo(git_repo, mem_store.list())
    pure_orphan = next(e for e in pure_listing.entries if e.record.id == mem_rec.id)
    assert pure_orphan.record.status == "orphaned"
    assert mem_store.get(mem_rec.id).status == "created"
    assert pure_orphan.record is not mem_rec


@pytest.mark.requires_git
def test_list_repo_orphan_status_override_without_reconcile(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """D3 guard: the in-memory status override is a read-view rule applied
    by list_repo() itself -- it must fire even when reconcile() has never
    run (reconcile_on_init=False), while the store keeps holding its
    original persisted status."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    rec = WorktreeRecord(
        id="tracked-linked-101-noreconcile",
        repo_root=git_repo.resolve().as_posix(),
        branch="feature/alpha",
        path=linked_worktree.resolve().as_posix(),
        backing="worktree",
        status="created",
    )
    store.add(rec)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )

    subprocess.run(
        ["git", "worktree", "remove", "--force", str(linked_worktree)],
        cwd=git_repo, check=True, capture_output=True,
    )

    listing = mgr.list_repo(str(git_repo))
    orphan = next(e for e in listing.entries if e.record.id == "tracked-linked-101-noreconcile")
    assert orphan.record.status == "orphaned"
    assert store.get("tracked-linked-101-noreconcile").status == "created"


@pytest.mark.requires_git
def test_list_repo_orphan_ordering_is_deterministic(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    """Two vanished records plus one live linked worktree: entries stay
    primary-first then path-sorted -- no orphan-last third dimension -- and
    two consecutive calls return an identical id sequence."""
    other_wt = tmp_path / "other-linked-wt-101"
    subprocess.run(
        ["git", "worktree", "add", str(other_wt), "-b", "feature/other-101"],
        cwd=git_repo, check=True, capture_output=True,
    )
    try:
        vanished_a = tmp_path / "vanished-a"
        vanished_b = tmp_path / "vanished-b"
        records = [
            WorktreeRecord(
                id="live-linked-101",
                repo_root=git_repo.resolve().as_posix(),
                branch="feature/other-101",
                path=other_wt.resolve().as_posix(),
                backing="worktree",
            ),
            WorktreeRecord(
                id="vanished-a-101",
                repo_root=git_repo.resolve().as_posix(),
                branch="feature/vanished-a",
                path=vanished_a.as_posix(),
                backing="worktree",
            ),
            WorktreeRecord(
                id="vanished-b-101",
                repo_root=git_repo.resolve().as_posix(),
                branch="feature/vanished-b",
                path=vanished_b.as_posix(),
                backing="worktree",
            ),
        ]

        listing_1 = _list_repo(git_repo, records)
        listing_2 = _list_repo(git_repo, records)

        ids_1 = [e.record.id for e in listing_1.entries]
        ids_2 = [e.record.id for e in listing_2.entries]
        assert ids_1 == ids_2

        assert listing_1.entries[0].record.backing == "primary"
        rest = listing_1.entries[1:]
        paths = [e.record.path for e in rest]
        assert paths == sorted(paths)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(other_wt)],
            cwd=git_repo, capture_output=True,
        )


@pytest.mark.requires_git
def test_list_repo_orphan_record_of_other_repo_not_surfaced(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    """The repo_root filter must still exclude a foreign repo's vanished
    record from the new orphan pass, same as it already does for the
    porcelain-driven first pass."""
    other_repo_root = (tmp_path / "other-repo-101").as_posix()
    vanished_path = (tmp_path / "other-repo-101" / "vanished-wt").as_posix()
    records = [
        WorktreeRecord(
            id="other-repo-vanished-101",
            repo_root=other_repo_root,
            branch="feature/x",
            path=vanished_path,
            backing="worktree",
        )
    ]

    listing = _list_repo(git_repo, records)
    ids = {e.record.id for e in listing.entries}
    assert "other-repo-vanished-101" not in ids


@pytest.mark.requires_git
def test_list_repo_stale_primary_backed_record_does_not_add_second_primary(
    tmp_path: Path, git_repo: Path, skip_if_no_git  # noqa: ARG001
):
    """A stale backing="primary" record under the same repo_root (pointing
    at a vanished path) must not add a second primary entry -- the
    single-primary invariant relied on elsewhere (`next(e for e in entries
    if e.record.backing == "primary")`) must hold."""
    stale_primary_path = (tmp_path / "stale-primary-101").as_posix()
    records = [
        WorktreeRecord(
            id="stale-primary-101",
            repo_root=git_repo.resolve().as_posix(),
            branch=None,
            path=stale_primary_path,
            backing="primary",
        )
    ]

    listing = _list_repo(git_repo, records)

    primary_entries = [e for e in listing.entries if e.record.backing == "primary"]
    assert len(primary_entries) == 1
    assert primary_entries[0].record.id != "stale-primary-101"


@pytest.mark.requires_git
def test_list_repo_healthy_worktree_still_single_entry(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Regression guard: a tracked, still-live linked worktree must not be
    emitted twice -- once by the porcelain pass, once by the new orphan
    pass."""
    records = [
        WorktreeRecord(
            id="healthy-linked-101",
            repo_root=git_repo.resolve().as_posix(),
            branch="feature/alpha",
            path=linked_worktree.resolve().as_posix(),
            backing="worktree",
        )
    ]

    listing = _list_repo(git_repo, records)

    matches = [e for e in listing.entries if e.record.id == "healthy-linked-101"]
    assert len(matches) == 1
    assert matches[0].record.status != "orphaned"


def test_list_repo_orphan_entry_has_no_new_fields():
    """D3 pin: no new EnvironmentEntry field is introduced to carry the
    orphan discriminator -- it is `record.status == "orphaned"` combined
    with `tracked is True`, nothing else."""
    field_names = {f.name for f in dataclasses.fields(EnvironmentEntry)}
    assert field_names == {"record", "is_current", "tracked"}
