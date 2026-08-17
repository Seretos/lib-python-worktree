"""Tests for ``WorktreeManager.remove(checkout_path=...)`` (ticket #88).

Addresses the "orphaned worktrees not removable" bug: ``list_repo()`` (and
thus ``environment_list``) reports an untracked linked worktree with a
synthesised id that ``remove(worktree_id=...)`` could never resolve, making
the documented "orphan worktree recovery" flow unusable for exactly the
case it exists to solve. These tests cover both the untracked (orphan)
path and the tracked path through the new ``checkout_path`` parameter, plus
the guards that must hold on every new path (primary refusal, id-mismatch,
missing-arguments).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lib_python_worktree.core._exceptions import (
    CheckoutTargetError,
    DirtyWorktreeError,
    InvalidRepoError,
    PrimaryCheckoutError,
)
from lib_python_worktree.core.checkout import untracked_id_for
from lib_python_worktree.core.manager import (
    ManagerConfig,
    WorktreeManager,
    WorktreeNotFoundError,
)
from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord
from lib_python_worktree.core.yaml_store import YamlStateStore


# ---------------------------------------------------------------------------
# Behavioural requirement 2 -- an orphaned (untracked) linked worktree is
# removable by checkout_path
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_remove_untracked_linked_worktree_by_checkout_path(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """Driving test (the ticket's regression test): an orphaned linked
    worktree -- on disk and reported by `git worktree list --porcelain`,
    but with no persisted WorktreeRecord -- can be torn down via
    remove(checkout_path=...) without needing adopt() first.

    The state store carries one unrelated tracked record throughout, so a
    before/after byte comparison of state.yaml proves the untracked removal
    never touches the store (mirrors the technique in test_list_repo.py)."""
    state_dir = tmp_path / "state"
    store = YamlStateStore(state_dir=state_dir)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=store,
        reconcile_on_init=False,
    )
    store.add(
        WorktreeRecord(
            id="unrelated-tracked",
            repo_root=git_repo.resolve().as_posix(),
            branch="feature/other",
            path="/fake/other-checkout",
            backing="worktree",
        )
    )

    listing = mgr.list_repo(str(git_repo))
    linked_entries = [e for e in listing.entries if e.record.backing == "worktree"]
    # Ticket #101: "unrelated-tracked" points at a path that was never a
    # real worktree, so it now also surfaces as a tracked, orphaned entry
    # (list_repo() no longer silently drops a persisted record with no live
    # git registration) -- that is this ticket's fix, not a regression here.
    # Narrow down to the untracked entry this test is actually about.
    assert len(linked_entries) == 2
    untracked_entries = [e for e in linked_entries if not e.tracked]
    assert len(untracked_entries) == 1
    orphan_id = untracked_entries[0].record.id
    assert orphan_id == untracked_id_for(linked_worktree)

    state_bytes_before = (state_dir / "state.yaml").read_bytes()

    removed = mgr.remove(checkout_path=str(linked_worktree), force=True)

    assert removed.status == "removed"
    assert removed.id == orphan_id
    assert not linked_worktree.exists()

    porcelain = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=git_repo, capture_output=True, text=True, check=True,
    ).stdout
    assert linked_worktree.resolve().as_posix() not in porcelain.replace("\\", "/")

    assert (state_dir / "state.yaml").read_bytes() == state_bytes_before


@pytest.mark.requires_git
def test_remove_untracked_linked_worktree_by_subdirectory_checkout_path(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """checkout_path pointing at a subdirectory of the orphan resolves to
    the containing worktree."""
    sub = linked_worktree / "sub"
    sub.mkdir()
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )

    removed = mgr.remove(checkout_path=str(sub), force=True)

    assert removed.status == "removed"
    assert removed.id == untracked_id_for(linked_worktree)
    assert not linked_worktree.exists()


@pytest.mark.requires_git
def test_remove_untracked_linked_worktree_preserves_branch(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """C5 -- the single most dangerous regression this change could
    introduce: an orphan's branch must never be deleted, even with
    force=True, because a synthesised record always has
    branch_created_by_us=False."""
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )

    mgr.remove(checkout_path=str(linked_worktree), force=True)

    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "refs/heads/feature/alpha"],
        cwd=git_repo, capture_output=True,
    )
    assert proc.returncode == 0, "orphan's branch must survive removal"


@pytest.mark.requires_git
def test_remove_untracked_linked_worktree_dirty_no_force_raises(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    (linked_worktree / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )

    with pytest.raises(DirtyWorktreeError):
        mgr.remove(checkout_path=str(linked_worktree), force=False)


@pytest.mark.requires_git
def test_remove_untracked_linked_worktree_matching_id_and_checkout_path_succeeds(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )
    orphan_id = untracked_id_for(linked_worktree)

    removed = mgr.remove(
        worktree_id=orphan_id, checkout_path=str(linked_worktree), force=True
    )

    assert removed.status == "removed"
    assert removed.id == orphan_id


@pytest.mark.requires_git
def test_remove_untracked_linked_worktree_mismatched_id_raises_id_mismatch(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """C7 for untracked targets: a mismatching (worktree_id, checkout_path)
    pair raises CheckoutTargetError(reason="id_mismatch")."""
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )

    with pytest.raises(CheckoutTargetError) as exc_info:
        mgr.remove(
            worktree_id="totally-wrong-id",
            checkout_path=str(linked_worktree),
            force=True,
        )
    assert exc_info.value.reason == "id_mismatch"
    # The worktree must survive an aborted, mismatched call.
    assert linked_worktree.exists()


@pytest.mark.requires_git
def test_remove_worktree_id_only_synthesised_untracked_id_raises_not_found(
    tmp_path: Path, git_repo: Path, linked_worktree: Path, skip_if_no_git  # noqa: ARG001
):
    """C8: remove(worktree_id=<synthesised untracked id>) -- no checkout_path
    -- still raises WorktreeNotFoundError, but the message names
    checkout_path= as the remedy."""
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )
    orphan_id = untracked_id_for(linked_worktree)

    with pytest.raises(WorktreeNotFoundError) as exc_info:
        mgr.remove(worktree_id=orphan_id)

    assert "checkout_path" in str(exc_info.value)
    assert linked_worktree.exists()


# ---------------------------------------------------------------------------
# Behavioural requirement 3 -- checkout_path works for tracked targets, and
# every guard holds
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_remove_tracked_worktree_by_checkout_path(
    manager: WorktreeManager, git_repo: Path
):
    """Driving test: a tracked worktree is removable via checkout_path
    alone. Since create() sets branch_created_by_us=True for a freshly
    created branch, the owned branch must also be deleted."""
    rec = manager.create(str(git_repo), "feature/new-tracked", base="main", fetch=False)

    removed = manager.remove(checkout_path=rec.path)

    assert removed.status == "removed"
    assert manager.state.get(rec.id) is None
    assert not Path(rec.path).exists()

    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "refs/heads/feature/new-tracked"],
        cwd=git_repo, capture_output=True,
    )
    assert proc.returncode != 0, "owned branch must be deleted"


@pytest.mark.requires_git
def test_remove_by_checkout_path_unaffected_by_orphan_entries(
    manager: WorktreeManager, git_repo: Path
):
    """Ticket #101 guard: the presence of a vanished (orphaned) tracked
    record for a different checkout must not confuse
    remove(checkout_path=...) resolution for a live, untracked linked
    worktree -- an orphan entry is always is_current=False, so
    _resolve_removal_target()'s containment-based fallback must still land
    on the live checkout, not the vanished one."""
    vanished_path = git_repo.parent / "vanished-checkout-101"
    manager.state.add(
        WorktreeRecord(
            id="vanished-101",
            repo_root=git_repo.resolve().as_posix(),
            branch="feature/vanished",
            path=vanished_path.as_posix(),
            backing="worktree",
        )
    )

    live_wt = git_repo.parent / "live-untracked-101"
    subprocess.run(
        ["git", "worktree", "add", str(live_wt), "-b", "feature/live-101"],
        cwd=git_repo, check=True, capture_output=True,
    )

    removed = manager.remove(checkout_path=str(live_wt), force=True)

    assert removed.status == "removed"
    assert not live_wt.exists()


@pytest.mark.requires_git
def test_remove_neither_argument_raises_missing(manager: WorktreeManager):
    with pytest.raises(CheckoutTargetError) as exc_info:
        manager.remove()
    assert exc_info.value.reason == "missing"


@pytest.mark.requires_git
def test_remove_checkout_path_never_started_primary_raises_primary_checkout_error(
    manager: WorktreeManager, git_repo: Path
):
    """C6, new behaviour: a never-started primary addressed by
    checkout_path now gets the correct structural refusal instead of a
    confusing WorktreeNotFoundError."""
    with pytest.raises(PrimaryCheckoutError):
        manager.remove(checkout_path=str(git_repo))


@pytest.mark.requires_git
def test_remove_checkout_path_started_primary_raises_primary_checkout_error_even_with_force(
    manager: WorktreeManager, git_repo: Path
):
    manager.start(checkout_path=str(git_repo))
    with pytest.raises(PrimaryCheckoutError):
        manager.remove(checkout_path=str(git_repo), force=True)


@pytest.mark.requires_git
def test_remove_checkout_path_outside_repo_raises_invalid_repo_error(
    manager: WorktreeManager, tmp_path: Path
):
    """C9, documented limitation: a directory that is not (or is no longer)
    inside any git repo cannot be classified; InvalidRepoError propagates
    unchanged."""
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    with pytest.raises(InvalidRepoError):
        manager.remove(checkout_path=str(outside))


@pytest.mark.requires_git
def test_remove_phantom_tracked_record_removable_by_id(
    manager: WorktreeManager, git_repo: Path, monkeypatch
):
    """A tracked-but-phantom record (record present, dir present, but git
    no longer lists it) is still removable by id -- unchanged pre-existing
    behaviour via _teardown's "is not a working tree" branch, verified here
    so a #88 regression can't quietly break it."""
    import lib_python_worktree.core.manager as manager_module

    real_run_git = manager_module._run_git
    rec = manager.create(str(git_repo), "feature/alpha")

    def _fake_run_git(args, cwd=None, **kwargs):
        if args[:2] == ["worktree", "remove"]:
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=128, stdout="",
                stderr=f"fatal: '{rec.path}' is not a working tree",
            )
        return real_run_git(args, cwd=cwd, **kwargs)

    monkeypatch.setattr(manager_module, "_run_git", _fake_run_git)

    removed = manager.remove(rec.id)

    assert removed.status == "removed"
    assert manager.state.get(rec.id) is None
    assert not Path(rec.path).exists()


@pytest.mark.requires_git
def test_remove_phantom_tracked_record_removable_by_checkout_path(
    manager: WorktreeManager, git_repo: Path, monkeypatch
):
    """Same phantom scenario as above, but addressed by checkout_path --
    _resolve_target() finds the tracked record via containment over
    self.state.list() regardless of git's live porcelain view, so this
    never needs the untracked fallback."""
    import lib_python_worktree.core.manager as manager_module

    real_run_git = manager_module._run_git
    rec = manager.create(str(git_repo), "feature/alpha")

    def _fake_run_git(args, cwd=None, **kwargs):
        if args[:2] == ["worktree", "remove"]:
            return subprocess.CompletedProcess(
                args=["git", *args], returncode=128, stdout="",
                stderr=f"fatal: '{rec.path}' is not a working tree",
            )
        return real_run_git(args, cwd=cwd, **kwargs)

    monkeypatch.setattr(manager_module, "_run_git", _fake_run_git)

    removed = manager.remove(checkout_path=rec.path)

    assert removed.status == "removed"
    assert manager.state.get(rec.id) is None
    assert not Path(rec.path).exists()
