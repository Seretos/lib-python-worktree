"""Pluggable state store for worktree records.

W2 ships an in-memory ``StateStore`` behind a small interface so that W7
(persistent state) can swap in a file-backed implementation without touching
any tool-level code. The interface is intentionally minimal — only the
operations the W2 tools need are exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Protocol

if TYPE_CHECKING:
    from .process_lifecycle import KilledProcessInfo


@dataclass
class WorktreeRecord:
    """A single tracked worktree (or, since ticket #84, a primary checkout).

    Fields ``ports``, ``pids``, and ``status`` exist for forward compatibility
    with W4/W5/W6 and are populated by later phases. W2 leaves them at their
    defaults.

    ``backing`` (ticket #84) is the checkout *substrate* -- ``"worktree"``
    (default, a linked ``git worktree`` checkout) or ``"primary"`` (the
    repo's main clone, addressed via ``checkout_path`` rather than created by
    ``create()``). It is explicitly orthogonal to ``branch_created_by_us``:
    the latter tracks whether *this manager* created the branch, the former
    tracks what kind of checkout the record describes. ``branch`` is
    ``Optional`` because a primary record's branch is never stored -- it is
    read live via ``_effective_branch`` so it can never go stale.
    """

    id: str
    repo_root: str
    branch: Optional[str]
    path: str
    status: str = "created"
    ports: Dict[str, int] = field(default_factory=dict)
    pids: Dict[str, int] = field(default_factory=dict)
    branch_created_by_us: bool = False
    killed_pids: List["KilledProcessInfo"] = field(default_factory=list)
    returncode: Optional[int] = None
    start_log_path: Optional[str] = None
    backing: str = "worktree"


class StateStore(Protocol):
    """Interface that W7 will re-implement against a persistent backing store."""

    def add(self, record: WorktreeRecord) -> None: ...

    def get(self, worktree_id: str) -> Optional[WorktreeRecord]: ...

    def remove(self, worktree_id: str) -> Optional[WorktreeRecord]: ...

    def list(self) -> List[WorktreeRecord]: ...

    def find_by_branch(
        self, repo_root: str, branch: str
    ) -> Optional[WorktreeRecord]: ...

    def update(self, record: WorktreeRecord) -> None: ...


class InMemoryStateStore:
    """Phase-1 in-memory store. Swapped out by W7."""

    def __init__(self) -> None:
        self._records: Dict[str, WorktreeRecord] = {}

    def add(self, record: WorktreeRecord) -> None:
        if record.id in self._records:
            raise ValueError(f"Worktree id already tracked: {record.id}")
        self._records[record.id] = record

    def get(self, worktree_id: str) -> Optional[WorktreeRecord]:
        return self._records.get(worktree_id)

    def remove(self, worktree_id: str) -> Optional[WorktreeRecord]:
        return self._records.pop(worktree_id, None)

    def list(self) -> List[WorktreeRecord]:
        return list(self._records.values())

    def find_by_branch(
        self, repo_root: str, branch: str
    ) -> Optional[WorktreeRecord]:
        for rec in self._records.values():
            if rec.backing == "primary":
                # A primary record's branch is never stored (read live via
                # _effective_branch) and must never shadow a create()
                # duplicate-branch check.
                continue
            if rec.repo_root == repo_root and rec.branch == branch:
                return rec
        return None

    def update(self, record: WorktreeRecord) -> None:
        if record.id not in self._records:
            raise KeyError(f"Worktree id not tracked: {record.id}")
        self._records[record.id] = record


__all__: Iterable[str] = (
    "InMemoryStateStore",
    "StateStore",
    "WorktreeRecord",
)
