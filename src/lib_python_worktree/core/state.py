"""Pluggable state store for worktree records.

W2 ships an in-memory ``StateStore`` behind a small interface so that W7
(persistent state) can swap in a file-backed implementation without touching
any tool-level code. The interface is intentionally minimal — only the
operations the W2 tools need are exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from .process_lifecycle import KilledProcessInfo


# Ticket #99: cap on how many survivor PIDs a persisted ``StopDetail`` may
# carry verbatim. The underlying causes of "stop_incomplete" can enumerate
# far more PIDs than are worth writing to state.yaml on every incomplete
# stop (a truncated Job Object member list alone can hold up to
# ``_JOB_MEMBER_LIST_MAX_SLOTS`` -- 4096 -- entries in
# ``process_lifecycle.py``). ``StopDetail.survivor_count`` always carries the
# true total even when ``survivor_pids`` itself is capped at this value.
_STOP_DETAIL_MAX_PIDS = 32

# Reason vocabulary for ``StopDetail.reason`` -- one tag per
# ``stop_incomplete``-setting branch in ``process_lifecycle.stop()``, listed
# here in the same if/elif precedence order those branches use.
STOP_REASON_SURVIVORS = "survivors"
STOP_REASON_TREE_TRUNCATED = "tree_truncated"
STOP_REASON_JOB_MEMBER_LIST_TRUNCATED = "job_member_list_truncated"
STOP_REASON_ORPHAN_SCAN_INCOMPLETE = "orphan_scan_incomplete"

STOP_REASONS: Tuple[str, ...] = (
    STOP_REASON_SURVIVORS,
    STOP_REASON_TREE_TRUNCATED,
    STOP_REASON_JOB_MEMBER_LIST_TRUNCATED,
    STOP_REASON_ORPHAN_SCAN_INCOMPLETE,
)


@dataclass(frozen=True)
class StopDetail:
    """Machine-readable reason a ``stop()`` call reported
    ``status="stop_incomplete"`` (ticket #99).

    Ticket #95 closed the mechanical half of #99 (Windows Job Object
    containment catches a reparented, ppid-independent grandchild). This
    dataclass closes the *reporting* half: before it existed, every one of
    ``process_lifecycle.stop()``'s four ``stop_incomplete`` branches only
    emitted a ``_logger.warning(...)`` -- nothing machine-readable reached
    the caller, so a caller (e.g. the ``agent-worktree`` MCP plugin) could
    not tell *why* a stop was incomplete or whether retrying with
    ``kill_orphans=True`` would help.

    ``reason`` is always one of :data:`STOP_REASONS`, except when loaded from
    a ``state.yaml`` written by a *future* engine version, in which case an
    unrecognised value is preserved verbatim rather than rejected (forward
    compatibility -- see ``yaml_store._stop_detail_from_dict``).

    ``message`` is the exact human-readable string also passed to
    ``_logger.warning(...)`` for the same branch, so the log line and this
    field can never drift apart.

    ``survivor_pids`` is capped at :data:`_STOP_DETAIL_MAX_PIDS`;
    ``survivor_count`` always holds the true total regardless of the cap
    (populated for ``reason="survivors"`` only -- ``0``/``()`` otherwise).

    ``truncated_at`` holds the cap that was hit -- ``_MAX_TREE_NODES`` for
    ``reason="tree_truncated"``, ``_JOB_MEMBER_LIST_MAX_SLOTS`` for
    ``reason="job_member_list_truncated"`` -- ``None`` for every other
    reason.

    ``skipped_passes`` mirrors ``_PartialList.skipped_passes`` and is only
    populated for ``reason="orphan_scan_incomplete"``.

    ``kill_orphans_may_help`` is a remediation hint: ``True`` when the
    orphan-scan pass (``kill_orphans=True``) had not yet run and might catch
    what this call missed; always ``False`` for
    ``reason="orphan_scan_incomplete"`` (that pass already ran and was
    starved -- the remediation there is a larger ``timeout``, which
    ``message`` says explicitly).

    Invariant (for any record **in the store**): ``stop_detail is not None``
    implies ``status == "stop_incomplete"``. Every call site that transitions
    a stored record's status away from ``"stop_incomplete"`` also resets
    ``stop_detail`` to ``None`` (``process_lifecycle.start``/``stop``,
    ``manager.WorktreeManager.stop``, ``yaml_store.reconcile``). A record
    already popped from the store (e.g. ``manager.remove``'s
    ``status="removed"`` return value) is exempt -- the detail survives there
    as forensic info about the stop that could not be confirmed. Sites where
    the status *stays* ``"stop_incomplete"`` (the sticky-preserve paths) must
    NOT touch this field -- an earlier call's detail remains the most recent
    honest explanation until a later call actually resolves or re-detects the
    problem.

    Bounded by construction, unlike ``WorktreeRecord.killed_pids`` (unbounded,
    transient, describes an *attempt* rather than a verdict): this dataclass
    is small enough to persist to ``state.yaml`` outright, so
    ``status="stop_incomplete"`` is never reported without also carrying a
    concrete, reloadable reason.
    """

    reason: str
    message: str
    role: Optional[str] = None
    survivor_pids: Tuple[int, ...] = ()
    survivor_count: int = 0
    truncated_at: Optional[int] = None
    skipped_passes: Tuple[str, ...] = ()
    kill_orphans_may_help: bool = False


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
    job_names: Dict[str, str] = field(default_factory=dict)
    """Windows-only (ticket #95): per-role mapping of ``role`` -> the name of
    the Job Object ``start()`` created and assigned that role's spawned
    process to. Mirrors ``pids`` -- one entry per currently-tracked role. A
    role with no Job Object (POSIX, or Job Object creation failed on
    Windows) has no key in this dict at all, never a ``None`` value. Used by
    ``stop()`` as a ppid-independent containment mechanism -- see
    ``process_lifecycle._create_job_object``'s docstring for the full
    rationale. Persisted (unlike ``killed_pids``): a record loaded by a
    *different* host process still names the job so ``stop()`` can attempt
    ``OpenJobObjectW`` on it, even though that process no longer holds the
    keeper handle itself (see ``_JOB_HANDLES``'s docstring for what this does
    and does not guarantee).

    Ticket #95 fix-cycle note: this was originally a single scalar
    ``job_name: Optional[str]`` field, unconditionally overwritten by every
    ``start()`` call regardless of role -- in a multi-role worktree, starting
    a second role silently lost the first role's job name, and ``stop()``
    for the first role would then open/terminate the *second* role's job
    instead. Changed to a per-role dict, consistent with ``pids``, to close
    that cross-role bleed."""

    stop_detail: Optional[StopDetail] = None
    """Ticket #99: machine-readable reason the most recent ``stop()`` call
    reported ``status="stop_incomplete"``, or ``None``. See
    :class:`StopDetail`'s own docstring for the full invariant and clearing
    semantics -- in short: present only while ``status == "stop_incomplete"``
    for a record still in the store, cleared by every call site that resolves
    or supersedes that status, and left untouched by every call site that
    merely re-confirms it (sticky-preserve). Persisted through
    ``state.yaml`` (unlike the transient ``killed_pids``) so the reason
    survives a host restart -- a legacy record with no ``stop_detail`` key
    deserialises to ``None``."""


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
    "StopDetail",
    "STOP_REASONS",
    "WorktreeRecord",
)
