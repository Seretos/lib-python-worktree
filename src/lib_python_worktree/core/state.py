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

# Reason vocabulary for ``ShadowedContract.reason`` (ticket #100) -- one tag
# per branch in ``manager._detect_shadowed_contract``.
SHADOW_REASON_DIFFERS = "differs"
SHADOW_REASON_UNREADABLE = "unreadable"

SHADOW_REASONS: Tuple[str, ...] = (
    SHADOW_REASON_DIFFERS,
    SHADOW_REASON_UNREADABLE,
)


# Status vocabulary for ``SetupOutcome.status`` (ticket #105) -- one tag per
# verdict ``WorktreeManager.create()`` reaches for the contract ``setup:``
# hook: it ran and every step succeeded, it ran and a step failed, or there
# were no ``setup:`` steps to run at all.
SETUP_STATUS_COMPLETED = "completed"
SETUP_STATUS_FAILED = "failed"
SETUP_STATUS_SKIPPED = "skipped"

SETUP_STATUSES: Tuple[str, ...] = (
    SETUP_STATUS_COMPLETED,
    SETUP_STATUS_FAILED,
    SETUP_STATUS_SKIPPED,
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


@dataclass(frozen=True)
class ShadowedContract:
    """A checkout-local contract copy that ``start()`` did NOT read (ticket
    #100).

    ``start()`` always reads the live contract from
    ``<repo_root>/.seretos/worktree-setup.yml`` -- never from a linked
    worktree's own checkout-local copy (the ``agent-worktree`` plugin's
    convenience copy dropped into every new checkout, see ``manager.create``'s
    docstring). When that checkout-local copy exists and parses to a
    *different* ``WorktreeContract`` than the one actually used, an agent that
    edited only the checkout-local file gets a clean-looking no-op
    ``status="ready"`` with no hint that its edit was never read. This
    dataclass is the machine-readable flag surfaced on the ``WorktreeRecord``
    :func:`~.manager.WorktreeManager.start` returns, so a caller (e.g. the
    ``environment_start`` MCP tool) can warn the agent explicitly.

    ``path`` is the checkout-local contract file that was found but not read;
    ``used_path`` is the repo-root file actually read (both forward-slash
    strings). ``reason`` is always one of :data:`SHADOW_REASONS`:

    - ``"differs"``: the checkout-local copy parses to a contract whose model
      fields differ from the one that was used.
    - ``"unreadable"``: the checkout-local copy exists but could not be
      parsed/validated (``ContractError``/``ContractValidationError``/
      ``OSError``) -- ``start()`` itself never raises for this; only the used
      (repo-root) contract's own load failures propagate.

    ``message`` is the exact human-readable string also passed to
    ``_logger.warning(...)`` for the same detection, so the log line and this
    field can never drift apart (mirrors :class:`StopDetail`'s message-parity
    convention).

    Deliberately **not** persisted to ``state.yaml`` -- see
    ``WorktreeRecord.shadowed_contract``'s docstring for why.
    """

    path: str
    used_path: str
    reason: str
    message: str


@dataclass(frozen=True)
class SetupOutcome:
    """Machine-readable verdict of the contract ``setup:`` hook run by
    ``WorktreeManager.create()`` (ticket #105).

    Orthogonal to the overloaded ``WorktreeRecord.status`` field, mirroring
    the relationship ``StopDetail`` has to ``status``. ``record.status`` is
    continuously rewritten by ``create``/``start``/``stop``/``reconcile`` for
    entirely different purposes (e.g. ``"created"``, ``"running"``,
    ``"stopped"``, ``"setup_failed"``), so it cannot answer "did the
    ``setup:`` hook itself ever run, and how did it end?" once later calls
    have moved ``status`` on. ``setup_outcome`` answers exactly that question
    and only that question -- it is written once, by ``create()``'s
    ``setup:`` hook block, and never touched again by any other call site.

    ``status`` is always one of :data:`SETUP_STATUSES`, except when loaded
    from a ``state.yaml`` written by a *future* engine version, in which case
    an unrecognised value is preserved verbatim rather than rejected (forward
    compatibility -- see ``yaml_store._setup_outcome_from_dict``, mirroring
    :class:`StopDetail`'s convention):

    - ``"completed"``: the contract had ``setup:`` steps and every one of
      them succeeded.
    - ``"failed"``: the contract had ``setup:`` steps and one of them raised
      (a :class:`~..setup.runner.SetupFailedError` or any other exception).
    - ``"skipped"``: the contract had no ``setup:`` steps at all (missing
      contract, empty contract, or an explicit ``setup: []``) -- ``create()``
      still ran and reached this decision, it just had nothing to execute.

    ``message`` is the exact human-readable string describing the outcome --
    for ``"failed"`` this is ``str()`` of the exception that was raised,
    mirroring :class:`StopDetail`'s message-parity convention.

    ``completed_at`` is an ISO-8601 UTC timestamp (via
    ``datetime.now(timezone.utc).isoformat()``), set for all three statuses.

    ``steps_run`` is the number of ``setup:`` steps that ran -- the length of
    the underlying ``SetupResult.steps`` on a ``"completed"`` run; always
    ``0`` for ``"skipped"``.

    ``failed_step_index``, ``failed_step_name``, ``log_path`` (a forward-slash
    string, like other path fields on :class:`WorktreeRecord`), ``returncode``,
    and ``timed_out`` are populated for ``"failed"`` only (and only when the
    failure was a :class:`~..setup.runner.SetupFailedError`, which is the only
    exception type that carries this detail) -- left at their defaults
    otherwise. ``timed_out`` is ``True`` when the failing step was killed by
    its own timeout rather than exiting non-zero.

    Deliberately **distinct** from ``None``: ``record.setup_outcome is None``
    means the ``setup:`` hook was never reached at all (a record predating
    this field, an adopted record, or a ``list_repo()``-synthesised entry),
    whereas ``status="skipped"`` means ``create()`` ran and positively
    determined there was nothing to run. Persisted through ``state.yaml``
    (mirrors ``stop_detail``, unlike the transient ``killed_pids``/
    ``shadowed_contract``) -- a legacy record with no ``setup_outcome`` key
    deserialises to ``None``.

    Invariant: only the ``setup:`` hook block in ``WorktreeManager.create()``
    ever assigns ``WorktreeRecord.setup_outcome``. ``start()``, ``stop()``,
    ``reconcile()``, ``adopt()``, and ``remove()`` never read or write it.
    """

    status: str
    message: str = ""
    completed_at: Optional[str] = None
    steps_run: int = 0
    failed_step_index: Optional[int] = None
    failed_step_name: Optional[str] = None
    log_path: Optional[str] = None
    returncode: Optional[int] = None
    timed_out: bool = False


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
    """Ticket #95: PIDs ``start()`` killed and retried past, describing an
    *attempt* rather than a verdict. Deliberately transient, like
    ``shadowed_contract``: never persisted through ``state.yaml`` (unlike
    ``stop_detail``). For an ``InMemoryStateStore``-backed manager this field
    is subject to the same store-reference caveat as ``shadowed_contract`` --
    ``InMemoryStateStore.update()`` stores the record by reference (no copy),
    so a value set by one call can keep appearing on later ``get()``/
    ``list()`` calls until the next call that assigns it overwrites it."""
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

    variants: Dict[str, str] = field(default_factory=dict)
    """Ticket #104: per-role mapping of ``role`` -> the ``variant`` name that
    ``start()`` was called with for that role (i.e. which named ``start:``
    contract step is currently running under that role). Mirrors ``pids`` and
    ``job_names`` exactly: one entry per currently-tracked role, and a role
    whose variant is unknown/unset has no key in this dict at all, never a
    ``None`` value. The invariant to maintain is
    ``set(record.variants) <= set(record.pids)`` -- an entry exists only
    while that role has a recorded pid.

    Direction is role -> variant (not the reverse) because two distinct
    roles can be started from the same variant, and a reverse mapping could
    not represent that without silently dropping one.

    Used by ``WorktreeManager.stop(variant=...)`` to resolve which ``role``
    to stop without the caller having to track the role<->variant pairing
    itself. Persisted through ``state.yaml`` (unlike the transient
    ``killed_pids``/``shadowed_contract``) so the mapping survives a host
    restart -- a legacy record with no ``variants`` key deserialises to
    ``{}``.
    """

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

    shadowed_contract: Optional[ShadowedContract] = None
    """Ticket #100: set by ``start()`` when a checkout-local
    ``.seretos/worktree-setup.yml`` copy exists and would differ from (or
    fails to parse against) the repo-root contract that was actually read.
    See :class:`ShadowedContract`'s own docstring for the full rationale.

    Deliberately **transient**, like ``killed_pids`` and unlike the persisted
    ``stop_detail``: this is a live observation recomputed on every
    ``start()`` call, not a stored verdict -- a persisted copy would go stale
    the moment the agent fixes or deletes the shadow file. ``_record_to_dict``
    in ``yaml_store.py`` never serialises this field, so a record round-
    tripped through ``YamlStateStore`` always comes back with
    ``shadowed_contract is None``, which is correct: nothing here needs a
    key in ``state.yaml`` for old records to remain compatible with.

    The transient guarantee above is delivered specifically by
    ``YamlStateStore``'s serialise/deserialise round-trip, not by this field
    itself. ``InMemoryStateStore.update()`` stores the record object by
    reference (no copy), so for an ``InMemoryStateStore``-backed manager a
    ``shadowed_contract`` set by one ``start()`` call can keep appearing on
    later ``get()``/``list()`` calls -- even after the checkout-local
    contract is fixed or deleted -- until the next ``start()`` recomputes it.
    This is an accepted limitation of the in-memory store, not a property of
    this field, and it applies equally to ``killed_pids`` (see that field's
    own docstring)."""

    setup_outcome: Optional[SetupOutcome] = None
    """Ticket #105: machine-readable verdict of the contract ``setup:`` hook
    run by ``WorktreeManager.create()``, or ``None``. See
    :class:`SetupOutcome`'s own docstring for the full invariant -- in short:
    written ONLY by the ``setup:`` hook block in ``create()``; orthogonal to
    ``status`` (which ``start``/``stop``/``reconcile`` continuously rewrite
    for unrelated reasons); never touched by ``start``/``stop``/
    ``reconcile``/``adopt``/``remove``. Persisted through ``state.yaml``
    (mirrors ``stop_detail``) -- a legacy record with no ``setup_outcome``
    key deserialises to ``None``. ``None`` deliberately means "the ``setup:``
    hook was never reached" (record predates this field, was ``adopt()``-ed,
    or was synthesised by ``list_repo()``) -- distinct from
    ``status="skipped"``, which means ``create()`` ran and found no
    ``setup:`` steps to run."""


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
    "SetupOutcome",
    "SETUP_STATUSES",
    "SETUP_STATUS_COMPLETED",
    "SETUP_STATUS_FAILED",
    "SETUP_STATUS_SKIPPED",
    "ShadowedContract",
    "SHADOW_REASONS",
    "StateStore",
    "StopDetail",
    "STOP_REASONS",
    "WorktreeRecord",
)
