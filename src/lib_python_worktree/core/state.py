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

# Outcome vocabulary for ``StopAttempt.outcome`` (ticket #110) -- distinguishes
# "nothing needed killing" from "the tracked PID had already gone stale (but
# other work from its tree/group/job was found and killed)" and from
# "process_lifecycle.stop() was never even reached" (the WorktreeManager.stop()
# no-op branch for a role with no recorded pid).
STOP_ATTEMPT_KILLED = "killed"
STOP_ATTEMPT_ALREADY_EXITED = "already_exited"
STOP_ATTEMPT_TRACKED_PID_MISSING = "tracked_pid_missing"
STOP_ATTEMPT_NO_PROCESS_RECORDED = "no_process_recorded"

STOP_ATTEMPT_OUTCOMES: Tuple[str, ...] = (
    STOP_ATTEMPT_KILLED,
    STOP_ATTEMPT_ALREADY_EXITED,
    STOP_ATTEMPT_TRACKED_PID_MISSING,
    STOP_ATTEMPT_NO_PROCESS_RECORDED,
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


# no_op_reason vocabulary for ``StopHookOutcome.no_op_reason`` (ticket #128) --
# distinguishes an ``isolation: none`` contract's "nothing to stop, by
# design" no-op from an ordinary "this role was never started" no-op. Both
# previously surfaced identically via
# ``StopAttempt(outcome=STOP_ATTEMPT_NO_PROCESS_RECORDED)`` -- this
# vocabulary lives on the new, orthogonal ``StopHookOutcome`` field instead
# of extending ``STOP_ATTEMPT_OUTCOMES`` (which keeps its original, narrower
# meaning unchanged).
STOP_NO_OP_ISOLATION_NONE = "isolation_none"
STOP_NO_OP_NO_PROCESS_RECORDED = "no_process_recorded"

STOP_NO_OP_REASONS = frozenset(
    {
        STOP_NO_OP_ISOLATION_NONE,
        STOP_NO_OP_NO_PROCESS_RECORDED,
    }
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
class StopAttempt:
    """Machine-readable verdict of what a ``stop()`` call found at the
    tracked PID itself (ticket #110), orthogonal to :class:`StopDetail`.

    ``StopDetail`` answers "did anything this call tried to kill survive?".
    ``StopAttempt`` answers a narrower, earlier question: "was the *tracked*
    PID actually alive when this call started, or had it already gone
    stale?" -- and, when it was already stale, "did that matter?" (i.e. did
    real work from its tree/process-group/Job Object still get found and
    killed, or was there genuinely nothing left).

    This closes an ambiguity finding #110-2: a composite/chained shell
    command (e.g. ``cmd /c "echo running>marker & ping -n 120 127.0.0.1
    >nul"``) can leave the tracked wrapper PID dead while the real
    backgrounded child keeps running under a different, untracked PID.
    Before this field existed, that case and the genuinely-nothing-to-do
    case both surfaced identically as ``killed_pids: []`` -- a caller could
    not tell them apart.

    ``outcome`` is always one of :data:`STOP_ATTEMPT_OUTCOMES`:

    - ``"killed"``: the tracked pid was alive at entry -- kills were
      attempted against it (and its tree/group/job).
    - ``"already_exited"``: the tracked pid was already dead at entry, AND
      nothing from its own tree/process-group/Job Object was found either --
      genuinely nothing to do for that pid's lineage. (This can still be
      true even when the unrelated, path-heuristic orphan scan separately
      found and killed something -- that scan is not evidence about the
      tracked pid's own tree/group/job, so it deliberately does not affect
      this outcome; see finding #110's fix cycle blocking finding #1.)
    - ``"tracked_pid_missing"``: the tracked pid was already dead at entry,
      BUT something from its OWN tree/process-group/Job Object was found and
      killed anyway -- the tracked pid itself had gone stale while real work
      it spawned kept running. This is the composite-command case above.
      Deliberately based on ``killed_tree`` only, never on the orphan scan's
      results.
    - ``"no_process_recorded"``: emitted only by
      ``WorktreeManager.stop()``'s no-op branch, for a role that has no
      entry in ``record.pids`` at all -- ``process_lifecycle.stop()`` (and
      therefore every other outcome above) was never even reached.

    ``message`` mirrors :class:`StopDetail`'s message-parity convention:
    the exact human-readable string also passed to a log call for the same
    branch (``_logger.warning`` for ``"tracked_pid_missing"``, `debug` for
    ``"already_exited"``), so the log line and this field can never drift.

    ``tracked_pid`` / ``tracked_pid_alive`` record the pid that was checked
    and whether it was alive at entry -- ``tracked_pid`` is ``None`` for
    ``"no_process_recorded"`` (there was no pid to check).
    ``kill_orphans_may_help`` mirrors :class:`StopDetail`'s hint: ``True``
    when a ``kill_orphans=True`` retry might catch something this call
    missed (only meaningful for ``"tracked_pid_missing"`` when the orphan
    scan did not already run).

    Deliberately **transient**, exactly like ``WorktreeRecord.killed_pids``:
    this describes a single call's attempt, not a durable verdict, so it is
    never persisted to ``state.yaml`` (``yaml_store._record_to_dict`` never
    serialises it) and carries the same in-memory-store-by-reference caveat
    documented on ``killed_pids`` for ``InMemoryStateStore``-backed
    managers. Deliberately kept as its OWN field rather than overloading
    ``stop_detail``: the invariant ``stop_detail is not None implies status
    == "stop_incomplete"`` must stay intact, and ``"tracked_pid_missing"``
    is a *successful* stop (something else was found and killed) that must
    never be reported as incomplete.
    """

    outcome: str
    message: str
    role: Optional[str] = None
    tracked_pid: Optional[int] = None
    tracked_pid_alive: bool = False
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


@dataclass(frozen=True)
class StopHookOutcome:
    """Machine-readable verdict of the contract ``stop:`` hook run by
    ``WorktreeManager.stop()`` -- or, since ticket #130, by
    ``WorktreeManager.remove()``/``_teardown()`` (Step 1b's best-effort
    hook run on the force-remove path) -- plus the contract diagnostics
    that verdict depends on (ticket #128).

    Before this field existed, ``stop()`` read ``contract.isolation``
    nowhere and discarded ``SetupRunner.run()``'s return value outright, so
    two very different situations surfaced identically as
    ``stop_attempt.outcome == "no_process_recorded"``: an ``isolation: none``
    contract's "there is nothing to stop, by design" no-op, and an ordinary
    "this role was simply never started" no-op. ``stop_hook_outcome``
    answers "did the ``stop:`` hook itself run, how did it end, and what did
    the contract actually say about isolation?" -- orthogonal to
    :class:`StopAttempt` (which answers a narrower question about the
    tracked PID itself) and to :class:`StopDetail` (which answers "did
    anything survive the kill attempt?").

    ``status`` is always one of :data:`SETUP_STATUSES` (the same vocabulary
    :class:`SetupOutcome` uses -- deliberately not a new one):

    - ``"completed"``: the contract had ``stop:`` steps and
      ``SetupRunner.run()`` returned without raising.
    - ``"failed"``: the contract's ``stop:`` steps could not be determined at
      all (the contract itself failed to load/parse), or the contract had
      ``stop:`` steps and running them raised -- either way, ``message`` is
      ``str()`` of the exception, mirroring :class:`SetupOutcome`'s
      message-parity convention. Never propagates past ``stop()``: exactly
      as before this field existed, a ``stop:`` hook failure is swallowed so
      it can never block the SIGTERM that follows.
    - ``"skipped"``: the contract loaded fine and simply had no ``stop:``
      steps to run (missing contract, empty contract, explicit
      ``stop: []``, or -- structurally, since the schema forbids ``stop:``
      entries under ``isolation: none`` -- any ``isolation: none`` contract).

    ``steps_run`` is the number of ``stop:`` steps that actually ran -- the
    length of the underlying ``SetupResult.steps`` on a ``"completed"`` run;
    always ``0`` for ``"skipped"``/``"failed"``.

    ``contract_found`` is the independent, filesystem-only
    ``contract_path.exists()`` check -- computed separately from the
    parse/validate attempt below so a filesystem error on the ``.exists()``
    check can never mask a successfully parsed contract (or vice versa).
    ``contract_path`` is the forward-slash ``<repo_root>/.seretos/
    worktree-setup.yml`` path that was probed, regardless of whether it was
    found or parsed. ``contract_isolation`` is the loaded contract's
    ``isolation`` value (``"full"``/``"partial"``/``"none"``), or ``None``
    when the contract could not be loaded/parsed at all.

    ``no_op_reason``, one of :data:`STOP_NO_OP_REASONS` or ``None``, is the
    actual point of ticket #128: set only on ``WorktreeManager.stop()``'s
    no-op branch (the resolved role has no entry in ``record.pids``) --

    - :data:`STOP_NO_OP_ISOLATION_NONE`: ``contract_isolation == "none"`` --
      there was never anything to start or stop for this role, by design.
    - :data:`STOP_NO_OP_NO_PROCESS_RECORDED`: any other contract isolation --
      an ordinary "this role was never started" no-op.

    ``None`` on the delegated path (the resolved role *does* have a recorded
    pid, so ``stop()`` was not a no-op at all) and on every other code path
    -- including, always, an outcome built by ``_teardown()`` (ticket #130):
    that path has no "role never started" no-op concept at all (Step 1b
    runs unconditionally, once, regardless of ``record.pids``), so
    ``no_op_reason`` is unconditionally ``None`` there.

    Deliberately **transient**, exactly like :class:`StopAttempt`: recomputed
    on every ``stop()`` call and never persisted to ``state.yaml``
    (``yaml_store._record_to_dict`` never serialises it) -- unlike
    :class:`SetupOutcome`, which is written once by ``create()`` and
    persisted. Subject to the same in-memory-store-by-reference caveat
    documented on ``killed_pids`` for ``InMemoryStateStore``-backed managers.
    """

    status: str
    message: str = ""
    steps_run: int = 0
    contract_found: bool = False
    contract_path: Optional[str] = None
    contract_isolation: Optional[str] = None
    no_op_reason: Optional[str] = None


# Reason vocabulary for ``BaseFetchFallback.reason`` (ticket #134) -- the two
# situations in which ``WorktreeManager.create()``'s best-effort
# ``git fetch origin <base>`` for an explicit ``base=`` falls back to the
# local ``base`` ref instead of hard-failing.
BASE_FETCH_FALLBACK_REASON_FETCH_FAILED = "fetch_failed"
BASE_FETCH_FALLBACK_REASON_FETCH_TIMEOUT = "fetch_timeout"

BASE_FETCH_FALLBACK_REASONS: Tuple[str, ...] = (
    BASE_FETCH_FALLBACK_REASON_FETCH_FAILED,
    BASE_FETCH_FALLBACK_REASON_FETCH_TIMEOUT,
)

# Cap on how many characters of git's stderr ``BaseFetchFallback.stderr`` may
# carry verbatim -- mirrors ``StopDetail``'s ``survivor_pids`` cap pattern:
# bound the persisted payload rather than writing an unbounded blob to
# state.yaml.
_BASE_FETCH_STDERR_MAX_CHARS = 2000


@dataclass(frozen=True)
class BaseFetchFallback:
    """Machine-readable record of a best-effort ``git fetch origin <base>``
    degrading to the local ``base`` ref instead of hard-failing
    ``WorktreeManager.create()`` (ticket #134).

    Before this field existed, an explicit ``base=`` whose ``git fetch``
    failed (no ``origin`` remote, auth failure, unknown remote ref, DNS/
    network error) or timed out unconditionally raised -- turning a purely
    local operation (branching off an already-locally-present ``base``) into
    a hard failure, or a ``WORKTREE_GIT_TIMEOUT_SEC``-bounded stall, for no
    reason beyond the remote being unreachable. ``create()`` now re-verifies
    the local ``base`` ref exists at fallback time and, when it does, warns
    and proceeds from the local ref instead.

    ``reason`` is always one of :data:`BASE_FETCH_FALLBACK_REASONS`, except
    when loaded from a ``state.yaml`` written by a *future* engine version,
    in which case an unrecognised value is preserved verbatim rather than
    rejected (forward compatibility -- mirrors :class:`StopDetail`'s
    convention, see ``yaml_store._base_fetch_fallback_from_dict``):

    - ``"fetch_failed"``: ``git fetch origin <base>`` exited non-zero.
      ``returncode``/``stderr`` are populated; ``elapsed_sec`` is ``None``.
    - ``"fetch_timeout"``: the fetch was killed by ``_run_git``'s own
      timeout (``GitTimeoutError``). ``elapsed_sec`` is populated;
      ``returncode``/``stderr`` are ``None``.

    ``base`` is the explicit base ref that was requested (the exact string
    passed as ``create(base=...)``).

    ``message`` is the exact human-readable string also passed to
    ``_logger.warning(...)`` for the same fallback, so the log line and this
    field can never drift apart (mirrors :class:`StopDetail`'s
    message-parity convention).

    ``stderr`` is git's stderr output, truncated to
    :data:`_BASE_FETCH_STDERR_MAX_CHARS` characters.

    Deliberately an ``Optional[BaseFetchFallback]`` rather than a bare
    ``bool`` on :class:`WorktreeRecord` -- the *reason* and git's stderr are
    the actionable part a caller needs to decide whether the created
    worktree's ``base`` might be stale.

    Persisted through ``state.yaml`` (mirrors ``stop_detail``) -- a legacy
    record with no ``base_fetch_fallback`` key deserialises to ``None``.
    Written **only** by the explicit-base fetch block in ``create()``; never
    touched by ``start``/``stop``/``reconcile``/``adopt``/``remove``/
    ``list_repo``. ``None`` means "no fallback happened" -- either the fetch
    succeeded, or no fetch was attempted at all (``fetch=False``, or the
    base was defaulted rather than explicit).
    """

    reason: str
    base: str
    message: str
    returncode: Optional[int] = None
    stderr: Optional[str] = None
    elapsed_sec: Optional[float] = None


@dataclass(frozen=True)
class OrphanScanEntry:
    """One process hit reported by the new all-platform, warn-only
    orphan-detection teardown phase (``_phase_orphan_scan``, ticket #140).

    ``info`` is the underlying :class:`~.process_lifecycle.KilledProcessInfo`
    (carrying its own ``source``/``match_pass`` provenance). ``owned`` is
    ``True`` iff ``info.pid`` was one of the pids ``record.pids`` tracked at
    context-build time (mirrors Gate A's own owned/foreign distinction --
    see ``teardown._phase_gate_a_blocking_preflight``). ``killed`` is
    ``True`` only for a pid the reused ``_kill_blocking_processes`` remedy
    actually confirmed killed on this attempt -- a pid seen only by the
    warn scan (no kill attempted, or attempted but not confirmed by the
    kill's own tighter rescan) is reported with ``killed=False``.
    """

    info: "KilledProcessInfo"
    owned: bool
    killed: bool


@dataclass(frozen=True)
class OrphanScanReport:
    """Machine-readable result of one ``_phase_orphan_scan`` invocation
    (ticket #140), attached to :attr:`WorktreeRecord.orphan_scan`.

    Unlike :class:`StopDetail`'s ``survivor_pids``, ``entries`` is
    deliberately **uncapped** -- a deliberate, documented divergence: the
    phase is warn-only and never blocks removal, so there is no bounded
    "how many can I afford to list before giving up and refusing" trade-off
    to make; every hit found is worth reporting.

    ``entries`` is the **union** of the initial full-budget warn scan and
    (when ``kill_blocking_processes=True`` and the scan produced at least
    one hit) the reused ``_kill_blocking_processes`` remedy's own result,
    first-wins pid-deduped, preserving scan order then kill order -- see
    ``teardown._phase_orphan_scan``'s docstring for the full merge
    rationale.

    ``skipped_passes`` carries forward the scan's own incompleteness tags
    (``"cwd:truncated"``, ``"handle_scan:skipped"``, ...) plus any the kill
    result added, deduped, order-preserving, plus the synthetic
    ``"scan:failed"`` marker (ticket #107) when an unexpected exception
    anywhere in the phase's body -- scan OR kill -- was caught rather than
    left to propagate and turn a working removal into a failure.

    ``kill_attempted`` is ``True`` whenever the kill was actually attempted
    on this invocation, **including when it raised** -- an honest record of
    intent, not of success (mirrors ``WorktreeDirLockedError``'s own
    ``kill_attempted`` semantics elsewhere in this package).

    ``message`` is the exact human-readable string also passed to the one
    ``_logger.warning(...)`` call the phase emits for an abnormal outcome
    (message parity, matching ``StopDetail``/``BaseFetchFallback``).

    Like ``killed_pids``/``stop_hook_outcome``, this is deliberately
    **transient**: ``yaml_store._record_to_dict`` is never taught about
    ``orphan_scan`` -- it describes one live removal attempt's scan, not a
    stored verdict, so no legacy-key deserialization path exists for it.
    """

    message: str
    entries: Tuple[OrphanScanEntry, ...] = ()
    skipped_passes: Tuple[str, ...] = ()
    kill_attempted: bool = False


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
    start_log_paths: Dict[str, str] = field(default_factory=dict)
    """Ticket #119: per-role mapping of ``role`` -> the absolute path of the
    ``start-<role>.log`` file ``start()`` wrote for that role. Mirrors
    ``pids``/``job_names``/``variants``: one entry per role that has ever
    been started, and a role with no recorded start log has no key in this
    dict at all, never a ``None`` value. Persisted through ``state.yaml`` --
    a legacy record written by an older engine (carrying only the removed
    scalar ``start_log_path:`` key) deserialises to ``{}``.

    Ticket #119 fix-cycle note: this was originally a single scalar
    ``start_log_path: Optional[str]`` field, unconditionally overwritten by
    every ``start()`` call regardless of role -- in a multi-role worktree,
    starting a second role silently overwrote it, so the field ended up
    naming only the most-recently-started role's log, and e.g.
    ``stop(role="main")`` could hand the caller ``ui``'s log path. Changed
    to a per-role dict, consistent with ``pids``, to close that cross-role
    bleed. The old scalar field is removed entirely (no alias, no
    deprecation shim).

    Deliberate invariant divergence from ``job_names``/``variants``: entries
    here are **retained** by ``stop()`` and by ``reconcile``'s dead-role
    sweep, so ``set(record.start_log_paths) <= set(record.pids)`` does
    *not* hold and is not meant to. The log file outlives the process it
    describes, and the ticket's primary use case is reading a role's log
    path off the response of the ``stop()`` call that just stopped it.
    Restarting the same role overwrites that role's single entry rather
    than accumulating."""
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

    stop_attempt: Optional[StopAttempt] = None
    """Ticket #110: machine-readable verdict of what the most recent
    ``stop()`` call found at the tracked PID itself, or ``None``. See
    :class:`StopAttempt`'s own docstring for the full rationale and outcome
    vocabulary. Deliberately **transient**, exactly like ``killed_pids``:
    never persisted to ``state.yaml`` (unlike ``stop_detail``), cleared by
    ``start()`` in lockstep with ``stop_detail``, and subject to the same
    in-memory-store-by-reference caveat documented on ``killed_pids`` for
    ``InMemoryStateStore``-backed managers."""

    teardown_ran: bool = False
    """Ticket #126: at-most-once-teardown marker. Set ``True`` immediately
    after ``WorktreeManager._teardown()`` actually runs the contract's
    ``teardown:`` steps, and persisted *before* the subsequent
    ``git worktree remove`` is attempted -- so if that removal then fails
    with a post-teardown dirty-tree error, a caller's ``force=True`` retry
    (which bypasses Gate B) still sees the marker and skips re-running
    ``teardown:``. Unconditional scalar (mirrors ``branch_created_by_us``,
    not the ``Optional``/nullable style of ``stop_detail``): persisted
    through ``state.yaml``, and a legacy record with no ``teardown_ran`` key
    deserialises to ``False``. Cleared back to ``False`` by ``start()``
    (both the normal lifecycle-start path and the no-op ``"ready"`` path) --
    a restarted environment is a new logical lifecycle and must earn a fresh
    teardown."""

    stop_hook_outcome: Optional[StopHookOutcome] = None
    """Ticket #128: machine-readable verdict of the most recent ``stop()``
    call's contract ``stop:`` hook, plus contract diagnostics (whether a
    contract file was found, its path, and its ``isolation``), or ``None``.
    Since ticket #130, also set by ``WorktreeManager.remove()`` /
    ``_teardown()`` (its own best-effort Step 1b hook run) -- so this field
    now answers "what did the most recent ``stop()`` OR ``remove()`` call's
    ``stop:`` hook do", not ``stop()`` alone. See :class:`StopHookOutcome`'s
    own docstring for the full rationale -- in short, this is orthogonal to
    ``stop_attempt``: ``stop_attempt`` answers what happened to the tracked
    PID itself, this field answers whether/how the contract's ``stop:``
    hook ran and what the contract's isolation was, which is what lets a
    caller tell an ``isolation: none`` "nothing to stop, by design" no-op
    apart from an ordinary "role never started" no-op (``stop()`` only --
    ``no_op_reason`` is always ``None`` when this outcome came from
    ``remove()``/``_teardown()``, which has no no-op concept of its own).
    Note also that ``stop_attempt`` itself stays whatever an earlier
    ``stop()`` call left it as (or ``None``) on the teardown path --
    ``_teardown()`` deliberately does not compute a new one, since it stops
    every role in a loop and ``StopAttempt`` is single-valued. Deliberately
    **transient**, exactly like ``stop_attempt``: recomputed on every
    ``stop()``/``remove()`` call and NOT persisted to ``state.yaml``."""

    base_fetch_fallback: Optional[BaseFetchFallback] = None
    """Ticket #134: machine-readable record of a best-effort
    ``git fetch origin <base>`` degrading to the local ``base`` ref instead
    of hard-failing ``create()``, or ``None``. See
    :class:`BaseFetchFallback`'s own docstring for the full rationale and
    invariant -- in short: written ONLY by the explicit-base fetch block in
    ``WorktreeManager.create()``; never touched by ``start``/``stop``/
    ``reconcile``/``adopt``/``remove``/``list_repo``. Persisted through
    ``state.yaml`` (mirrors ``stop_detail``) -- a legacy record with no
    ``base_fetch_fallback`` key deserialises to ``None``. ``None`` means "no
    fallback happened" (fetch succeeded, or no fetch was attempted at all --
    ``fetch=False``, or a defaulted rather than explicit ``base``)."""

    orphan_scan: Optional[OrphanScanReport] = None
    """Ticket #140: machine-readable result of the most recent
    ``_phase_orphan_scan`` invocation (the new all-platform, warn-only
    teardown phase, index 6), or ``None``. See :class:`OrphanScanReport`'s
    own docstring for the full rationale -- in short: written ONLY by
    ``teardown._phase_orphan_scan``; never touched by ``start``/``stop``/
    ``reconcile``/``adopt``/``list_repo``. Deliberately **transient**,
    exactly like ``killed_pids``/``stop_hook_outcome``: never persisted to
    ``state.yaml``, and subject to the same in-memory-store-by-reference
    caveat documented on ``killed_pids`` for ``InMemoryStateStore``-backed
    managers. ``None`` means either the phase has not yet run for this
    record, or its most recent run found a clean, complete scan (no hits,
    no incompleteness to report) -- both indistinguishable and both
    correctly "nothing to warn about"."""

    start_variants: List[str] = field(default_factory=list)
    """Ticket #146: the full list of variant strings that WOULD resolve
    against ``start()``'s step-selection tiers for the contract ``create()``
    or ``start()`` just loaded -- i.e. ``manager.available_variants(contract
    .start)`` (the promoted, public form of the former private
    ``_available_variants`` helper). Populated by ``create()`` (from the
    contract it loads while building the record) and by both of ``start()``'s
    return paths (the no-op ``"ready"`` path and the real-spawn path), always
    from the FULL available-variants list for the contract, not just the one
    variant that was selected/started.

    Non-Optional, defaulting to ``[]`` -- deliberately never ``None``, unlike
    the ``Optional[...]`` fields above: closing the ``worktree_create``
    ``[]``-vs-``null`` inconsistency at its source is the entire point of
    this field, so any caller can rely on ``list(record.start_variants)``
    unconditionally, with no ``is None`` guard.

    Deliberately **transient**, exactly like ``killed_pids``/
    ``shadowed_contract``/``stop_attempt``/``stop_hook_outcome``/
    ``orphan_scan``: never persisted to ``state.yaml`` (``yaml_store
    ._record_to_dict`` never serialises it) -- it is a live recomputation
    from the contract, not a stored verdict, so a legacy record round-tripped
    through ``YamlStateStore`` always comes back with ``start_variants ==
    []``. Subject to the same in-memory-store-by-reference caveat documented
    on ``killed_pids`` for ``InMemoryStateStore``-backed managers: because
    ``InMemoryStateStore.update()`` stores the record object by reference (no
    copy), a value set by one ``create()``/``start()`` call can keep
    appearing on later ``get()``/``list()`` calls until the next call that
    assigns it overwrites it."""


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
    "BaseFetchFallback",
    "BASE_FETCH_FALLBACK_REASONS",
    "BASE_FETCH_FALLBACK_REASON_FETCH_FAILED",
    "BASE_FETCH_FALLBACK_REASON_FETCH_TIMEOUT",
    "InMemoryStateStore",
    "OrphanScanEntry",
    "OrphanScanReport",
    "SetupOutcome",
    "SETUP_STATUSES",
    "SETUP_STATUS_COMPLETED",
    "SETUP_STATUS_FAILED",
    "SETUP_STATUS_SKIPPED",
    "ShadowedContract",
    "SHADOW_REASONS",
    "StateStore",
    "StopAttempt",
    "STOP_ATTEMPT_OUTCOMES",
    "StopDetail",
    "STOP_REASONS",
    "StopHookOutcome",
    "STOP_NO_OP_REASONS",
    "STOP_NO_OP_ISOLATION_NONE",
    "STOP_NO_OP_NO_PROCESS_RECORDED",
    "WorktreeRecord",
)
