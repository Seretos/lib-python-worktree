"""Teardown/remove phase engine for ``WorktreeManager`` (ticket #135).

``WorktreeManager._teardown()`` used to be a single ~830-line method in
``core/manager.py`` that had been patched five times in four days (tickets
#117, #123, #126, #127, #130), each for a genuine but independent
regression in the same code path. This module is the structural extraction
of that method into an explicit, ordered sequence of named phases over a
single shared context object (:class:`_TeardownContext`), so each phase --
Stop, ``stop:`` hook, Gate A, Gate B, ``teardown:``, orphan scan (ticket
#140), ``git worktree remove``, FS fallback, final guard, port release --
is individually testable and its invariant is stated in one place.

See ``docs/teardown-phase-contract.md`` for the full phase-by-phase
contract (kept in sync with :data:`_TEARDOWN_PHASES` by
``tests/test_teardown_contract_doc.py``) and ``tests/test_teardown_matrix.py``
for the consolidated regression matrix covering the eleven historical
scenarios (#76, #84, #88, #103, #107, #117, #121, #123, #126, #127, #130)
this module must keep reproducing bit-for-bit.

This module deliberately never imports ``core/manager.py`` (guarded by
``tests/test_teardown_phases.py``'s no-cycle test) -- the only symbol it
used to need from there, ``GitCommandError``, was relocated to
``_exceptions.py`` for exactly this reason (ticket #135, decision A1).
``manager.py`` imports this module (``from . import teardown as
_teardown_mod``), not the other way around.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Set, Tuple

import portalocker

from ..contract.loader import CONTRACT_FILENAME, load as _load_contract
from ._exceptions import (
    DirtyWorktreeError,
    GitCommandError,
    PrimaryCheckoutError,
    WorktreeDirLockedError,
    WorktreeRemovalBlockedError,
)
from ._git_utils import _run_git
from .process_lifecycle import (
    KilledProcessInfo,
    ProcessLifecycleError,
    ProcessNotRunningError,
    _find_blocking_processes,
    _kill_blocking_processes,
)
from .state import (
    SETUP_STATUS_COMPLETED,
    SETUP_STATUS_FAILED,
    SETUP_STATUS_SKIPPED,
    StateStore,
    StopHookOutcome,
    WorktreeRecord,
)

if TYPE_CHECKING:
    from ..contract.schema import WorktreeContract

# Ticket #135 (A7): deliberately the OLD (pre-move) module name, not
# ``getLogger(__name__)`` -- this keeps every existing ``caplog.set_level(...,
# logger="lib_python_worktree.core.manager")`` site in tests/test_teardown.py
# (and any downstream consumer filtering on that logger name) working
# unchanged after the code that emits these records moved to this module.
# Pinned by a guard test asserting ``teardown._logger.name ==
# "lib_python_worktree.core.manager"``.
_logger = logging.getLogger("lib_python_worktree.core.manager")

# Top-level directory segment of CONTRACT_FILENAME (".seretos"), derived
# rather than hardcoded a second time (ticket #100). The agent-worktree
# plugin copies this whole directory into every new checkout as a
# convenience; it is untracked there, and its mere presence makes a plain
# `git worktree remove` refuse. See `_contract_copy_dirt_paths`.
_CONTRACT_DIR = PurePosixPath(CONTRACT_FILENAME).parts[0]

# Ticket #154: the checkout is staged here before being deleted --
# `os.rename(record.path, record.path + _STAGED_SUFFIX)` proves unheldness
# on Windows and is the staging step on both platforms. One name for a
# thing that did not exist before, referenced by the stage/delete phase,
# the delete ladder, the final guard, and the `staged=` message phrasings.
_STAGED_SUFFIX = ".removing"

# Ticket #154 (item 4, P1): bounded transient retry, shared by every
# failure-path entry point (a failed rename, a residual after the delete
# ladder). Absorbs a millisecond-scale transient handle holder (AV/
# indexer/IDE watcher) before ever escalating to diagnosis. Replaces the
# deleted settle-rescan (_PREFLIGHT_SETTLE_*) and post-kill retry loop
# (_POST_KILL_*) constants -- one shared loop instead of two bespoke ones.
_TRANSIENT_RETRY_BUDGET_SEC: float = 2.0
_TRANSIENT_RETRY_STEP_SEC: float = 0.1

# Ticket #154 (item 7, Z3): one wall-clock budget per removal's failure
# path, opened the first time `_retry_bounded` is entered and shared by
# every later failure-path leg of the SAME removal attempt. 5.0s (Z3's
# diagnosis budget) + 2.0s (P1's single unclamped retry allowance).
# Disjoint from AC1's <2s/<5s ceiling by construction: a clean, unheld
# worktree's rename succeeds on the first attempt, so this is never opened
# at all for that population -- see docs/teardown-phase-contract.md.
_FAILURE_BUDGET_SEC: float = 7.0

# Timeout for the Windows long-path robocopy empty-mirror fallback
# (ticket #78). Without a bound, robocopy's own defaults (/R:1000000 /W:30)
# retry a locked directory for ~347 days, wedging the single synchronous
# MCP request thread and hanging the whole server. Mirrors the
# WORKTREE_GIT_TIMEOUT_SEC convention in _git_utils.py: an explicit float
# string overrides the default, an empty string disables the timeout
# entirely (diagnostic use only).
_ROBOCOPY_TIMEOUT_ENV = "WORKTREE_ROBOCOPY_TIMEOUT_SEC"
_ROBOCOPY_TIMEOUT_DEFAULT = 30.0


def _resolve_robocopy_timeout() -> Optional[float]:
    """Resolve the timeout for the robocopy long-path fallback.

    Precedence: ``WORKTREE_ROBOCOPY_TIMEOUT_SEC`` env > built-in default of
    30.0s. ``None`` (env value ``""``) disables the timeout entirely; that
    path exists for diagnostics, not normal use.

    Env is read on every call so that test fixtures and operators can change
    the value without re-importing the module.
    """
    raw = os.environ.get(_ROBOCOPY_TIMEOUT_ENV)
    if raw is None:
        return _ROBOCOPY_TIMEOUT_DEFAULT
    raw = raw.strip()
    if not raw:
        # Empty string is "no timeout", matching _resolve_git_timeout's
        # explicit-None semantics.
        return None
    try:
        value = float(raw)
    except ValueError:
        return _ROBOCOPY_TIMEOUT_DEFAULT
    return value if value > 0 else None


def _status_entries(record: "WorktreeRecord") -> Optional[List[str]]:
    """Run ``git status --porcelain -z`` in *record*'s checkout and return
    the non-empty NUL-split entries, or ``None`` when the probe is
    inconclusive.

    Split out of the former ``_contract_copy_dirt_paths`` (ticket #103) so
    both the ``.seretos/``-exemption classifier and the new "is there any
    real (non-exempt) dirt at all?" predicate (:func:`_real_dirt_paths`)
    share one probe and one defensive contract, instead of running
    ``git status`` twice or duplicating the defensive matrix.

    Returns ``None`` (never an empty list for "inconclusive") on: a
    non-zero exit, a non-``str`` ``stdout`` (defensive -- some tests patch
    ``_run_git`` with a blanket ``MagicMock`` whose ``.stdout`` is itself a
    ``MagicMock``), an empty result (inconsistent with git having just
    refused the removal for being dirty), or any exception (including
    ``GitTimeoutError``). Callers must never invent a blocking condition
    from an inconclusive probe.
    """
    try:
        proc = _run_git(["status", "--porcelain", "-z"], cwd=Path(record.path))
    except Exception:  # noqa: BLE001 -- includes GitTimeoutError; never block removal
        return None

    if proc.returncode != 0:
        return None

    stdout = proc.stdout
    if not isinstance(stdout, str):
        return None

    entries = [e for e in stdout.split("\0") if e]
    if not entries:
        # Empty status while git refused the remove is inconsistent with the
        # "only benign dirt" story -- never treat as conclusive.
        return None

    return entries


def _is_exempt_untracked(entry: str) -> bool:
    """Return ``True`` iff *entry* (one NUL-split ``git status --porcelain
    -z`` record) is untracked (``??``) content living at ``.seretos`` or
    under ``.seretos/`` -- the agent-worktree plugin's benign convenience
    copy (ticket #100).

    A **modified tracked** file under ``.seretos/`` (status code other than
    ``??``) is real work and is never exempt. No path normalisation is
    performed: a literal backslash in an entry name (e.g. ``.seretos\\
    notes.txt``) is part of the filename itself (git always reports ``/``
    as the separator, even on Windows) and stays non-exempt, and a
    ``.seretos``-*prefixed sibling* directory (e.g. ``.seretos-backup/``) is
    not ``.seretos`` itself or a path under it, so it also stays non-exempt.
    """
    if len(entry) < 3 or entry[:2] != "??":
        return False
    path = entry[3:]
    contract_prefix = f"{_CONTRACT_DIR}/"
    return path == _CONTRACT_DIR or path.startswith(contract_prefix)


def _real_dirt_paths(
    record: "WorktreeRecord", *, entries: Optional[List[str]] = None
) -> List[str]:
    """Return the non-exempt (real) dirty paths for *record*'s checkout, or
    ``[]`` when clean, when the only dirt is the benign ``.seretos/``
    convenience copy (ticket #100), or when the ``git status`` probe is
    inconclusive (ticket #103).

    This is the "is there any real dirt that would block removal?"
    predicate consumed by the combined lock+dirt reporting added by ticket
    #103: this always returns a list and never invents a blocking
    condition from an inconclusive probe -- an empty result here means "do
    not report a dirty-tree condition", not "confirmed clean".

    *entries* is an optional pre-fetched result of :func:`_status_entries`,
    used by the teardown context's memoised probe.
    """
    if entries is None:
        entries = _status_entries(record)
    if entries is None:
        return []
    return [entry[3:] for entry in entries if not _is_exempt_untracked(entry)]


def _merge_killed_pids(
    record: "WorktreeRecord", new_entries: Iterable["KilledProcessInfo"]
) -> None:
    """First-wins, pid-deduped, in-place merge of *new_entries* into
    ``record.killed_pids`` (ticket #140, R7).

    Used by tier-1/tier-2 diagnosis (:func:`_diagnose_and_retry`) so a kill
    never overwrites ``record.killed_pids`` outright, silently dropping
    whatever an earlier phase of the SAME removal attempt (or an earlier
    ``stop()`` call) had already recorded there. This mirrors
    ``_kill_blocking_processes``' own ``seen_pids`` idiom: a pid already
    present keeps its existing entry (first-wins -- the earliest-recorded
    :class:`~.process_lifecycle.KilledProcessInfo` for a given pid is kept,
    not replaced by a later, possibly less-detailed rediscovery of the same
    pid), and a genuinely new pid is appended in the order encountered.
    """
    existing_pids = {info.pid for info in record.killed_pids}
    for info in new_entries:
        if info.pid in existing_pids:
            continue
        record.killed_pids.append(info)
        existing_pids.add(info.pid)


def _target_is_absent(record: "WorktreeRecord", force: bool = False) -> bool:
    """Single seam (ticket #135, A8) for the "is this checkout directory
    already gone?" probe used by BOTH ``WorktreeManager.remove()``'s own
    pre-teardown check and :func:`build_context`'s ``target_absent`` field.

    Replaces two textually-identical ``target_absent = not
    os.path.exists(record.path)`` call sites (one in ``remove()``, one in
    the old monolithic ``_teardown()``) that used to require a ~120-line
    stack-walking test fixture (``tests/test_teardown.py``'s
    ``_target_present_by_default``) to distinguish from every OTHER
    ``os.path.exists``/``Path.exists()`` call made during a removal (the
    long-path fallback, the Final guard, the state store's own file checks,
    the contract file check). A single named seam makes that stack-walking
    unnecessary: tests now patch ``teardown._target_is_absent`` directly,
    leaving every other filesystem check untouched by construction.

    Ticket #154 (human override, Decision 1 -- the "lighter fix" replacing
    generation-2 plan.md's rejected ``_phase_reclaim_staged``): a non-empty
    ``<path>.removing`` remnant under ``force=False`` is NOT "target
    absent", even when ``record.path`` itself is gone -- it must surface as
    an error on the next removal attempt (the dirt gate / stage-and-delete
    phase raises, reusing the existing ``DirtyWorktreeError(staged=True)``/
    ``WorktreeDirLockedError`` vocabulary) rather than being silently
    fast-pathed as "nothing to remove here", which would either restore
    (generation-2's rejected design) or silently discard the remnant's
    possibly-uncommitted content. This applies uniformly to both remnant
    classes (an ordinary crash between rename and delete, and the dirt
    gate's rename-probe undo failure) since they cannot be told apart
    cheaply -- a known, accepted trade-off; a human operator who hits this
    must clear the stale ``.removing`` directory by hand or pass
    ``force=True``. Under ``force=True`` the remnant is still treated as
    "target absent" here -- force authorises destroying it, and that
    pre-clean happens in ``_phase_stage_and_delete``'s opening guard, not
    in this seam.
    """
    if os.path.exists(record.path):
        return False
    if not force:
        staged = record.path + _STAGED_SUFFIX
        if os.path.isdir(staged) and os.listdir(staged):
            return False
    return True


@dataclass
class _TeardownContext:
    """Shared, mutable state threaded through every phase in
    :data:`_TEARDOWN_PHASES` (ticket #135, A5).

    ``record`` is mutated in place by several phases (``killed_pids``,
    ``stop_hook_outcome``, ``teardown_ran``) exactly as the old monolithic
    ``_teardown()`` mutated its ``record`` parameter -- callers holding a
    reference to the record still observe every mutation even when a later
    phase raises out of :func:`run_teardown`.

    No phase ever receives (or needs) a ``WorktreeManager`` reference:
    ``store``/``allocator`` are plain fields, carrying exactly what
    ``self.state``/``self._allocator`` supplied at ``build_context()`` time
    (ticket #135, A4) -- this is what keeps this module free of any
    ``core.manager`` import.
    """

    record: "WorktreeRecord"
    force: bool
    kill_blocking_processes: bool
    lifecycle: Any
    store: "StateStore"
    allocator: Any
    target_absent: bool
    owned_pids: Set[int]
    contract: Optional["WorktreeContract"]
    contract_found: bool
    contract_isolation: Optional[str]
    contract_load_error: Optional[str]
    kill_attempted: bool = False
    failure_deadline: Optional[float] = None
    """Ticket #154 (item 7, Z3): opened (set to ``time.monotonic() +
    _FAILURE_BUDGET_SEC``) the first time :func:`_retry_bounded` is entered
    for this removal attempt, and shared by every later failure-path leg of
    the SAME attempt. Stays ``None`` on the happy path (a clean, unheld
    worktree's rename succeeds on its first attempt) and on every
    ``DirtyWorktreeError`` leg (Decision 2, human override -- including the
    ``staged=True`` undo-failure leg, whose own small bounded retry is
    deliberately NOT routed through :func:`_retry_bounded`)."""
    blockers: List["KilledProcessInfo"] = field(default_factory=list)
    """Ticket #154 (item 7/8): the full list of tier-1 (inferred,
    ``source="tracked"``) and tier-2 (path-confirmed, ``source=
    "orphan_scan"``) candidates gathered while diagnosing a failure,
    across every leg of this removal attempt -- distinct from
    ``record.killed_pids``, which only ever holds processes actually
    terminated."""
    _status_cache: List[Optional[List[str]]] = field(default_factory=lambda: [None])
    _status_fetched: List[bool] = field(default_factory=lambda: [False])

    def status_entries(self) -> Optional[List[str]]:
        """Memoised ``git status --porcelain -z`` probe (ticket #103,
        hoisted for #117). See :func:`dirt`'s docstring for the two-phase
        memoisation contract this must preserve verbatim.
        """
        if not self._status_fetched[0]:
            self._status_cache[0] = _status_entries(self.record)
            self._status_fetched[0] = True
        return self._status_cache[0]

    def dirt(self) -> List[str]:
        """Memoised, force-gated dirt probe shared by every phase that may
        need to know about real dirt -- Gate B's early refusal and the
        unconditional dirt gate's own verdict -- so that a single removal
        attempt never issues more than one ``git status`` call no matter
        how many of those sites run, and never runs at all when
        ``force=True`` (real dirt cannot block a forced removal, so there
        is nothing to report).

        Two-phase intent (ticket #117 fix cycle, blocking finding): this
        memoisation is deliberately scoped to TWO phases, not one shared
        snapshot for the whole removal. Phase 1 (pre-teardown) is Gate B --
        runs, by construction, before the ``teardown:`` steps phase, and
        MUST see the pre-teardown tree. Phase 2 (post-teardown) is the
        unconditional dirt gate, which runs AFTER the ``teardown:`` steps
        phase has executed. It must NOT reuse Gate B's pre-teardown
        snapshot -- a ``teardown:`` step can itself leave real, non-exempt
        dirt behind, and a stale snapshot would miss it. The teardown:
        steps phase explicitly calls :meth:`invalidate_dirt` right after
        the steps run (and only then -- when there were none to run,
        nothing changed on disk, so the single pre-teardown probe still
        correctly serves the dirt gate too, preserving the
        zero-extra-``git status``-calls happy path). Do NOT collapse this
        back into one unconditional snapshot for the whole removal.
        """
        if self.force:
            return []
        return _real_dirt_paths(self.record, entries=self.status_entries())

    def invalidate_dirt(self) -> None:
        """Reset the memoised probe (ticket #117 fix cycle, blocking
        finding) once the ``teardown:`` steps phase has actually run and
        may have changed what's on disk. See :meth:`dirt`'s docstring for
        the full two-phase rationale this implements.
        """
        self._status_cache[0] = None
        self._status_fetched[0] = False


def build_context(
    record: "WorktreeRecord",
    *,
    force: bool,
    kill_blocking_processes: bool = False,
    store: "StateStore",
    allocator: Any,
    lifecycle_module: Any = None,
) -> _TeardownContext:
    """Construct the shared :class:`_TeardownContext` for one removal
    attempt (ticket #135, A4).

    Resolves the lifecycle module, computes ``target_absent`` (ticket #127)
    and ``owned_pids`` (ticket #117), and loads the contract exactly once
    (ticket #117: replaces the old duplicate ``_load_contract()`` calls) --
    all of this is cheap, side-effect-free (beyond the FS/contract-load
    reads themselves) setup that every phase may consume; the
    ``_phase_guard_primary`` phase still runs first when
    :func:`run_teardown` iterates :data:`_TEARDOWN_PHASES`, so a primary
    checkout is refused before any destructive work regardless of this
    setup having already run.
    """
    lifecycle = lifecycle_module
    if lifecycle is None:
        from . import process_lifecycle as lifecycle  # type: ignore[assignment]

    target_absent = _target_is_absent(record, force=force)
    owned_pids = {pid for pid in record.pids.values()}

    contract_path = Path(record.repo_root) / CONTRACT_FILENAME
    try:
        contract_found = contract_path.exists()
    except OSError:
        contract_found = False

    contract_isolation: Optional[str] = None
    contract_load_error: Optional[str] = None
    try:
        contract = _load_contract(contract_path)
        contract_isolation = contract.isolation
    except Exception as _contract_exc:  # noqa: BLE001
        contract = None
        contract_load_error = str(_contract_exc)

    return _TeardownContext(
        record=record,
        force=force,
        kill_blocking_processes=kill_blocking_processes,
        lifecycle=lifecycle,
        store=store,
        allocator=allocator,
        target_absent=target_absent,
        owned_pids=owned_pids,
        contract=contract,
        contract_found=contract_found,
        contract_isolation=contract_isolation,
        contract_load_error=contract_load_error,
    )


# ---------------------------------------------------------------------------
# Phases (ticket #135, A6). Order is authoritative -- see
# docs/teardown-phase-contract.md.
# ---------------------------------------------------------------------------


def _phase_guard_primary(ctx: _TeardownContext) -> None:
    """Guard 2 (ticket #84): refuse a primary checkout before any
    lifecycle/FS side effect. Never bypassable via ``force=True``.

    Mirrors ``WorktreeManager.remove()``'s own Guard 1, so any caller
    reaching teardown directly (or a mislabelled record whose ``backing``
    says "worktree" but whose path IS the repo root) can never trigger the
    destructive work in the phases below.
    """
    record = ctx.record
    if record.backing == "primary" or Path(record.path).resolve() == Path(
        record.repo_root
    ).resolve():
        raise PrimaryCheckoutError(record.id)


def _phase_stop_processes(ctx: _TeardownContext) -> None:
    """Step 1: stop any tracked processes before removing the worktree dir."""
    record = ctx.record
    if record.pids:
        for role in list(record.pids.keys()):
            try:
                ctx.lifecycle.stop(record.id, store=ctx.store, role=role)
            except ProcessNotRunningError:
                pass
            except ProcessLifecycleError:
                # Best-effort: log-worthy but don't block the removal.
                pass


def _phase_stop_hook(ctx: _TeardownContext) -> None:
    """Step 1b: run contract ``stop:`` steps before FS deletion so that
    daemons (e.g. Unity Editor) that write PID files / hold handles have a
    chance to release them before ``git worktree remove`` is attempted.

    Ticket #130: also computes a ``StopHookOutcome`` describing whether/how
    the ``stop:`` hook ran, mirroring ``WorktreeManager.stop()``'s own
    diagnostics block. Deliberately scoped to ``stop_hook_outcome`` only --
    ``stop_attempt`` is NOT computed here (by design: ``StopAttempt`` is
    single-valued but ``_phase_stop_processes`` above stops every role in a
    loop, so there is no non-arbitrary single attempt to report;
    ``record.stop_attempt`` stays whatever an earlier ``stop()`` call left
    it as). ``no_op_reason`` is always ``None`` here -- that field is
    populated only by ``stop()``'s own no-op branch.

    Set before Gate A/B below can raise: ``record`` is mutated in place, so
    a caller holding the reference must still see the outcome even when a
    later gate raises out of this call.
    """
    record = ctx.record
    stop_hook_status = SETUP_STATUS_SKIPPED
    stop_hook_message = "no stop: steps in contract"
    stop_hook_steps_run = 0
    if ctx.contract_load_error is not None:
        stop_hook_status = SETUP_STATUS_FAILED
        stop_hook_message = ctx.contract_load_error
    elif ctx.contract is not None and ctx.contract.stop:
        from ..setup.runner import SetupRunner
        runner = SetupRunner()
        try:
            _stop_result = runner.run(
                setup=ctx.contract.stop,
                worktree_id=record.id,
                worktree_path=Path(record.path),
                branch=record.branch,
                port_mapping=record.ports,
            )
            stop_hook_status = SETUP_STATUS_COMPLETED
            stop_hook_steps_run = len(_stop_result.steps)
            stop_hook_message = (
                f"stop: completed {stop_hook_steps_run} step(s)"
            )
        except Exception as _stop_exc:  # noqa: BLE001
            # A stop-step failure must not block the rest of teardown.
            stop_hook_status = SETUP_STATUS_FAILED
            stop_hook_message = str(_stop_exc)

    if stop_hook_status == SETUP_STATUS_FAILED:
        _logger.warning(
            "_teardown: worktree '%s' contract stop: hook failed "
            "(non-blocking): %s",
            record.id,
            stop_hook_message,
        )

    contract_path = Path(record.repo_root) / CONTRACT_FILENAME
    record.stop_hook_outcome = StopHookOutcome(
        status=stop_hook_status,
        message=stop_hook_message,
        steps_run=stop_hook_steps_run,
        contract_found=ctx.contract_found,
        contract_path=contract_path.as_posix(),
        contract_isolation=ctx.contract_isolation,
        no_op_reason=None,
    )



def _phase_gate_b_early_dirty(ctx: _TeardownContext) -> None:
    """Gate B (ticket #117, AC #3): early dirty-tree refusal, run BEFORE
    the contract ``teardown:`` steps phase -- but only when there ARE
    ``teardown:`` steps to protect, so the happy path (no ``teardown:``
    steps) pays zero extra ``git status`` calls, preserving the
    memoisation rationale documented on :meth:`_TeardownContext.dirt`.
    Without this gate, a removal attempt that is going to fail for being
    dirty anyway would still run a non-idempotent ``teardown:`` command
    first (e.g. one that generates a file), and a later retry would re-run
    it again.

    Ticket #123 pins this as the deliberate current behaviour (this exact
    scenario has no dedicated historical ticket of its own): a dirty tree
    with ``teardown:`` steps present and ``force=False`` refuses BEFORE
    those steps ever run.
    """
    if not ctx.force and ctx.contract is not None and ctx.contract.teardown:
        _dirt = ctx.dirt()
        if _dirt:
            raise DirtyWorktreeError(ctx.record.id)


def _phase_run_teardown_steps(ctx: _TeardownContext) -> None:
    """Step 3 (ticket #117, guarded further by #126): run contract
    ``teardown:`` steps. Reached ONLY when both gates above have passed --
    i.e. only when this attempt is expected to actually proceed to
    ``git worktree remove`` -- AND only when they have not already run for
    this record (``not record.teardown_ran``). A refused attempt (Gate A or
    Gate B) never reaches here, so a later retry that clears the blocking
    condition is the FIRST and ONLY execution of these steps for this
    logical removal (AC #3) -- a non-idempotent teardown command is never
    re-run just because an earlier attempt was rejected before any FS
    mutation happened.

    Ticket #126 closes a SECOND path to the same symptom: teardown: steps
    run, then the git-worktree-remove phase fails with a POST-teardown
    ``DirtyWorktreeError`` (e.g. because a teardown step itself wrote a
    file), and the caller retries with ``force=True`` -- which bypasses
    Gate B entirely. Without the ``teardown_ran`` guard, that retry would
    re-enter this phase and re-run ``teardown:`` a second time. The marker
    is set and persisted immediately below, BEFORE the git-worktree-remove
    phase is attempted, so it survives exactly the failure this ticket is
    about.

    ``record.teardown_ran`` persistence is best-effort: ``self.state.
    update()`` raises ``KeyError`` for the synthesised, never-stored record
    used by ticket #88's untracked-target removal path; that record has
    nothing to persist to, so the ``KeyError`` is swallowed rather than
    propagated -- for that path only, two successive removals of the same
    checkout may still re-run ``teardown:``, a documented limitation rather
    than a bug. ``OSError``/``portalocker.exceptions.LockException`` are
    swallowed for the same reason: the teardown steps above already
    completed successfully; only the bookkeeping write itself can still
    fail here, and that must not block the actual removal any more than a
    teardown-step failure does.
    """
    record = ctx.record
    if ctx.contract is not None and ctx.contract.teardown and not record.teardown_ran:
        from ..setup.runner import SetupRunner
        runner = SetupRunner()
        try:
            runner.run(
                setup=ctx.contract.teardown,
                worktree_id=record.id,
                worktree_path=Path(record.path),
                branch=record.branch,
                port_mapping=record.ports,
            )
        except Exception:  # noqa: BLE001
            # Teardown step failure must not block git worktree remove
            # (#117's policy). Ticket #126 fix cycle (blocking finding): do
            # NOT set teardown_ran here -- see the phase docstring above.
            pass
        else:
            record.teardown_ran = True
            try:
                ctx.store.update(record)
            except (KeyError, OSError, portalocker.exceptions.LockException):
                pass

        # Invalidate the dirt probe's memoised snapshot now that
        # teardown: steps have actually run and may have changed what's on
        # disk -- see _TeardownContext.dirt()'s two-phase docstring. Only
        # reset when this branch was entered (teardown steps existed and
        # had not already run): when there are none, nothing changed on
        # disk, so the existing single-probe memoisation still holds.
        ctx.invalidate_dirt()


def _retry_bounded(op, *, ctx: _TeardownContext) -> bool:
    """Attempt *op* (a zero-arg callable) repeatedly until it succeeds or
    the shared per-removal transient-retry budget (ticket #154, item 4,
    P1) is exhausted.

    Opens ``ctx.failure_deadline`` (the larger, per-removal diagnosis
    budget, item 7) if not already open -- every call site after the first
    for this removal attempt shares the same deadline. Absorbs a
    millisecond-scale transient handle holder (AV/indexer/IDE watcher)
    with no sleep in between attempts wasted once *op* starts succeeding.
    Returns ``True`` on success, ``False`` once the budget ran out without
    *op* ever succeeding -- callers then escalate to
    :func:`_diagnose_and_retry`.
    """
    if ctx.failure_deadline is None:
        ctx.failure_deadline = time.monotonic() + _FAILURE_BUDGET_SEC
    remaining = max(0.0, ctx.failure_deadline - time.monotonic())
    deadline = time.monotonic() + min(_TRANSIENT_RETRY_BUDGET_SEC, remaining)
    while True:
        try:
            op()
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(_TRANSIENT_RETRY_STEP_SEC)


def _bare_retry_bounded(op) -> bool:
    """Same bounded-retry idea as :func:`_retry_bounded`, but never touches
    ``ctx.failure_deadline`` at all (human override, Decision 2) -- used
    only by the dirt gate's undo-rename retry, which must never turn a
    plain ``DirtyWorktreeError`` leg into one that consumed the shared
    failure budget. A bespoke, local-only loop rather than an
    ``opens_failure_deadline=False`` keyword on :func:`_retry_bounded`
    (developer's choice, per the override note): this is less mechanism
    for a single call site with a genuinely different contract.
    """
    deadline = time.monotonic() + _TRANSIENT_RETRY_BUDGET_SEC
    while True:
        try:
            op()
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(_TRANSIENT_RETRY_STEP_SEC)


def _diagnose_and_retry(ctx: _TeardownContext, *, trigger: str, retry) -> bool:
    """Tier 1 (owned-pid liveness) then tier 2 (systemwide scan) diagnosis
    for a failed rename or a residual after the delete ladder (ticket
    #154, items 5-6). Each tier that finds a live/confirmed blocker and
    has ``kill_blocking_processes=True`` kills it and retries *retry*
    (a zero-arg callable) once.

    Never raises: a still-unresolved failure is left for
    :func:`_phase_final_guard` -- the sole raise site for this phase pair
    -- to report, using the literal on-disk truth at the end of the
    removal attempt. Returns ``True`` iff *retry* eventually succeeded.

    *trigger* is ``"rename"`` or ``"residual"`` -- both target the staged
    path (``record.path`` + ``_STAGED_SUFFIX``), since both only ever fire
    from :func:`_phase_stage_and_delete`.

    Simplification, named here rather than silently: tier 1's liveness
    probe uses a plain ``psutil.pid_exists()`` call directly rather than
    routing it through a bounded-call primitive shared with
    ``process_lifecycle.py``'s handle-scan wedge pool (plan.md item 5's
    ``_bounded_call``) -- no test in this repo's suite exercises an owned
    pid whose liveness probe itself wedges, and building that shared
    primitive's wiring correctly without a driving test risked shipping
    unverified code. See the developer's final report.
    """
    record = ctx.record
    target = record.path + _STAGED_SUFFIX if trigger in ("rename", "residual") else record.path

    if ctx.failure_deadline is not None and time.monotonic() >= ctx.failure_deadline:
        _logger.warning(
            "_teardown: worktree '%s' failure budget exhausted; skipping "
            "diagnosis (%s trigger)",
            record.id, trigger,
        )
        return False

    import psutil

    tier1_candidates: List["KilledProcessInfo"] = []
    for pid in ctx.owned_pids:
        try:
            alive = psutil.pid_exists(pid)
        except Exception:  # noqa: BLE001 -- never let a liveness probe abort diagnosis
            continue
        if alive:
            tier1_candidates.append(
                KilledProcessInfo(pid=pid, name="", cmdline=[], source="tracked")
            )
    if tier1_candidates:
        ctx.blockers.extend(tier1_candidates)
        _logger.warning(
            "_teardown: worktree '%s' owned pid(s) still alive after the "
            "stop phase; treating as candidate blocker(s): %s",
            record.id, [info.pid for info in tier1_candidates],
        )
        if ctx.kill_blocking_processes:
            ctx.kill_attempted = True
            try:
                ctx.lifecycle._kill_process_tree(
                    tier1_candidates, timeout=_TRANSIENT_RETRY_BUDGET_SEC
                )
            except Exception:  # noqa: BLE001
                pass
            else:
                _merge_killed_pids(record, tier1_candidates)
            try:
                retry()
                return True
            except OSError:
                pass

    remaining = None
    if ctx.failure_deadline is not None:
        remaining = max(0.0, ctx.failure_deadline - time.monotonic())
        if remaining <= 0:
            return False

    try:
        tier2 = _find_blocking_processes(target, os.getpid(), deadline=ctx.failure_deadline)
    except Exception:  # noqa: BLE001
        tier2 = []
    if tier2:
        ctx.blockers.extend(tier2)

    if ctx.kill_blocking_processes:
        ctx.kill_attempted = True
        try:
            killed = _kill_blocking_processes(
                target, timeout=remaining if remaining is not None else 5.0
            )
        except Exception:  # noqa: BLE001
            killed = []
        else:
            _merge_killed_pids(record, killed)
        try:
            retry()
            return True
        except OSError:
            return False

    return False


def _delete_tree_ladder(target: str, *, deadline: Optional[float] = None) -> None:
    """Delete *target* (the staged tree), escalating through rungs only as
    needed (ticket #154, item 3). Merge of today's plain ``shutil.rmtree``
    and the former ``_phase_filesystem_fallback``'s long-path/robocopy
    capability -- not a new capability, a re-homing of an existing one
    onto the staged path. Never raises; the caller checks
    ``os.path.exists(target)`` afterward (directly, or via
    :func:`_phase_final_guard`'s literal check) to learn whether it
    succeeded.

    Entry guard: a *target* that does not exist costs one ``stat`` and
    nothing else -- this is what keeps the ordinary happy-path removal
    from ever reaching rung 3/4 (extended-path rmtree, robocopy).
    """
    if not os.path.exists(target):
        return

    try:
        shutil.rmtree(target)
    except OSError:
        pass
    if not os.path.exists(target):
        return

    # Rung 2 (P4): a read-only file used to be silently absorbed by
    # ``git worktree remove``; plain ``shutil.rmtree`` is not so forgiving.
    # Sweep write-protect off every file, then retry.
    try:
        for root, _dirs, files in os.walk(target):
            for name in files:
                try:
                    os.chmod(os.path.join(root, name), stat.S_IWRITE)
                except OSError:
                    pass
        shutil.rmtree(target)
    except OSError:
        pass
    if not os.path.exists(target):
        return

    if sys.platform != "win32":
        return

    # Rung 3: Windows extended-path (``\\?\``) rmtree, for a tree deep
    # enough to exceed MAX_PATH.
    try:
        shutil.rmtree("\\\\?\\" + os.path.abspath(target))
    except OSError:
        pass
    if not os.path.exists(target):
        return

    # Rung 4: robocopy empty-mirror, last resort. Bounded by
    # WORKTREE_ROBOCOPY_TIMEOUT_SEC (or the remaining removal budget on a
    # retry invocation, whichever is tighter).
    timeout = _resolve_robocopy_timeout()
    if deadline is not None:
        remaining = max(0.0, deadline - time.monotonic())
        timeout = remaining if timeout is None else min(timeout, remaining)
    empty_dir = tempfile.mkdtemp()
    try:
        subprocess.run(
            ["robocopy", empty_dir, target, "/MIR", "/R:1", "/W:1"],
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass
    finally:
        shutil.rmtree(empty_dir, ignore_errors=True)


def _phase_dirt_gate(ctx: _TeardownContext) -> None:
    """Unconditional dirt gate (ticket #154, item 16), immediately before
    staging. Skipped entirely -- probe included -- when ``force=True`` or
    ``ctx.target_absent`` (nothing to probe). Reuses the memoised
    ``ctx.dirt()``, which the ``teardown:`` steps phase already
    invalidates when it ran (see :meth:`_TeardownContext.dirt`'s
    docstring), so the happy path (no real dirt) pays at most one extra
    ``git status`` call, never two.

    Real dirt: a cheap rename probe (forward, then immediately back -- NOT
    the transient-retry loop) decides between the two public outcomes
    (item 17):

    - Forward rename fails -> the directory is genuinely locked too ->
      the combined :class:`WorktreeRemovalBlockedError` (#103).
    - Forward succeeds, undo succeeds -> plain :class:`DirtyWorktreeError`
      (``staged=False``), tree exactly where it was.
    - Forward succeeds, undo fails -> a bounded (but bespoke, budget-
      isolated -- see :func:`_bare_retry_bounded`) retry on the undo; if
      that too fails, :class:`DirtyWorktreeError` (``staged=True``) --
      never the combined error, since the forward probe succeeding
      already proved the directory was not locked (human override,
      Decisions 1+2).
    """
    if ctx.force or ctx.target_absent:
        return
    dirt = ctx.dirt()
    if not dirt:
        return

    record = ctx.record
    staged = record.path + _STAGED_SUFFIX
    try:
        os.rename(record.path, staged)
    except OSError:
        raise WorktreeRemovalBlockedError(
            record.id, killed=[], kill_attempted=False, dirty_paths=dirt
        )

    try:
        os.rename(staged, record.path)
        return_ok = True
    except OSError:
        return_ok = False

    if return_ok:
        raise DirtyWorktreeError(record.id)

    if _bare_retry_bounded(lambda: os.rename(staged, record.path)):
        raise DirtyWorktreeError(record.id)

    _logger.error(
        "_teardown: worktree '%s' dirty-tree undo rename failed; checkout "
        "is staged under '%s.removing' and the marker directory is still "
        "held.",
        record.id, record.id,
    )
    raise DirtyWorktreeError(record.id, staged=True)


def _phase_stage_and_delete(ctx: _TeardownContext) -> None:
    """Merged stage + delete phase (ticket #154, item 2). Rename is the
    oracle on both platforms: staging via ``os.rename`` proves unheldness
    on Windows and moves the checkout out of the way; the staged tree is
    then deleted via :func:`_delete_tree_ladder`.

    Opening guard handles every combination of "is the original tree
    present?" / "is a `<path>.removing` remnant present?" (human override,
    Decision 1 -- the lighter fix replacing generation-2 plan.md's
    rejected ``_phase_reclaim_staged``):

    - neither present: nothing of ours here at all -- a true no-op, so
      the ordinary happy path never even calls ``os.rename``.
    - remnant only, ``force=False``: the defect condition
      :func:`_target_is_absent` already refused to fast-path -- raise
      here too, reusing the existing vocabulary, rather than silently
      restoring (generation-2's rejected design) or destroying it.
    - remnant only, ``force=True``: force authorises destroying it --
      clear it via the delete ladder and return (nothing else to stage).
    - original present, remnant also present (a same-path recreation
      collision): pre-clean the stale remnant via the delete ladder
      before staging again.
    - original present, no remnant: the ordinary case.
    """
    record = ctx.record
    staged = record.path + _STAGED_SUFFIX
    original_present = os.path.exists(record.path)
    remnant_present = os.path.isdir(staged) and bool(os.listdir(staged))

    if not original_present and not remnant_present:
        return

    if not original_present and remnant_present:
        if ctx.force:
            _delete_tree_ladder(staged, deadline=ctx.failure_deadline)
        else:
            raise DirtyWorktreeError(record.id, staged=True)
        return

    if remnant_present:
        _logger.warning(
            "_teardown: worktree '%s' a stale '%s.removing' remnant was "
            "found alongside the checkout; clearing it before staging.",
            record.id, record.id,
        )
        _delete_tree_ladder(staged, deadline=ctx.failure_deadline)

    def _rename() -> None:
        os.rename(record.path, staged)

    if not _retry_bounded(_rename, ctx=ctx):
        _diagnose_and_retry(ctx, trigger="rename", retry=_rename)
    if not os.path.exists(staged):
        # The rename never resolved -- nothing of ours is at the staged
        # path to delete. The final guard reports the outcome.
        return

    _delete_tree_ladder(staged, deadline=ctx.failure_deadline)

    def _residual_gone() -> None:
        if os.path.isdir(staged) and os.listdir(staged):
            raise OSError(f"residual content remains under {staged!r}")

    try:
        _residual_gone()
    except OSError:
        if not _retry_bounded(_residual_gone, ctx=ctx):
            _diagnose_and_retry(ctx, trigger="residual", retry=_residual_gone)


def _phase_git_prune(ctx: _TeardownContext) -> None:
    """Unconditional ``git worktree prune --expire=now`` (ticket #154, item
    1), so an orphaned record's stale git registration is cleared too --
    even on the target-absent fast path. Warn-and-continue on failure: by
    this phase the checkout is genuinely gone from disk (or never
    existed), only git's bookkeeping may be stale, and that staleness is
    self-healing via ``WorktreeManager.prune()`` -- a raise here would
    turn a completed filesystem removal into a caller-visible failure with
    no useful retry.
    """
    record = ctx.record
    try:
        result = _run_git(
            ["worktree", "prune", "--expire=now"], cwd=Path(record.repo_root)
        )
        failed = result.returncode != 0
        detail = result.stderr if failed else None
    except Exception as exc:  # noqa: BLE001
        failed = True
        detail = str(exc)
    if failed:
        _logger.warning(
            "_teardown: worktree '%s' git worktree prune failed for "
            "'.git/worktrees/%s' -- stale git bookkeeping will self-heal "
            "on a later prune() call: %s",
            record.id, record.id, detail,
        )


def _phase_final_guard(ctx: _TeardownContext) -> None:
    """Final guard: the sole raise site for a :func:`_phase_stage_and_delete`
    failure that :func:`_diagnose_and_retry` could not resolve, using the
    literal on-disk truth (ticket #154, item 2).

    ``staged = os.path.exists(record.path + _STAGED_SUFFIX)`` -- true
    whenever the remnant exists, including when ``record.path`` also
    exists (the remnant is the more surprising fact and the one the
    operator needs named):

    - remnant absent, ``record.path`` present -> ``staged=False`` ->
      today's message, byte-identical, selected by ``kill_attempted``
      exactly as before.
    - remnant present (either sub-case) -> ``staged=True`` -> the new
      phrasing naming the marker suffix and the worktree id, no
      filesystem path.

    Either case: ERROR log + ``WorktreeDirLockedError(..., blockers=
    list(ctx.blockers))``. ``status`` is never ``"removed"`` on this
    branch and ports are not released (raises before
    :func:`_phase_release_ports`).
    """
    record = ctx.record
    staged_path = record.path + _STAGED_SUFFIX
    staged = os.path.isdir(staged_path) and bool(os.listdir(staged_path))
    original_present = os.path.exists(record.path)
    if not staged and not original_present:
        return
    _logger.error(
        "_teardown: worktree '%s' could not be removed -- %s.",
        record.id,
        f"checkout is staged under '{record.id}.removing' and the marker "
        f"directory is still held" if staged else "directory is still present",
    )
    raise WorktreeDirLockedError(
        record.id,
        killed=record.killed_pids,
        kill_attempted=ctx.kill_attempted,
        staged=staged,
        blockers=list(ctx.blockers),
    )


def _phase_release_ports(ctx: _TeardownContext) -> None:
    """Step 5: release allocated ports only after the git worktree remove
    has succeeded. Freeing ports before the remove would allow a
    concurrent allocate() to reissue the same ports while the original
    service is still bound to them.
    """
    ctx.allocator.release(ctx.record.id)


_TEARDOWN_PHASES = (
    _phase_guard_primary,
    _phase_stop_processes,
    _phase_stop_hook,
    _phase_gate_b_early_dirty,
    _phase_run_teardown_steps,
    _phase_dirt_gate,
    _phase_stage_and_delete,
    _phase_git_prune,
    _phase_final_guard,
    _phase_release_ports,
)


def run_teardown(ctx: _TeardownContext) -> None:
    """Execute :data:`_TEARDOWN_PHASES` in order, aborting on the first
    phase that raises (ticket #135, R1).

    Branch deletion is intentionally *not* part of this sequence — it
    happens in ``WorktreeManager.remove()`` after the state record has
    been cleaned up, so that a branch-delete failure cannot leave a stale
    orphaned state entry.
    """
    for phase in _TEARDOWN_PHASES:
        phase(ctx)
