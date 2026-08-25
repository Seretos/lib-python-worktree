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
import subprocess
import sys
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
    OrphanScanEntry,
    OrphanScanReport,
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

# Retry constants for the post-kill directory-unlock loop (ticket #51).
_POST_KILL_RETRIES: int = 5      # attempts after kill before giving up
_POST_KILL_SLEEP: float = 0.5    # seconds to wait between retries

# Settle-window constants for Gate A's confirmed-blocker pre-flight check
# (ticket #117): a foreign (non-owned) process found by the initial scan is
# re-scanned after a brief wait, and only a pid present in BOTH scans is
# treated as a genuine blocker -- see _phase_gate_a_blocking_preflight for
# the full rationale.
_PREFLIGHT_SETTLE_SLEEP: float = 1.0     # seconds to wait before the re-scan
_PREFLIGHT_SETTLE_RETRIES: int = 1       # number of settle re-scans

# Ticket #148: tags a _find_blocking_processes result may carry
# (result.skipped_passes) that mean "this scan did not genuinely look" --
# as opposed to "looked and found nothing" -- and must therefore never be
# treated as proof that a previously observed foreign blocker went away.
# "open_files:degraded" (ticket #117) was the original, sole member; ticket
# #148 adds "handle_scan:capped" (the process-wide wedged-worker cap was
# hit -- the handle-table scan refused before ever dumping the table) and
# "handle_scan:busy" (the scan-level lock was contended -- another scan was
# already running, so this one gave up immediately without looking either).
# Deliberately excludes "handle_scan:truncated" (this scan's own deadline
# expired mid-enumeration -- it DID genuinely look, just ran out of time,
# which is a materially weaker signal than never having looked at all) and
# "handle_scan:masked_deferred_capped" (a transient, per-scan degradation
# in the GrantedAccess deferred-resolution pre-filter -- the scan still
# looked at everything else) -- see _scan_coverage_blind's docstring.
_BLIND_SCAN_TAGS: Tuple[str, ...] = (
    "open_files:degraded",
    "handle_scan:capped",
    "handle_scan:busy",
)


def _scan_coverage_blind(scan) -> Optional[str]:
    """Return the first tag in *scan*'s ``skipped_passes`` that means "this
    scan never genuinely looked" (ticket #148), or ``None`` if none apply.

    See ``_BLIND_SCAN_TAGS`` for the exact set and the rationale for what is
    -- and is deliberately NOT -- included. Used at both Gate A sites: the
    pre-flight warning (log-only, never a refusal on its own) and the
    confirming settle-window rescan guard (load-bearing -- a blind rescan
    must not be allowed to clear a pending, previously observed foreign
    blocker).
    """
    for tag in getattr(scan, "skipped_passes", ()):
        if tag in _BLIND_SCAN_TAGS:
            return tag
    return None


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


def _contract_copy_dirt_paths(
    record: "WorktreeRecord", *, entries: Optional[List[str]] = None
) -> Optional[List[str]]:
    """Return the untracked path(s) iff *record*'s checkout's only "dirt"
    (per ``git status``) lives under the ``.seretos/`` convenience copy
    (ticket #100); ``None`` (falsy, same as an empty list) otherwise.

    The ``agent-worktree`` plugin copies the whole ``.seretos/`` directory
    into every new checkout after ``create()`` returns -- purely a
    convenience so the checkout has its own readable copy of the contract
    that created it. Everything under that directory is untracked, so its
    mere presence makes a plain ``git worktree remove`` (``force=False``)
    refuse with "contains modified or untracked files" even when the agent
    made zero real edits. This helper lets teardown recognise that
    specific, benign case and auto-escalate to ``--force`` rather than
    surfacing ``DirtyWorktreeError`` on every single removal.

    This intentionally exempts **any** untracked content under ``.seretos/``,
    not merely the copied contract file -- the plugin copies the directory,
    not a single file, and an agent may itself drop other untracked material
    there (notes, caches, ...). That is a deliberate, documented trade-off:
    such content is discarded here exactly as an explicit ``force=True``
    would discard it. The caller is expected to warning-log the returned
    paths so the discard is observable rather than silent.

    Reimplemented (ticket #103) on top of :func:`_status_entries` and
    :func:`_is_exempt_untracked`; semantics and the "empty/inconclusive
    status ⇒ ``None``" rule are unchanged. Returns the matched paths only
    when **every** reported entry is exempt untracked ``.seretos/`` dirt;
    any other entry anywhere, or an inconclusive probe, returns ``None`` so
    the ordinary ``DirtyWorktreeError`` path is unaffected.

    *entries* is an optional pre-fetched result of :func:`_status_entries`
    (ticket #103) -- the teardown context passes its memoised probe result
    here so a single ``git status`` call can serve both this classifier and
    :func:`_real_dirt_paths` within one removal attempt. Defaults to
    ``None``, which fetches fresh exactly as before -- every existing
    caller/test that calls ``_contract_copy_dirt_paths(record)`` with no
    second argument is unaffected.
    """
    if entries is None:
        entries = _status_entries(record)
    if entries is None:
        return None

    matched: List[str] = []
    for entry in entries:
        if not _is_exempt_untracked(entry):
            return None
        matched.append(entry[3:])
    return matched


def _real_dirt_paths(
    record: "WorktreeRecord", *, entries: Optional[List[str]] = None
) -> List[str]:
    """Return the non-exempt (real) dirty paths for *record*'s checkout, or
    ``[]`` when clean, when the only dirt is the benign ``.seretos/``
    convenience copy (ticket #100), or when the ``git status`` probe is
    inconclusive (ticket #103).

    This is the "is there any real dirt that would block removal?"
    predicate consumed by the combined lock+dirt reporting added by ticket
    #103 -- unlike :func:`_contract_copy_dirt_paths`, which answers "is
    *all* dirt benign?" and returns ``None`` for clean, real-dirt, AND
    inconclusive alike, this always returns a list and never invents a
    blocking condition from an inconclusive probe: an empty result here
    means "do not report a dirty-tree condition", not "confirmed clean".

    *entries*: see :func:`_contract_copy_dirt_paths` -- same optional
    pre-fetched-result seam, used by the teardown context's memoised probe.
    """
    if entries is None:
        entries = _status_entries(record)
    if entries is None:
        return []
    return [entry[3:] for entry in entries if not _is_exempt_untracked(entry)]


def _is_lock_signal(stderr: str) -> bool:
    """Classify a failed ``git worktree remove`` attempt's stderr as an
    OS-level directory-lock signal (ticket #72, Befund 2) rather than some
    other git failure.

    Windows: ``"Permission denied"`` or ``"Invalid argument"`` in stderr
    (both are NTFS/Win32 delete-failure strings). No exit-code requirement
    -- real-world Win32 lock failures have been observed on exit codes other
    than 255.

    POSIX: ``"lock"`` in stderr (case-insensitive) -- git worktree reports a
    held directory as "locked" / "unable to lock" / "cannot lock" /
    "worktree is locked". Unrelated git failures (broken metadata, network
    FS errors, "not a git repository", etc.) match neither arm.

    Shared by both call sites in the git-worktree-remove phase that can hit
    a genuine directory lock: the primary ``git worktree remove`` attempt,
    and the ``.seretos/``-exemption auto-force retry (ticket #100) -- a lock
    can just as easily be held during the retry (e.g. an AV scanner grabs a
    handle on a file under ``.seretos/`` exactly when the forced delete
    runs), and it must be classified identically at either site.
    """
    return (
        sys.platform == "win32"
        and ("Permission denied" in stderr or "Invalid argument" in stderr)
    ) or (
        sys.platform != "win32"
        and "lock" in stderr.lower()
    )


def _merge_killed_pids(
    record: "WorktreeRecord", new_entries: Iterable["KilledProcessInfo"]
) -> None:
    """First-wins, pid-deduped, in-place merge of *new_entries* into
    ``record.killed_pids`` (ticket #140, R7).

    Two plain-assignment sites (``_resolve_lock_or_raise`` and Gate A) used
    to overwrite ``record.killed_pids`` outright on every kill, silently
    dropping whatever an earlier phase of the SAME removal attempt (or an
    earlier ``stop()`` call) had already recorded there. This mirrors
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


def _resolve_lock_or_raise(
    args: List[str],
    record: "WorktreeRecord",
    kill_blocking_processes: bool,
    phantom_cleanup,
    *,
    dirt_probe,
) -> bool:
    """Apply the kill-and-retry directory-lock remedy (ticket #51/#72) for a
    ``git worktree remove`` invocation (*args*) whose failure was already
    classified as a lock signal by :func:`_is_lock_signal`, or raise
    :class:`WorktreeDirLockedError` (single condition) /
    :class:`WorktreeRemovalBlockedError` (lock AND real dirt, ticket #103).

    Shared by both lock-signal call sites: the primary removal attempt and
    the ``.seretos/``-exemption auto-force retry (ticket #100) -- both need
    the SAME remedy instead of the retry site leaking git internals via a
    bare ``GitCommandError``.

    - ``kill_blocking_processes=False``: no remedy was opted into. Raises
      immediately, naming ``kill_blocking_processes=True`` as the way to
      retry (``kill_attempted=False``). No kill or retry is attempted.
    - ``kill_blocking_processes=True``: kills the blocking processes
      (recording ``record.killed_pids``) and retries *args* up to
      ``_POST_KILL_RETRIES`` times, sleeping ``_POST_KILL_SLEEP`` seconds
      between attempts. A retry that fails with "is not a working tree"
      (git deregistered the worktree mid-retry -- phantom-state) is treated
      as success via *phantom_cleanup*, a zero-arg callable that removes the
      leftover directory and prunes stale git metadata. Raises on exhausted
      retries (see below).

    *dirt_probe* is a zero-arg callable (the teardown context's memoised,
    force-gated dirt probe -- see :func:`_real_dirt_paths`) that returns the
    non-exempt dirty paths, or ``[]`` when clean/benign/inconclusive/
    ``force=True``. At each raise site, ``bool(dirt_probe())`` decides
    between the single-condition ``WorktreeDirLockedError`` (no real dirt)
    and the combined ``WorktreeRemovalBlockedError`` (real dirt too) -- both
    name every currently-blocking condition and remedy in one raise so a
    single informed retry suffices.

    Returns ``True`` (the caller's ``kill_attempted`` flag) on any
    non-raising exit; the caller falls through to the long-path fallback /
    the git-worktree-remove phase exactly as the ordinary success path does.
    """
    if not kill_blocking_processes:
        dirt = dirt_probe()
        if dirt:
            raise WorktreeRemovalBlockedError(
                record.id, killed=[], kill_attempted=False, dirty_paths=dirt
            )
        raise WorktreeDirLockedError(record.id, killed=[], kill_attempted=False)

    killed = _kill_blocking_processes(record.path)
    _merge_killed_pids(record, killed)
    retry_result = None
    phantom_on_retry = False
    for attempt in range(_POST_KILL_RETRIES):
        retry_result = _run_git(args, cwd=Path(record.repo_root))
        if retry_result.returncode == 0:
            break
        if (
            retry_result.returncode == 128
            and "is not a working tree" in retry_result.stderr
        ):
            phantom_cleanup()
            phantom_on_retry = True
            break
        if attempt < _POST_KILL_RETRIES - 1:
            time.sleep(_POST_KILL_SLEEP)
    if not phantom_on_retry and (retry_result is None or retry_result.returncode != 0):
        dirt = dirt_probe()
        if dirt:
            raise WorktreeRemovalBlockedError(
                record.id, killed=killed, kill_attempted=True, dirty_paths=dirt
            )
        raise WorktreeDirLockedError(record.id, killed=killed)
    return True


def _target_is_absent(record: "WorktreeRecord") -> bool:
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
    """
    return not os.path.exists(record.path)


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
        need to know about real dirt -- Gate A's confirmed-blocker dirt
        check, Gate B's early refusal, both ``_resolve_lock_or_raise`` raise
        sites, AND the ``.seretos/``-exemption classifier in the
        git-worktree-remove phase -- so that a single removal attempt never
        issues more than one ``git status`` call no matter how many of
        those sites run, and never runs at all when ``force=True`` (real
        dirt cannot block a forced removal, so there is nothing to report).

        Two-phase intent (ticket #117 fix cycle, blocking finding): this
        memoisation is deliberately scoped to TWO phases, not one shared
        snapshot for the whole removal. Phase 1 (pre-teardown) is Gate A
        and Gate B -- both run, by construction, before the ``teardown:``
        steps phase, and MUST see the pre-teardown tree. Phase 2
        (post-teardown) is every consumer that can only run AFTER the
        ``teardown:`` steps phase has executed: the ``.seretos/``-exemption
        classifier and both ``_resolve_lock_or_raise`` ``dirt_probe``
        arguments. Those must NOT reuse Gate B's pre-teardown snapshot -- a
        ``teardown:`` step can itself leave real, non-exempt dirt behind,
        and a stale snapshot would make the exemption classifier believe
        that new dirt is the old, already-cleared ``.seretos/``-only dirt,
        silently force-discarding it instead of raising
        ``DirtyWorktreeError``. The teardown: steps phase explicitly calls
        :meth:`invalidate_dirt` right after the steps run (and only then --
        when there were none to run, nothing changed on disk, so the single
        pre-teardown probe still correctly serves any Phase-2 consumer too,
        preserving the zero-extra-``git status``-calls happy path). Do NOT
        collapse this back into one unconditional snapshot for the whole
        removal.
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

    def phantom_cleanup(self) -> None:
        """Remove the leftover directory and prune stale git metadata.

        Called when git reports 'is not a working tree' — meaning it has
        already deregistered the worktree from its internal registry on a
        prior attempt, but the directory and YAML state record were never
        cleaned up. Both operations are best-effort; errors are swallowed
        so that port release and state removal still occur.
        """
        shutil.rmtree(self.record.path, ignore_errors=True)
        _run_git(["worktree", "prune"], cwd=Path(self.record.repo_root))


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

    target_absent = _target_is_absent(record)
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


def _phase_gate_a_blocking_preflight(ctx: _TeardownContext) -> None:
    """Gate A (Windows only, ticket #76, rewritten for #117): pre-flight
    blocking-process check BEFORE the destructive ``git worktree remove``
    phase below.

    Root cause this guards against: on Windows, ``git worktree remove
    --force`` can return exit 0 while still leaving a content-less locked
    directory behind — an open file handle from a blocking process blocks
    only the *final* directory removal, not the individual file unlinks.
    That falls past the lock-detection branch in the git-worktree-remove
    phase (which only triggers on ``returncode != 0``) into the Final
    guard, which used to raise ``WorktreeDirLockedError`` with no way to
    know a kill was never attempted — and by then the destructive partial
    removal has already happened, so no later retry (even with
    ``kill_blocking_processes=True``) can ever recover: the files are gone
    but the directory is still locked.

    Running this check first means an errored removal has no destructive
    side effect, and a later ``kill_blocking_processes=True`` retry can
    still find and kill the still-alive blocker.

    Placed after the contract ``stop:`` steps (so daemons that release
    handles during that step are not falsely flagged) but before the git
    call. POSIX is intentionally excluded: POSIX unlinks files even under
    an open handle, so ``git worktree remove`` succeeds there — a naive
    pre-flight applied to POSIX would incorrectly block/kill on a merely
    cwd'd process rather than a genuine locker.

    Ticket #127: skipped entirely when ``ctx.target_absent`` -- there is
    nothing on disk that could hold a Windows file handle.

    Ticket #117: rewritten as a confirmed-blocker gate:
      - a hit whose pid is one WE tracked (``ctx.owned_pids``, captured at
        context-build time, before Step 1 stopped them) is trusted
        immediately, no delay -- that is precisely the "our own daemon is
        still shutting down" case, and killing it is the correct remedy;
      - a hit whose pid is NOT ours (a foreign/transient process -- AV
        scanner, indexer, editor) goes through a bounded settle window:
        wait briefly, then re-scan; only a pid present in BOTH scans is
        treated as a genuine, confirmed blocker (AC #4). A process that
        merely happened to touch the path at the instant of the first scan
        and is gone by the second is never reported as blocking;
      - ``"open_files:degraded"`` is logged (reduced coverage this attempt)
        but is never, by itself, a refusal condition (Q2 Option A) -- it
        can still contribute to a confirmation via the owned/foreign logic
        above, and the Final guard after the destructive git call remains
        the backstop for an actually-locked directory this degraded scan
        could not see coming (ticket #121: a degraded/partial scan result
        never blocks removal on its own).

    Ticket #140: both scan calls below (initial pre-flight and the settle-
    window confirm rescan) and the confirmed-blocker kill call further down
    are individually guarded against an unexpected exception (e.g. the bare
    ``RuntimeError`` ticket #107 documents) -- degrading to "no confirmed
    blocker"/"kill not confirmed" for THIS gate rather than propagating and
    aborting the whole removal before the new all-platform orphan-scan
    phase (index 6) gets a chance to run and report the same condition.
    """
    record = ctx.record
    if sys.platform == "win32" and not ctx.target_absent:
        try:
            _preflight_scan = _find_blocking_processes(record.path, os.getpid())
        except Exception as _scan_exc:  # noqa: BLE001 -- ticket #107/#140
            _logger.warning(
                "_teardown: worktree '%s' Gate A blocking-process scan "
                "failed unexpectedly (%s); treating as no confirmed "
                "blocker for this attempt.",
                record.id, _scan_exc,
            )
            _preflight_scan = []
        # Ticket #148: log-only pre-flight warning, generalized from the
        # ticket #117 "open_files:degraded"-only check to any tag in
        # _BLIND_SCAN_TAGS -- see _scan_coverage_blind's docstring. Never a
        # refusal on its own (Q2 Option A, unchanged): it can still
        # contribute to a confirmation via the owned/foreign logic below,
        # and the Final guard after the destructive git call remains the
        # backstop for an actually-locked directory this blind scan could
        # not see coming.
        _preflight_blind_tag = _scan_coverage_blind(_preflight_scan)
        if _preflight_blind_tag is not None:
            _logger.warning(
                "_teardown: worktree '%s' blocking-process scan did not "
                "genuinely look (%s) — reduced coverage for this attempt; "
                "not treated as a blocker on its own (ticket #117/#148).",
                record.id, _preflight_blind_tag,
            )

        _owned = [info for info in _preflight_scan if info.pid in ctx.owned_pids]
        _foreign = [info for info in _preflight_scan if info.pid not in ctx.owned_pids]
        _confirmed = list(_owned)

        if _foreign:
            # Bounded settle window (ticket #117, AC #4). A transient
            # foreign process must not permanently hard-block removal --
            # only a pid still present on the confirming re-scan counts.
            for _ in range(_PREFLIGHT_SETTLE_RETRIES):
                time.sleep(_PREFLIGHT_SETTLE_SLEEP)
                try:
                    _confirm_scan = _find_blocking_processes(record.path, os.getpid())
                except Exception as _scan_exc:  # noqa: BLE001 -- ticket #107/#140
                    # Same conservative treatment as a degraded confirming
                    # rescan just below: an unexpected failure here tells us
                    # nothing about whether the pending foreign hit(s) are
                    # still there, so do NOT clear them.
                    _logger.warning(
                        "_teardown: worktree '%s' Gate A confirming rescan "
                        "failed unexpectedly (%s); conservatively treating "
                        "%d pending foreign hit(s) as still-blocking rather "
                        "than clearing them.",
                        record.id, _scan_exc, len(_foreign),
                    )
                    break
                _confirm_blind_tag = _scan_coverage_blind(_confirm_scan)
                if _confirm_blind_tag is not None:
                    # Ticket #117 fix cycle (blocking finding 1), generalized
                    # by ticket #148 from "open_files:degraded" alone to any
                    # tag in _BLIND_SCAN_TAGS (see _scan_coverage_blind): a
                    # BLIND CONFIRMING rescan returns an empty/near-empty
                    # result because the scan never actually looked at
                    # anything -- not because the foreign blocker went away.
                    # This is true whether the scan degraded
                    # (open_files:degraded), was refused outright by the
                    # process-wide wedged-worker cap (handle_scan:capped),
                    # or gave up immediately on scan-lock contention
                    # (handle_scan:busy) -- none of these are evidence the
                    # foreign hit cleared. Intersecting a blind result with
                    # the first scan's hits would silently clear a real
                    # blocker and fall straight through to the destructive
                    # `git worktree remove` call, reproducing the exact
                    # locked-directory failure this ticket exists to
                    # prevent. Do NOT let a blind rescan clear a previously
                    # observed foreign hit: stop settling and conservatively
                    # treat the still-pending `_foreign` hits from the last
                    # real (non-blind) scan as confirmed, rather than as
                    # disproven. This is deliberately narrower than a
                    # blanket blind-scan refusal: it only fires when a real
                    # foreign hit is already pending confirmation. A blind
                    # scan with NO prior hit never enters this `if
                    # _foreign:` branch at all, so AC #1's de-escalation (a
                    # degraded scan alone must not block removal) is
                    # unaffected.
                    _logger.warning(
                        "_teardown: worktree '%s' confirming rescan did not "
                        "genuinely look (%s) while %d foreign hit(s) were "
                        "pending confirmation; treating as still-blocking "
                        "rather than clearing.",
                        record.id,
                        _confirm_blind_tag,
                        len(_foreign),
                    )
                    break
                _rescan_pids = {info.pid for info in _confirm_scan}
                _foreign = [info for info in _foreign if info.pid in _rescan_pids]
                if not _foreign:
                    break
            if _foreign:
                _confirmed.extend(_foreign)
            else:
                _logger.debug(
                    "_teardown: worktree '%s' had transient blocking "
                    "process(es) settle during the pre-flight check; "
                    "proceeding without refusal.",
                    record.id,
                )

        if _confirmed:
            _logger.warning(
                "_teardown: worktree '%s' blocked by %d confirmed "
                "process(es): %s",
                record.id,
                len(_confirmed),
                ", ".join(
                    "pid=%d name=%r cmd=%r"
                    % (
                        info.pid,
                        info.name,
                        info.cmdline[0] if info.cmdline else "",
                    )
                    for info in _confirmed
                ),
            )
            # Q2 (ticket #103): probe for real dirt BEFORE taking any
            # irreversible action -- including the kill below. If real dirt
            # is ALSO present, this removal cannot succeed on this attempt
            # no matter what kill_blocking_processes says (git worktree
            # remove will still refuse for being dirty), so report both
            # conditions now, with nothing killed and nothing removed,
            # rather than killing a process for a removal that was always
            # going to fail.
            _dirt = ctx.dirt()
            if _dirt:
                raise WorktreeRemovalBlockedError(
                    record.id, killed=[], kill_attempted=False, dirty_paths=_dirt
                )
            if ctx.kill_blocking_processes:
                # Ticket #140: kill_attempted is set BEFORE the call (not
                # after, as before this ticket) so an attempt that raises
                # mid-way is still recorded honestly -- mirrors the new
                # orphan-scan phase's own convention.
                ctx.kill_attempted = True
                try:
                    killed = _kill_blocking_processes(record.path)
                except Exception as _kill_exc:  # noqa: BLE001 -- ticket #107/#140
                    # An unexpected kill failure must not block removal --
                    # the new orphan-scan phase (index 6) makes its own
                    # kill attempt and reports the degradation there.
                    _logger.warning(
                        "_teardown: worktree '%s' Gate A kill attempt "
                        "failed unexpectedly (%s); proceeding without "
                        "confirming a kill.",
                        record.id, _kill_exc,
                    )
                else:
                    _merge_killed_pids(record, killed)
                    if not killed:
                        # Ticket #148 scope note: this third site -- the
                        # post-kill check on `_kill_blocking_processes`'s
                        # own return value -- is deliberately LEFT
                        # UNCHANGED (still "open_files:degraded" only, not
                        # `_scan_coverage_blind`/`_BLIND_SCAN_TAGS`). The
                        # ticket's Gate A scope is the two sites above only
                        # (pre-flight warning and the confirming-rescan
                        # guard); widening this site too was considered and
                        # explicitly deferred, not overlooked.
                        if "open_files:degraded" in getattr(
                            killed, "skipped_passes", ()
                        ):
                            # Ticket #117 fix cycle (blocking finding 2):
                            # `_kill_blocking_processes` returns a
                            # `_PartialList` -- a `list` subclass -- so an empty
                            # `killed` is falsy-equal both when the blocker
                            # genuinely exited before the kill ran AND when the
                            # kill's own internal rescan was degraded and
                            # therefore never actually looked for anything to
                            # kill. `not killed` alone cannot tell those apart.
                            # Restore the pre-#117 guard for the degraded case:
                            # a still-live, twice-confirmed blocker (Gate A
                            # above already confirmed it via two scans) must not
                            # be silently treated as cleared just because the
                            # kill's rescan happened to be blind. Raise rather
                            # than fall through to the destructive git remove
                            # call.
                            raise WorktreeDirLockedError(
                                record.id, killed=list(killed), kill_attempted=True
                            )
                        # Ticket #117, AC #2: kill_blocking_processes was opted
                        # into, the blocker was twice-confirmed live above, and
                        # the kill's own rescan was NOT degraded (handled
                        # above) -- so an empty result here means it genuinely
                        # exited in the interval between confirmation and the
                        # kill call, not that nothing was ever really there.
                        # Log and proceed rather than raise (the pre-#117
                        # behaviour treated this the same as a still-degraded
                        # rescan and always refused, which is what made
                        # kill_blocking_processes=True + empty killed_pids the
                        # normal, misleading outcome this ticket fixes).
                        _logger.warning(
                            "_teardown: worktree '%s' kill_blocking_processes "
                            "found nothing left to kill (confirmed "
                            "blocker(s) exited before the kill ran); "
                            "proceeding.",
                            record.id,
                        )
                # Fall through to the (now-safe) git remove phase below.
            else:
                # Caller did not opt into the kill-and-retry remedy: raise
                # immediately, before the destructive git call runs, so no
                # partial removal can occur on this attempt.
                raise WorktreeDirLockedError(
                    record.id, killed=[], kill_attempted=False
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


def _phase_orphan_scan(ctx: _TeardownContext) -> None:
    """Phase (index 6, ticket #140): all-platform, warn-only scan for a
    process holding *record.path* (as cwd, cmdline token, Windows handle,
    or open file) that was never tracked by this engine -- previously
    silently orphaned once ``git worktree remove`` succeeded around it.

    Runs on EVERY platform (unlike win32-only Gate A), AFTER the contract
    ``teardown:`` steps have run but BEFORE the destructive
    ``git worktree remove`` phase -- a fresh, full-budget
    (``_find_blocking_processes(record.path, os.getpid())``, no
    ``deadline``) scan, deliberately not a reuse of Gate A's own result:
    Gate A never runs on POSIX or against an absent target, and the
    ``teardown:`` steps that just ran may have changed the picture. On
    win32 with a present target this means a removal now pays two full
    scans -- an accepted, documented cost.

    No-ops (``record.orphan_scan`` stays ``None``) when ``ctx.target_absent``
    -- there is nothing on disk to hold a handle on -- mirroring Gate A's
    own ticket #127 skip.

    ``_kill_blocking_processes(record.path)`` (the SAME reused remedy Gate A
    and ``_resolve_lock_or_raise`` use -- no hand-rolled signal/wait logic)
    is called only when ``ctx.kill_blocking_processes`` AND the scan found
    at least one hit -- a clean scan skips the ~5s call entirely even with
    the flag set. This is what makes ``kill_blocking_processes=True``
    meaningful on POSIX for the first time (ticket #140's second goal).
    ``ctx.kill_attempted`` is set immediately before that call so an
    attempt that raises mid-way is still recorded honestly. Only the
    entries the kill call actually confirms killed are merged into
    ``record.killed_pids`` (via :func:`_merge_killed_pids` -- first-wins,
    pid-deduped, never overwriting an earlier phase's kills).

    ``record.orphan_scan`` (an :class:`~.state.OrphanScanReport`) is the
    **union** of the warn scan and the kill result, first-wins pid-deduped,
    preserving scan order then kill-only order: a pid the scan found but the
    kill's own tighter rescan did not confirm is reported ``killed=False``;
    a pid the scan found AND the kill confirmed is upgraded to
    ``killed=True``; a lineage-only pid the kill's ``_process_tree``
    expansion added (never seen by the scan itself) is reported
    ``killed=True`` with ``owned`` computed the same way as any other entry
    (``ctx.owned_pids`` membership). ``owned`` reflects Gate A's own
    owned/foreign distinction, not a NEW classification.

    **Warn-only, hard invariant:** this phase never raises and never adds a
    new refusal condition or a new ``force`` requirement. The entire body
    (scan AND kill) is wrapped in a single ``except Exception`` that logs
    once, appends the synthetic ``"scan:failed"`` marker to
    ``skipped_passes``, and proceeds with whatever partial results were
    already collected before the failure -- this is what protects against
    ``_find_blocking_processes`` (or the reused kill call) re-raising a bare
    ``RuntimeError`` (ticket #107) and turning a working removal into a
    failure.

    ``record.orphan_scan`` is assigned exactly once per invocation (to
    ``None`` for a clean, complete scan with no hits and no incompleteness
    to report, or to a populated :class:`~.state.OrphanScanReport`
    otherwise) so a stale report from an earlier failed attempt can never
    leak forward. Exactly one ``_logger.warning(...)`` is emitted for an
    abnormal outcome, with the identical string also stored as the report's
    ``message`` (message-parity convention, matching
    ``StopDetail``/``BaseFetchFallback``).

    **Kill-target caveat:** with ``kill_blocking_processes=True``, a
    heuristic hit from Pass 1b (``cmdline``), Pass 1c (``handle_scan``), or
    Pass 2 (``open_files``) -- not just a Pass 1 exact-cwd match -- becomes
    a KILL TARGET, not merely a warning: an editor or AV scanner that merely
    holds a handle or an open file under the checkout can be terminated by
    an opt-in caller. See ``WorktreeManager.remove()``'s docstring for the
    same caveat stated at the public API surface.
    """
    record = ctx.record
    record.orphan_scan = None
    if ctx.target_absent:
        return

    entries: List[OrphanScanEntry] = []
    skipped_passes: List[str] = []
    kill_attempted = False
    fail_reason: Optional[str] = None

    try:
        scan_result = _find_blocking_processes(record.path, os.getpid())
        for _tag in getattr(scan_result, "skipped_passes", ()):
            if _tag not in skipped_passes:
                skipped_passes.append(_tag)

        scan_pid_seen: Set[int] = set()
        for info in scan_result:
            if info.pid in scan_pid_seen:
                continue
            scan_pid_seen.add(info.pid)
            entries.append(
                OrphanScanEntry(
                    info=info, owned=info.pid in ctx.owned_pids, killed=False,
                )
            )

        if ctx.kill_blocking_processes and scan_pid_seen:
            kill_attempted = True
            ctx.kill_attempted = True
            kill_result = _kill_blocking_processes(record.path)
            _merge_killed_pids(record, kill_result)
            for _tag in getattr(kill_result, "skipped_passes", ()):
                if _tag not in skipped_passes:
                    skipped_passes.append(_tag)

            kill_pid_seen: Set[int] = set()
            for info in kill_result:
                if info.pid in kill_pid_seen:
                    continue
                kill_pid_seen.add(info.pid)
                if info.pid in scan_pid_seen:
                    # Upgrade the existing (killed=False) scan entry in
                    # place -- the kill's own tighter rescan confirmed it.
                    for _idx, _existing in enumerate(entries):
                        if _existing.info.pid == info.pid:
                            entries[_idx] = OrphanScanEntry(
                                info=_existing.info,
                                owned=_existing.owned,
                                killed=True,
                            )
                            break
                else:
                    # Lineage-only pid (_process_tree expansion): never seen
                    # by the scan itself.
                    entries.append(
                        OrphanScanEntry(
                            info=info, owned=info.pid in ctx.owned_pids, killed=True,
                        )
                    )
    except Exception as _exc:  # noqa: BLE001 -- ticket #107/#140: never let scan/kill failure block removal; partial results already collected are kept.
        fail_reason = str(_exc)
        if "scan:failed" not in skipped_passes:
            skipped_passes.append("scan:failed")

    if not entries and not skipped_passes:
        # Clean, complete scan: nothing to warn about.
        return

    if fail_reason is not None:
        message = (
            f"_teardown: worktree '{record.id}' orphan scan degraded "
            f"unexpectedly ({fail_reason}); reporting best-effort results "
            f"({len(entries)} process(es) found before the failure)."
        )
    else:
        message = (
            f"_teardown: worktree '{record.id}' orphan scan found "
            f"{len(entries)} untracked process(es) holding the checkout."
        )
    _logger.warning("%s", message)

    record.orphan_scan = OrphanScanReport(
        message=message,
        entries=tuple(entries),
        skipped_passes=tuple(skipped_passes),
        kill_attempted=kill_attempted,
    )


def _phase_git_worktree_remove(ctx: _TeardownContext) -> None:
    """Step 4: remove the git worktree checkout directory."""
    record = ctx.record
    args = ["worktree", "remove"]
    if ctx.force:
        args.append("--force")
    args.append(record.path)
    proc = _run_git(args, cwd=Path(record.repo_root))
    if proc.returncode != 0:
        _triage_remove_failure(ctx, args, proc)


def _triage_remove_failure(ctx: _TeardownContext, args: List[str], proc) -> None:
    """Classify and remedy a non-zero ``git worktree remove`` exit.

    Branch order (preserved verbatim from the pre-#135 monolithic
    ``_teardown``) matters: phantom-state is checked first so it can never
    be shadowed by the widened exit-128 fallback below it; the lock-signal
    classification runs BEFORE the dirty-tree phrase check (ticket #72,
    Befund 1) so a lock-induced failure is never misread as a dirty
    working tree just because git's dirty-tree phrase happens to also be
    present in stderr.
    """
    record = ctx.record
    if proc.returncode == 128 and "is not a working tree" in proc.stderr:
        # git has already deregistered this worktree (phantom-state
        # scenario from ticket #51). Treat as already-gone: clean up the
        # leftover directory and stale metadata, then fall through to port
        # release.
        ctx.phantom_cleanup()
        return

    if proc.returncode == 128 and (
        ctx.force
        or (
            ctx.target_absent
            and "contains modified or untracked files" not in proc.stderr
        )
    ):
        # The .git link is already gone (worktree dir was wiped
        # externally). Fall back: delete the directory ourselves, then
        # prune the stale git metadata. Both steps are best-effort so that
        # port release and state removal still occur even in a degraded
        # state.
        #
        # Ticket #127: this fallback used to require force=True, but an
        # orphaned record (target_absent=True) hits this exact exit-128
        # shape through no fault of the caller -- the directory is gone,
        # git still has a `prunable` admin dir under
        # `.git/worktrees/<id>`, and the stderr matches neither the
        # phantom-state branch above nor a dirty-tree phrase (an absent
        # directory cannot be dirty). Widening the guard to also fire when
        # target_absent is True lets a plain remove(worktree_id=...)
        # recover this case without forcing the caller to pass
        # force=True. When the directory is genuinely present
        # (target_absent is False), this branch stays gated on force
        # exactly as before, and the fall-through GitCommandError below
        # still raises.
        #
        # The explicit dirty-tree-phrase exclusion on the target_absent arm
        # (but NOT on the force arm, which must keep bypassing everything
        # exactly as it did before this ticket) guards against a
        # nonsensical co-occurrence: a directory that does not exist cannot
        # genuinely be dirty, so this can only happen when a caller's own
        # `_run_git` fake asserts both conditions at once (as one
        # pre-existing regression test does, pinning the
        # DirtyWorktreeError message format). Without this exclusion,
        # target_absent (evaluated on a real, non-existent test path)
        # would steal that case away from the nested dirty-tree-phrase
        # check below and silently swallow a refusal the caller needs to
        # see and act on.
        shutil.rmtree(record.path, ignore_errors=True)
        _run_git(["worktree", "prune"], cwd=Path(record.repo_root))
        return

    # Determine whether this exit qualifies as a directory-lock signal (an
    # OS-level lock held by another process on the worktree directory).
    # Classification and remedy are factored into module-level helpers
    # (_is_lock_signal / _resolve_lock_or_raise) so the SAME logic also
    # covers the `.seretos/`-exemption auto-force retry below
    # (ticket #100).
    if _is_lock_signal(proc.stderr):
        # A lock's remedy (kill_blocking_processes=True) is independent of
        # --force, so this branch applies for BOTH force=True and
        # force=False (ticket #72, Befund 2).
        ctx.kill_attempted = _resolve_lock_or_raise(
            args,
            record,
            ctx.kill_blocking_processes,
            ctx.phantom_cleanup,
            dirt_probe=ctx.dirt,
        )
        # Retry succeeded (or phantom-state cleanup ran) — fall through to
        # the long-path fallback phase then the Final guard.
        return

    if (
        proc.returncode == 128
        and not ctx.force
        and "contains modified or untracked files" in proc.stderr
    ):
        # git refused because the worktree has uncommitted changes. Before
        # surfacing DirtyWorktreeError, check whether the ONLY dirt is
        # untracked content under `.seretos/` -- the convenience copy the
        # agent-worktree plugin drops into every checkout after create()
        # returns (ticket #100), or anything else an agent left there.
        # None of that is ever read by start()/setup -- it holds no
        # engine-authoritative state -- yet its mere presence would
        # otherwise force force=True on every single plain
        # create() -> remove() cycle with zero real edits.
        dirt_paths = _contract_copy_dirt_paths(
            record, entries=ctx.status_entries()
        )
        if dirt_paths:
            _seretos_exemption_retry(ctx, args, dirt_paths)
            return
        # Real dirt beyond the benign contract copy. Surface a structured
        # error naming only the engine parameter (force=True), not the raw
        # git command, path, or exit code.
        raise DirtyWorktreeError(record.id)

    raise GitCommandError(["git", *args], proc.returncode, proc.stderr)


def _seretos_exemption_retry(
    ctx: _TeardownContext, args: List[str], dirt_paths: List[str]
) -> None:
    """Auto-escalate to ``--force`` when the ONLY dirty-tree refusal cause
    is the benign ``.seretos/`` convenience copy (ticket #100), then
    classify the retry's own outcome exactly like the primary attempt.

    Deliberate, documented trade-off (ticket #100): this auto-escalation is
    unconditional and exempts ANY untracked content under ``.seretos/``,
    not merely the copied contract file -- an agent's own hand-added
    notes/cache/etc. dropped there are discarded here too, without a
    prompt, exactly as an explicit ``force=True`` would discard them. The
    discard itself is warning-logged below, naming the actual paths, so it
    is observable rather than silent.
    """
    record = ctx.record
    _logger.warning(
        "_teardown: worktree '%s' removed without "
        "force=True; discarding untracked path(s) under "
        "%s/ that were the only dirty-tree refusal "
        "cause: %s",
        record.id, _CONTRACT_DIR, ", ".join(dirt_paths),
    )
    retry_args = ["worktree", "remove", "--force", record.path]
    retry_result = _run_git(retry_args, cwd=Path(record.repo_root))
    if retry_result.returncode != 0:
        if (
            retry_result.returncode == 128
            and "is not a working tree" in retry_result.stderr
        ):
            # git deregistered the worktree between the first refusal and
            # this retry (phantom-state mid-retry). Treat as already-gone
            # and fall through to port release.
            ctx.phantom_cleanup()
            return
        if _is_lock_signal(retry_result.stderr):
            # The retry itself hit a genuine directory lock (e.g. an AV
            # scanner/editor holding a handle on a file under `.seretos/`
            # exactly when the forced delete runs). Route through the SAME
            # lock remedy as the primary removal attempt, rather than
            # falling through to a bare GitCommandError that would leak
            # git internals and skip kill_blocking_processes entirely
            # (ticket #100 follow-up).
            ctx.kill_attempted = _resolve_lock_or_raise(
                retry_args,
                record,
                ctx.kill_blocking_processes,
                ctx.phantom_cleanup,
                dirt_probe=ctx.dirt,
            )
            return
        raise GitCommandError(
            ["git", *retry_args],
            retry_result.returncode,
            retry_result.stderr,
        )
    # Retry succeeded — fall through to the long-path fallback phase then
    # the Final guard, exactly as the ordinary success path does.


def _phase_filesystem_fallback(ctx: _TeardownContext) -> None:
    """Long-path fallback: on Windows, 'git worktree remove' can succeed
    (exit 0) but leave the directory behind when paths exceed MAX_PATH. In
    that case attempt \\\\?\\ prefixed deletion; if that also fails, try
    the robocopy empty-mirror trick. On POSIX, shutil.rmtree is the simple
    fallback.
    """
    record = ctx.record
    if os.path.exists(record.path):
        if sys.platform == "win32":
            extended_path = "\\\\?\\" + os.path.abspath(record.path)
            try:
                shutil.rmtree(extended_path)
            except OSError:
                # Extended-path rmtree failed — try robocopy empty-mirror.
                import tempfile
                try:
                    with tempfile.TemporaryDirectory() as empty_tmp:
                        # Ticket #78: bound this call two independent ways
                        # so a locked directory can never wedge the single
                        # synchronous MCP request thread. /R:1 /W:1
                        # override robocopy's own default of a million
                        # retries at 30s apart; the explicit timeout= is a
                        # second, independent backstop. subprocess.
                        # TimeoutExpired is caught by the broad except
                        # below, so the Final guard's
                        # WorktreeDirLockedError still fires correctly when
                        # this call times out.
                        subprocess.run(
                            ["robocopy", empty_tmp, record.path, "/MIR", "/R:1", "/W:1"],
                            capture_output=True,
                            timeout=_resolve_robocopy_timeout(),
                            # getattr(..., 0), not subprocess.CREATE_NO_WINDOW
                            # directly: this branch is guarded by
                            # `sys.platform == "win32"` above, but tests
                            # (and the real world, under a mocked
                            # sys.platform) can reach this line while the
                            # *real* subprocess module has no such
                            # attribute (e.g. running on Linux CI with
                            # sys.platform patched to "win32" to exercise
                            # this code path). A direct attribute access
                            # would raise AttributeError there, which the
                            # broad `except Exception` below silently
                            # swallows -- skipping the robocopy call
                            # entirely without any indication why.
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                        shutil.rmtree(record.path, ignore_errors=True)
                except Exception:  # noqa: BLE001
                    pass  # Best-effort: don't block port release.
        else:
            shutil.rmtree(record.path, ignore_errors=True)


def _phase_final_guard(ctx: _TeardownContext) -> None:
    """Final guard: if the directory is still present after all deletion
    attempts, the worktree is locked by an external process. Raise rather
    than returning a false status: "removed".
    """
    record = ctx.record
    if os.path.exists(record.path):
        raise WorktreeDirLockedError(
            record.id, killed=record.killed_pids, kill_attempted=ctx.kill_attempted
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
    _phase_gate_a_blocking_preflight,
    _phase_gate_b_early_dirty,
    _phase_run_teardown_steps,
    _phase_orphan_scan,
    _phase_git_worktree_remove,
    _phase_filesystem_fallback,
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
