"""Process lifecycle engine layer (W6/W8 — ticket #8).

Public API
----------
- ``start(worktree_id, cmd, *, store, role="main", env=None, cwd=None)``
  Spawns a detached process, persists ``pids[role]`` and ``status="running"``
  to the state store, returns the updated ``WorktreeRecord``. The captured
  output log is written to
  ``<log_dir_for(worktree_id)>/start-<sanitized role, case preserved>.log``
  (ticket #111): the role is sanitized for filesystem-safety (unsafe
  characters collapsed to ``-``) but never lower-cased, so the filename's
  role token stays identical to the literal ``pids`` dict key -- a caller
  holding a role name can always reconstruct the log path without querying
  the record. Note this is ambiguous on case-insensitive filesystems for
  roles that differ only by case; see ``_role_log_slug``'s docstring.

- ``stop(worktree_id, *, store, role="main", timeout=10.0, kill_orphans=False)``
  Gracefully terminates the process (SIGTERM/CTRL_BREAK), waits up to
  ``timeout`` seconds, then force-kills if still alive.  Also snapshots and
  kills the tracked PID's whole descendant process tree (ticket #87) so a
  grandchild spawned by a nested shell (e.g. a ``run:`` step whose command
  itself invokes another shell) cannot survive by being reparented once the
  tracked PID dies -- this tree kill is unconditional, not gated on
  ``kill_orphans``.  Clears ``pids[role]``; sets ``status="stopped"`` only
  when no other roles remain AND nothing this call tried to kill is still
  alive -- otherwise ``status="stop_incomplete"`` and a warning is logged
  naming the survivor PIDs, so a leaked process is never silently reported
  as stopped.  Ticket #99: whenever ``status`` is set to
  ``"stop_incomplete"``, ``record.stop_detail`` (a
  ``state.StopDetail``) is populated with the same information as the
  warning -- a machine-readable ``reason`` (one of ``state.STOP_REASONS``),
  the formatted ``message``, and reason-specific evidence (survivor PIDs, the
  truncation cap hit, or the orphan scan's skipped passes) -- so a caller
  does not have to parse log lines to learn why.  Returns the updated
  ``WorktreeRecord``.

Platform differences
--------------------
- Windows: ``CREATE_NEW_PROCESS_GROUP`` to detach from the MCP host's
  process group while still allowing ``CTRL_BREAK_EVENT`` delivery;
  ``TerminateProcess`` (via ctypes) for force-kill. Ticket #147:
  ``CTRL_BREAK_EVENT`` targets a console PROCESS GROUP id, not an arbitrary
  pid, so it is only ever sent to a pid the caller has explicitly confirmed
  is the leader of a group WE created -- in practice, only ``stop()``'s own
  tracked pid. Every other pid this module signals (descendant-tree nodes,
  path-heuristic-discovered orphans) falls straight through to the
  wait/force-kill fallback instead. See ``_send_graceful_signal``'s
  docstring. ``start()`` also creates
  a Windows Job Object (ticket #95) and assigns the spawned process to it --
  a ppid-INDEPENDENT containment mechanism that ``stop()`` enumerates and
  terminates as a unit, closing the gap the ppid-derived process-tree walk
  cannot: a ``ShellExecuteEx``-delegated launch (what ``Start-Process`` uses
  without stream redirection) lands outside our ppid lineage entirely, no
  matter how deep the tree walk recurses. See ``_create_job_object``'s
  docstring for why no limit flags (in particular, never
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``) are ever set on it.
- POSIX:   ``start_new_session=True``; ``SIGTERM`` for graceful stop;
  ``SIGKILL`` for force-kill.

The ``_pid_alive`` helper is imported from ``yaml_store`` so there is a
single, tested implementation.

No ``mcp`` imports; returns plain dataclasses (``WorktreeRecord``).
"""

from __future__ import annotations

import base64
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .state import (
    STOP_ATTEMPT_ALREADY_EXITED,
    STOP_ATTEMPT_KILLED,
    STOP_ATTEMPT_TRACKED_PID_MISSING,
    STOP_REASON_HANDLE_SCAN_EXHAUSTED,
    STOP_REASON_JOB_MEMBER_LIST_TRUNCATED,
    STOP_REASON_ORPHAN_SCAN_INCOMPLETE,
    STOP_REASON_SURVIVORS,
    STOP_REASON_TREE_TRUNCATED,
    StateStore,
    StopAttempt,
    StopDetail,
    WorktreeRecord,
    _STOP_DETAIL_MAX_PIDS,
)
from .yaml_store import _pid_alive

_logger = logging.getLogger(__name__)

# The role key used when the caller does not supply an explicit role.
DEFAULT_ROLE = "main"

# Ticket #111: sanitizer for the *log filename's* role component. Unlike
# setup.runner._slug (lower-cases the value), this preserves case so that
# start-<role>.log's role token stays identical to the literal record.pids
# key -- see _role_log_slug's docstring for the full rationale.
_ROLE_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


def _role_log_slug(role: str, max_len: int = 40) -> str:
    """Sanitize *role* for use in the ``start-<role>.log`` filename.

    Case-preserving counterpart to ``setup.runner._slug``: replaces runs of
    filesystem-unsafe characters with a single ``-`` and strips leading/
    trailing ``-``, but -- unlike ``_slug`` -- does NOT lower-case the
    result. This keeps the log filename's role token identical to the raw
    ``role`` string used as the ``record.pids`` dict key (ticket #111: the
    two had drifted because ``start()`` previously slugified the filename
    with ``_slug``, which lower-cases, while ``pids`` is keyed by the raw,
    original-case role).

    Falls back to ``"_"`` (a single underscore -- not ``_slug``'s
    ``"step"``) when *role* contains no alphanumeric characters at all.
    ``"_"`` is deliberate, not arbitrary: ``"_"`` never appears as output of
    the substitution step alone -- that step's output alphabet is exactly
    ``[A-Za-z0-9-]``, since every other character, ``_`` included, is
    collapsed to ``-``. The only way to produce ``"_"`` is the explicit
    degenerate-role fallback below. A role that is itself entirely
    non-alphanumeric (including the literal string ``"_"``) is a degenerate
    role by this function's own definition and is therefore already covered
    by, not exempt from, the "degenerate roles collide with each other"
    caveat further down -- it is not a separate collision class. What this
    does guarantee is narrower but still useful: a *non-degenerate* role
    (one with at least one alphanumeric character) can never sanitize to
    ``"_"``, so the fallback can never collide with a non-degenerate role's
    filename -- an earlier version of this helper fell back to ``"role"``,
    which DID collide with an actual non-degenerate role literally named
    ``"role"``: both produced ``start-role.log`` while ``record.pids`` kept
    them as distinct keys, undermining the very pids-key/filename
    traceability this ticket exists to restore. This is deliberately not a
    hashing/suffixing scheme, which every consumer that reconstructs the log
    path from a role name would have to reimplement.

    Degenerate roles still collide *with each other* (and, as just noted,
    with the fallback token itself): ``"!!!"``, ``"---"``, and the literal
    role ``"_"`` all sanitize to ``"_"`` and therefore all name
    ``start-_.log``. That is the same accepted, documented lossy-
    sanitization ambiguity already noted below for roles like ``"role a"``
    vs ``"role-a"`` vs ``"role_a"`` -- not something this helper tries to
    fix.

    Truncates to *max_len* characters -- mirroring ``_slug``'s 40-char
    default for parity -- and re-strips any trailing ``-`` a truncation cut
    exposes (e.g. ``"a" * 39 + "-" + "b" * 10`` truncates to
    ``"a" * 39 + "-"`` before this second strip, which would otherwise leave
    a filename ending in ``-.log``). Order is: replace -> strip ->
    fallback-if-empty -> truncate -> strip again. This intentionally
    diverges from ``setup.runner._slug``'s ordering (which strips only once,
    before truncating) -- do not "fix" ``_slug`` to match; that module is
    out of this ticket's scope. The final strip can never empty the string:
    after the first strip the leading character is alphanumeric, and the
    fallback token ``"_"`` contains no ``-``.

    Case-insensitive filesystems (Windows, default macOS): this is a pure
    string transform, so two roles that differ only by case (e.g. ``"roleA"``
    and ``"rolea"``) still produce two *distinct filename strings*, but on
    a case-insensitive filesystem those two strings name the *same physical
    file* and their log output will interleave. This is an accepted,
    documented limitation, not a bug: disambiguating case-only-distinct
    roles would require pushing a suffixing/hashing scheme onto every
    consumer that reconstructs the log path from a role name, for an
    ambiguity that already exists today for roles like ``"role a"`` vs
    ``"role-a"`` vs ``"role_a"``. Callers needing guaranteed per-role log
    isolation must pick role names that differ by more than case.
    """
    s = _ROLE_SLUG_RE.sub("-", role).strip("-")
    if not s:
        s = "_"
    s = s[:max_len]
    return s.strip("-")

# Bounded wait, right after spawning, for the child to prove it has already
# exited (ticket #81). Keeps start() from blocking meaningfully while still
# catching a process that dies on launch, so that outcome is surfaced as
# status="exited" (with a returncode) instead of a false "running".
_EARLY_EXIT_WAIT_SEC = 0.25

# Per-query timeout for the NtQueryObject watchdog inside _win_handle_holders
# (ticket #71). Measured empirically against a real, ordinary Windows dev
# machine: a non-trivial fraction (measured up to ~270, out of ~6.5k queries
# against a ~140k-entry system handle table) of handle queries genuinely do
# not answer NtQueryObject within tens of milliseconds -- not just the
# classically-documented unconnected-named-pipe case, but evidently other
# slow/blocked handles too (network/reparse-point-backed files, filter-
# driver-hooked handles, etc.). Larger per-query timeouts (tried up to
# 0.25s) do NOT reliably convert these into successful results -- most are
# still slow or hung at that scale too -- while multiplying the total cost
# by the number of such handles, which is what made more generous timeouts
# impractically slow (multiple seconds to tens of minutes, and in one case
# slow enough that the scan never reached the specific handle a real,
# targeted end-to-end test needed to find). A small timeout keeps the fixed
# per-hang cost low while still reliably resolving legitimate (fast)
# handles, confirmed by a real end-to-end test against a live held-open
# file that consistently completes in single-digit seconds at this value.
#
# Ticket #90: this is now stage 1 of a two-stage wait (see
# _HANDLE_QUERY_GRACE_SEC / _HANDLE_QUERY_GRACE_BUDGET_SEC below). #71's
# finding above still stands and is NOT contradicted by adding stage 2: a
# *flat* larger per-query timeout is still not affordable, because its cost
# multiplies by however many slow handles the table holds (the exact
# behaviour #71 measured as impractically slow). Stage 2 sidesteps that by
# drawing from a small, fixed *per-scan* pool instead of inflating every
# query's own timeout -- the total extra wall clock a single scan can ever
# spend in stage 2 is capped by _HANDLE_QUERY_GRACE_BUDGET_SEC regardless of
# how many handles are slow, which is precisely what a flat per-query
# timeout cannot bound.
_HANDLE_QUERY_TIMEOUT_SEC = 0.01

# Ticket #90 -- stage 2 of the bounded NtQueryObject wait. When a query
# does not resolve within _HANDLE_QUERY_TIMEOUT_SEC (stage 1), it is given
# one more bounded chance to resolve -- up to this many additional seconds
# -- before being treated as wedged. This recovers merely-slow handles
# (e.g. under transient contention) that stage 1 alone would misclassify as
# hung, without paying the unbounded-multiplication cost #71 found with a
# flat larger stage-1 timeout, because the *total* stage-2 time a scan may
# spend is capped separately (see _HANDLE_QUERY_GRACE_BUDGET_SEC).
_HANDLE_QUERY_GRACE_SEC = 0.10

# Ticket #90 -- total stage-2 grace a single _win_handle_holders scan may
# spend across ALL of its queries combined, drawn from a per-scan
# _GraceBudget seeded to this value. This is what keeps stage 2 affordable
# where a flat larger per-query timeout was not (#71): the extra wall clock
# a pathological handle table with many slow handles can add to one scan is
# capped at this constant regardless of how many such handles exist, rather
# than scaling with their count.
_HANDLE_QUERY_GRACE_BUDGET_SEC = 1.0

# Ticket #90 -- process-wide ceiling on how many permanently-wedged
# NtQueryObject worker threads (see _BoundedQueryWorker) may exist at once,
# across every _win_handle_holders scan in this process's lifetime. Before
# this cap existed, a long-lived host process invoking _win_handle_holders
# many times (many worktree teardowns) could accumulate one abandoned
# daemon thread per genuine hang encountered, unboundedly, for as long as
# the process ran -- see the ticket for the observed real-world blowup
# (thousands of threads after 36 minutes of otherwise-idle teardown churn).
# Once this many wedged workers are outstanding, a scan degrades gracefully
# -- stops attempting further queries and returns whatever it already
# found -- rather than creating yet another worker on top of the pile. This
# cap gates *replacement*-worker creation once a scan's own worker wedges
# (see _win_handle_holders' _bounded_query) -- its original, ticket #90
# role, unchanged by #121 or #148 below.
#
# Historical note, superseded by ticket #148 -- kept for the record, not as
# current guidance: earlier drafts of this comment said this cap "must
# NEVER be used to refuse to start a scan's initial worker at all", because
# a *silent* scan-start gate on it was tried and rejected (it would latch
# permanently, forever, the moment the cap first filled, since a wedged
# worker may by definition never return). That rejection of a SILENT gate
# still stands. Ticket #148 adds a *loud* one instead -- see
# _acquire_scan_worker and _HANDLE_SCAN_MAX_LIVE_WORKERS below -- which is
# a different, acceptable thing: it reports complete=False, tags the
# result "handle_scan:capped", logs a warning, and stop() surfaces a
# dedicated STOP_REASON_HANDLE_SCAN_EXHAUSTED reason naming host-restart
# remediation, so a caller can always distinguish "refused to look" from
# "looked and found nothing" -- exactly what the rejected silent gate could
# not do.
#
# Ticket #121 -- this cap alone was only ever half of the fix. Before #121,
# every scan created a brand-new *initial* worker unconditionally (this
# cap only ever bounded *replacement* workers created mid-scan), so a
# handle that wedged one scan's initial worker just wedged the *next*
# scan's fresh initial worker the same way -- one leaked thread per
# affected _win_handle_holders call, unbounded, with this cap doing
# nothing to stop it. #121 adds a *separate*, complementary bound one
# level up in _win_handle_holders: a process-wide registry of individual
# kernel objects that have wedged at least once (see _wedging_handle_key /
# _remember_wedging_handle / _is_known_wedging_handle below), so an initial
# worker can still wedge, but only ever once per distinct pathological
# object *for as long as that object's key remains in the registry*, not
# once per call to this function. The registry is bounded FIFO (see
# _MAX_REMEMBERED_WEDGED_OBJECTS below), so this is not an unconditional
# one-ever guarantee: churn through more than
# _MAX_REMEMBERED_WEDGED_OBJECTS distinct wedging objects evicts the
# oldest key and can re-admit one further wedge for that evicted object if
# it is encountered again later.
#
# Ticket #148 -- closes the remaining gap #121 left open: even with the
# per-object registry, #121 alone still created a brand-new *initial*
# worker/thread on every single scan (bounded per-object, but not bounded
# per-thread) -- if that fresh initial worker's very first query wedged on
# a NOT-yet-registered object, its thread was never itself counted against
# this cap (which only ever gated *replacement* workers) and, once
# genuinely wedged, was never joined. #148 replaces the always-fresh
# initial worker with a single, process-wide, lock-guarded PERSISTENT
# worker (see _persistent_query_worker/_acquire_scan_worker/
# _replace_scan_worker below), reused by every later scan instead of
# recreated -- so live query-worker threads are now bounded by
# construction to _HANDLE_SCAN_MAX_LIVE_WORKERS (== 1 +
# _MAX_WEDGED_HANDLE_WORKERS) process-wide, not per-object and not
# per-scan. This cap keeps doing its now-narrower job: bounding
# *replacement*-worker churn (both mid-scan, in _bounded_query, and at the
# very next scan's start, in _acquire_scan_worker) within and across
# concurrently-running scans.
_MAX_WEDGED_HANDLE_WORKERS = 8

# Ticket #90 -- bounded join _BoundedQueryWorker.close() waits for a
# healthy (non-wedged) worker's thread to actually exit after being sent
# the shutdown sentinel. A worker that is not wedged drains its queue and
# exits almost immediately, so this only needs to be small; it exists so
# close() itself can never block indefinitely even in an unexpected case.
_WORKER_CLOSE_JOIN_SEC = 0.05

# Ceiling on the wall-clock budget for _win_handle_holders' system-wide handle
# scan. Defense in depth on top of the per-query watchdog timeout above:
# bounds the *whole* scan so a pathological handle table (e.g. an unusually
# large number of slow/hanging handles) cannot block the caller far beyond
# this ceiling. The scan degrades gracefully -- returns whatever it found so
# far -- once its budget is exceeded. This pass only runs as a last-resort
# Windows-only detection step during worktree teardown (not a hot path), so a
# multi-second worst case here is an acceptable trade for correctness.
# Measured to comfortably cover a full scan (~140k system handles) on an
# ordinary dev machine in well under half this budget.
#
# This constant is only a *ceiling*, not an entitlement: the actual per-call
# budget passed to _win_handle_holders is
# ``min(_HANDLE_SCAN_BUDGET_SEC, <time remaining from the caller's overall
# timeout>)`` -- see the ``deadline`` parameter threaded through
# ``_find_blocking_processes`` and ``_kill_blocking_processes``. This keeps
# the scan from ever independently consuming up to this ceiling *on top of*
# a caller-supplied ``timeout`` (e.g. ``stop(timeout=...)``).
_HANDLE_SCAN_BUDGET_SEC = 15.0

# Ticket #87: cap on how many descendant processes _process_tree will ever
# collect for a single root pid. Defense in depth against a pathological or
# cyclic process tree turning a single stop() call into an unbounded scan.
_MAX_TREE_NODES = 256

# Ticket #110: cap on how many PIDs a single _describe_pid() enrichment pass
# (job-object / process-group member snapshotting) will look up name/cmdline
# for before a stop() call. Bounds the extra psutil.Process(...) calls the
# snapshot-time enrichment adds -- beyond the cap, entries keep empty
# name/cmdline but still carry an accurate `source`.
_DESCRIBE_MAX_PIDS = 256

# Ticket #154: `deadline` is now a REQUIRED keyword of _find_blocking_processes
# (see below) -- every call site (teardown's dirt-gate/tier-2 diagnosis, and
# _kill_blocking_processes) always supplies a real, caller-derived deadline,
# so there is no "no deadline at all" case left for a private absolute
# ceiling to cover. _DISCOVERY_MAX_SEC is deleted outright, not kept as a
# fallback default.

# Ticket #154 (item 11): per-call wall-clock bound for a blocking psutil
# read (proc.cwd()/cmdline()/name()/parents(), tier-1's psutil.pid_exists),
# dispatched through _bounded_call. Cannot reuse _HANDLE_QUERY_TIMEOUT_SEC
# (0.01s, tuned for one NtQueryObject on an already-duplicated handle --
# applying it to a psutil call would misclassify almost every healthy read
# as wedged). Value composition: generous enough that a healthy psutil read
# on a loaded host (well under 5ms) is never misclassified; small enough
# that the worst case (_MAX_WEDGED_HANDLE_WORKERS x this = 2.0s) fits
# inside the 7s per-removal failure budget (teardown._FAILURE_BUDGET_SEC).
# Deliberately never combined with a grace budget -- see _bounded_call's
# docstring: grace exists for Pass 1c's ~10^5-per-scan NtQueryObject
# multiplication problem, not for the ~10^2-10^3-per-call psutil population.
_BLOCKING_CALL_TIMEOUT_SEC = 0.25

# Ticket #95, finding 2 (budget starvation): stop() used to hand the primary
# _wait_or_kill call the ENTIRE remaining timeout budget. On Windows, a
# CTRL_BREAK_EVENT to a CREATE_NEW_PROCESS_GROUP child sharing no console
# frequently does nothing, so _wait_or_kill polls the full timeout before
# force-killing -- leaving 0.0s for the subsequent tree kill and orphan scan,
# whose own deadline-driven guards then all fail immediately (Pass 1c, the
# Windows handle-table scan, never executes). _compute_stop_budget below caps
# the primary wait's share and reserves floors for the other two steps so
# neither can ever be starved down to zero by an adversarial primary step.
#
# The shares are chosen so _PRIMARY_WAIT_SHARE == 1 - (_TREE_KILL_FLOOR_SHARE
# + _ORPHAN_SCAN_FLOOR_SHARE): the share-based and absolute-second caps
# coincide exactly for kill_orphans=True at any timeout <= 10s; the absolute
# floors only bind (clamp below the share) once timeout > 10s.
_PRIMARY_WAIT_SHARE = 0.6
_TREE_KILL_FLOOR_SEC = 1.0
_TREE_KILL_FLOOR_SHARE = 0.10
_ORPHAN_SCAN_FLOOR_SEC = 3.0
_ORPHAN_SCAN_FLOOR_SHARE = 0.30


def _compute_stop_budget(timeout: float, kill_orphans: bool) -> Tuple[float, float, float]:
    """Split *timeout* across stop()'s primary wait, tree kill, and orphan
    scan steps (ticket #95, finding 2).

    Returns ``(primary_cap, tree_floor, orphan_floor)``:

    - ``primary_cap`` -- the maximum seconds the primary ``_wait_or_kill``
      call may be granted, regardless of how much of *timeout* remains at
      that point.
    - ``tree_floor`` -- seconds of *timeout* reserved so the tree-kill step
      (``_kill_process_tree``) is never handed a budget below this, even if
      the primary wait consumed its entire cap.
    - ``orphan_floor`` -- seconds of *timeout* reserved for the orphan scan
      (``_kill_blocking_processes``) in the same way. Always ``0.0`` when
      *kill_orphans* is ``False`` -- there is no point reserving room for a
      scan that will not run, and the tree kill should get the full
      remainder instead.

    ``timeout <= 0`` collapses every value to ``0.0``, preserving today's
    immediate-force-kill behaviour for ``stop(timeout=0)``.
    """
    if timeout <= 0:
        return 0.0, 0.0, 0.0

    tree_floor = min(_TREE_KILL_FLOOR_SEC, _TREE_KILL_FLOOR_SHARE * timeout)
    orphan_floor = (
        min(_ORPHAN_SCAN_FLOOR_SEC, _ORPHAN_SCAN_FLOOR_SHARE * timeout)
        if kill_orphans
        else 0.0
    )
    primary_cap = max(
        0.0, min(timeout * _PRIMARY_WAIT_SHARE, timeout - (tree_floor + orphan_floor))
    )
    return primary_cap, tree_floor, orphan_floor


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class ProcessLifecycleError(RuntimeError):
    """Base error for process lifecycle operations."""


class ProcessAlreadyRunningError(ProcessLifecycleError):
    """Raised by ``start`` when the role's process is already alive.

    ``.pid`` carries the PID of the existing process.
    """

    def __init__(self, worktree_id: str, role: str, pid: int) -> None:
        super().__init__(
            f"process already running for worktree '{worktree_id}' role '{role}'"
            f" (pid={pid})"
        )
        self.worktree_id = worktree_id
        self.role = role
        self.pid = pid


class ProcessNotRunningError(ProcessLifecycleError):
    """Raised by ``stop`` when no PID is recorded for the given role."""

    def __init__(self, worktree_id: str, role: str) -> None:
        super().__init__(
            f"no running process for worktree '{worktree_id}' role '{role}'"
        )
        self.worktree_id = worktree_id
        self.role = role


# ---------------------------------------------------------------------------
# Internal spawn helper
# ---------------------------------------------------------------------------

def _spawn_detached(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    log_path: Optional[Path] = None,
) -> subprocess.Popen:
    """Spawn *cmd* as a fully detached process and return its ``Popen`` object.

    The child process is detached from the caller's process group so it
    survives if the MCP host exits.  ``stdin`` is always ``subprocess.DEVNULL``
    — a detached child must never inherit the caller's stdin.  When
    *log_path* is given, ``stdout``/``stderr`` are redirected to that file
    (opened append-binary) so a spawned process's output and early-exit
    reason are recoverable; otherwise they are also ``subprocess.DEVNULL``.
    """
    if not cmd:
        raise ValueError("cmd must be a non-empty list")

    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if env is not None:
        kwargs["env"] = env
    if cwd is not None:
        kwargs["cwd"] = cwd

    if sys.platform == "win32":
        # Use CREATE_NEW_PROCESS_GROUP alone (without DETACHED_PROCESS).
        # DETACHED_PROCESS severs the child from *all* consoles, which
        # means GenerateConsoleCtrlEvent (CTRL_BREAK_EVENT) is never
        # delivered — the graceful-stop path would always fall through to
        # force-kill.  With DEVNULL stdio the child still survives parent
        # exit and is independent, but retains a process group so it can
        # receive CTRL_BREAK_EVENT.
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        )
    else:
        kwargs["start_new_session"] = True

    if log_path is not None:
        log_file = open(log_path, "ab")
        try:
            kwargs["stdout"] = log_file
            kwargs["stderr"] = log_file
            proc = subprocess.Popen(cmd, **kwargs)
        finally:
            # The child inherits its own duplicate of the handle; the parent
            # closes its copy immediately after Popen returns.
            log_file.close()
    else:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
        proc = subprocess.Popen(cmd, **kwargs)

    # Ticket #95, R5: Windows Job Object containment -- a ppid-independent
    # mechanism that closes the gap _process_tree's ppid-walk cannot: a
    # ShellExecuteEx-delegated launch (what `Start-Process` uses without
    # stream redirection) lands OUTSIDE our ppid lineage entirely, so no
    # amount of recursion depth in _process_tree can ever find it. Every
    # process this job contains -- however it was spawned, however deeply
    # nested, regardless of ppid -- is enumerable and terminable as a unit
    # by stop() (see _job_object_member_pids / _terminate_job_object).
    #
    # job_name is stashed as an attribute on the returned Popen rather than
    # changing this function's return type, to keep every existing caller
    # (including tests that do `_spawn_detached(...).pid`) working
    # unchanged. `None` means "no containment available" -- POSIX, or any
    # Windows failure -- callers must always treat it as optional.
    job_name: Optional[str] = None
    if sys.platform == "win32":
        candidate_job_name = f"Local\\worktree-{uuid.uuid4().hex}"
        job_handle = _create_job_object(candidate_job_name)
        if job_handle is not None:
            if _assign_process_to_job(job_handle, proc.pid):
                job_name = candidate_job_name
            else:
                # Assignment failed (e.g. lost the sub-millisecond race
                # against the child already exiting) -- leave job_name None
                # so nothing persists a job with no member in it. Ticket #95
                # fix cycle (blocking finding, round 4): job_name staying
                # None means record.job_names[role] is never populated for
                # this attempt, so stop()'s job-handle-close logic (which
                # only runs when record.job_names.get(role) is truthy) can
                # never discover this handle -- it and its _JOB_HANDLES
                # entry would otherwise leak indefinitely on a long-lived
                # host. Close and evict it here instead, on this
                # assignment-failure path specifically, since this is the
                # only place that will ever know about it.
                _close_job_object_handle(job_handle)
                _JOB_HANDLES.pop(candidate_job_name, None)
    proc._worktree_job_name = job_name  # type: ignore[attr-defined]

    return proc


# ---------------------------------------------------------------------------
# Windows Job Object containment -- ticket #95, R5
# ---------------------------------------------------------------------------

# A named Job Object dies the moment its LAST HANDLE closes -- there is no
# per-process "owner" tie-in the way there is for e.g. a child process. The
# spawning host process must therefore keep at least one handle open for as
# long as containment should hold; this module-level dict is that keeper,
# indexed by job_name.
#
# Consequence, documented rather than worked around: containment holds only
# for the lifetime of the HOST process that spawned it. A stop() call issued
# by a RESTARTED host degrades gracefully to the ppid/process-group path --
# the job object itself is already gone by the time a new host process
# starts (its last handle, held by the now-dead old host, already closed) --
# this is rule N7, not a stop_incomplete-triggering condition.
_JOB_HANDLES: Dict[str, int] = {}

# QueryInformationJobObject(JobObjectBasicProcessIdList) buffer sizing
# (ticket #95): start generous enough to avoid a second round-trip in the
# common case, grow to the OS-reported exact size on ERROR_MORE_DATA, but
# never past this cap -- mirrors _process_tree's _MAX_TREE_NODES treatment:
# hitting it is never silently trusted as "that's everyone" (see stop()'s
# job-member-list truncation handling, the same class as
# tree_possibly_truncated).
_JOB_MEMBER_LIST_INITIAL_SLOTS = 64
_JOB_MEMBER_LIST_MAX_SLOTS = 4096


def _create_job_object(job_name: str) -> Optional[int]:
    """Windows-only: create a new Job Object named *job_name* with NO limit
    flags set, and register its handle in :data:`_JOB_HANDLES`.

    Deliberately sets no limits -- in particular, this NEVER calls
    ``SetInformationJobObject`` to set ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``.
    That flag would terminate every process still assigned to the job the
    moment ITS OWN handle closes (e.g. when this host process exits) --
    which would directly contradict this module's core detachment
    invariant: a process spawned via :func:`_spawn_detached` must survive
    the host process exiting. This job object exists purely as a
    containment/enumeration mechanism for :func:`stop`, never as an
    auto-kill-on-host-exit mechanism.

    Returns the handle (``int``) on success, ``None`` on any failure --
    best-effort; POSIX never calls this, and a Windows failure here simply
    means the caller falls back to the ppid-tree-only containment path
    (rule N7).
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        handle = kernel32.CreateJobObjectW(None, job_name)
    except Exception:  # noqa: BLE001 -- best-effort, never propagate
        return None
    if not handle:
        return None
    _JOB_HANDLES[job_name] = handle
    return handle


def _assign_process_to_job(job_handle: int, pid: int) -> bool:
    """Windows-only: ``AssignProcessToJobObject(job_handle, pid)``.

    Opens *pid* with ``PROCESS_SET_QUOTA | PROCESS_TERMINATE`` -- the
    minimal access ``AssignProcessToJobObject`` needs.

    A sub-millisecond race is accepted and documented rather than worked
    around: ``subprocess.Popen`` closes the child's thread handle once it
    returns, so ``CREATE_SUSPENDED``/``PROC_THREAD_ATTRIBUTE_JOB_LIST`` are
    not reachable through it -- the child could in principle spawn its own
    descendants in the brief window before this assignment lands. Those very
    early descendants would not be job members, but they remain covered by
    the existing ppid-tree path (:func:`_process_tree`) regardless -- this
    job-object mechanism is additive containment, not a replacement for it.

    Returns ``True`` on success. Best-effort; never raises.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    PROCESS_SET_QUOTA = 0x0100
    PROCESS_TERMINATE = 0x0001

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        proc_handle = kernel32.OpenProcess(
            PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid
        )
        if not proc_handle:
            return False
        try:
            return bool(kernel32.AssignProcessToJobObject(job_handle, proc_handle))
        finally:
            kernel32.CloseHandle(proc_handle)
    except Exception:  # noqa: BLE001 -- best-effort, never propagate
        return False


def _open_job_object(job_name: str) -> Optional[int]:
    """Windows-only: return a handle usable for
    ``QueryInformationJobObject``/``TerminateJobObject`` on *job_name*.

    Prefers the :data:`_JOB_HANDLES` registry's own handle -- the common
    case, when this is the same host process that created the job (and
    therefore the only process guaranteed to still have it alive at all;
    see that dict's docstring). Falls back to a fresh ``OpenJobObjectW``
    attempt otherwise, which is a harmless no-op failure in the
    restarted-host case (the object's last handle -- held by the now-dead
    old host -- has already closed by then).

    Returns ``None`` on any failure. Best-effort; never raises.
    """
    if sys.platform != "win32":
        return None
    if job_name in _JOB_HANDLES:
        return _JOB_HANDLES[job_name]

    import ctypes
    from ctypes import wintypes

    JOB_OBJECT_QUERY = 0x0004
    JOB_OBJECT_TERMINATE = 0x0001

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.OpenJobObjectW.restype = wintypes.HANDLE
        kernel32.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        handle = kernel32.OpenJobObjectW(
            JOB_OBJECT_QUERY | JOB_OBJECT_TERMINATE, False, job_name
        )
    except Exception:  # noqa: BLE001 -- best-effort, never propagate
        return None
    return handle or None


def _job_object_member_pids(job_handle: int) -> "_PartialList":
    """Windows-only: return every PID currently assigned to *job_handle* via
    ``QueryInformationJobObject(JobObjectBasicProcessIdList)``.

    Buffer sizing starts at :data:`_JOB_MEMBER_LIST_INITIAL_SLOTS`, retried
    at the OS-reported ``NumberOfAssignedProcesses`` on ``ERROR_MORE_DATA``,
    capped at :data:`_JOB_MEMBER_LIST_MAX_SLOTS`. Reaching the cap without
    room for every assigned process is reported the same way
    :func:`_process_tree` reports hitting ``_MAX_TREE_NODES`` -- a
    :class:`_PartialList` with ``complete=False`` -- so ``stop()`` can treat
    it the same, conservative way (never silently trusted as "that's
    everyone").

    Never raises: any ctypes/structure failure degrades to
    ``_PartialList([], complete=False)`` -- "never looked", not "found
    nothing".
    """
    import ctypes
    from ctypes import wintypes

    JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
    ERROR_MORE_DATA = 234

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]

        header_size = ctypes.sizeof(ctypes.c_ulong) * 2
        slot_size = ctypes.sizeof(ctypes.c_size_t)
        slots = _JOB_MEMBER_LIST_INITIAL_SLOTS

        for _attempt in range(6):
            buf_size = header_size + slots * slot_size
            buf = ctypes.create_string_buffer(buf_size)
            return_length = wintypes.DWORD(0)
            ctypes.set_last_error(0)
            ok = kernel32.QueryInformationJobObject(
                job_handle,
                JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                buf,
                buf_size,
                ctypes.byref(return_length),
            )
            if ok:
                number_in_list = ctypes.c_ulong.from_buffer_copy(buf, 4).value
                pids = [
                    ctypes.c_size_t.from_buffer_copy(
                        buf, header_size + i * slot_size
                    ).value
                    for i in range(number_in_list)
                ]
                return _PartialList(pids, complete=True)

            err = ctypes.get_last_error()
            if err != ERROR_MORE_DATA:
                return _PartialList([], complete=False)

            number_assigned = ctypes.c_ulong.from_buffer_copy(buf, 0).value
            if slots >= _JOB_MEMBER_LIST_MAX_SLOTS:
                # Already at the cap and still not enough room -- give up
                # rather than looping forever; report as incomplete.
                return _PartialList([], complete=False)
            slots = min(_JOB_MEMBER_LIST_MAX_SLOTS, max(slots * 2, number_assigned))

        return _PartialList([], complete=False)
    except Exception:  # noqa: BLE001 -- best-effort, never propagate
        return _PartialList([], complete=False)


def _terminate_job_object(job_handle: int) -> None:
    """Windows-only: ``TerminateJobObject(job_handle, 1)``.

    The ppid-independent force step -- terminates every process still
    assigned to the job in one call, regardless of how deeply nested or how
    it was spawned. Best-effort; never raises.
    """
    import ctypes
    from ctypes import wintypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject(job_handle, 1)
    except Exception:  # noqa: BLE001 -- best-effort, never propagate
        pass


def _close_job_object_handle(job_handle: int) -> None:
    """Windows-only: ``CloseHandle(job_handle)`` for a Job Object handle
    (ticket #95 fix cycle -- blocking finding: Job Object handles leak in
    the common path).

    Every job handle :func:`stop` obtains via :func:`_open_job_object` --
    whether served from the :data:`_JOB_HANDLES` keeper registry (the
    common path) or opened fresh via the ``OpenJobObjectW`` fallback (the
    narrower restarted-host path round 1 nit-picked) -- must be closed once
    :func:`_terminate_job_object` has run against it. Before this fix,
    nothing anywhere called ``CloseHandle`` on a job handle: on a
    long-lived host process (e.g. an MCP server that never restarts), every
    ``start()``/``stop()`` cycle for a Windows worktree leaked one kernel
    HANDLE indefinitely. Mirrors the explicit ``argtypes`` pattern used for
    other HANDLE-typed bindings in this module (e.g.
    :func:`_assign_process_to_job`'s own ``CloseHandle`` usage) rather than
    relying on ctypes' default int guessing, which silently truncates 64-bit
    HANDLE values on x64. Best-effort; never raises.
    """
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle(job_handle)
    except Exception:  # noqa: BLE001 -- best-effort, never propagate
        pass


# ---------------------------------------------------------------------------
# Internal signal / kill helpers
# ---------------------------------------------------------------------------

def _send_graceful_signal(pid: int, *, group_leader: bool = False) -> bool:
    """Send the platform-appropriate graceful-stop signal to *pid*.

    Windows: CTRL_BREAK_EVENT (sent to the process group).
    POSIX:   SIGTERM.

    ``GenerateConsoleCtrlEvent`` (what ``os.kill(pid, CTRL_BREAK_EVENT)``
    invokes on win32) targets a console PROCESS GROUP id, not an arbitrary
    pid. Handing it a pid we never confirmed is the leader of a process
    group WE created (via ``CREATE_NEW_PROCESS_GROUP``) is at best a no-op
    and at worst misdirects the break to an unrelated process group sharing
    the same console. ``group_leader=True`` is therefore how the caller
    explicitly asserts that *pid* is such a confirmed leader; on win32 the
    break is only ever issued when that assertion is made. ``group_leader``
    is ignored on POSIX, where ``SIGTERM`` is already pid-precise.

    Refused unconditionally (no ``os.kill`` call, returns ``False``) when
    *pid* is non-positive or is our own pid — the latter mirrors
    ``_signal_process_group``'s "never our own group" rule and applies even
    when ``group_leader=True``.

    Returns ``True`` if the signal was actually delivered (``os.kill``
    raised no exception), ``False`` otherwise (refused, or the process had
    already exited and ``os.kill`` raised ``OSError``).

    Known limitation: like ``stop()`` itself (see its docstring), the
    tracked pid is trusted as-is with no identity check — ``group_leader``
    confirms only that the CALLER asserts leadership, not that *pid* has
    not been reused by an unrelated process since it was last observed
    alive.
    """
    if pid <= 0 or pid == os.getpid():
        return False

    if sys.platform == "win32":
        if group_leader is not True:
            _logger.debug(
                "skipping CTRL_BREAK_EVENT for pid %s: process-group "
                "leadership not confirmed; falling through to "
                "wait/force-kill",
                pid,
            )
            return False
        try:
            os.kill(pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        except OSError:
            # Process may have already exited between the liveness check and
            # the signal call — treat as a no-op.
            return False
        return True
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False
        return True


def _force_kill(pid: int) -> None:
    """Unconditionally kill *pid*.

    Windows: TerminateProcess via ctypes.
    POSIX:   SIGKILL.
    """
    if sys.platform == "win32":
        import ctypes
        PROCESS_TERMINATE = 0x0001
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if handle:
            try:
                kernel32.TerminateProcess(handle, 1)
            finally:
                kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _reap(pid: int) -> bool:
    """Best-effort reap of our own child *pid* (POSIX only).

    A process we spawned that has exited but has not been waited on lingers
    as a *zombie*: ``os.kill(pid, 0)`` (used by ``_pid_alive``) still
    succeeds for it, so the process reads as "alive" forever.  Reaping it
    with a non-blocking ``waitpid`` removes the zombie so liveness checks
    report the truth.

    Returns ``True`` only when *pid* was actually reaped (it is now gone).
    Returns ``False`` when the process is still running, or when ``waitpid``
    cannot reap it (e.g. ``ECHILD`` — not our child, already reaped by init
    after an MCP restart); in that case callers fall back to ``_pid_alive``.

    No-op on Windows, which has no zombies — a process object disappears
    once the last handle to it is closed.
    """
    if sys.platform == "win32":
        return False
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except OSError:
        # ECHILD (not our child) or similar — we cannot reap it here.
        return False
    return waited == pid


def _reap_until_gone(pid: int, attempts: int = 50) -> None:
    """Briefly poll-reap a just-force-killed child so it does not linger.

    After ``_force_kill`` the child dies imminently but may not be a
    reapable zombie for a few milliseconds.  Poll a short while so that, by
    the time we return, ``_pid_alive`` reflects the death.  No-op on Windows.
    """
    if sys.platform == "win32":
        return
    for _ in range(attempts):
        if _reap(pid) or not _pid_alive(pid):
            return
        time.sleep(0.01)


def _wait_or_kill(pid: int, timeout: float) -> None:
    """Wait up to *timeout* seconds for *pid* to die; force-kill if it doesn't.

    Uses a polling loop (0.1 s sleep) so there is no hard dependency on
    psutil or OS-specific wait APIs.  Each poll reaps *pid* if it has become
    a zombie child of ours, so a graceful exit is detected promptly instead
    of reading as "alive" indefinitely.  ``timeout <= 0`` goes straight to
    force-kill.
    """
    if timeout <= 0:
        _force_kill(pid)
        _reap_until_gone(pid)
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _reap(pid) or not _pid_alive(pid):
            return
        time.sleep(0.1)

    # Still alive after the timeout — escalate to force-kill, then reap so
    # the killed child does not linger as a zombie.
    if not _reap(pid) and _pid_alive(pid):
        _force_kill(pid)
        _reap_until_gone(pid)


# ---------------------------------------------------------------------------
# Blocking-process detection and kill helpers
# ---------------------------------------------------------------------------

# Ticket #132: interpreter basenames _decode_encoded_command recognizes as
# the PowerShell/pwsh -EncodedCommand transport this library itself emits
# (setup.runner._build_step_command, ticket #109). psutil reports the full
# interpreter path (e.g. "C:\\Windows\\...\\powershell.exe"), so detection
# matches on os.path.basename(...).lower() rather than the raw token.
_ENCODED_COMMAND_INTERPRETER_BASENAMES = frozenset(
    {"powershell.exe", "powershell", "pwsh.exe", "pwsh"}
)

# Ticket #132: upper bound on an -EncodedCommand payload token's length
# before _decode_encoded_command gives up on it without attempting to
# base64-decode it at all. Defense in depth against spending decode work on
# an implausibly large token that was never actually produced by
# _build_step_command.
_MAX_ENCODED_COMMAND_PAYLOAD_LEN = 65536

# Ticket #132: recognized short-flag aliases for -EncodedCommand that are
# NOT literal prefixes of the string "encodedcommand" -- PowerShell's own
# CLI documents "-ec" as a first-class alias (see `pwsh -h`), not merely an
# abbreviation of the full switch name (its own prefix-of-full-name check
# would land on "en", not "ec"). Combined with the prefix check in
# _decode_encoded_command, this set plus "is a prefix of encodedcommand"
# together cover every variant this ticket enumerates: -EncodedCommand,
# -enc, -ec, -e, /EncodedCommand. Note "-e" needs no entry here -- it IS
# already a literal prefix of "encodedcommand", so the prefix check alone
# covers it.
_ENCODED_COMMAND_FLAG_ALIASES = frozenset({"ec"})


def _try_decode_encoded_command_payload(payload: str) -> Optional[str]:
    """Best-effort inverse of ``_build_step_command``'s
    ``base64.b64encode(run_line.encode("utf-16-le")).decode("ascii")``
    (ticket #109/#132).

    Returns the decoded run-line text on success, or ``None`` when *payload*
    does not look like a genuine ``-EncodedCommand`` blob this library
    produced -- never raises. Validation, in order: strict base64 (no
    non-alphabet characters silently ignored), a length cap, an even decoded
    byte length (UTF-16 code units are 2 bytes each), strict UTF-16LE
    decoding, and a check that the decoded text contains no control
    characters other than ``\\t``/``\\r``/``\\n`` (a genuine PowerShell
    run-line payload is command text, not arbitrary binary wearing a text
    disguise).

    Ticket #134 regression fix: ``_build_step_command`` appends
    ``_PS_EXIT_CODE_EPILOGUE`` to every PowerShell/pwsh run line before
    encoding it, so a genuine inverse must also undo that append -- a
    trailing exact match of the epilogue is stripped from the decoded text
    before it is returned, so ``KilledProcessInfo.cmdline`` keeps showing
    the plain, human-readable run line (this function's only callsite is
    already gated to the PowerShell/pwsh interpreters by
    :func:`_decode_encoded_command`, so a bash/sh payload is never routed
    through here and needs no special-casing).
    """
    if not payload or len(payload) > _MAX_ENCODED_COMMAND_PAYLOAD_LEN:
        return None
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:  # noqa: BLE001 -- e.g. binascii.Error
        return None
    if len(raw) % 2 != 0:
        return None
    try:
        text = raw.decode("utf-16-le", errors="strict")
    except Exception:  # noqa: BLE001 -- e.g. UnicodeDecodeError
        return None
    for ch in text:
        if ch in ("\t", "\r", "\n"):
            continue
        if ord(ch) < 0x20 or ord(ch) == 0x7F:
            return None
    from ..setup.runner import _PS_EXIT_CODE_EPILOGUE  # noqa: PLC0415

    if text.endswith(_PS_EXIT_CODE_EPILOGUE):
        text = text[: -len(_PS_EXIT_CODE_EPILOGUE)]
    return text


def _decode_encoded_command(cmdline: List[str]) -> Tuple[List[str], bool]:
    """Rewrite a PowerShell/pwsh ``-EncodedCommand <base64 blob>`` argv into
    a human-readable one, for ``KilledProcessInfo.cmdline`` (ticket #132).

    Detection is deliberately narrow, to avoid mangling unrelated argv:

    1. ``os.path.basename(cmdline[0]).lower()`` must be one of
       :data:`_ENCODED_COMMAND_INTERPRETER_BASENAMES`.
    2. A later token, after stripping a leading ``-``/``/`` and lowercasing,
       must be a non-empty prefix of ``"encodedcommand"`` OR a member of
       :data:`_ENCODED_COMMAND_FLAG_ALIASES` -- together these cover
       ``-EncodedCommand``, ``-enc``, ``-ec``, ``-e``, ``/EncodedCommand``.
    3. The token immediately after it (the payload) must pass
       :func:`_try_decode_encoded_command_payload`'s validation.

    Every occurrence in *cmdline* is handled, not just the first. The
    ``-EncodedCommand``-equivalent flag token itself is always kept
    unchanged -- only the payload token right after it is replaced with the
    decoded text -- mirroring ``SetupRunner._invoke``'s existing log-swap.

    Returns ``(argv, changed)``: *argv* is a NEW list (the input is never
    mutated in place) equal to *cmdline* except for any substituted payload
    tokens; *changed* is ``True`` only when at least one substitution
    actually happened. Never raises -- any unexpected failure degrades to
    ``(list(cmdline), False)``.
    """
    try:
        if not cmdline:
            return list(cmdline), False
        basename = os.path.basename(cmdline[0]).lower()
        if basename not in _ENCODED_COMMAND_INTERPRETER_BASENAMES:
            return list(cmdline), False

        result = list(cmdline)
        changed = False
        i = 1
        while i < len(result) - 1:
            flag_token = result[i].lstrip("-/").lower()
            if flag_token and (
                "encodedcommand".startswith(flag_token)
                or flag_token in _ENCODED_COMMAND_FLAG_ALIASES
            ):
                decoded = _try_decode_encoded_command_payload(result[i + 1])
                if decoded is not None:
                    result[i + 1] = decoded
                    changed = True
                    i += 2
                    continue
            i += 1
        return result, changed
    except Exception:  # noqa: BLE001 -- must never raise into KilledProcessInfo
        return list(cmdline), False


@dataclass
class KilledProcessInfo:
    """Information about a process that was killed to unblock worktree removal.

    ``source`` (ticket #110) is provenance: which discovery mechanism found
    this pid -- one of ``"tree"`` (ppid-descendant walk, :func:`_process_tree`),
    ``"process_group"`` (POSIX process-group snapshot,
    :func:`_process_group_members`), ``"job_object"`` (Windows Job Object
    member enumeration), ``"tracked"`` (the pid ``stop()`` was originally
    given), or ``"orphan_scan"`` (the path-heuristic scan,
    :func:`_find_blocking_processes` / :func:`_kill_blocking_processes`).
    Defaults to ``"unknown"`` only for callers/tests constructed before this
    field existed. A ``"job_object"``-sourced entry may legitimately be an
    OS-created artifact of the job (e.g. a stray ``conhost.exe``) rather than
    anything this module spawned directly -- ``source`` is what a caller
    should filter/group on, never an empty ``name`` alone (an empty name can
    also mean "psutil could not be queried in time", not "nothing there").

    ``cmdline``/``cmdline_raw`` (ticket #132): PowerShell/pwsh accept a
    ``-EncodedCommand`` (base64+UTF-16LE) blob in place of a plain
    ``-Command`` argument -- this library's own setup-step transport
    (ticket #109, ``setup.runner._build_step_command``) is one source of
    such argv, but the raw ``psutil.cmdline()`` payload token it produces is
    an opaque blob, not human-readable, regardless of who emitted it.
    ``__post_init__`` runs :func:`_decode_encoded_command` against
    ``cmdline`` and, whenever an argv is ``-EncodedCommand``-shaped
    (recognized PowerShell/pwsh interpreter basename, a recognized flag
    token, and a payload that passes
    :func:`_try_decode_encoded_command_payload`'s strict base64/UTF-16LE
    validation) and a substitution actually happens, rewrites ``cmdline`` in
    place to the human-readable run line and stashes the original,
    untouched argv in ``cmdline_raw``. The heuristic has no way to verify
    true provenance -- it decodes any argv that fits the shape, including a
    genuine, unrelated third-party process that happens to invoke
    PowerShell with ``-EncodedCommand`` and a validly-formed payload, not
    only commands this library itself built. This is an accepted,
    intentional trade-off, not a bug: even for such a third-party
    invocation the decoded text is still an accurate rendering of what
    actually ran, and the original raw argv remains available via
    ``cmdline_raw`` for anyone who needs the byte-for-byte original.
    ``cmdline_raw`` stays ``None`` when no substitution happened --
    including for every non-PowerShell process, and for a PowerShell
    process whose argv was never ``-EncodedCommand``-shaped in the first
    place -- so ``cmdline_raw is None`` is the caller-facing contract for
    "``cmdline`` is exactly what psutil reported, unmodified". Decoding can
    never raise (see
    :func:`_decode_encoded_command`'s own never-raises contract), so this
    can never turn a previously-safe ``KilledProcessInfo`` construction into
    one that fails. ``compare=False`` on ``cmdline_raw`` keeps ``__eq__``
    semantics identical to before this field existed -- two entries that
    differ only in whether/how their raw argv was captured still compare
    equal by their (possibly decoded) ``cmdline``, as they always did.

    ``match_pass`` (ticket #140) is finer-grained provenance than
    ``source``: which of :func:`_find_blocking_processes`' four detection
    passes produced this hit -- ``"cwd"`` (Pass 1, cwd match), ``"cmdline"``
    (Pass 1b, Windows cmdline-token fallback), ``"handle_scan"`` (Pass 1c,
    Windows OS-level handle-table scan), or ``"open_files"`` (Pass 2, open
    file handle match) -- using the same literal tags the existing
    ``skipped_passes`` machinery already uses. It is set only at those four
    construction sites; every other construction site (``_process_tree``'s
    ``source="tree"``, ``"tracked"``, ``"process_group"``, ``"job_object"``)
    leaves it ``None``. ``source`` itself stays the uniform
    ``"orphan_scan"`` literal for all four passes -- ``match_pass`` is the
    intentionally new, more granular axis, not a replacement for ``source``.
    ``compare=False`` for the same reason as ``cmdline_raw``: two entries
    differing only in which pass found them still compare equal.
    """

    pid: int
    name: str
    cmdline: List[str] = field(default_factory=list)
    source: str = "unknown"
    match_pass: Optional[str] = field(default=None, compare=False)
    cmdline_raw: Optional[List[str]] = field(default=None, compare=False, init=False)

    def __post_init__(self) -> None:
        decoded_cmdline, changed = _decode_encoded_command(self.cmdline)
        if changed:
            self.cmdline_raw = list(self.cmdline)
            self.cmdline = decoded_cmdline


class _PartialList(list):
    """A ``list`` subclass that also tracks discovery *coverage* (ticket #95,
    finding 3).

    ``_find_blocking_processes``, ``_kill_blocking_processes``, and
    ``_win_handle_holders`` used to return a plain ``[]`` in both the
    "genuinely found nothing" case and the "discovery was starved by the
    deadline (or a worker-cap/exception) before it could look" case --
    callers (``stop()``) could not tell them apart, so a starved scan
    silently read as a clean, confirmed-empty result.

    ``complete`` tracks discovery COVERAGE only -- never kill efficacy
    (whether a killed process is actually dead is the survivor re-probe's
    job, not this flag's) and never platform applicability (a pass simply
    not running on this OS, e.g. Pass 1b/1c on POSIX, is not incompleteness
    -- see the D1-D9 / N1-N8 rules documented on ``_find_blocking_processes``
    and ``_kill_blocking_processes``).

    ``skipped_passes`` names which pass(es) contributed to ``complete=False``,
    using short tags (e.g. ``"cwd:truncated"``, ``"handle_scan:skipped"``) so
    a caller's warning log can point an operator at the specific pass that
    degraded, not just "something, somewhere, might be incomplete".

    This deliberately keeps every existing call site working unchanged: a
    plain ``[]`` returned by an old test mock, or received by
    ``manager._teardown``'s positional ``_kill_blocking_processes(record.path)``
    call, has no ``.complete``/``.skipped_passes`` attributes at all --
    every reader in this module uses ``getattr(result, "complete", True)``
    (default: trust it, exactly what a bare list has always implicitly
    meant) rather than requiring this subclass.
    """

    def __init__(
        self,
        iterable=(),
        *,
        complete: bool = True,
        skipped_passes: Tuple[str, ...] = (),
    ) -> None:
        super().__init__(iterable)
        self.complete = complete
        self.skipped_passes = tuple(skipped_passes)


def _describe_pid(pid: int) -> Tuple[str, List[str]]:
    """Best-effort ``(name, cmdline)`` lookup for *pid* (ticket #110).

    Never raises: any psutil failure (``NoSuchProcess``, ``AccessDenied``, or
    anything else) degrades to ``("", [])`` -- mirrors the per-field
    try/except style already used in :func:`_process_tree` and the Pass 1c
    handle-scan enrichment in :func:`_find_blocking_processes`. Callers are
    expected to invoke this BEFORE any signal is sent, while there is still a
    chance the OS can answer -- once a process has been killed, both
    ``name()`` and ``cmdline()`` reliably fail.
    """
    import psutil

    name = ""
    cmdline: List[str] = []
    try:
        proc = psutil.Process(pid)
    except Exception:  # noqa: BLE001 -- e.g. NoSuchProcess/AccessDenied
        return name, cmdline
    try:
        name = proc.name() or ""
    except Exception:  # noqa: BLE001
        pass
    try:
        cmdline = proc.cmdline() or []
    except Exception:  # noqa: BLE001
        pass
    return name, cmdline


def _process_tree(pid: int) -> List[KilledProcessInfo]:
    """Return the descendant process tree of *pid*, deepest-first (ticket #87).

    This is the fix for the double-shell-nesting problem: ``stop()`` used to
    only ever signal the single PID it tracked, so a grandchild spawned by a
    nested shell (e.g. ``run: powershell ... -Command "..."`` where the
    ``run:`` string itself invokes another shell) survived being reparented
    once the tracked PID died, along with any ports it held. Snapshotting the
    *whole* tree while the root is still alive means reparenting after the
    kill can no longer hide anything.

    Primary implementation: ``psutil.Process(pid).children(recursive=True)``.
    Falls back to a manual ``ppid``-walk via
    ``psutil.process_iter(["pid", "ppid"])`` when the primary call raises or
    returns nothing -- e.g. because psutil's own recursive walk lost a race
    with a process exiting mid-scan.

    Ordering is deepest-first: a node whose parent is also in the collected
    set sorts after that parent, so callers can signal/kill children before
    their own ancestors without orphaning a node mid-walk. Depth is derived
    purely from each collected node's own ``ppid()`` relative to the other
    *collected* nodes -- no extra ``psutil.Process`` calls beyond what was
    already gathered.

    Capped at ``_MAX_TREE_NODES`` entries. The host process (our own PID) and
    all of its OS-level ancestors are always excluded, even if they somehow
    appear as a "descendant" (e.g. PID reuse) -- this function must never
    target the caller's own lineage.

    Truncation is never silent (ticket #87 follow-up, finding F2): when the
    cap is actually hit -- i.e. more descendants exist than the cap allows,
    not merely "exactly ``_MAX_TREE_NODES`` found, no more" -- a ``warning``
    is logged naming *pid* and the cap, so an operator can tell from the
    logs alone that a snapshot may be incomplete. Callers that need to know
    this programmatically (rather than just log it) -- e.g. ``stop()``,
    which must not report a clean ``"stopped"`` purely on the strength of a
    capped snapshot -- treat ``len(result) >= _MAX_TREE_NODES`` as "cannot
    guarantee completeness": that check is a cheap, self-contained proxy for
    the same condition this warning logs, with no need to change this
    function's return type (and no risk of the boundary case where the tree
    happens to have *exactly* ``_MAX_TREE_NODES`` descendants and no more --
    that case is, conservatively, still treated as "not guaranteed
    complete").

    Never raises: ``NoSuchProcess``/``AccessDenied``/anything else degrades
    to a best-effort partial list, or ``[]`` if nothing could be gathered.
    ``pid <= 0`` (including ``None``) returns ``[]`` immediately.
    """
    if not pid or pid <= 0:
        return []

    import psutil

    host_pid = os.getpid()
    excluded: "set[int]" = {host_pid}
    try:
        for ancestor in psutil.Process(host_pid).parents():
            excluded.add(ancestor.pid)
    except Exception:  # noqa: BLE001 -- best-effort, never propagate
        pass

    descendants: list = []
    truncated = False
    try:
        descendants = list(psutil.Process(pid).children(recursive=True))
    except Exception:  # noqa: BLE001 -- e.g. NoSuchProcess/AccessDenied
        descendants = []

    # Only attempt the (expensive, full-system) ppid-walk fallback when *pid*
    # plausibly still exists. A dead/nonexistent pid cannot have live
    # descendants reachable via a ppid-walk either (any children it once had
    # are already reparented elsewhere by the time it is gone), so spending
    # a full psutil.process_iter() sweep on that case would be pure waste --
    # this is also the common case (an already-dead root, or a
    # blocking-process pid supplied by _kill_blocking_processes that turns
    # out not to exist), so skipping it here matters for real wall-clock
    # cost, not just a theoretical corner case. psutil.pid_exists() is a
    # single cheap existence check, not an enumeration.
    if not descendants and psutil.pid_exists(pid):
        # Fallback: manual ppid-walk. Covers the case where
        # children(recursive=True) itself raised or returned nothing (e.g.
        # a race with a process exiting mid-scan) even though *pid* is
        # still alive.
        try:
            by_ppid: Dict[int, List[int]] = {}
            for proc in psutil.process_iter(["pid", "ppid"]):
                try:
                    ppid = proc.info.get("ppid")
                except Exception:  # noqa: BLE001
                    continue
                if ppid is None:
                    continue
                by_ppid.setdefault(ppid, []).append(proc.info["pid"])
            frontier = [pid]
            seen_pids: "set[int]" = set()
            while frontier and len(seen_pids) < _MAX_TREE_NODES:
                next_frontier: List[int] = []
                for parent_pid in frontier:
                    for child_pid in by_ppid.get(parent_pid, []):
                        if child_pid == pid or child_pid in seen_pids:
                            continue
                        seen_pids.add(child_pid)
                        next_frontier.append(child_pid)
                frontier = next_frontier
            if frontier:
                # The while loop above exited with unexplored children still
                # queued in `frontier` -- it stopped because `seen_pids` hit
                # the cap, not because the tree was exhausted. The true tree
                # is larger than what was collected (finding F2).
                truncated = True
            for child_pid in seen_pids:
                try:
                    descendants.append(psutil.Process(child_pid))
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001 -- best-effort, never propagate
            pass

    # De-duplicate by pid, drop the host process / its ancestors (defensive
    # -- should never legitimately appear), cap at _MAX_TREE_NODES. Unlike a
    # `break` once the cap is hit, this keeps scanning the (already
    # in-memory, cheap -- no further psutil calls) remainder of *descendants*
    # so `truncated` accurately reflects whether more than the cap actually
    # existed (finding F2), not just "cap coincidentally == count found".
    by_pid: Dict[int, object] = {}
    for proc in descendants:
        try:
            proc_pid = proc.pid
        except Exception:  # noqa: BLE001
            continue
        if proc_pid in excluded or proc_pid in by_pid:
            continue
        if len(by_pid) >= _MAX_TREE_NODES:
            truncated = True
            continue
        by_pid[proc_pid] = proc

    known_pids = set(by_pid)
    depth_cache: Dict[int, int] = {}

    def _depth(proc_pid: int) -> int:
        if proc_pid in depth_cache:
            return depth_cache[proc_pid]
        depth_cache[proc_pid] = 0  # cycle guard while this node is computing
        proc = by_pid[proc_pid]
        try:
            parent_pid = proc.ppid()
        except Exception:  # noqa: BLE001
            parent_pid = None
        if parent_pid in known_pids and parent_pid != proc_pid:
            depth = 1 + _depth(parent_pid)
        else:
            depth = 0
        depth_cache[proc_pid] = depth
        return depth

    ordered_pids = sorted(by_pid, key=_depth, reverse=True)

    result: List[KilledProcessInfo] = []
    for proc_pid in ordered_pids:
        proc = by_pid[proc_pid]
        name = ""
        cmdline: List[str] = []
        try:
            name = proc.name() or ""
        except Exception:  # noqa: BLE001
            pass
        try:
            cmdline = proc.cmdline() or []
        except Exception:  # noqa: BLE001
            pass
        result.append(
            KilledProcessInfo(pid=proc_pid, name=name, cmdline=cmdline, source="tree")
        )

    if truncated:
        # Ticket #87 follow-up, finding F2: make the cap's effect observable
        # instead of silently dropping the excess. `len(result) ==
        # _MAX_TREE_NODES` here always -- see the corresponding
        # `len(...) >= _MAX_TREE_NODES` proxy check callers (e.g. stop())
        # use to treat this snapshot as "cannot guarantee completeness".
        _logger.warning(
            "_process_tree(pid=%s): descendant tree truncated at %s nodes -- "
            "the true tree is larger than what was collected; some "
            "descendants were not snapshotted and will not be signalled or "
            "killed by this pass",
            pid, _MAX_TREE_NODES,
        )

    return result


def _signal_process_group(pid: int, *, force: bool = False) -> bool:
    """POSIX-only: signal the process group of *pid* if it is the leader.

    ``start_new_session=True`` (used by ``_spawn_detached`` on POSIX) makes
    the spawned process the leader of its own new session and process group
    (``pgid == pid``). When that holds, this signals the *whole group* in one
    shot via ``os.killpg`` -- catching descendants that already detached into
    that same group even before an individual-PID tree snapshot could see
    them.

    Returns ``True`` if a signal was actually sent, ``False`` otherwise:
    always ``False`` on Windows (no process groups in the POSIX sense);
    ``False`` when *pid* no longer exists; ``False`` when *pid* is not the
    leader of its own group (``os.getpgid(pid) != pid``) -- signalling a
    group we did not create could hit unrelated processes; and ``False`` when
    that group is our *own* group (``os.getpgid(0)``) -- this function must
    never be able to signal the caller's own process group.
    """
    if sys.platform == "win32":
        return False
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return False
    if pgid != pid:
        return False
    try:
        own_pgid = os.getpgid(0)
    except OSError:
        own_pgid = None
    if own_pgid is not None and pgid == own_pgid:
        return False
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(pgid, sig)
    except OSError:
        return False
    return True


def _process_group_members(pid: int) -> List[int]:
    """POSIX-only: snapshot the PIDs sharing *pid*'s process group.

    Companion to :func:`_signal_process_group` -- ticket #87 follow-up
    (finding B3): that function fires a single ``os.killpg`` SIGTERM at
    *pid*'s whole process group, catching a descendant that already
    detached out of *pid*'s ppid lineage (so it is absent from
    :func:`_process_tree`) but still shares *pid*'s pgid. Nothing folded
    those group members into the subsequent force-kill step or the final
    survivor re-probe, so a stubborn same-group descendant that ignored
    that one SIGTERM could survive ``stop()`` while it still reported
    ``"stopped"``. Callers snapshot this list BEFORE
    :func:`_signal_process_group` is invoked -- same rationale as
    :func:`_process_tree` being snapshotted before any signal -- then fold
    the result into both the force-kill path and the candidate PIDs
    checked by the survivor re-probe.

    Mirrors :func:`_signal_process_group`'s own guards, with one deliberate
    divergence (ticket #110, finding #110-1): when ``os.getpgid(pid)`` raises
    ``OSError`` -- the group LEADER has already been reaped -- this function
    does NOT bail out to ``[]`` the way :func:`_signal_process_group` still
    does. ``start_new_session=True`` (used by ``_spawn_detached`` for every
    process this module spawns) guarantees the group *pid* originally
    created has ``pgid == pid``, so *pid* remains a valid probe for
    surviving members even after the leader itself has exited -- POSIX does
    not recycle a pgid number while members of that group still exist. This
    is exactly the composite/chained-shell-command case: a wrapper such as
    ``sh -c "... & long_running_child"`` can exit (reaping the leader)
    while the backgrounded child it started keeps running under a
    different, untracked pid but the SAME process group. Before this
    change, that survivor was invisible to this scan, so ``stop()`` would
    silently report ``killed_pids: []`` / ``status="stopped"`` while the
    child kept running.

    Otherwise mirrors :func:`_signal_process_group`'s guards exactly: ``[]``
    on Windows; ``[]`` when *pid* is alive but not the leader of its own
    group (``os.getpgid(pid)`` succeeds and is ``!= pid`` -- signalling/
    scanning a group we did not create could hit unrelated processes);
    ``[]`` when that group is our own process group. The host process, its
    ancestors, and *pid* itself are always excluded from the result --
    callers already handle *pid* separately.

    Known limitation -- PID reuse (accepted trade-off, same class already
    documented on ``stop()``'s own module-level docstring): if the OS has
    recycled *pid*'s integer for an unrelated process by the time this scan
    runs, ``os.getpgid(pid)`` (dead-leader branch) or the per-member
    ``os.getpgid(member_pid) == pgid`` comparisons below could match that
    unrelated process's group instead. Not newly introduced by this change --
    the identity of *pid* is trusted as-is throughout this module.

    Never raises: any psutil/OS failure degrades to a best-effort partial (or
    empty) list rather than propagating out of ``stop()``.
    """
    if sys.platform == "win32":
        return []
    try:
        pgid = os.getpgid(pid)
    except OSError:
        # Ticket #110: the leader has already been reaped -- do not bail out
        # here. start_new_session=True guarantees pgid == pid for any group
        # this module created, so pid itself remains a valid pgid probe for
        # the process_iter scan below (see the docstring above).
        pgid = pid
    else:
        if pgid != pid:
            return []
    try:
        own_pgid = os.getpgid(0)
    except OSError:
        own_pgid = None
    if own_pgid is not None and pgid == own_pgid:
        return []

    import psutil

    host_pid = os.getpid()
    excluded: "set[int]" = {host_pid, pid}
    try:
        for ancestor in psutil.Process(host_pid).parents():
            excluded.add(ancestor.pid)
    except Exception:  # noqa: BLE001 -- best-effort, never propagate
        pass

    members: List[int] = []
    try:
        for proc in psutil.process_iter(["pid"]):
            member_pid = proc.info.get("pid")
            if member_pid is None or member_pid in excluded or member_pid in members:
                continue
            try:
                if os.getpgid(member_pid) == pgid:
                    members.append(member_pid)
            except OSError:
                continue
    except Exception:  # noqa: BLE001 -- best-effort, never propagate
        return members

    return members


def _kill_process_tree(
    tree: List[KilledProcessInfo], *, timeout: float
) -> List[KilledProcessInfo]:
    """Gracefully signal, then wait/force-kill, every node in *tree*.

    *tree* is expected to already be in deepest-first order (see
    :func:`_process_tree`) so a child is always signalled before its own
    ancestor. Mirrors the budget discipline of :func:`_kill_blocking_processes`:
    each node is signalled immediately, then the remaining *timeout* budget is
    split evenly across the nodes for the subsequent wait/force-kill step;
    once the shared deadline passes, any remaining nodes are still signalled
    but no longer waited on.

    Returns *tree* unchanged so callers can use it for a post-hoc survivor
    check.
    """
    if not tree:
        return tree

    deadline = time.monotonic() + max(0.0, timeout)
    n = len(tree)

    for info in tree:
        delivered = _send_graceful_signal(info.pid, group_leader=False)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Budget exhausted. If the graceful signal was actually
            # delivered, leave this node (and all subsequent ones) merely
            # signalled, as before. If it was refused/not delivered (e.g. a
            # withheld win32 CTRL_BREAK_EVENT for a non-leader pid), there is
            # no wait budget left to fall back on -- force-kill immediately
            # instead of leaving the node running unmanaged.
            if not delivered:
                _wait_or_kill(info.pid, timeout=0.0)
            continue
        per_pid_budget = min(remaining, timeout / n)
        _wait_or_kill(info.pid, timeout=per_pid_budget)

    return tree


# ---------------------------------------------------------------------------
# Bounded background-query worker -- ticket #90
#
# _win_handle_holders' NtQueryObject calls can hang indefinitely (see that
# function's docstring). _BoundedQueryWorker is the OS-agnostic plumbing
# that bounds a single such call without ever leaking the thread that ran
# it, whether the call resolves quickly, resolves late (within a bounded
# "grace" second chance), or never resolves at all. It is deliberately
# generic -- it runs an arbitrary zero-arg callable, not anything
# ctypes/Windows-specific -- so it (and its cross-platform test coverage)
# has no dependency on running on Windows at all.
# ---------------------------------------------------------------------------

class _QueryStatus:
    """Tri-state result of a :meth:`_BoundedQueryWorker.submit` call."""

    RESOLVED = "resolved"
    ABANDONED = "abandoned"
    CAPPED = "capped"


@dataclass
class _QueryOutcome:
    """Result of one :meth:`_BoundedQueryWorker.submit` call.

    ``status`` is one of :class:`_QueryStatus`. ``value`` carries the
    submitted callable's return value when ``status == RESOLVED``; it is
    always ``None`` otherwise (a query that was not resolved in time never
    got its answer *back to the caller* -- see ``on_abandoned_done`` on
    :meth:`_BoundedQueryWorker.submit` for how a late answer is still
    delivered, out of band, once it eventually arrives).
    """

    status: str
    value: Optional[str] = None


@dataclass
class _GraceBudget:
    """Mutable per-scan pool of stage-2 grace time (ticket #90).

    A single instance is shared across every :meth:`_BoundedQueryWorker.submit`
    call made during one scan. Each call that needs stage 2 deducts its
    *actually elapsed* wait from ``remaining``, so the total extra wall
    clock a scan can ever spend across all of its slow queries combined is
    capped at whatever ``remaining`` was seeded to -- this is what makes
    stage 2 affordable where a flat larger per-query timeout was not (see
    ``_HANDLE_QUERY_GRACE_BUDGET_SEC``'s docstring): the cost is bounded
    per scan, not multiplied by the number of slow handles.
    """

    remaining: float


# Process-wide ceiling accounting (ticket #90, revised #154 item 14): how
# many _BoundedQueryWorker threads, across every concurrent
# _win_handle_holders scan AND every bounded psutil call (_bounded_call,
# including teardown's tier-1 liveness probes) in this process, are
# currently permanently blocked inside a wedged callable. One shared
# accounting for every bounded OS call in the module -- not two pools.
# Guarded by _wedged_worker_lock; incremented by submit() unconditionally,
# for every worker it retires -- ABANDONED or CAPPED alike, since both
# leave a thread genuinely and permanently blocked in fn() for as long as
# that real call takes.
#
# A one-element list, not a bare int (ticket #154 item 14, replacing the
# ticket #148 attempt-2 generation guard): submit() captures the CELL
# OBJECT ITSELF -- not a version number -- into the job's state at the
# moment it retires the worker, and _run() decrements THAT captured cell,
# unconditionally, whenever its wedged callable finally returns. A test
# fixture that wants isolation from a straggler rebinds
# `_wedged_worker_slots` to a fresh `[0]` cell (see
# tests/test_process_lifecycle.py's `_reset_handle_scan_state_for_test()`)
# rather than zeroing the old one in place -- the straggler's eventual
# decrement then lands on the cell it captured (now orphaned, read by
# nobody), never on the new cell a later scan is counting against. This
# has no production caller of a reset at all: production never replaces
# the cell, so every decrement always lands where it was counted --
# bit-identical production behaviour to a bare int, with the isolation
# problem solved structurally instead of via a version-number comparison.
_wedged_worker_slots: "List[int]" = [0]
_wedged_worker_lock = threading.Lock()


def _wedged_slot_available() -> bool:
    """``True`` iff another worker may currently be created without the
    process-wide live-blocked-thread count reaching ``_MAX_WEDGED_HANDLE_WORKERS``
    (ticket #90).

    A cheap, best-effort peek used by callers (e.g. _win_handle_holders'
    scan loop, for in-scan *replacement* workers only -- never for a scan's
    initial worker, which must always be created regardless of this check;
    see the note on ``_MAX_WEDGED_HANDLE_WORKERS`` above) to avoid spinning
    up a brand-new worker/thread that would immediately be at risk of
    pushing the live blocked-thread count over the cap. The authoritative
    check is still the atomic one inside ``submit()`` itself -- concurrent
    scans in different threads can race this peek, in which case
    ``submit()``'s own check is what actually decides ``ABANDONED`` vs
    ``CAPPED`` for that specific worker.
    """
    with _wedged_worker_lock:
        return _wedged_worker_slots[0] < _MAX_WEDGED_HANDLE_WORKERS


# Ticket #148 -- process-wide persistent query worker + scan lock, closing
# the residual per-scan daemon-thread leak. Before this fix,
# _win_handle_holders created a brand-new _BoundedQueryWorker on every
# single call ("the initial worker"), unconditionally, regardless of the
# process-wide _wedged_worker_count cap above -- see the historical "do not
# reintroduce a scan-start gate" comment inside _win_handle_holders' body
# (kept, rewritten, for the record: the gate rejected there was a *silent*
# forever-`[]` gate; what this ticket adds is a *loud* one -- see
# _acquire_scan_worker below). If that initial worker's very first query
# wedged, its thread was never itself gated by _MAX_WEDGED_HANDLE_WORKERS
# (which only ever bounded *replacement* workers created mid-scan, inside
# _bounded_query) and was never joined (a wedged thread cannot be joined) --
# so a long-lived host process making many remove()/_find_blocking_processes
# calls over its life accumulated one leaked daemon thread per affected
# call, unbounded.
#
# The fix: exactly one persistent worker slot, shared by every scan in this
# process, guarded by _handle_scan_lock, which also fully serializes scans
# -- only one _win_handle_holders call may be mid-flight at a time. A
# healthy worker parks on queue.get() between scans and is reused
# indefinitely by every later scan -- the "+1" in
# _HANDLE_SCAN_MAX_LIVE_WORKERS below. _persistent_query_worker is only
# ever read/written while holding _handle_scan_lock.
_handle_scan_lock = threading.Lock()
_persistent_query_worker: "Optional[_BoundedQueryWorker]" = None

# Ceiling on how long a scan waits to acquire _handle_scan_lock before
# giving up and reporting handle_scan:busy (ticket #148). The actual wait
# passed to Lock.acquire(True, wait) is additionally clamped by the scan's
# own budget_sec via scan_deadline -- see _win_handle_holders -- so a scan
# started with little budget left never waits longer than that budget, even
# though this constant nominally allows up to 2s.
_HANDLE_SCAN_LOCK_WAIT_MAX_SEC = 2.0

# Bound on how many query-worker daemon threads may be alive process-wide at
# once (ticket #148): the one persistent worker, plus however many
# replacement workers _MAX_WEDGED_HANDLE_WORKERS allows to be wedged
# concurrently.
_HANDLE_SCAN_MAX_LIVE_WORKERS = 1 + _MAX_WEDGED_HANDLE_WORKERS


def _acquire_scan_worker() -> "Optional[_BoundedQueryWorker]":
    """Return the worker this scan should use (ticket #148).

    Caller must hold ``_handle_scan_lock``. Reuses the live persistent
    worker if one already exists; otherwise creates one, but only when the
    process-wide wedged-thread count has not yet reached
    ``_MAX_WEDGED_HANDLE_WORKERS`` -- i.e. only when creating it cannot by
    itself push that count over the cap. Returns ``None`` when the slot is
    empty and no capacity remains: the caller must then refuse the scan
    (``handle_scan:capped``) loudly, before ever dumping the handle table,
    rather than run the (heavy) dump only to immediately re-hit the same
    cap on the first query.

    Deliberately checks ``_wedged_worker_count`` against
    ``_MAX_WEDGED_HANDLE_WORKERS`` directly here, rather than calling
    ``_wedged_slot_available()`` (which does the same comparison) -- this
    keeps that helper's established, narrower role scoped to exactly what
    it already gated before this ticket: whether ``_bounded_query`` may
    create a *replacement* worker mid-scan (see ``_replace_scan_worker``
    below). Sharing one function for both would mean a caller that peeks
    or patches ``_wedged_slot_available()`` to reason about replacement
    capacity alone would, as an unintended side effect, also change
    whether a scan is allowed to start at all -- two genuinely different
    decisions that happen to use the same arithmetic.
    """
    global _persistent_query_worker
    if _persistent_query_worker is not None:
        return _persistent_query_worker
    if _wedged_worker_slots[0] >= _MAX_WEDGED_HANDLE_WORKERS:
        return None
    _persistent_query_worker = _BoundedQueryWorker()
    return _persistent_query_worker


def _replace_scan_worker() -> "Optional[_BoundedQueryWorker]":
    """Replace a just-wedged persistent worker with a fresh one, if capacity
    allows (ticket #148).

    Caller must hold ``_handle_scan_lock``. Always clears the module slot
    first -- the previous worker is permanently retired (see
    ``_BoundedQueryWorker.submit``'s ABANDONED/CAPPED outcomes and
    ``_bounded_query`` below) -- then creates and stores a replacement only
    when ``_wedged_slot_available()``, else stores (and returns) ``None``.
    This keeps the module slot always holding the truth on scan exit: the
    next scan's ``_acquire_scan_worker`` call sees exactly what this one
    left behind.
    """
    global _persistent_query_worker
    _persistent_query_worker = None
    if _wedged_slot_available():
        _persistent_query_worker = _BoundedQueryWorker()
    return _persistent_query_worker


def _clear_scan_worker_slot() -> None:
    """Clear the process-wide persistent-worker slot without attempting to
    create a replacement (ticket #148).

    Caller must hold ``_handle_scan_lock``. Used for a genuinely ``CAPPED``
    outcome (``_QueryStatus.CAPPED``) -- ``submit()`` itself already decided
    that no replacement worker may be attempted at all (this worker's own
    slot claim filled the last available process-wide capacity), so unlike
    ``_replace_scan_worker`` this never re-checks ``_wedged_slot_available()``
    or creates a new worker; it only ever empties the slot.
    """
    global _persistent_query_worker
    _persistent_query_worker = None


# Ticket #154 (item 14): the test-only `_reset_handle_scan_state()` helper
# that used to live here is deleted outright -- production must not ship a
# test-only entry point. Its replacement,
# `_reset_handle_scan_state_for_test()`, lives in
# tests/test_process_lifecycle.py and rebuilds `_wedged_worker_slots` (a
# fresh cell) rather than zeroing a shared counter -- see that function and
# the cell-capture design documented on `_wedged_worker_slots` above.

# Ticket #121 -- process-wide registry of kernel objects whose NtQueryObject
# call has already wedged at least once, so a *later* scan never re-queries
# the same object and never risks leaking another permanently-blocked
# thread for it. This is the residual half of #90: #90 bounded only
# *replacement*-worker creation once a scan's own worker had already
# wedged (see _wedged_worker_count/_wedged_slot_available above); it did
# nothing to stop a *fresh* scan's *initial* worker from wedging on the
# exact same pathological handle every time _win_handle_holders is called
# again (one leaked thread per affected call, unbounded over a long-lived
# host process's lifetime).
#
# Memory policy (deliberate, not a placeholder): a key is remembered
# permanently -- never expired by time or by scan -- and the only eviction
# is FIFO once the container exceeds _MAX_REMEMBERED_WEDGED_OBJECTS. This
# gives the strongest available plateau guarantee: a given pathological
# kernel object can cost at most one leaked thread per concurrently-running
# scan that races the first wedge against it -- converging to a true
# process-wide one-leaked-thread plateau the moment any one of those racing
# scans records the key -- no matter how many later, non-overlapping
# _win_handle_holders calls are made -- *for as long as its key remains in
# the registry* (see the FIFO caveat below). This is the same class of
# bounded overshoot already accepted for _wedged_worker_count above (each
# concurrently-running scan may hold one worker beyond the cap, bounded by
# the small number of concurrent scans): _process_handle's
# _is_known_wedging_handle check and the matching _remember_wedging_handle
# call inside _bounded_query are seconds apart -- spanning the query's
# timeout+grace window, not atomic together -- so two scans already in
# flight against the same object when it first wedges can both observe
# "not yet known", both submit, and each leak its own worker before either
# records. The accepted cost: a handle that was merely very slow once
# (rather than genuinely, permanently wedged) is skipped by every later scan
# for the remaining life of the process (or until its key is evicted), so
# that handle's holder may be missed by _find_blocking_processes from then
# on. This is deliberately preferred over unbounded thread growth, and it is
# strictly narrower than the scan-start gate rejected in
# _win_handle_holders' docstring (which would silently return `[]` for
# *every* handle, forever, the moment the unrelated
# _MAX_WEDGED_HANDLE_WORKERS cap filled) -- this registry only ever
# suppresses the *specific* objects that have individually proven themselves
# wedging, nothing else. The FIFO bound also means the registry itself
# cannot grow without limit even if a long-lived process churns through many
# thousands of distinct wedging objects over its life -- but that same FIFO
# eviction is a residual gap, not just a memory safeguard: once a process
# has churned through more than _MAX_REMEMBERED_WEDGED_OBJECTS *distinct*
# wedging objects, the oldest-recorded key is evicted and, if that same
# object wedges again later, it can re-admit exactly one further leaked
# thread for it. So the true guarantee is "at most one leaked thread per
# distinct wedging object, per admission window between two evictions of its
# key" -- not an unconditional one-ever bound. This is accepted for the same
# reason as the slow-handle cost above: it only degrades a best-effort,
# last-resort diagnostic pass (_find_blocking_processes may miss one
# holder), whereas the unbounded thread leak this ticket fixes degrades the
# whole host process.
#
# Key-collision caveats (both accepted, for the same best-effort-diagnostic
# reason as above -- neither is worth the cost of making the fallback
# non-permanent, which would reinstate the unbounded leak for exactly the
# non-elevated callers this ticket targets): the ``("pid", pid,
# handle_value, type_index)`` fallback key (see _wedging_handle_key) can
# still alias a different, unrelated object if a handle value is recycled
# by the OS onto a new object of the *same* type within the *same* pid,
# permanently suppressing that unrelated, healthy handle. The preferred
# ``("obj", object_ptr)`` key is likewise theoretically susceptible to the
# kernel reusing a freed object's pointer for a new, unrelated object, far
# rarer in practice than handle-value reuse. The fallback key is equally
# susceptible to PID reuse: Windows recycles PIDs just as it recycles handle
# values, so a short-lived process that wedges a handle once, exits, and is
# later replaced by an unrelated process that the OS happens to assign the
# same PID -- which then opens a handle with the same (handle_value,
# type_index) -- will be suppressed too, even though it is a different
# process holding a different, unrelated kernel object. As with the
# handle-value-reuse case above, this fails in the safe direction: it only
# ever causes extra suppression of this best-effort diagnostic pass (a
# healthy handle silently skipped), never extra thread leaking.
_MAX_REMEMBERED_WEDGED_OBJECTS = 4096

# Insertion-ordered bounded set (values are unused, always None) of keys
# identifying kernel objects that have wedged NtQueryObject at least once.
# See _wedging_handle_key for how a key is derived from one handle-table
# entry, and _remember_wedging_handle/_is_known_wedging_handle for the only
# two operations performed against it. Guarded by _wedged_object_lock.
# Deliberately pure Python (no ctypes) so it -- and the suppression/
# recording logic around it -- is exercised by CI on every platform, not
# just Windows.
_wedged_object_keys: "OrderedDict[tuple, None]" = OrderedDict()
_wedged_object_lock = threading.Lock()


def _wedging_handle_key(
    pid: int, handle_value: int, object_ptr: int, type_index: int
) -> tuple:
    """Derive the dedup key used by the wedged-handle registry (ticket #121).

    Prefers ``("obj", object_ptr)`` -- identifying the underlying kernel
    object itself, independent of which process/handle-value currently
    references it -- whenever the handle table entry's raw ``Object``
    pointer is non-zero. Falls back to ``("pid", pid, handle_value,
    type_index)`` when it is zero, which is the common case for a
    non-elevated caller on modern Windows: the kernel zeroes ``Object`` in
    ``SYSTEM_HANDLE_TABLE_ENTRY_INFO_EX`` for callers without the
    ``SeDebugPrivilege``-equivalent elevation, as a kernel-pointer-leak
    mitigation. This fallback is required, not merely defensive -- without
    it, the registry would silently never suppress anything on an ordinary
    unprivileged host process, which is the normal case this ticket exists
    to fix.

    ``type_index`` is included in the fallback key -- not just ``(pid,
    handle_value)`` -- because Windows recycles small handle values almost
    immediately after ``CloseHandle``: without it, a future, completely
    unrelated object landing on the same now-reused handle value in the
    same pid would be silently and permanently suppressed too, which is a
    wider and more common failure mode than the single-object cost this
    registry already accepts (see the comment above
    ``_MAX_REMEMBERED_WEDGED_OBJECTS``). Including the type index narrows
    that aliasing to a recycled handle value that also happens to land on
    an object of the *same* type -- it does not eliminate it; see that same
    comment block for the accepted residual risk.

    The ``pid`` component of the fallback key is subject to the same kind
    of reuse: Windows recycles PIDs, not just handle values, so this key
    can also alias across two entirely unrelated *processes* if the OS
    later reassigns a wedged, now-exited process's PID to a new process
    that happens to open a handle with the same ``(handle_value,
    type_index)`` -- see the same comment block above
    ``_MAX_REMEMBERED_WEDGED_OBJECTS``, which names this PID-reuse vector
    explicitly alongside handle-value reuse. Like that vector, it fails in
    the safe direction: extra suppression of a diagnostic pass, never extra
    leaking.
    """
    if object_ptr:
        return ("obj", object_ptr)
    return ("pid", pid, handle_value, type_index)


def _remember_wedging_handle(key: tuple) -> None:
    """Record *key* as belonging to a kernel object that has wedged
    NtQueryObject at least once (ticket #121), so no later scan -- in this
    or any future call to _win_handle_holders -- re-queries it.

    Bounded FIFO: once the registry exceeds _MAX_REMEMBERED_WEDGED_OBJECTS,
    the oldest-recorded key is evicted first. See the constant's own
    comment for the accepted-cost rationale.
    """
    with _wedged_object_lock:
        if key not in _wedged_object_keys:
            _wedged_object_keys[key] = None
        while len(_wedged_object_keys) > _MAX_REMEMBERED_WEDGED_OBJECTS:
            _wedged_object_keys.popitem(last=False)


def _is_known_wedging_handle(key: tuple) -> bool:
    """``True`` iff *key* was previously recorded by
    :func:`_remember_wedging_handle` and has not since been evicted."""
    with _wedged_object_lock:
        return key in _wedged_object_keys


class _BoundedQueryWorker:
    """One daemon thread that runs submitted zero-arg callables, bounded.

    Owns exactly one ``queue.Queue`` and one daemon ``threading.Thread``.
    :meth:`submit` dispatches a callable to that thread and waits for it,
    first for up to ``_HANDLE_QUERY_TIMEOUT_SEC`` (stage 1), then --  if a
    *grace* budget was supplied and has room -- for a further bounded
    "second chance" (stage 2, ``_HANDLE_QUERY_GRACE_SEC`` at most, further
    clamped by the caller's *scan_deadline* and by however much of *grace*
    remains). :meth:`close` always shuts the thread down cleanly.

    Every worker, whether it turns out healthy or wedged, is guaranteed to
    eventually receive its own shutdown sentinel and exit -- a *healthy*
    worker via :meth:`close` (called by callers, e.g. in a ``finally``, once
    they are done with it), a worker whose current job goes unresolved past
    both stages via ``submit`` itself, immediately at the moment it gives
    up waiting (the job is already the only thing dispatched to that
    worker's queue at that point, so posting the sentinel right behind it is
    safe -- once the wedged callable finally returns, the worker's loop
    drains straight to that sentinel and exits, instead of parking on
    ``queue.get()`` forever).

    That "eventually" is doing real work, though: a genuinely wedged
    callable (e.g. ``NtQueryObject`` against a named pipe with no
    listener) may never return, in which case this worker's thread stays
    alive -- blocked inside the callable, sentinel or not -- for as long as
    the process runs. Ticket #90 alone therefore does *not* guarantee that
    a long-lived caller creating many of these over its lifetime never
    accumulates threads without bound: nothing here stops the *same*
    callable from wedging again on the next worker created for it. What
    actually bounds that (ticket #121) lives one level up, in the caller --
    see ``_win_handle_holders``' module-level wedged-handle registry, which
    remembers a wedging target so no later caller submits the same callable
    to a fresh worker again *for as long as that target's key remains
    recorded* -- the registry is bounded FIFO (see
    ``_MAX_REMEMBERED_WEDGED_OBJECTS``), so eviction after enough distinct
    wedging targets have been recorded can re-admit one further wedge for a
    target evicted earlier; it is not an unconditional guarantee across the
    process's entire lifetime. This class's own guarantee is narrower, and
    still accurate: every worker it creates eventually receives its
    shutdown sentinel exactly once, and exits on its own the moment its
    callable returns -- it is the caller's responsibility not to keep
    resubmitting to the same wedging target.
    """

    def __init__(self) -> None:
        self._job_queue: "queue.Queue" = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._job_queue.get()
            if item is None:
                return
            fn, state = item
            try:
                value = fn()
            except Exception:  # noqa: BLE001 -- best-effort, never propagate
                value = None
            with state["lock"]:
                state["value"] = value
                state["completed"] = True
                was_abandoned = state["abandoned"]
                callback = state["on_abandoned_done"] if was_abandoned else None
                slot_acquired = state["slot_acquired"]
                slot_cell = state["slot_cell"]
            state["done"].set()
            if was_abandoned:
                # on_abandoned_done fires unconditionally -- it closes the
                # duplicated kernel handle this worker's wedged call was
                # holding; gating it would leak that handle.
                if callback is not None:
                    try:
                        callback(value)
                    except Exception:  # noqa: BLE001 -- best-effort cleanup
                        pass
                if slot_acquired:
                    # Decrement the CELL this slot claim captured at
                    # submit() time (ticket #154 item 14) -- not whatever
                    # `_wedged_worker_slots` currently points at. A test
                    # fixture that rebinds `_wedged_worker_slots` to a fresh
                    # cell between tests/scans leaves this straggler
                    # decrementing the orphaned old cell (read by nobody),
                    # never corrupting the new cell a later scan is
                    # counting against -- no version-number comparison
                    # needed, because object identity already gives the
                    # same guarantee structurally.
                    with _wedged_worker_lock:
                        slot_cell[0] = max(0, slot_cell[0] - 1)

    def submit(
        self,
        fn,
        *,
        grace: Optional[_GraceBudget] = None,
        scan_deadline: Optional[float] = None,
        on_abandoned_done=None,
        timeout: float = _HANDLE_QUERY_TIMEOUT_SEC,
    ) -> _QueryOutcome:
        """Run *fn* (a zero-arg callable) through this worker, bounded.

        *timeout* (ticket #154, item 11) overrides the stage-1 wait,
        defaulting to ``_HANDLE_QUERY_TIMEOUT_SEC`` (0.01s, tuned for a
        single NtQueryObject call on an already-duplicated handle) so Pass
        1c's own callers are unaffected byte-for-byte. A caller bounding a
        blocking psutil read (via :func:`_bounded_call`) passes
        ``_BLOCKING_CALL_TIMEOUT_SEC`` instead -- applying the handle-query
        timeout to a psutil call would misclassify nearly every healthy
        read as wedged.

        Returns a :class:`_QueryOutcome`:

        - ``RESOLVED`` -- *fn* returned within stage 1 or stage 2; ``value``
          carries its return value (or ``None`` if it raised).
        - ``ABANDONED`` -- *fn* did not resolve within either stage. This
          worker is now retired (see the class docstring) -- callers must
          not submit to it again. *on_abandoned_done*, if given, is invoked
          (on the worker's own thread) with *fn*'s eventual return value
          once it finally returns. This worker's own thread is counted
          against the process-wide ``_wedged_worker_count`` for as long as
          it remains blocked in *fn* (decremented by ``_run()`` the moment
          *fn* finally returns), and the cap had room left *after* counting
          it in -- so a caller may safely create one more replacement
          worker for this scan's next query.
        - ``CAPPED`` -- *fn* did not resolve within either stage, and this
          worker's own thread -- counted in exactly the same way as the
          ABANDONED case above, including *on_abandoned_done* still being
          invoked once *fn* finally returns -- filled the last available
          process-wide slot (or the cap was already full before it).
          ``CAPPED`` differs from ``ABANDONED`` *only* in that signal to
          the caller: do not create a replacement worker for this scan's
          next query right now. It never means this worker's own thread
          goes uncounted -- every non-resolved outcome is tracked and
          released the same way, or ``_MAX_WEDGED_HANDLE_WORKERS`` would
          not actually bound the number of live blocked threads.

        Calling ``submit`` on an already-closed/retired worker returns
        ``ABANDONED`` immediately without running *fn* at all, rather than
        raising.
        """
        if self._closed:
            return _QueryOutcome(_QueryStatus.ABANDONED, None)

        state = {
            "lock": threading.Lock(),
            "done": threading.Event(),
            "value": None,
            "completed": False,
            "abandoned": False,
            "on_abandoned_done": on_abandoned_done,
            "slot_acquired": False,
            "slot_cell": None,
        }
        self._job_queue.put((fn, state))

        if state["done"].wait(timeout):
            return _QueryOutcome(_QueryStatus.RESOLVED, state["value"])

        # Stage 2: a single bounded "second chance", drawn from the shared
        # per-scan grace pool and clamped by whatever remains of the
        # caller's own scan deadline -- neither of which this single query
        # may exceed.
        allowance = 0.0
        if grace is not None:
            allowance = min(_HANDLE_QUERY_GRACE_SEC, grace.remaining)
            if scan_deadline is not None:
                allowance = min(allowance, scan_deadline - time.monotonic())
            allowance = max(0.0, allowance)

        if allowance > 0:
            t0 = time.monotonic()
            got = state["done"].wait(allowance)
            elapsed = time.monotonic() - t0
            grace.remaining = max(0.0, grace.remaining - elapsed)
            if got:
                return _QueryOutcome(_QueryStatus.RESOLVED, state["value"])

        # Neither stage resolved it -- this worker's thread is now
        # permanently tied up in *fn*. Decide retirement atomically against
        # _run() possibly completing the job at this exact moment.
        with state["lock"]:
            if state["completed"]:
                return _QueryOutcome(_QueryStatus.RESOLVED, state["value"])
            # Every retiring worker's thread is genuinely, permanently
            # blocked in *fn* for as long as that real call takes --
            # regardless of whether the process-wide cap already had room.
            # It must therefore always be counted in here (and decremented
            # by _run() once fn() finally returns), or the cell stops
            # reflecting the true number of live blocked threads and
            # _MAX_WEDGED_HANDLE_WORKERS stops being a real bound on them.
            # The cap only ever governs whether a *replacement* worker may
            # be created for this scan's next query -- that decision is the
            # CAPPED vs ABANDONED distinction below, checked *after* this
            # worker's own slot is already claimed.
            with _wedged_worker_lock:
                _wedged_worker_slots[0] += 1
                at_cap = _wedged_worker_slots[0] >= _MAX_WEDGED_HANDLE_WORKERS
                # Capture the CELL OBJECT itself (ticket #154 item 14) --
                # not a version number -- read under the same lock as the
                # count increment above, so _run() later decrements exactly
                # this cell regardless of whatever `_wedged_worker_slots`
                # points at by the time this worker's wedged call finally
                # returns.
                state["slot_cell"] = _wedged_worker_slots
            state["abandoned"] = True
            state["slot_acquired"] = True

        # Retire this worker now: it must never process another job, and
        # must never park on queue.get() forever once *fn* finally returns.
        # The queue is otherwise empty at this point -- the only job ever
        # put on it is the one still running -- so posting the sentinel now
        # guarantees _run() drains straight to it next.
        self._closed = True
        self._job_queue.put(None)

        if at_cap:
            return _QueryOutcome(_QueryStatus.CAPPED, None)
        return _QueryOutcome(_QueryStatus.ABANDONED, None)

    def close(self) -> None:
        """Shut this worker down. Idempotent; never raises.

        A healthy worker (no job in flight, or its most recent job resolved
        normally) drains to the sentinel and exits almost immediately. A
        worker already retired by :meth:`submit` has already been sent its
        own sentinel -- this is then a no-op beyond the bounded join, which
        only succeeds once the wedged call eventually returns.
        """
        if not self._closed:
            self._closed = True
            self._job_queue.put(None)
        self._thread.join(_WORKER_CLOSE_JOIN_SEC)


# NTSTATUS values used by both _enumerate_handle_table and _query_object_raw
# below. Plain ints -- no ctypes needed to define these -- so, unlike the
# structures below, they are simple module constants rather than living
# behind the lazy-ctypes-import memoised builder.
_STATUS_SUCCESS = 0x00000000
_STATUS_INFO_LENGTH_MISMATCH = 0xC0000004

# SystemInformationClass value for NtQuerySystemInformation's
# SystemExtendedHandleInformation dump, used by _enumerate_handle_table.
_SYSTEM_EXTENDED_HANDLE_INFORMATION = 64

# Memoised (ticket #121) module-global cache for the two ctypes.Structure
# subclasses _enumerate_handle_table and _query_object_raw both need.
# Populated lazily by _win_handle_structs() below.
_win_handle_structs_cache: "Optional[Tuple[type, type]]" = None


def _win_handle_structs() -> "Tuple[type, type]":
    """Return ``(SystemHandleTableEntryInfoEx, UnicodeString)`` ctypes
    structures, building and caching them on first call (ticket #121).

    Both structures used to be defined fresh, as local classes, on every
    single call to ``_win_handle_holders`` -- harmless when they were only
    ever constructed once per scan, but ``_query_object_raw`` is now called
    once per handle after being hoisted to module scope, so redefining
    ``ctypes.Structure`` subclasses on every call would be wasteful. This
    builder imports ``ctypes`` itself, on first use only, preserving this
    module's "no ctypes import at module load time" invariant (the module
    is imported and exercised by the test suite on non-Windows CI too,
    where ``ctypes.wintypes``/``ctypes.WinDLL`` are unavailable). The cache
    assignment is idempotent -- two threads racing this on first use may
    each build an equivalent pair of classes and one simply overwrites the
    other's assignment, which is harmless since neither pair carries any
    shared mutable state -- so no lock is used here.
    """
    global _win_handle_structs_cache
    if _win_handle_structs_cache is not None:
        return _win_handle_structs_cache

    import ctypes

    class _SystemHandleTableEntryInfoEx(ctypes.Structure):
        _fields_ = [
            ("Object", ctypes.c_void_p),
            ("UniqueProcessId", ctypes.c_size_t),
            ("HandleValue", ctypes.c_size_t),
            ("GrantedAccess", ctypes.c_ulong),
            ("CreatorBackTraceIndex", ctypes.c_ushort),
            ("ObjectTypeIndex", ctypes.c_ushort),
            ("HandleAttributes", ctypes.c_ulong),
            ("Reserved", ctypes.c_ulong),
        ]

    class _UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", ctypes.c_ushort),
            ("MaximumLength", ctypes.c_ushort),
            ("Buffer", ctypes.c_wchar_p),
        ]

    _win_handle_structs_cache = (_SystemHandleTableEntryInfoEx, _UnicodeString)
    return _win_handle_structs_cache


def _enumerate_handle_table(
    ntdll, excluded_pids: "set[int]"
) -> "Optional[Dict[int, List[Tuple[int, int, int, int]]]]":
    """Dump the system-wide handle table, grouped by owning PID.

    Hoisted to module scope (ticket #121) from ``_win_handle_holders``' old
    inline Step 1 so it is directly mockable in tests (e.g. to inject a
    fixed, single-entry table without touching the real system handle
    table). Behaviour is unchanged: grows the ``NtQuerySystemInformation``
    buffer on ``STATUS_INFO_LENGTH_MISMATCH`` with bounded retries, and
    returns ``None`` -- never an empty dict -- when the one-shot dump
    itself never succeeds, so the caller can tell "found nothing" apart
    from "never looked" (mapped by the caller to
    ``_PartialList([], complete=False)``).

    Each grouped tuple is ``(handle_value, type_index, object_ptr,
    granted_access)`` -- widened (ticket #148) from the ticket #121 3-tuple
    ``(handle_value, type_index, object_ptr)`` to also carry the handle
    table entry's raw ``GrantedAccess`` mask (the struct already declared
    this field; only the emitted tuple dropped it), which
    ``_process_handle`` uses as a type-probe pre-filter for the specific
    ``GrantedAccess`` bitmask observed to be hang-prone
    (``_HANG_PRONE_GRANTED_ACCESS``). ``object_ptr`` remains the
    wedged-handle registry's primary dedup key (see ``_wedging_handle_key``),
    falling back to ``(pid, handle_value, type_index)`` only when it is
    zero.
    """
    import ctypes

    _SystemHandleTableEntryInfoEx, _ = _win_handle_structs()

    buf_size = 1 << 20  # 1 MiB initial guess
    buf = None
    for _attempt in range(8):
        buf = ctypes.create_string_buffer(buf_size)
        return_length = ctypes.c_ulong(0)
        status = ntdll.NtQuerySystemInformation(
            _SYSTEM_EXTENDED_HANDLE_INFORMATION,
            buf,
            buf_size,
            ctypes.byref(return_length),
        ) & 0xFFFFFFFF
        if status == _STATUS_INFO_LENGTH_MISMATCH:
            buf_size = max(buf_size * 2, return_length.value + (1 << 16))
            continue
        if status != _STATUS_SUCCESS:
            # The one-shot dump itself failed -- the caller never looked at
            # any handle at all, not merely "found nothing".
            return None
        break
    else:
        return None

    size_t_size = ctypes.sizeof(ctypes.c_size_t)
    entry_size = ctypes.sizeof(_SystemHandleTableEntryInfoEx)
    handles_offset = 2 * size_t_size
    num_handles = ctypes.c_size_t.from_buffer_copy(buf, 0).value
    # Defend against a corrupt/short buffer reporting more handles than it
    # actually holds -- clamp rather than read out of bounds.
    max_fit = max(0, (buf_size - handles_offset) // entry_size)
    num_handles = min(num_handles, max_fit)

    # Group (handle value, object type index, object pointer, granted
    # access) quadruples by owning PID so each foreign process is opened
    # (OpenProcess) at most once regardless of how many of its handles end
    # up inspected.
    by_pid: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for i in range(num_handles):
        offset = handles_offset + i * entry_size
        entry = _SystemHandleTableEntryInfoEx.from_buffer_copy(buf, offset)
        pid = int(entry.UniqueProcessId)
        if pid <= 0 or pid in excluded_pids:
            continue
        by_pid.setdefault(pid, []).append(
            (
                int(entry.HandleValue),
                int(entry.ObjectTypeIndex),
                int(entry.Object or 0),
                int(entry.GrantedAccess or 0),
            )
        )
    return by_pid


def _query_object_raw(ntdll, dup_handle, info_class: int) -> Optional[str]:
    """Run one raw ``NtQueryObject`` call and parse its ``UNICODE_STRING``
    result.

    Hoisted to module scope (ticket #121) from a closure defined fresh
    inside every ``_win_handle_holders`` scan, so it is directly mockable/
    countable in tests without a live ``ctypes``/``ntdll`` call. Behaviour
    is unchanged -- only ``ntdll`` became an explicit parameter (previously
    captured from the enclosing scope), and the ``UNICODE_STRING`` ctypes
    structure it parses now comes from the memoised
    :func:`_win_handle_structs` instead of being redefined on every call.
    """
    import ctypes

    _, _UnicodeString = _win_handle_structs()

    size = 1024
    for _attempt in range(4):
        name_buf = ctypes.create_string_buffer(size)
        returned = ctypes.c_ulong(0)
        status = ntdll.NtQueryObject(
            dup_handle,
            info_class,
            name_buf,
            size,
            ctypes.byref(returned),
        ) & 0xFFFFFFFF
        if status == _STATUS_SUCCESS:
            # Both ObjectNameInformation and ObjectTypeInformation begin
            # with a UNICODE_STRING as their first field, so the same
            # parsing applies to either info_class.
            uni = _UnicodeString.from_buffer_copy(name_buf, 0)
            if not uni.Buffer or uni.Length == 0:
                return None
            return ctypes.wstring_at(uni.Buffer, uni.Length // 2)
        if status == _STATUS_INFO_LENGTH_MISMATCH:
            size = max(size * 2, returned.value + 256)
            continue
        return None
    return None


def _win_handle_holders(
    path: str,
    excluded_pids: "set[int]",
    *,
    budget_sec: float = _HANDLE_SCAN_BUDGET_SEC,
) -> "_PartialList":
    """Windows-only: system-wide OS handle-table scan (ticket #71 — Pass 1c).

    Returns ``(pid, name)`` pairs for processes holding an *open OS handle*
    to a file at or under *path*, catching processes that evade both
    ``proc.cwd()`` and ``proc.open_files()`` (Pass 1/2) as well as the
    cmdline-token scan (Pass 1b, ticket #57) — e.g. a process launched via
    ``Start-Process -WorkingDirectory <worktree_dir>`` whose real OS-level
    cwd is inside the worktree, but which raises ``psutil.AccessDenied`` for
    both ``cwd()`` and ``open_files()`` and has no worktree path as a
    cmdline token.

    Implementation, using raw ``ctypes`` calls into ``ntdll``/``kernel32``
    (no new pip dependency — no ``pywin32``), mirroring this module's and
    ``yaml_store.py``'s existing raw-ctypes style:

    1. ``NtQuerySystemInformation(SystemExtendedHandleInformation, ...)``
       dumps every open handle on the system as ``(pid, handle value)``
       pairs.
    2. For each owning PID (excluding *excluded_pids*), ``OpenProcess`` with
       ``PROCESS_DUP_HANDLE`` is attempted so its handles can be duplicated
       into our own process via ``DuplicateHandle`` — this is required
       before the handle can be queried. When ``OpenProcess`` is denied
       (elevated or other-user process, and we are not elevated ourselves)
       that PID's handles are skipped entirely: this is a hard OS
       permission boundary, not a bug, and is a residual limitation of this
       pass (see ``_find_blocking_processes``).
    3. Each duplicated handle is resolved to its underlying NT device path
       via ``NtQueryObject(ObjectNameInformation)``, then translated to a
       drive-letter path via a ``QueryDosDevice``-built device map, and
       compared against the normalized *path* using the same
       ``normalized + os.sep`` prefix-boundary check the other passes use
       (so a sibling directory sharing a name prefix cannot false-match).

    Performance: a full system handle table routinely holds 100k+ entries
    (observed ~143k on an ordinary dev machine), and only a small fraction
    are file handles -- resolving *every* handle's name would be far too
    slow for a removal-path check. Each handle's ``ObjectTypeIndex`` (read
    directly out of the system handle table, no extra syscall) is looked up
    in a per-call cache keyed by type index: the *first* handle seen for a
    given type index is duplicated and probed once via
    ``NtQueryObject(ObjectTypeInformation)`` to learn its type name (e.g.
    ``"File"``, ``"Event"``, ``"Key"``); the result is cached, and every
    subsequent handle sharing that type index is accepted or skipped from
    the cache alone with no further ``DuplicateHandle``/``NtQueryObject``
    call. Only handles resolved as type ``"File"`` are individually
    name-queried and path-compared. This typically prunes well over 90% of
    the handle table before any per-handle name resolution is attempted.

    ``NtQueryObject`` is documented to hang indefinitely for certain handle
    types (named pipes with no listener, in particular). There is no
    OS-level way to cancel a blocked kernel call, so each query -- both the
    one-per-type-index type probe and the per-handle name resolution -- is
    dispatched through a single reusable :class:`_BoundedQueryWorker`
    background thread per scan (rather than spawning a new OS thread per
    handle -- thread-creation overhead alone made a naive one-thread-per-
    handle implementation take upwards of tens of minutes against a normal
    handle table, confirmed impractically slow in practice).

    Each query first gets ``_HANDLE_QUERY_TIMEOUT_SEC`` (stage 1); one that
    does not resolve in time gets a second, bounded chance
    (``_HANDLE_QUERY_GRACE_SEC``, stage 2) drawn from a small per-scan grace
    pool (``_HANDLE_QUERY_GRACE_BUDGET_SEC`` total, clamped by
    ``scan_deadline``) that recovers merely-slow handles without the
    unbounded-multiplication cost ticket #71 found with a flat larger
    per-query timeout -- see the constants' own docstrings. A query that
    still has not resolved after both stages is treated as wedged: the
    worker that ran it is retired (ticket #90) rather than abandoned in
    place -- it is sent its own shutdown sentinel immediately, so it exits
    on its own the moment the wedged call finally returns instead of
    parking on ``queue.get()`` forever, and a fresh worker takes over for
    subsequent queries, up to a process-wide ceiling of
    ``_MAX_WEDGED_HANDLE_WORKERS`` outstanding retired workers. Once that
    ceiling is hit, the scan stops issuing further queries and returns
    whatever it already found, exactly like exhausting ``scan_deadline``.
    Ownership of the underlying duplicated handle transfers to the retired
    worker in this case too (via an ``on_abandoned_done`` callback that
    closes it once the wedged call returns) -- the scan loop itself must
    never close a handle whose query it gave up waiting on, since Windows
    can recycle that same handle *value* onto a different, unrelated object
    in the meantime (a real use-after-close/false-match bug fixed by this
    same change). Every worker created during a scan -- healthy or retired
    -- is guaranteed to eventually receive a shutdown sentinel; a *healthy*
    worker then exits essentially immediately. A *retired* one does not: its
    thread is genuinely, permanently blocked inside the still-running
    wedged ``NtQueryObject`` call for as long as that real call takes --
    which, for a handle type documented to hang indefinitely (named pipes
    with no listener, in particular), may be the remaining life of the
    process. Ticket #90 alone did not make that safe to repeat: a retired
    worker's thread is real and outstanding regardless of how quickly this
    *scan* itself returns, and a long-lived host process that called this
    function again against the *same* pathological handle would spin up a
    fresh initial worker that wedged on it all over again -- one leaked
    thread per affected call, unbounded (the residual gap ticket #121
    closes). What actually bounds this now is the module-level wedged-
    handle registry (see ``_wedging_handle_key`` /
    ``_remember_wedging_handle`` / ``_is_known_wedging_handle`` above
    ``_BoundedQueryWorker``): the moment a query against a given kernel
    object fails to resolve, that object's key is recorded, and every
    *later*, non-overlapping call to this function skips it outright in
    ``_process_handle``, before ever duplicating or querying it again, for
    as long as that key remains in the registry. So a single pathological
    kernel object can leak at most one permanently-blocked thread per
    distinct object *per concurrently-running scan that races the first
    wedge against it* -- converging to one leaked thread process-wide the
    moment any one of those racing scans records the key -- *for as long as
    its key remains in the registry*, not one per call. This is the same
    class of bounded overshoot already accepted for
    ``_wedged_worker_count`` above (each concurrently-running scan may hold
    one worker beyond the cap, bounded by the small number of concurrent
    scans): the check in ``_process_handle`` and the record in
    ``_bounded_query`` are seconds apart, spanning the query's
    timeout+grace window rather than being atomic together, so two scans
    already in flight against the same object when it first wedges can
    both observe "not yet known" and both submit, each leaking its own
    worker before either records -- the residual bound is one leaked
    thread per concurrently-running scan racing that wedge, bounded by the
    (small) number of concurrent scans. On top of that, the registry is
    bounded FIFO (see ``_MAX_REMEMBERED_WEDGED_OBJECTS``), so this is not
    an unconditional one-ever-across-the-process's-whole-lifetime
    guarantee either: eviction after more than
    ``_MAX_REMEMBERED_WEDGED_OBJECTS`` distinct wedging objects have been
    recorded can re-admit one further wedge for an object evicted earlier.
    An overall wall-clock budget (``_HANDLE_SCAN_BUDGET_SEC``) additionally
    bounds the whole function as defense in depth: once exceeded, the scan
    stops early and returns whatever it has found so far rather than
    continuing indefinitely.

    Never raises: any unexpected ctypes/structure failure (missing API,
    unexpected buffer layout, etc.) is expected to be caught by the caller
    and treated as "found nothing new via this pass" — see the try/except
    around the call site in ``_find_blocking_processes``.

    Parameters
    ----------
    budget_sec:
        Wall-clock seconds allotted to this scan's per-PID/per-handle loop
        (the one-shot ``NtQuerySystemInformation`` dump itself always runs
        regardless). Defaults to ``_HANDLE_SCAN_BUDGET_SEC`` for direct
        callers (e.g. tests calling this function standalone). Callers
        reached through ``_find_blocking_processes`` instead pass
        ``min(_HANDLE_SCAN_BUDGET_SEC, <time remaining from the caller's
        overall timeout>)``, so this scan can never independently spend up
        to its own ``_HANDLE_SCAN_BUDGET_SEC`` ceiling on top of a
        caller-supplied ``timeout`` (see ``_find_blocking_processes`` and
        ``_kill_blocking_processes``). A value ``<= 0`` still performs the
        handle-table dump but the per-handle loop below exits immediately
        without inspecting any individual handle.
    """
    import ctypes
    from ctypes import wintypes

    scan_deadline = time.monotonic() + max(0.0, budget_sec)
    # Ticket #90: total stage-2 grace this scan may spend, drawn down as
    # queries need it. Created here (not module scope) so it is scoped to
    # exactly one scan -- see _HANDLE_QUERY_GRACE_BUDGET_SEC's docstring.
    grace_budget = _GraceBudget(remaining=_HANDLE_QUERY_GRACE_BUDGET_SEC)

    normalized = os.path.normcase(os.path.normpath(path))

    # Ticket #148 (R3): acquire the process-wide scan lock BEFORE any real
    # work -- this fully serializes scans (only one _win_handle_holders call
    # may be mid-flight at a time) and is the sole guard for the
    # persistent-worker slot (_persistent_query_worker; see the module-level
    # comment above _handle_scan_lock). The wait is bounded by both
    # _HANDLE_SCAN_LOCK_WAIT_MAX_SEC and this scan's OWN remaining budget
    # (via scan_deadline) -- a caller with little time left never waits
    # longer than it has. budget_sec<=0 (already-expired deadline) collapses
    # the wait to 0.0, which is still a legal non-blocking
    # Lock.acquire(True, 0.0) that succeeds immediately on a free lock (so
    # test_zero_budget_skips_per_handle_loop_entirely's path/outcome is
    # unaffected by this gate). A genuinely contended lock reports
    # handle_scan:busy -- never dumps the handle table concurrently with
    # another in-flight scan.
    _lock_wait = max(
        0.0, min(_HANDLE_SCAN_LOCK_WAIT_MAX_SEC, scan_deadline - time.monotonic())
    )
    if not _handle_scan_lock.acquire(True, _lock_wait):
        _logger.warning(
            "_win_handle_holders: could not acquire the process-wide scan "
            "lock within %.2fs (another scan is already running) -- "
            "reporting handle_scan:busy without dumping the handle table",
            _lock_wait,
        )
        return _PartialList([], complete=False, skipped_passes=("handle_scan:busy",))

    try:
        # Ticket #148 (R2): scan-start capacity gate. Reuses the process-wide
        # persistent worker if one already exists (regardless of the
        # _MAX_WEDGED_HANDLE_WORKERS cap -- a *healthy* persistent worker is
        # always safe to reuse); creates a fresh one only when the cap has
        # room; refuses loudly, before ever touching ctypes.WinDLL or
        # dumping the handle table, when neither is true.
        #
        # Historical note (ticket #90, superseded by this gate): a
        # *silent*, forever-`[]` scan-start gate was tried and rejected --
        # once _MAX_WEDGED_HANDLE_WORKERS workers had ever accumulated
        # process-wide, every later call would have returned `[]` forever,
        # for the remaining life of the process, with no way for a caller
        # to tell "found nothing" apart from "refused to look". That
        # rejection still holds for a SILENT gate. This gate is different
        # in the one way that matters: it is loud. It reports
        # complete=False, tags the result "handle_scan:capped", logs a
        # warning naming the live count and the cap, and stop() surfaces a
        # dedicated STOP_REASON_HANDLE_SCAN_EXHAUSTED reason naming
        # host-restart remediation (see process_lifecycle.stop()) -- a
        # caller can always tell this condition apart from a genuine
        # "looked and found nothing", which is exactly what the old
        # rejected gate could not do.
        worker = _acquire_scan_worker()
        if worker is None:
            _logger.warning(
                "_win_handle_holders: process-wide wedged NtQueryObject "
                "worker cap (_MAX_WEDGED_HANDLE_WORKERS=%s, live=%s) is "
                "full and no persistent worker is available to reuse -- "
                "refusing this scan before dumping the handle table; this "
                "will not clear until the host process is restarted",
                _MAX_WEDGED_HANDLE_WORKERS, _wedged_worker_slots[0],
            )
            return _PartialList(
                [], complete=False, skipped_passes=("handle_scan:capped",)
            )

        ntdll = ctypes.WinDLL("ntdll")  # type: ignore[attr-defined]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

        # HANDLE-typed parameters MUST have explicit argtypes/restype: ctypes'
        # default guess for a bare Python int is 32-bit c_int, which silently
        # corrupts 64-bit HANDLE values on x64 (observed in practice: every
        # DuplicateHandle call failed with ERROR_INVALID_HANDLE (6) without
        # this). GetLogicalDrives/QueryDosDeviceW/NtQuerySystemInformation are
        # left with ctypes' default guessing (as elsewhere in this module) since
        # none of their parameters are 64-bit HANDLE values. NtQueryObject IS
        # called with a HANDLE parameter (``dup_handle``, its first argument at
        # every call site below) and is also left with ctypes' default guessing
        # -- but that is safe today only because every call site passes an
        # already-typed ``wintypes.HANDLE`` instance (produced by
        # ``DuplicateHandle``'s out-parameter), never a bare Python int. Do not
        # start passing a raw int handle value to NtQueryObject without adding
        # explicit argtypes first, or this reintroduces the exact x64
        # handle-truncation bug class this comment exists to warn about.
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.DuplicateHandle.restype = wintypes.BOOL
        kernel32.DuplicateHandle.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        OBJECT_NAME_INFORMATION = 1
        OBJECT_TYPE_INFORMATION = 2
        PROCESS_DUP_HANDLE = 0x0040
        DUPLICATE_SAME_ACCESS = 0x00000002

        # --- Step 1: dump the system-wide handle table, grouped by owning PID.
        # Hoisted to module scope as _enumerate_handle_table (ticket #121) so it
        # is directly mockable in tests; see that function's docstring for the
        # (handle_value, type_index, object_ptr, granted_access) tuple shape
        # and the None-means-"dump itself failed" contract.
        by_pid = _enumerate_handle_table(ntdll, excluded_pids)
        if by_pid is None:
            # The one-shot dump itself failed -- this scan never looked at any
            # handle at all, not merely "found nothing".
            return _PartialList([], complete=False)

        # --- Step 2: build the NT-device -> drive-letter map used to translate
        # resolved object names (e.g. "\Device\HarddiskVolume3\...") into
        # ordinary drive-letter paths comparable against *path*.
        device_map: Dict[str, str] = {}
        bitmask = kernel32.GetLogicalDrives()
        for i in range(26):
            if not (bitmask & (1 << i)):
                continue
            drive = f"{chr(65 + i)}:"
            dev_buf = ctypes.create_unicode_buffer(260)
            if kernel32.QueryDosDeviceW(drive, dev_buf, 260):
                device_map[dev_buf.value] = drive

        def _nt_path_to_dos(nt_path: str) -> Optional[str]:
            for device, drive in device_map.items():
                if nt_path.startswith(device):
                    return drive + nt_path[len(device):]
            return None

        # Ticket #148: the worker used by this scan is the process-wide
        # persistent worker acquired by the scan-start gate above (or a
        # replacement created mid-scan by _bounded_query, see below) -- NOT
        # a fresh worker created unconditionally per scan, which was the
        # unbounded per-scan daemon-thread leak this ticket closes (a
        # wedged initial worker's thread was never gated by
        # _MAX_WEDGED_HANDLE_WORKERS, which only ever bounded *replacement*
        # workers, and was never joined). This worker is deliberately never
        # closed here: a healthy persistent worker parks on queue.get()
        # between scans and is reused by every later scan in this process
        # -- the "+1" in _HANDLE_SCAN_MAX_LIVE_WORKERS. Only a worker that
        # this scan itself retires (via _bounded_query's ABANDONED/CAPPED
        # handling) stops being reused -- and even then its thread is not
        # joined (a wedged thread cannot be joined); it simply self-
        # terminates, on its own, whenever its wedged call eventually
        # returns.
        #
        # Ticket #121's wedged-handle registry (_wedged_object_keys, see
        # _wedging_handle_key/_remember_wedging_handle/_is_known_wedging_handle
        # above) remains in force unchanged: _process_handle below still
        # checks it before ever duplicating/querying a handle, and
        # _bounded_query still records a handle's key the moment its query
        # fails to resolve, so a given pathological kernel object still
        # only ever wedges once (per the same bounded-overshoot/FIFO-
        # eviction caveats documented above _MAX_REMEMBERED_WEDGED_OBJECTS)
        # rather than once per call -- this ticket's persistent-worker fix
        # is what now also bounds the *thread* cost of that first wedge to
        # _HANDLE_SCAN_MAX_LIVE_WORKERS process-wide, not just the query
        # cost.
        stop_scan = False
        cap_already_logged = False
        capped_mid_scan = False

        _RESOLVED, _MATCHED, _CONTINUE, _STOP = (
            "resolved", "matched", "continue", "stop",
        )

        def _make_handle_closer(dup_handle):
            # Bound to *this* dup_handle via the default-argument trick isn't
            # needed here since each call site builds a fresh closure per
            # handle -- captured by simple closure instead.
            def _closer(_value: Optional[str]) -> None:
                kernel32.CloseHandle(dup_handle)
            return _closer

        def _bounded_query(dup_handle, info_class: int, key: tuple):
            """Run one NtQueryObject call through the scan's current worker.

            Returns ``(_RESOLVED, value)`` on success, or ``(_STOP, None)``
            once this scan has no usable worker left (process-wide wedged-
            worker cap hit, with or without this specific query being the
            one that hit it) -- callers must stop issuing further queries
            in that case. A wedged query's handle ownership is transferred
            to the retired worker via ``on_abandoned_done`` in every
            non-resolved case; the caller must NOT close *dup_handle*
            itself when this does not return ``_RESOLVED`` (ticket #90's
            handle-reuse / false-match fix).

            Ticket #121: *key* (see ``_wedging_handle_key``) identifies the
            kernel object this query targets. On every non-``RESOLVED``
            outcome -- ``ABANDONED`` and ``CAPPED`` alike, mirroring how
            ``submit()`` already counts both against ``_wedged_worker_count``,
            since both leave a thread genuinely blocked -- *key* is recorded
            via ``_remember_wedging_handle`` so no *later* scan (in this
            process's lifetime) ever queries this same object again.

            Ticket #148: on an ``ABANDONED`` outcome, the process-wide
            persistent-worker slot is handed to ``_replace_scan_worker`` --
            which clears the (now permanently retired) worker from the slot
            and creates a replacement only when capacity allows -- exactly
            mirroring the pre-#148 separate ``_wedged_slot_available()``
            check + bare ``_BoundedQueryWorker()`` construction it replaces.
            A genuine ``CAPPED`` outcome never attempts a replacement at
            all -- ``submit()`` itself already decided the process-wide cap
            left no room -- so the slot is instead cleared directly via
            ``_clear_scan_worker_slot``, without ever re-consulting
            ``_wedged_slot_available()`` a second time (which could only
            ever apply once ``submit()`` has already made its own
            ABANDONED/CAPPED decision). Either way, the module slot
            always ends up holding the truth on scan exit, and the retiring
            worker is explicitly ``close()``d: a fast, safe no-op for a
            genuinely wedged real worker (submit() already sent it its own
            shutdown sentinel), but the only thing that promptly shuts down
            a healthy worker's thread when ``submit()`` itself has been
            replaced by a test double that never actually dispatched
            anything to it.
            """
            nonlocal worker, cap_already_logged, capped_mid_scan
            outcome = worker.submit(
                lambda: _query_object_raw(ntdll, dup_handle, info_class),
                grace=grace_budget,
                scan_deadline=scan_deadline,
                on_abandoned_done=_make_handle_closer(dup_handle),
            )
            if outcome.status == _QueryStatus.RESOLVED:
                return _RESOLVED, outcome.value
            _remember_wedging_handle(key)
            retiring_worker = worker
            if outcome.status == _QueryStatus.ABANDONED:
                replacement = _replace_scan_worker()
                if replacement is not None:
                    # This worker is now permanently tied up in the
                    # still-running wedged call (and has already been sent
                    # its own shutdown sentinel by submit()) -- the
                    # replacement lets subsequent queries in this scan
                    # proceed.
                    worker = replacement
                    retiring_worker.close()
                    return _CONTINUE, None
                # ABANDONED but no replacement capacity left --
                # _replace_scan_worker already cleared the slot.
            else:
                # Genuinely CAPPED: submit() itself already decided no
                # replacement may be attempted at all.
                _clear_scan_worker_slot()
            # Either genuinely CAPPED, or ABANDONED with no capacity left
            # for a replacement worker -- either way this scan cannot make
            # further progress on new queries. Log once per scan, at
            # debug, so an operator can see from the logs alone that a
            # scan degraded early.
            retiring_worker.close()
            capped_mid_scan = True
            if not cap_already_logged:
                _logger.debug(
                    "_win_handle_holders: hit the process-wide wedged NtQueryObject "
                    "worker cap (_MAX_WEDGED_HANDLE_WORKERS=%s, live=%s) -- "
                    "stopping this scan early and returning partial results",
                    _MAX_WEDGED_HANDLE_WORKERS, _wedged_worker_slots[0],
                )
                cap_already_logged = True
            return _STOP, None

        found: List[Tuple[int, str]] = []
        current_process = kernel32.GetCurrentProcess()
        # Per-call cache: object-type-index -> resolved type name, or None
        # if a probe against a handle of that type failed to resolve
        # (ticket #154, the C3 dedup fix: this is deliberately WRITTEN, not
        # left _UNSET, on a non-resolved outcome too -- see below). Every
        # later handle sharing that type index is accepted/skipped from
        # this cache alone, with no further DuplicateHandle/NtQueryObject
        # call -- including a sibling of a type whose probe wedged.
        _UNSET = object()
        type_name_cache: Dict[int, Optional[str]] = {}
        type_unresolved = False

        def _process_handle(
            proc_handle, pid: int, handle_value: int, type_index: int,
            object_ptr: int, granted_access: int,
        ) -> str:
            """Resolve one (handle_value, type_index, object_ptr) for *pid*.

            Returns ``_MATCHED`` (pid appended to *found*), ``_CONTINUE``
            (nothing to report, keep scanning this pid's other handles), or
            ``_STOP`` (this scan has no usable worker left -- callers must stop
            issuing further queries entirely, not just for this pid).

            Ticket #121: before touching this handle at all -- no
            ``DuplicateHandle``, no worker traffic -- checks whether it is
            already known to belong to a kernel object that wedged a previous
            scan's query (``_is_known_wedging_handle``); if so, skips it
            outright via ``_CONTINUE``.

            *granted_access* is accepted for tuple-shape compatibility with
            ``_enumerate_handle_table``'s 4-tuple entries but is otherwise
            unused -- ticket #154 deletes the former GrantedAccess-mask
            deferral special case (a hardcoded workaround for one observed
            hang-prone mask) in favour of the general per-type-index
            suppression fix below, which covers a probe that fails to
            resolve for *any* reason, not just that one mask.
            """
            nonlocal type_unresolved
            key = _wedging_handle_key(pid, handle_value, object_ptr, type_index)
            if _is_known_wedging_handle(key):
                return _CONTINUE  # known-wedging object -- skip without duplicating

            cached_type = type_name_cache.get(type_index, _UNSET)
            if cached_type is not _UNSET and cached_type != "File":
                return _CONTINUE  # known non-file type (or unresolved) -- skip

            dup_handle = wintypes.HANDLE()
            ok = kernel32.DuplicateHandle(
                proc_handle,
                wintypes.HANDLE(handle_value),
                current_process,
                ctypes.byref(dup_handle),
                0,
                False,
                DUPLICATE_SAME_ACCESS,
            )
            if not ok:
                return _CONTINUE

            owns_handle = True
            nt_name: Optional[str] = None
            try:
                if cached_type is _UNSET:
                    status, type_name = _bounded_query(dup_handle, OBJECT_TYPE_INFORMATION, key)
                    if status != _RESOLVED:
                        owns_handle = False
                        # Ticket #154 (C3 fix): write the cache even on a
                        # non-resolved probe -- as `None`, the value this
                        # cache is already typed and documented for -- so
                        # no later same-type-index sibling this scan
                        # re-probes (and re-wedges) the same pathological
                        # type. Previously this branch returned without
                        # writing the cache at all, which was the bug: a
                        # sibling handle of the same type re-triggered a
                        # fresh DuplicateHandle/NtQueryObject cycle here,
                        # up to _MAX_WEDGED_HANDLE_WORKERS times per scan.
                        type_name_cache[type_index] = None
                        type_unresolved = True
                        return _STOP if status == _STOP else _CONTINUE
                    type_name_cache[type_index] = type_name
                    if type_name != "File":
                        return _CONTINUE

                status, nt_name = _bounded_query(dup_handle, OBJECT_NAME_INFORMATION, key)
                if status != _RESOLVED:
                    owns_handle = False
                    return _STOP if status == _STOP else _CONTINUE
            finally:
                if owns_handle:
                    kernel32.CloseHandle(dup_handle)

            if not nt_name:
                return _CONTINUE
            dos_path = _nt_path_to_dos(nt_name)
            if not dos_path:
                return _CONTINUE
            norm = os.path.normcase(os.path.normpath(dos_path))
            if norm == normalized or norm.startswith(normalized + os.sep):
                found.append((pid, ""))
                return _MATCHED
            return _CONTINUE

        # Ticket #95, finding 3 (D5 vs N2/N3, REVERSED by ticket #148):
        # distinguish this scan's OWN budget_sec expiring mid-enumeration
        # (deadline_truncated -- genuine incompleteness, "ran out of time
        # before it could look") from the process-wide wedged-worker cap
        # being hit (stop_scan via a CAPPED/ABANDONED-no-capacity verdict).
        # Both cause the same early `break`, but only the deadline case sets
        # deadline_truncated -- checked as a distinct condition, and only
        # when stop_scan is not ALSO already true (i.e. the deadline is the
        # actual reason this iteration stopped, not a cap hit on a prior
        # handle that would have broken the loop already). Ticket #148
        # reverses the OLD N2/N3 rule that a cap hit left complete=True (an
        # "internal degradation" the table was still enumerated up to) --
        # see capped_mid_scan below and the tag-driven return at the bottom
        # of this function: a cap hit now honestly reports complete=False,
        # tagged "handle_scan:capped", distinct from "handle_scan:truncated".
        #
        # Ticket #106: a *small* budget_sec cannot be used to reliably exercise
        # the CAPPED/ABANDONED stop_scan path in a test. scan_deadline is armed
        # at the top of this function, before the handle-table dump and the
        # pure-Python parse loop above even run -- so on a loaded machine a
        # tight budget is spent (or overspent) by the dump/parse alone, before
        # the per-handle loop below starts, and the scan ends up here via
        # deadline_truncated instead. A test that wants the CAPPED/ABANDONED
        # branch must force the query *outcome* (patch
        # ``_BoundedQueryWorker.submit``), not starve the *budget* -- see
        # ``TestDiscoveryCompleteness`` N2/N3 in test_process_lifecycle.py.
        deadline_truncated = False
        for pid, handle_entries in by_pid.items():
            if stop_scan:
                break
            if time.monotonic() > scan_deadline:
                deadline_truncated = True
                break
            proc_handle = kernel32.OpenProcess(PROCESS_DUP_HANDLE, False, pid)
            if not proc_handle:
                # No PROCESS_DUP_HANDLE access -- elevated/other-user process
                # we cannot cross without running elevated ourselves.
                continue
            try:
                for entry in handle_entries:
                    if stop_scan:
                        break
                    if time.monotonic() > scan_deadline:
                        deadline_truncated = True
                        break
                    # Ticket #148: tolerate both the current 4-tuple shape
                    # ((handle_value, type_index, object_ptr, granted_access))
                    # and the pre-#148 3-tuple shape (no granted_access) --
                    # a missing granted_access means "no access info" and
                    # therefore "pre-filter disabled", i.e. pre-#148
                    # behaviour, which keeps every hand-built 3-tuple
                    # fake_table fixture elsewhere in the test suite working
                    # unchanged.
                    handle_value, type_index, object_ptr = entry[0], entry[1], entry[2]
                    granted_access = entry[3] if len(entry) > 3 else 0
                    verdict = _process_handle(
                        proc_handle, pid, handle_value, type_index, object_ptr,
                        granted_access,
                    )
                    if verdict == _STOP:
                        stop_scan = True
                        break
                    if verdict == _MATCHED:
                        break  # one matching handle is enough to flag this pid
            finally:
                kernel32.CloseHandle(proc_handle)

        # Ticket #148 (R2/R4), revised #154 (R16b): tag-driven return -- each
        # distinct degradation this scan may have hit contributes its own
        # tag, and `complete` is honest: True only when none did.
        # `handle_scan:type_unresolved` (new, #154) replaces
        # `handle_scan:masked_deferred_capped`: per-object suppression skips
        # one kernel object and leaves `complete` alone, but per-*type*
        # suppression (the C3 fix above) can skip an entire type -- possibly
        # "File" itself -- so a scan that suppressed a type index must
        # honestly report `complete=False` rather than silently under-
        # counting; see #148's own precedent for "a caller can always tell
        # 'refused to look' from 'looked and found nothing'".
        tags: List[str] = []
        if deadline_truncated:
            tags.append("handle_scan:truncated")
        if capped_mid_scan:
            tags.append("handle_scan:capped")
        if type_unresolved:
            tags.append("handle_scan:type_unresolved")
        return _PartialList(found, complete=not tags, skipped_passes=tuple(tags))
    finally:
        _handle_scan_lock.release()


def _bounded_call(fn, *, deadline: Optional[float] = None) -> "Tuple[bool, Any]":
    """Thin, locked wrapper (ticket #154, item 11) for a single blocking
    psutil read (``proc.cwd()``, ``cmdline()``, ``name()``, the ancestor
    walk's ``parents()``, tier-1's ``psutil.pid_exists``, ...), dispatched
    through a short-lived :class:`_BoundedQueryWorker` at
    ``_BLOCKING_CALL_TIMEOUT_SEC`` -- never ``_HANDLE_QUERY_TIMEOUT_SEC``,
    which is tuned for a single already-duplicated handle, not a psutil
    read.

    Returns ``(completed, value)``: ``(True, <fn's return value>)`` if *fn*
    resolved in time, ``(False, None)`` if it did not (or if a worker
    could not even be started). Never raises.

    Deliberately never passed a *grace* budget: grace exists for Pass 1c's
    ~10^5-per-scan ``NtQueryObject`` multiplication problem (see
    ``_HANDLE_QUERY_GRACE_SEC``'s docstring), not for the ~10^2-10^3-
    per-call psutil population, where a flat, generous per-call timeout is
    affordable on its own.

    A non-resolved outcome's thread joins the SAME shared
    ``_wedged_worker_slots`` cell as a wedged handle-scan worker -- one
    accounting for every bounded OS call in this module, not two pools
    (item 14). The worker is intentionally NOT the persistent
    ``_win_handle_holders`` worker (a separate, freshly-created one per
    call): sharing that one would serialize ordinary psutil reads behind
    ``_handle_scan_lock``, which guards a genuinely different resource.
    """
    try:
        worker = _BoundedQueryWorker()
    except Exception:  # noqa: BLE001 -- never let bounding itself fail
        return False, None
    outcome = worker.submit(fn, scan_deadline=deadline, timeout=_BLOCKING_CALL_TIMEOUT_SEC)
    if outcome.status == _QueryStatus.RESOLVED:
        worker.close()
        return True, outcome.value
    # Not resolved -- submit() already retired this worker's thread and
    # counted it against the shared _wedged_worker_slots cell; do not
    # close() it (it may still be genuinely blocked in fn()).
    return False, None


def _find_blocking_processes(
    path: str,
    host_pid: int,
    *,
    deadline: Optional[float] = None,
) -> "_PartialList":
    """Return processes whose cwd or open file handles are under *path*.

    Detection passes:
    1. CWD match — processes whose working directory is at or under *path*.
    1b. (Windows only, ticket #57) cmdline-token scan — a fallback for when
        ``proc.cwd()`` raises ``AccessDenied`` (the common case for foreign
        processes on Windows): if any cmdline token resolves to a path under
        *path*, the process is treated as blocking.
    1c. (Windows only, ticket #71) OS-level handle-table scan via
        :func:`_win_handle_holders` — catches a process with a real OS-level
        cwd inside *path* (e.g. started via
        ``Start-Process -WorkingDirectory <path>``) that holds an open file
        handle inside *path* but evades both Pass 1 (``cwd()``/
        ``open_files()`` denied) and Pass 1b (the worktree path is not a
        cmdline token). This pass is best-effort and wrapped so it can never
        raise out of this function or make the result worse than the other
        passes alone; elevated/other-user processes remain undetectable by
        it (a hard OS permission boundary — see ``_win_handle_holders``). Its
        scan budget is governed by *deadline* (see below): capped at
        ``_HANDLE_SCAN_BUDGET_SEC`` and skipped entirely once no budget
        remains.
    2. Open-file match — processes holding an open file handle inside *path*
       (via ``psutil.open_files()``). On Windows, a single call can raise a
       bare ``RuntimeError`` when the OS-wide handle table is too large for
       psutil's query to succeed (D9) -- this is process-independent, not a
       per-PID failure; see the Returns section below.

    All passes exclude the host process and all its OS-level ancestors.
    Results are de-duplicated by PID.

    Parameters
    ----------
    path:
        The worktree directory path.
    host_pid:
        The PID of the current (MCP host) process; it and all its OS-level
        ancestors are always excluded from the returned list.
    deadline:
        Optional ``time.monotonic()``-based absolute deadline for this call.
        Supplied by :func:`_kill_blocking_processes` so that Pass 1c's
        Windows handle-table scan — the only detection pass with a
        meaningful per-call cost — respects whatever time remains from the
        *caller's* overall timeout (see ``_kill_blocking_processes`` and
        ``stop()``) instead of always independently spending up to its own
        fixed ``_HANDLE_SCAN_BUDGET_SEC`` ceiling on top of everything else.
        ``None`` (the default, used by direct/test callers) means "use the
        full ``_HANDLE_SCAN_BUDGET_SEC`` ceiling, no external deadline".
        Passes 1, 1b, and 2 are now ALSO bounded by *deadline* (ticket #87):
        before this fix they were unbounded linear scans, and against a
        large/slow ambient process list that measured ~75s of CPU for a
        single call that found nothing. See ``_DISCOVERY_MAX_SEC`` /
        ``_DISCOVERY_RESERVE_SEC`` below.

    Returns
    -------
    A :class:`_PartialList` (ticket #95, finding 3). ``.complete`` is
    ``False`` when any pass's entry guard was false (never ran at all -- e.g.
    ``"cwd:skipped"``) or its inner per-process loop broke on ``scan_stop``
    (ran, but did not finish -- e.g. ``"cwd:truncated"``), and likewise for
    Pass 1b (``"cmdline:..."``), Pass 1c (``"handle_scan:..."``, including
    ``"handle_scan:failed"`` when :func:`_win_handle_holders` raised and was
    swallowed -- zero coverage from that pass, not a mere degradation), and
    Pass 2 (``"open_files:..."``, including ``"open_files:degraded"`` (D9)
    when ``proc.open_files()`` raised a bare ``RuntimeError`` -- see below).
    A pass simply not applicable on this OS (Pass 1b/1c off Windows) never
    contributes a tag. An individual PID raising
    ``AccessDenied``/``NoSuchProcess`` within a pass is caught and skipped
    (``continue``) without affecting that pass's completeness -- only a
    whole pass being skipped/truncated/failed/degraded does. A
    ``RuntimeError`` from ``proc.open_files()`` on Windows (D9) is different
    from those: it is *not* per-PID -- it reflects an OS-wide condition (the
    handle table is too large for a single query) that will recur
    identically for every remaining process, so Pass 2 stops early
    (``break``, not ``continue``) and reports ``"open_files:degraded"``
    rather than either silently continuing to burn the scan budget on
    guaranteed failures or letting the exception escape and crash the
    caller (ticket #107). This is distinct from ``"open_files:truncated"``
    (D7): truncated means the loop ran out of clock (``scan_stop`` was
    reached); degraded means the OS refused the query outright, independent
    of remaining budget.
    """
    import psutil

    normalized = os.path.normcase(os.path.normpath(path))
    skipped_passes: List[str] = []

    # Ancestor-exclusion walk (ticket #154, item 11): a SAFETY precondition,
    # not an enrichment -- without it, an ancestor of the calling process
    # whose cwd lies inside the worktree could be reported as a blocker and
    # then signalled. Bounded via _bounded_call; if it does not complete,
    # discovery aborts BEFORE Pass 1 even starts -- an unknown exclusion
    # set must never become an empty one. `psutil.AccessDenied`/
    # `NoSuchProcess` are definitive OS answers ("the walk finished as far
    # as permitted"), not "we never got an answer" -- kept unchanged,
    # deliberately not routed through the abort rule below.
    excluded_pids: set[int] = {host_pid}
    ancestor_deadline = deadline if deadline is not None else time.monotonic() + _HANDLE_SCAN_BUDGET_SEC
    completed, ancestors = _bounded_call(
        lambda: list(psutil.Process(host_pid).parents()), deadline=ancestor_deadline
    )
    if not completed:
        return _PartialList([], complete=False, skipped_passes=("discovery:unresponsive",))
    if ancestors is not None:
        for ancestor in ancestors:
            excluded_pids.add(ancestor.pid)

    # Ticket #154 (item 10): `_DISCOVERY_MAX_SEC`/`_DISCOVERY_RESERVE_SEC`
    # are deleted -- the caller-derived *deadline* is the only budget story
    # now (every real call site supplies one). `scan_stop` reserves
    # `0.2 * remaining` unconditionally (no min-of-two against a deleted
    # absolute constant) so that discovery never eats into the time the
    # caller needs afterward for the actual signal/kill step. Checked at
    # the top of every pass and inside every per-process loop below so a
    # slow ambient process list degrades gracefully to "whatever was found
    # so far" instead of blowing through the caller's budget. `deadline is
    # None` (direct/test callers with no caller-supplied timeout) falls
    # back to `_HANDLE_SCAN_BUDGET_SEC` -- the one ceiling this module
    # keeps -- as the overall discovery ceiling too.
    entry_ts = time.monotonic()
    if deadline is None:
        scan_stop = entry_ts + _HANDLE_SCAN_BUDGET_SEC
    else:
        # Clamp to 0 (ticket #87 follow-up, finding B4): when *deadline* has
        # already elapsed by entry (remaining < 0), the un-clamped algebra
        # below still resolves to a scan_stop at or before entry_ts -- so
        # every pass's own `time.monotonic() <= scan_stop` guard already
        # skips it, and this is not a live bug -- but leaving `remaining`
        # negative makes that non-obvious from the arithmetic alone. Clamping
        # here makes the "no discovery once the deadline is already gone"
        # intent unambiguous without changing behaviour.
        remaining = max(0.0, deadline - entry_ts)
        scan_stop = deadline - 0.2 * remaining

    seen_pids: set[int] = set()
    result: List[KilledProcessInfo] = []
    # Ticket #95, finding 3: names of passes that were skipped entirely
    # (entry guard false), truncated (inner loop broke on scan_stop), failed
    # (raised and was swallowed), or degraded (OS-wide condition made
    # continuing pointless -- Pass 2 only, ticket #107), in the order
    # encountered. See the D1-D9 / N1-N8 rules documented above.
    # (skipped_passes was already initialised above the ancestor walk.)

    # Ticket #154 (item 11): a pid whose bounded call did not complete this
    # call is added here, and later passes skip it -- without this, one
    # genuinely hung process burns a worker slot in Pass 1, again in Pass
    # 1b, and again in Pass 1c's enrichment. Local, not a registry: no
    # lock, no cap, no eviction, dies with this call.
    unresponsive_pids: set[int] = set()

    # Pass 1: CWD match. proc.cwd() is dispatched through _bounded_call
    # (item 11) -- a single hung cwd() must not hang the whole scan.
    pass_unresponsive = False
    if time.monotonic() <= scan_stop:
        pass_truncated = False
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if time.monotonic() > scan_stop:
                pass_truncated = True
                break
            try:
                pid = proc.info["pid"]
                if pid in excluded_pids:
                    continue
                completed, cwd = _bounded_call(proc.cwd, deadline=scan_stop)
                if not completed:
                    unresponsive_pids.add(pid)
                    pass_unresponsive = True
                    continue
                if cwd is None:
                    continue
                norm_cwd = os.path.normcase(os.path.normpath(cwd))
                # Match if the process cwd equals the target path or is under it.
                if norm_cwd == normalized or norm_cwd.startswith(normalized + os.sep):
                    seen_pids.add(pid)
                    result.append(
                        KilledProcessInfo(
                            pid=pid,
                            name=proc.info["name"] or "",
                            cmdline=proc.info["cmdline"] or [],
                            source="orphan_scan",
                            match_pass="cwd",
                        )
                    )
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        if pass_truncated:  # D2
            skipped_passes.append("cwd:truncated")
    else:
        skipped_passes.append("cwd:skipped")  # D1
    if pass_unresponsive:
        skipped_passes.append("discovery:unresponsive")

    # Pass 1b (Windows only): cmdline token scan.
    # proc.cwd() raises AccessDenied for almost all foreign processes on Windows.
    # Scan cmdline tokens as a fallback: if any token resolves to a path under
    # the worktree directory, treat the process as blocking.
    if sys.platform == "win32":
        if time.monotonic() <= scan_stop:
            pass_truncated = False
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                if time.monotonic() > scan_stop:
                    pass_truncated = True
                    break
                try:
                    pid = proc.info["pid"]
                    if pid in excluded_pids or pid in seen_pids or pid in unresponsive_pids:
                        continue
                    cmdline = proc.info["cmdline"] or []
                    for token in cmdline:
                        try:
                            norm_token = os.path.normcase(os.path.normpath(token))
                        except (ValueError, TypeError):
                            continue
                        if norm_token == normalized or norm_token.startswith(normalized + os.sep):
                            seen_pids.add(pid)
                            result.append(KilledProcessInfo(
                                pid=pid,
                                name=proc.info["name"] or "",
                                cmdline=cmdline,
                                source="orphan_scan",
                                match_pass="cmdline",
                            ))
                            break
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
            if pass_truncated:  # D3
                skipped_passes.append("cmdline:truncated")
        else:
            skipped_passes.append("cmdline:skipped")  # D3

    # Pass 1c (Windows only): OS-level handle-table scan (ticket #71). See
    # _win_handle_holders' docstring for the full rationale/mechanism. This
    # call is never allowed to raise out of _find_blocking_processes -- any
    # ctypes/structure failure degrades gracefully to whatever Pass 1/1b
    # already found.
    #
    # Budget (ticket #154, item 10): capped at _HANDLE_SCAN_BUDGET_SEC, but
    # shrunk further to whatever remains of `scan_stop` -- NOT `deadline`
    # directly -- so this scan can never independently add up to
    # _HANDLE_SCAN_BUDGET_SEC on top of the caller's own budget, and never
    # eats into the reserve `scan_stop` already set aside for the caller's
    # own kill step afterward (the real inconsistency this fixes: computing
    # the budget from `deadline` let Pass 1c legally spend time out of that
    # reserve). When no time remains at all, the scan is skipped outright
    # rather than spending any of the remaining time on a scan that has no
    # chance to report back before that time is needed elsewhere.
    if sys.platform == "win32":
        if time.monotonic() <= scan_stop:
            handle_scan_budget = min(
                _HANDLE_SCAN_BUDGET_SEC, max(0.0, scan_stop - time.monotonic())
            )
            if handle_scan_budget <= 0:
                handle_holders: "_PartialList" = _PartialList([], complete=True)
                skipped_passes.append("handle_scan:skipped")  # D4
            else:
                try:
                    handle_holders = _win_handle_holders(
                        path, excluded_pids, budget_sec=handle_scan_budget
                    )
                except Exception:  # noqa: BLE001 -- best-effort, never propagate
                    handle_holders = _PartialList([], complete=True)
                    skipped_passes.append("handle_scan:failed")  # D6
                else:
                    if not getattr(handle_holders, "complete", True):
                        # Ticket #148: propagate _win_handle_holders' OWN
                        # skipped_passes tags (e.g. "handle_scan:capped",
                        # "handle_scan:busy", "handle_scan:masked_deferred_capped")
                        # instead of collapsing every incomplete inner result
                        # to the single hardcoded "handle_scan:truncated"
                        # string -- callers (stop(), teardown's Gate A) need
                        # to distinguish these conditions. Falls back to the
                        # legacy "handle_scan:truncated" tag only when the
                        # inner result carries no tags of its own (a bare/
                        # legacy incomplete list, e.g. from a test mock or
                        # the _enumerate_handle_table-is-None path above,
                        # which itself returns a tagless incomplete
                        # _PartialList).
                        inner_tags = getattr(handle_holders, "skipped_passes", ())
                        if inner_tags:
                            skipped_passes.extend(inner_tags)  # D5
                        else:
                            skipped_passes.append("handle_scan:truncated")  # D5
            for pid, name in handle_holders:
                if pid in excluded_pids or pid in seen_pids:
                    continue
                cmdline: List[str] = []
                proc_name = name
                try:
                    p = psutil.Process(pid)
                    cmdline = p.cmdline()
                    if not proc_name:
                        proc_name = p.name()
                except Exception:  # noqa: BLE001
                    # Best-effort info gathering only -- a psutil failure here
                    # must not prevent the PID from being reported as a blocker.
                    pass
                seen_pids.add(pid)
                result.append(
                    KilledProcessInfo(
                        pid=pid, name=proc_name or "", cmdline=cmdline,
                        source="orphan_scan",
                        match_pass="handle_scan",
                    )
                )
        else:
            skipped_passes.append("handle_scan:skipped")  # D4

    # Ticket #154 (item 9): Pass 2 (psutil.open_files()) is deleted outright,
    # not bounded -- it was the actual hang site (the ticket's own repro
    # stack-dumped inside psutil's isfile_strict() under open_files()).
    # Named consequence: on POSIX, discovery now degrades to cwd-only, so
    # stop(kill_orphans=True) no longer finds a daemon that chdir'd away but
    # still holds an open file inside the worktree.

    return _PartialList(
        result, complete=not skipped_passes, skipped_passes=tuple(skipped_passes)
    )


def _kill_blocking_processes(
    path: str,
    *,
    timeout: float = 5.0,
) -> "_PartialList":
    """Kill all processes whose cwd is under *path* and return their info.

    Sends the graceful signal first, waits, then force-kills any survivors.
    The MCP host process and its ancestors are never killed.  Returns an empty
    list when no blocking processes are found.

    The total runtime of this function — including the ``_find_blocking_processes``
    discovery scan, not just the subsequent signal/wait step — is bounded by
    *timeout* seconds. The deadline is computed once, before discovery even
    starts, and threaded into ``_find_blocking_processes`` as *deadline* so
    that Windows Pass 1c's handle-table scan (the only discovery pass with a
    meaningful per-call cost) never independently spends up to its own fixed
    ``_HANDLE_SCAN_BUDGET_SEC`` ceiling on top of this function's budget.
    Whatever time remains after discovery completes is then distributed
    evenly across the found orphans for the signal/wait step; once the
    deadline has passed any remaining orphans receive only the graceful
    signal (no wait).

    Parameters
    ----------
    path:
        The worktree directory path.
    timeout:
        Maximum seconds to spend on discovery (including Pass 1c) plus
        waiting across *all* found orphans combined. Defaults to 5.0.  Pass
        ``0.0`` to send graceful signals without waiting — discovery itself
        may still spend a small, unavoidable amount of time (e.g. Pass 1/1b/2
        scans), but Pass 1c is skipped outright since no budget remains for
        it.

    Lineage expansion (ticket #87)
    -------------------------------
    Each blocker found by :func:`_find_blocking_processes` (a path-heuristic
    match) has its own descendant tree collected via :func:`_process_tree`
    and folded into the result (de-duplicated by PID) before any signalling
    happens. This catches a blocker's own children even when they evade the
    path heuristics themselves (e.g. a grandchild that changed its cwd away
    from the worktree and holds no matching open file handle).

    This loop is itself bounded by the same *deadline* as everything else in
    this function (ticket #87 follow-up, finding F1): before that fix it ran
    AFTER discovery with no budget check of its own, so a host with many
    path-heuristic blockers could make this loop alone blow through the
    caller's *timeout* -- contradicting the "total runtime ... is bounded by
    *timeout* seconds" guarantee above. Once the deadline passes, expansion
    stops issuing further :func:`_process_tree` calls and degrades
    gracefully to whatever lineage was already expanded so far -- the same
    pattern :func:`_find_blocking_processes` already uses for its own Pass
    1/1b/1c/2 discovery loops. Ticket #95, finding 3 (D8): when this happens
    it is folded into the returned :class:`_PartialList` as
    ``"lineage:truncated"`` and ``complete=False`` -- carrying forward
    whatever incompleteness :func:`_find_blocking_processes` itself already
    reported, if any.
    """
    deadline = time.monotonic() + timeout
    found = _find_blocking_processes(path, os.getpid(), deadline=deadline)
    if not found:
        return found

    discovery_complete = getattr(found, "complete", True)
    discovery_skipped_passes = tuple(getattr(found, "skipped_passes", ()))

    seen_pids = {info.pid for info in found}
    expanded: List[KilledProcessInfo] = list(found)
    lineage_truncated = False
    for info in found:
        if time.monotonic() >= deadline:
            # Budget exhausted -- stop issuing further _process_tree() calls;
            # whatever lineage was expanded so far is kept, not discarded.
            lineage_truncated = True
            break
        for descendant in _process_tree(info.pid):
            if descendant.pid in seen_pids:
                continue
            seen_pids.add(descendant.pid)
            expanded.append(descendant)
    found_list = expanded

    n = len(found_list)
    for info in found_list:
        # Always send the graceful signal, even if the budget is exhausted.
        # This ensures every orphan is notified regardless of how much time
        # is left.  Only the _wait_or_kill call is gated on remaining budget.
        delivered = _send_graceful_signal(info.pid, group_leader=False)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Budget exhausted. If the graceful signal was actually
            # delivered, leave this orphan (and all subsequent ones) merely
            # signalled, as before. If it was refused/not delivered (e.g. a
            # withheld win32 CTRL_BREAK_EVENT for an unconfirmed orphan),
            # there is no wait budget left to fall back on -- force-kill
            # immediately instead of leaving the orphan running unmanaged.
            if not delivered:
                _wait_or_kill(info.pid, timeout=0.0)
            continue
        per_pid_budget = min(remaining, timeout / n)
        _wait_or_kill(info.pid, timeout=per_pid_budget)

    skipped_passes = discovery_skipped_passes
    if lineage_truncated:  # D8
        skipped_passes = skipped_passes + ("lineage:truncated",)
    return _PartialList(
        found_list,
        complete=discovery_complete and not lineage_truncated,
        skipped_passes=skipped_passes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(
    worktree_id: str,
    cmd: List[str],
    *,
    store: StateStore,
    role: str = DEFAULT_ROLE,
    variant: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> WorktreeRecord:
    """Spawn a detached process for *worktree_id* and record its PID.

    Parameters
    ----------
    worktree_id:
        The ID of the worktree record in *store*.
    cmd:
        Command + arguments to run.  Must be non-empty.
    store:
        The active ``StateStore`` instance (carries ``WorktreeRecord``).
    role:
        Identifies the process within the worktree (e.g. ``"main"``).
    variant:
        The name of the ``start:`` contract step that produced *cmd*, if
        any. Recorded under ``record.variants[role]`` so a later
        ``stop(variant=...)`` can resolve back to this *role* -- see
        ``WorktreeRecord.variants``'s docstring. ``None`` (the default)
        means no variant is being tracked for this call, and any stale
        entry left over from a previous ``start()`` of this same *role* is
        cleared.
    env:
        Full environment for the child process.  ``None`` inherits the
        current process environment.
    cwd:
        Working directory for the child.  ``None`` inherits the current
        directory.

    Log filename contract (ticket #111)
    ------------------------------------
    Captured output is written to
    ``<log_dir_for(worktree_id)>/start-<sanitized role, case preserved>.log``.
    The role is sanitized for filesystem safety (runs of characters other
    than ``[A-Za-z0-9]`` collapse to a single ``-``, leading/trailing ``-``
    stripped, empty result falls back to ``"_"`` -- the substitution step
    alone can never produce ``"_"`` (its output alphabet is exactly
    ``[A-Za-z0-9-]``), so a non-degenerate role (one with at least one
    alphanumeric character) can never collide with the fallback; a role
    that is itself entirely non-alphanumeric is degenerate and already
    collides with other degenerate roles rather than being a separate case;
    see ``_role_log_slug``'s docstring) but is never
    lower-cased, so the filename's role token is always identical to the
    literal key used in ``record.pids``/``record.job_names``/
    ``record.variants``/``record.start_log_paths``, and is the key under
    which that path is recorded in ``record.start_log_paths``. A consumer
    holding a role name can therefore reconstruct the log path itself, or
    read it directly off ``record.start_log_paths[role]``. Roles that
    differ only by case (e.g. ``"roleA"`` vs ``"rolea"``) produce distinct
    filename strings but name the same physical file on a case-insensitive
    filesystem (Windows, default macOS) -- pick role names that differ by
    more than case if per-role log isolation matters.

    Raises
    ------
    WorktreeNotFoundError
        If *worktree_id* is not in *store*.
    ProcessAlreadyRunningError
        If ``record.pids[role]`` already exists AND the process is alive.
    ValueError
        If *cmd* is empty.
    """
    # Import here to avoid a circular-import at module level (manager imports
    # us and manager defines WorktreeNotFoundError).
    from .manager import WorktreeNotFoundError
    # Lazy import mirrors the existing _resolve_shell lazy import in
    # manager.start, avoiding a circular import (setup.runner does not import
    # process_lifecycle, but importing at module level here would still tie
    # this module's import order to setup.runner's).
    from ..setup.runner import log_dir_for

    if not cmd:
        raise ValueError("cmd must be a non-empty list")

    record = store.get(worktree_id)
    if record is None:
        raise WorktreeNotFoundError(
            f"No worktree tracked with id '{worktree_id}'"
        )

    existing_pid = record.pids.get(role, 0)
    if existing_pid and _pid_alive(existing_pid):
        raise ProcessAlreadyRunningError(worktree_id, role, existing_pid)

    log_dir = log_dir_for(worktree_id, env=env)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"start-{_role_log_slug(role)}.log"

    proc = _spawn_detached(cmd, env=env, cwd=cwd, log_path=log_path)

    try:
        proc.wait(timeout=_EARLY_EXIT_WAIT_SEC)
    except subprocess.TimeoutExpired:
        status = "running"
        returncode = None
    else:
        status = "exited"
        returncode = proc.returncode

    record.pids[role] = proc.pid
    record.status = status
    # Ticket #99: status is unconditionally overwritten above (it always was,
    # even before this ticket -- a start() for any role has always reset the
    # record-wide status), so any stop_detail attached by an earlier
    # stop_incomplete outcome is now stale by the same invariant: stop_detail
    # is not None implies status == "stop_incomplete". Clear it in lockstep.
    record.stop_detail = None
    # Ticket #110: mirror the stop_detail clear immediately above -- a
    # leftover stop_attempt from a previous stop() call on this role is
    # equally stale once a new process has been spawned for it.
    record.stop_attempt = None
    # Ticket #126: a restarted environment is a new logical lifecycle and
    # must earn a fresh teardown -- a `teardown_ran=True` marker left over
    # from a prior remove() attempt (or synthesised test state) must not
    # suppress teardown: steps for THIS lifecycle's eventual removal.
    record.teardown_ran = False
    record.returncode = returncode
    # Ticket #119: unconditional per-role assignment -- unlike job_names/
    # variants below, there is no "not available this time" branch here:
    # start() always creates a log file for this role, so there is nothing
    # to pop when absent. Restarting the same role overwrites that role's
    # single entry rather than accumulating.
    record.start_log_paths[role] = str(log_path)
    # Ticket #95, R5 (fix cycle: per-role, not a record-wide scalar --
    # mirrors how `pids` is keyed by role): persist the Job Object name (if
    # one was successfully created and the child assigned to it) under this
    # specific role's key, so stop() can later enumerate/terminate its
    # members without disturbing any OTHER role's job. On POSIX, or on any
    # Windows failure, no Job Object exists for this role -- pop any stale
    # entry left over from a previous start() of this same role rather than
    # storing a None (job_names values are always real job names, never
    # None -- absence of the key IS "no containment available").
    spawned_job_name = getattr(proc, "_worktree_job_name", None)
    if spawned_job_name is not None:
        record.job_names[role] = spawned_job_name
    else:
        record.job_names.pop(role, None)
    # Ticket #104: mirrors job_names' role-keyed, no-None-value convention --
    # record which variant started this role so stop(variant=...) can
    # resolve back to it, and clear any stale entry from a previous start()
    # of this same role when no variant is given this time.
    if variant is not None:
        record.variants[role] = variant
    else:
        record.variants.pop(role, None)
    store.update(record)

    return record


def stop(
    worktree_id: str,
    *,
    store: StateStore,
    role: str = DEFAULT_ROLE,
    timeout: float = 10.0,
    kill_orphans: bool = False,
) -> WorktreeRecord:
    """Stop the process recorded under *role* for *worktree_id*.

    Sends a graceful signal, waits up to *timeout* seconds, then force-kills
    if the process is still alive.  Clears ``pids[role]``; sets
    ``status="stopped"`` only when no other roles remain in ``pids`` AND
    nothing this call tried to kill is still alive.

    If the PID is already dead, clears the record gracefully without raising.

    Ticket #119: unlike ``job_names[role]``/``variants[role]``, the stopping
    role's ``start_log_paths[role]`` entry is deliberately *retained* rather
    than cleared, so the caller of this very call can still read the log
    path of the process it just stopped off the returned record.

    Process-tree kill (ticket #87)
    -------------------------------
    Before any signal is sent, the tracked PID's descendant process tree is
    snapshotted via :func:`_process_tree` -- this is what makes a grandchild
    spawned by a nested shell (e.g. a ``run:`` step whose command string
    itself invokes another shell) reachable: once the tracked (wrapper) PID
    is killed, the OS would otherwise reparent that grandchild away, making
    it invisible to anything that only looks at the tracked PID. On POSIX,
    :func:`_signal_process_group` additionally signals the tracked PID's
    whole process group in one shot when it is the group's leader (as
    ``start_new_session=True`` guarantees for processes this module spawns);
    :func:`_process_group_members` snapshots that group's *other* member
    PIDs before that signal is sent, so a same-group descendant absent from
    the ppid-tree snapshot (e.g. already detached out of the tracked PID's
    lineage) is not just signalled once and forgotten -- it is folded into
    the tree-kill step below (force-killed if it ignores the graceful
    signal) and into the final survivor re-probe. After the tracked PID
    itself is handled, the snapshotted tree -- plus those process-group-only
    members -- is killed via :func:`_kill_process_tree` -- this step is
    UNCONDITIONAL (it runs regardless of *kill_orphans*): killing the
    tree/group of the process we ourselves spawned is the meaning of "stop",
    not orphan-hunting.

    What the unconditional kill actually covers, per platform:

    - **Windows, job assigned and a live handle available** (``job_name`` is
      truthy and :func:`_open_job_object` returned a handle): every
      descendant of the tracked process is a job member at any depth,
      regardless of ppid lineage -- this is what reaches a
      ``Start-Process``/ShellExecuteEx-delegated launch, which lands outside
      the ppid tree entirely and so cannot be found by :func:`_process_tree`
      at any recursion depth. :func:`_terminate_job_object`'s single
      ``TerminateJobObject`` call kills every one of them together.
      ``CREATE_BREAKAWAY_FROM_JOB`` cannot be used to escape this
      particular job: :func:`_create_job_object` sets no limit flags and
      never calls ``SetInformationJobObject``, so without
      ``JOB_OBJECT_LIMIT_BREAKAWAY_OK`` the OS refuses any breakaway
      ``CreateProcess`` outright. A job member list that hit the
      ``_JOB_MEMBER_LIST_MAX_SLOTS`` cap (reported as the
      ``"job_member_list_truncated"`` ``stop_detail`` reason -- see "Status
      honesty" below) is an enumeration/reporting gap only:
      :func:`_terminate_job_object` still terminates the job as a whole, so
      members past the cap are killed regardless of whether they were ever
      enumerated.
    - **Windows, no job or no live handle** (rule N7 -- job creation or
      assignment failed at :func:`_spawn_detached` time, or
      :func:`_open_job_object` returns ``None`` here): containment degrades
      to the ppid tree snapshot alone. :func:`_process_group_members` and
      :func:`_signal_process_group` are both unconditional no-ops on
      Windows (``sys.platform == "win32"`` makes them return ``[]``/
      ``False`` immediately), so there is no group-based fallback the way
      POSIX has below.
    - **POSIX**: there is no Job Object mechanism at all
      (:func:`_create_job_object`/:func:`_open_job_object` both return
      ``None`` off ``win32``). Containment is the ppid tree snapshot plus,
      when the tracked pid is the leader of its own process group
      (``start_new_session=True`` guarantees this for everything this
      module spawns), that group's other members via
      :func:`_signal_process_group`/:func:`_process_group_members`. A
      descendant that calls ``setsid()`` itself leaves that group before
      this snapshot is taken, and is invisible to both mechanisms if its
      intermediate parent has already exited (classic double-fork
      daemonization).

    ``kill_orphans`` is **path-scoped, not lineage-scoped**: when *True*, a
    further pass using :func:`_kill_blocking_processes` is run against
    ``record.path`` after the tree kill and the Job Object terminate step --
    the path-heuristic (cwd/cmdline-token/open-file/Windows handle-table)
    orphan scan implemented by :func:`_find_blocking_processes`. It kills
    anything it finds under ``record.path``, regardless of who started it,
    and never consults ``record.pids`` -- a fundamentally different scope
    from the ppid/group/job containment above, not a deeper version of it.
    Concretely, on Windows with a successfully-assigned job and a live
    handle, there is no process this ``stop()`` call is responsible for that
    escapes the unconditional kill above -- ``kill_orphans`` there widens
    *scope* (anything under the path), not containment depth. The genuine
    gaps it closes are: a POSIX descendant that detached via ``setsid()``
    out of both the ppid tree and the process group (the canonical case,
    ticket #87); a Windows role whose job was never created or never
    successfully assigned (:func:`_assign_process_to_job` returned
    ``False``); a Windows role whose job handle is not available at
    ``stop()`` time (:func:`_open_job_object` returned ``None``); a process
    spawned by a ``setup:`` step via ``SetupRunner``'s default runner, which
    never creates or joins a Job Object and whose pid never enters
    ``record.pids`` -- entirely outside every mechanism above, reachable
    only because it still runs with the worktree as its cwd; and the
    sub-millisecond window documented on :func:`_assign_process_to_job`
    between a child's ``Popen`` returning and its job assignment landing.
    It does **not** help with a ``"job_member_list_truncated"`` outcome:
    those members were already killed by the unconditional
    ``TerminateJobObject`` call above, regardless of enumeration -- there is
    nothing left for the orphan scan to find.

    Do not pass ``kill_orphans=True`` defensively on every call: on Windows
    its cost is dominated by Pass 1c, a **system-wide** OS handle-table scan
    (:func:`_win_handle_holders` -- a full system handle table routinely
    holds 100k+ entries), budgeted at ``_HANDLE_SCAN_BUDGET_SEC`` (15.0s)
    within an overall ``_DISCOVERY_MAX_SEC`` (20.0s) discovery ceiling. This
    pass exists because ``proc.cwd()`` raises ``AccessDenied`` for most
    foreign processes on Windows, so it is often the scan's only real
    coverage there. It also carries a real cost against *timeout* itself:
    whenever ``kill_orphans=True``, ``_ORPHAN_SCAN_FLOOR_SEC`` (3.0s) of the
    caller's *timeout* budget is reserved for this pass alone -- see the
    budget paragraph below.

    The *timeout* budget is shared across the primary signal/wait step, the
    tree kill, and the optional orphan scan: each later step receives only
    the time that remains after the previous one completes, so the total
    operation is always bounded by *timeout* seconds. Ticket #95, finding 2:
    the primary step is additionally capped by :func:`_compute_stop_budget`
    so it can never consume the *entire* budget and starve the tree kill /
    orphan scan down to a 0.0s slice -- see that function's docstring for
    the exact split. ``_wait_or_kill`` polls on a 0.1s tick, so the primary
    step can overshoot its own cap by up to one tick plus the force-kill
    cost. Since ticket #95's round-3 starvation fix, the tree-kill step's
    budget is ``max(tree_floor, deadline - time.monotonic() - orphan_floor)``
    rather than a plain ``max(0.0, ...)`` clamp, so in the pathological case
    where the primary wait already overran into or past the deadline, the
    tree kill still gets its full ``tree_floor`` seconds rather than a
    near-zero slice. That means the "total bounded by *timeout*" guarantee
    above holds only to within one poll tick plus force-kill cost in the
    common case, but can stretch up to ``tree_floor`` seconds (up to 1.0s by
    default) past *timeout* in that pathological overrun case -- an
    intentional trade-off protecting the tree kill from starvation, not a
    regression. With ``kill_orphans=True``
    on a tight *timeout*, the orphan scan's Windows handle-table pass (Pass
    1c) may only get a truncated slice of its own budget -- pass a larger
    *timeout* (e.g. 20.0) for a caller that needs a guaranteed-complete
    Windows orphan scan.

    ``record.killed_pids`` (ticket #95, finding 1) is set to every PID this
    call actually attempted to terminate -- the tree kill, the tracked PID
    itself (only when it was alive at entry), and the orphan scan (when
    ``kill_orphans=True``) -- de-duplicated, deepest-first. This is a
    transient, in-memory-only field: it is never persisted to ``state.yaml``
    (see ``WorktreeRecord.killed_pids``'s own docstring). Ticket #110,
    finding #110-3: every entry carries a ``source`` -- ``"tree"``,
    ``"process_group"``, ``"job_object"``, ``"tracked"``, or
    ``"orphan_scan"`` -- naming which discovery mechanism found it, and
    name/cmdline are captured at snapshot time (before any signal is sent)
    wherever the OS can still answer, so a ``"process_group"``/
    ``"job_object"`` entry is no longer an anonymous bare PID. A
    ``"job_object"``-sourced entry may legitimately be an OS-created
    artifact of the job (e.g. a stray ``conhost.exe``), not something this
    module spawned directly -- ``source`` is what a caller should
    filter/group on, never an empty ``name`` alone. A pid discovered by more
    than one mechanism keeps whichever entry carries non-empty name/cmdline
    (dedup prefers the richest entry, not first-wins).

    ``record.stop_attempt`` (ticket #110, finding #110-2) is a
    ``state.StopAttempt`` distinguishing "the tracked pid was alive at
    entry" (``"killed"``) from "the tracked pid was already dead AND
    genuinely nothing else was found" (``"already_exited"``) from "the
    tracked pid was already dead BUT other process(es) from its tree/
    process-group/job were found and killed anyway" (``"tracked_pid_missing"``
    -- e.g. a composite/chained shell command whose wrapper exited while a
    backgrounded child it spawned kept running under a different, untracked
    pid). Orthogonal to ``stop_detail``/``status`` above and equally
    transient -- see :class:`state.StopAttempt`'s own docstring.

    Status honesty (ticket #87)
    -----------------------------
    After every kill step, liveness of the tracked PID, every node in the
    snapshotted tree, and everything the orphan scan (if run) reported is
    re-probed. If anything survived, ``status`` is set to
    ``"stop_incomplete"`` (sticky until the next ``start()``) and a warning
    naming the survivor PIDs is logged -- ``stop()`` must never report
    ``"stopped"`` while a process it was responsible for is demonstrably
    still running. ``pids[role]`` is still cleared either way -- retaining a
    possibly-dead/reused PID there would be worse (see the module-level
    docstring / ticket for the postcondition this preserves).

    Ticket #99 (reporting half of #99, mechanical half already closed by
    #95's Job Object containment): every one of the four branches below that
    sets ``status = "stop_incomplete"`` also attaches a
    ``state.StopDetail`` to ``record.stop_detail`` -- ``reason`` names
    exactly which branch fired (``"survivors"``, ``"tree_truncated"``,
    ``"job_member_list_truncated"``, or ``"orphan_scan_incomplete"``, in that
    if/elif precedence order, unchanged from before this ticket), ``message``
    is the identical string also passed to ``_logger.warning(...)`` so the
    log and the field can never drift, and ``kill_orphans_may_help`` hints
    whether re-calling with ``kill_orphans=True`` might resolve it. The
    field is cleared (``None``) the moment ``status`` is set to anything
    other than ``"stop_incomplete"`` by this function or by
    ``start()``/``manager.WorktreeManager.stop()``/``yaml_store.reconcile()``,
    and is otherwise left untouched (sticky, exactly like ``status`` itself)
    -- see ``state.StopDetail``'s own docstring for the full invariant.

    A capped tree snapshot is likewise never trusted as clean (ticket #87
    follow-up, finding F2): when the pre-kill :func:`_process_tree` snapshot
    collected exactly ``_MAX_TREE_NODES`` entries, this call cannot
    guarantee it saw the whole tree -- the excess beyond the cap was never
    even examined, let alone killed -- so ``status`` is set to
    ``"stop_incomplete"`` (with its own warning) even when every collected
    PID came back dead. Reporting ``"stopped"`` purely on the strength of a
    capped snapshot would reintroduce exactly the false-positive class this
    ticket exists to eliminate.

    Known limitation -- PID reuse (out of scope for this ticket): the
    tracked PID is trusted as-is, with no identity check against the
    process it originally belonged to. If the OS has recycled that PID
    number for an unrelated process by the time ``stop()`` runs, this call
    will signal/kill *that* process's tree/group instead, believing it to
    be the one it started. A ``psutil.Process(pid).create_time()`` comparison
    against a start-time recorded by :func:`start` would close this gap; it
    is a known, pre-existing follow-up, not introduced by this ticket, and
    intentionally not implemented here (it would touch the record schema and
    :func:`start`).

    Parameters
    ----------
    worktree_id:
        The ID of the worktree record in *store*.
    store:
        The active ``StateStore`` instance.
    role:
        Identifies the process within the worktree.
    timeout:
        Seconds to bound the complete stop operation (primary kill + tree
        kill + orphan scan combined).  Graceful exit is attempted first;
        force-kill is used if a process has not exited by the deadline.
    kill_orphans:
        When ``True``, additionally run the path-scoped orphan scan (path
        heuristics against ``record.path``, not process lineage) after the
        tree kill -- see "Process-tree kill (ticket #87)" above for exactly
        what this does and does not add over the unconditional kill.
        Defaults to ``False`` to preserve backward-compatible behaviour.

    Raises
    ------
    WorktreeNotFoundError
        If *worktree_id* is not in *store*.
    ProcessNotRunningError
        If no PID is recorded for *role* (``pids`` has no entry for the role).
    """
    from .manager import WorktreeNotFoundError

    record = store.get(worktree_id)
    if record is None:
        raise WorktreeNotFoundError(
            f"No worktree tracked with id '{worktree_id}'"
        )

    if role not in record.pids:
        raise ProcessNotRunningError(worktree_id, role)

    pid = record.pids[role]

    # Compute a shared deadline so that the primary kill step, the tree kill,
    # and the optional orphan scan together never exceed the caller-supplied
    # timeout.
    deadline = time.monotonic() + timeout

    # Ticket #95, finding 2: pre-split the budget so the primary wait can
    # never starve the tree kill / orphan scan down to zero. See
    # _compute_stop_budget's docstring for the full rationale. tree_floor is
    # actually enforced at the _kill_process_tree call site below (ticket
    # #95 fix cycle, blocking finding) -- it used to be unpacked into a
    # throwaway variable and discarded, so an overrun of the primary wait
    # past its own primary_cap (acknowledged as possible -- _wait_or_kill
    # polls on a 0.1s tick) could still starve the tree kill down to 0.0s,
    # reintroducing a narrower version of this ticket's own original bug.
    primary_cap, tree_floor, orphan_floor = _compute_stop_budget(timeout, kill_orphans)

    # Snapshot the descendant tree BEFORE any signal is sent (ticket #87):
    # once the root dies, a reparented grandchild becomes invisible to
    # anything that only looks at the tracked PID.
    tree = _process_tree(pid)

    # Ticket #110, finding #110-3: describe the tracked pid itself BEFORE any
    # signal is sent -- once it is killed, psutil can no longer answer
    # name()/cmdline() for it. Used below to enrich the tracked-pid
    # KilledProcessInfo entry instead of constructing it with an empty
    # name/cmdline (the root cause of "killed_pids" entries that look like
    # anonymous bare PIDs).
    tracked_pid_name, tracked_pid_cmdline = _describe_pid(pid)

    # Ticket #87 follow-up, finding F2: _process_tree's _MAX_TREE_NODES cap
    # truncates silently -- a tree with more descendants than the cap allows
    # leaves the excess neither collected, killed, nor checked below. This
    # snapshot alone does not distinguish "truncated" from "the caller's
    # own list" (that would require widening _process_tree's return type --
    # see its docstring); instead, treat "collected exactly the cap" as
    # "cannot guarantee completeness": a genuinely smaller real tree would
    # never have produced this many entries. The only false positive this
    # can produce is the rare boundary case of a real tree that happens to
    # be exactly cap-sized with no more descendants -- accepted as the
    # conservative, cheap trade-off documented on _process_tree.
    tree_possibly_truncated = len(tree) >= _MAX_TREE_NODES

    # Ticket #95, R5: Windows Job Object containment -- a ppid-independent
    # mechanism that closes the gap _process_tree's ppid-walk cannot by
    # construction: a ShellExecuteEx-delegated launch (what `Start-Process`
    # uses without stream redirection) lands OUTSIDE our ppid lineage
    # entirely, so no amount of recursion depth in _process_tree can ever
    # find it. Enumerated BEFORE any signal is sent -- same rationale as the
    # tree/group-member snapshots below: once a member dies, the job's own
    # membership list shrinks, so this must be captured first.
    # Ticket #95 fix cycle: look up THIS role's job name from the per-role
    # mapping, not a record-wide scalar -- a record-wide field would have
    # stop(role="main") open/terminate whichever role's job happened to be
    # written there last (e.g. role "worker"'s job, if "worker" was started
    # after "main"), corrupting an unrelated role's containment.
    job_name = record.job_names.get(role)
    job_handle: Optional[int] = None
    job_member_pids: List[int] = []
    job_list_truncated = False
    job_pid_descriptions: Dict[int, Tuple[str, List[str]]] = {}
    if job_name:
        job_handle = _open_job_object(job_name)
        if job_handle is not None:
            job_members = _job_object_member_pids(job_handle)
            job_list_truncated = not getattr(job_members, "complete", True)
            job_member_pids = list(job_members)
            # Ticket #110, finding #110-3: describe each member BEFORE
            # _terminate_job_object runs below -- same snapshot-time
            # rationale as tracked_pid_name/tracked_pid_cmdline above.
            # Bounded by _DESCRIBE_MAX_PIDS: beyond the cap, an entry keeps
            # empty metadata but still carries source="job_object".
            for _jidx, _jpid in enumerate(job_member_pids):
                if _jidx >= _DESCRIBE_MAX_PIDS:
                    break
                job_pid_descriptions[_jpid] = _describe_pid(_jpid)
        # else: no live handle available (POSIX, or OpenJobObjectW failed --
        # e.g. a restarted host whose predecessor's keeper handle already
        # closed the object). This is rule N7 -- a fallback degrading to the
        # ppid-tree/process-group path, never itself a stop_incomplete
        # trigger.

    # POSIX only, also snapshotted BEFORE any signal (ticket #87 follow-up,
    # finding B3): the PIDs of any OTHER process sharing *pid*'s process
    # group. _signal_process_group below fires a single SIGTERM at the whole
    # group -- this snapshot is what lets a same-group descendant that
    # ignores that SIGTERM (and is not also reachable via the ppid-tree
    # snapshot above, e.g. because it already detached out of *pid*'s
    # lineage) still get force-killed and checked by the survivor re-probe,
    # instead of silently surviving while stop() still reports "stopped".
    group_member_pids = _process_group_members(pid)
    # Ticket #110, finding #110-3: describe each member BEFORE the graceful
    # signal below -- same snapshot-time rationale as tracked_pid_name/
    # tracked_pid_cmdline and job_pid_descriptions above. Bounded by
    # _DESCRIBE_MAX_PIDS: beyond the cap, an entry keeps empty metadata but
    # still carries source="process_group".
    group_pid_descriptions: Dict[int, Tuple[str, List[str]]] = {}
    for _gidx, _gpid in enumerate(group_member_pids):
        if _gidx >= _DESCRIBE_MAX_PIDS:
            break
        group_pid_descriptions[_gpid] = _describe_pid(_gpid)

    # POSIX only: if *pid* is itself the leader of its own process group (as
    # start_new_session=True guarantees for processes we spawned), signal
    # the whole group in one shot. No-op on Windows, when *pid* is not a
    # group leader, or when doing so would signal our own group.
    _signal_process_group(pid, force=False)

    pid_was_alive = _pid_alive(pid)
    if pid_was_alive:
        _send_graceful_signal(pid, group_leader=True)
        _wait_or_kill(pid, max(0.0, min(deadline - time.monotonic(), primary_cap)))

    # Kill the snapshotted descendant tree, plus any process-group members
    # not already covered by it (finding B3) -- both get the identical
    # signal/wait/force-kill treatment via _kill_process_tree. This is
    # UNCONDITIONAL -- not gated on kill_orphans -- because it is completing
    # "stop" for the process tree/group we ourselves are responsible for,
    # not orphan-hunting.
    tree_pids = {info.pid for info in tree}
    group_only_infos = [
        KilledProcessInfo(
            pid=gpid,
            name=group_pid_descriptions.get(gpid, ("", []))[0],
            cmdline=group_pid_descriptions.get(gpid, ("", []))[1],
            source="process_group",
        )
        for gpid in group_member_pids
        if gpid not in tree_pids and gpid != pid
    ]
    # Ticket #95, R5: fold in any Job Object member not already covered by
    # the ppid tree or the process-group snapshot -- this is precisely the
    # ShellExecuteEx-delegated-grandchild case the job object exists to
    # catch. Host process and the tracked pid itself are always excluded
    # (both already handled elsewhere).
    already_covered = tree_pids | {info.pid for info in group_only_infos}
    job_only_infos = [
        KilledProcessInfo(
            pid=jpid,
            name=job_pid_descriptions.get(jpid, ("", []))[0],
            cmdline=job_pid_descriptions.get(jpid, ("", []))[1],
            source="job_object",
        )
        for jpid in job_member_pids
        if jpid not in already_covered and jpid != pid and jpid != os.getpid()
    ]
    killed_tree = tree + group_only_infos + job_only_infos
    # Ticket #95 fix cycle (blocking finding): enforce tree_floor here, not
    # just reserve orphan_floor -- the pre-fix `max(0.0, ...)` clamp let an
    # overrun of the primary wait (past its own primary_cap; _wait_or_kill
    # only polls on a 0.1s tick, so this is acknowledged as possible) push
    # `deadline - time.monotonic()` to zero or negative, silently starving
    # the tree kill down to 0.0s despite _compute_stop_budget's tree_floor
    # promising it would never happen. max(tree_floor, ...) guarantees this
    # step always gets at least tree_floor seconds, exactly as documented.
    _kill_process_tree(
        killed_tree,
        timeout=max(tree_floor, deadline - time.monotonic() - orphan_floor),
    )

    # Ticket #95, R5: the ppid-independent force step -- terminates every
    # process still assigned to the job in one call, catching anything that
    # ignored the graceful signal above (or was never reachable by it at
    # all, e.g. it holds no console to deliver CTRL_BREAK_EVENT to). Run
    # AFTER the tree kill (which already gave everything a graceful chance)
    # and BEFORE the orphan scan.
    if job_handle is not None:
        _terminate_job_object(job_handle)
        # Ticket #95 fix cycle (blocking finding): release the handle this
        # call opened/queried, and evict it from the _JOB_HANDLES keeper
        # registry when it was that registry's own handle -- otherwise this
        # leaks one kernel HANDLE (and, in the registry case, one
        # _JOB_HANDLES entry) per stop() call, indefinitely, on a
        # long-lived host process. Covers both paths _open_job_object can
        # serve a handle from: the common keeper-registry case (this host
        # created the job at start() time) and the OpenJobObjectW fallback
        # (a restarted host querying afresh) -- only the former also needs
        # a registry eviction, since the fallback handle was never in
        # _JOB_HANDLES to begin with.
        _close_job_object_handle(job_handle)
        if _JOB_HANDLES.get(job_name) == job_handle:
            _JOB_HANDLES.pop(job_name, None)

    # Orphan scan: kill processes that survived because they detached far
    # enough (e.g. into their own session/process group) to evade both the
    # tree snapshot above and the primary signal. Run this whether the
    # tracked PID was alive or dead -- it's a no-op when there are no
    # orphans. Pass remaining budget so the scan is also bounded.
    orphan_found: List[KilledProcessInfo] = []
    if kill_orphans:
        # Ticket #95 fix cycle, R5 (blocking finding): enforce orphan_floor
        # here, mirroring the tree_floor enforcement above -- the pre-fix
        # `max(0.0, ...)` clamp let an overrun of the primary wait (plus the
        # tree-kill step legitimately consuming up to tree_floor extra
        # seconds past the deadline to satisfy ITS OWN floor) push
        # `deadline - time.monotonic()` to zero or negative, silently
        # starving the orphan scan down to 0.0s despite
        # _compute_stop_budget's docstring promising it would never happen.
        # max(orphan_floor, ...) guarantees this step always gets at least
        # orphan_floor seconds, exactly as documented.
        orphan_budget = max(orphan_floor, deadline - time.monotonic())
        orphan_found = _kill_blocking_processes(record.path, timeout=orphan_budget)

    # Never report a false "stopped" (ticket #87): re-probe liveness of
    # everything this call attempted to kill. Any survivor means the
    # environment is not actually torn down, even though the tracked pid
    # entry is about to be cleared below.
    candidate_pids = [pid] + [info.pid for info in killed_tree] + [info.pid for info in orphan_found]
    survivor_pids = [p for p in candidate_pids if _pid_alive(p)]

    # Ticket #95, finding 1: killed_pids was never populated by stop() --
    # only ever written by manager.py's separate _kill_blocking_processes
    # call sites (the _teardown kill-and-retry paths). Populate it here too
    # so a caller inspecting the returned record can see exactly what this
    # call attempted to terminate. Deepest-first: the tree (already
    # deepest-first from _process_tree) comes first, then the tracked pid
    # itself (the root -- only when something was actually attempted
    # against it, i.e. it was alive at entry), then the orphan scan's
    # results. De-duplicated by pid. Deliberately transient -- see
    # WorktreeRecord.killed_pids' docstring; _record_to_dict never persists
    # it, so this never round-trips through state.yaml.
    _attempted_infos = list(killed_tree)
    if pid_was_alive:
        _attempted_infos.append(
            KilledProcessInfo(
                pid=pid,
                name=tracked_pid_name,
                cmdline=tracked_pid_cmdline,
                source="tracked",
            )
        )
    _attempted_infos.extend(orphan_found)
    # Ticket #110, finding #110-3: prefer the RICHEST entry per pid rather
    # than first-wins -- a pid can legitimately appear from more than one
    # discovery mechanism (e.g. a process-group member that the orphan scan
    # also matched by cwd), and before this fix whichever source happened to
    # be appended first silently won even when it carried no name/cmdline
    # while a later duplicate had real metadata. Richness is scored by how
    # many of {name, cmdline} are populated (0, 1, or 2) so a candidate with
    # BOTH beats one with only one populated, which beats one with neither --
    # not just an any-vs-none check, which would (wrongly) keep a
    # partially-populated first entry over a strictly richer later one.
    # Insertion order (and therefore killed_pids' existing deepest-first
    # ordering) is preserved -- only the winning entry's CONTENT at that
    # position can change, never its position.
    def _richness(info: "KilledProcessInfo") -> int:
        return (1 if info.name else 0) + (1 if info.cmdline else 0)

    _seen_killed_pids: "Dict[int, KilledProcessInfo]" = {}
    for info in _attempted_infos:
        existing = _seen_killed_pids.get(info.pid)
        if existing is None:
            _seen_killed_pids[info.pid] = info
            continue
        if _richness(info) > _richness(existing):
            _seen_killed_pids[info.pid] = info
    killed_pids: List[KilledProcessInfo] = list(_seen_killed_pids.values())
    record.killed_pids = killed_pids

    # Clear the role regardless of whether the process was alive — the
    # important postcondition is that the record no longer references it.
    del record.pids[role]
    # Ticket #95 fix cycle: mirror the pids[role] clear above for this
    # role's job_names entry -- once stop() has attempted to open/enumerate/
    # terminate this role's job, the record must no longer name it, exactly
    # as pids[role] is unconditionally cleared regardless of survivor
    # status. Uses .pop(role, None) rather than `del` since job_name may be
    # absent entirely (no Job Object existed for this role -- POSIX, or
    # Windows Job Object creation failed at start() time).
    # Safe to do unconditionally even on a stop_incomplete outcome below:
    # `del record.pids[role]` above already unconditionally removed this
    # role, so no later stop(role=...) call can ever reach this record state
    # for that role again, and no reconcile/adopt pathway reads job_names
    # outside of stop() itself -- there is nothing left that depends on this
    # entry surviving a stop.
    record.job_names.pop(role, None)
    # Ticket #104: mirror the job_names[role] clear immediately above --
    # once this role's pid entry is gone, the variant that started it is no
    # longer meaningful and would otherwise leave the
    # set(variants) <= set(pids) invariant violated.
    record.variants.pop(role, None)
    # Ticket #119 -- start_log_paths[role] is intentionally NOT popped here
    # (contrast job_names/variants immediately above): the log file outlives
    # its process and callers read it off this very stop() response, so this
    # field deliberately does not honour the set(x) <= set(pids) invariant
    # that job_names/variants do.

    if survivor_pids:
        message = (
            f"stop(worktree_id={worktree_id}, role={role}): process(es) "
            f"survived termination: {survivor_pids}"
        )
        _logger.warning(message)
        record.status = "stop_incomplete"
        record.stop_detail = StopDetail(
            reason=STOP_REASON_SURVIVORS,
            message=message,
            role=role,
            survivor_pids=tuple(survivor_pids[:_STOP_DETAIL_MAX_PIDS]),
            survivor_count=len(survivor_pids),
            kill_orphans_may_help=not kill_orphans,
        )
    elif tree_possibly_truncated:
        # Ticket #87 follow-up, finding F2: every candidate we DID collect
        # came back dead, but the tree snapshot hit the _MAX_TREE_NODES cap,
        # so an unknown number of descendants beyond it were never even
        # examined. Reporting "stopped" here would be exactly the silent
        # false positive this ticket exists to eliminate.
        message = (
            f"stop(worktree_id={worktree_id}, role={role}): descendant tree "
            f"for pid {pid} was truncated at the {_MAX_TREE_NODES}-node cap "
            f"-- cannot guarantee every descendant was found and killed, "
            f"reporting stop_incomplete instead of stopped"
        )
        _logger.warning(message)
        record.status = "stop_incomplete"
        record.stop_detail = StopDetail(
            reason=STOP_REASON_TREE_TRUNCATED,
            message=message,
            role=role,
            truncated_at=_MAX_TREE_NODES,
            kill_orphans_may_help=not kill_orphans,
        )
    elif job_list_truncated:
        # Ticket #95, R5: same class as tree_possibly_truncated above --
        # every candidate we DID collect came back dead, but the Job
        # Object's member list hit the _JOB_MEMBER_LIST_MAX_SLOTS cap, so an
        # unknown number of members beyond it were never even examined.
        message = (
            f"stop(worktree_id={worktree_id}, role={role}): Job Object "
            f"'{job_name}' member list was truncated at the "
            f"{_JOB_MEMBER_LIST_MAX_SLOTS}-slot cap -- cannot guarantee "
            f"every job member was found and killed, reporting "
            f"stop_incomplete instead of stopped"
        )
        _logger.warning(message)
        record.status = "stop_incomplete"
        record.stop_detail = StopDetail(
            reason=STOP_REASON_JOB_MEMBER_LIST_TRUNCATED,
            message=message,
            role=role,
            truncated_at=_JOB_MEMBER_LIST_MAX_SLOTS,
            kill_orphans_may_help=not kill_orphans,
        )
    elif (
        kill_orphans
        and "handle_scan:capped" in getattr(orphan_found, "skipped_passes", ())
    ):
        # Ticket #148: a more specific variant of the generic
        # orphan_scan_incomplete branch just below. "handle_scan:capped"
        # means the process-wide wedged-worker cap (_MAX_WEDGED_HANDLE_WORKERS)
        # was hit and no persistent worker was available to reuse -- unlike
        # the generic condition, this will NOT clear itself by retrying
        # kill_orphans or waiting longer: the cap is a process-wide
        # accumulation of permanently-blocked threads that only clears when
        # the host process itself restarts. kill_orphans_may_help is
        # therefore forced False (not merely "not kill_orphans", as the
        # generic branch computes) and the message names the actual
        # remediation. Still sets status="stop_incomplete", exactly like
        # every other stop_detail-setting branch (ticket #148 round-3
        # clarification: this is additive detail, not a different status).
        skipped_passes = getattr(orphan_found, "skipped_passes", ())
        message = (
            f"stop(worktree_id={worktree_id}, role={role}): orphan scan's "
            f"handle-table pass exhausted the process-wide wedged-worker "
            f"cap (skipped_passes={skipped_passes}) -- this will not clear "
            f"until the host process is restarted; reporting "
            f"stop_incomplete instead of stopped"
        )
        _logger.warning(message)
        record.status = "stop_incomplete"
        record.stop_detail = StopDetail(
            reason=STOP_REASON_HANDLE_SCAN_EXHAUSTED,
            message=message,
            role=role,
            skipped_passes=tuple(skipped_passes),
            kill_orphans_may_help=False,
        )
    elif kill_orphans and not getattr(orphan_found, "complete", True):
        # Ticket #95, finding 3: every candidate we DID collect came back
        # dead, but the orphan scan itself did not have full discovery
        # coverage (a pass was skipped/truncated/failed against the
        # deadline -- see _PartialList/_find_blocking_processes' D1-D9
        # rules). "Found nothing" and "never looked" used to be
        # indistinguishable ([] either way); reporting "stopped" here would
        # reintroduce exactly that silent false positive.
        skipped_passes = getattr(orphan_found, "skipped_passes", ())
        message = (
            f"stop(worktree_id={worktree_id}, role={role}): orphan scan "
            f"discovery was incomplete (skipped_passes={skipped_passes}) -- "
            f"cannot guarantee no orphans were missed, reporting "
            f"stop_incomplete instead of stopped"
        )
        _logger.warning(message)
        record.status = "stop_incomplete"
        # Ticket #99: this pass already ran (and was starved) -- unlike the
        # other three reasons, retrying with kill_orphans=True again offers
        # no new information; the remediation (a larger timeout) is stated
        # in `message` instead.
        record.stop_detail = StopDetail(
            reason=STOP_REASON_ORPHAN_SCAN_INCOMPLETE,
            message=message,
            role=role,
            skipped_passes=tuple(skipped_passes),
            kill_orphans_may_help=False,
        )
    elif not record.pids and record.status not in ("stop_incomplete", "orphaned"):
        # Only mark the whole worktree as "stopped" when all roles are gone.
        # In a multi-role worktree, stopping one role must not mask the fact
        # that other processes are still alive.
        #
        # The status exclusion mirrors yaml_store.reconcile()'s identical
        # guard (ticket #87 follow-up, finding B2): "stop_incomplete" is a
        # sticky, deliberately-honest status set by an EARLIER stop() call
        # (possibly for a different role in this same multi-role worktree)
        # that could not confirm everything it tried to kill actually died.
        # This call's own kill being clean does not re-verify that earlier
        # survivor -- clearing the last pid entry here must not silently
        # overwrite that guarantee back to "stopped". Likewise "orphaned"
        # (path gone, set by reconcile()) must not be masked by a stop()
        # call racing against it.
        record.status = "stopped"
        # Ticket #99: this call's own kill was clean and every other role is
        # gone too -- status is genuinely leaving "stop_incomplete" (if it
        # was ever set), so any stale reason from an earlier incomplete stop
        # no longer applies. Safe unconditionally: the sticky-preserve paths
        # never reach this branch -- they hit one of the elif arms above, or
        # fail the `record.status not in (...)` guard and fall through
        # untouched.
        record.stop_detail = None

    # Ticket #110: StopAttempt -- orthogonal to stop_detail/status above,
    # this answers "was the TRACKED pid itself alive at entry, or had it
    # already gone stale?" and, when stale, "did that matter?". Closes
    # finding #110-2: before this field existed, "genuinely nothing to kill"
    # and "the tracked pid had gone stale but real work it spawned (e.g. a
    # composite/chained shell command's backgrounded child) was still found
    # and killed" both surfaced identically as killed_pids: []. See
    # StopAttempt's own docstring for the full outcome vocabulary and why
    # this is a separate field rather than overloading stop_detail --
    # "tracked_pid_missing" is a SUCCESSFUL stop (something else was found
    # and killed) and must never be reported as incomplete.
    # Ticket #110 fix cycle (blocking finding #1): base this solely on
    # killed_tree (the descendant tree/process-group/job recovery this
    # message describes), NOT on orphan_found. orphan_found comes from the
    # unrelated path-heuristic orphan scan (_kill_blocking_processes,
    # cwd/cmdline matching against record.path) -- a hit there says nothing
    # about whether the TRACKED pid's own tree/group/job was still alive.
    # Folding it in mislabeled "tracked pid missing, but its tree/group/job
    # was found and killed" for a case where the tree/group/job was in fact
    # empty and only an unrelated orphan was found -- that is genuinely
    # "already_exited", just with an orphan-scan hit alongside it.
    _something_else_found = bool(killed_tree)
    if pid_was_alive:
        _stop_attempt_outcome = STOP_ATTEMPT_KILLED
        _stop_attempt_message = (
            f"stop(worktree_id={worktree_id}, role={role}): tracked pid "
            f"{pid} was alive at entry; termination attempted"
        )
    elif _something_else_found:
        _stop_attempt_outcome = STOP_ATTEMPT_TRACKED_PID_MISSING
        _stop_attempt_message = (
            f"stop(worktree_id={worktree_id}, role={role}): tracked pid "
            f"{pid} had already exited at entry, but other process(es) "
            f"from its descendant tree/process-group/job were found and "
            f"killed -- the tracked pid may have gone stale (e.g. a "
            f"composite/chained shell command) while real work it spawned "
            f"kept running"
        )
        _logger.warning(_stop_attempt_message)
    else:
        _stop_attempt_outcome = STOP_ATTEMPT_ALREADY_EXITED
        _stop_attempt_message = (
            f"stop(worktree_id={worktree_id}, role={role}): tracked pid "
            f"{pid} had already exited at entry; nothing found to kill"
        )
        _logger.debug(_stop_attempt_message)
    record.stop_attempt = StopAttempt(
        outcome=_stop_attempt_outcome,
        message=_stop_attempt_message,
        role=role,
        tracked_pid=pid,
        tracked_pid_alive=pid_was_alive,
        kill_orphans_may_help=(
            _stop_attempt_outcome == STOP_ATTEMPT_TRACKED_PID_MISSING
            and not kill_orphans
        ),
    )

    store.update(record)

    return record


__all__ = (
    "DEFAULT_ROLE",
    "KilledProcessInfo",
    "ProcessAlreadyRunningError",
    "ProcessLifecycleError",
    "ProcessNotRunningError",
    "start",
    "stop",
)
