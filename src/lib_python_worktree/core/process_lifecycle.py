"""Process lifecycle engine layer (W6/W8 — ticket #8).

Public API
----------
- ``start(worktree_id, cmd, *, store, role="main", env=None, cwd=None)``
  Spawns a detached process, persists ``pids[role]`` and ``status="running"``
  to the state store, returns the updated ``WorktreeRecord``.

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
  as stopped.  Returns the updated ``WorktreeRecord``.

Platform differences
--------------------
- Windows: ``CREATE_NEW_PROCESS_GROUP`` to detach from the MCP host's
  process group while still allowing ``CTRL_BREAK_EVENT`` delivery;
  ``TerminateProcess`` (via ctypes) for force-kill. ``start()`` also creates
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

import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .state import StateStore, WorktreeRecord
from .yaml_store import _pid_alive

_logger = logging.getLogger(__name__)

# The role key used when the caller does not supply an explicit role.
DEFAULT_ROLE = "main"

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
# cap gates only *replacement*-worker creation once a scan's own worker
# wedges (see _win_handle_holders' _bounded_query). It must NEVER be used
# to refuse to start a scan's initial worker at all: a wedged worker may,
# by definition, never return from its NtQueryObject call, so a scan-start
# gate on this cap would eventually latch permanently once
# _MAX_WEDGED_HANDLE_WORKERS wedged workers had ever accumulated -- every
# later call to _win_handle_holders would then silently return `[]` forever
# for the rest of the process's life. That was tried and rejected; do not
# reintroduce it.
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

# Ticket #87: hard ceiling (seconds) on _find_blocking_processes' own
# discovery cost -- Pass 1 (cwd), Pass 1b (cmdline tokens), Pass 1c (Windows
# handle-table scan), and Pass 2 (open files) combined. Before this fix,
# these passes had no cap of their own beyond Pass 1c's per-call budget --
# a large/slow ambient process list (observed ~75s of CPU for a single call
# that found nothing) could make discovery alone blow through whatever
# timeout the caller (e.g. stop(timeout=...)) requested. _DISCOVERY_MAX_SEC
# bounds discovery independent of any caller-supplied deadline; when a
# deadline *is* supplied, discovery is bounded by whichever of the two is
# tighter (see _find_blocking_processes).
_DISCOVERY_MAX_SEC = 20.0

# Ticket #87: seconds of the caller's deadline reserved (not spent on
# discovery) so that time remains for the actual signal/kill step once
# discovery completes. Shrunk to at most 20% of whatever time remains when
# the caller's deadline itself leaves less than this much room, so the
# reserve can never itself consume the entire budget on a very tight
# deadline.
_DISCOVERY_RESERVE_SEC = 1.0

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

def _send_graceful_signal(pid: int) -> None:
    """Send the platform-appropriate graceful-stop signal to *pid*.

    Windows: CTRL_BREAK_EVENT (sent to the process group).
    POSIX:   SIGTERM.
    """
    if sys.platform == "win32":
        try:
            os.kill(pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        except OSError:
            # Process may have already exited between the liveness check and
            # the signal call — treat as a no-op.
            pass
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


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

@dataclass
class KilledProcessInfo:
    """Information about a process that was killed to unblock worktree removal."""

    pid: int
    name: str
    cmdline: List[str] = field(default_factory=list)


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
    -- see the D1-D8 / N1-N8 rules documented on ``_find_blocking_processes``
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
        result.append(KilledProcessInfo(pid=proc_pid, name=name, cmdline=cmdline))

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

    Mirrors :func:`_signal_process_group`'s own guards exactly, so this only
    ever returns members in the same situations that function would actually
    signal the group: ``[]`` on Windows; ``[]`` when *pid* is not the leader
    of its own group (``os.getpgid(pid) != pid``); ``[]`` when that group is
    our own process group. The host process, its ancestors, and *pid* itself
    are always excluded from the result -- callers already handle *pid*
    separately.

    Never raises: any psutil/OS failure degrades to a best-effort partial (or
    empty) list rather than propagating out of ``stop()``.
    """
    if sys.platform == "win32":
        return []
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return []
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
        _send_graceful_signal(info.pid)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Budget exhausted -- signal already sent; skip the wait for this
            # node and all subsequent ones (they will also only be signalled).
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


# Process-wide ceiling accounting (ticket #90): how many _BoundedQueryWorker
# threads, across every concurrent _win_handle_holders scan in this process,
# are currently permanently blocked inside a wedged callable. Guarded by
# _wedged_worker_lock; incremented by submit() unconditionally, for every
# worker it retires -- ABANDONED or CAPPED alike, since both leave a thread
# genuinely and permanently blocked in fn() for as long as that real call
# takes (review finding, ticket #90 fix pass: a CAPPED worker's thread is
# just as live as an ABANDONED one and must be tracked the same way, or
# _MAX_WEDGED_HANDLE_WORKERS stops bounding the true number of live blocked
# threads). Decremented by the retired worker's own thread, in a
# ``finally``, the moment its wedged callable finally returns. The
# ABANDONED/CAPPED distinction governs only whether a caller may create a
# *replacement* worker for the scan's next query -- never whether this
# worker's own thread is counted.
_wedged_worker_count = 0
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
        return _wedged_worker_count < _MAX_WEDGED_HANDLE_WORKERS


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
    ``queue.get()`` forever). This is what makes a long-lived caller that
    creates many of these over its lifetime never accumulate threads
    without bound (ticket #90).
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
            state["done"].set()
            if was_abandoned:
                if callback is not None:
                    try:
                        callback(value)
                    except Exception:  # noqa: BLE001 -- best-effort cleanup
                        pass
                if slot_acquired:
                    global _wedged_worker_count
                    with _wedged_worker_lock:
                        _wedged_worker_count = max(0, _wedged_worker_count - 1)

    def submit(
        self,
        fn,
        *,
        grace: Optional[_GraceBudget] = None,
        scan_deadline: Optional[float] = None,
        on_abandoned_done=None,
    ) -> _QueryOutcome:
        """Run *fn* (a zero-arg callable) through this worker, bounded.

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
        }
        self._job_queue.put((fn, state))

        if state["done"].wait(_HANDLE_QUERY_TIMEOUT_SEC):
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
        global _wedged_worker_count
        with state["lock"]:
            if state["completed"]:
                return _QueryOutcome(_QueryStatus.RESOLVED, state["value"])
            # Every retiring worker's thread is genuinely, permanently
            # blocked in *fn* for as long as that real call takes --
            # regardless of whether the process-wide cap already had room.
            # It must therefore always be counted in here (and decremented
            # by _run() once fn() finally returns), or _wedged_worker_count
            # stops reflecting the true number of live blocked threads and
            # _MAX_WEDGED_HANDLE_WORKERS stops being a real bound on them
            # (review finding, ticket #90 fix pass). The cap only ever
            # governs whether a *replacement* worker may be created for
            # this scan's next query -- that decision is the CAPPED vs
            # ABANDONED distinction below, checked *after* this worker's
            # own slot is already claimed.
            with _wedged_worker_lock:
                _wedged_worker_count += 1
                at_cap = _wedged_worker_count >= _MAX_WEDGED_HANDLE_WORKERS
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
    -- is guaranteed to eventually receive a shutdown sentinel and exit, so
    a long-lived host process invoking this function many times over its
    lifetime does not accumulate threads without bound (the operational
    limitation this module used to accept — see the ticket for the
    real-world blowup that made it no longer acceptable). An overall
    wall-clock budget (``_HANDLE_SCAN_BUDGET_SEC``) additionally bounds the
    whole function as defense in depth: once exceeded, the scan stops early
    and returns whatever it has found so far rather than continuing
    indefinitely.

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

    SYSTEM_EXTENDED_HANDLE_INFORMATION = 64
    STATUS_SUCCESS = 0x00000000
    STATUS_INFO_LENGTH_MISMATCH = 0xC0000004
    OBJECT_NAME_INFORMATION = 1
    OBJECT_TYPE_INFORMATION = 2
    PROCESS_DUP_HANDLE = 0x0040
    DUPLICATE_SAME_ACCESS = 0x00000002

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

    # --- Step 1: dump the system-wide handle table, growing the buffer on
    # STATUS_INFO_LENGTH_MISMATCH. Bounded retries so a hostile/changing
    # buffer size can never loop forever.
    buf_size = 1 << 20  # 1 MiB initial guess
    buf = None
    for _attempt in range(8):
        buf = ctypes.create_string_buffer(buf_size)
        return_length = ctypes.c_ulong(0)
        status = ntdll.NtQuerySystemInformation(
            SYSTEM_EXTENDED_HANDLE_INFORMATION,
            buf,
            buf_size,
            ctypes.byref(return_length),
        ) & 0xFFFFFFFF
        if status == STATUS_INFO_LENGTH_MISMATCH:
            buf_size = max(buf_size * 2, return_length.value + (1 << 16))
            continue
        if status != STATUS_SUCCESS:
            # The one-shot dump itself failed -- this scan never looked at
            # any handle at all, not merely "found nothing".
            return _PartialList([], complete=False)
        break
    else:
        return _PartialList([], complete=False)

    size_t_size = ctypes.sizeof(ctypes.c_size_t)
    entry_size = ctypes.sizeof(_SystemHandleTableEntryInfoEx)
    handles_offset = 2 * size_t_size
    num_handles = ctypes.c_size_t.from_buffer_copy(buf, 0).value
    # Defend against a corrupt/short buffer reporting more handles than it
    # actually holds -- clamp rather than read out of bounds.
    max_fit = max(0, (buf_size - handles_offset) // entry_size)
    num_handles = min(num_handles, max_fit)

    # Group (handle value, object type index) pairs by owning PID so each
    # foreign process is opened (OpenProcess) at most once regardless of how
    # many of its handles we end up inspecting. ObjectTypeIndex is read
    # directly from the system handle table -- no syscall needed -- and
    # drives the type-index cache below that skips non-File handles without
    # ever duplicating them.
    by_pid: Dict[int, List[Tuple[int, int]]] = {}
    for i in range(num_handles):
        offset = handles_offset + i * entry_size
        entry = _SystemHandleTableEntryInfoEx.from_buffer_copy(buf, offset)
        pid = int(entry.UniqueProcessId)
        if pid <= 0 or pid in excluded_pids:
            continue
        by_pid.setdefault(pid, []).append(
            (int(entry.HandleValue), int(entry.ObjectTypeIndex))
        )

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

    def _query_object_raw(dup_handle, info_class: int) -> Optional[str]:
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
            if status == STATUS_SUCCESS:
                # Both ObjectNameInformation and ObjectTypeInformation begin
                # with a UNICODE_STRING as their first field, so the same
                # parsing applies to either info_class.
                uni = _UnicodeString.from_buffer_copy(name_buf, 0)
                if not uni.Buffer or uni.Length == 0:
                    return None
                return ctypes.wstring_at(uni.Buffer, uni.Length // 2)
            if status == STATUS_INFO_LENGTH_MISMATCH:
                size = max(size * 2, returned.value + 256)
                continue
            return None
        return None

    # Ticket #90: exactly one _BoundedQueryWorker services this whole scan.
    # A scan always creates and uses its initial worker, even when the
    # process-wide wedged-worker cap (_MAX_WEDGED_HANDLE_WORKERS) is
    # already full: refusing to scan at the cap was tried and rejected,
    # because a wedged worker is -- by definition -- stuck in an
    # NtQueryObject call that may NEVER return. If a scan-start check
    # instead skipped scanning once the cap filled, then once
    # _MAX_WEDGED_HANDLE_WORKERS workers accumulated process-wide,
    # every later call to this function would return `[]` forever, for
    # the remaining life of the process -- a silent, permanent, wrong
    # answer that is worse than the thread leak this ticket set out to
    # fix. The cap instead gates only *replacement*-worker creation
    # inside `_bounded_query` below (once this scan's own worker wedges),
    # never the initial worker. The accepted tradeoff is a small, bounded
    # overshoot: each concurrently-running scan may hold one worker
    # beyond the cap, bounded by the (small) number of concurrent scans.
    # Do not reintroduce a scan-start gate here.
    # It is always shut down via the try/finally below -- a healthy worker
    # drains and exits promptly; a worker retired because a query wedged
    # (see _BoundedQueryWorker.submit) has already been sent its own
    # shutdown sentinel at retirement time, so close() on it is a fast
    # no-op and the retired thread still self-terminates on its own once
    # the wedged call finally returns. Either way this scan leaks no
    # thread. See _BoundedQueryWorker's docstring for the full mechanism.
    worker = _BoundedQueryWorker()
    stop_scan = False
    cap_already_logged = False

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

    def _bounded_query(dup_handle, info_class: int):
        """Run one NtQueryObject call through the per-scan worker.

        Returns ``(_RESOLVED, value)`` on success, or ``(_STOP, None)``
        once this scan has no usable worker left (process-wide wedged-
        worker cap hit, with or without this specific query being the one
        that hit it) -- callers must stop issuing further queries in that
        case. A wedged query's handle ownership is transferred to the
        retired worker via ``on_abandoned_done`` in every non-resolved
        case; the caller must NOT close *dup_handle* itself when this
        does not return ``_RESOLVED`` (ticket #90's handle-reuse /
        false-match fix).
        """
        nonlocal worker, cap_already_logged
        outcome = worker.submit(
            lambda: _query_object_raw(dup_handle, info_class),
            grace=grace_budget,
            scan_deadline=scan_deadline,
            on_abandoned_done=_make_handle_closer(dup_handle),
        )
        if outcome.status == _QueryStatus.RESOLVED:
            return _RESOLVED, outcome.value
        if outcome.status == _QueryStatus.ABANDONED and _wedged_slot_available():
            # This worker is now permanently tied up in the still-running
            # wedged call (and has already been sent its own shutdown
            # sentinel by submit()) -- replace it so subsequent queries in
            # this scan are not blocked.
            worker = _BoundedQueryWorker()
            return _CONTINUE, None
        # Either genuinely CAPPED, or ABANDONED with no capacity left for a
        # replacement worker -- either way this scan cannot make further
        # progress on new queries. Log once per scan, at debug, so an
        # operator can see from the logs alone that a scan degraded early.
        if not cap_already_logged:
            _logger.debug(
                "_win_handle_holders: hit the process-wide wedged NtQueryObject "
                "worker cap (_MAX_WEDGED_HANDLE_WORKERS=%s, live=%s) -- "
                "stopping this scan early and returning partial results",
                _MAX_WEDGED_HANDLE_WORKERS, _wedged_worker_count,
            )
            cap_already_logged = True
        return _STOP, None

    found: List[Tuple[int, str]] = []
    current_process = kernel32.GetCurrentProcess()
    # Per-call cache: object-type-index -> resolved type name (or None if
    # unresolved). Populated lazily from the first handle seen for each type
    # index; every later handle of that type index is accepted/skipped from
    # this cache alone, with no further DuplicateHandle/NtQueryObject call.
    _UNSET = object()
    type_name_cache: Dict[int, Optional[str]] = {}

    def _process_handle(proc_handle, pid: int, handle_value: int, type_index: int) -> str:
        """Resolve one (handle_value, type_index) for *pid*.

        Returns ``_MATCHED`` (pid appended to *found*), ``_CONTINUE``
        (nothing to report, keep scanning this pid's other handles), or
        ``_STOP`` (this scan has no usable worker left -- callers must stop
        issuing further queries entirely, not just for this pid).
        """
        cached_type = type_name_cache.get(type_index, _UNSET)
        if cached_type is not _UNSET and cached_type != "File":
            return _CONTINUE  # known non-file type -- skip without duplicating

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
                status, type_name = _bounded_query(dup_handle, OBJECT_TYPE_INFORMATION)
                if status != _RESOLVED:
                    owns_handle = False
                    return _STOP if status == _STOP else _CONTINUE
                type_name_cache[type_index] = type_name
                if type_name != "File":
                    return _CONTINUE

            status, nt_name = _bounded_query(dup_handle, OBJECT_NAME_INFORMATION)
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

    # Ticket #95, finding 3 (D5 vs N2): distinguish this scan's OWN
    # budget_sec expiring mid-enumeration (deadline_truncated -- genuine
    # incompleteness, "ran out of time before it could look") from the
    # process-wide wedged-worker cap being hit (stop_scan via a CAPPED
    # verdict -- an internal degradation the design explicitly does NOT
    # treat as incompleteness, since the handle table up to that point WAS
    # actually enumerated; see _bounded_query's CAPPED branch). Both cause
    # the same early `break`, but only the deadline case sets
    # deadline_truncated -- checked as a distinct condition, and only when
    # stop_scan is not ALSO already true (i.e. the deadline is the actual
    # reason this iteration stopped, not a cap hit on a prior handle that
    # would have broken the loop already).
    deadline_truncated = False
    try:
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
                for handle_value, type_index in handle_entries:
                    if stop_scan:
                        break
                    if time.monotonic() > scan_deadline:
                        deadline_truncated = True
                        break
                    verdict = _process_handle(proc_handle, pid, handle_value, type_index)
                    if verdict == _STOP:
                        stop_scan = True
                        break
                    if verdict == _MATCHED:
                        break  # one matching handle is enough to flag this pid
            finally:
                kernel32.CloseHandle(proc_handle)
    finally:
        worker.close()

    return _PartialList(found, complete=not deadline_truncated)


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
       (via ``psutil.open_files()``).

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
    Pass 2 (``"open_files:..."``). A pass simply not applicable on this OS
    (Pass 1b/1c off Windows) never contributes a tag. An individual PID
    raising ``AccessDenied``/``NoSuchProcess`` within a pass is caught and
    skipped (``continue``) without affecting that pass's completeness -- only
    a whole pass being skipped/truncated/failed does.
    """
    import psutil

    normalized = os.path.normcase(os.path.normpath(path))

    # Build the set of PIDs to exclude: the host process and all its ancestors.
    excluded_pids: set[int] = {host_pid}
    try:
        for ancestor in psutil.Process(host_pid).parents():
            excluded_pids.add(ancestor.pid)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass

    # Ticket #87: hard-bound the wall-clock cost of discovery as a whole
    # (Pass 1, 1b, 1c, 2 combined), independent of any per-pass cost of its
    # own. `scan_stop` is computed once, at entry, from two ceilings:
    #  - `_DISCOVERY_MAX_SEC`, an absolute cap that applies even with no
    #    caller-supplied *deadline* at all;
    #  - when *deadline* IS supplied, `deadline` minus a small reserve, so
    #    that discovery never eats into the time the caller needs afterward
    #    for the actual signal/kill step. The reserve shrinks to at most 20%
    #    of whatever time remains on a very tight deadline, so it can never
    #    by itself consume the whole budget.
    # Checked at the top of every pass and inside every per-process loop
    # below so a slow ambient process list degrades gracefully to "whatever
    # was found so far" instead of blowing through the caller's budget.
    entry_ts = time.monotonic()
    if deadline is None:
        scan_stop = entry_ts + _DISCOVERY_MAX_SEC
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
        reserve = min(_DISCOVERY_RESERVE_SEC, 0.2 * remaining)
        scan_stop = min(entry_ts + _DISCOVERY_MAX_SEC, deadline - reserve)

    seen_pids: set[int] = set()
    result: List[KilledProcessInfo] = []
    # Ticket #95, finding 3: names of passes that were skipped entirely
    # (entry guard false) or truncated (inner loop broke on scan_stop),
    # failed (raised and was swallowed), in the order encountered. See the
    # D1-D8 / N1-N8 rules documented above.
    skipped_passes: List[str] = []

    # Pass 1: CWD match.
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
                try:
                    cwd = proc.cwd()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
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
                        )
                    )
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        if pass_truncated:  # D2
            skipped_passes.append("cwd:truncated")
    else:
        skipped_passes.append("cwd:skipped")  # D1

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
                    if pid in excluded_pids or pid in seen_pids:
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
    # Budget: capped at _HANDLE_SCAN_BUDGET_SEC, but shrunk further to
    # whatever remains of *deadline* (the caller's overall timeout) when one
    # was supplied, so this scan can never independently add up to
    # _HANDLE_SCAN_BUDGET_SEC on top of the caller's own budget. When no
    # time remains at all, the scan is skipped outright rather than spending
    # any of the remaining time on a scan that has no chance to report back
    # before the deadline is needed elsewhere (Pass 2 / the kill/wait step).
    if sys.platform == "win32":
        if time.monotonic() <= scan_stop:
            if deadline is None:
                handle_scan_budget = _HANDLE_SCAN_BUDGET_SEC
            else:
                handle_scan_budget = min(
                    _HANDLE_SCAN_BUDGET_SEC, max(0.0, deadline - time.monotonic())
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
                    KilledProcessInfo(pid=pid, name=proc_name or "", cmdline=cmdline)
                )
        else:
            skipped_passes.append("handle_scan:skipped")  # D4

    # Pass 2: open file handles — catches daemons that have changed their cwd
    # away from the worktree but still hold file locks inside it.
    if time.monotonic() <= scan_stop:
        pass_truncated = False
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if time.monotonic() > scan_stop:
                pass_truncated = True
                break
            try:
                pid = proc.info["pid"]
                if pid in excluded_pids or pid in seen_pids:
                    continue
                try:
                    open_files = proc.open_files()
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
                for finfo in open_files:
                    norm_fpath = os.path.normcase(os.path.normpath(finfo.path))
                    if norm_fpath.startswith(normalized + os.sep) or norm_fpath == normalized:
                        seen_pids.add(pid)
                        result.append(
                            KilledProcessInfo(
                                pid=pid,
                                name=proc.info["name"] or "",
                                cmdline=proc.info["cmdline"] or [],
                            )
                        )
                        break
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        if pass_truncated:  # D7
            skipped_passes.append("open_files:truncated")
    else:
        skipped_passes.append("open_files:skipped")  # D7

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
        _send_graceful_signal(info.pid)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Budget exhausted — signal already sent; skip the wait for this
            # orphan and all subsequent ones (they will also only be signalled).
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
    env:
        Full environment for the child process.  ``None`` inherits the
        current process environment.
    cwd:
        Working directory for the child.  ``None`` inherits the current
        directory.

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
    from ..setup.runner import _slug, log_dir_for

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
    log_path = log_dir / f"start-{_slug(role)}.log"

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
    record.returncode = returncode
    record.start_log_path = str(log_path)
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

    When *kill_orphans* is ``True``, a further pass using
    :func:`_kill_blocking_processes` is run against ``record.path`` after the
    tree kill.  This is the path-heuristic (cwd/cmdline/open-file) orphan
    scan for processes that detached far enough (e.g. into their own
    session/process group) to evade even the tree snapshot.

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
    (see ``WorktreeRecord.killed_pids``'s own docstring).

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
        When ``True``, additionally scan for and kill any orphaned processes
        under ``record.path`` (path heuristics) after the tree kill.
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
    if job_name:
        job_handle = _open_job_object(job_name)
        if job_handle is not None:
            job_members = _job_object_member_pids(job_handle)
            job_list_truncated = not getattr(job_members, "complete", True)
            job_member_pids = list(job_members)
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

    # POSIX only: if *pid* is itself the leader of its own process group (as
    # start_new_session=True guarantees for processes we spawned), signal
    # the whole group in one shot. No-op on Windows, when *pid* is not a
    # group leader, or when doing so would signal our own group.
    _signal_process_group(pid, force=False)

    pid_was_alive = _pid_alive(pid)
    if pid_was_alive:
        _send_graceful_signal(pid)
        _wait_or_kill(pid, max(0.0, min(deadline - time.monotonic(), primary_cap)))

    # Kill the snapshotted descendant tree, plus any process-group members
    # not already covered by it (finding B3) -- both get the identical
    # signal/wait/force-kill treatment via _kill_process_tree. This is
    # UNCONDITIONAL -- not gated on kill_orphans -- because it is completing
    # "stop" for the process tree/group we ourselves are responsible for,
    # not orphan-hunting.
    tree_pids = {info.pid for info in tree}
    group_only_infos = [
        KilledProcessInfo(pid=gpid, name="", cmdline=[])
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
        KilledProcessInfo(pid=jpid, name="", cmdline=[])
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
        _attempted_infos.append(KilledProcessInfo(pid=pid, name="", cmdline=[]))
    _attempted_infos.extend(orphan_found)
    _seen_killed_pids: "set[int]" = set()
    killed_pids: List[KilledProcessInfo] = []
    for info in _attempted_infos:
        if info.pid in _seen_killed_pids:
            continue
        _seen_killed_pids.add(info.pid)
        killed_pids.append(info)
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

    if survivor_pids:
        _logger.warning(
            "stop(worktree_id=%s, role=%s): process(es) survived termination: %s",
            worktree_id, role, survivor_pids,
        )
        record.status = "stop_incomplete"
    elif tree_possibly_truncated:
        # Ticket #87 follow-up, finding F2: every candidate we DID collect
        # came back dead, but the tree snapshot hit the _MAX_TREE_NODES cap,
        # so an unknown number of descendants beyond it were never even
        # examined. Reporting "stopped" here would be exactly the silent
        # false positive this ticket exists to eliminate.
        _logger.warning(
            "stop(worktree_id=%s, role=%s): descendant tree for pid %s was "
            "truncated at the %s-node cap -- cannot guarantee every "
            "descendant was found and killed, reporting stop_incomplete "
            "instead of stopped",
            worktree_id, role, pid, _MAX_TREE_NODES,
        )
        record.status = "stop_incomplete"
    elif job_list_truncated:
        # Ticket #95, R5: same class as tree_possibly_truncated above --
        # every candidate we DID collect came back dead, but the Job
        # Object's member list hit the _JOB_MEMBER_LIST_MAX_SLOTS cap, so an
        # unknown number of members beyond it were never even examined.
        _logger.warning(
            "stop(worktree_id=%s, role=%s): Job Object '%s' member list was "
            "truncated at the %s-slot cap -- cannot guarantee every job "
            "member was found and killed, reporting stop_incomplete instead "
            "of stopped",
            worktree_id, role, job_name, _JOB_MEMBER_LIST_MAX_SLOTS,
        )
        record.status = "stop_incomplete"
    elif kill_orphans and not getattr(orphan_found, "complete", True):
        # Ticket #95, finding 3: every candidate we DID collect came back
        # dead, but the orphan scan itself did not have full discovery
        # coverage (a pass was skipped/truncated/failed against the
        # deadline -- see _PartialList/_find_blocking_processes' D1-D8
        # rules). "Found nothing" and "never looked" used to be
        # indistinguishable ([] either way); reporting "stopped" here would
        # reintroduce exactly that silent false positive.
        _logger.warning(
            "stop(worktree_id=%s, role=%s): orphan scan discovery was "
            "incomplete (skipped_passes=%s) -- cannot guarantee no orphans "
            "were missed, reporting stop_incomplete instead of stopped",
            worktree_id, role, getattr(orphan_found, "skipped_passes", ()),
        )
        record.status = "stop_incomplete"
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
