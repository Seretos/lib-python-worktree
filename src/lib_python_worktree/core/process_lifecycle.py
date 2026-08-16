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
  ``TerminateProcess`` (via ctypes) for force-kill.
- POSIX:   ``start_new_session=True``; ``SIGTERM`` for graceful stop;
  ``SIGKILL`` for force-kill.

The ``_pid_alive`` helper is imported from ``yaml_store`` so there is a
single, tested implementation.

No ``mcp`` imports; returns plain dataclasses (``WorktreeRecord``).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
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
_HANDLE_QUERY_TIMEOUT_SEC = 0.01

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

    return proc


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


def _win_handle_holders(
    path: str,
    excluded_pids: "set[int]",
    *,
    budget_sec: float = _HANDLE_SCAN_BUDGET_SEC,
) -> List[Tuple[int, str]]:
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
    dispatched through a single reusable background worker thread guarded
    by a bounded queue/event timeout, rather than spawning a new OS thread
    per handle (thread-creation overhead alone made a naive one-thread-per-
    handle implementation take upwards of tens of minutes against a normal
    handle table -- confirmed impractically slow in practice). When a query
    does not return within the timeout, the worker thread is permanently
    wedged on it; that single call is abandoned (the thread is a daemon and
    is simply replaced with a fresh worker for subsequent queries) rather
    than blocking the rest of the scan. An overall wall-clock budget
    (``_HANDLE_SCAN_BUDGET_SEC``) additionally bounds the whole function as
    defense in depth: once exceeded, the scan stops early and returns
    whatever it has found so far rather than continuing indefinitely.

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
    import queue
    import threading
    from ctypes import wintypes

    scan_deadline = time.monotonic() + max(0.0, budget_sec)

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
            return []
        break
    else:
        return []

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

    # A single reusable background worker thread services bounded queries.
    # Spawning a fresh OS thread per handle was measured to be prohibitively
    # slow against a full system handle table (thread-creation overhead
    # dominates at that volume); reusing one worker via a queue/event makes
    # the common (fast, non-hanging) case near-zero overhead. If a query
    # ever wedges the worker (timeout elapses while it's still running), that
    # worker is abandoned -- it is a daemon thread, so it either eventually
    # returns and is garbage-collected or blocks forever harmlessly -- and a
    # fresh worker replaces it for subsequent queries.
    #
    # Known operational limitation (accepted, not an oversight): this
    # replacement has no cross-call cap. A long-lived host process that
    # invokes _win_handle_holders many times over its lifetime (many
    # worktree teardowns) can accumulate one permanently-wedged daemon
    # thread per genuine NtQueryObject hang encountered, for as long as the
    # host process keeps running -- each abandoned thread is never joined or
    # cleaned up. In practice this is judged acceptable: each such thread is
    # small and inert (blocked in a kernel call, never scheduled again,
    # daemon so it cannot block interpreter shutdown), and genuine hangs are
    # rare per call (see _HANDLE_QUERY_TIMEOUT_SEC's docstring). A hard cap
    # / bounded thread pool would add real complexity for a failure mode
    # that has not been observed to matter in practice; revisit only if a
    # host process is seen accumulating a large number of these threads
    # (e.g. very many teardowns against systems with many
    # persistently-hanging handles, such as many unconnected named pipes).
    def _make_worker():
        job_queue: "queue.Queue" = queue.Queue()

        def _run() -> None:
            while True:
                item = job_queue.get()
                if item is None:
                    return
                dup_handle, info_class, outcome, done = item
                try:
                    outcome["value"] = _query_object_raw(dup_handle, info_class)
                except Exception:  # noqa: BLE001 -- best-effort, never propagate
                    outcome["value"] = None
                done.set()

        worker_thread = threading.Thread(target=_run, daemon=True)
        worker_thread.start()
        return job_queue

    worker_queue = _make_worker()

    def _query_object_bounded(
        dup_handle, info_class: int, timeout: float = _HANDLE_QUERY_TIMEOUT_SEC
    ) -> Optional[str]:
        nonlocal worker_queue
        outcome: dict = {}
        done = threading.Event()
        worker_queue.put((dup_handle, info_class, outcome, done))
        if done.wait(timeout):
            return outcome.get("value")
        # The worker is now permanently wedged inside NtQueryObject for this
        # handle -- replace it so subsequent queries are not blocked.
        worker_queue = _make_worker()
        return None

    found: List[Tuple[int, str]] = []
    current_process = kernel32.GetCurrentProcess()
    # Per-call cache: object-type-index -> resolved type name (or None if
    # unresolved). Populated lazily from the first handle seen for each type
    # index; every later handle of that type index is accepted/skipped from
    # this cache alone, with no further DuplicateHandle/NtQueryObject call.
    _UNSET = object()
    type_name_cache: Dict[int, Optional[str]] = {}

    for pid, handle_entries in by_pid.items():
        if time.monotonic() > scan_deadline:
            break
        proc_handle = kernel32.OpenProcess(PROCESS_DUP_HANDLE, False, pid)
        if not proc_handle:
            # No PROCESS_DUP_HANDLE access -- elevated/other-user process we
            # cannot cross without running elevated ourselves.
            continue
        try:
            for handle_value, type_index in handle_entries:
                if time.monotonic() > scan_deadline:
                    break

                cached_type = type_name_cache.get(type_index, _UNSET)
                if cached_type is not _UNSET and cached_type != "File":
                    continue  # known non-file type -- skip without duplicating

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
                    continue
                try:
                    if cached_type is _UNSET:
                        type_name = _query_object_bounded(dup_handle, OBJECT_TYPE_INFORMATION)
                        type_name_cache[type_index] = type_name
                        if type_name != "File":
                            continue
                    nt_name = _query_object_bounded(dup_handle, OBJECT_NAME_INFORMATION)
                finally:
                    kernel32.CloseHandle(dup_handle)
                if not nt_name:
                    continue
                dos_path = _nt_path_to_dos(nt_name)
                if not dos_path:
                    continue
                norm = os.path.normcase(os.path.normpath(dos_path))
                if norm == normalized or norm.startswith(normalized + os.sep):
                    found.append((pid, ""))
                    break  # one matching handle is enough to flag this pid
        finally:
            kernel32.CloseHandle(proc_handle)

    return found


def _find_blocking_processes(
    path: str,
    host_pid: int,
    *,
    deadline: Optional[float] = None,
) -> List[KilledProcessInfo]:
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

    # Pass 1: CWD match.
    if time.monotonic() <= scan_stop:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if time.monotonic() > scan_stop:
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

    # Pass 1b (Windows only): cmdline token scan.
    # proc.cwd() raises AccessDenied for almost all foreign processes on Windows.
    # Scan cmdline tokens as a fallback: if any token resolves to a path under
    # the worktree directory, treat the process as blocking.
    if sys.platform == "win32" and time.monotonic() <= scan_stop:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if time.monotonic() > scan_stop:
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
    if sys.platform == "win32" and time.monotonic() <= scan_stop:
        if deadline is None:
            handle_scan_budget = _HANDLE_SCAN_BUDGET_SEC
        else:
            handle_scan_budget = min(
                _HANDLE_SCAN_BUDGET_SEC, max(0.0, deadline - time.monotonic())
            )
        if handle_scan_budget <= 0:
            handle_holders: List[Tuple[int, str]] = []
        else:
            try:
                handle_holders = _win_handle_holders(
                    path, excluded_pids, budget_sec=handle_scan_budget
                )
            except Exception:  # noqa: BLE001 -- best-effort, never propagate
                handle_holders = []
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

    # Pass 2: open file handles — catches daemons that have changed their cwd
    # away from the worktree but still hold file locks inside it.
    if time.monotonic() <= scan_stop:
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if time.monotonic() > scan_stop:
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

    return result


def _kill_blocking_processes(
    path: str,
    *,
    timeout: float = 5.0,
) -> List[KilledProcessInfo]:
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
    1/1b/1c/2 discovery loops.
    """
    deadline = time.monotonic() + timeout
    found = _find_blocking_processes(path, os.getpid(), deadline=deadline)
    if not found:
        return found

    seen_pids = {info.pid for info in found}
    expanded: List[KilledProcessInfo] = list(found)
    for info in found:
        if time.monotonic() >= deadline:
            # Budget exhausted -- stop issuing further _process_tree() calls;
            # whatever lineage was expanded so far is kept, not discarded.
            break
        for descendant in _process_tree(info.pid):
            if descendant.pid in seen_pids:
                continue
            seen_pids.add(descendant.pid)
            expanded.append(descendant)
    found = expanded

    n = len(found)
    for info in found:
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
    return found


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
    operation is always bounded by *timeout* seconds.

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

    if _pid_alive(pid):
        _send_graceful_signal(pid)
        _wait_or_kill(pid, max(0.0, deadline - time.monotonic()))

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
    killed_tree = tree + group_only_infos
    _kill_process_tree(killed_tree, timeout=max(0.0, deadline - time.monotonic()))

    # Orphan scan: kill processes that survived because they detached far
    # enough (e.g. into their own session/process group) to evade both the
    # tree snapshot above and the primary signal. Run this whether the
    # tracked PID was alive or dead -- it's a no-op when there are no
    # orphans. Pass remaining budget so the scan is also bounded.
    orphan_found: List[KilledProcessInfo] = []
    if kill_orphans:
        orphan_budget = max(0.0, deadline - time.monotonic())
        orphan_found = _kill_blocking_processes(record.path, timeout=orphan_budget)

    # Never report a false "stopped" (ticket #87): re-probe liveness of
    # everything this call attempted to kill. Any survivor means the
    # environment is not actually torn down, even though the tracked pid
    # entry is about to be cleared below.
    candidate_pids = [pid] + [info.pid for info in killed_tree] + [info.pid for info in orphan_found]
    survivor_pids = [p for p in candidate_pids if _pid_alive(p)]

    # Clear the role regardless of whether the process was alive — the
    # important postcondition is that the record no longer references it.
    del record.pids[role]

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
