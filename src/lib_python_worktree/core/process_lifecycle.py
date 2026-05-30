"""Process lifecycle engine layer (W6/W8 — ticket #8).

Public API
----------
- ``start(worktree_id, cmd, *, store, role="main", env=None, cwd=None)``
  Spawns a detached process, persists ``pids[role]`` and ``status="running"``
  to the state store, returns the updated ``WorktreeRecord``.

- ``stop(worktree_id, *, store, role="main", timeout=10.0)``
  Gracefully terminates the process (SIGTERM/CTRL_BREAK), waits up to
  ``timeout`` seconds, then force-kills if still alive.  Clears
  ``pids[role]``; sets ``status="stopped"`` only when no other roles
  remain.  Returns the updated ``WorktreeRecord``.

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

import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

from .state import StateStore, WorktreeRecord
from .yaml_store import _pid_alive

# The role key used when the caller does not supply an explicit role.
DEFAULT_ROLE = "main"


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
) -> int:
    """Spawn *cmd* as a fully detached process and return its PID.

    The child process does not inherit standard streams and is detached from
    the caller's process group so it survives if the MCP host exits.
    """
    if not cmd:
        raise ValueError("cmd must be a non-empty list")

    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
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

    proc = subprocess.Popen(cmd, **kwargs)
    return proc.pid


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


def _wait_or_kill(pid: int, timeout: float) -> None:
    """Wait up to *timeout* seconds for *pid* to die; force-kill if it doesn't.

    Uses a polling loop (0.1 s sleep) so there is no hard dependency on
    psutil or OS-specific wait APIs.  ``timeout <= 0`` goes straight to
    force-kill.
    """
    if timeout <= 0:
        _force_kill(pid)
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)

    # Still alive after the timeout — escalate to force-kill.
    if _pid_alive(pid):
        _force_kill(pid)


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

    pid = _spawn_detached(cmd, env=env, cwd=cwd)

    record.pids[role] = pid
    record.status = "running"
    store.update(record)

    return record


def stop(
    worktree_id: str,
    *,
    store: StateStore,
    role: str = DEFAULT_ROLE,
    timeout: float = 10.0,
) -> WorktreeRecord:
    """Stop the process recorded under *role* for *worktree_id*.

    Sends a graceful signal, waits up to *timeout* seconds, then force-kills
    if the process is still alive.  Clears ``pids[role]``; sets
    ``status="stopped"`` only when no other roles remain in ``pids``.

    If the PID is already dead, clears the record gracefully without raising.

    Parameters
    ----------
    worktree_id:
        The ID of the worktree record in *store*.
    store:
        The active ``StateStore`` instance.
    role:
        Identifies the process within the worktree.
    timeout:
        Seconds to wait for graceful exit before force-killing.

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

    if _pid_alive(pid):
        _send_graceful_signal(pid)
        _wait_or_kill(pid, timeout)

    # Clear the role regardless of whether the process was alive — the
    # important postcondition is that the record no longer references it.
    del record.pids[role]
    # Only mark the whole worktree as "stopped" when all roles are gone.
    # In a multi-role worktree, stopping one role must not mask the fact
    # that other processes are still alive.
    if not record.pids:
        record.status = "stopped"
    store.update(record)

    return record


__all__ = (
    "DEFAULT_ROLE",
    "ProcessAlreadyRunningError",
    "ProcessLifecycleError",
    "ProcessNotRunningError",
    "start",
    "stop",
)
