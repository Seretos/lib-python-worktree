"""File-backed state store persisting worktree records to YAML files.

W7 implementation of the ``StateStore`` protocol.  Two files live under
``~/.agent-worktree/`` (or a caller-specified directory):

* ``state.yaml``  — worktree records, schema version 1.
* ``ports.yaml``  — allocated port assignments, schema version 1.

Every read-modify-write cycle acquires an exclusive ``portalocker.Lock`` on
the target file with a 10-second timeout so that two MCP instances cannot
corrupt each other's data.  Writes go to a temp file first; ``os.replace()``
makes the swap atomic (avoids partial-write corruption).

The module also provides a ``reconcile()`` function that detects and corrects
stale state at startup:

* Worktrees whose on-disk path has vanished → status set to ``"orphaned"``.
* PIDs that are no longer alive → removed from ``pids``, status ``"stopped"``.
* Ports that are not in use and have no surviving PID → removed from
  ``ports.yaml``.

Each inconsistency is logged at WARNING level.
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import portalocker
import yaml

from .state import WorktreeRecord

_STATE_FILE = "state.yaml"
_PORTS_FILE = "ports.yaml"
_STATE_SCHEMA_VERSION = 1
_PORTS_SCHEMA_VERSION = 1
_LOCK_TIMEOUT = 10.0  # seconds
# EXCLUSIVE|NON_BLOCKING lets portalocker poll with its own retry loop and
# honour the timeout argument.  Pure LOCK_EX (blocking) makes the OS block
# indefinitely and portalocker cannot interrupt it for the timeout.
_LOCK_FLAGS = portalocker.LOCK_EX | portalocker.LOCK_NB

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portable PID liveness check
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Return True if the given PID corresponds to a running process.

    Works on both POSIX and Windows without third-party dependencies.
    On POSIX:  ``os.kill(pid, 0)`` succeeds → alive; EPERM → alive (no perms);
               ESRCH → dead.
    On Windows: uses ``OpenProcess`` via ctypes; ``PROCESS_QUERY_INFORMATION``
               access right is enough to tell whether the handle opened.
               An exit-code of STILL_ACTIVE (259) means the process is alive.
    """
    if pid <= 0:
        return False

    if sys.platform == "win32":
        return _pid_alive_windows(pid)

    # POSIX path
    import errno
    try:
        os.kill(pid, 0)
        return True
    except OSError as exc:
        import errno as _errno
        if exc.errno == _errno.EPERM:
            # We lack permission to signal it, but it exists.
            return True
        # ESRCH (no such process) or anything else → dead
        return False


def _pid_alive_windows(pid: int) -> bool:
    """Windows-specific PID liveness check via ctypes."""
    import ctypes
    import ctypes.wintypes

    PROCESS_QUERY_INFORMATION = 0x0400
    STILL_ACTIVE = 259

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.wintypes.DWORD()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return exit_code.value == STILL_ACTIVE
        return False
    finally:
        kernel32.CloseHandle(handle)


# ---------------------------------------------------------------------------
# Internal helpers: serialise / deserialise WorktreeRecord
# ---------------------------------------------------------------------------

def _record_to_dict(rec: WorktreeRecord) -> Dict[str, Any]:
    return {
        "id": rec.id,
        "repo_root": rec.repo_root,
        "branch": rec.branch,
        "path": rec.path,
        "status": rec.status,
        "ports": dict(rec.ports),
        "pids": dict(rec.pids),
        "branch_created_by_us": rec.branch_created_by_us,
    }


def _record_from_dict(d: Dict[str, Any]) -> WorktreeRecord:
    return WorktreeRecord(
        id=d["id"],
        repo_root=d["repo_root"],
        branch=d["branch"],
        path=d["path"],
        status=d.get("status", "created"),
        ports=dict(d.get("ports") or {}),
        pids=dict(d.get("pids") or {}),
        branch_created_by_us=bool(d.get("branch_created_by_us", False)),
    )


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------

def _atomic_write_yaml(path: Path, data: Any) -> None:
    """Write ``data`` as YAML to ``path`` atomically (temp file + os.replace).

    The temp file is created in the same directory as ``path`` so that
    ``os.replace`` is guaranteed to be on the same filesystem (a requirement
    on some platforms).
    """
    dir_ = path.parent
    dir_.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile with delete=False so we can replace after closing.
    fd, tmp = tempfile.mkstemp(dir=str(dir_), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, allow_unicode=True, default_flow_style=False)
        os.replace(tmp, str(path))
    except Exception:
        # Best-effort cleanup of the temp file so we don't litter.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# _PortsFile: locked access to ports.yaml
# ---------------------------------------------------------------------------

class _PortsFile:
    """Provides locked read/write access to ``ports.yaml``."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _load(self) -> Dict[str, int]:
        if not self._path.exists():
            return {}
        with open(self._path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return dict(data.get("allocated") or {})

    def _save(self, allocated: Dict[str, int]) -> None:
        _atomic_write_yaml(
            self._path,
            {"version": _PORTS_SCHEMA_VERSION, "allocated": dict(allocated)},
        )

    def get_all(self) -> Dict[str, int]:
        """Return all allocated ports (name → port) under a lock."""
        with portalocker.Lock(
            str(self._path) + ".lock",
            timeout=_LOCK_TIMEOUT,
            flags=_LOCK_FLAGS,
        ):
            return self._load()

    def set_all(self, allocated: Dict[str, int]) -> None:
        """Replace the entire ports allocation under a lock."""
        with portalocker.Lock(
            str(self._path) + ".lock",
            timeout=_LOCK_TIMEOUT,
            flags=_LOCK_FLAGS,
        ):
            self._save(allocated)

    def ensure_file(self) -> None:
        """Create the file with an empty schema if it doesn't exist yet."""
        with portalocker.Lock(
            str(self._path) + ".lock",
            timeout=_LOCK_TIMEOUT,
            flags=_LOCK_FLAGS,
        ):
            if not self._path.exists():
                self._save({})


# ---------------------------------------------------------------------------
# ReconcileReport
# ---------------------------------------------------------------------------

@dataclass
class ReconcileReport:
    """Result of a ``reconcile()`` pass."""

    orphaned: List[str] = field(default_factory=list)
    stopped: List[str] = field(default_factory=list)
    freed_ports: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YamlStateStore
# ---------------------------------------------------------------------------

class YamlStateStore:
    """File-backed ``StateStore`` persisting to YAML files under ``state_dir``.

    The default ``state_dir`` is ``~/.agent-worktree/``.  Pass an explicit
    path in tests so that no test touches the real home directory.
    """

    def __init__(self, state_dir: Optional[Path] = None) -> None:
        if state_dir is None:
            state_dir = Path.home() / ".agent-worktree"
        self._state_dir = state_dir
        self._state_path = state_dir / _STATE_FILE
        self._ports_path = state_dir / _PORTS_FILE
        self._ports = _PortsFile(self._ports_path)
        # Ensure the directory exists; individual files are created lazily.
        state_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal locked helpers for state.yaml
    # ------------------------------------------------------------------

    def _load_state(self) -> Dict[str, WorktreeRecord]:
        """Load and return records dict from state.yaml (caller holds lock)."""
        if not self._state_path.exists():
            return {}
        with open(self._state_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        raw_worktrees = data.get("worktrees") or {}
        return {k: _record_from_dict(v) for k, v in raw_worktrees.items()}

    def _save_state(self, records: Dict[str, WorktreeRecord]) -> None:
        """Persist records dict to state.yaml atomically (caller holds lock)."""
        _atomic_write_yaml(
            self._state_path,
            {
                "version": _STATE_SCHEMA_VERSION,
                "worktrees": {k: _record_to_dict(v) for k, v in records.items()},
            },
        )

    def _with_state_lock(self):
        """Context manager: exclusive lock on state.yaml's lock file."""
        return portalocker.Lock(
            str(self._state_path) + ".lock",
            timeout=_LOCK_TIMEOUT,
            flags=_LOCK_FLAGS,
        )

    # ------------------------------------------------------------------
    # StateStore Protocol
    # ------------------------------------------------------------------

    def add(self, record: WorktreeRecord) -> None:
        with self._with_state_lock():
            records = self._load_state()
            if record.id in records:
                raise ValueError(f"Worktree id already tracked: {record.id}")
            records[record.id] = record
            self._save_state(records)

    def get(self, worktree_id: str) -> Optional[WorktreeRecord]:
        with self._with_state_lock():
            records = self._load_state()
        return records.get(worktree_id)

    def remove(self, worktree_id: str) -> Optional[WorktreeRecord]:
        with self._with_state_lock():
            records = self._load_state()
            rec = records.pop(worktree_id, None)
            if rec is not None:
                self._save_state(records)
        return rec

    def list(self) -> List[WorktreeRecord]:
        with self._with_state_lock():
            records = self._load_state()
        return list(records.values())

    def find_by_branch(
        self, repo_root: str, branch: str
    ) -> Optional[WorktreeRecord]:
        with self._with_state_lock():
            records = self._load_state()
        for rec in records.values():
            if rec.repo_root == repo_root and rec.branch == branch:
                return rec
        return None

    def update(self, record: WorktreeRecord) -> None:
        with self._with_state_lock():
            records = self._load_state()
            if record.id not in records:
                raise KeyError(f"Worktree id not tracked: {record.id}")
            records[record.id] = record
            self._save_state(records)

    @property
    def state_dir(self) -> Path:
        return self._state_dir


# ---------------------------------------------------------------------------
# reconcile()
# ---------------------------------------------------------------------------

def reconcile(
    store: YamlStateStore,
    *,
    logger: Optional[logging.Logger] = None,
) -> ReconcileReport:
    """Scan persisted state and correct stale entries.

    Detects:
    * Worktrees whose on-disk path no longer exists → status ``"orphaned"``.
    * PIDs that are no longer alive → removed; status ``"stopped"`` (unless
      already ``"orphaned"``).
    * Port allocations whose port is not in use and has no surviving PID →
      removed from ports.yaml.

    Every inconsistency is logged at WARNING level via ``logger`` (or the
    module-level logger if not supplied).

    Returns a ``ReconcileReport`` describing what was changed.
    """
    _log = logger if logger is not None else logging.getLogger(__name__)
    report = ReconcileReport()

    # --- Phase 1: state.yaml ---
    with store._with_state_lock():
        records = store._load_state()
        changed = False
        for wt_id, rec in records.items():
            if not Path(rec.path).exists():
                if rec.status != "orphaned":
                    _log.warning(
                        "reconcile: worktree '%s' path not found (%s) → orphaned",
                        wt_id, rec.path,
                    )
                    rec.status = "orphaned"
                    changed = True
                report.orphaned.append(wt_id)

            dead_roles = [
                role for role, pid in rec.pids.items()
                if not _pid_alive(pid)
            ]
            for role in dead_roles:
                pid = rec.pids.pop(role)
                _log.warning(
                    "reconcile: worktree '%s' PID %d (role '%s') is dead → removed",
                    wt_id, pid, role,
                )
                changed = True
                if rec.status not in ("orphaned",):
                    rec.status = "stopped"
                if wt_id not in report.stopped:
                    report.stopped.append(wt_id)

        if changed:
            store._save_state(records)

    # --- Phase 2: ports.yaml ---
    # Build the set of all surviving PIDs (from the freshly-reconciled records)
    surviving_pids: set[int] = set()
    for rec in records.values():
        surviving_pids.update(rec.pids.values())

    ports_lock = portalocker.Lock(
        str(store._ports_path) + ".lock",
        timeout=_LOCK_TIMEOUT,
        flags=_LOCK_FLAGS,
    )
    with ports_lock:
        allocated = store._ports._load()
        to_free: List[str] = []
        for name, port in list(allocated.items()):
            in_use = _port_in_use(port)
            if not in_use and not surviving_pids:
                # Port not listening and no surviving PIDs remain in any
                # worktree record — safe to release.  When a live process
                # is still tracked (surviving_pids is non-empty) we keep
                # the allocation because the process may not have bound
                # yet (race between startup and reconcile).
                to_free.append(name)
        for name in to_free:
            port = allocated.pop(name)
            _log.warning(
                "reconcile: port allocation '%s'=%d not in use → freed",
                name, port,
            )
            report.freed_ports.append(name)
        if to_free:
            store._ports._save(allocated)

    return report


def _port_in_use(port: int) -> bool:
    """Return True if something is listening on 127.0.0.1:<port>."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


__all__ = [
    "ReconcileReport",
    "YamlStateStore",
    "reconcile",
]
