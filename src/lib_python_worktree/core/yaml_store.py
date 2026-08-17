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
import re
import socket
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import portalocker
import yaml

from ._git_utils import _run_git
from .state import StopDetail, WorktreeRecord, _STOP_DETAIL_MAX_PIDS

_STATE_FILE = "state.yaml"
_PORTS_FILE = "ports.yaml"
_STATE_SCHEMA_VERSION = 1
_PORTS_SCHEMA_VERSION = 1
_LOCK_TIMEOUT = 10.0  # seconds
# Canonical separator between a ports.yaml key's owner id and its slot name
# (e.g. "wt-a:web"). Owned by this module; port_allocator.py aliases it as
# _KEY_SEP so the import direction (port_allocator -> yaml_store) stays intact.
_PORT_KEY_SEP = ":"
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


def _kernel32_lasterror():
    """Return ``kernel32`` bound with ``use_last_error=True`` (ticket #95).

    ``ctypes.windll.kernel32`` cannot carry ``use_last_error`` -- it is a
    fixed, process-wide singleton binding. A dedicated ``ctypes.WinDLL(...,
    use_last_error=True)`` call is needed so ``ctypes.get_last_error()``
    reliably reflects the outcome of the most recent call made through the
    handle this function returns. Factored into its own function (rather
    than inlined) purely to give tests a patch seam that works even when not
    running on Windows.
    """
    import ctypes

    return ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]


def _pid_alive_windows(pid: int) -> bool:
    """Windows-specific PID liveness check via ctypes (hardened -- ticket #95).

    Tries ``PROCESS_QUERY_LIMITED_INFORMATION`` (0x1000, introduced in
    Vista -- requires fewer privileges) first, then falls back to the
    classic ``PROCESS_QUERY_INFORMATION`` (0x0400) if that fails.

    Finding 4 (ticket #95): the previous implementation treated *any*
    ``OpenProcess`` failure as "dead". But ``OpenProcess`` failing with
    ``ERROR_ACCESS_DENIED`` actually proves the PID still exists -- the OS
    returns ``ERROR_INVALID_PARAMETER`` for a PID that is genuinely gone.
    A live process we cannot open (e.g. spawned by another user or an
    elevated context) therefore used to read as already dead, silently
    masking a real survivor from callers such as ``stop()``'s post-kill
    re-probe. When both ``OpenProcess`` attempts fail, ``GetLastError()`` is
    now classified:

    - ``ERROR_ACCESS_DENIED`` (5)      -> ``True``  (the PID exists)
    - ``ERROR_INVALID_PARAMETER`` (87) -> ``False`` (no such process)
    - anything else                    -> ``False``, logged at debug level

    Note: classifying ACCESS_DENIED as "alive" makes a PID-reuse false
    positive marginally more likely (an unrelated, inaccessible process
    happens to reuse the tracked PID number) -- consistent with the
    PID-reuse limitation already documented as out of scope on ``stop()``.
    """
    import ctypes
    import ctypes.wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_QUERY_INFORMATION = 0x0400
    STILL_ACTIVE = 259
    ERROR_ACCESS_DENIED = 5
    ERROR_INVALID_PARAMETER = 87

    kernel32 = _kernel32_lasterror()

    ctypes.set_last_error(0)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        first_err = ctypes.get_last_error()
        if first_err == ERROR_ACCESS_DENIED:
            # The less-privileged query already proves the PID exists --
            # PROCESS_QUERY_INFORMATION requires strictly MORE privilege, so
            # retrying with it can only fail the same way. Short-circuit
            # rather than making a second, guaranteed-redundant syscall.
            return True
        ctypes.set_last_error(0)
        handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)

    if not handle:
        err = ctypes.get_last_error()
        if err == ERROR_ACCESS_DENIED:
            return True
        if err == ERROR_INVALID_PARAMETER:
            return False
        logger.debug(
            "_pid_alive_windows(pid=%s): OpenProcess failed with unrecognized "
            "GetLastError()=%s -- treating as dead",
            pid, err,
        )
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

def _stop_detail_to_dict(detail: Optional[StopDetail]) -> Optional[Dict[str, Any]]:
    """Serialise a :class:`~.state.StopDetail` to a plain nested dict of
    scalars/lists (ticket #99), or ``None`` when *detail* is ``None``.

    ``survivor_pids`` is written as a plain ``list`` (YAML has no tuple type);
    re-capped defensively at :data:`_STOP_DETAIL_MAX_PIDS` even though the
    dataclass is already constructed within that cap, so a hand-edited or
    future-version state.yaml can never blow past it on the way back in via
    :func:`_stop_detail_from_dict`.
    """
    if detail is None:
        return None
    return {
        "reason": detail.reason,
        "message": detail.message,
        "role": detail.role,
        "survivor_pids": list(detail.survivor_pids)[:_STOP_DETAIL_MAX_PIDS],
        "survivor_count": detail.survivor_count,
        "truncated_at": detail.truncated_at,
        "skipped_passes": list(detail.skipped_passes),
        "kill_orphans_may_help": detail.kill_orphans_may_help,
    }


def _stop_detail_from_dict(d: Optional[Dict[str, Any]]) -> Optional[StopDetail]:
    """Reconstruct a :class:`~.state.StopDetail` from its serialised dict, or
    ``None`` when *d* is ``None``/absent/empty (ticket #99).

    Field-by-field with defaults for missing keys -- deliberately NOT
    ``StopDetail(**d)`` -- so a legacy record with no ``stop_detail`` key at
    all, or a state.yaml written by a future engine version carrying extra
    keys this version does not know about, both deserialise without raising:
    unknown keys are silently ignored, missing keys fall back to the same
    defaults :class:`~.state.StopDetail` itself uses. ``reason`` is preserved
    verbatim even if it is not a value in ``state.STOP_REASONS`` -- forward
    compatibility with a reason vocabulary this version predates.
    """
    if not d:
        return None
    return StopDetail(
        reason=d.get("reason", ""),
        message=d.get("message", ""),
        role=d.get("role"),
        survivor_pids=tuple(d.get("survivor_pids") or ())[:_STOP_DETAIL_MAX_PIDS],
        survivor_count=d.get("survivor_count", 0),
        truncated_at=d.get("truncated_at"),
        skipped_passes=tuple(d.get("skipped_passes") or ()),
        kill_orphans_may_help=bool(d.get("kill_orphans_may_help", False)),
    )


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
        "returncode": rec.returncode,
        "start_log_path": rec.start_log_path,
        "backing": rec.backing,
        "job_names": dict(rec.job_names),
        "stop_detail": _stop_detail_to_dict(rec.stop_detail),
    }


def _record_from_dict(d: Dict[str, Any]) -> WorktreeRecord:
    # Normalize repo_root and path to forward slashes on load so that records
    # written by a pre-fix version of the engine (which used str(Path(...)) and
    # therefore produced OS-native backslashes on Windows) are self-healing the
    # first time they are read back.  str.replace is used rather than
    # Path(...).as_posix() because on Linux/macOS PosixPath treats backslash as
    # an ordinary filename character and would leave legacy backslash strings
    # unchanged.  A plain replace("\\", "/") is platform-independent and is a
    # no-op for strings that are already forward-slash.
    return WorktreeRecord(
        id=d["id"],
        repo_root=d["repo_root"].replace("\\", "/"),
        branch=d.get("branch"),
        path=d["path"].replace("\\", "/"),
        status=d.get("status", "created"),
        ports=dict(d.get("ports") or {}),
        pids=dict(d.get("pids") or {}),
        branch_created_by_us=bool(d.get("branch_created_by_us", False)),
        returncode=d.get("returncode"),
        start_log_path=d.get("start_log_path"),
        backing=d.get("backing", "worktree"),
        job_names=dict(d.get("job_names") or {}),
        stop_detail=_stop_detail_from_dict(d.get("stop_detail")),
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
# AdoptReport
# ---------------------------------------------------------------------------

@dataclass
class AdoptReport:
    """Result of an ``adopt()`` pass."""

    adopted: List[str] = field(default_factory=list)
    # Count of worktree blocks that had no branch (detached HEAD, corrupt blocks,
    # etc.) and were therefore skipped rather than imported.
    skipped_detached: int = 0
    # Count of worktree blocks marked ``prunable`` by git (their on-disk directory
    # is gone) and therefore skipped — they do not "exist on-disk".
    skipped_prunable: int = 0


# ---------------------------------------------------------------------------
# Private helpers for adopt() — avoid circular import from manager.py
# ---------------------------------------------------------------------------

_SLUG_RE_YS = re.compile(r"[^a-z0-9]+")


def _slug_ys(value: str, *, max_len: int = 40) -> str:
    """Lower-case ASCII slug, duplicated from manager.py to avoid circular import."""
    s = _SLUG_RE_YS.sub("-", value.lower()).strip("-")
    if not s:
        s = "x"
    return s[:max_len]


def _short_uuid_ys() -> str:
    return uuid.uuid4().hex[:8]


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

    def _add_unlocked(
        self, records: Dict[str, WorktreeRecord], record: WorktreeRecord
    ) -> None:
        """Add ``record`` to an already-loaded ``records`` dict without acquiring
        the lock.  The caller MUST hold ``_with_state_lock()`` and call
        ``_save_state(records)`` after all mutations are complete.

        Raises ``ValueError`` (same as the public ``add()``) if the id is
        already present.
        """
        if record.id in records:
            raise ValueError(f"Worktree id already tracked: {record.id}")
        records[record.id] = record

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
            if rec.backing == "primary":
                # A primary record's branch is never stored (read live via
                # _effective_branch) and must never shadow a create()
                # duplicate-branch check.
                continue
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
                    # Ticket #99: "orphaned" supersedes "stop_incomplete"
                    # (the path itself is gone, so no earlier "process may
                    # still be alive" reason remains actionable) -- clear any
                    # stale stop_detail in lockstep with the status change.
                    rec.stop_detail = None
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
                # Ticket #87 follow-up (finding B2): "stop_incomplete" is a
                # sticky, deliberately-honest status set by process_lifecycle
                # .stop() when it could not confirm everything it tried to
                # kill actually died. A dead role discovered here later must
                # not silently overwrite that guarantee back to "stopped" --
                # only "orphaned" (path gone) is allowed to already outrank
                # the default "stopped" outcome of this branch.
                if rec.status not in ("orphaned", "stop_incomplete"):
                    rec.status = "stopped"
                    # Ticket #99: this branch only runs when status is
                    # actually transitioning to "stopped" (the guard above
                    # already excludes the sticky "stop_incomplete" case) --
                    # clear stop_detail in lockstep, mirroring
                    # process_lifecycle.stop()'s own clean-"stopped" branch.
                    rec.stop_detail = None
                if wt_id not in report.stopped:
                    report.stopped.append(wt_id)

        if changed:
            store._save_state(records)

    # --- Phase 2: ports.yaml ---
    # Free a port allocation only when its *owning record* no longer exists in
    # state.yaml — not based on whether the port is currently listening or
    # whether any PID happens to still be alive somewhere. Each ports.yaml key
    # is `<owner_id><_PORT_KEY_SEP><slot>`; a legacy/hand-written key with no
    # separator is treated as its own owner id (unprefixed keys have no
    # matching record and are therefore always freed). This fixes a bug where
    # a stopped-but-still-tracked environment (no live PIDs) would have its
    # port reservation wiped by reconcile() even though the record — and
    # therefore the reservation's rightful owner — still exists.
    surviving_ids: set[str] = set(records.keys())

    ports_lock = portalocker.Lock(
        str(store._ports_path) + ".lock",
        timeout=_LOCK_TIMEOUT,
        flags=_LOCK_FLAGS,
    )
    with ports_lock:
        allocated = store._ports._load()
        to_free: List[str] = []
        for name, _port in list(allocated.items()):
            owner_id = name.split(_PORT_KEY_SEP, 1)[0]
            if owner_id not in surviving_ids:
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


# ---------------------------------------------------------------------------
# adopt()
# ---------------------------------------------------------------------------

def adopt(
    store: YamlStateStore,
    repo_path: Path,
    *,
    logger: Optional[logging.Logger] = None,
) -> AdoptReport:
    """Discover git worktrees that exist on-disk but are not tracked in ``store``.

    Calls ``git worktree list --porcelain`` against ``repo_path`` and imports
    any unknown worktrees as ``WorktreeRecord`` entries with
    ``status="adopted"`` and ``branch_created_by_us=False``.

    ``repo_path`` MUST be the exact git top-level directory (as returned by
    ``git rev-parse --show-toplevel``).  Passing a sub-directory will cause
    the ``(repo_root, branch)`` idempotency check to diverge from records
    written by ``WorktreeManager.create()``.  ``WorktreeManager.adopt()``
    guarantees this by running ``_validate_repo()`` first.

    Silently skips:
    - Any block whose resolved path equals the PRIMARY checkout (always
      ``blocks[0]`` in porcelain output, regardless of cwd) OR ``repo_path``
      itself.  Both are skipped because: (a) the primary checkout is the repo
      and must never be adopted as a managed worktree; (b) if ``repo_path`` is
      a linked worktree, its own block must also be excluded.
    - Prunable blocks (on-disk directory deleted; use ``prune()`` instead).
    - Blocks with no ``branch`` line (detached HEAD, corrupt entries, etc.).
    - Worktrees already in the store (same path or same ``(repo_root, branch)``
      pair — idempotent, checked atomically under the store lock).

    Atomicity: the entire read-state → conflict-check → write-new-records
    cycle runs under a single ``_with_state_lock()`` acquisition, so two
    concurrent ``adopt()`` calls for the same repo cannot produce duplicate
    records.

    Best-effort: any git failure returns an empty ``AdoptReport``; does not raise.
    """
    _log = logger if logger is not None else logging.getLogger(__name__)
    report = AdoptReport()

    # --- Phase 1: run git (outside the store lock — no I/O under the lock) ---
    try:
        proc = _run_git(["worktree", "list", "--porcelain"], cwd=repo_path)
    except Exception as exc:  # noqa: BLE001
        _log.debug("adopt: git worktree list failed with exception: %s", exc)
        return report

    if proc.returncode != 0:
        _log.debug(
            "adopt: git worktree list returned %d: %s",
            proc.returncode, (proc.stderr or "").strip(),
        )
        return report

    main_path = repo_path.resolve()
    repo_root_str = main_path.as_posix()

    # --- Phase 2: parse porcelain output (CPU only, no I/O) ---
    # Format (one block per worktree, blank line between blocks):
    #   worktree /path/to/wt
    #   HEAD <sha>
    #   branch refs/heads/<name>   -- OR --
    #   detached
    #
    # The first block is always the main worktree.

    blocks: List[Dict[str, Optional[str]]] = []
    current: Dict[str, Optional[str]] = {}

    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.rstrip()
        if not line:
            if current:
                blocks.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = line[len("worktree "):].strip()
        elif line.startswith("branch "):
            # "branch refs/heads/<name>"
            ref = line[len("branch "):].strip()
            prefix = "refs/heads/"
            if ref.startswith(prefix):
                current["branch"] = ref[len(prefix):]
            else:
                current["branch"] = ref
        elif line == "detached":
            current["detached"] = "true"
        elif line.startswith("prunable"):
            current["prunable"] = "true"

    # Flush the last block (no trailing blank line at EOF in some git versions).
    if current:
        blocks.append(current)

    if not blocks:
        return report

    # git worktree list --porcelain ALWAYS emits the primary checkout as
    # blocks[0], regardless of which worktree's cwd the command ran from.
    # Resolve it here so we can skip it unconditionally in the loop below.
    primary_path = Path(blocks[0].get("path") or "").resolve()

    # Pre-classify blocks.
    # Skip any block whose resolved path equals EITHER:
    #   • primary_path (blocks[0]) — the actual repo directory; must never be
    #     adopted as a managed worktree because remove(force=True) would
    #     shutil.rmtree it.
    #   • main_path (repo_path.resolve()) — if adopt() was called from a linked
    #     worktree, repo_path differs from primary_path and appears somewhere in
    #     blocks[1:]; we must not adopt the worktree we were invoked from.
    skip_paths = {primary_path, main_path}

    candidates: List[tuple[str, str]] = []   # (wt_path_str, branch)
    for block in blocks:
        wt_path_raw = block.get("path")
        if not wt_path_raw:
            continue

        # Skip the primary checkout and the repo we were called with.
        if Path(wt_path_raw).resolve() in skip_paths:
            continue

        # Skip prunable blocks — the on-disk directory is gone; these worktrees
        # do not "exist on-disk" and should be removed with prune(), not adopted.
        if block.get("prunable"):
            _log.debug(
                "adopt: skipping prunable worktree at %s", wt_path_raw
            )
            report.skipped_prunable += 1
            continue

        if block.get("detached") or not block.get("branch"):
            _log.debug(
                "adopt: skipping branchless worktree at %s", wt_path_raw
            )
            report.skipped_detached += 1
            continue

        candidates.append(
            (Path(wt_path_raw).resolve().as_posix(), block["branch"])
        )

    if not candidates:
        return report

    # --- Phase 3: single-lock critical section — snapshot, check, write ---
    # All conflict detection AND the appended records are committed in one
    # atomic load → mutate → save cycle so two concurrent adopt() calls on
    # the same repo cannot both see a worktree as untracked and each add it.
    repo_slug = _slug_ys(main_path.name)
    with store._with_state_lock():
        records = store._load_state()
        existing_paths = {r.path for r in records.values()}
        existing_branches = {(r.repo_root, r.branch) for r in records.values()}

        changed = False
        for wt_path_str, branch in candidates:
            if wt_path_str in existing_paths:
                _log.debug("adopt: already tracked (path match) %s", wt_path_str)
                continue
            if (repo_root_str, branch) in existing_branches:
                _log.debug(
                    "adopt: already tracked (branch match) %s @ %s",
                    branch, repo_root_str,
                )
                continue

            worktree_id = f"{repo_slug}-{_slug_ys(branch)}-{_short_uuid_ys()}"
            record = WorktreeRecord(
                id=worktree_id,
                repo_root=repo_root_str,
                branch=branch,
                path=wt_path_str,
                status="adopted",
                branch_created_by_us=False,
                ports={},
                pids={},
            )
            store._add_unlocked(records, record)
            existing_paths.add(wt_path_str)
            existing_branches.add((repo_root_str, branch))
            report.adopted.append(worktree_id)
            changed = True
            _log.debug(
                "adopt: imported worktree '%s' branch '%s'", worktree_id, branch
            )

        if changed:
            store._save_state(records)

    return report


__all__ = [
    "AdoptReport",
    "ReconcileReport",
    "YamlStateStore",
    "adopt",
    "reconcile",
]
