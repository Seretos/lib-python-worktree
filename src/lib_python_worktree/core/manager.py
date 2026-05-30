"""Thin wrapper around ``git worktree`` plus canonical id allocation.

W2 keeps this module strictly mechanical: ``subprocess`` calls to ``git`` and
the in-memory state store from ``state.py``. Setup-script execution (W5),
port allocation (W4), process lifecycle (W6) and full teardown semantics (W8)
will hook in around ``WorktreeManager`` later — the seams are documented at
``_teardown`` and ``create`` so future phases know where to inject.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..contract.loader import CONTRACT_FILENAME, load as _load_contract
from .port_allocator import PortAllocationError, PortAllocator, _NoOpPortAllocator
from .process_lifecycle import (
    ProcessAlreadyRunningError,
    ProcessLifecycleError,
    ProcessNotRunningError,
    start as _lifecycle_start,
    stop as _lifecycle_stop,
)
from .state import InMemoryStateStore, StateStore, WorktreeRecord
from .yaml_store import YamlStateStore, reconcile

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_DEFAULT_STORE_ROOT_ENV = "WORKTREE_STORE_ROOT"
_DEFAULT_STORE_DIR_NAME = "agent-worktree-store"
_GIT_TIMEOUT_ENV = "WORKTREE_GIT_TIMEOUT_SEC"
_GIT_TIMEOUT_DEFAULT = 30.0
_PORT_RANGE_ENV = "WORKTREE_PORT_RANGE"
_PORT_RANGE_DEFAULT = (30000, 40000)

# Stable since git 2.5 (builtin/worktree.c). Captures branch + path from
# the two variants git emits when refusing `worktree add` on a conflict:
#   fatal: 'feature/x' is already checked out at '/path/to/wt'
#   fatal: 'feature/x' is already used by worktree at '/path/to/wt'
# The "used by worktree at" wording is what modern git (2.40+) emits in
# practice; the older "checked out at" still appears in some code paths.
_ALREADY_CHECKED_OUT_RE = re.compile(
    r"fatal: '([^']+)' is already (?:checked out|used by worktree) at '([^']+)'"
)


class WorktreeError(RuntimeError):
    """Base class for ``WorktreeManager`` errors surfaced to MCP clients."""


class BranchNotFoundError(WorktreeError):
    pass


class BranchAlreadyCheckedOutError(WorktreeError):
    """Raised when ``git worktree add`` refuses because the branch is checked
    out in another worktree.

    Ticket #18: the raw ``GitCommandError`` is opaque ("fatal: 'X' is already
    checked out at '...'") and the MCP client cannot programmatically react.
    This carries the parsed branch + path plus a ``prunable`` flag derived
    from ``git worktree list --porcelain``, so callers can offer a "prune
    and retry" affordance.
    """

    def __init__(
        self, branch: str, path: str, prunable: Optional[bool]
    ) -> None:
        super().__init__(
            f"branch_already_checked_out: '{branch}' is checked out at "
            f"'{path}' (prunable={prunable}). "
            f"Hint: 'git worktree prune' or 'git worktree remove {path}'."
        )
        self.branch = branch
        self.path = path
        self.prunable = prunable


class DuplicateWorktreeError(WorktreeError):
    pass


class WorktreeNotFoundError(WorktreeError):
    pass


class GitCommandError(WorktreeError):
    def __init__(self, command: List[str], returncode: int, stderr: str) -> None:
        super().__init__(
            f"git command failed (exit {returncode}): {' '.join(command)}\n{stderr.strip()}"
        )
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


class GitTimeoutError(WorktreeError):
    """Raised when a ``git`` subprocess exceeds the configured timeout.

    Ticket #19: the Windows PyInstaller binary was hanging because the spawned
    ``git`` inherited the MCP client's stdin pipe and waited forever for input.
    ``_run_git`` now closes stdin, runs via ``Popen.communicate(timeout=...)``,
    and raises this on overrun so the MCP tool can surface a real error rather
    than blocking the client forever.
    """

    def __init__(self, command: List[str], elapsed: float) -> None:
        super().__init__(
            f"git command timed out after {elapsed:.1f}s: {' '.join(command)}"
        )
        self.command = command
        self.elapsed = elapsed


@dataclass
class ManagerConfig:
    """Runtime configuration for ``WorktreeManager``.

    ``store_root`` is the directory under which per-repo worktree checkouts
    live (decision D2, Option B). Resolved from ``WORKTREE_STORE_ROOT`` if
    unset on construction, falling back to ``~/agent-worktree-store``.

    ``port_range`` is the inclusive ``(low, high)`` range from which the port
    allocator draws ports. Resolved from ``WORKTREE_PORT_RANGE`` (format
    ``"30000-40000"``), falling back to ``(30000, 40000)``.
    """

    store_root: Path
    port_range: tuple = _PORT_RANGE_DEFAULT  # type: ignore[assignment]

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "ManagerConfig":
        environ = env if env is not None else os.environ
        raw = environ.get(_DEFAULT_STORE_ROOT_ENV)
        if raw:
            root = Path(raw).expanduser().resolve()
        else:
            root = (Path.home() / _DEFAULT_STORE_DIR_NAME).resolve()

        port_range: tuple[int, int] = _PORT_RANGE_DEFAULT
        raw_range = environ.get(_PORT_RANGE_ENV)
        if raw_range:
            try:
                low_s, high_s = raw_range.split("-", 1)
                port_range = (int(low_s.strip()), int(high_s.strip()))
            except (ValueError, TypeError):
                port_range = _PORT_RANGE_DEFAULT

        return cls(store_root=root, port_range=port_range)


def _slug(value: str, *, max_len: int = 40) -> str:
    """Lower-case ASCII slug suitable for filesystem use and IDs."""

    s = _SLUG_RE.sub("-", value.lower()).strip("-")
    if not s:
        s = "x"
    return s[:max_len]


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


def _resolve_git_timeout(explicit: Optional[float]) -> Optional[float]:
    """Resolve the timeout for a single ``_run_git`` call.

    Precedence: explicit kwarg > ``WORKTREE_GIT_TIMEOUT_SEC`` env > built-in
    default of 30.0 s. ``None`` (either as kwarg or env value ``""``) disables
    the timeout entirely; that path exists for diagnostics, not normal use.

    Env is read on every call so that test fixtures and operators can change
    the value without re-importing the module.
    """

    if explicit is not None:
        return explicit
    raw = os.environ.get(_GIT_TIMEOUT_ENV)
    if raw is None:
        return _GIT_TIMEOUT_DEFAULT
    raw = raw.strip()
    if not raw:
        # Empty string is "no timeout", matching the explicit-None semantics.
        return None
    try:
        value = float(raw)
    except ValueError:
        return _GIT_TIMEOUT_DEFAULT
    return value if value > 0 else None


def _run_git(
    args: List[str],
    cwd: Optional[Path] = None,
    *,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Run ``git <args>`` and return a ``CompletedProcess``.

    Ticket #19 hardenings:
    * ``stdin=DEVNULL`` so ``git`` can never inherit the MCP client's stdin
      pipe and wedge waiting on input -- this was the Windows-exe hang root
      cause.
    * Explicit ``stdout=PIPE, stderr=PIPE`` (rather than ``capture_output``)
      because we now drive a ``Popen`` directly to keep a clean kill path.
    * On Windows: ``creationflags=CREATE_NO_WINDOW`` so packaged-exe runs
      don't briefly flash a console window per git call.
    * ``timeout`` defaults from ``WORKTREE_GIT_TIMEOUT_SEC`` (30 s if unset);
      on overrun the process is killed and ``GitTimeoutError`` is raised.
    """

    effective_timeout = _resolve_git_timeout(timeout)

    popen_kwargs: dict = {
        "cwd": str(cwd) if cwd else None,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if sys.platform == "win32":
        # Suppress the brief console-window flash when the packaged worktree.exe
        # spawns git from a GUI MCP host.
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    cmd = ["git", *args]
    start = time.monotonic()
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=effective_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        # Drain the pipes after kill so the child fully reaps; ignore output.
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        elapsed = time.monotonic() - start
        raise GitTimeoutError(cmd, elapsed) from None

    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr
    )


def _parse_already_checked_out(stderr: str) -> Optional[tuple[str, str]]:
    """Return ``(branch, path)`` if stderr matches the git "already checked
    out" error, else ``None``.

    Ticket #18: stderr-parse is the primary path. We deliberately avoid a
    pre-check (`git worktree list` before `git worktree add`) because that
    would race with other processes; let git fail and read its verdict.
    """

    match = _ALREADY_CHECKED_OUT_RE.search(stderr or "")
    if match is None:
        return None
    return match.group(1), match.group(2)


def _is_path_prunable(repo_path: Path, target_path: str) -> Optional[bool]:
    """Probe ``git worktree list --porcelain`` for whether ``target_path``
    carries the ``prunable`` marker.

    Returns ``True`` / ``False`` if the path is found, ``None`` if it isn't
    listed at all (which itself usually means the worktree dir was wiped and
    a ``git worktree prune`` will clear the stale ref). The probe itself is
    best-effort: any failure returns ``None`` rather than masking the original
    "already checked out" error.
    """

    try:
        proc = _run_git(["worktree", "list", "--porcelain"], cwd=repo_path)
    except WorktreeError:
        return None
    if proc.returncode != 0:
        return None

    # Porcelain format: blocks separated by blank lines, each block starting
    # with `worktree <path>`. A `prunable <reason>` line within the block
    # marks it as removable.
    target_norm = str(Path(target_path)).replace("\\", "/").lower()
    current_path: Optional[str] = None
    current_prunable = False
    found: Optional[bool] = None

    def _flush() -> None:
        nonlocal found
        if current_path is None:
            return
        if current_path.replace("\\", "/").lower() == target_norm:
            found = current_prunable

    for raw_line in proc.stdout.splitlines():
        line = raw_line.rstrip()
        if not line:
            _flush()
            current_path = None
            current_prunable = False
            continue
        if line.startswith("worktree "):
            _flush()
            current_path = line[len("worktree "):].strip()
            current_prunable = False
        elif line.startswith("prunable"):
            current_prunable = True
    _flush()
    return found


class WorktreeManager:
    """High-level facade used by the FastMCP tools.

    Decision D1 (Option C): id = ``<repo-slug>-<branch-slug>-<short-uuid8>``.
    Decision D2 (Option B): worktree checkouts live under
    ``<store_root>/<repo-slug>/<id>/``.
    """

    def __init__(
        self,
        config: Optional[ManagerConfig] = None,
        state: Optional[StateStore] = None,
        *,
        reconcile_on_init: bool = True,
    ) -> None:
        self.config = config or ManagerConfig.from_env()
        resolved_state: StateStore = state if state is not None else YamlStateStore()
        self.state = resolved_state
        if reconcile_on_init and isinstance(resolved_state, YamlStateStore):
            reconcile(resolved_state)

        # Construct the port allocator.  When the state store is file-backed
        # (YamlStateStore) we use the real allocator backed by its _PortsFile.
        # For InMemoryStateStore (unit tests) we use a no-op stub so that
        # tests never touch the filesystem.
        if isinstance(resolved_state, YamlStateStore):
            self._allocator: object = PortAllocator(
                resolved_state._ports,
                port_range=self.config.port_range,
            )
        else:
            self._allocator = _NoOpPortAllocator()

    # ---- public API used by the FastMCP tools ----

    def create(
        self,
        repo_root: str,
        branch: str,
        base: Optional[str] = None,
    ) -> WorktreeRecord:
        repo_path = self._validate_repo(repo_root)

        branch = branch.strip()
        if not branch:
            raise WorktreeError("branch must be a non-empty string")

        repo_slug = _slug(repo_path.name)

        if self.state.find_by_branch(str(repo_path), branch) is not None:
            raise DuplicateWorktreeError(
                f"A worktree for branch '{branch}' already exists in {repo_path}"
            )

        branch_exists = self._branch_exists(repo_path, branch)
        if not branch_exists and base is None:
            raise BranchNotFoundError(
                f"Branch '{branch}' does not exist in {repo_path}. "
                "Pass `base` to create it."
            )
        if not branch_exists and base is not None and not self._branch_exists(
            repo_path, base
        ):
            raise BranchNotFoundError(
                f"Base branch '{base}' does not exist in {repo_path}."
            )

        worktree_id = f"{repo_slug}-{_slug(branch)}-{_short_uuid()}"
        target_path = self.config.store_root / repo_slug / worktree_id
        target_path.parent.mkdir(parents=True, exist_ok=True)

        git_args = ["worktree", "add"]
        if not branch_exists:
            git_args += ["-b", branch, str(target_path), base]  # type: ignore[list-item]
        else:
            git_args += [str(target_path), branch]

        proc = _run_git(git_args, cwd=repo_path)
        if proc.returncode != 0:
            # Ticket #18: surface the specific "branch already checked out
            # elsewhere" condition as a structured error so callers can offer
            # prune/remove affordances. Falls through to GitCommandError for
            # any other failure.
            parsed = _parse_already_checked_out(proc.stderr)
            if parsed is not None:
                conflict_branch, conflict_path = parsed
                prunable = _is_path_prunable(repo_path, conflict_path)
                raise BranchAlreadyCheckedOutError(
                    branch=conflict_branch,
                    path=conflict_path,
                    prunable=prunable,
                )
            raise GitCommandError(["git", *git_args], proc.returncode, proc.stderr)

        # Load the contract, allocate ports, and persist the state record.
        # All three steps are inside the same try/except so that ANY failure
        # (ContractError, PortAllocationError, state.add failure) triggers the
        # same git-worktree rollback.  A missing contract file is silently
        # treated as an implicit isolation:none contract with no ports.
        port_mapping: dict = {}
        try:
            contract_path = repo_path / CONTRACT_FILENAME
            contract = _load_contract(contract_path)

            if contract.ports:
                slot_names = [slot.name for slot in contract.ports]
                port_mapping = self._allocator.allocate(slot_names, worktree_id)

            record = WorktreeRecord(
                id=worktree_id,
                repo_root=str(repo_path),
                branch=branch,
                path=str(target_path),
                branch_created_by_us=not branch_exists,
                ports=port_mapping,
            )
            self.state.add(record)
        except Exception:
            # Roll back: remove the git worktree we just created (--force
            # because the checkout may be empty / partially written), release
            # any ports already written by allocate(), then delete the branch
            # if this manager created it.  Failures in the rollback itself are
            # swallowed so we always re-raise the original exception.
            try:
                _run_git(
                    ["worktree", "remove", "--force", str(target_path)],
                    cwd=repo_path,
                )
            except Exception:  # noqa: BLE001
                pass
            try:
                self._allocator.release(worktree_id)
            except Exception:  # noqa: BLE001
                pass
            if not branch_exists:
                try:
                    _run_git(["branch", "-D", branch], cwd=repo_path)
                except Exception:  # noqa: BLE001
                    pass
            raise

        return record

    def list(self) -> List[WorktreeRecord]:
        return self.state.list()

    def remove(self, worktree_id: str, force: bool = False) -> WorktreeRecord:
        record = self.state.get(worktree_id)
        if record is None:
            raise WorktreeNotFoundError(
                f"No worktree tracked with id '{worktree_id}'"
            )
        # Phase 1: remove the git worktree checkout.  If this raises the
        # directory still exists, so we keep the state record and propagate.
        self._teardown(record, force=force)
        # Phase 2: the worktree directory is now gone.  Remove the state record
        # *before* the branch-delete step so that a branch-delete failure
        # (e.g. ``git branch -d`` refusing an unmerged branch when force=False)
        # does not leave a stale orphaned record in the state store.
        removed = self.state.remove(worktree_id)
        assert removed is not None  # state.get returned record above
        # Phase 3: delete the owned branch (if any).  May raise GitCommandError
        # (e.g. unmerged + force=False); the record is already gone from state.
        self._delete_owned_branch(record, force=force)
        return removed

    def start(
        self,
        worktree_id: str,
        cmd: List[str],
        *,
        role: str = "main",
        env: Optional[dict] = None,
        cwd: Optional[str] = None,
    ) -> WorktreeRecord:
        """Spawn a detached process for *worktree_id* and record its PID.

        Delegates to ``process_lifecycle.start`` with ``store=self.state``.
        """
        return _lifecycle_start(
            worktree_id,
            cmd,
            store=self.state,
            role=role,
            env=env,
            cwd=cwd,
        )

    def stop(
        self,
        worktree_id: str,
        *,
        role: str = "main",
        timeout: float = 10.0,
    ) -> WorktreeRecord:
        """Stop the process recorded under *role* for *worktree_id*.

        Delegates to ``process_lifecycle.stop`` with ``store=self.state``.
        """
        return _lifecycle_stop(
            worktree_id,
            store=self.state,
            role=role,
            timeout=timeout,
        )

    # ---- seams for later phases ----

    def _teardown(
        self,
        record: WorktreeRecord,
        *,
        force: bool,
        _lifecycle_module=None,
    ) -> None:
        """Remove the git worktree checkout directory.

        Sequence (W8):
        1. Stop any tracked processes (process lifecycle).
        2. Run any contract ``teardown:`` steps via ``SetupRunner``.
        3. Remove the git worktree checkout.
        4. Release allocated ports (only after step 3 succeeds).

        Branch deletion is intentionally *not* done here — it happens in
        ``remove()`` after the state record has been cleaned up, so that a
        branch-delete failure cannot leave a stale orphaned state entry.

        ``_lifecycle_module`` is an injection seam for tests; callers should
        leave it as ``None`` (the real ``process_lifecycle`` module is used).
        """
        lifecycle = _lifecycle_module
        if lifecycle is None:
            from . import process_lifecycle as lifecycle  # type: ignore[assignment]

        # Step 1: stop any tracked processes before removing the worktree dir.
        if record.pids:
            for role in list(record.pids.keys()):
                try:
                    lifecycle.stop(record.id, store=self.state, role=role)
                except ProcessNotRunningError:
                    pass
                except ProcessLifecycleError:
                    # Best-effort: log-worthy but don't block the removal.
                    pass

        # Step 2: run contract teardown: steps.
        # A missing contract is treated as isolation:none (no teardown steps).
        try:
            contract_path = Path(record.repo_root) / CONTRACT_FILENAME
            contract = _load_contract(contract_path)
            if contract.teardown:
                from ..setup.runner import SetupRunner
                runner = SetupRunner()
                try:
                    runner.run(
                        setup=contract.teardown,
                        worktree_id=record.id,
                        worktree_path=Path(record.path),
                        branch=record.branch,
                        port_mapping=record.ports,
                    )
                except Exception:  # noqa: BLE001
                    # Teardown step failure must not block git worktree remove.
                    pass
        except Exception:  # noqa: BLE001
            # Any contract load failure is silently skipped — same pattern as
            # create().
            pass

        # Step 3: remove the git worktree checkout directory.
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(record.path)
        proc = _run_git(args, cwd=Path(record.repo_root))
        if proc.returncode != 0:
            raise GitCommandError(
                ["git", *args], proc.returncode, proc.stderr
            )

        # Step 4: release allocated ports only after the git worktree remove
        # has succeeded.  Freeing ports before the remove would allow a
        # concurrent allocate() to reissue the same ports while the original
        # service is still bound to them.
        self._allocator.release(record.id)

    def _delete_owned_branch(self, record: WorktreeRecord, *, force: bool) -> None:
        """Delete the branch if we created it (``git worktree add -b``).

        Branches that pre-existed (reuse path, no ``base`` supplied) are left
        untouched. A future ``keep_branch`` parameter on ``remove`` is the
        intended per-call opt-out hook — deferred to a follow-up ticket.

        Raises ``GitCommandError`` if the branch exists but deletion fails
        (e.g. ``git branch -d`` refuses an unmerged branch with force=False).
        Skips silently if the branch is already gone (idempotent).
        """

        if not record.branch_created_by_us:
            return
        repo_path = Path(record.repo_root)
        if not self._branch_exists(repo_path, record.branch):
            # Already gone — skip silently (idempotent).
            return
        delete_flag = "-D" if force else "-d"
        del_args = ["branch", delete_flag, record.branch]
        del_proc = _run_git(del_args, cwd=repo_path)
        if del_proc.returncode != 0:
            raise GitCommandError(
                ["git", *del_args], del_proc.returncode, del_proc.stderr
            )

    # ---- helpers ----

    def _validate_repo(self, repo_root: str) -> Path:
        if not repo_root:
            raise WorktreeError("repo_root must be a non-empty path")
        path = Path(repo_root).expanduser().resolve()
        if not path.exists():
            raise WorktreeError(f"repo_root does not exist: {path}")
        proc = _run_git(["rev-parse", "--show-toplevel"], cwd=path)
        if proc.returncode != 0:
            raise WorktreeError(f"Not a git repository: {path}")
        return Path(proc.stdout.strip()).resolve()

    def _branch_exists(self, repo_path: Path, branch: str) -> bool:
        proc = _run_git(
            ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=repo_path,
        )
        return proc.returncode == 0


__all__ = (
    "BranchAlreadyCheckedOutError",
    "BranchNotFoundError",
    "DuplicateWorktreeError",
    "GitCommandError",
    "GitTimeoutError",
    "ManagerConfig",
    "WorktreeError",
    "WorktreeManager",
    "WorktreeNotFoundError",
    "PortAllocationError",
    "ProcessAlreadyRunningError",
    "ProcessLifecycleError",
    "ProcessNotRunningError",
)
