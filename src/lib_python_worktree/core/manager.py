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
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# `subprocess` is kept for CompletedProcess / DEVNULL references inside this
# module even though _run_git now lives in _git_utils.

from ..contract.loader import CONTRACT_FILENAME, load as _load_contract
from ._env_utils import _get_user_profile_env
from ._exceptions import DirtyWorktreeError, GitTimeoutError, InvalidRepoError, UnknownVariantError, WorktreeDirLockedError, WorktreeError  # noqa: F401 — re-exported
from ._git_utils import _resolve_git_timeout, _run_git  # noqa: F401 — re-exported
from .port_allocator import PortAllocationError, PortAllocator, _NoOpPortAllocator
from .process_lifecycle import (
    ProcessAlreadyRunningError,
    ProcessLifecycleError,
    ProcessNotRunningError,
    _find_blocking_processes,
    _kill_blocking_processes,
    start as _lifecycle_start,
    stop as _lifecycle_stop,
)
from .state import InMemoryStateStore, StateStore, WorktreeRecord
from .yaml_store import AdoptReport, YamlStateStore, adopt as _yaml_adopt, reconcile

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_DEFAULT_STORE_ROOT_ENV = "WORKTREE_STORE_ROOT"
_DEFAULT_STORE_DIR_NAME = "agent-worktree-store"
_PORT_RANGE_ENV = "WORKTREE_PORT_RANGE"
_PORT_RANGE_DEFAULT = (30000, 40000)

# Retry constants for the post-kill directory-unlock loop (ticket #51).
_POST_KILL_RETRIES: int = 5      # attempts after kill before giving up
_POST_KILL_SLEEP: float = 0.5    # seconds to wait between retries

# Stable since git 2.5 (builtin/worktree.c). Captures branch + path from
# the two variants git emits when refusing `worktree add` on a conflict:
#   fatal: 'feature/x' is already checked out at '/path/to/wt'
#   fatal: 'feature/x' is already used by worktree at '/path/to/wt'
# The "used by worktree at" wording is what modern git (2.40+) emits in
# practice; the older "checked out at" still appears in some code paths.
_ALREADY_CHECKED_OUT_RE = re.compile(
    r"fatal: '([^']+)' is already (?:checked out|used by worktree) at '([^']+)'"
)


class BranchNotFoundError(WorktreeError):
    pass


class InvalidBranchError(WorktreeError):
    """Raised when ``branch`` is an empty or whitespace-only string."""

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

    for raw_line in (proc.stdout or "").splitlines():
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


def _build_worktree_env(
    record: "WorktreeRecord",
    caller_env: "Optional[Dict[str, str]]",
) -> "Dict[str, str]":
    """Build the child-process environment for a worktree start call.

    Merge order (rightmost wins per key):
        _get_user_profile_env()  <--  worktree identity/port vars  <--  caller_env

    ``_get_user_profile_env()`` returns a complete user-profile environment
    (registry-sourced on Windows, ``dict(os.environ)`` elsewhere) so that
    child processes spawned via the ``start:`` step inherit Windows user-profile
    vars (``APPDATA``, ``LOCALAPPDATA``, ``USERPROFILE``, etc.) that are absent
    from a headless MCP server's ``os.environ``.

    Variable names mirror ``SetupRunner._build_env`` in ``setup/runner.py``
    (the sibling implementation of this convention).  Do NOT extract a shared
    helper — the two are peers in separate layers; mirroring the few lines
    here avoids new coupling and circular imports.
    """
    env: Dict[str, str] = _get_user_profile_env()
    env["WORKTREE_ID"] = record.id
    env["WORKTREE_PATH"] = record.path
    env["WORKTREE_BRANCH"] = record.branch
    for slot, port in record.ports.items():
        env[f"WORKTREE_PORT_{slot.upper()}"] = str(port)
    if caller_env is not None:
        env.update(caller_env)
    return env


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
        _plugin_seed_config_dir: Optional[Path] = None,
        _plugin_install_config_dir: Optional[Path] = None,
        _plugin_install_which: Optional[object] = None,
        _plugin_install_runner: Optional[object] = None,
    ) -> None:
        self.config = config or ManagerConfig.from_env()
        resolved_state: StateStore = state if state is not None else YamlStateStore()
        self.state = resolved_state
        self._plugin_seed_config_dir = _plugin_seed_config_dir
        self._plugin_install_config_dir = _plugin_install_config_dir
        # Test seams only: let integration tests inject fake `which`/subprocess
        # runners for install_enabled_plugins() without spawning a real
        # `claude` process. Left None in production so the real
        # shutil.which/subprocess path is used.
        self._plugin_install_which = _plugin_install_which
        self._plugin_install_runner = _plugin_install_runner
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
        *,
        fetch: bool = True,
    ) -> WorktreeRecord:
        repo_path = self._validate_repo(repo_root)

        branch = branch.strip()
        if not branch:
            raise InvalidBranchError("branch must be a non-empty string")

        repo_slug = _slug(repo_path.name)

        if self.state.find_by_branch(repo_path.as_posix(), branch) is not None:
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

        # When creating a new branch from a base and fetch=True, fetch the
        # base branch from origin so the new worktree starts from the latest
        # remote commit rather than a potentially stale local ref.
        if not branch_exists and base is not None and fetch:
            fetch_proc = _run_git(["fetch", "origin", base], cwd=repo_path)
            if fetch_proc.returncode != 0:
                raise GitCommandError(
                    ["git", "fetch", "origin", base],
                    fetch_proc.returncode,
                    fetch_proc.stderr,
                )

        worktree_id = f"{repo_slug}-{_slug(branch)}-{_short_uuid()}"
        target_path = self.config.store_root / repo_slug / worktree_id
        target_path.parent.mkdir(parents=True, exist_ok=True)

        git_args = ["worktree", "add"]
        if not branch_exists:
            # When fetch=True use origin/<base> so the new branch starts from
            # the freshly-fetched remote tip, not the (possibly stale) local ref.
            base_ref = f"origin/{base}" if fetch else base
            git_args += ["-b", branch, str(target_path), base_ref]  # type: ignore[list-item]
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
                repo_root=repo_path.as_posix(),
                branch=branch,
                path=target_path.as_posix(),
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

        # Run contract setup: steps right after the record is persisted.
        # Contract is loaded from repo_path (not worktree_path) because the
        # plugin layer copies .seretos into the worktree *after* create()
        # returns.  A missing/empty contract (isolation:none) is a no-op.
        # On step failure: leave the worktree, ports, and state record intact
        # for user inspection; update status to "setup_failed" and re-raise
        # SetupFailedError so the caller knows setup did not complete.
        _setup_contract = _load_contract(repo_path / CONTRACT_FILENAME)
        if _setup_contract.setup:
            from ..setup.runner import SetupRunner  # noqa: PLC0415
            _setup_runner = SetupRunner()
            try:
                _setup_runner.run(
                    setup=_setup_contract.setup,
                    worktree_id=record.id,
                    worktree_path=Path(record.path),
                    branch=record.branch,
                    port_mapping=record.ports,
                )
            except Exception:  # noqa: BLE001
                record.status = "setup_failed"
                self.state.update(record)
                raise

        # Install the worktree's enabledPlugins so that project-scoped
        # plugins are active without a manual /reload-plugins. Clone-first
        # mechanism (ticket #64): registers each key by cloning an existing,
        # structurally-valid registry entry under a lock, falling back to
        # `claude plugin install --scope project` only when no valid clone
        # source exists. This is now self-sufficient (no `claude` CLI on
        # PATH required), so the old plugin_seed fallback wiring has been
        # retired. Best-effort — failures here must never fail create().
        try:
            from .plugin_install import install_enabled_plugins  # noqa: PLC0415
            install_enabled_plugins(
                record.repo_root,
                record.path,
                worktree_id=record.id,
                config_dir=self._plugin_install_config_dir,
                which=self._plugin_install_which,
                runner=self._plugin_install_runner,
            )
        except Exception:  # noqa: BLE001
            pass

        return record

    def list(self) -> List[WorktreeRecord]:
        return self.state.list()

    def remove(
        self,
        worktree_id: str,
        force: bool = False,
        kill_blocking_processes: bool = False,
    ) -> WorktreeRecord:
        record = self.state.get(worktree_id)
        if record is None:
            raise WorktreeNotFoundError(
                f"No worktree tracked with id '{worktree_id}'"
            )
        # Phase 1: remove the git worktree checkout.  If this raises the
        # directory still exists, so we keep the state record and propagate.
        self._teardown(record, force=force, kill_blocking_processes=kill_blocking_processes)
        # Phase 2: the worktree directory is now gone.  Remove the state record
        # *before* the branch-delete step so that a branch-delete failure
        # (e.g. ``git branch -d`` refusing an unmerged branch when force=False)
        # does not leave a stale orphaned record in the state store.
        removed = self.state.remove(worktree_id)
        assert removed is not None  # state.get returned record above
        removed.status = "removed"
        # Copy killed_pids from the in-memory record: YamlStateStore.remove()
        # returns a freshly-deserialized object that never carries killed_pids
        # (the field is transient and not written to state.yaml), so we must
        # propagate it explicitly from the object _teardown mutated.
        removed.killed_pids = record.killed_pids
        # Phase 3: delete the owned branch (if any).  May raise GitCommandError
        # (e.g. unmerged + force=False); the record is already gone from state.
        self._delete_owned_branch(record, force=force)
        return removed

    def adopt(self, repo_root: str) -> "AdoptReport":
        """Discover git worktrees that exist on-disk but are not in the store.

        Calls ``git worktree list --porcelain`` against ``repo_root`` and
        imports any unknown worktrees as ``WorktreeRecord`` entries with
        ``status="adopted"`` and ``branch_created_by_us=False``.

        Only available when the state store is a file-backed ``YamlStateStore``.
        Raises ``WorktreeError`` for any other store type.
        """
        if not isinstance(self.state, YamlStateStore):
            raise WorktreeError("adopt() requires a file-backed YamlStateStore")
        repo_path = self._validate_repo(repo_root)
        return _yaml_adopt(self.state, repo_path)

    def prune(self, repo_root: str) -> None:
        """Run ``git worktree prune --expire=now`` against ``repo_root``.

        Removes stale worktree registrations from git's internal metadata (the
        ``.git/worktrees/`` directory).  ``--expire=now`` overrides git's default
        3-month grace period (``gc.worktreePruneExpire``) so that worktrees whose
        directory was deleted moments ago are pruned immediately rather than being
        kept as "recently used".  Raises ``GitCommandError`` on non-zero returncode.
        """
        repo_path = self._validate_repo(repo_root)
        proc = _run_git(["worktree", "prune", "--expire=now"], cwd=repo_path)
        if proc.returncode != 0:
            raise GitCommandError(
                ["git", "worktree", "prune", "--expire=now"],
                proc.returncode,
                proc.stderr,
            )

    def start(
        self,
        worktree_id: str,
        *,
        role: str = "main",
        env: Optional[dict] = None,
        cwd: Optional[str] = None,
        variant: str = "default",
    ) -> WorktreeRecord:
        """Spawn a detached process for *worktree_id* using the contract's
        ``start:`` step, and record its PID.

        The command is read from the ``start:`` field of the worktree contract
        at ``<repo_root>/.seretos/worktree-setup.yml``.

        *variant* selects which step to run (default ``"default"``):

        - If *variant* is ``"default"`` and exactly one step has no ``name``
          set, that step is used (backward-compatibility path).
        - Otherwise the step whose ``name`` equals *variant* is used.
        - If no matching step is found, ``UnknownVariantError`` is raised
          listing the available named steps. ``UnknownVariantError`` is both
          a ``WorktreeError`` and a ``ValueError``, so callers may catch
          either base.

        When no ``start:`` step is configured at all (missing
        ``.seretos/worktree-setup.yml`` or an empty ``start:`` list), there is
        nothing meaningful to run.  Rather than erroring, this is treated as a
        **no-op start**: no process is spawned, the worktree is marked
        ``status="ready"`` (usable, with no managed process), and the record is
        returned.  This makes "just give me a worktree I can work in" work out
        of the box for simple repos (e.g. dependency-bump chores).  See
        ticket #41.

        Delegates to ``process_lifecycle.start`` with ``store=self.state``
        only when a concrete ``start:`` step is selected.
        """
        record = self.state.get(worktree_id)
        if record is None:
            raise WorktreeNotFoundError(
                f"No worktree tracked with id '{worktree_id}'"
            )

        contract = _load_contract(Path(record.repo_root) / CONTRACT_FILENAME)

        if not contract.start:
            # No start: step configured — nothing to run.  Treat as a no-op
            # start so worktree creation + start works without a contract:
            # mark the worktree usable and return without spawning a process.
            record.status = "ready"
            self.state.update(record)
            return record

        # Step selection
        # (1) Exact name match wins (covers explicit name: "default" and any named variant)
        step = None
        for s in contract.start:
            if s.name == variant:
                step = s
                break

        # (2) Back-compat: variant defaulted and no exact match → use the lone unnamed step
        if step is None and variant == "default":
            unnamed_steps = [s for s in contract.start if s.name is None]
            if len(unnamed_steps) == 1:
                step = unnamed_steps[0]

        if step is None:
            available = [s.name for s in contract.start if s.name]
            raise UnknownVariantError(variant, available)

        from ..setup.runner import _resolve_shell
        cmd = [*_resolve_shell(step.shell), step.run]

        return _lifecycle_start(
            worktree_id,
            cmd,
            store=self.state,
            role=role,
            env=_build_worktree_env(record, env),
            cwd=cwd,
        )

    def stop(
        self,
        worktree_id: str,
        *,
        role: str = "main",
        timeout: float = 10.0,
        kill_orphans: bool = False,
    ) -> WorktreeRecord:
        """Stop the process recorded under *role* for *worktree_id*.

        If the contract defines ``stop:`` steps, they are run (best-effort,
        errors are swallowed) before sending the stop signal.

        When *kill_orphans* is ``True``, a cwd/open-file scan is run after
        the primary signal to terminate any orphaned grandchild processes that
        survived because the tracked shell wrapper already exited and they were
        reparented away from it.

        When no process is recorded for *role* (e.g. after a no-op ``"ready"``
        start with no ``start:`` step — ticket #41), stopping is a graceful
        no-op: contract ``stop:`` steps are still run best-effort, but no
        signal is sent and ``ProcessNotRunningError`` is *not* raised.  The
        worktree is marked ``"stopped"`` when no other roles remain.

        This is the engine's documented and intentional behavior (ticket
        #41) and this method's return type is fixed: it always returns a
        ``WorktreeRecord`` (or raises ``WorktreeNotFoundError`` for an
        unknown *worktree_id*), never a dict-shaped "soft error" result.
        Any dict-shaped soft-error contract for a never-started role (e.g.
        ``{"error": ..., "code": ...}``) is owned by the MCP wrapper layer in
        the separate ``agent-worktree`` plugin repo, which translates this
        engine's return values/exceptions into whatever shape its tool
        surface promises callers — it is not this engine's concern and is
        not implemented here (see ``AGENTS.md``'s "Layering" section).

        Delegates to ``process_lifecycle.stop`` with ``store=self.state``.
        """
        record = self.state.get(worktree_id)
        if record is None:
            raise WorktreeNotFoundError(
                f"No worktree tracked with id '{worktree_id}'"
            )

        # Run contract stop: steps (best-effort — a failure must not prevent
        # the SIGTERM from being sent).
        try:
            contract_path = Path(record.repo_root) / CONTRACT_FILENAME
            contract = _load_contract(contract_path)
            if contract.stop:
                from ..setup.runner import SetupRunner
                runner = SetupRunner()
                try:
                    runner.run(
                        setup=contract.stop,
                        worktree_id=record.id,
                        worktree_path=Path(record.path),
                        branch=record.branch,
                        port_mapping=record.ports,
                    )
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass

        # No process recorded for this role → graceful no-op (symmetric with
        # the no-op "ready" start).  Avoid delegating to _lifecycle_stop, which
        # would raise ProcessNotRunningError.
        if role not in record.pids:
            if not record.pids:
                record.status = "stopped"
            self.state.update(record)
            return record

        return _lifecycle_stop(
            worktree_id,
            store=self.state,
            role=role,
            timeout=timeout,
            kill_orphans=kill_orphans,
        )

    def run_seed_postprocess(self, worktree_id: str) -> "SetupResult":
        """Run the contract's ``seed_postprocess:`` steps in isolation.

        Loads the contract, builds the same ``WORKTREE_*`` environment as
        ``setup:``, and delegates to ``SetupRunner``.  Raises
        ``SetupFailedError`` on the first non-zero step exit.  Raises
        ``WorktreeNotFoundError`` if ``worktree_id`` is unknown.  Steps are
        expected to be idempotent (delete-then-insert style) so this can be
        called repeatedly.

        An empty ``seed_postprocess:`` list is a silent no-op: an empty
        ``SetupResult`` is returned and ``SetupRunner`` is never invoked.
        """
        record = self.state.get(worktree_id)
        if record is None:
            raise WorktreeNotFoundError(
                f"No worktree tracked with id '{worktree_id}'"
            )

        contract_path = Path(record.repo_root) / CONTRACT_FILENAME
        contract = _load_contract(contract_path)

        if not contract.seed_postprocess:
            from ..setup.runner import SetupResult
            return SetupResult(worktree_id=worktree_id)

        from ..setup.runner import SetupRunner
        runner = SetupRunner()
        return runner.run(
            setup=contract.seed_postprocess,
            worktree_id=record.id,
            worktree_path=Path(record.path),
            branch=record.branch,
            port_mapping=record.ports,
        )

    # ---- seams for later phases ----

    def _teardown(
        self,
        record: WorktreeRecord,
        *,
        force: bool,
        kill_blocking_processes: bool = False,
        _lifecycle_module=None,
    ) -> None:
        """Remove the git worktree checkout directory.

        Sequence (W8):
        1. Stop any tracked processes (process lifecycle).
        1b. Run contract ``stop:`` steps via ``SetupRunner`` (best-effort,
            before any FS delete so that daemons with PID files can release
            file handles gracefully).
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

        # Step 1b: run contract stop: steps before FS deletion so that daemons
        # (e.g. Unity Editor) that write PID files / hold handles have a chance
        # to release them before git worktree remove is attempted.
        try:
            contract_path = Path(record.repo_root) / CONTRACT_FILENAME
            contract = _load_contract(contract_path)
            if contract.stop:
                from ..setup.runner import SetupRunner
                runner = SetupRunner()
                try:
                    runner.run(
                        setup=contract.stop,
                        worktree_id=record.id,
                        worktree_path=Path(record.path),
                        branch=record.branch,
                        port_mapping=record.ports,
                    )
                except Exception:  # noqa: BLE001
                    # A stop-step failure must not block the rest of teardown.
                    pass
        except Exception:  # noqa: BLE001
            # Any contract load failure is silently skipped.
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

        # Step 2b (Windows only, ticket #76): pre-flight blocking-process check
        # BEFORE the destructive `git worktree remove` call below.
        #
        # Root cause this guards against: on Windows, `git worktree remove
        # --force` can return exit 0 while still leaving a content-less
        # locked directory behind — an open file handle from a blocking
        # process blocks only the *final* directory removal, not the
        # individual file unlinks. That falls past the lock-detection branch
        # below (which only triggers on `returncode != 0`) into the Final
        # guard, which used to raise WorktreeDirLockedError with no way to
        # know a kill was never attempted — and by then the destructive
        # partial removal has already happened, so no later retry (even with
        # kill_blocking_processes=True) can ever recover: the files are gone
        # but the directory is still locked.
        #
        # Running this check first means an errored removal has no
        # destructive side effect, and a later kill_blocking_processes=True
        # retry can still find and kill the still-alive blocker.
        #
        # Placed after the contract stop:/teardown: steps (so daemons that
        # release handles during those steps are not falsely flagged) but
        # before the git call. POSIX is intentionally excluded: POSIX unlinks
        # files even under an open handle, so `git worktree remove` succeeds
        # there — a naive pre-flight applied to POSIX would incorrectly
        # block/kill on a merely-cwd'd process rather than a genuine locker.
        kill_attempted = False
        if sys.platform == "win32":
            _preflight_blockers = _find_blocking_processes(record.path, os.getpid())
            if _preflight_blockers:
                if kill_blocking_processes:
                    killed = _kill_blocking_processes(record.path)
                    record.killed_pids = killed
                    kill_attempted = True
                    # Fall through to the (now-safe) git remove call below.
                else:
                    # Caller did not opt into the kill-and-retry remedy: raise
                    # immediately, before the destructive git call runs, so
                    # no partial removal can occur on this attempt.
                    raise WorktreeDirLockedError(
                        record.id, killed=[], kill_attempted=False
                    )

        # Step 3: remove the git worktree checkout directory.
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(record.path)
        proc = _run_git(args, cwd=Path(record.repo_root))
        def _phantom_state_cleanup() -> None:
            """Remove the leftover directory and prune stale git metadata.

            Called when git reports 'is not a working tree' — meaning it has
            already deregistered the worktree from its internal registry on a
            prior attempt, but the directory and YAML state record were never
            cleaned up.  Both operations are best-effort; errors are swallowed
            so that port release and state removal still occur.
            """
            shutil.rmtree(record.path, ignore_errors=True)
            _run_git(["worktree", "prune"], cwd=Path(record.repo_root))

        if proc.returncode != 0:
            if proc.returncode == 128 and "is not a working tree" in proc.stderr:
                # git has already deregistered this worktree (phantom-state
                # scenario from ticket #51).  Treat as already-gone: clean up
                # the leftover directory and stale metadata, then fall through
                # to port release.
                _phantom_state_cleanup()
                # Fall through to step 4.
            elif proc.returncode == 128 and force:
                # The .git link is already gone (worktree dir was wiped
                # externally).  Fall back: delete the directory ourselves,
                # then prune the stale git metadata.  Both steps are
                # best-effort so that port release and state removal still
                # occur even in a degraded state.
                shutil.rmtree(record.path, ignore_errors=True)
                _run_git(
                    ["worktree", "prune"],
                    cwd=Path(record.repo_root),
                )
                # Fall through to step 4.
            else:
                # Determine whether this exit qualifies as a directory-lock
                # signal (an OS-level lock held by another process on the
                # worktree directory). This is checked BEFORE the dirty-tree
                # check (ticket #72, Befund 1): a lock-induced failure must
                # never be misread as a dirty working tree just because
                # git's dirty-tree phrase happens to also be present in
                # stderr (e.g. a stale git error string bundled alongside
                # a Win32 delete-failure string).
                # Windows: "Permission denied" or "Invalid argument" in
                #   stderr (both are NTFS/Win32 delete-failure strings).
                #   No exit-code requirement — real-world Win32 lock
                #   failures have been observed on exit codes other than
                #   255, so the strict `returncode == 255` check has been
                #   dropped (ticket #72, Befund 2).
                # POSIX: "lock" in stderr (case-insensitive) — git worktree
                #   reports a held directory as "locked" / "unable to lock" /
                #   "cannot lock" / "worktree is locked".  All variants contain
                #   the substring "lock".  Unrelated git failures (broken
                #   metadata, network FS errors, "not a git repository", etc.)
                #   match neither arm, so they fall through past the
                #   dirty-tree check to GitCommandError unchanged.
                _is_lock_signal = (
                    sys.platform == "win32"
                    and (
                        "Permission denied" in proc.stderr
                        or "Invalid argument" in proc.stderr
                    )
                ) or (
                    sys.platform != "win32"
                    and "lock" in proc.stderr.lower()
                )
                if _is_lock_signal:
                    # A lock's remedy (kill_blocking_processes=True) is
                    # independent of --force, so this branch applies for
                    # BOTH force=True and force=False (ticket #72, Befund 2).
                    if kill_blocking_processes:
                        killed = _kill_blocking_processes(record.path)
                        record.killed_pids = killed
                        kill_attempted = True
                        retry_result = None
                        _phantom_on_retry = False
                        for _attempt in range(_POST_KILL_RETRIES):
                            retry_result = _run_git(args, cwd=Path(record.repo_root))
                            if retry_result.returncode == 0:
                                break
                            if (
                                retry_result.returncode == 128
                                and "is not a working tree" in retry_result.stderr
                            ):
                                # git deregistered the worktree between the kill
                                # and this retry (phantom-state mid-loop).  Treat
                                # as already-gone and fall through to port release.
                                _phantom_state_cleanup()
                                _phantom_on_retry = True
                                break
                            if _attempt < _POST_KILL_RETRIES - 1:
                                time.sleep(_POST_KILL_SLEEP)
                        if not _phantom_on_retry and (
                            retry_result is None or retry_result.returncode != 0
                        ):
                            raise WorktreeDirLockedError(record.id, killed=killed)
                        # Retry succeeded (or phantom-state cleanup ran) —
                        # fall through to long-path check then step 4.
                    else:
                        # Lock detected but the caller did not opt into the
                        # kill-and-retry remedy: raise a clean domain error
                        # naming the remedy (kill_blocking_processes=True)
                        # rather than leaking git's raw stderr via a bare
                        # GitCommandError (ticket #72, Befund 2). No kill is
                        # attempted and no retry is performed.
                        raise WorktreeDirLockedError(
                            record.id, killed=[], kill_attempted=False
                        )
                elif (
                    proc.returncode == 128
                    and not force
                    and "contains modified or untracked files" in proc.stderr
                ):
                    # git refused because the worktree has uncommitted changes.
                    # Surface a structured error naming only the engine parameter
                    # (force=True), not the raw git command, path, or exit code.
                    raise DirtyWorktreeError(record.id)
                else:
                    raise GitCommandError(
                        ["git", *args], proc.returncode, proc.stderr
                    )

        # Long-path fallback: on Windows, 'git worktree remove' can succeed
        # (exit 0) but leave the directory behind when paths exceed MAX_PATH.
        # In that case attempt \\?\ prefixed deletion; if that also fails, try
        # the robocopy empty-mirror trick.  On POSIX, shutil.rmtree is the
        # simple fallback.
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
                            subprocess.run(
                                ["robocopy", empty_tmp, record.path, "/MIR"],
                                capture_output=True,
                            )
                            shutil.rmtree(record.path, ignore_errors=True)
                    except Exception:  # noqa: BLE001
                        pass  # Best-effort: don't block port release.
            else:
                shutil.rmtree(record.path, ignore_errors=True)

        # Final guard: if the directory is still present after all deletion
        # attempts, the worktree is locked by an external process.  Raise
        # rather than returning a false status: "removed".
        if os.path.exists(record.path):
            raise WorktreeDirLockedError(
                record.id, killed=record.killed_pids, kill_attempted=kill_attempted
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
            raise InvalidRepoError(repo_root, "repo_root must be a non-empty path")
        path = Path(repo_root).expanduser().resolve()
        if not path.exists():
            raise InvalidRepoError(repo_root, f"repo_root does not exist: {path}")
        proc = _run_git(["rev-parse", "--show-toplevel"], cwd=path)
        if proc.returncode != 0:
            raise InvalidRepoError(repo_root, f"not a git repository: {path}")
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
    "DirtyWorktreeError",
    "DuplicateWorktreeError",
    "GitCommandError",
    "GitTimeoutError",
    "InvalidBranchError",
    "InvalidRepoError",
    "ManagerConfig",
    "UnknownVariantError",
    "WorktreeDirLockedError",
    "WorktreeError",
    "WorktreeManager",
    "WorktreeNotFoundError",
    "PortAllocationError",
    "ProcessAlreadyRunningError",
    "ProcessLifecycleError",
    "ProcessNotRunningError",
)
