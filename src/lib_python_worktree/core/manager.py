"""Thin wrapper around ``git worktree`` plus canonical id allocation.

W2 keeps this module strictly mechanical: ``subprocess`` calls to ``git`` and
the in-memory state store from ``state.py``. Setup-script execution (W5),
port allocation (W4), process lifecycle (W6) and full teardown semantics (W8)
will hook in around ``WorktreeManager`` later — the seams are documented at
``_teardown`` and ``create`` so future phases know where to inject.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..contract.schema import Step

# `subprocess` is kept for CompletedProcess / DEVNULL references inside this
# module even though _run_git now lives in _git_utils.

from ..contract.loader import (
    CONTRACT_FILENAME,
    ContractError,
    ContractValidationError,
    load as _load_contract,
)
from ._env_utils import _get_user_profile_env
from ._exceptions import CheckoutTargetError, DirtyWorktreeError, GitCommandError, GitTimeoutError, InvalidRepoError, PrimaryCheckoutError, UnknownVariantError, VariantResolutionError, WorktreeDirLockedError, WorktreeError, WorktreeRemovalBlockedError  # noqa: F401 — re-exported
from ._git_utils import _resolve_git_timeout, _run_git  # noqa: F401 — re-exported
from .checkout import (
    CheckoutInfo,
    EnvironmentEntry,
    RepoListing,
    _path_contains,
    classify_checkout,
    list_repo as _list_repo,
    primary_id_for,
    untracked_id_for,
)
from .port_allocator import (
    PinnedPortUnavailableError,
    PortAllocationError,
    PortAllocator,
    _NoOpPortAllocator,
)
from .process_lifecycle import (
    ProcessAlreadyRunningError,
    ProcessLifecycleError,
    ProcessNotRunningError,
    _find_blocking_processes,
    _kill_blocking_processes,
    start as _lifecycle_start,
    stop as _lifecycle_stop,
)
from . import teardown as _teardown_mod
from .state import (
    BASE_FETCH_FALLBACK_REASON_FETCH_FAILED,
    BASE_FETCH_FALLBACK_REASON_FETCH_TIMEOUT,
    SETUP_STATUS_COMPLETED,
    SETUP_STATUS_FAILED,
    SETUP_STATUS_SKIPPED,
    SHADOW_REASON_DIFFERS,
    SHADOW_REASON_UNREADABLE,
    STOP_ATTEMPT_NO_PROCESS_RECORDED,
    STOP_NO_OP_ISOLATION_NONE,
    STOP_NO_OP_NO_PROCESS_RECORDED,
    BaseFetchFallback,
    InMemoryStateStore,
    _BASE_FETCH_STDERR_MAX_CHARS,
    SetupOutcome,
    ShadowedContract,
    StateStore,
    StopAttempt,
    StopHookOutcome,
    WorktreeRecord,
)
from .yaml_store import AdoptReport, YamlStateStore, adopt as _yaml_adopt, reconcile

_logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")
# Matches the synthesised-id suffix minted by checkout.untracked_id_for()
# (ticket #88), so remove()'s id-only path can name the right remedy
# (checkout_path=...) instead of a generic "not found" for this case.
_UNTRACKED_ID_RE = re.compile(r"-untracked-[0-9a-f]{8}$")
_DEFAULT_STORE_ROOT_ENV = "WORKTREE_STORE_ROOT"
_DEFAULT_STORE_DIR_NAME = "agent-worktree-store"
_PORT_RANGE_ENV = "WORKTREE_PORT_RANGE"
_PORT_RANGE_DEFAULT = (30000, 40000)

# record.status value written by create() when a contract's setup: step(s)
# fail (ticket #105). Distinct from the setup_outcome.status values
# (SETUP_STATUS_*, imported above) -- this is the *record*-level status,
# not the nested setup_outcome verdict. Ticket #118: start() gates on this
# constant to refuse silently proceeding past a half-provisioned checkout.
_STATUS_SETUP_FAILED = "setup_failed"

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


class SetupIncompleteError(WorktreeError):
    """Raised by ``start()`` when the target's ``record.status`` is
    ``"setup_failed"`` (ticket #118).

    ``create()`` sets this status when a contract's ``setup:`` step(s) fail,
    leaving the checkout half-provisioned. Without this gate, ``start()``
    would silently overwrite the status to ``"running"`` (a real spawn) or
    ``"ready"`` (the no-op path), erasing the only top-level evidence that
    setup never completed -- the nested, easy-to-miss
    ``record.setup_outcome.status == "failed"`` would be all that remained.

    Deliberately does not read or carry ``record.setup_outcome`` -- callers
    are pointed at ``list()`` / ``environment_list`` for that detail instead,
    keeping this error a pointer, not a duplicate of the detail record.
    """

    def __init__(self, worktree_id: str, status: str) -> None:
        super().__init__(
            f"worktree '{worktree_id}' has status '{status}': its last "
            "setup did not complete. Refusing to start it, to avoid "
            "masking a half-provisioned checkout as healthy. See "
            "list()/environment_list for the setup_outcome detail. "
            "Remedies: re-provision the worktree (recreate it, or re-run "
            "its setup: steps), or call start(..., allow_setup_failed=True) "
            "to start it anyway (this is logged as a warning)."
        )
        self.worktree_id = worktree_id
        self.status = status


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



def _detect_shadowed_contract(
    record: "WorktreeRecord", used_contract: "object"
) -> Optional[ShadowedContract]:
    """Return a :class:`ShadowedContract` when *record*'s checkout carries a
    checkout-local ``.seretos/worktree-setup.yml`` copy that ``start()``
    never reads and that would actually change behaviour (ticket #100).

    ``start()`` always loads the live contract from
    ``<repo_root>/.seretos/worktree-setup.yml`` -- never from a linked
    worktree's own checkout-local copy (the ``agent-worktree`` plugin's
    convenience copy, see ``create()``'s docstring). An agent that edits only
    the checkout-local file gets a clean-looking no-op ``status="ready"``
    with no hint the edit was never read. This flags exactly that footgun.

    Returns ``None`` (no flag) when:
    - *record* is a primary checkout (``backing == "primary"``) or its
      checkout path resolves to the same directory as ``repo_root`` --
      nothing is shadowed because there is no separate checkout-local copy.
    - no ``.seretos/worktree-setup.yml`` exists inside the checkout at all.
    - the checkout-local copy parses to a ``WorktreeContract`` equal to
      *used_contract* -- since the plugin copies ``.seretos/`` into *every*
      checkout, an unconditional flag would fire on every single ``start()``
      and be pure noise; the identical-copy case is not a footgun.

    Returns a ``ShadowedContract`` with ``reason="unreadable"`` when the
    checkout-local copy exists but fails to parse/validate
    (``ContractError``/``ContractValidationError``, or an ``OSError``
    reading it) -- this never raises through ``start()``; only the *used*
    (repo-root) contract's own load failures propagate there. A dangling
    symlink at the checkout-local path is also treated as "unreadable"
    (checked via ``os.path.lexists``, not ``Path.exists``, precisely so a
    broken symlink is not misread as "no checkout-local contract at all").

    Returns a ``ShadowedContract`` with ``reason="differs"`` when the
    checkout-local copy parses cleanly but is a different contract (Pydantic
    model ``==``, not raw bytes, so line-ending/formatting differences alone
    never trigger a false positive).

    The whole body is wrapped so that detection can never make ``start()``
    fail -- same defensive posture as ``_teardown()``'s contract-load blocks.
    """
    try:
        if record.backing == "primary":
            return None
        checkout_path = Path(record.path).resolve()
        repo_root_path = Path(record.repo_root).resolve()
        if checkout_path == repo_root_path:
            return None

        shadow_file = Path(record.path) / CONTRACT_FILENAME
        if not os.path.lexists(shadow_file):
            # lexists (not exists) so a broken symlink still counts as
            # "present" here -- exists() follows symlinks and would report
            # False for a dangling link, which would wrongly read as "no
            # checkout-local contract at all" instead of falling through to
            # the unreadable-contract handling below.
            return None

        used_path = (Path(record.repo_root) / CONTRACT_FILENAME).as_posix()
        shadow_path = shadow_file.as_posix()

        if not shadow_file.exists():
            # Present per lexists() but exists() (which follows symlinks)
            # says otherwise -- a dangling symlink. _load_contract()/loader
            # .load() has its own `exists()` guard that would silently treat
            # this as "no file" (implicit isolation: none) rather than
            # raising, so it is handled explicitly here instead of being
            # handed to _load_contract.
            message = (
                f"start(): checkout-local contract '{shadow_path}' exists "
                f"but could not be read (broken symlink); the contract "
                f"actually used is '{used_path}'."
            )
            return ShadowedContract(
                path=shadow_path,
                used_path=used_path,
                reason=SHADOW_REASON_UNREADABLE,
                message=message,
            )

        try:
            shadow_contract = _load_contract(shadow_file)
        except (ContractError, OSError) as exc:
            # ContractValidationError is a ContractError subclass, so this
            # arm covers both parse and schema failures.
            message = (
                f"start(): checkout-local contract '{shadow_path}' exists "
                f"but could not be read ({exc}); the contract actually used "
                f"is '{used_path}'."
            )
            return ShadowedContract(
                path=shadow_path,
                used_path=used_path,
                reason=SHADOW_REASON_UNREADABLE,
                message=message,
            )

        if shadow_contract == used_contract:
            return None

        message = (
            f"start(): checkout-local contract '{shadow_path}' differs from "
            f"the contract actually used ('{used_path}'); the checkout-local "
            f"copy is never read by start()."
        )
        return ShadowedContract(
            path=shadow_path,
            used_path=used_path,
            reason=SHADOW_REASON_DIFFERS,
            message=message,
        )
    except Exception:  # noqa: BLE001 -- detection must never fail start()
        return None


def _effective_branch(record: "WorktreeRecord") -> str:
    """Return *record*'s effective branch name, read live for a primary.

    A primary ``WorktreeRecord`` never stores its branch (see the ``backing``
    docstring on ``WorktreeRecord``) precisely so it can never go stale
    across a ``git checkout``/branch switch done directly in the main clone
    -- it is read fresh, on every call, via ``git rev-parse --abbrev-ref
    HEAD`` in ``record.path``. A linked worktree's ``record.branch`` is
    always set at creation time and is returned as-is with **no** git call
    at all (a worktree's branch cannot change out from under it the same
    way).

    Detached HEAD reads back as ``"HEAD"`` from ``--abbrev-ref``; that case
    falls back to the short commit sha via ``git rev-parse --short HEAD`` so
    callers still get a meaningful, non-generic value. Any git failure
    (unreadable repo, etc.) returns ``""`` rather than raising -- this value
    only feeds environment variables / contract-step context, never a
    correctness gate.
    """
    if record.branch:
        return record.branch

    proc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=Path(record.path))
    if proc.returncode != 0:
        return ""
    ref = proc.stdout.strip()
    if ref and ref != "HEAD":
        return ref

    # Detached HEAD (or unexpectedly empty output): fall back to short sha.
    short_proc = _run_git(["rev-parse", "--short", "HEAD"], cwd=Path(record.path))
    if short_proc.returncode != 0:
        return ""
    return short_proc.stdout.strip()


def _current_branch(repo_path: Path) -> str:
    """Return the branch currently checked out at *repo_path*, or ``""``.

    Used by ``WorktreeManager.create()`` to default an omitted ``base`` to
    the branch checked out at the main clone. Unlike ``_effective_branch``,
    this value feeds a **correctness gate** (it becomes the actual branch
    start point for a new worktree), not a cosmetic env-var value -- so,
    deliberately, there is no short-SHA fallback for a detached HEAD here.
    ``create()`` treats an empty return as "could not determine a default
    base" and raises ``BranchNotFoundError`` accordingly; this helper never
    decides that policy itself, and never raises on its own account for the
    two *expected* "no default base available" cases (a ``GitTimeoutError``
    from ``_run_git`` still propagates naturally, as always):

    * **Unborn HEAD** (no commits yet) -- ``git rev-parse --abbrev-ref HEAD``
      exits 128 and prints git's standard "ambiguous argument 'HEAD':
      unknown revision or path not in the working tree" message to stderr
      in this case (verified directly: ``git init`` + ``git rev-parse
      --abbrev-ref HEAD`` on this system).
    * **Detached HEAD** -- exits 0 but prints the literal ``"HEAD"``.

    Any *other* non-zero exit (corrupt ``.git/HEAD``, broken refs,
    permission error, ...) is a genuine repo problem, not a "no default
    base" situation -- it is raised as ``GitCommandError`` so it surfaces to
    the caller instead of being silently folded into the "detached or
    unborn HEAD" bucket. Exit code 128 alone is not a precise-enough signal
    for "unborn HEAD" (git also uses 128 for e.g. "not a git repository" or
    corrupt-ref failures), so the unborn-HEAD branch additionally requires
    the stderr text above -- mirroring the stderr-substring pattern used
    elsewhere in this module for disambiguating an exit code that git
    overloads across multiple distinct failures (see the
    ``"is not a working tree"`` / lock-signal checks in ``remove()``).
    Note: in the actual ``create()`` call path, ``_validate_repo()`` /
    ``classify_checkout()`` already runs an equivalent ``git rev-parse
    --git-dir --git-common-dir`` probe before this function is ever
    reached, and it fails identically on a corrupted repo -- so a
    corrupted-repo 128 is normally intercepted upstream as
    ``InvalidRepoError`` and never reaches here. The stderr check below is
    still applied so this function is correct standalone, independent of
    that upstream gate.
    """
    proc = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path)
    if proc.returncode != 0:
        if proc.returncode == 128 and (
            "unknown revision or path not in the working tree" in proc.stderr
        ):
            return ""  # unborn HEAD: no commits yet
        raise GitCommandError(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            proc.returncode,
            proc.stderr,
        )
    ref = (proc.stdout or "").strip()  # `or ""` guards mocked/patched _run_git
    if not ref or ref == "HEAD":
        return ""  # detached HEAD prints "HEAD" at exit 0
    return ref


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
    env["WORKTREE_BRANCH"] = _effective_branch(record)
    for slot, port in record.ports.items():
        env[f"WORKTREE_PORT_{slot.upper()}"] = str(port)
    if caller_env is not None:
        env.update(caller_env)
    return env


def available_variants(steps: "List[Step]") -> "List[str]":
    """Compute the variant strings that WOULD resolve against ``start()``'s
    step-selection block (the tiers at the "Step selection" comment inside
    ``WorktreeManager.start()``, around lines 1601-1622; ticket #131).

    Returns every explicitly-set step ``name`` (tier 1, in contract order)
    plus the literal ``"default"`` when either fallback tier is reachable:
    tier 2 (exactly one step has ``name is None``) or tier 3 (the contract
    has exactly one ``start:`` step total -- ticket #112). Order-preserving
    de-dup, so a step explicitly named ``"default"`` is never listed twice.

    This mirrors the selection tiers for *reporting* purposes only -- it is
    read-only introspection and does not own or alter the real selection
    logic in ``start()``.

    Ticket #146: promoted from the former private ``_available_variants`` to
    a public, package-exported helper -- ``create()``/``start()`` now call it
    directly to populate ``WorktreeRecord.start_variants``, and callers other
    than this module's own ``UnknownVariantError`` raise site need to be able
    to reach it too. ``_available_variants`` remains as a back-compat module
    attribute alias below.
    """
    unnamed_steps = [s for s in steps if s.name is None]
    default_reachable = len(unnamed_steps) == 1 or len(steps) == 1

    candidates = [s.name for s in steps if s.name]
    if default_reachable:
        candidates.append("default")

    seen: "set[str]" = set()
    available: "List[str]" = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            available.append(name)
    return available


# Ticket #146: back-compat alias -- `_available_variants` was the private
# name before this helper was promoted to the public `available_variants`.
_available_variants = available_variants


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
        # Ticket #110: remembered so list() can re-run the same reconcile()
        # gate on every call, not just at construction time -- see list()'s
        # own comment for why.
        self._reconcile_on_init = reconcile_on_init
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
        """Create a new worktree for *branch*, creating the branch if needed.

        If *branch* already exists, a worktree is checked out on it directly
        and *base*/*fetch* are ignored for branch-creation purposes.

        If *branch* does not exist and *base* is given explicitly, the new
        branch is created from *base*. When ``fetch=True`` (the default),
        ``origin/<base>`` is fetched first and used as the start point, so
        the new branch reflects the latest remote tip rather than a possibly
        stale local ref. Pass ``fetch=False`` to branch from the local ref
        without any network call.

        **Best-effort fetch (ticket #134):** when that fetch fails (no
        ``origin`` remote, auth failure, unknown remote ref, DNS/network
        error) or times out (``WORKTREE_GIT_TIMEOUT_SEC``), ``create()`` no
        longer hard-fails outright as long as *base* still exists as a local
        branch: it logs a ``WARNING``, falls back to branching from the
        local ``base`` ref instead of ``origin/<base>``, and records the
        degradation on the returned record's ``base_fetch_fallback``
        (:class:`~.state.BaseFetchFallback`, reason ``"fetch_failed"`` or
        ``"fetch_timeout"``) so a caller can tell the branch may not reflect
        the latest remote tip. ``base_fetch_fallback`` is ``None`` on every
        other path (fetch succeeded, ``fetch=False``, or a defaulted base).
        Still fatal, and never downgraded to a fallback: the local ``base``
        ref not existing either (raises ``GitCommandError``/
        ``GitTimeoutError``, exactly as before this ticket), and a missing
        ``git`` binary (``OSError``/``FileNotFoundError`` -- an environment
        fault, not a remote-reachability degradation).

        If *branch* does not exist and *base* is omitted (``None``), it
        defaults to the branch currently checked out at the main clone (the
        repo resolved from *repo_root* by ``_validate_repo()`` -- not
        whichever linked worktree *repo_root* may point into). This defaulted
        base is always resolved from the **local** ref and never fetched,
        regardless of *fetch*, since HEAD's local ref is already the answer
        -- there is no "stale" remote counterpart to catch up to. Raises
        ``BranchNotFoundError`` if the main clone's HEAD is detached or
        unborn (no commits yet), since no sensible default branch can be
        determined in either case; pass `base` explicitly to create the
        branch in that situation.
        """
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
        base_defaulted = False
        if not branch_exists and base is None:
            base = _current_branch(repo_path)
            if not base:
                raise BranchNotFoundError(
                    f"Branch '{branch}' does not exist in {repo_path} and the "
                    "current branch could not be determined (detached or "
                    "unborn HEAD). Pass `base` to create it."
                )
            base_defaulted = True
        if (
            not branch_exists
            and not base_defaulted
            and base is not None
            and not self._branch_exists(repo_path, base)
        ):
            raise BranchNotFoundError(
                f"Base branch '{base}' does not exist in {repo_path}."
            )

        # A defaulted base (the branch already checked out at the main
        # clone) is resolved locally and never fetched: there is no remote
        # ref to fetch from -- HEAD's already-local ref *is* the freshest
        # answer. An explicit base keeps today's fetch-from-origin behavior.
        fetch_base = fetch and not base_defaulted

        # When creating a new branch from a base and fetch=True, fetch the
        # base branch from origin so the new worktree starts from the latest
        # remote commit rather than a potentially stale local ref.
        #
        # Ticket #134 (Befund 2, Option A -- best-effort fetch with visible
        # degradation): a fetch failure or timeout no longer unconditionally
        # hard-fails create(). Exactly two situations count as "fetch
        # failed" here: a non-zero `git fetch` returncode, and a
        # `GitTimeoutError` raised out of `_run_git`. On either, the local
        # `base` ref is re-verified (closes a TOCTOU window against the
        # branch_exists guard above -- a concurrent branch deletion between
        # that guard and this fetch) -- if it's still there, this degrades to
        # a warning + fallback (fetch_base is flipped to False so the
        # ref-selection below picks the local `base`, not `origin/<base>`);
        # if it's gone, this re-raises exactly as before this ticket. Never
        # caught here: OSError/FileNotFoundError (missing git binary -- an
        # environment fault, not a remote-reachability degradation) and any
        # exception raised outside this single `_run_git(["fetch", ...])`
        # call.
        base_fetch_fallback: Optional[BaseFetchFallback] = None
        if not branch_exists and base is not None and fetch_base:
            try:
                fetch_proc = _run_git(["fetch", "origin", base], cwd=repo_path)
            except GitTimeoutError as _fetch_timeout_exc:
                if not self._branch_exists(repo_path, base):
                    raise
                message = (
                    f"git fetch origin {base} timed out after "
                    f"{_fetch_timeout_exc.elapsed:.1f}s. Falling back to the "
                    f"local base ref '{base}'; branch '{branch}' may not "
                    f"reflect the latest origin/{base} tip."
                )
                _logger.warning("%s", message)
                base_fetch_fallback = BaseFetchFallback(
                    reason=BASE_FETCH_FALLBACK_REASON_FETCH_TIMEOUT,
                    base=base,
                    message=message,
                    elapsed_sec=_fetch_timeout_exc.elapsed,
                )
                fetch_base = False
            else:
                if fetch_proc.returncode != 0:
                    if not self._branch_exists(repo_path, base):
                        raise GitCommandError(
                            ["git", "fetch", "origin", base],
                            fetch_proc.returncode,
                            fetch_proc.stderr,
                        )
                    stderr = fetch_proc.stderr or ""
                    stripped = stderr.strip()
                    first_line = stripped.splitlines()[0] if stripped else ""
                    message = (
                        f"git fetch origin {base} failed (exit "
                        f"{fetch_proc.returncode}): {first_line}. Falling "
                        f"back to the local base ref '{base}'; branch "
                        f"'{branch}' may not reflect the latest "
                        f"origin/{base} tip."
                    )
                    _logger.warning("%s", message)
                    base_fetch_fallback = BaseFetchFallback(
                        reason=BASE_FETCH_FALLBACK_REASON_FETCH_FAILED,
                        base=base,
                        message=message,
                        returncode=fetch_proc.returncode,
                        # Truncated here (not just at YAML serialize/deserialize
                        # time) so the in-process record create() returns --
                        # before any disk round-trip -- already carries a
                        # bounded stderr, matching this dataclass's own
                        # docstring claim (ticket #134 review fix-loop).
                        stderr=stderr[:_BASE_FETCH_STDERR_MAX_CHARS],
                    )
                    fetch_base = False

        worktree_id = f"{repo_slug}-{_slug(branch)}-{_short_uuid()}"
        target_path = self.config.store_root / repo_slug / worktree_id
        target_path.parent.mkdir(parents=True, exist_ok=True)

        git_args = ["worktree", "add"]
        if not branch_exists:
            # When fetch=True use origin/<base> so the new branch starts from
            # the freshly-fetched remote tip, not the (possibly stale) local ref.
            base_ref = f"origin/{base}" if fetch_base else base
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
                pins = {
                    slot.name: slot.port
                    for slot in contract.ports
                    if slot.port is not None
                }
                port_mapping = self._allocator.allocate(
                    slot_names, worktree_id, pinned=pins
                )

            record = WorktreeRecord(
                id=worktree_id,
                repo_root=repo_path.as_posix(),
                branch=branch,
                path=target_path.as_posix(),
                branch_created_by_us=not branch_exists,
                ports=port_mapping,
                base_fetch_fallback=base_fetch_fallback,
                start_variants=available_variants(contract.start),
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
        #
        # Ticket #105: every one of the three branches below (completed /
        # failed / skipped) also records a `SetupOutcome` verdict on
        # `record.setup_outcome` -- a first-class, persisted record of the
        # setup: hook's own outcome, independent of `record.status` (which
        # start()/stop()/reconcile() continuously rewrite for unrelated
        # reasons afterwards).
        _setup_contract = _load_contract(repo_path / CONTRACT_FILENAME)
        if _setup_contract.setup:
            from ..setup.runner import SetupFailedError, SetupRunner  # noqa: PLC0415
            _setup_runner = SetupRunner()
            try:
                _setup_result = _setup_runner.run(
                    setup=_setup_contract.setup,
                    worktree_id=record.id,
                    worktree_path=Path(record.path),
                    branch=_effective_branch(record),
                    port_mapping=record.ports,
                )
            except SetupFailedError as _setup_exc:
                record.status = _STATUS_SETUP_FAILED
                record.setup_outcome = SetupOutcome(
                    status=SETUP_STATUS_FAILED,
                    message=str(_setup_exc),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    failed_step_index=_setup_exc.step_index,
                    failed_step_name=_setup_exc.step_name,
                    log_path=Path(_setup_exc.log_path).as_posix(),
                    returncode=_setup_exc.returncode,
                    timed_out=_setup_exc.timeout is not None,
                )
                self.state.update(record)
                raise
            except Exception as _setup_exc:  # noqa: BLE001
                # Never crash inside this handler; always re-raise the
                # original exception unchanged. Any exception type other
                # than SetupFailedError carries no step-level detail, so the
                # step fields stay at their None/0 defaults.
                record.status = _STATUS_SETUP_FAILED
                record.setup_outcome = SetupOutcome(
                    status=SETUP_STATUS_FAILED,
                    message=f"{type(_setup_exc).__name__}: {_setup_exc}",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
                self.state.update(record)
                raise
            else:
                record.setup_outcome = SetupOutcome(
                    status=SETUP_STATUS_COMPLETED,
                    message=f"setup: completed {len(_setup_result.steps)} step(s)",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    steps_run=len(_setup_result.steps),
                )
        else:
            record.setup_outcome = SetupOutcome(
                status=SETUP_STATUS_SKIPPED,
                message="no setup: steps in contract",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        # Completed and skipped share one write here (the success path
        # previously performed no write at all). Placed BEFORE the
        # best-effort plugin-install block below so a plugin-install hiccup
        # can never lose the setup verdict. The failure path above already
        # performed its own update() inside its except block before
        # re-raising.
        self.state.update(record)

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

    def _list_reconciled(self) -> List[WorktreeRecord]:
        """Return every tracked ``WorktreeRecord``, reconciled first.

        Ticket #110: for a ``YamlStateStore``-backed manager, this first
        re-runs ``reconcile()`` -- exactly as ``__init__`` already does --
        so a caller that only ever polls a listing method repeatedly (never
        re-instantiating the manager) still sees a dead pid cleared /
        ``status`` normalized promptly rather than only at construction
        time. Gated on the same ``reconcile_on_init`` flag the constructor
        was given, so a caller that explicitly opted out of reconcile at
        construction time keeps that opt-out on every call here too.
        A no-op for an ``InMemoryStateStore``-backed manager (``reconcile()``
        only accepts a ``YamlStateStore``). Shared by ``list()`` and
        ``list_repo()`` so both listing paths get the same staleness fix.
        """
        if self._reconcile_on_init and isinstance(self.state, YamlStateStore):
            reconcile(self.state)
        return self.state.list()

    def list(self) -> List[WorktreeRecord]:
        """Return every tracked ``WorktreeRecord``.

        See ``_list_reconciled()`` for the reconcile-before-listing
        behaviour (ticket #110).
        """
        return self._list_reconciled()

    def remove(
        self,
        worktree_id: Optional[str] = None,
        force: bool = False,
        kill_blocking_processes: bool = False,
        *,
        checkout_path: Optional[str] = None,
    ) -> WorktreeRecord:
        """Remove a worktree, addressed by *worktree_id* and/or
        *checkout_path* (ticket #88).

        A *tracked* target (a persisted ``WorktreeRecord`` exists) behaves as
        before: the git worktree checkout is torn down, the state record is
        removed, and an owned branch (``branch_created_by_us``) is deleted.

        An *untracked* target -- a linked worktree that is on disk (per
        ``git worktree list --porcelain``, as ``list_repo()``/
        ``environment_list`` would report it with ``tracked=False``) but has
        no persisted record -- is addressable **only** via *checkout_path*:
        its ``list_repo()``-displayed id (``untracked_id_for(path)``) is not
        a state-store key and cannot be looked up. For such a target this
        method synthesises an ephemeral ``WorktreeRecord`` (mirroring
        ``list_repo()``'s own synthesis, so the id reported back matches the
        id ``environment_list`` displayed), tears down the checkout, and
        returns that record with ``status="removed"`` -- **without ever
        writing to the state store**. Because the synthesised record always
        has ``branch_created_by_us=False``, an orphan's branch is never
        deleted, even with ``force=True``.

        A primary checkout is refused on every path (by id, by
        *checkout_path* on a tracked primary, or by *checkout_path* on a
        never-started primary) with ``PrimaryCheckoutError`` --
        ``force=True`` never bypasses this.

        Raises ``CheckoutTargetError`` if neither argument is given
        (``reason="missing"``), or if both are given and disagree
        (``reason="id_mismatch"``). Raises ``WorktreeNotFoundError`` if
        *worktree_id* does not resolve to any tracked record (its message
        names ``checkout_path=`` as the remedy when the id looks like a
        synthesised untracked id) or if *checkout_path* does not resolve to
        any environment at all.

        A blocked attempt (a confirmed Windows pre-flight blocker, or real
        uncommitted dirt detected *before* teardown runs) is guaranteed
        side-effect-free with respect to the contract's ``teardown:`` steps
        (ticket #117): those steps run only once removal is confirmed to
        proceed, so such a refused attempt never executes them.

        The contract's ``teardown:`` steps run **at most once per logical
        removal** (ticket #126), enforced by Gates A/B plus a persisted
        ``WorktreeRecord.teardown_ran`` marker that is written *before* the
        actual ``git worktree remove`` is attempted. This closes a second
        failure path distinct from the pre-teardown guard above: if
        ``teardown:`` runs, then a *post*-teardown dirty-tree check raises
        ``DirtyWorktreeError`` (e.g. because a teardown step itself wrote a
        file), a caller's ``force=True`` retry -- the error's own suggested
        remedy -- bypasses Gate B but still observes ``teardown_ran=True``
        and skips re-running the steps. Limitation: for an *untracked*
        (ticket #88) removal target, the marker cannot survive the call
        (that path never writes to the state store), so two successive
        untracked ``remove()`` calls on the same checkout may still re-run
        ``teardown:`` -- this is a deliberate, documented limitation, not a
        bug.

        Orphaned records (ticket #127): when the checkout directory was
        already deleted externally before this call (``status="orphaned"``
        per ``reconcile()``), ``force=True`` is **not** required. The
        missing directory is treated as already torn down -- the leftover
        git worktree registration is pruned, the state record is deleted,
        and reserved ports are released, all without ``force``. In that
        same situation, an owned branch that turns out to be unmerged
        (``git branch -d`` refusing it) does **not** raise: the removal
        still completes and returns ``status="removed"``, with a warning
        logged naming the branch and the manual ``git branch -D`` remedy.
        This non-fatal handling applies only when the checkout was actually
        absent; when the checkout is present, an unmerged owned branch still
        raises ``GitCommandError`` exactly as before.

        Ticket #130: the returned record's ``stop_hook_outcome`` (a
        ``state.StopHookOutcome``) now reports the outcome of the contract's
        ``stop:`` hook that ``_teardown()`` runs best-effort at Step 1b --
        including on a ``force=True`` removal of a still-running
        environment, where it previously stayed ``None``. Set on both the
        tracked and untracked (ticket #88) removal paths. ``no_op_reason``
        is always ``None`` on this path (that field only distinguishes
        ``stop()``'s own no-op branch), and ``stop_attempt`` stays whatever
        an earlier ``stop()`` call left it as -- ``_teardown()`` does not
        recompute it (see its own docstring for why).

        Ticket #140: ``force`` and ``kill_blocking_processes`` guard TWO
        DIFFERENT gates, and neither substitutes for the other. Gate A
        (Windows only) refuses on a confirmed blocking process and is
        bypassed only by ``kill_blocking_processes=True`` -- ``force=True``
        alone still raises ``WorktreeDirLockedError``. Gate B refuses on
        real uncommitted dirt and is bypassed only by ``force=True`` --
        ``kill_blocking_processes=True`` alone still raises
        ``DirtyWorktreeError``. Neither gate runs on POSIX at all (Gate A is
        win32-only; POSIX unlinks files even under an open handle, so a
        blocking process there was previously silently orphaned rather than
        refused or reported).

        Ticket #154: replaces the previous ``git worktree remove`` +
        Windows-only pre-flight-scan + all-platform warn-only orphan-scan
        mechanism with a single, identical-on-both-platforms sequence:
        ``os.rename(record.path, "<path>.removing")`` proves unheldness
        (Windows) and stages the checkout (both platforms), then the staged
        tree is deleted and ``git worktree prune --expire=now`` clears the
        registration. A clean, unheld worktree's removal performs zero
        systemwide process/handle scans; scanning happens only as bounded
        diagnosis after a rename or delete failure. The all-platform
        warn-only orphan-scan phase (and the transient
        ``WorktreeRecord.orphan_scan``/``state.OrphanScanReport``/
        ``state.OrphanScanEntry`` surface it populated) is removed with no
        replacement.
        """
        record, tracked = self._resolve_removal_target(worktree_id, checkout_path)
        # Guard 1 (ticket #84): refuse a primary checkout before any
        # teardown/FS work runs -- never honours force=True. A primary
        # checkout IS the repo; deleting it would be catastrophic and
        # structurally different from removing a disposable linked worktree.
        # _teardown() carries the same guard (guard 2) for callers that reach
        # it directly, so this refusal holds even if a mislabelled record
        # (backing="worktree" but path == repo_root) slips through here.
        if record.backing == "primary" or Path(record.path).resolve() == Path(
            record.repo_root
        ).resolve():
            raise PrimaryCheckoutError(record.id)
        # Ticket #127: captured before Phase 1 tears down (and thus changes
        # the on-disk existence of) record.path -- this must reflect whether
        # the checkout was ALREADY gone (the orphaned-record case) walking
        # in, not whatever _teardown() leaves behind afterwards.
        # Ticket #135: routed through the shared teardown._target_is_absent
        # seam so both this probe and _teardown()'s own probe are covered
        # by a single patch target in tests. Ticket #154: passes force=
        # through -- a non-empty `.removing` remnant under force=False is
        # not "target absent" (see _target_is_absent's docstring).
        target_absent = _teardown_mod._target_is_absent(record, force=force)
        # Phase 1: remove the git worktree checkout.  If this raises the
        # directory still exists, so we keep the state record and propagate.
        self._teardown(record, force=force, kill_blocking_processes=kill_blocking_processes)
        # Phase 2: the worktree directory is now gone.  Remove the state
        # record *before* the branch-delete step so that a branch-delete
        # failure (e.g. ``git branch -d`` refusing an unmerged branch when
        # force=False) does not leave a stale orphaned record in the state
        # store. An untracked target (ticket #88) never touched the store in
        # the first place, so there is nothing to remove there.
        if tracked:
            removed = self.state.remove(record.id)
            assert removed is not None  # _resolve_removal_target confirmed it exists
            removed.status = "removed"
            # Copy killed_pids from the in-memory record: YamlStateStore.remove()
            # returns a freshly-deserialized object that never carries killed_pids
            # (the field is transient and not written to state.yaml), so we must
            # propagate it explicitly from the object _teardown mutated.
            removed.killed_pids = record.killed_pids
            # Ticket #130: same transient-field rationale as killed_pids
            # above -- stop_hook_outcome is not serialised to state.yaml, so
            # the fresh object self.state.remove() returns never carries the
            # verdict _teardown() computed and assigned onto the in-memory
            # `record` a moment ago. Copy it forward explicitly.
            removed.stop_hook_outcome = record.stop_hook_outcome
        else:
            removed = record
            removed.status = "removed"
        # Phase 3: delete the owned branch (if any).  May raise GitCommandError
        # (e.g. unmerged + force=False); the record is already gone from state.
        # A synthesised (untracked) record always has branch_created_by_us=False,
        # so an orphan's branch is never deleted here.
        #
        # Ticket #127: on the orphaned-record path (target_absent=True), an
        # unmerged owned branch (the common case for every create(base=...)
        # environment) must not turn an otherwise-successful cleanup into a
        # raised exception after the state record has already been deleted
        # above -- that would leave the caller with neither a usable record
        # nor a clean error to retry against. Log a warning naming the
        # remedy and continue; the checkout is genuinely gone either way.
        #
        # This tolerance is deliberately narrow: it only covers git's actual
        # refusal to delete an unmerged branch ("error: The branch 'X' is
        # not fully merged...", from `git branch -d` without --force), which
        # is the one failure mode #127's plan (Q2) authorizes as non-fatal.
        # Any OTHER GitCommandError from _delete_owned_branch -- lockfile
        # contention ("Unable to create '.git/refs/heads/X.lock'"), the
        # branch being checked out elsewhere ("Cannot delete branch 'X'
        # checked out at ..."), or anything else -- is a real failure and
        # must still propagate even when target_absent is True; swallowing
        # it would silently report status="removed" for a removal that did
        # not actually succeed as claimed.
        # When the checkout was actually present (target_absent=False), this
        # is unchanged from today: re-raise exactly as before.
        try:
            self._delete_owned_branch(record, force=force)
        except GitCommandError as exc:
            if not target_absent or "not fully merged" not in exc.stderr:
                raise
            _logger.warning(
                "remove(): worktree '%s' checkout was already absent "
                "(orphaned record); leaving unmerged branch '%s' in place "
                "rather than failing the removal. Delete it manually if "
                "desired: git branch -D %s",
                record.id, record.branch, record.branch,
            )
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
        worktree_id: Optional[str] = None,
        *,
        checkout_path: Optional[str] = None,
        role: str = "main",
        env: Optional[dict] = None,
        cwd: Optional[str] = None,
        variant: str = "default",
        allow_setup_failed: bool = False,
    ) -> WorktreeRecord:
        """Spawn a detached process for the target environment, using the
        contract's ``start:`` step, and record its PID.

        The target is resolved by ``_resolve_target()`` from *worktree_id*
        and/or *checkout_path* (ticket #84):

        - *worktree_id* alone: today's exact-path lookup, unchanged.
        - *checkout_path* alone: classified via ``classify_checkout()``. A
          primary checkout resolves to its deterministic id
          (``primary_id_for()``) and is **materialised** here (this is the
          only place a primary ``WorktreeRecord`` is ever written) if it has
          never been started before. A linked worktree resolves to whichever
          tracked record's path matches exactly; an untracked one raises
          ``WorktreeNotFoundError`` naming ``adopt()``.
        - Both given: resolved via *checkout_path* as above, then the
          resolved id must equal *worktree_id* or ``CheckoutTargetError`` is
          raised.
        - Neither given: ``CheckoutTargetError``.

        The command is read from the ``start:`` field of the worktree contract
        at ``<repo_root>/.seretos/worktree-setup.yml``.

        *variant* selects which step to run (default ``"default"``):

        - The step whose ``name`` equals *variant* is used, if any.
        - Otherwise, when *variant* is ``"default"``, a two-tier fallback
          applies: (1) if exactly one step has no ``name`` set, that step is
          used (the original backward-compatibility path -- an unnamed step
          always wins over a named sibling); (2) else, if the contract has
          exactly one ``start:`` step total, that step is used regardless of
          whether it is named (ticket #112 -- so a contract with a single
          named step, e.g. ``name: "main"``, works out of the box without
          requiring ``variant="main"``).
        - If no matching step is found (multiple named and/or unnamed steps
          with no exact match), ``UnknownVariantError`` is raised. Its
          ``.available`` lists every variant string that WOULD have resolved
          against this contract -- each step's own ``name`` (tier 1) plus the
          literal ``"default"`` (ticket #131) when either fallback tier above
          would have reached it, even though the specific *variant* passed to
          this call didn't match. ``UnknownVariantError`` is both a
          ``WorktreeError`` and a ``ValueError``, so callers may catch either
          base.

        ``role`` vs ``variant`` (ticket #104)
        --------------------------------------
        *role* is the tracking/addressing key under which the spawned
        process's pid is recorded (``record.pids[role]``); it defaults to
        ``"main"`` **regardless of which** *variant* was requested. *variant*
        only selects which contract ``start:`` step is run. The two are
        independent: two variants started concurrently against the same
        worktree must be given two distinct *role*s, or the second call
        raises ``ProcessAlreadyRunningError`` (a role already has a live pid).
        Whichever *variant* actually started a given *role* is recorded under
        ``record.variants[role]`` -- this is exactly what
        ``WorktreeManager.stop(variant=...)`` resolves against, so a caller
        that started ``variant="web"`` under some role can later stop it
        without separately tracking which role it used. The no-op "ready"
        start below (no ``start:`` step configured) spawns nothing and
        therefore records no variant for *role*. When the ticket #112
        fallback resolves a named step from a bare ``variant="default"``
        call, ``record.variants[role]`` records the resolved step's own
        name (e.g. ``"main"``), not the literal ``"default"`` -- so a
        subsequent ``stop(variant="main")`` resolves it correctly. This is a
        deliberate, documented asymmetry: ``stop(variant="default")`` --
        the same literal the caller passed to ``start()`` -- will **not**
        resolve in this case, because no role is ever recorded under the
        variant ``"default"`` once the fallback substitutes the step's own
        name. Only ``stop(variant=<the step's actual name>)`` finds it.

        When no ``start:`` step is configured at all (missing
        ``.seretos/worktree-setup.yml`` or an empty ``start:`` list), there is
        nothing meaningful to run.  Rather than erroring, this is treated as a
        **no-op start**: no process is spawned, the worktree is marked
        ``status="ready"`` (usable, with no managed process), and the record is
        returned.  This makes "just give me a worktree I can work in" work out
        of the box for simple repos (e.g. dependency-bump chores).  See
        ticket #41.

        Before either the no-op or a real spawn, any contract ``ports:``
        slot the resolved record does not have yet is allocated (ticket #84,
        R9): incremental, so a contract gaining a new slot picks it up on
        the next ``start()`` without disturbing existing slots, and a
        stopped environment's already-allocated ports survive to the next
        start (``stop()``/``_teardown()`` never release them for a still-
        tracked record — see ``reconcile()``'s per-owner rule).

        Delegates to ``process_lifecycle.start`` with ``store=self.state``
        only when a concrete ``start:`` step is selected. ``cwd=None``
        (the default) defaults to ``record.path`` (ticket #81), not the
        caller's own working directory.

        Ticket #118 -- refusing a half-provisioned checkout
        -----------------------------------------------------
        If the resolved record's ``status`` is ``"setup_failed"`` (set by
        ``create()`` when a contract's ``setup:`` step(s) failed), ``start()``
        raises ``SetupIncompleteError`` instead of proceeding -- by default,
        silently overwriting ``record.status`` to ``"running"`` or ``"ready"``
        would erase the only top-level evidence that setup never completed,
        leaving just the easy-to-miss nested
        ``record.setup_outcome.status == "failed"``. The check runs before
        any mutation (port allocation, the no-op ``"ready"`` write, or a real
        spawn), so a refused call is completely side-effect free. Pass
        ``allow_setup_failed=True`` to start it anyway -- this is a one-shot
        override: it is logged as a ``_logger.warning`` each time it is used,
        and once a forced start succeeds, ``process_lifecycle.start`` rewrites
        ``status`` to ``"running"`` (or the no-op path sets ``"ready"``), so a
        *subsequent* ``start()`` call is no longer refused. The override
        acknowledges the failure once per call, not permanently.
        """
        record = self._resolve_target(worktree_id, checkout_path, materialise=True)

        if record.status == _STATUS_SETUP_FAILED:
            if not allow_setup_failed:
                raise SetupIncompleteError(record.id, record.status)
            _logger.warning(
                "start(): worktree '%s' has status 'setup_failed' but "
                "allow_setup_failed=True was passed -- starting it anyway. "
                "Its last setup did not complete; see "
                "list()/environment_list for the setup_outcome detail.",
                record.id,
            )

        contract = _load_contract(Path(record.repo_root) / CONTRACT_FILENAME)

        # Ticket #100: detect a checkout-local contract copy that this
        # start() call never reads (the plugin's convenience copy) and that
        # would actually change behaviour. Computed once, before either
        # return path, and logged immediately so the warning always fires
        # regardless of which branch below returns.
        shadowed_contract = _detect_shadowed_contract(record, contract)
        if shadowed_contract is not None:
            _logger.warning(shadowed_contract.message)

        if contract.ports:
            # A slot is re-allocated when it is entirely missing, OR when it
            # has a pin (ticket #120) that disagrees with the persisted
            # value -- the pin is authoritative, so a stale auto-allocated
            # port (or a stale different pin) is not left behind.
            needs_alloc = [
                s.name
                for s in contract.ports
                if s.name not in record.ports
                or (s.port is not None and record.ports[s.name] != s.port)
            ]
            if needs_alloc:
                pins = {
                    s.name: s.port
                    for s in contract.ports
                    if s.port is not None and s.name in needs_alloc
                }
                allocated = self._allocator.allocate(
                    needs_alloc, record.id, pinned=pins
                )
                record.ports.update(allocated)
                self.state.update(record)

        if not contract.start:
            # No start: step configured — nothing to run.  Treat as a no-op
            # start so worktree creation + start works without a contract:
            # mark the worktree usable and return without spawning a process.
            record.status = "ready"
            record.shadowed_contract = shadowed_contract
            record.start_variants = available_variants(contract.start)
            # Ticket #126: this no-op path never reaches _lifecycle_start
            # (where the equivalent reset lives for a real start), so it
            # needs its own reset -- a restarted environment is a new
            # logical lifecycle and must earn a fresh teardown, regardless
            # of whether the restart spawned a process or not.
            record.teardown_ran = False
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

        # (3) Ticket #112: variant still defaulted and no exact/unnamed match →
        # if the contract has exactly one start: step total, use it regardless
        # of naming. Multi-step contracts (named or unnamed) keep raising
        # UnknownVariantError below, unchanged.
        if step is None and variant == "default" and len(contract.start) == 1:
            step = contract.start[0]

        if step is None:
            available = available_variants(contract.start)
            raise UnknownVariantError(variant, available)

        from ..setup.runner import _build_step_command, _resolve_shell
        cmd = _build_step_command(_resolve_shell(step.shell), step.run)

        result = _lifecycle_start(
            record.id,
            cmd,
            store=self.state,
            role=role,
            variant=step.name or variant,
            env=_build_worktree_env(record, env),
            cwd=cwd if cwd is not None else record.path,
        )
        # _lifecycle_start() hands back a store-loaded record (a fresh
        # deserialisation for YamlStateStore), which never carries this
        # transient field -- assign it here, mirroring how killed_pids is
        # propagated explicitly elsewhere for the same reason.
        result.shadowed_contract = shadowed_contract
        result.start_variants = available_variants(contract.start)
        return result

    def stop(
        self,
        worktree_id: Optional[str] = None,
        *,
        checkout_path: Optional[str] = None,
        role: Optional[str] = None,
        variant: Optional[str] = None,
        timeout: float = 10.0,
        kill_orphans: bool = False,
    ) -> WorktreeRecord:
        """Stop the process recorded under *role* for the target environment.

        The target is resolved by ``_resolve_target()`` exactly as in
        ``start()`` (ticket #84), with ``materialise=False``: stopping an
        environment that was never started has no record to operate on and
        raises ``WorktreeNotFoundError`` — consistent with today's
        unknown-id contract, and it keeps "a primary record is written only
        by the first ``start()``" literally true.

        ``role`` vs ``variant`` (ticket #104)
        --------------------------------------
        *role* defaults to ``"main"`` -- **``role=None`` (the parameter's
        actual default) means "use ``main``"**, not "no role" / a no-op.
        This mirrors ``start()``'s ``role="main"`` default exactly; the
        sentinel exists only so this method can tell "the caller didn't pass
        ``role``" apart from "the caller explicitly passed ``role="main"``"
        when *variant* is also given (see below). Existing callers of
        ``stop(id)`` or ``stop(id, role="web")`` are unaffected -- both
        behave byte-for-byte as before.

        *variant* resolves to the ``role`` that was started with it, via
        ``record.variants`` (populated by ``start(variant=...)``, see that
        method's docstring), so a caller that started ``variant="web"``
        under some role does not need to separately remember which role it
        used:

        - ``variant=None`` (the default): no resolution; *role* alone
          selects the target, defaulting to ``"main"`` as described above.
        - ``variant`` given and it matches no currently-running role's
          recorded variant: ``VariantResolutionError`` is raised (covers an
          unknown/typo'd variant, and a role started before this ticket
          shipped with no recorded variant).
        - ``variant`` given and it matches more than one currently-running
          role: ``VariantResolutionError`` is raised (ambiguous) -- pass
          ``role=`` to disambiguate.
        - ``variant`` given and it matches exactly one role: that role is
          used. If *role* was also given explicitly and disagrees with the
          resolved role, ``VariantResolutionError`` is raised rather than
          silently picking one -- mirroring ``_resolve_target()``'s existing
          "*worktree_id* and *checkout_path*, if both given, must agree"
          rule.

        This resolution happens immediately after ``_resolve_target()`` and
        before the best-effort contract ``stop:`` steps below, so a
        resolution failure raises before anything has been attempted.

        If the contract defines ``stop:`` steps, they are run (best-effort,
        errors are swallowed) before sending the stop signal.

        When *kill_orphans* is ``True``, a cwd/open-file scan is run after
        the primary signal to terminate any orphaned grandchild processes that
        survived because the tracked shell wrapper already exited and they were
        reparented away from it.

        When no process is recorded for the resolved role (e.g. after a
        no-op ``"ready"`` start with no ``start:`` step — ticket #41),
        stopping is a graceful no-op: contract ``stop:`` steps are still run
        best-effort, but no signal is sent and ``ProcessNotRunningError`` is
        *not* raised.  The worktree is marked ``"stopped"`` when no other
        roles remain.

        This is the engine's documented and intentional behavior (ticket
        #41) and this method's return type is fixed: it always returns a
        ``WorktreeRecord`` (or raises ``WorktreeNotFoundError`` for an
        unknown target), never a dict-shaped "soft error" result.
        Any dict-shaped soft-error contract for a never-started role (e.g.
        ``{"error": ..., "code": ...}``) is owned by the MCP wrapper layer in
        the separate ``agent-worktree`` plugin repo, which translates this
        engine's return values/exceptions into whatever shape its tool
        surface promises callers — it is not this engine's concern and is
        not implemented here (see ``AGENTS.md``'s "Layering" section).

        Delegates to ``process_lifecycle.stop`` with ``store=self.state``.
        Ports are deliberately **not** released here — see ``start()``'s
        docstring and ``reconcile()``'s per-owner rule (ticket #84, R9).

        Ticket #99: when the delegated ``process_lifecycle.stop`` call
        reports ``status="stop_incomplete"``, the returned record's
        ``stop_detail`` (a ``state.StopDetail``) carries a machine-readable
        ``reason`` and evidence for why — see that function's own docstring.

        Ticket #128: every returned record also carries ``stop_hook_outcome``
        (a ``state.StopHookOutcome``) describing whether/how the contract's
        ``stop:`` hook ran (``status`` one of ``"completed"``/``"failed"``/
        ``"skipped"``) and the contract diagnostics behind that verdict
        (``contract_found``, ``contract_path``, ``contract_isolation``). On
        the no-op path (no process recorded for the resolved role),
        ``stop_hook_outcome.no_op_reason`` additionally distinguishes an
        ``isolation: none`` contract's "nothing to stop, by design" no-op
        (``"isolation_none"``) from an ordinary "role never started" no-op
        (``"no_process_recorded"``) — both of which otherwise surface
        identically via ``stop_attempt.outcome ==
        "no_process_recorded"``. ``no_op_reason`` is ``None`` whenever this
        call was not a no-op. Deliberately transient, like ``stop_attempt``:
        never persisted to ``state.yaml``.
        """
        record = self._resolve_target(worktree_id, checkout_path, materialise=False)

        # Ticket #104: resolve variant -> role before any side effect (the
        # best-effort contract stop: steps below), so a resolution failure
        # raises cleanly with nothing having been attempted yet. See this
        # method's docstring ("role vs variant") for the full algorithm.
        if variant is None:
            effective_role = role if role is not None else "main"
        else:
            matches = sorted(r for r, v in record.variants.items() if v == variant)
            if not matches or len(matches) > 1:
                running_roles = sorted(record.pids.keys())
                raise VariantResolutionError(
                    variant, matches, requested_role=role, running_roles=running_roles
                )
            resolved = matches[0]
            if role is not None and role != resolved:
                raise VariantResolutionError(
                    variant, matches, requested_role=role, running_roles=sorted(record.pids.keys())
                )
            effective_role = resolved

        # Run contract stop: steps (best-effort — a failure must not prevent
        # the SIGTERM from being sent).
        #
        # Ticket #128: also compute a StopHookOutcome describing whether/how
        # the stop: hook ran and what the contract's isolation actually was.
        # contract_found is deliberately computed via its OWN try/except,
        # independent of the contract-parse try/except immediately below --
        # a filesystem error on the .exists() check must never masquerade as
        # (or mask) a contract parse outcome, or vice versa.
        contract_path = Path(record.repo_root) / CONTRACT_FILENAME
        try:
            contract_found = contract_path.exists()
        except OSError:
            contract_found = False

        contract_isolation: Optional[str] = None
        stop_hook_status = SETUP_STATUS_SKIPPED
        stop_hook_message = "no stop: steps in contract"
        stop_hook_steps_run = 0
        try:
            contract = _load_contract(contract_path)
            contract_isolation = contract.isolation
            if contract.stop:
                from ..setup.runner import SetupRunner
                runner = SetupRunner()
                try:
                    _stop_result = runner.run(
                        setup=contract.stop,
                        worktree_id=record.id,
                        worktree_path=Path(record.path),
                        branch=_effective_branch(record),
                        port_mapping=record.ports,
                    )
                    stop_hook_status = SETUP_STATUS_COMPLETED
                    stop_hook_steps_run = len(_stop_result.steps)
                    stop_hook_message = (
                        f"stop: completed {stop_hook_steps_run} step(s)"
                    )
                except Exception as _stop_exc:  # noqa: BLE001
                    stop_hook_status = SETUP_STATUS_FAILED
                    stop_hook_message = str(_stop_exc)
        except Exception as _contract_exc:  # noqa: BLE001
            stop_hook_status = SETUP_STATUS_FAILED
            stop_hook_message = str(_contract_exc)
            contract_isolation = None

        def _stop_hook_outcome(no_op_reason: Optional[str]) -> StopHookOutcome:
            return StopHookOutcome(
                status=stop_hook_status,
                message=stop_hook_message,
                steps_run=stop_hook_steps_run,
                contract_found=contract_found,
                contract_path=contract_path.as_posix(),
                contract_isolation=contract_isolation,
                no_op_reason=no_op_reason,
            )

        # No process recorded for this role → graceful no-op (symmetric with
        # the no-op "ready" start).  Avoid delegating to _lifecycle_stop, which
        # would raise ProcessNotRunningError.
        if effective_role not in record.pids:
            # Ticket #104: keep the set(variants) <= set(pids) invariant
            # intact even on this no-op path -- a stale variants[role] entry
            # left over from a role that never got a pid recorded (or whose
            # pid entry was already cleared by a previous stop/reconcile)
            # must not survive here.
            record.variants.pop(effective_role, None)
            if not record.pids and record.status not in ("stop_incomplete", "orphaned"):
                # Ticket #95, finding 6: mirror process_lifecycle.stop()'s own
                # guard (see its identical exclusion). "stop_incomplete" and
                # "orphaned" are sticky, deliberately-honest statuses set by
                # an earlier stop()/reconcile() call -- this no-op branch
                # clearing the last pid entry for an unrelated role must not
                # silently overwrite that guarantee back to "stopped".
                record.status = "stopped"
                # Ticket #99: status is actually transitioning to "stopped"
                # here (the guard above already excludes the sticky
                # "stop_incomplete"/"orphaned" cases) -- clear any stale
                # stop_detail in lockstep, mirroring
                # process_lifecycle.stop()'s own clean-"stopped" branch.
                record.stop_detail = None
            # Ticket #110: process_lifecycle.stop() (and therefore every
            # StopAttempt outcome it can set -- "killed"/"already_exited"/
            # "tracked_pid_missing") is never reached on this no-op path --
            # there is no pid to check. Record that explicitly rather than
            # leaving stop_attempt at whatever a previous stop() call left
            # it as (or None, for a role that was never started).
            record.stop_attempt = StopAttempt(
                outcome=STOP_ATTEMPT_NO_PROCESS_RECORDED,
                message=(
                    f"stop(worktree_id={record.id}, role={effective_role}): "
                    f"no process recorded for this role; nothing to stop"
                ),
                role=effective_role,
            )
            # Ticket #128: distinguish an isolation: none contract's "nothing
            # to stop, by design" no-op from an ordinary "role never
            # started" no-op -- both set the same STOP_ATTEMPT_NO_PROCESS_
            # RECORDED above, so this is the only place that tells them
            # apart.
            record.stop_hook_outcome = _stop_hook_outcome(
                STOP_NO_OP_ISOLATION_NONE
                if contract_isolation == "none"
                else STOP_NO_OP_NO_PROCESS_RECORDED
            )
            # Ticket #110 (review round 2, blocking finding 2): mirror
            # process_lifecycle.stop(), which unconditionally recomputes
            # record.killed_pids on every path -- including to an empty
            # list. This no-op branch used to skip that refresh, so a
            # record left with stale killed_pids from an earlier real
            # stop()/start() call would still report those stale pids here
            # even though this call reports "nothing to stop". Clear it so
            # stale data from an older call can't leak forward.
            record.killed_pids = []
            self.state.update(record)
            return record

        result = _lifecycle_stop(
            record.id,
            store=self.state,
            role=effective_role,
            timeout=timeout,
            kill_orphans=kill_orphans,
        )
        # Ticket #128: _lifecycle_stop() hands back a store-loaded record (a
        # fresh deserialisation for YamlStateStore), which never carries this
        # transient field -- assign it onto the returned object, not the
        # pre-call `record` variable, mirroring how shadowed_contract is
        # propagated explicitly in start() for the same reason. This was not
        # a no-op, so no_op_reason stays None.
        result.stop_hook_outcome = _stop_hook_outcome(None)
        return result

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
            branch=_effective_branch(record),
            port_mapping=record.ports,
        )

    def list_repo(self, path: str) -> "RepoListing":
        """Return the repo-scoped listing (primary + linked worktrees) for
        any *path* inside a repo (ticket #84, §8).

        Thin wrapper around the pure ``checkout.list_repo()`` function, fed
        with the current store's records — the actual join/synthesis logic
        lives there (store-free, independently unit-testable) so this method
        stays a two-line, read-only pass-through.

        Ticket #110: the records are fetched via ``_list_reconciled()``, the
        same reconcile-before-listing gate ``list()`` uses, so this
        repo-scoped listing doesn't return a stale ``status: running`` for a
        dead pid either.
        """
        return _list_repo(path, self._list_reconciled())

    def _resolve_target(
        self,
        worktree_id: Optional[str],
        checkout_path: Optional[str],
        *,
        materialise: bool,
    ) -> WorktreeRecord:
        """Resolve a ``start()``/``stop()`` target from *worktree_id* and/or
        *checkout_path* (ticket #84). See the resolution table in the
        ticket's plan; behaviour is exhaustively covered by the
        ``test_target_resolution_*`` tests in
        ``tests/test_primary_environment.py``.

        *materialise*: when ``True`` (only ``start()`` passes this), an
        unresolved primary checkout is created and persisted here — the
        **only** place a primary ``WorktreeRecord`` is ever written.
        ``False`` (``stop()``) never writes a record; an unresolved primary
        raises ``WorktreeNotFoundError`` instead.
        """
        if checkout_path is None:
            if worktree_id is None:
                raise CheckoutTargetError(
                    worktree_id=None, checkout_path=None, reason="missing"
                )
            record = self.state.get(worktree_id)
            if record is None:
                raise WorktreeNotFoundError(
                    f"No worktree tracked with id '{worktree_id}'. "
                    f"Pass checkout_path=... to start the primary checkout."
                )
            return record

        # checkout_path given (worktree_id may or may not also be given).
        # classify_checkout() runs before any state access, per the
        # resolution table's "checkout_path at a non-repo dir" row.
        info = classify_checkout(checkout_path)

        existing_record: Optional[WorktreeRecord] = None
        resolved_id: Optional[str] = None
        if info.backing == "primary":
            resolved_id = primary_id_for(info.repo_root)
            existing_record = self.state.get(resolved_id)
        else:
            # Match on containment (path == rec.path, or path is a
            # subdirectory of rec.path), not exact equality -- mirrors how
            # the primary branch above resolves subdirectories via
            # primary_id_for()/repo_root. Scoped to this repo's records
            # (rec.repo_root == info.repo_root) so a same-prefix path under
            # a different repo's worktree can never match. When several
            # tracked worktrees could contain the path (nested worktrees),
            # prefer the most specific (longest/deepest) match so a nested
            # worktree isn't shadowed by an ancestor.
            target_path = info.checkout_path
            repo_root_str = info.repo_root.as_posix()
            best_rec: Optional[WorktreeRecord] = None
            best_depth = -1
            for rec in self.state.list():
                if rec.backing != "worktree" or rec.repo_root != repo_root_str:
                    continue
                rec_path = Path(rec.path).resolve()
                if rec_path != target_path and not _path_contains(rec_path, target_path):
                    continue
                depth = len(rec_path.parts)
                if depth > best_depth:
                    best_depth = depth
                    best_rec = rec
            if best_rec is not None:
                existing_record = best_rec
                resolved_id = best_rec.id

        if (
            worktree_id is not None
            and resolved_id is not None
            and resolved_id != worktree_id
        ):
            raise CheckoutTargetError(
                worktree_id=worktree_id,
                checkout_path=str(checkout_path),
                resolved_id=resolved_id,
                reason="id_mismatch",
            )

        if existing_record is not None:
            return existing_record

        if info.backing == "primary":
            if not materialise:
                raise WorktreeNotFoundError(
                    f"No primary checkout environment started yet for "
                    f"'{info.repo_root}'. Call start(checkout_path=...) first."
                )
            record = WorktreeRecord(
                id=resolved_id,
                repo_root=info.repo_root.as_posix(),
                branch=None,
                path=info.repo_root.as_posix(),
                backing="primary",
            )
            self.state.add(record)
            return record

        raise WorktreeNotFoundError(
            f"No tracked worktree at '{checkout_path}'. "
            f"Use adopt() to import it first."
        )

    def _resolve_removal_target(
        self,
        worktree_id: Optional[str],
        checkout_path: Optional[str],
    ) -> Tuple[WorktreeRecord, bool]:
        """Resolve a ``remove()`` target from *worktree_id* and/or
        *checkout_path* (ticket #88).

        Returns ``(record, tracked)``. ``tracked=True`` means *record* came
        from the state store, so ``remove()`` must delete it there;
        ``tracked=False`` means *record* was freshly synthesised (never
        persisted) for an untracked linked worktree discovered via
        ``list_repo()``, so ``remove()`` must not touch the store at all.

        By id only (``checkout_path is None``): looked up directly via
        ``self.state.get()``, same as before ticket #88 -- an id can only
        ever resolve to a *tracked* record. Unlike ``start()``/``stop()``'s
        ``_resolve_target()``, this method never materialises anything, so
        it does not need a ``materialise`` flag.

        By *checkout_path* (with or without *worktree_id*): first delegates
        to the unmodified ``_resolve_target(materialise=False)`` -- so #84's
        containment/longest-match rules and its ``id_mismatch`` check apply
        to a tracked target exactly as they do for ``start()``/``stop()``.
        If that raises ``WorktreeNotFoundError`` (the checkout exists but
        has no persisted record -- an untracked linked worktree, or a
        never-started primary), falls back to ``self.list_repo()`` --
        reusing ``list_repo()``'s own id synthesis so the id this method
        (and thus ``remove()``) reports back is, by construction, the same
        id ``environment_list``/``list_repo()`` displayed for that checkout.
        """
        if checkout_path is None:
            if worktree_id is None:
                raise CheckoutTargetError(
                    worktree_id=None, checkout_path=None, reason="missing"
                )
            record = self.state.get(worktree_id)
            if record is None:
                if _UNTRACKED_ID_RE.search(worktree_id):
                    raise WorktreeNotFoundError(
                        f"'{worktree_id}' is a synthesised id for an "
                        f"untracked checkout and is not a state-store key; "
                        f"pass checkout_path=... instead."
                    )
                raise WorktreeNotFoundError(
                    f"No worktree tracked with id '{worktree_id}'"
                )
            return record, True

        try:
            record = self._resolve_target(worktree_id, checkout_path, materialise=False)
            return record, True
        except WorktreeNotFoundError:
            pass

        # Untracked fallback: the checkout exists on disk but has no
        # persisted record. Reuse list_repo()'s own join/synthesis rather
        # than re-implementing it, so the id is guaranteed consistent with
        # what environment_list already showed for this checkout.
        listing = self.list_repo(checkout_path)
        # is_current already implements the same containment rule
        # (checkout_path == entry path, or a subdirectory of it) that
        # _resolve_target() uses for tracked targets above; picking the
        # deepest/most-specific match mirrors _resolve_target()'s own
        # longest-match rule so a nested worktree isn't shadowed.
        candidates = [e for e in listing.entries if e.is_current]
        if not candidates:
            raise WorktreeNotFoundError(
                f"No tracked worktree at '{checkout_path}'. "
                f"Use adopt() to import it first."
            )
        best_entry = max(candidates, key=lambda e: len(Path(e.record.path).parts))

        if best_entry.record.backing == "primary":
            # Improvement over the pre-#88 behaviour: a never-started
            # primary addressed by checkout_path now gets the correct
            # structural refusal instead of a confusing not-found.
            raise PrimaryCheckoutError(best_entry.record.id)

        if worktree_id is not None and worktree_id != best_entry.record.id:
            raise CheckoutTargetError(
                worktree_id=worktree_id,
                checkout_path=str(checkout_path),
                resolved_id=best_entry.record.id,
                reason="id_mismatch",
            )

        return best_entry.record, best_entry.tracked

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

        Thin delegator (ticket #135): builds a ``teardown._TeardownContext``
        from this manager's ``state``/``_allocator`` and the call's
        parameters, then runs the ordered phase sequence in
        ``teardown.run_teardown``. See ``docs/teardown-phase-contract.md``
        for the full phase-by-phase contract (Stop, ``stop:`` hook, Gate A,
        Gate B, ``teardown:``, orphan scan (ticket #140), ``git worktree
        remove``, FS fallback, final guard, port release) and
        ``tests/test_teardown_matrix.py`` for the
        consolidated regression matrix covering the eleven historical
        scenarios (#76, #84, #88, #103, #107, #117, #121, #123, #126, #127,
        #130) this delegation must keep reproducing bit-for-bit.

        ``_lifecycle_module`` is an injection seam for tests; callers should
        leave it as ``None`` (the real ``process_lifecycle`` module is used).
        """
        ctx = _teardown_mod.build_context(
            record,
            force=force,
            kill_blocking_processes=kill_blocking_processes,
            store=self.state,
            allocator=self._allocator,
            lifecycle_module=_lifecycle_module,
        )
        _teardown_mod.run_teardown(ctx)

    def _delete_owned_branch(self, record: WorktreeRecord, *, force: bool) -> None:
        """Delete the branch if we created it (``git worktree add -b``).

        Branches that pre-existed (reuse path, no ``base`` supplied) are left
        untouched. A future ``keep_branch`` parameter on ``remove`` is the
        intended per-call opt-out hook — deferred to a follow-up ticket.

        Raises ``GitCommandError`` if the branch exists but deletion fails
        (e.g. ``git branch -d`` refuses an unmerged branch with force=False).
        Skips silently if the branch is already gone (idempotent).
        """

        if record.backing == "primary" or not record.branch:
            # Defense-in-depth (ticket #84): remove()/_teardown() already
            # refuse a primary record before this could ever be reached, and
            # a primary's branch_created_by_us is never True in practice --
            # this guard exists so _delete_owned_branch() is safe even if
            # called directly (e.g. from a test) against a primary record.
            return
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
        # Ticket #84 (B2): classify_checkout() resolves the MAIN clone's root
        # from any path inside the repo -- including a linked worktree. The
        # previous `git rev-parse --show-toplevel` call resolved *whichever*
        # checkout `path` happened to be inside, so calling create()/adopt()/
        # prune() from within a linked worktree silently targeted that
        # worktree instead of the main clone.
        info = classify_checkout(path)
        return info.repo_root

    def _branch_exists(self, repo_path: Path, branch: str) -> bool:
        proc = _run_git(
            ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
            cwd=repo_path,
        )
        return proc.returncode == 0


__all__ = (
    "BranchAlreadyCheckedOutError",
    "BranchNotFoundError",
    "CheckoutInfo",
    "CheckoutTargetError",
    "DirtyWorktreeError",
    "DuplicateWorktreeError",
    "EnvironmentEntry",
    "GitCommandError",
    "GitTimeoutError",
    "InvalidBranchError",
    "InvalidRepoError",
    "ManagerConfig",
    "PrimaryCheckoutError",
    "RepoListing",
    "SetupIncompleteError",
    "UnknownVariantError",
    "VariantResolutionError",
    "WorktreeDirLockedError",
    "WorktreeError",
    "WorktreeManager",
    "WorktreeNotFoundError",
    "PinnedPortUnavailableError",
    "PortAllocationError",
    "ProcessAlreadyRunningError",
    "ProcessLifecycleError",
    "ProcessNotRunningError",
    "available_variants",
    "classify_checkout",
    "primary_id_for",
    "untracked_id_for",
)
