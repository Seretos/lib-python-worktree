"""Base exception classes for the worktree engine.

Placed in their own module so that both ``_git_utils`` (which raises
``GitTimeoutError``) and ``manager`` (which defines the full exception
hierarchy and catches ``WorktreeError``) can share a single class identity
without creating a circular import.

Import hierarchy:
    _exceptions  ← _git_utils  ← yaml_store
                 ← manager (re-exports everything)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .process_lifecycle import KilledProcessInfo


class WorktreeError(RuntimeError):
    """Base class for all worktree-engine errors surfaced to MCP clients."""


_NETWORK_SUBCOMMANDS = frozenset({"clone", "fetch", "ls-remote", "pull", "push"})
"""Git subcommands that talk to a remote and are worth retrying on timeout.

Deliberately excludes ``remote`` and ``submodule``: both are local in their
common invocations (``git remote -v``, ``git submodule status``), and
flagging them as network would produce a false retry signal.
"""


def _git_subcommand(command: List[str]) -> Optional[str]:
    """Best-effort extraction of the git subcommand from an argv list.

    Skips a leading ``"git"`` token (if present) and any leading global
    options -- ``-c key=val``, ``--no-pager``, ``-C <dir>``, ``--git-dir
    <path>``, or any other ``-``/``--`` prefixed token (consuming its value
    when one is expected, but only when that value is a SEPARATE token --
    an ``=``-joined form like ``--git-dir=/path`` is a single token and
    never consumes the next one) -- then returns the first remaining bare
    token. Returns ``None`` for an empty or all-options command rather than
    raising, so a malformed/empty ``command`` never breaks exception
    construction.
    """
    tokens = list(command)
    if tokens and tokens[0] == "git":
        tokens = tokens[1:]

    # Global options that take a separate (space-delimited) value argument.
    # https://git-scm.com/docs/git#_options
    _takes_value = {
        "-c",
        "-C",
        "--git-dir",
        "--work-tree",
        "--namespace",
        "--super-prefix",
        "--config-env",
    }

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("-"):
            i += 1
            if tok in _takes_value and i < len(tokens):
                i += 1
            continue
        return tok
    return None


class GitTimeoutError(WorktreeError):
    """Raised when a ``git`` subprocess exceeds the configured timeout.

    Ticket #19: the Windows PyInstaller binary was hanging because the spawned
    ``git`` inherited the MCP client's stdin pipe and waited forever for input.
    ``_run_git`` now closes stdin, runs via ``Popen.communicate(timeout=...)``,
    and raises this on overrun so the MCP tool can surface a real error rather
    than blocking the client forever.

    Ticket #102: a single E2E sweep observed one ``git fetch`` timeout that
    didn't reproduce on retry, suggesting transient network contention rather
    than a genuine hang. There was no way for a caller to tell "this timeout
    was on a network operation, retrying may help" from "this timeout was on
    a purely local command, retrying won't help" -- so this now also derives
    (or accepts an explicit override for) a ``.network`` flag and the
    ``.subcommand`` it was derived from, purely as a diagnostic signal. No
    retry/backoff behaviour is implemented here.
    """

    def __init__(
        self,
        command: List[str],
        elapsed: float,
        *,
        network: Optional[bool] = None,
    ) -> None:
        self.subcommand = _git_subcommand(command)
        self.network = (
            network if network is not None else self.subcommand in _NETWORK_SUBCOMMANDS
        )
        message = f"git command timed out after {elapsed:.1f}s: {' '.join(command)}"
        if self.network:
            message += (
                " (network operation -- this may be transient remote/network "
                "contention; retrying may succeed)"
            )
        super().__init__(message)
        self.command = command
        self.elapsed = elapsed


class GitCommandError(WorktreeError):
    """Raised when a ``git`` subprocess exits non-zero for a reason that has
    no more specific engine-level exception.

    Ticket #135: relocated here (from ``core/manager.py``) as the sole
    prerequisite for ``core/teardown.py`` to depend on nothing in
    ``manager.py`` -- ``manager.py`` re-exports this class via its existing
    ``from ._exceptions import (...)  # noqa: F401 -- re-exported`` line, so
    ``manager.GitCommandError`` keeps the same class identity as before.
    """

    def __init__(self, command: List[str], returncode: int, stderr: str) -> None:
        super().__init__(
            f"git command failed (exit {returncode}): {' '.join(command)}\n{stderr.strip()}"
        )
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


class DirtyWorktreeError(WorktreeError):
    """Raised when ``git worktree remove`` refuses because the worktree has
    uncommitted changes and ``force=False`` was passed.

    The message names only the engine-level parameter (``force=True``) and
    the worktree id — no raw git command text, absolute paths, or exit codes
    are surfaced so that callers can react programmatically without parsing
    implementation details.
    """

    def __init__(self, worktree_id: str) -> None:
        super().__init__(
            f"worktree '{worktree_id}' has uncommitted changes. "
            f"Pass force=True to remove it anyway."
        )
        self.worktree_id = worktree_id


class InvalidRepoError(WorktreeError):
    """Raised when ``repo_root`` is not a valid git repository.

    Covers three distinct failure modes surfaced by ``_validate_repo``:
    - empty string passed as ``repo_root``
    - path does not exist on the filesystem
    - path exists but is not a git repository (``git rev-parse`` non-zero)

    Both ``repo_root`` (the raw value passed by the caller) and a ``reason``
    string describing the failure are stored as attributes so callers can
    react programmatically without parsing the message text.
    """

    def __init__(self, repo_root: str, reason: str) -> None:
        super().__init__(f"invalid repo_root {repo_root!r}: {reason}")
        self.repo_root = repo_root
        self.reason = reason


class WorktreeDirLockedError(WorktreeError):
    """Raised when the worktree checkout directory is held by an OS-level
    lock (e.g. a Windows process still has a handle open on a file inside
    it), which ``git worktree remove`` reports distinctly from a dirty
    working tree.

    Two phrasings, selected by ``kill_attempted``:

    - ``kill_attempted=True`` (default): a kill-and-retry cycle was
      attempted (``kill_blocking_processes=True`` was passed) and the
      directory is *still* locked after killing the blocking processes.
      The message names how many processes were killed (``len(killed)``).
    - ``kill_attempted=False``: a lock was detected but the caller did not
      opt into the kill-and-retry remedy (``kill_blocking_processes=False``,
      ticket #72). The message points at ``kill_blocking_processes=True`` as
      the way to retry, without attempting a kill itself.

    In both cases the message names only the worktree id and (when
    applicable) how many processes were killed — no raw paths, exit codes,
    or git command text are surfaced so that callers can react
    programmatically without parsing implementation details.
    """

    def __init__(
        self,
        worktree_id: str,
        killed: "List[KilledProcessInfo]",
        *,
        kill_attempted: bool = True,
    ) -> None:
        if kill_attempted:
            n = len(killed)
            message = (
                f"worktree '{worktree_id}' directory is still locked after killing"
                f" {n} blocking process(es)."
            )
        else:
            message = (
                f"worktree '{worktree_id}' directory is locked by another process. "
                f"Pass kill_blocking_processes=True to kill the blocking process(es) "
                f"and retry."
            )
        super().__init__(message)
        self.worktree_id = worktree_id
        self.killed = killed
        self.kill_attempted = kill_attempted


class WorktreeRemovalBlockedError(WorktreeDirLockedError, DirtyWorktreeError):
    """Raised when TWO OR MORE conditions are simultaneously blocking a
    ``remove()`` call -- currently: an OS-level directory lock AND real
    uncommitted/untracked changes (not merely the benign ``.seretos/``
    convenience copy exempted by ticket #100).

    Ticket #103: without this, a caller who hits both conditions has to
    discover each one via a separate round-trip -- plain ``remove()`` ->
    ``WorktreeDirLockedError`` -> retry with ``kill_blocking_processes=True``
    -> ``DirtyWorktreeError`` -> retry with ``force=True`` too. This
    exception instead names EVERY condition currently blocking removal and
    the flag needed to clear each, in a single raise, so one informed retry
    suffices.

    Deliberately inherits from both ``WorktreeDirLockedError`` and
    ``DirtyWorktreeError`` (MRO: ``WorktreeRemovalBlockedError`` ->
    ``WorktreeDirLockedError`` -> ``DirtyWorktreeError`` -> ``WorktreeError``
    -> ``RuntimeError``; C3-valid since both parents derive only from
    ``WorktreeError``) so existing ``except WorktreeDirLockedError:`` and
    ``except DirtyWorktreeError:`` call sites both keep catching it without
    any change on their part.

    The constructor bypasses both parents' ``__init__`` -- neither accepts
    ``**kwargs`` or forwards cooperatively, so a ``super()`` chain would
    either double-set the message or raise ``TypeError`` -- and instead
    calls ``WorktreeError.__init__`` directly, then sets all four
    structured attributes explicitly: ``worktree_id``, ``killed``,
    ``kill_attempted`` (mirroring ``WorktreeDirLockedError``) plus the new
    ``dirty_paths`` (the non-exempt untracked/modified paths ``git status``
    reported). The human-readable message itself stays free of filesystem
    paths -- it names only engine-level flags and the worktree id; a caller
    that wants the actual paths reads the structured ``dirty_paths``
    attribute.

    Raised only when two or more conditions block removal at once; a
    single blocking condition keeps raising the existing single-condition
    exception (``WorktreeDirLockedError`` or ``DirtyWorktreeError``)
    unchanged.
    """

    def __init__(
        self,
        worktree_id: str,
        killed: "List[KilledProcessInfo]",
        *,
        kill_attempted: bool = False,
        dirty_paths: "Optional[List[str]]" = None,
    ) -> None:
        if kill_attempted:
            n = len(killed)
            message = (
                f"worktree '{worktree_id}' cannot be removed: 2 conditions "
                f"are blocking it. The directory is still locked after "
                f"killing {n} blocking process(es) (retry with "
                f"kill_blocking_processes=True) and it has uncommitted "
                f"changes (pass force=True)."
            )
        else:
            message = (
                f"worktree '{worktree_id}' cannot be removed: 2 conditions "
                f"are blocking it. The directory is locked by another "
                f"process (pass kill_blocking_processes=True) and it has "
                f"uncommitted changes (pass force=True). Pass both to "
                f"remove it in a single retry."
            )
        WorktreeError.__init__(self, message)
        self.worktree_id = worktree_id
        self.killed = list(killed)
        self.kill_attempted = kill_attempted
        self.dirty_paths = list(dirty_paths or [])


class UnknownVariantError(WorktreeError, ValueError):
    """Raised when ``WorktreeManager.start()`` is given a ``variant`` that
    does not match any ``start:`` step name in the contract.

    Ticket #70: the original code raised a plain ``WorktreeError``
    (``RuntimeError``-based) here even though ``start()``'s docstring
    documents a ``ValueError`` contract for unknown variants. This class is
    both, so callers catching either base keep working.

    Both ``variant`` (the requested, unmatched name) and ``available`` are
    stored as attributes so callers can react programmatically without
    parsing the message text. ``available`` is the variant strings that
    would resolve against this contract, including the implicit
    ``"default"`` when a fallback tier makes it reachable (ticket #131) --
    not merely the step names that are literally set. See
    ``manager._available_variants()`` for exactly how it is computed.

    When ``available`` is genuinely empty -- no step name matches and no
    fallback tier is reachable either, so the contract has no addressable
    ``start:`` step at all -- the message appends a short remediation clause
    (ticket #131) so a bare ``available: []`` isn't itself misleading; the
    ``available`` attribute itself stays ``[]`` either way.
    """

    def __init__(self, variant: str, available: "List[str]") -> None:
        message = (
            f"no start: step named '{variant}' found in contract "
            f"(available: {available})"
        )
        if not available:
            message += (
                "; no start: step is addressable -- give the steps distinct "
                "name: values"
            )
        super().__init__(message)
        self.variant = variant
        self.available = available


class VariantResolutionError(WorktreeError, ValueError):
    """Raised when ``WorktreeManager.stop(variant=...)`` cannot uniquely
    resolve *variant* to the ``role`` that started it (ticket #104).

    ``record.variants`` maps ``role -> variant`` for every currently-running
    role (see ``WorktreeRecord.variants``'s docstring). This error covers the
    three ways that lookup can fail to produce a single, unambiguous role:

    * **zero match** -- no currently-running role was started with
      *variant*. ``roles`` is ``[]``. This also covers a role started before
      this ticket shipped (a live pid but no ``variants`` entry) and a typo'd
      variant name; the message points the caller at ``role=`` and names the
      roles that *are* currently running (``running_roles``) so a legacy
      record remains actionable.
    * **multiple match** -- more than one currently-running role was started
      with *variant*. ``roles`` lists every matching role (sorted).
    * **disagreement** -- the caller passed an explicit ``role=`` alongside
      ``variant=`` and they do not agree: *variant* resolves to exactly one
      role, but it is not the one the caller named. ``roles`` is that single
      resolved role (as a one-element list) and ``requested_role`` is the
      role the caller explicitly asked for.

    Callers can distinguish the three modes programmatically via
    ``len(e.roles)`` (``0`` → zero match, ``>1`` → ambiguous) and
    ``e.requested_role`` (non-``None`` with ``len(e.roles) == 1`` →
    disagreement).
    """

    def __init__(
        self,
        variant: str,
        roles: "List[str]",
        *,
        requested_role: "Optional[str]" = None,
        running_roles: "Optional[List[str]]" = None,
    ) -> None:
        self.variant = variant
        self.roles = roles
        self.requested_role = requested_role

        if requested_role is not None and len(roles) == 1:
            message = (
                f"variant='{variant}' resolves to role='{roles[0]}', which "
                f"disagrees with the explicitly requested role="
                f"'{requested_role}' -- when both role and variant are "
                f"given they must agree"
            )
        elif len(roles) > 1:
            message = (
                f"variant='{variant}' is ambiguous: it was used to start "
                f"multiple currently-running roles {roles} -- pass "
                f"role=<one of these> to disambiguate"
            )
        else:
            running = running_roles if running_roles is not None else []
            message = (
                f"no currently-running role was started with variant="
                f"'{variant}' -- pass role=<role name> instead "
                f"(currently-running roles: {running})"
            )
        super().__init__(message)


class PrimaryCheckoutError(WorktreeError):
    """Raised when an operation that would delete a checkout is attempted
    against a primary (main-clone) record.

    Ticket #84: unlike a linked worktree, a primary checkout IS the repo --
    ``remove()`` must refuse it structurally, and this refusal must never be
    bypassable via ``force=True`` (the parameter that overrides every other
    removal safeguard). The message names only the worktree id, not the raw
    path, so callers can react programmatically without parsing implementation
    details.
    """

    def __init__(self, worktree_id: str) -> None:
        super().__init__(
            f"worktree '{worktree_id}' is a primary checkout and cannot be "
            f"removed. Primary checkouts are never deleted, even with "
            f"force=True."
        )
        self.worktree_id = worktree_id


class CheckoutTargetError(WorktreeError, ValueError):
    """Raised by ``start()``/``stop()``/``remove()`` when the ``(worktree_id,
    checkout_path)`` pair does not resolve to a single, unambiguous target.

    Modelled on ``UnknownVariantError``'s dual base (``WorktreeError`` and
    ``ValueError``) so callers catching either base keep working.

    ``reason`` is one of:
    - ``"missing"``: neither ``worktree_id`` nor ``checkout_path`` was given.
    - ``"id_mismatch"``: both were given, but the id resolved from
      ``checkout_path`` (``resolved_id``) does not equal ``worktree_id``.

    Both ``worktree_id``/``checkout_path`` (the raw values passed by the
    caller) and ``resolved_id`` (populated only for ``"id_mismatch"``) are
    stored as attributes so callers can react programmatically without
    parsing the message text.
    """

    def __init__(
        self,
        *,
        worktree_id: "Optional[str]",
        checkout_path: "Optional[str]",
        resolved_id: "Optional[str]" = None,
        reason: str,
    ) -> None:
        if reason == "missing":
            message = (
                "start()/stop()/remove() require either worktree_id or "
                "checkout_path to address a target; neither was given."
            )
        else:
            message = (
                f"checkout_path resolved to id '{resolved_id}' which does "
                f"not match the given worktree_id '{worktree_id}'."
            )
        super().__init__(message)
        self.worktree_id = worktree_id
        self.checkout_path = checkout_path
        self.resolved_id = resolved_id
        self.reason = reason


__all__ = [
    "CheckoutTargetError",
    "DirtyWorktreeError",
    "GitCommandError",
    "GitTimeoutError",
    "InvalidRepoError",
    "PrimaryCheckoutError",
    "UnknownVariantError",
    "VariantResolutionError",
    "WorktreeDirLockedError",
    "WorktreeError",
    "WorktreeRemovalBlockedError",
]
