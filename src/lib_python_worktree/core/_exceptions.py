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


class UnknownVariantError(WorktreeError, ValueError):
    """Raised when ``WorktreeManager.start()`` is given a ``variant`` that
    does not match any ``start:`` step name in the contract.

    Ticket #70: the original code raised a plain ``WorktreeError``
    (``RuntimeError``-based) here even though ``start()``'s docstring
    documents a ``ValueError`` contract for unknown variants. This class is
    both, so callers catching either base keep working.

    Both ``variant`` (the requested, unmatched name) and ``available`` (the
    list of step names that *do* exist) are stored as attributes so callers
    can react programmatically without parsing the message text.
    """

    def __init__(self, variant: str, available: "List[str]") -> None:
        super().__init__(
            f"no start: step named '{variant}' found in contract "
            f"(available: {available})"
        )
        self.variant = variant
        self.available = available


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
    "GitTimeoutError",
    "InvalidRepoError",
    "PrimaryCheckoutError",
    "UnknownVariantError",
    "WorktreeDirLockedError",
    "WorktreeError",
]
