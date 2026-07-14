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

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .process_lifecycle import KilledProcessInfo


class WorktreeError(RuntimeError):
    """Base class for all worktree-engine errors surfaced to MCP clients."""


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


__all__ = [
    "DirtyWorktreeError",
    "GitTimeoutError",
    "InvalidRepoError",
    "UnknownVariantError",
    "WorktreeDirLockedError",
    "WorktreeError",
]
