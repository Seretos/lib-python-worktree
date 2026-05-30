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

from typing import List


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


__all__ = [
    "DirtyWorktreeError",
    "GitTimeoutError",
    "WorktreeError",
]
