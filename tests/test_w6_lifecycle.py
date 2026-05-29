"""Stub test file for W6 (process lifecycle).

This module is not implemented yet. Tests are skipped until the W6 process
lifecycle module lands.

Intended coverage targets when implemented:
- start_process(worktree_id, command) launches the configured command in the
  worktree directory and stores a PID reference.
- stop_process(worktree_id) terminates the process (SIGTERM then SIGKILL after
  timeout).
- Status transitions: created -> running -> stopped / failed.
- Re-starting an already-running worktree raises an appropriate error.
- Process state is reflected in WorktreeRecord.status after create/remove.
"""

import pytest

pytestmark = pytest.mark.skip(reason="W6 process lifecycle module not implemented yet")


def test_placeholder_w6():
    pass
