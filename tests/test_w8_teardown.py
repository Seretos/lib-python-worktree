"""Stub test file for W8 (full teardown semantics).

This module is not implemented yet. Tests are skipped until the W8 teardown
hooks land in WorktreeManager._teardown().

Intended coverage targets when implemented:
- Teardown executes contract teardown: steps before git worktree remove.
- Teardown stops any running process (W6 hook) before filesystem cleanup.
- Teardown releases allocated ports (W4 hook) after git remove.
- Teardown failure in a step leaves the worktree in a "teardown_failed" status
  rather than deleting it silently.
- force=True on teardown skips steps and forcibly removes even with changes.
- WorktreeManager.remove() calls _teardown() then state.remove(), in that
  order.
"""

import pytest

pytestmark = pytest.mark.skip(reason="W8 full teardown semantics not implemented yet")


def test_placeholder_w8():
    pass
