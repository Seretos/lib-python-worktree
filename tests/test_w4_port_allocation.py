"""Stub test file for W4 (port allocation).

This module is not implemented yet. Tests are skipped until the W4 module
lands under src/lib_python_worktree/ports/ (or equivalent).

Intended coverage targets when implemented:
- allocate_ports(repo_root, slots) -> dict[str, int]: each slot gets a unique
  port in the configured range.
- Ports are not reused by other live worktrees tracked in state.
- Out-of-range / exhausted pool raises an appropriate error.
- Port release on worktree removal makes the port available again.
- Integration with WorktreeManager.create()'s port_mapping result.
"""

import pytest

pytestmark = pytest.mark.skip(reason="W4 port allocation module not implemented yet")


def test_placeholder_w4():
    pass
