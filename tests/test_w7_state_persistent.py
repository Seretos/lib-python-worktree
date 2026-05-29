"""Stub test file for W7 (persistent state store).

This module is not implemented yet. Tests are skipped until the W7 file-backed
StateStore implementation lands.

Intended coverage targets when implemented:
- FileStateStore(path) survives a Python process restart: records added in one
  instance are readable from a new instance pointed at the same file.
- add() / get() / remove() / list() / find_by_branch() behave identically to
  InMemoryStateStore but persist across instances.
- Concurrent writes do not corrupt the backing file (locking / atomic write).
- A corrupt or missing state file is handled gracefully (empty store, not
  crash).
- StateStore protocol compliance: FileStateStore satisfies the same interface
  as InMemoryStateStore.
"""

import pytest

pytestmark = pytest.mark.skip(reason="W7 persistent state store module not implemented yet")


def test_placeholder_w7():
    pass
