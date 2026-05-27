"""Contract module: parser + schema for the worktree contract at
``<repo-root>/.seretos/worktree-setup.yml``.

Public surface:
- ``WorktreeContract`` — the validated top-level model.
- ``load`` / ``load_text`` — file/string loaders.
- ``CONTRACT_FILENAME`` — relative path below the repo root.
- ``ContractError`` / ``ContractValidationError`` — typed errors.
"""

from __future__ import annotations

from .loader import (
    CONTRACT_FILENAME,
    ContractError,
    ContractValidationError,
    load,
    load_text,
)
from .schema import (
    Isolation,
    PortSlot,
    Step,
    WorktreeContract,
)

__all__ = (
    "CONTRACT_FILENAME",
    "ContractError",
    "ContractValidationError",
    "Isolation",
    "PortSlot",
    "Step",
    "WorktreeContract",
    "load",
    "load_text",
)
