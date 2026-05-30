"""lib-python-worktree -- git-worktree lifecycle + contract engine.

The reusable engine extracted from the `agent-worktree` MCP plugin:
git-worktree lifecycle (create/list/remove), the `.seretos/` YAML setup
contract (schema + loader), the setup-script runner, and the pluggable
state store. Port allocation, process lifecycle, and full teardown semantics
hook in around these seams in later phases.

Primary entry point for consumers:

    >>> from lib_python_worktree import WorktreeManager
    >>> m = WorktreeManager()
    >>> rec = m.create("/path/to/repo", "feature/x", base="main")
    >>> m.list()
    >>> m.remove(rec.id)

The package is **MCP-agnostic**: no `mcp` import belongs here. The
`agent-worktree` plugin is a separate repo that wraps `WorktreeManager` (and
the contract/setup pieces) as `@mcp.tool()`s -- behaviour and the data model
live here, the MCP/stdio glue lives in the plugin.
"""
from __future__ import annotations

from .contract import (
    CONTRACT_FILENAME,
    ContractError,
    ContractValidationError,
    Isolation,
    PortSlot,
    Step,
    WorktreeContract,
    load,
    load_text,
)
from .core.manager import (
    BranchAlreadyCheckedOutError,
    BranchNotFoundError,
    DirtyWorktreeError,
    DuplicateWorktreeError,
    GitCommandError,
    GitTimeoutError,
    ManagerConfig,
    PortAllocationError,
    WorktreeError,
    WorktreeManager,
    WorktreeNotFoundError,
)
from .core.process_lifecycle import (
    ProcessAlreadyRunningError,
    ProcessLifecycleError,
    ProcessNotRunningError,
)
from .core.port_allocator import PortAllocator
from .core.state import InMemoryStateStore, StateStore, WorktreeRecord
from .core.yaml_store import AdoptReport, ReconcileReport, YamlStateStore, adopt, reconcile
from .setup import (
    SetupFailedError,
    SetupResult,
    SetupRunner,
    SetupStep,
    SetupStepResult,
    log_dir_for,
)

__version__ = "0.1.0"

__all__ = [
    # core / manager
    "WorktreeManager",
    "ManagerConfig",
    "WorktreeError",
    "BranchNotFoundError",
    "BranchAlreadyCheckedOutError",
    "DirtyWorktreeError",
    "DuplicateWorktreeError",
    "WorktreeNotFoundError",
    "GitCommandError",
    "GitTimeoutError",
    # process lifecycle
    "ProcessLifecycleError",
    "ProcessAlreadyRunningError",
    "ProcessNotRunningError",
    # port allocator
    "PortAllocator",
    "PortAllocationError",
    # state
    "WorktreeRecord",
    "StateStore",
    "InMemoryStateStore",
    # yaml state store (W7)
    "YamlStateStore",
    "ReconcileReport",
    "reconcile",
    "AdoptReport",
    "adopt",
    # contract
    "WorktreeContract",
    "Step",
    "PortSlot",
    "Isolation",
    "load",
    "load_text",
    "CONTRACT_FILENAME",
    "ContractError",
    "ContractValidationError",
    # setup runner
    "SetupRunner",
    "SetupResult",
    "SetupStep",
    "SetupStepResult",
    "SetupFailedError",
    "log_dir_for",
    "__version__",
]
