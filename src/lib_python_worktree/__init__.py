"""lib-python-worktree -- git-worktree lifecycle + contract engine.

The reusable engine being extracted from the `agent-worktree` MCP plugin:
git-worktree lifecycle (create/list/remove), the `.seretos/` YAML setup
contract (schema + loader), port allocation, the setup-script runner, and
the pluggable state store.

This is the initial empty frame. The engine modules and their public
re-exports land here in a follow-up migration; until then the package
exposes only its version.

    >>> import lib_python_worktree
    >>> lib_python_worktree.__version__
    '0.1.0'

The `agent-worktree` plugin is a separate repo that wraps this engine as
`@mcp.tool()`s -- behaviour and the data model live here, the MCP/stdio
glue lives in the plugin.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
