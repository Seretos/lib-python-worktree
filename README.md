# lib-python-worktree

Git-worktree lifecycle + contract engine for the Seretos agent-plugin
ecosystem. Holds the reusable engine being extracted from the `agent-worktree`
MCP plugin: worktree lifecycle (create / list / remove), the `.seretos/` YAML
setup contract (schema + loader), port allocation, the setup-script runner, and
the pluggable state store.

Extracted from `agent-worktree` so the engine can be reused and unit-tested
independently of the MCP server. The plugin becomes a thin wrapper that exposes
the engine as MCP tools.

## Status

Initial frame -- the package currently exposes only `__version__`. The engine
modules land here in a follow-up migration (driven by the first release
ticket).

## Install

```bash
pip install -e ".[test]"
```

## Usage

```python
import lib_python_worktree

lib_python_worktree.__version__  # '0.1.0'
```

The public engine API will be re-exported from `lib_python_worktree` once the
logic migration lands.

## Release

Releases are pipeline-owned (`.github/workflows/release.yml`, manual dispatch
with `version=X.Y.Z`). See `AGENTS.md` for the release + downstream-ticket
mechanics.
