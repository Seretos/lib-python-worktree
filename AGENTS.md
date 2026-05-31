# lib-python-worktree -- agent guide

Git-worktree lifecycle + contract engine, being extracted from the
`agent-worktree` MCP plugin. README.md covers what it does and how to use it;
`pyproject.toml` and `.github/workflows/` are the source of truth for structure,
testing, and release. This file records only the non-obvious invariants a
contributor must not silently break.

## Tool priority

Skills and MCP tools take priority over raw file tools — and this **explicitly overrides** the generic harness default that says "prefer the dedicated file/search tools (Glob/Grep/Read)". When a skill or MCP tool covers the task, reach for it first; fall back to raw Glob/Grep/Read only when none applies.

Concretely: any *"where is X defined / what does the code support / which Y exist / how does X work / find the callers of X"* question is a **code-understanding task → use the matching skill first** (e.g. the `serena-wrapper` symbol-aware tools), never raw Glob/Grep/Read.

## Status: implemented

The engine is fully implemented under `src/lib_python_worktree/`. Key modules:

- `core/manager.py` — `WorktreeManager` (public facade: create / list / remove / adopt / prune / start / stop)
- `core/state.py` — `StateStore` protocol + `WorktreeRecord` dataclass
- `core/yaml_store.py` — `YamlStateStore` (file-backed store), `reconcile`, `adopt`, `ReconcileReport`, `AdoptReport`
- `core/port_allocator.py` — `PortAllocator` (locked, atomic read-modify-write against `ports.yaml`)
- `core/process_lifecycle.py` — `start` / `stop` (detached process management, cross-platform)
- `core/_git_utils.py` — `_run_git` (timeout-hardened git subprocess runner)
- `core/_exceptions.py` — `WorktreeError` base hierarchy (`GitTimeoutError`, `DirtyWorktreeError`, etc.)
- `contract/schema.py` — `WorktreeContract`, `Step`, `PortSlot` (Pydantic v2 models)
- `contract/loader.py` — `load()`, `load_text()` (missing/empty file → implicit `isolation: none`)
- `setup/runner.py` — `SetupRunner`, `SetupResult`, `SetupFailedError`

All symbols above except the module-level `start`/`stop` functions from
`core/process_lifecycle.py` are re-exported from `lib_python_worktree.__init__`.
Those two functions are wrapped by `WorktreeManager.start`/`stop` and are not
in `__all__` directly.

## Layering (read before grounding a change)

- The engine lives here, under `src/lib_python_worktree/`. It is registry- and
  filesystem-aware but **MCP-agnostic**: no `mcp` import belongs in this
  package, and engine functions return plain dataclasses (`WorktreeRecord`,
  `AdoptReport`, `ReconcileReport`, `SetupResult`) or Pydantic models
  (`WorktreeContract`).
- The `agent-worktree` plugin is a **separate repo** and only wraps the engine
  as `@mcp.tool()`s (`worktree_create`/`worktree_list`/`worktree_remove`).
  Behaviour, the git subprocess handling, the contract data model, and the
  setup runner are changed **here**, not in the plugin. The MCP tool docstrings
  (the LLM-facing descriptions) live in the plugin.

## Repo specifics (minimal by design)

- **Language:** Python, src-layout under `src/`, package `lib_python_worktree`.
- **Tests:** `python -m pytest`. Install dev deps with
  `pip install -e ".[test]"`.
- **Branch discipline:** All feature work happens on a feature branch in a git
  worktree, never on `main`. Assume the worktree and branch already exist and
  that you are inside them.
- **AI attribution:** The project-issues MCP automatically prefixes every
  comment and PR body with `#ai-generated`. Never type that prefix yourself.

### Env vars

| Variable | Default | Format | Effect |
|----------|---------|--------|--------|
| `WORKTREE_STORE_ROOT` | `~/agent-worktree-store` | filesystem path | Root directory under which per-repo worktree checkouts are created. |
| `WORKTREE_PORT_RANGE` | `30000-40000` | `"<low>-<high>"` | Inclusive TCP port range from which `PortAllocator` draws ports. |
| `WORKTREE_LOG_ROOT` | `~/.agent-worktree/logs` | filesystem path | Root directory for per-step setup/teardown log files. |
| `WORKTREE_GIT_TIMEOUT_SEC` | `30.0` | float string or `""` | Seconds before a `git` subprocess is killed and `GitTimeoutError` raised. Empty string disables the timeout. |

## Release is pipeline-owned

`release.yml` (manual dispatch, `version=X.Y.Z`) stamps the version in CI, tags
`vX.Y.Z`, force-pushes `release/Nx`, publishes a GitHub Release, then opens a
dependency-update ticket in the consumer (`Seretos/agent-worktree`). Never
hand-bump `version` in `pyproject.toml`.

The ticket step authenticates with the **`WORKTREE_TICKET_TOKEN`** repo secret
(a fine-grained PAT with **Issues: write** on `Seretos/agent-worktree`);
`GITHUB_TOKEN` cannot open cross-repo issues. The step is
`continue-on-error: true`, so a missing or invalid token never blocks the
release itself.

**If the automatic step was skipped or failed**, re-file manually by running
the `open-dep-ticket` workflow (`.github/workflows/ticket.yml`) via "Run
workflow" in GitHub Actions. Supply:

- `version` -- the semver string (no leading `v`), e.g. `0.2.0`.
- `consumers` -- space-separated `owner/repo` targets (default:
  `Seretos/agent-worktree`).

The workflow is idempotent: it checks for an open issue with the exact same
title before creating one, so running it twice is safe.

**Human prerequisite -- `WORKTREE_TICKET_TOKEN`:** create this repository secret
(Settings -> Secrets -> Actions) once before the first release.
