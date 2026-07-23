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
- `core/plugin_install.py` — `install_enabled_plugins` (CLI-driven `enabledPlugins` install; primary mechanism, with `plugin_seed` as fallback)
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
| `WORKTREE_PLUGIN_INSTALL_TIMEOUT_SEC` | `60.0` | float string or `""` | Seconds before a `claude plugin install` subprocess is killed. Empty string disables the timeout. |
| `WORKTREE_SETUP_TIMEOUT_SEC` | `300.0` | float string or `""` | Seconds before a setup/teardown step subprocess is killed and SetupFailedError raised. Empty string disables the timeout. |
| `WORKTREE_SETUP_LOWER_PRIORITY` | `true` | unset or any value; `"0"`/`"false"`/`"no"`/`"off"`/empty (case-insensitive, whitespace-stripped) disable it, any other value enables it | Lowers OS scheduling + I/O priority of setup-step subprocesses spawned by `SetupRunner`, so a heavy step doesn't starve unrelated concurrent work in the calling application. |
| `WORKTREE_ROBOCOPY_TIMEOUT_SEC` | `30.0` | float string or `""` | Seconds before the Windows long-path `robocopy` fallback subprocess (used when `_teardown`'s extended-path `shutil.rmtree` fails) is killed, falling through to `WorktreeDirLockedError`. Empty string disables the timeout. |

## Release is pipeline-owned

`release.yml` (manual dispatch, `version=X.Y.Z`) stamps the version in CI, tags
`vX.Y.Z`, force-pushes `release/Nx`, publishes a GitHub Release, then opens a
`chore(deps): bump lib-python-worktree to vX.Y.Z` issue in **both**
`Seretos/agent-worktree` and `Seretos/workboard`. Never hand-bump `version` in
`pyproject.toml`.

Each consumer has its own dedicated ticket step with `continue-on-error: true`,
so a broken or missing token for one consumer never blocks the other or the
release itself.

- **`WORKTREE_TICKET_TOKEN`** — classic PAT with the **`repo`** scope
  (Issues: write on `Seretos/agent-worktree`) **and** the **`project`**
  scope. Used for the agent-worktree ticket step and its board-add follow-up.
- **`WORKBOARD_TICKET_TOKEN`** — classic PAT with the **`repo`** scope
  (Issues: write on `Seretos/workboard`) **and** the **`project`** scope.
  Used for the workboard ticket step and its board-add follow-up.

`GITHUB_TOKEN` cannot open cross-repo issues, so both PATs are required.
Fine-grained PATs cannot be used here — they have no "Projects" permission
at all, a hard GitHub platform limitation, not a setting to look for harder
in the UI; only classic PATs (Tokens (classic)) expose the `project` scope.

Right after filing (or finding) each ticket, a follow-up step adds it to the
`users/Seretos/projects/2` board via `gh project item-add`, reusing that
consumer's own ticket token — no separate board secret. Each per-consumer
classic PAT above carries both `repo` and `project` scopes, so it covers both
its ticket step and its board-add. Missing `project` scope → the board-add is
skipped or logged as a `::warning::`, never fails the run — the ticket itself
still opens normally.

**If the automatic step was skipped or failed**, re-file manually by running
the `open-dep-ticket` workflow (`.github/workflows/ticket.yml`) via "Run
workflow" in GitHub Actions. Supply:

- `version` -- the semver string (no leading `v`), e.g. `0.2.0`.
- `consumers` -- space-separated `owner/repo` targets (default:
  `Seretos/agent-worktree Seretos/workboard`).

The workflow is idempotent: it checks for an open issue with the exact same
title before creating one, so running it twice is safe. It selects the correct
token per consumer automatically and marks the run red if any consumer fails,
naming the offending consumer in the error output.

**Human prerequisite -- `WORKTREE_TICKET_TOKEN`:** create this repository secret
(Settings -> Secrets -> Actions) once before the first release. Generate a
**classic PAT** (Settings -> Developer settings -> Personal access tokens ->
**Tokens (classic)**) with the `repo` scope (Issues: write on
`Seretos/agent-worktree`) and the `project` scope so the same token can add
the ticket to project board 2.

**Human prerequisite -- `WORKBOARD_TICKET_TOKEN`:** create this repository secret
(Settings -> Secrets -> Actions) once before the first release. Generate a
**classic PAT** with the `repo` scope (Issues: write on `Seretos/workboard`)
and the `project` scope so the same token can add the ticket to project
board 2.
