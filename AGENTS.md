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

- `core/manager.py` — `WorktreeManager` (public facade: create / list / list_repo / remove / adopt / prune / start / stop); `available_variants()` — public helper (promoted from the former private `_available_variants`, ticket #146) computing the variant strings that would resolve against `start()`'s step-selection tiers for a contract's `start:` steps; `_available_variants` remains as a back-compat module-attribute alias
- `core/teardown.py` — the `_teardown`/`remove()` phase engine (ticket #135, rewritten for #154): `_TeardownContext` (`failure_deadline`/`blockers`, ticket #154), `build_context()`, `_target_is_absent(record, force=...)`, `_TEARDOWN_PHASES` (10 phases: `_phase_guard_primary`, `_phase_stop_processes`, `_phase_stop_hook`, `_phase_gate_b_early_dirty`, `_phase_run_teardown_steps`, `_phase_dirt_gate`, `_phase_stage_and_delete`, `_phase_git_prune`, `_phase_final_guard`, `_phase_release_ports`), `run_teardown()`, `_retry_bounded`/`_bare_retry_bounded`, `_diagnose_and_retry`, `_delete_tree_ladder`. Removal is now rename-based (`os.rename(record.path, "<path>.removing")` proves unheldness on Windows and stages the checkout on both platforms) rather than `git worktree remove`-based; a systemwide process scan is only ever consulted as bounded diagnosis after a rename/delete failure, never on the happy path. See "Teardown/remove changes" below before touching this path.
- `core/checkout.py` — `classify_checkout`, `CheckoutInfo`, `primary_id_for` (primary-vs-linked-worktree classification, ticket #84); `EnvironmentEntry`, `RepoListing`, `list_repo` (repo-scoped listing)
- `core/state.py` — `StateStore` protocol + `WorktreeRecord` dataclass (`backing: "worktree" | "primary"`, ticket #84; `stop_detail: Optional[StopDetail]` — machine-readable reason for a `"stop_incomplete"` status, ticket #99; `teardown_ran: bool` — at-most-once-teardown marker persisted before the checkout is staged for removal, so a `force=True` retry after a post-teardown `DirtyWorktreeError` never re-runs `teardown:`, ticket #126; `stop_hook_outcome: Optional[StopHookOutcome]` — stop-hook + contract diagnostics, ticket #128; `base_fetch_fallback: Optional[BaseFetchFallback]` — best-effort `git fetch origin <base>` degrading to the local `base` ref instead of hard-failing `create()`, ticket #134; `start_variants: List[str]` — non-Optional, defaults to `[]`, populated by `create()`/`start()` from `manager.available_variants(contract.start)`, transient/never persisted, ticket #146. `orphan_scan`/`OrphanScanReport`/`OrphanScanEntry` (ticket #140) are removed with no replacement, ticket #154)
- `core/yaml_store.py` — `YamlStateStore` (file-backed store), `reconcile`, `adopt`, `ReconcileReport` (`healed_branches: List[str]`, ticket #139 -- ids of records whose non-ASCII-only-gated mojibake branch `reconcile()`'s best-effort, never-raising branch-healing phase rewrote to git's live value), `AdoptReport`
- `core/port_allocator.py` — `PortAllocator` (locked, atomic read-modify-write against `ports.yaml`)
- `core/process_lifecycle.py` — `start` / `stop` (detached process management, cross-platform); on Windows, `start()` also assigns the spawned process to a Job Object (ticket #95) that `stop()` enumerates/terminates as a ppid-independent containment mechanism (catches `Start-Process`/ShellExecuteEx-delegated grandchildren the ppid-derived process-tree walk cannot reach)
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

### Teardown/remove changes (ticket #135 process rule)

`WorktreeManager._teardown()`/`remove()` had been patched five times in
four days (tickets #117, #123, #126, #127, #130) before ticket #135
extracted the logic into `core/teardown.py` as an explicit, ordered
sequence of named phases (`_TEARDOWN_PHASES`) over a shared
`_TeardownContext`. Ticket #154 replaced the removal mechanism itself
(Windows-only pre-flight scan + `git worktree remove` + all-platform
warn-only orphan scan + long-path fallback → rename-stages-the-checkout
+ a merged delete-and-prune phase, diagnosing via a systemwide scan only
after a rename/delete failure) without touching this extraction's own
discipline -- the phase count changed (11 → 10) but the pattern (named
phases over `_TeardownContext`, documented in
`docs/teardown-phase-contract.md`, pinned by
`tests/test_teardown_matrix.py`) did not. Before changing anything on
this path:

- Read `docs/teardown-phase-contract.md` first -- it names every phase, in
  order, with its invariant, and is kept in sync with
  `teardown._TEARDOWN_PHASES` by `tests/test_teardown_contract_doc.py`.
- `tests/test_teardown_matrix.py` is the **authoritative** consolidated
  regression matrix for this code path -- add any new teardown/remove
  scenario there first, as a parametrized row, not as an ad-hoc one-off
  test elsewhere.
- `core/teardown.py` must never import `core/manager.py` (AST-guarded by
  `tests/test_teardown_phases.py`); it depends only on `_exceptions.py`,
  `_git_utils.py`, `state.py`, `process_lifecycle.py`, and the contract
  loader. `manager.py` imports it as `from . import teardown as
  _teardown_mod` and calls its public functions by module-attribute
  access (`_teardown_mod.build_context(...)`, `_teardown_mod.run_teardown(...)`)
  so a single `patch("lib_python_worktree.core.teardown.<X>")` covers every
  call site of a shared symbol.
- `teardown._logger` is deliberately `logging.getLogger(
  "lib_python_worktree.core.manager")`, not `getLogger(__name__)` -- this
  keeps every `caplog` site (and any downstream consumer) that filters on
  that logger name working across the module split. Do not "fix" this to
  `__name__`.

## Repo specifics (minimal by design)

- **Language:** Python, src-layout under `src/`, package `lib_python_worktree`.
- **Tests:** `python -m pytest`. Install dev deps with
  `pip install -e ".[test]"`. **Never run the whole suite in one go from an
  agent session** -- see "Running the suite" below; it is the single most
  common way an automated run on this repo dies.
- **Branch discipline:** All feature work happens on a feature branch in a git
  worktree, never on `main`. Assume the worktree and branch already exist and
  that you are inside them.
- **AI attribution:** The project-issues MCP automatically prefixes every
  comment and PR body with `#ai-generated`. Never type that prefix yourself.

### Running the suite (agent sessions: read this before you run pytest)

The full suite takes **~567 s for 1162 tests**. That is longer than an agent
session can wait for, and working around it the obvious way is fatal:

> **Never start the full suite as a background task and end your turn waiting
> for it.** In a headless session (`claude -p`, which is how every automated
> run on this repo executes) **ending the turn ends the process**. Nothing
> wakes the session back up: the background task is not suspended, it is
> orphaned along with the run. No error is written -- the session just stops,
> mid-pipeline, with its work uncommitted.

This is not theoretical. On 2026-08-24 five consecutive automated sessions
across tickets #139 and #140 died at exactly this point, each having reached
its own conclusion that it should "wait for the suite and pick up after".
One of them even diagnosed the mechanism correctly and still lost, by moving
the background run from a subagent into its own turn -- the turn still ended.
Raising `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS` does not help (tried at 2 h;
the session died without ever reaching the ceiling), and setting it to `0`
makes it worse -- `0` means *wait zero*, not *wait forever*.

**Instead: run the suite in chunks, synchronously, inside a single turn.** Each
chunk must finish well under the ~600 s ceiling a blocking call has, so target
roughly 3-4 minutes per chunk. Measured chunks:

| chunk | files | tests | time |
|---|---|---|---|
| A | `test_process_lifecycle.py` | 273 | 100 s (measured) |
| B | `test_manager.py`, `test_teardown*.py` | 403 | 345 s (measured) |
| C | everything else under `tests/` | 486 | ~120 s (remainder of the 567 s) |

Chunk B is the one to watch: at 345 s it still fits, but not comfortably once
the machine is busy, and it is worth splitting -- the five slowest cases in it
are all `test_manager.py` remove tests at ~20.5 s each, so `test_manager.py`
alone carries most of that number and `test_teardown*.py` is cheap.

Note that test *count* does not predict runtime here: chunk A holds the most
tests but is the second cheapest, because the expensive tests are the few that
spawn real processes and enumerate the Windows handle table (a single
`TestWindowsJobObjectContainment` case costs ~30 s). When you add tests, check
your chunk with `--durations=10` rather than assuming.

Run the chunks one after another in the same turn, and treat a chunk boundary
as a good place to commit. **Commit and push before any turn that might end**
-- the cost of the mechanism above is not the lost minutes, it is the
uncommitted work that goes with them.

CI runs the suite as a whole; chunking is a constraint on *agent sessions*
only, not on the workflow.

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

Each bump ticket's body now carries a `### What changed` section populated
from `gh release view`'s changelog, degrading to a `::warning::` plus a
release-page link (and truncating at 30000 characters) when the fetch fails
or returns nothing. The board-add step also resolves the `Status` field's
`Backlog` option id at runtime and moves the card there instead of leaving it
in automation's default `Todo` column, degrading to a `::warning::` (card
left in its default column) if the field/option lookup or the move itself
fails.

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
