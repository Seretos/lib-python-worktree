# lib-python-worktree

Git-worktree lifecycle + contract engine for the Seretos agent-plugin
ecosystem. Holds the reusable engine being extracted from the `agent-worktree`
MCP plugin: worktree lifecycle (create / list / remove), the `.seretos/` YAML
setup contract (schema + loader), port allocation, the setup-script runner, and
the pluggable state store.

Extracted from `agent-worktree` so the engine can be reused and unit-tested
independently of the MCP server. The plugin becomes a thin wrapper that exposes
the engine as MCP tools.

## Install

```bash
pip install -e ".[test]"
```

## Usage

```python
from lib_python_worktree import WorktreeManager

# WorktreeManager reads WORKTREE_STORE_ROOT and WORKTREE_PORT_RANGE from the
# environment; both have sensible defaults (see "On-disk layout" below).
m = WorktreeManager()

# Create a worktree for an existing branch:
rec_x = m.create("/path/to/repo", "feature/x")

# Create a worktree and a new branch at the same time:
rec_y = m.create("/path/to/repo", "feature/y", base="main")

# List all tracked worktrees:
records = m.list()

# Spawn and stop a detached process inside a live worktree:
m.start(rec_y.id, ["python", "server.py"])
m.stop(rec_y.id)

# Remove a worktree (pass force=True to remove despite uncommitted changes).
# An untracked `.seretos/` convenience copy left in the checkout does NOT
# require force=True -- see "DirtyWorktreeError" below.
removed = m.remove(rec_y.id)

# Adopt worktrees that exist on disk but are not yet tracked:
report = m.adopt("/path/to/repo")

# Prune stale git worktree metadata:
m.prune("/path/to/repo")
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Public surface  (lib_python_worktree/__init__.py)       │
│  Re-exports everything below; MCP-agnostic boundary.    │
│  No `mcp` import belongs in this package.               │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│  Engine core  (lib_python_worktree/core/)                │
│                                                          │
│  manager.py          — WorktreeManager (public facade)   │
│    └─ state.py       — StateStore protocol + WorktreeRecord│
│    └─ yaml_store.py  — YamlStateStore, reconcile, adopt  │
│    └─ port_allocator.py — PortAllocator                  │
│    └─ process_lifecycle.py — start / stop                │
│    └─ checkout.py    — classify_checkout, list_repo      │
│    └─ _git_utils.py  — _run_git (timeout-hardened)       │
│    └─ _exceptions.py — WorktreeError hierarchy           │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│  Contract + setup  (lib_python_worktree/contract/        │
│                     lib_python_worktree/setup/)          │
│                                                          │
│  contract/schema.py  — WorktreeContract, Step, PortSlot  │
│  contract/loader.py  — load(), load_text()               │
│  setup/runner.py     — SetupRunner, SetupResult          │
└─────────────────────────────────────────────────────────┘
```

`manager.py` imports from `state.py`, `yaml_store.py`, `port_allocator.py`,
`process_lifecycle.py`, `checkout.py`, `_git_utils.py`, and `contract/loader.py`.
`yaml_store.py` imports from `_git_utils.py`; `port_allocator.py` imports from
`yaml_store.py` (sharing `_LOCK_FLAGS`, `_LOCK_TIMEOUT`, `_PortsFile`,
`_PORT_KEY_SEP`, and `_port_in_use`). `checkout.py` imports only
`_exceptions.py`, `_git_utils.py`, and `state.py` -- never `manager.py` -- so
`manager.py` can depend on it without a cycle.
`setup/runner.py` is independent of `core/`; `manager.py` imports it lazily
inside `_teardown` to avoid a circular import.

`reconcile()`'s port-freeing rule (ticket #84): a `ports.yaml` entry is freed
only when its *owning record* (parsed from the `<owner_id>:<slot>` key,
matching `port_allocator`'s `_PORT_KEY_SEP`) no longer exists in
`state.yaml` -- not based on whether the port is currently listening
(`_port_in_use`, still used by `PortAllocator.allocate()`'s own
collision-avoidance check) or whether any PID happens to be alive. A
stopped-but-still-tracked environment's reservation is therefore never
wiped, and it survives to its next `start()`.

## On-disk layout

| Root | Default path | Env var override |
|------|-------------|-----------------|
| Worktree checkouts | `~/agent-worktree-store/<repo-slug>/<id>/` | `WORKTREE_STORE_ROOT` |
| State files (`state.yaml`, `ports.yaml`) | `~/.agent-worktree/` | none (hardcoded) |
| Step logs | `~/.agent-worktree/logs/<id>/setup-NN-<slug>.log` | `WORKTREE_LOG_ROOT` |

The state directory is not overridable via environment variable; pass an
explicit `state_dir` to `YamlStateStore()` in tests.

## Contract schema

The contract file lives at `.seretos/worktree-setup.yml` relative to the repo
root. A missing file or an empty file is treated as an implicit
`isolation: none` contract with no setup, teardown, or ports.

### Top-level fields (`WorktreeContract`)

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `version` | `1` | yes | Must be the integer `1`; load-bearing for future migrations. |
| `isolation` | `full \| partial \| none` | yes | `none` forbids `setup`, `teardown`, and `ports`. |
| `setup` | list of `Step` | no | Setup steps defined in the contract schema; NOT executed by `WorktreeManager.create()` — consumed by external callers or reserved for a future runner phase. |
| `teardown` | list of `Step` | no | Steps run before the worktree directory is removed. |
| `ports` | list of `PortSlot` | no | Named TCP port slots allocated from the configured range. |

### `Step` fields

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `run` | `str` | yes | Shell command to execute (non-empty). |
| `name` | `str` | no | Human label; used in log filenames. |
| `shell` | `bash \| sh \| pwsh \| powershell` | no | Per-step shell override. |

### `PortSlot` fields

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | `str` | yes | Must match `^[a-z][a-z0-9_]{0,31}$`. Names must be unique within the contract. |

### `isolation: none` constraint

When `isolation` is `none`, the `setup`, `teardown`, and `ports` fields must
all be absent or empty. Providing any of them raises `ContractValidationError`.

### Example

```yaml
version: 1
isolation: full
ports:
  - name: api
  - name: db
setup:
  - name: install
    run: pip install -e .
  - name: migrate
    run: python manage.py migrate
    shell: bash
teardown:
  - name: cleanup
    run: docker compose down
```

## Public API

### `WorktreeManager`

```python
WorktreeManager(
    config: Optional[ManagerConfig] = None,
    state: Optional[StateStore] = None,
    *,
    reconcile_on_init: bool = True,
)
```

Defaults to `ManagerConfig.from_env()` and a `YamlStateStore()`. When
`reconcile_on_init=True` and the store is a `YamlStateStore`, runs `reconcile`
at construction to clean up stale records.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `create` | `(repo_root: str, branch: str, base: Optional[str] = None, *, fetch: bool = True)` | `WorktreeRecord` | Add a git worktree, allocate ports per the contract, and persist state. If `branch` doesn't exist, pass `base` to create it from; if `base` is omitted, it defaults to the branch currently checked out at the main clone (local ref only, never fetched). With an explicit `base`, `fetch=True` (default) fetches `origin/<base>` first; `fetch=False` branches from the local ref. Raises `BranchNotFoundError` if `branch` is missing and no `base` is given while the main clone's HEAD is detached or unborn. |
| `list` | `()` | `List[WorktreeRecord]` | Return all tracked worktree records. |
| `list_repo` | `(path: str)` | `RepoListing` | Repo-scoped listing (primary + linked worktrees) for any path inside a repo, joining git's live `worktree list --porcelain` view against tracked records. Entries not yet persisted (the primary before its first `start()`, or a linked worktree never `adopt()`-ed) are synthesised with `tracked=False`. A tracked record whose checkout is no longer registered with git (deregistered outside this tool) is still listed too, as `tracked=True` with `record.status == "orphaned"`. See "Orphan worktree recovery" below for both flavours. |
| `remove` | `(worktree_id: Optional[str] = None, force: bool = False, kill_blocking_processes: bool = False, *, checkout_path: Optional[str] = None)` | `WorktreeRecord` | Run teardown, remove git worktree, release ports, delete state. Target by `worktree_id` **or** `checkout_path`; a `checkout_path` pointing at an untracked/orphaned linked worktree is removed without needing `adopt()` first (see "Orphan worktree recovery" below — `checkout_path` for an untracked orphan, `worktree_id` for a tracked-but-deregistered one). `force=True` removes despite uncommitted changes; never bypasses the primary-checkout refusal. |
| `adopt` | `(repo_root: str)` | `AdoptReport` | Import untracked on-disk worktrees into the state store. Requires `YamlStateStore`. |
| `prune` | `(repo_root: str)` | `None` | Run `git worktree prune --expire=now` to clear stale git metadata. |
| `start` | `(worktree_id: Optional[str] = None, *, checkout_path: Optional[str] = None, role: str = "main", env: Optional[dict] = None, cwd: Optional[str] = None, variant: str = "default")` | `WorktreeRecord` | Resolve the target environment by `worktree_id` or `checkout_path`, then spawn a detached process (per the contract's `start:` step for `variant`) and record its PID under `role`, and records the variant under `record.variants[role]`, which is what `stop(variant=...)` resolves against. |
| `stop` | `(worktree_id: Optional[str] = None, *, checkout_path: Optional[str] = None, role: Optional[str] = None, variant: Optional[str] = None, timeout: float = 10.0, kill_orphans: bool = False)` | `WorktreeRecord` | Gracefully stop the process for `role` (or the role `variant` resolves to); force-kills if it does not exit within `timeout` seconds. `role=None` (the default) means `"main"`, same as `start()`. `variant=` resolves to the role that was started with it via `record.variants`; if `role` is also given, both must agree or `VariantResolutionError` is raised. An unknown or ambiguous `variant` also raises `VariantResolutionError` — see "`role` vs `variant`" below. On `status="stop_incomplete"`, see `stop_detail` below. |

### Orphan worktree recovery

"Orphan" covers two distinct situations, told apart by an entry's `tracked`
and `record.status` fields, and recovered differently:

| Flavour | `tracked` | `record.status` | Recover with |
|---------|-----------|------------------|---------------|
| A — untracked but on disk | `False` | `"created"` | `remove(checkout_path=...)` or `adopt()` |
| B — tracked but deregistered outside this tool | `True` | `"orphaned"` | `remove(worktree_id=...)` |

#### Flavour A — untracked but on disk

A linked worktree that was created outside this tool (by hand, or by a
process that crashed before persisting its record) shows up in
`list_repo()` — and thus in `environment_list` — as an entry with
`tracked=False`. Its `record.id` is a deterministic, location-hashed id
(`untracked_id_for(path)`), but that id is display/correlation only: it is
**not** a state-store key and cannot be passed as `remove(worktree_id=...)`.
Recover (or discard) it via `checkout_path` instead:

```python
listing = manager.list_repo(repo_root)
orphan = next(e for e in listing.entries if not e.tracked)

# Discard it directly -- no adopt() needed:
manager.remove(checkout_path=orphan.record.path, force=True)

# ...or import it into the state store first, if you want to keep it:
manager.adopt(repo_root)
```

`remove(checkout_path=...)` tears the checkout down without ever writing to
the state store for an untracked target, and never deletes its branch
(a synthesised record always has `branch_created_by_us=False`) — even with
`force=True`.

#### Flavour B — tracked but deregistered outside this tool

A worktree this library created, whose git registration is gone — the
checkout may or may not still exist on disk — e.g. someone ran
`git worktree remove --force <path>` directly in a shell, deleted its
`.git/worktrees/<id>` administrative directory directly, or deleted the
checkout directory so its porcelain block reads `prunable` — stays visible
rather than silently vanishing from `list_repo()`/`environment_list`. Its
`WorktreeRecord` survives in
`state.yaml`; `reconcile()` (which runs by default at `WorktreeManager`
construction) marks it `status="orphaned"`, and `list_repo()` applies the
same rule as an in-memory-only read view even when `reconcile()` hasn't run.
The entry has **`tracked=True`**, and its id **is** a real state-store key:

```python
listing = manager.list_repo(repo_root)
orphan = next(
    e for e in listing.entries
    if e.tracked and e.record.status == "orphaned"
)
manager.remove(orphan.record.id, force=True)
```

The discriminator is `record.status` — there is no dedicated boolean field
on `EnvironmentEntry` for this. `record.path` for such an entry may not
exist on disk, so a consumer that stats it directly must tolerate
`FileNotFoundError`. Orphan entries of both flavours are always present in
every `list_repo()` call — there is no opt-in flag to request them.

### `ManagerConfig`

```python
@dataclass
class ManagerConfig:
    store_root: Path          # where worktree checkouts are created
    port_range: tuple         # inclusive (low, high), default (30000, 40000)

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> ManagerConfig: ...
```

`from_env` reads `WORKTREE_STORE_ROOT` (default `~/agent-worktree-store`) and
`WORKTREE_PORT_RANGE` (format `"30000-40000"`, default `(30000, 40000)`).

### Contract loader functions

```python
load(path: Union[str, Path]) -> WorktreeContract
load_text(text: str, *, source: str = "<string>") -> WorktreeContract
```

`load` treats a missing file as `isolation: none`. `load_text` treats an empty
string as `isolation: none`.

### `SetupRunner`

```python
SetupRunner(
    *,
    log_root: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
)

runner.run(
    *,
    setup: Sequence[SetupStep],
    worktree_id: str,
    worktree_path: Path,
    branch: str,
    port_mapping: Optional[Dict[str, int]] = None,
    isolation: str = "full",
) -> SetupResult
```

Runs steps sequentially; raises `SetupFailedError` on the first non-zero exit.

### Exception hierarchy

```
Exception
├── RuntimeError
│   ├── WorktreeError                    (base for all engine errors)
│   │   ├── GitTimeoutError              (git subprocess exceeded WORKTREE_GIT_TIMEOUT_SEC; carries .network, .subcommand)
│   │   ├── DirtyWorktreeError           (remove refused; pass force=True -- an untracked-only `.seretos/` copy is exempt, see below)
│   │   │   └── WorktreeRemovalBlockedError  (ALSO a WorktreeDirLockedError, see below -- multiple inheritance)
│   │   ├── WorktreeDirLockedError       (remove refused; directory locked by another process -- carries .worktree_id, .killed, .kill_attempted)
│   │   │   └── WorktreeRemovalBlockedError  (lock AND real dirt both blocking at once; carries .dirty_paths too -- see "Reporting every blocking condition at once" below)
│   │   ├── BranchNotFoundError
│   │   ├── BranchAlreadyCheckedOutError (carries .branch, .path, .prunable)
│   │   ├── DuplicateWorktreeError
│   │   ├── WorktreeNotFoundError
│   │   ├── GitCommandError              (carries .command, .returncode, .stderr)
│   │   ├── UnknownVariantError          (ALSO a ValueError; start(variant=...) matched no start: step -- carries .variant, .available)
│   │   └── VariantResolutionError       (ALSO a ValueError; stop(variant=...) could not resolve to one role -- carries .variant, .roles, .requested_role)
│   ├── ProcessLifecycleError            (base for process lifecycle errors)
│   │   ├── ProcessAlreadyRunningError   (carries .worktree_id, .role, .pid)
│   │   └── ProcessNotRunningError       (carries .worktree_id, .role)
│   ├── PortAllocationError
│   └── SetupFailedError
└── ContractError                        (base for contract loading errors)
    └── ContractValidationError          (carries .path, .errors)
```

All public exception classes are re-exported from `lib_python_worktree`.

### `DirtyWorktreeError` and the `.seretos/` convenience copy

The `agent-worktree` plugin copies the whole `.seretos/` directory into every
new checkout right after `create()` returns, purely as a convenience so the
checkout has its own readable copy of the contract that created it. That copy
is untracked, and `remove(force=False)` no longer treats its mere presence as
"uncommitted changes": if the checkout's **only** dirt (per `git status`) is
untracked content under `.seretos/`, `_teardown` auto-escalates to
`git worktree remove --force` on the caller's behalf, so a plain
`create()` -> `remove()` cycle with zero real edits succeeds without ever
passing `force=True`. Any other dirt -- an untracked file elsewhere, or a
**modified** (tracked) file anywhere including inside `.seretos/` -- still
raises `DirtyWorktreeError` exactly as before.

This is a deliberate, documented trade-off, and the exemption covers **any**
untracked content under `.seretos/`, not merely the copied contract file --
the plugin copies the whole directory, not a single file, and an agent may
itself drop other untracked material there (notes, a cache directory, ...).
The exemption is unconditional, so all of that untracked content, including
a hand-edit an agent made only to the checkout-local contract copy, is
discarded along with the checkout on `remove()`, without a prompt -- exactly
as an explicit `force=True` would discard it. That is safe precisely because
none of it is read by `start()`/setup in the first place (see "Shadowed
checkout-local contract" below); it holds no engine-authoritative state. The
discard is not silent: `_teardown` emits a `WARNING`-level log line naming
the exact untracked path(s) it discarded, so it is observable after the
fact even though `remove()` does not prompt or fail. The shadowed-contract
warning below is what surfaces the contract-specific case of that divergence
to the agent, much earlier in its workflow than removal time.

### Reporting every blocking condition at once

`remove()` can be blocked by two independent conditions at the same time: an
OS-level directory lock (another process still has a handle open inside the
checkout) and real, uncommitted/untracked changes (`force=False`). Without
special handling, a caller who hits both would have to discover each one via
a separate round-trip -- a plain `remove()` raises `WorktreeDirLockedError`,
a retry with `kill_blocking_processes=True` then raises `DirtyWorktreeError`,
and only a third retry with `force=True` too finally succeeds.

When both conditions are detected at once, `_teardown` instead raises a
single `WorktreeRemovalBlockedError` naming **both** conditions and the flag
that clears each, so one retry with `force=True` and
`kill_blocking_processes=True` suffices. `WorktreeRemovalBlockedError`
inherits from **both** `WorktreeDirLockedError` and `DirtyWorktreeError`, so
existing `except WorktreeDirLockedError:` and `except DirtyWorktreeError:`
call sites keep catching it without any change. It adds one new attribute,
`.dirty_paths` (`List[str]`), on top of `WorktreeDirLockedError`'s
`.worktree_id`, `.killed`, and `.kill_attempted` -- the human-readable
message itself stays free of filesystem paths (it names only the engine-level
flags and the worktree id), so a caller that wants the actual dirty paths
reads `.dirty_paths` directly.

This is raised only when the removal genuinely cannot succeed on the current
attempt for **two or more** reasons at once; a single blocking condition
still raises the corresponding single-condition exception unchanged. On
Windows, the combined check runs as part of the Step 2b pre-flight (before
any process is killed) and at both of `_teardown`'s lock-signal raise sites;
detecting real dirt never kills a blocking process or attempts the
destructive `git worktree remove` for a removal that cannot succeed anyway.
The dirt probe itself is gated on `force=False` (a forced removal can never
be blocked by dirt, so there is nothing to probe or report) and is
memoised per removal attempt, so it never issues more than one extra
`git status` call.

## Cross-platform notes

### Shell auto-detection

When a `Step` does not specify `shell:`, `SetupRunner` picks:

- **Windows:** `powershell.exe -NoProfile -Command`
- **POSIX:** `bash -c`

Per-step overrides (the `shell:` field):

| Value | Command used |
|-------|-------------|
| `bash` | `bash -c` |
| `sh` | `sh -c` |
| `pwsh` | `pwsh -NoProfile -Command` |
| `powershell` | `powershell.exe -NoProfile -Command` |

### Git timeout

`WORKTREE_GIT_TIMEOUT_SEC` controls how long each `git` subprocess may run
before being killed and raising `GitTimeoutError`. Default: `30.0` seconds.
Set to an empty string to disable the timeout entirely (diagnostic use only).

`GitTimeoutError` also carries two programmatic diagnostic attributes,
derived from the timed-out command: `.subcommand` (the git subcommand, e.g.
`"fetch"`, or `None` if it couldn't be determined) and `.network` (`bool`,
`True` for remote-talking subcommands -- `clone`, `fetch`, `ls-remote`,
`pull`, `push`). A network timeout's message also gains a suffix noting it
may be transient and worth retrying. This is a diagnostic signal only --
the engine does not itself retry or change timeout behaviour based on it.

### Setup timeout

`WORKTREE_SETUP_TIMEOUT_SEC` controls how long each `setup:`/`stop:`/
`teardown:`/`seed_postprocess:` step's subprocess may run before being killed
and raising `SetupFailedError`. Default: `300.0` seconds. Set to an empty
string to disable the timeout entirely (diagnostic use only). Precedence:
an explicit `timeout=` kwarg passed to `SetupRunner(...)`/`SetupRunner.run(...)`
wins over this env var, which wins over the built-in default.

### Robocopy timeout

`WORKTREE_ROBOCOPY_TIMEOUT_SEC` controls how long the Windows long-path
`robocopy` fallback in `_teardown` may run (used when the extended-path
`shutil.rmtree` fails to remove a worktree checkout). Default: `30.0`
seconds; the call also passes `/R:1 /W:1` so robocopy itself fails fast
instead of retrying a locked directory for its default of ~347 days. On
timeout, teardown falls through to the existing `WorktreeDirLockedError`
contract rather than hanging. Set to an empty string to disable the timeout
entirely (diagnostic use only).

### Process lifecycle

Process detachment on Windows uses `CREATE_NEW_PROCESS_GROUP` so that
`CTRL_BREAK_EVENT` can be delivered for graceful stop. On POSIX,
`start_new_session=True` is used and `SIGTERM` / `SIGKILL` are used for
graceful and force stops respectively.

### `role` vs `variant`

`role` is the tracking/addressing key a spawned process's pid is recorded
under (`record.pids[role]`); it defaults to `"main"` regardless of which
`variant` was started. `variant` only selects which contract `start:` step
runs. The two are independent — two variants started concurrently against
the same worktree need two distinct `role`s, or the second `start()` call
raises `ProcessAlreadyRunningError`.

Whichever `variant` actually started a given `role` is recorded under
`record.variants[role]` (persisted through `state.yaml`, mirroring
`record.pids`/`record.job_names`: one entry per currently-tracked role, no
entry at all for a role with no known variant). `stop(variant=...)` resolves
against this mapping so a caller that started `variant="web"` under some
role does not have to separately track which role it used. Resolution can
fail three ways, and each raises `VariantResolutionError` (carries
`.variant`, `.roles`, `.requested_role`) before anything is attempted (no
contract `stop:` steps run, no process is signalled):

- **unknown** — no currently-running role was started with that variant
  (`.roles == []`); the message points at `role=` and names the roles that
  *are* running. This also covers a role started before this mapping
  existed (a live pid with no `variants` entry).
- **ambiguous** — more than one currently-running role was started with
  that variant (`.roles` lists every match); pass `role=` to disambiguate.
- **disagreement** — an explicit `role=` was also given and it does not
  match the single role `variant=` resolves to; the pair is never silently
  resolved one way.

`role=None` (`stop()`'s actual default) means `"main"`, exactly like
`start()`'s `role="main"` default — it is not a no-op.

### Stop status and `stop_detail`

When `stop()` cannot confirm that everything it tried to kill actually died,
it reports `status="stop_incomplete"` instead of `"stopped"` and attaches a
`stop_detail` (a `StopDetail`) to the returned `WorktreeRecord` naming why:

| `reason` | Meaning |
|---|---|
| `survivors` | One or more tracked PIDs were still alive after every kill attempt (`survivor_pids`, capped at 32, plus the true `survivor_count`). |
| `tree_truncated` | The descendant-process-tree snapshot hit its node cap, so some descendants were never even examined. |
| `job_member_list_truncated` | Windows-only: the Job Object's member list hit its slot cap. |
| `orphan_scan_incomplete` | `kill_orphans=True` was passed but the orphan scan's own discovery pass was starved before finishing. |

`stop_detail.kill_orphans_may_help` hints whether retrying with
`kill_orphans=True` might resolve it — `False` for `orphan_scan_incomplete`,
since that pass already ran. `stop_detail` is persisted to `state.yaml` and
is cleared as soon as the record's status moves away from
`"stop_incomplete"`.

`stop_detail` does not address *which* process the tracked PID identifies —
PID reuse (trusting a stale PID number without a process-identity check
against its recorded start time) remains a known limitation, out of scope
here; see ticket #87.

### Shadowed checkout-local contract

`start()` always reads the live contract from
`<repo_root>/.seretos/worktree-setup.yml` — never from a linked worktree's
own checkout-local copy (see the `.seretos/` convenience copy note above).
If an agent edits only the checkout-local file, that edit is silently never
read. To surface this footgun, every `start()` call sets a transient
`shadowed_contract` (a `ShadowedContract`, or `None`) on the returned
`WorktreeRecord` whenever a checkout-local copy exists and would actually
change behaviour:

| `reason` | Meaning |
|---|---|
| `differs` | The checkout-local copy parses cleanly but is a different contract from the one actually used. |
| `unreadable` | The checkout-local copy exists but fails to parse/validate — this never raises through `start()` itself; only the *used* (repo-root) contract's own load failures do. |

`shadowed_contract` is `None` when there is no checkout-local copy, when it
is identical to the contract actually used (the plugin copies `.seretos/`
into *every* checkout, so the identical case is not a footgun and would
otherwise be pure per-`start()` noise), or for a primary checkout (which has
no separate checkout-local copy at all). The identical message string is
also logged at `WARNING`. Like `killed_pids` (and unlike the persisted
`stop_detail`), `shadowed_contract` is **not** persisted to `state.yaml` — it
is a live observation recomputed on every `start()` call, not a stored
verdict. That guarantee comes from `YamlStateStore`'s round-trip specifically;
an `InMemoryStateStore`-backed manager stores records by reference, so a
`shadowed_contract` (or `killed_pids`) value can keep showing up on later
`get()`/`list()` calls until the next `start()` recomputes it.

### Setup outcome and `setup_outcome`

`record.status` is continuously rewritten by `create()`/`start()`/`stop()`/
`reconcile()` for unrelated reasons (`"created"`, `"running"`, `"stopped"`,
`"setup_failed"`, ...), so it cannot answer, on its own, "did the contract's
`setup:` hook ever run, and how did it end?" once later calls have moved
`status` on. `create()` also sets `setup_outcome` (a `SetupOutcome`, or
`None`) on the returned/persisted `WorktreeRecord` — written once, by the
`setup:` hook block only, and never touched again by `start`/`stop`/
`reconcile`/`adopt`/`remove`:

| `record.setup_outcome` | Meaning |
|---|---|
| `None` | The `setup:` hook was never reached — the record predates this field, was `adopt()`-ed, or was synthesised by `list_repo()`. |
| `status="skipped"` | `create()` ran and found no `setup:` steps to run (missing contract, empty contract, or an explicit `setup: []`). |
| `status="completed"` | Every `setup:` step succeeded (`steps_run` holds the count). |
| `status="failed"` | A `setup:` step raised. `message` mirrors the exception's `str()`; for a `SetupFailedError` specifically, `failed_step_index`, `failed_step_name`, `log_path`, `returncode`, and `timed_out` are also populated. |

Consumers should read `setup_outcome.status` directly rather than inferring
the setup hook's outcome from `record.status` — the latter is overwritten by
every later lifecycle call and cannot distinguish "setup never ran" from
"setup ran and succeeded" once, say, `stop()` has since set
`status="stopped"`. `setup_outcome` is persisted to `state.yaml` (mirrors
`stop_detail`, unlike the transient `killed_pids`/`shadowed_contract`) — a
legacy record with no `setup_outcome` key deserialises to `None`.

## Release

Releases are pipeline-owned (`.github/workflows/release.yml`, manual dispatch
with `version=X.Y.Z`). See `AGENTS.md` for the release + downstream-ticket
mechanics.
