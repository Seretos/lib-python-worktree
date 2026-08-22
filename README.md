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

Both `start()` and `stop()` succeed silently under `isolation: none` — there
is no `error` field and no exception: `start()` returns a no-op `"ready"`
record (there is no `start:` step to run), and `stop()` returns a no-op
`"stopped"` record (there is no `stop:` step to run, and the schema forbids
one under `isolation: none` in the first place). On the `stop()` side, a
caller distinguishes this "nothing to stop, by design" case from an ordinary
never-started role via `record.stop_hook_outcome.no_op_reason ==
"isolation_none"` — see "Stop hook outcome and `stop_hook_outcome`" below.
`start()` has no analogous diagnostic field today — its `isolation: none`
no-op is not currently distinguishable from any other no-op `"ready"` start.

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
| `start` | `(worktree_id: Optional[str] = None, *, checkout_path: Optional[str] = None, role: str = "main", env: Optional[dict] = None, cwd: Optional[str] = None, variant: str = "default")` | `WorktreeRecord` | Resolve the target environment by `worktree_id` or `checkout_path`, then spawn a detached process (per the contract's `start:` step for `variant`) and record its PID under `role`, and records the variant under `record.variants[role]`, which is what `stop(variant=...)` resolves against. With `variant="default"` and no exact `name` match, a two-tier fallback applies: a lone unnamed step wins if present (back-compat), else a contract with exactly one `start:` step total resolves to that step regardless of its name (ticket #112) — multi-step contracts still raise `UnknownVariantError`. |
| `stop` | `(worktree_id: Optional[str] = None, *, checkout_path: Optional[str] = None, role: Optional[str] = None, variant: Optional[str] = None, timeout: float = 10.0, kill_orphans: bool = False)` | `WorktreeRecord` | Gracefully stop the process for `role` (or the role `variant` resolves to); force-kills if it does not exit within `timeout` seconds. `role=None` (the default) means `"main"`, same as `start()`. `variant=` resolves to the role that was started with it via `record.variants`; if `role` is also given, both must agree or `VariantResolutionError` is raised. An unknown or ambiguous `variant` also raises `VariantResolutionError` — see "`role` vs `variant`" below. On `status="stop_incomplete"`, see `stop_detail` below. See "Orphan scan and `kill_orphans`" above for exactly what `kill_orphans=True` adds over the unconditional kill, per platform. |

### Orphan worktree recovery

"Orphan" covers two distinct situations, told apart by an entry's `tracked`
and `record.status` fields, and recovered differently:

| Flavour | `tracked` | `record.status` | Recover with |
|---------|-----------|------------------|---------------|
| A — untracked but on disk | `False` | `"created"` | `remove(checkout_path=...)` or `adopt()` |
| B — tracked but deregistered outside this tool | `True` | `"orphaned"` | `remove(worktree_id=...)` — no `force` needed |

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
manager.remove(orphan.record.id)
```

`force=True` is **not** required here (ticket #127): when the checkout
directory is already gone, `remove()` treats the leftover git registration
as already torn down, releases the reserved ports, and deletes the state
record without it. If the record's branch was created by this tool
(`branch_created_by_us=True`) and turns out to be unmerged, that no longer
blocks the removal either — the branch is left in place and a warning is
logged naming it and the manual `git branch -D <branch>` remedy, while
`remove()` still returns `status="removed"`. `force=True` still works as
before and still removes despite uncommitted changes on a checkout that
does exist.

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

- **Windows:** `powershell.exe -NoProfile -NonInteractive -EncodedCommand <base64 blob>`
- **POSIX:** `bash -c`

Per-step overrides (the `shell:` field):

| Value | Command used |
|-------|-------------|
| `bash` | `bash -c` |
| `sh` | `sh -c` |
| `pwsh` | `pwsh -NoProfile -NonInteractive -EncodedCommand <base64 blob>` |
| `powershell` | `powershell.exe -NoProfile -NonInteractive -EncodedCommand <base64 blob>` |

For `pwsh`/`powershell`, the step's `run:` line is base64-encoded (UTF-16LE)
and passed via `-EncodedCommand` rather than appended as raw `-Command`
text — including when the `run:` line is itself a self-wrapped/nested
PowerShell invocation, e.g. `run: powershell -Command "..."`. A raw
`-Command <text>` argument round-trips through both `subprocess`'s Windows
argv re-quoting (`list2cmdline`) and PowerShell's own `-Command` re-parsing,
which can mangle such a self-wrapped run line's quote structure and cause
the step to fail silently. `-EncodedCommand` carries no spaces or quotes, so
neither re-quoting pass can corrupt it, and previously-silent steps like
this now run correctly. Exit-code semantics are unchanged from `-Command`.

`stop()`'s reported `killed_pids[].cmdline` (ticket #132) reverses this
transport for readability: when a killed process's argv carries this
`-EncodedCommand <base64 blob>` shape, the entry's `cmdline` shows the
decoded, human-readable run line instead of the opaque blob, and the
original raw argv `psutil` actually reported is preserved in that entry's
`cmdline_raw` (`None` when no such substitution happened). Detection is
narrow and never raises — an argv that only looks similar (wrong
interpreter, invalid/foreign base64 payload, non-UTF-16LE bytes) is left
untouched — but it has no way to verify true provenance: it decodes any
argv that fits the `-EncodedCommand` shape, not only commands this
library's own setup-step transport built, so a genuine, unrelated
third-party process using `-EncodedCommand` with a validly-formed payload
gets decoded too. This is an accepted, intentional trade-off: the decoded
text is still an accurate rendering of what actually ran, and the
original argv is always available via `cmdline_raw` for anyone who needs
the byte-for-byte original.

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

### Orphan scan and `kill_orphans`

`stop()`'s unconditional kill (ppid-tree snapshot, POSIX process-group
signal, and — on Windows — the per-role Job Object's `TerminateJobObject`)
already covers a different set of processes on each platform:

| Platform | Unconditional coverage |
|---|---|
| Windows, job assigned + live handle | Every descendant of the tracked process at any depth, including a `Start-Process`/ShellExecuteEx-delegated launch outside the ppid tree — one `TerminateJobObject` call kills the whole job. `CREATE_BREAKAWAY_FROM_JOB` cannot escape it: the job is created with no limit flags, so without `JOB_OBJECT_LIMIT_BREAKAWAY_OK` the OS refuses the breakaway outright. |
| Windows, no job / no live handle | Degrades to the ppid tree snapshot alone — the process-group path is an unconditional no-op on Windows. |
| POSIX | The ppid tree snapshot plus, when the tracked pid is its own process-group leader, that group's other members. |

`kill_orphans=True` runs a further, **path-scoped, not lineage-scoped** pass
against `record.path` (the cwd/cmdline-token/open-file/Windows
handle-table orphan scan) after the unconditional kill — it kills anything
it finds under the worktree path regardless of who started it, and never
consults `record.pids`. On Windows, when the job was assigned successfully
and a live handle is available, this makes `kill_orphans` a *different
scope* rather than deeper containment — no process that `stop()` call is
responsible for escapes the unconditional kill there. Its genuine value is
closing these gaps:

- **POSIX daemonized descendant** — a child that called `setsid()` itself,
  leaving both the ppid tree and the process group (the canonical case,
  ticket #87).
- **Windows job never assigned** — job creation or `AssignProcessToJobObject`
  failed when the process was started.
- **Windows job handle unavailable at `stop()` time** — `OpenJobObjectW`
  returned no handle to enumerate or terminate.
- **Windows `setup:`-step process** — spawned by `SetupRunner`'s default
  runner, which never creates or joins a Job Object and whose pid never
  enters `record.pids`; entirely outside every other mechanism, reachable
  only because it still runs with the worktree as its cwd.
- **The sub-millisecond job-assignment race** — a descendant spawned in the
  brief window between `Popen` returning and `AssignProcessToJobObject`
  landing; narrow, and only relevant when that descendant is also outside
  the ppid tree.

It does **not** help with a `job_member_list_truncated` outcome (see below):
`TerminateJobObject` already killed every member of that job regardless of
how many were enumerated, so there is nothing left for the orphan scan to
find.

Do not pass `kill_orphans=True` on every call defensively — on Windows its
cost is dominated by a **system-wide** OS handle-table scan, budgeted at 15s
within a 20s overall discovery ceiling, and it reserves 3s of the caller's
`timeout` budget whenever requested even if nothing is found.

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
entry at all for a role with no known variant). When `start()`'s
`variant="default"` fallback resolves a *named* step (ticket #112 — a
contract with exactly one `start:` step total, named or not),
`record.variants[role]` records that step's own name (e.g. `"main"`), not
the literal `"default"` — so `stop(variant="main")` resolves it correctly
afterward. Note: after this fallback resolves a named step,
`stop(variant="default")` will **not** find it — no role is ever recorded
under the literal `"default"` in this case, so you must stop it with the
step's actual name (`stop(variant="main")` in this example).

`stop(variant=...)` resolves
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

### Per-role start logs

`record.start_log_paths[role]` gives that role's own start-log path — the
absolute path of the `start-<role>.log` file `start()` wrote when it started
that role. It mirrors `pids`/`job_names`/`variants`: one entry per role
that has ever been started, no key at all for a role with no recorded start
log. The filename's role token is sanitized (filesystem-unsafe characters
collapsed to `-`), but the dict *key* is always the raw, unslugged role
string.

Unlike `pids`/`job_names`/`variants`, entries here are **retained** by
`stop()` and by `reconcile`'s dead-role sweep — `set(record.start_log_paths)
<= set(record.pids)` does not hold and is not meant to. The log file
outlives the process it describes, and the main use case is reading a
role's log path off the response of the `stop()` call that just stopped it.
Restarting the same role overwrites that role's single entry rather than
accumulating.

This replaces the older `start_log_path: Optional[str]` scalar, which was
overwritten by every `start()` call regardless of role and so, in a
multi-role worktree, named whichever role started last. The scalar is
removed entirely — no alias, no deprecation shim — so a `state.yaml`
written by an older engine deserializes to an empty `start_log_paths: {}`.

### Stop status and `stop_detail`

When `stop()` cannot confirm that everything it tried to kill actually died,
it reports `status="stop_incomplete"` instead of `"stopped"` and attaches a
`stop_detail` (a `StopDetail`) to the returned `WorktreeRecord` naming why:

| `reason` | Meaning |
|---|---|
| `survivors` | One or more tracked PIDs were still alive after every kill attempt (`survivor_pids`, capped at 32, plus the true `survivor_count`). |
| `tree_truncated` | The descendant-process-tree snapshot hit its node cap, so some descendants were never even examined. |
| `job_member_list_truncated` | Windows-only: the Job Object's member list hit its slot cap. `kill_orphans` does not help here — `TerminateJobObject` already killed every member of that job regardless of enumeration. |
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

### Stop hook outcome and `stop_hook_outcome`

Every `stop()` call sets a transient `stop_hook_outcome` (a `StopHookOutcome`)
on the returned `WorktreeRecord`, describing whether/how the contract's
`stop:` hook ran and the contract diagnostics behind that verdict:

| Field | Meaning |
|---|---|
| `status` | One of `"completed"`/`"failed"`/`"skipped"` — reuses the same vocabulary as `setup_outcome.status` rather than minting a new one. `"failed"` covers both a `stop:` step raising and the contract itself failing to load/parse. |
| `message` | `str()` of the exception for `"failed"`; a short summary otherwise. |
| `steps_run` | The number of `stop:` steps that actually ran (`0` for `"skipped"`/`"failed"`). |
| `contract_found` | Whether `<repo_root>/.seretos/worktree-setup.yml` exists on disk, checked independently of the parse attempt so a filesystem error on one check can never mask the other. |
| `contract_path` | The forward-slash contract path that was probed, regardless of whether it was found. |
| `contract_isolation` | The loaded contract's `isolation` value, or `None` when the contract could not be loaded/parsed. |
| `no_op_reason` | Set only on `stop()`'s no-op branch (the resolved role has no recorded pid): `"isolation_none"` when `contract_isolation == "none"`, else `"no_process_recorded"`. `None` whenever the call was not a no-op. |

This is what lets a caller tell an `isolation: none` "nothing to stop, by
design" no-op apart from an ordinary "role never started" no-op — both
otherwise set the identical `stop_attempt.outcome == "no_process_recorded"`
on the record's `stop_attempt` (a `StopAttempt`, answering the narrower,
tracked-PID-only question of what `stop()` found at the tracked PID itself).
`stop_hook_outcome` is deliberately **transient**, exactly like
`stop_attempt`: recomputed on every `stop()` call and **not** persisted to
`state.yaml`.

## Release

Releases are pipeline-owned (`.github/workflows/release.yml`, manual dispatch
with `version=X.Y.Z`). See `AGENTS.md` for the release + downstream-ticket
mechanics.
