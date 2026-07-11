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

# Remove a worktree (pass force=True to remove despite uncommitted changes):
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
`process_lifecycle.py`, `_git_utils.py`, and `contract/loader.py`.
`yaml_store.py` imports from `_git_utils.py`; `port_allocator.py` imports from
`yaml_store.py` (sharing `_LOCK_FLAGS`, `_LOCK_TIMEOUT`, `_PortsFile`, and
`_port_in_use`).
`setup/runner.py` is independent of `core/`; `manager.py` imports it lazily
inside `_teardown` to avoid a circular import.

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
| `create` | `(repo_root: str, branch: str, base: Optional[str] = None)` | `WorktreeRecord` | Add a git worktree, allocate ports per the contract, and persist state. Pass `base` to create the branch. |
| `list` | `()` | `List[WorktreeRecord]` | Return all tracked worktree records. |
| `remove` | `(worktree_id: str, force: bool = False)` | `WorktreeRecord` | Run teardown, remove git worktree, release ports, delete state. `force=True` removes despite uncommitted changes. |
| `adopt` | `(repo_root: str)` | `AdoptReport` | Import untracked on-disk worktrees into the state store. Requires `YamlStateStore`. |
| `prune` | `(repo_root: str)` | `None` | Run `git worktree prune --expire=now` to clear stale git metadata. |
| `start` | `(worktree_id: str, cmd: List[str], *, role: str = "main", env: Optional[dict] = None, cwd: Optional[str] = None)` | `WorktreeRecord` | Spawn a detached process and record its PID. |
| `stop` | `(worktree_id: str, *, role: str = "main", timeout: float = 10.0)` | `WorktreeRecord` | Gracefully stop the process for `role`; force-kills if it does not exit within `timeout` seconds. |

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
│   │   ├── GitTimeoutError              (git subprocess exceeded WORKTREE_GIT_TIMEOUT_SEC)
│   │   ├── DirtyWorktreeError           (remove refused; pass force=True)
│   │   ├── BranchNotFoundError
│   │   ├── BranchAlreadyCheckedOutError (carries .branch, .path, .prunable)
│   │   ├── DuplicateWorktreeError
│   │   ├── WorktreeNotFoundError
│   │   └── GitCommandError              (carries .command, .returncode, .stderr)
│   ├── ProcessLifecycleError            (base for process lifecycle errors)
│   │   ├── ProcessAlreadyRunningError   (carries .worktree_id, .role, .pid)
│   │   └── ProcessNotRunningError       (carries .worktree_id, .role)
│   ├── PortAllocationError
│   └── SetupFailedError
└── ContractError                        (base for contract loading errors)
    └── ContractValidationError          (carries .path, .errors)
```

All public exception classes are re-exported from `lib_python_worktree`.

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

### Setup timeout

`WORKTREE_SETUP_TIMEOUT_SEC` controls how long each `setup:`/`stop:`/
`teardown:`/`seed_postprocess:` step's subprocess may run before being killed
and raising `SetupFailedError`. Default: `300.0` seconds. Set to an empty
string to disable the timeout entirely (diagnostic use only). Precedence:
an explicit `timeout=` kwarg passed to `SetupRunner(...)`/`SetupRunner.run(...)`
wins over this env var, which wins over the built-in default.

### Process lifecycle

Process detachment on Windows uses `CREATE_NEW_PROCESS_GROUP` so that
`CTRL_BREAK_EVENT` can be delivered for graceful stop. On POSIX,
`start_new_session=True` is used and `SIGTERM` / `SIGKILL` are used for
graceful and force stops respectively.

## Release

Releases are pipeline-owned (`.github/workflows/release.yml`, manual dispatch
with `version=X.Y.Z`). See `AGENTS.md` for the release + downstream-ticket
mechanics.
