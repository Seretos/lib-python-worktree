# Teardown phase contract (ticket #135, rewritten for #154)

This document is the central contract for `core/teardown.py`'s removal
sequence. It names every phase in `teardown._TEARDOWN_PHASES`, in order,
with the invariant it must uphold. `tests/test_teardown_contract_doc.py`
parses the phase names below and asserts they match
`[f.__name__ for f in teardown._TEARDOWN_PHASES]` exactly -- keep this file
and that tuple in sync.

Any new teardown/remove scenario belongs in
`tests/test_teardown_matrix.py` first (as a characterization/regression
row), not as an ad-hoc test elsewhere -- see that file's own docstring.

**Ticket #154 rewrite, in one sentence:** the mechanism changed from "ask
the whole system who holds this directory" (a Windows-only pre-flight scan,
a `git worktree remove` call with its own dirty/lock triage, an
all-platform warn-only orphan scan, and a long-path filesystem fallback) to
"try to get rid of it, and only diagnose when that fails" -- `os.rename`
proves unheldness on Windows and stages the checkout on both platforms; a
scan (`_find_blocking_processes`) is now only ever consulted as bounded
diagnosis after a rename or delete failure, never on the happy path. The
phase count is **11 -> 10**: four phases deleted
(`_phase_gate_a_blocking_preflight`, `_phase_orphan_scan`,
`_phase_git_worktree_remove`, `_phase_filesystem_fallback`), three added
(`_phase_dirt_gate`, `_phase_stage_and_delete`, `_phase_git_prune`).
`_phase_reclaim_staged` (a phase generation-2 planning considered and a
later human-override decision explicitly rejected) does **not** exist --
see the "non-empty `.removing` remnant" invariant below.

## Phases, in order

1. `_phase_guard_primary`
   Refuse a primary (main-clone) checkout before any lifecycle/FS side
   effect. Never bypassable via `force=True` (ticket #84).

2. `_phase_stop_processes`
   Stop every tracked process (by role) before touching the filesystem.
   `ProcessNotRunningError`/`ProcessLifecycleError` are swallowed --
   best-effort, never blocks removal.

3. `_phase_stop_hook`
   Run the contract's `stop:` steps (best-effort) so daemons holding file
   handles / PID files can release them before deletion. Always records a
   `StopHookOutcome` onto `record.stop_hook_outcome`, even on failure
   (ticket #130) -- a hook failure is warn-logged but never blocks
   teardown.

4. `_phase_gate_b_early_dirty`
   Unchanged from before #154. Early dirty-tree refusal, run BEFORE the
   `teardown:` steps phase, but only when the contract actually has
   `teardown:` steps to protect (ticket #117, AC #3; pinned for ticket
   #123). Raises `DirtyWorktreeError` when real (non-`.seretos/`-only)
   dirt is present and `force=False`.

5. `_phase_run_teardown_steps`
   Run the contract's `teardown:` steps at most once per logical removal
   (ticket #126): gated on `not record.teardown_ran`, and the marker is
   persisted immediately after the steps complete, before the dirt-gate/
   stage-and-delete phases below are attempted, so a later `force=True`
   retry after a *post*-teardown `DirtyWorktreeError` never re-runs these
   steps. A step failure never blocks the rest of teardown. Invalidates
   the memoised dirt-probe snapshot afterward so the dirt gate sees fresh
   disk state.

6. `_phase_dirt_gate`
   New in #154 (item 16-17). The unconditional second dirt verdict,
   immediately before staging. Skipped entirely -- probe included -- when
   `force=True` or `ctx.target_absent`. Reuses the memoised `ctx.dirt()`;
   the `teardown:` steps phase above already invalidated it when it ran,
   so the happy path (no real dirt) pays at most one `git status` call,
   never two.

   Real dirt found: a **bare rename pair** (forward, then immediately
   back) -- deliberately NOT the transient-retry loop, no sleep -- decides
   between three outcomes:
   - forward rename fails: the directory is genuinely locked too ->
     `WorktreeRemovalBlockedError` (#103's combined verdict), `blockers`
     populated, `staged=False`.
   - forward succeeds, undo succeeds: plain `DirtyWorktreeError`,
     `staged=False`, tree exactly where it was.
   - forward succeeds, undo fails: a bounded but **budget-isolated**
     retry on the undo (`_bare_retry_bounded` -- reuses
     `_TRANSIENT_RETRY_BUDGET_SEC`/`_TRANSIENT_RETRY_STEP_SEC` but never
     opens `ctx.failure_deadline`, human override Decision 2). Still
     unresolved: `DirtyWorktreeError(staged=True)` -- never the combined
     error, since the forward probe succeeding already proved the
     directory was not locked (human override Decision 1+2).

7. `_phase_stage_and_delete`
   New in #154 (item 2), replacing `_phase_git_worktree_remove` and
   `_phase_filesystem_fallback` together. Rename is the oracle on both
   platforms -- `os.rename(record.path, record.path + ".removing")`
   proves unheldness on Windows and moves the checkout out of the way on
   both; the staged tree is then deleted via the 4-rung
   `_delete_tree_ladder` (plain `shutil.rmtree`; a chmod sweep + retry for
   a read-only file, ticket #78's P4; win32 extended-path `rmtree`; win32
   robocopy empty-mirror, bounded by `WORKTREE_ROBOCOPY_TIMEOUT_SEC`).

   Opening guard (human override, Decision 1 -- the lighter fix that
   replaces generation-2 planning's rejected `_phase_reclaim_staged`):
   - neither the original tree nor a `.removing` remnant present: true
     no-op, `os.rename` is never called.
   - remnant only, `force=False`: `_target_is_absent` already refused to
     fast-path this as "target absent" -- raise here too
     (`DirtyWorktreeError(staged=True)`), reusing the existing
     vocabulary rather than silently restoring or destroying it.
   - remnant only, `force=True`: force authorises destroying it -- clear
     it via the delete ladder and return.
   - original present, remnant also present (a same-path recreation
     collision): pre-clean the stale remnant via the delete ladder before
     staging again.
   - original present, no remnant: the ordinary case.

   A failed rename, or a residual left behind after the delete ladder,
   goes through `_retry_bounded` (opens `ctx.failure_deadline` if not
   already open, shared by every failure-path leg of this removal
   attempt) and then, if still unresolved, `_diagnose_and_retry` (tier 1:
   bounded owned-pid liveness via `psutil.pid_exists`, kill + one retry
   if `kill_blocking_processes=True`; tier 2: systemwide
   `_find_blocking_processes`/`_kill_blocking_processes`, kill + one
   retry, only ever attempted when tier 2 actually found a candidate).
   **Neither this phase nor `_diagnose_and_retry` raises on an
   unresolved failure** -- `_phase_final_guard` is the sole raise site,
   using the literal on-disk truth at the end of the removal attempt.

8. `_phase_git_prune`
   New in #154 (item 1). Unconditional `git worktree prune --expire=now`
   -- runs even on the target-absent fast path, so a stale git
   registration is always cleared. Warn-and-continue on failure: by this
   phase the checkout is genuinely gone from disk (or never existed),
   only git's bookkeeping may be stale, and that staleness is self-healing
   via `WorktreeManager.prune()`.

9. `_phase_final_guard`
   Rewritten for #154. The sole raise site for a `_phase_stage_and_delete`
   failure `_diagnose_and_retry` could not resolve, using the literal
   on-disk truth: `staged = os.path.exists(record.path + ".removing")`
   (true whenever the remnant exists, including when `record.path` also
   exists -- the remnant is the more surprising fact). Remnant absent,
   `record.path` present -> `staged=False` -> byte-identical to v0.3.12's
   `kill_attempted`-selected message. Remnant present (either sub-case)
   -> `staged=True` -> a phrasing naming the `.removing` marker suffix and
   the worktree id, never a filesystem path. Either case: ERROR log +
   `WorktreeDirLockedError(..., blockers=list(ctx.blockers))`. `status` is
   never `"removed"` on this branch and ports are not released.

10. `_phase_release_ports`
    Release the record's allocated ports -- only reached once every phase
    above has succeeded, so a concurrent `allocate()` can never reissue a
    port that is still bound.

## Cross-phase invariants

- **Ordering is load-bearing.** `run_teardown` aborts on the first phase
  that raises; branch deletion is deliberately NOT one of these phases --
  it happens in `WorktreeManager.remove()`, after the state record is
  removed, so a branch-delete failure never leaves a stale orphaned
  record.
- **Dirt memoisation is two-phase**, not one snapshot for the whole
  removal: Gate B consumes the pre-teardown probe; the dirt gate must see
  a fresh, post-teardown probe (`_TeardownContext.dirt` docstring has the
  full rationale). Collapsing this back into a single snapshot
  reintroduces the ticket #117 regression.
- **`teardown._target_is_absent(record, force=...)`** is the single seam
  for "is the checkout directory already gone?", shared by
  `WorktreeManager.remove()` and `build_context()`. Ticket #154 adds one
  rule: a non-empty `<path>.removing` remnant under `force=False` is NOT
  "target absent", even when `record.path` itself is gone -- it must
  surface as an error on the next removal attempt (the dirt gate /
  stage-and-delete phase raises, reusing the existing
  `DirtyWorktreeError(staged=True)`/`WorktreeDirLockedError` vocabulary)
  rather than being silently fast-pathed as "nothing to remove here".
  This is a **named, accepted limitation** (human override, Decision 1):
  the self-healing generation-2 planning wanted (auto-restore a clean
  crash remnant) is not delivered by this lighter fix -- an operator who
  hits it must clear the stale `.removing` directory by hand or pass
  `force=True`. Under `force=True` the remnant is still treated as
  "target absent" here; the pre-clean happens in
  `_phase_stage_and_delete`'s opening guard, not in this seam.
- **No `WorktreeManager` reference crosses into this module.** `store`/
  `allocator` are plain fields on `_TeardownContext`, read once at
  `build_context()` time.
- **The two trigger kinds and the retry-then-diagnose call sites.** A
  failed rename and a residual-after-delete are the only two triggers
  `_diagnose_and_retry` ever sees, both from `_phase_stage_and_delete`.
  `_retry_bounded` has two call sites (the rename, the residual check);
  the dirt gate's undo retry is a third, **bespoke** bounded-retry call
  site (`_bare_retry_bounded`) that deliberately does NOT share
  `_retry_bounded`'s `ctx.failure_deadline`-opening contract (human
  override, Decision 2) -- `ctx.failure_deadline` stays `None` on every
  `DirtyWorktreeError` leg, `staged=True` included.
- **One failure budget per removal (`_FAILURE_BUDGET_SEC = 7.0`).**
  Opened the first time `_retry_bounded` is entered for a given removal
  attempt; every later failure-path leg of the SAME attempt shares it.
  Disjoint from AC1's `<2s`/`<5s` ceiling by construction: a clean,
  unheld worktree's rename succeeds on the first attempt, so the failure
  budget is never opened at all for that population.
- **`record.killed_pids`/`ctx.blockers` merge, never overwrite.** Every
  call site that records a kill goes through `_merge_killed_pids`
  (first-wins, pid-deduped, in-place append), never a plain
  `record.killed_pids = killed` assignment.
