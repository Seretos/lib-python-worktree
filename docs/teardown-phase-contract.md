# Teardown phase contract (ticket #135)

This document is the central contract for `core/teardown.py`'s removal
sequence. It names every phase in `teardown._TEARDOWN_PHASES`, in order,
with the invariant it must uphold. `tests/test_teardown_contract_doc.py`
parses the phase names below and asserts they match
`[f.__name__ for f in teardown._TEARDOWN_PHASES]` exactly -- keep this file
and that tuple in sync.

Any new teardown/remove scenario belongs in
`tests/test_teardown_matrix.py` first (as a characterization/regression
row), not as an ad-hoc test elsewhere -- see that file's own docstring.

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

4. `_phase_gate_a_blocking_preflight`
   Windows-only pre-flight check for a confirmed blocking process, run
   BEFORE the destructive `git worktree remove` call. Skipped entirely
   when the target directory is already absent (ticket #127). A pid this
   environment owns is trusted immediately; a foreign pid must survive a
   bounded settle-and-rescan window before being treated as a confirmed
   blocker (ticket #117). A degraded/partial scan is never, by itself, a
   blocking condition (ticket #121). Raises `WorktreeDirLockedError` /
   `WorktreeRemovalBlockedError` (ticket #103, when real dirt is also
   present) when a confirmed blocker cannot be cleared.

5. `_phase_gate_b_early_dirty`
   Early dirty-tree refusal, run BEFORE the `teardown:` steps phase, but
   only when the contract actually has `teardown:` steps to protect
   (ticket #117, AC #3; pinned for ticket #123). Raises
   `DirtyWorktreeError` when real (non-`.seretos/`-only) dirt is present
   and `force=False`.

6. `_phase_run_teardown_steps`
   Run the contract's `teardown:` steps at most once per logical removal
   (ticket #126): gated on `not record.teardown_ran`, and the marker is
   persisted immediately after the steps complete, before the git-remove
   phase is attempted, so a later `force=True` retry after a
   *post*-teardown `DirtyWorktreeError` never re-runs these steps. A step
   failure never blocks the rest of teardown. Invalidates the memoised
   dirt-probe snapshot afterward so later phases see fresh disk state.

7. `_phase_git_worktree_remove`
   Run `git worktree remove` (with `--force` iff `force=True`). On
   failure, triages via `_triage_remove_failure`: phantom-state
   deregistration (`is not a working tree`) is treated as already-gone;
   the exit-128 fallback widens to fire without `force=True` when the
   target was already absent (ticket #127); a directory-lock signal routes
   through the shared kill-and-retry remedy (`_resolve_lock_or_raise`,
   ticket #72); dirt that is *only* the benign `.seretos/` convenience
   copy auto-escalates to a forced retry via `_seretos_exemption_retry`
   (ticket #100); anything else raises `DirtyWorktreeError` or a bare
   `GitCommandError`.

8. `_phase_filesystem_fallback`
   Long-path (Windows `MAX_PATH`) fallback: extended-path `rmtree`, then a
   bounded robocopy empty-mirror trick (ticket #78). Plain `shutil.rmtree`
   on POSIX. Best-effort; never raises.

9. `_phase_final_guard`
   If the checkout directory is still present after every deletion
   attempt, raise `WorktreeDirLockedError` rather than silently reporting
   `"removed"`.

10. `_phase_release_ports`
    Release the record's allocated ports -- only reached once the git
    remove (and fallback/final-guard) phases above have succeeded, so a
    concurrent `allocate()` can never reissue a port that is still bound.

## Cross-phase invariants

- **Ordering is load-bearing.** `run_teardown` aborts on the first phase
  that raises; branch deletion is deliberately NOT one of these phases --
  it happens in `WorktreeManager.remove()`, after the state record is
  removed, so a branch-delete failure never leaves a stale orphaned
  record.
- **Dirt memoisation is two-phase**, not one snapshot for the whole
  removal: Gate A/Gate B consume the pre-teardown probe; every phase-6+
  consumer must see a fresh, post-teardown probe (`_TeardownContext.dirt`
  docstring has the full rationale). Collapsing this back into a single
  snapshot reintroduces the ticket #117 regression.
- **`teardown._target_is_absent(record)`** is the single seam for "is the
  checkout directory already gone?", shared by `WorktreeManager.remove()`
  and `build_context()`. The long-path-fallback and Final-guard
  `os.path.exists` checks are deliberately NOT routed through this seam --
  they must always see the real filesystem.
- **No `WorktreeManager` reference crosses into this module.** `store`/
  `allocator` are plain fields on `_TeardownContext`, read once at
  `build_context()` time.
