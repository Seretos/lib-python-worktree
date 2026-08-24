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

7. `_phase_orphan_scan`
   All-platform (not just Windows), **warn-only** scan for a process
   holding the checkout (as cwd, cmdline token, Windows handle, or open
   file) that this engine never tracked -- previously silently orphaned
   once `git worktree remove` succeeded around it (ticket #140). Runs
   AFTER the `teardown:` steps phase but BEFORE the destructive
   `git worktree remove` call, with a fresh, full-budget
   `_find_blocking_processes(record.path, os.getpid())` call (no
   `deadline`) -- deliberately not a reuse of Gate A's own result, since
   Gate A never runs on POSIX or against an absent target, and the
   `teardown:` steps that just ran may have changed the picture. On
   win32 with a present target this means a removal now pays TWO full
   scans (Gate A's own pre-flight plus this phase's) -- an accepted,
   documented cost. Skipped entirely (`record.orphan_scan` stays `None`)
   when the target is already absent (mirrors Gate A's own ticket #127
   skip).

   `_kill_blocking_processes(record.path)` (the SAME reused remedy Gate A
   and `_resolve_lock_or_raise` use) is invoked only when
   `kill_blocking_processes=True` **and** the scan found at least one hit
   -- a clean scan skips the ~5s kill call entirely even with the flag
   set. This is what makes `kill_blocking_processes=True` meaningful on
   POSIX for the first time. `ctx.kill_attempted` is set immediately
   before that call, so an attempt that raises mid-way is still recorded
   honestly. Only entries the kill call actually confirms killed are
   merged into `record.killed_pids` (via `_merge_killed_pids` -- see the
   cross-phase invariant below).

   `record.orphan_scan` (a `state.OrphanScanReport`) is the **union** of
   the warn scan and the kill result, first-wins pid-deduped, preserving
   scan order then kill-only order: a pid the scan found but the kill's
   own tighter rescan did not confirm is reported `killed=False`; a pid
   both the scan and the kill confirmed is reported `killed=True`; a
   lineage-only pid the kill's `_process_tree` expansion added (never
   seen by the scan itself) is reported `killed=True` too. Unlike
   `StopDetail`'s `survivor_pids`, the hit list is deliberately
   **uncapped**.

   **Warn-only, hard invariant:** this phase never raises and never adds
   a new refusal condition or a new `force` requirement. The entire body
   (scan AND kill) is wrapped in a single `except Exception` that logs
   once, appends the synthetic `"scan:failed"` marker to
   `skipped_passes`, and proceeds with whatever partial results were
   already collected -- this is what protects against
   `_find_blocking_processes` (or the kill call) re-raising a bare
   `RuntimeError` (ticket #107) and turning a working removal into a
   failure. Exactly one `_logger.warning(...)` is emitted for an abnormal
   outcome (no hits, or `skipped_passes` non-empty, or the failure case),
   with the identical string also stored as the report's `message`.

   **Kill-target caveat:** with `kill_blocking_processes=True`, a
   heuristic hit from Pass 1b (`cmdline`), Pass 1c (`handle_scan`), or
   Pass 2 (`open_files`) -- not only an exact Pass 1 cwd match -- becomes
   a KILL TARGET, not merely a warning: an editor or AV scanner that
   merely holds a handle or an open file under the checkout can be
   terminated by an opt-in caller.

8. `_phase_git_worktree_remove`
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

9. `_phase_filesystem_fallback`
   Long-path (Windows `MAX_PATH`) fallback: extended-path `rmtree`, then a
   bounded robocopy empty-mirror trick (ticket #78). Plain `shutil.rmtree`
   on POSIX. Best-effort; never raises.

10. `_phase_final_guard`
    If the checkout directory is still present after every deletion
    attempt, raise `WorktreeDirLockedError` rather than silently reporting
    `"removed"`.

11. `_phase_release_ports`
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
- **The Gate A / Gate B trigger split (ticket #140).** `force` and
  `kill_blocking_processes` guard TWO DIFFERENT gates, and neither
  substitutes for the other. Gate A (win32-only) refuses on a confirmed
  blocking process and is bypassed only by `kill_blocking_processes=True`
  -- `force=True` alone still raises `WorktreeDirLockedError`. Gate B
  refuses on real uncommitted dirt and is bypassed only by `force=True`
  -- `kill_blocking_processes=True` alone still raises
  `DirtyWorktreeError`. Neither gate runs on POSIX; phase 7
  (`_phase_orphan_scan`) is what surfaces a blocking process there
  instead, warn-only rather than as a refusal.
- **`record.killed_pids` merges, never overwrites (ticket #140).** Every
  call site that assigns to `record.killed_pids` -- `_resolve_lock_or_raise`,
  Gate A's confirmed-blocker kill, and `_phase_orphan_scan`'s own kill --
  goes through `_merge_killed_pids` (first-wins, pid-deduped, in-place
  append), never a plain `record.killed_pids = killed` assignment. A pid
  an earlier phase (or an earlier `stop()` call) already recorded is
  preserved, not clobbered by a later phase's own kill result.
