"""Ticket #135 -- canonical parametrized regression matrix for teardown/remove.

This file is the AUTHORITATIVE consolidation of the eleven historical
teardown/remove scenarios fixed across tickets #76, #84, #88, #103, #107,
#117, #121, #123, #126, #127, #130, plus a set of boundary-condition rows
(force True/False x platform x target_absent, contract absent/unparseable/
isolation:none, and the teardown_ran-already-True idempotency guard).

It was landed FIRST, against the UNMODIFIED (pre-refactor) ``manager.py``,
as the characterization baseline for ticket #135's structural extraction
into ``core/teardown.py``. Any future teardown/remove regression should add
its scenario HERE rather than as an ad-hoc one-off test elsewhere (see
``docs/teardown-phase-contract.md`` and ``AGENTS.md``).

Uses the same InMemoryStateStore + ``_make_record`` pattern as
``tests/test_teardown.py`` -- no real git required.
"""

from __future__ import annotations

import logging
import os
import shutil

from pathlib import Path
from unittest.mock import MagicMock, patch

# Ticket #137: bind the REAL shutil.rmtree function object now, before any
# test patches "lib_python_worktree.core.teardown.shutil.rmtree" (which
# patches the attribute on this same `shutil` module, since teardown.py just
# does a plain `import shutil`). A lookup of `shutil.rmtree` performed later,
# inside a test, would resolve to the patched mock instead of the real
# function and recurse infinitely.
_real_rmtree = shutil.rmtree

import pytest

from lib_python_worktree.core.manager import (
    DirtyWorktreeError,
    GitCommandError,
    ManagerConfig,
    PrimaryCheckoutError,
    WorktreeDirLockedError,
    WorktreeManager,
    WorktreeRemovalBlockedError,
)
from lib_python_worktree.core.process_lifecycle import (
    KilledProcessInfo,
    ProcessNotRunningError,
    _PartialList,
)
from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord
from lib_python_worktree.contract.schema import Step, WorktreeContract


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/test_teardown.py's own helper pair)
# ---------------------------------------------------------------------------

def _make_manager(tmp_path: Path) -> WorktreeManager:
    return WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )


def _make_record(wt_id: str = "wt-matrix", **kwargs) -> WorktreeRecord:
    defaults = dict(
        id=wt_id,
        repo_root="/fake/repo",
        branch="feature/x",
        path="/fake/store/" + wt_id,
    )
    defaults.update(kwargs)
    return WorktreeRecord(**defaults)


def _ok(*a, **kw):
    return MagicMock(returncode=0, stderr="")


@pytest.fixture(autouse=True)
def _no_real_settle_sleep():
    """No-op the Gate A settle-window sleep so rows exercising it stay fast."""
    with patch("lib_python_worktree.core.teardown.time.sleep"):
        yield


# ---------------------------------------------------------------------------
# Part 1: one row per historical ticket
# ---------------------------------------------------------------------------

class TestMatrixHistoricalTickets:
    """Each test id names the ticket it consolidates/pins."""

    def test_ticket_76_gate_a_confirmed_blocker_without_kill_flag_raises(
        self, tmp_path
    ):
        """#76: a confirmed Windows blocking process, kill_blocking_processes
        not opted into -> WorktreeDirLockedError, kill_attempted=False, no
        destructive git call fired for the blocked attempt's own removal."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-t76")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        blocker = [KilledProcessInfo(pid=4242, name="node.exe", cmdline=["node"])]

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=blocker,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.teardown.os.path.exists",
                return_value=True,
            ),
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )
        assert excinfo.value.kill_attempted is False

    def test_ticket_84_primary_checkout_never_removed_even_forced(self, tmp_path):
        """#84: a primary-backing record is refused, and force=True does not
        bypass the refusal."""
        manager = _make_manager(tmp_path)
        record = _make_record(
            "wt-t84", path="/fake/repo", repo_root="/fake/repo", backing="primary"
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        with pytest.raises(PrimaryCheckoutError):
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)

    def test_ticket_88_untracked_synthesised_record_teardown_marker_swallowed(
        self, tmp_path
    ):
        """#88: a synthesised record never added to the store still tears
        down cleanly -- the teardown_ran persistence KeyError is swallowed,
        not propagated."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-t88-untracked")
        # Deliberately NOT added to manager.state -- mirrors the untracked
        # removal path's synthesised, never-stored record.
        mock_lifecycle = MagicMock()

        fake_contract = WorktreeContract(
            version=1, isolation="full", teardown=[Step(run="echo bye", name="bye")]
        )
        mock_runner_instance = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
        ):
            mock_git.return_value = _ok()
            # Must not raise despite self.state.update(record) hitting KeyError.
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert record.teardown_ran is True

    def test_ticket_103_seretos_only_dirt_auto_forces_without_dirtyworktreeerror(
        self, tmp_path
    ):
        """#103: the ONLY dirt is the benign .seretos/ convenience copy ->
        auto-escalates to --force instead of raising DirtyWorktreeError."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-t103")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        calls = []

        def _side_effect(args, cwd=None, **kwargs):
            calls.append(list(args))
            if args == ["worktree", "remove", str(record.path)]:
                return MagicMock(
                    returncode=128,
                    stderr="fatal: contains modified or untracked files",
                )
            if args[:2] == ["status", "--porcelain"]:
                return MagicMock(returncode=0, stdout="?? .seretos\x00")
            return MagicMock(returncode=0, stderr="")

        with patch(
            "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
        ):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        retry = [c for c in calls if c == ["worktree", "remove", "--force", record.path]]
        assert retry, f"expected an auto --force retry, got calls={calls}"

    def test_ticket_107_ordinary_truncation_falls_through_to_normal_removal(
        self, tmp_path
    ):
        """#107: a degraded scan (open_files:degraded) with NO prior hit is
        NOT by itself a refusal condition -- removal proceeds normally."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-t107")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        degraded_empty = _PartialList([], skipped_passes=("open_files:degraded",))

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=degraded_empty,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "win32"
            # Must not raise -- a degraded scan alone never blocks.
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_ticket_117_gate_b_dirt_check_precedes_teardown_steps(self, tmp_path):
        """#117: Gate B's dirty-tree check runs BEFORE contract teardown:
        steps -- a doomed (dirty) attempt never executes a non-idempotent
        teardown command first."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-t117")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        fake_contract = WorktreeContract(
            version=1, isolation="full", teardown=[Step(run="echo bye", name="bye")]
        )
        mock_runner_instance = MagicMock()

        def _git_side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["status", "--porcelain"]:
                return MagicMock(returncode=0, stdout="M  dirty.txt\x00")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
        ):
            with pytest.raises(DirtyWorktreeError):
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )

        mock_runner_instance.run.assert_not_called()

    def test_ticket_121_degraded_partial_scan_never_blocks_alone(self, tmp_path):
        """#121: a degraded/partial process-scan result from the blocking-
        process boundary never blocks removal on its own (same guarantee as
        #107, pinned separately since #121's fix lived entirely inside
        process_lifecycle.py's scan internals -- this row only asserts the
        _teardown()-side contract: a degraded _PartialList never refuses)."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-t121")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        degraded_partial = _PartialList(
            [], skipped_passes=("cwd:skipped", "open_files:degraded")
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=degraded_partial,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "win32"
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

    def test_ticket_123_gate_b_refuses_before_teardown_pinning_the_decision(
        self, tmp_path
    ):
        """#123 does not appear anywhere in src/ or tests/ under that literal
        name -- this row derives its matrix entry from the current code path
        (Gate B: refuse before teardown: when not force and contract.teardown
        and dirt) and labels it as PINNING that deliberate decision, not as
        fixing a regression. Duplicate-in-spirit of the #117 row above,
        kept as its own named row per the plan's matrix enumeration."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-t123")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        fake_contract = WorktreeContract(
            version=1, isolation="full", teardown=[Step(run="echo bye", name="bye")]
        )

        def _git_side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["status", "--porcelain"]:
                return MagicMock(returncode=0, stdout="M  dirty.txt\x00")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
        ):
            with pytest.raises(DirtyWorktreeError):
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )

    def test_ticket_126_teardown_steps_not_rerun_on_forced_retry(self, tmp_path):
        """#126: teardown: steps run at most once per logical removal. A
        first attempt that fails with a POST-teardown DirtyWorktreeError,
        retried with force=True, must NOT re-run the teardown: steps."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-t126")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        fake_contract = WorktreeContract(
            version=1, isolation="full", teardown=[Step(run="echo bye", name="bye")]
        )
        mock_runner_instance = MagicMock()
        teardown_has_run = [False]

        def _mark_ran(**kw):
            teardown_has_run[0] = True
            return MagicMock(steps=[MagicMock()])

        mock_runner_instance.run.side_effect = _mark_ran

        def _git_side_effect_first(args, cwd=None, **kwargs):
            if args == ["worktree", "remove", record.path]:
                return MagicMock(
                    returncode=128,
                    stderr="fatal: contains modified or untracked files",
                )
            if args[:2] == ["status", "--porcelain"]:
                if not teardown_has_run[0]:
                    # Pre-teardown (Gate B) probe(s): an empty/inconclusive
                    # status is ALWAYS treated as "no dirt" by
                    # _real_dirt_paths (see _status_entries's own
                    # "never invent a blocking condition from an
                    # inconclusive probe" contract) -- however many times
                    # Gate B's own dirt_probe() internally re-fetches it,
                    # it stays non-blocking, letting the attempt proceed to
                    # the teardown: steps below.
                    return MagicMock(returncode=0, stdout="")
                # Post-teardown probe (re-fetched because Step 3 invalidates
                # the memoised snapshot): the teardown: step itself left real
                # dirt behind, which is what #126 guards must still tolerate
                # without re-running the steps on a later retry.
                return MagicMock(returncode=0, stdout="M  dirty.txt\x00")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect_first,
            ),
        ):
            with pytest.raises(DirtyWorktreeError):
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )

        assert mock_runner_instance.run.call_count == 1
        assert record.teardown_ran is True

        # Retry with force=True: Gate B is bypassed, but teardown_ran==True
        # must gate the steps from running again.
        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git2,
        ):
            mock_git2.return_value = _ok()
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)

        assert mock_runner_instance.run.call_count == 1, (
            "teardown: steps must not run a second time for the same "
            "logical removal"
        )

    def test_ticket_127_orphaned_target_absent_removable_without_force(
        self, tmp_path
    ):
        """#127: a tracked record whose checkout dir was deleted externally
        (target_absent) removes cleanly with force=False, releasing ports."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-t127")
        manager.state.add(record)
        mock_lifecycle = MagicMock()
        mock_allocator = MagicMock()
        manager._allocator = mock_allocator

        def _side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                return MagicMock(returncode=128, stderr="fatal: unable to stat")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git", side_effect=_side_effect
            ),
            patch(
                "lib_python_worktree.core.teardown.os.path.exists", return_value=False
            ),
            patch("lib_python_worktree.core.teardown.shutil.rmtree"),
        ):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        mock_allocator.release.assert_called_once_with(record.id)

    def test_ticket_130_stop_hook_outcome_captured_on_forced_removal(
        self, tmp_path, caplog
    ):
        """#130: record.stop_hook_outcome reports the stop: hook's outcome
        even on a force=True removal of a still-"running" environment, and
        is never left None on this path."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-t130", pids={"main": 4321})
        manager.state.add(record)
        mock_lifecycle = MagicMock()
        mock_lifecycle.stop.return_value = None

        fake_contract = WorktreeContract(
            version=1, isolation="full", stop=[Step(run="echo stopping", name="s")]
        )
        mock_runner_instance = MagicMock()
        mock_runner_instance.run.return_value = MagicMock(steps=[MagicMock()])

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
        ):
            mock_git.return_value = _ok()
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)

        assert record.stop_hook_outcome is not None
        assert record.stop_hook_outcome.status == "completed"
        assert record.stop_hook_outcome.steps_run == 1


# ---------------------------------------------------------------------------
# Part 2: boundary-condition rows
# ---------------------------------------------------------------------------

class TestMatrixBoundaryConditions:
    @pytest.mark.parametrize("force", [True, False])
    @pytest.mark.parametrize("platform", ["win32", "linux"])
    @pytest.mark.parametrize("target_absent", [True, False])
    def test_gate_a_runs_only_on_windows_with_present_target(
        self, tmp_path, force, platform, target_absent
    ):
        """Gate A's pre-flight scan runs iff (platform == win32) AND (not
        target_absent) -- regardless of force. This is the full
        force x platform x target_absent cross-product boundary the plan
        names explicitly."""
        manager = _make_manager(tmp_path)
        # A REAL directory (created iff target_absent is False) rather than a
        # patched os.path.exists -- so target_absent's own probe AND the
        # long-path-fallback/Final-guard checks stay consistent with each
        # other exactly as they do in production, with no global stdlib
        # patch needed.
        checkout_path = tmp_path / "checkout"
        if not target_absent:
            checkout_path.mkdir()
        record = _make_record(
            f"wt-boundary-{platform}-{force}-{target_absent}",
            path=str(checkout_path),
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        def _real_rmtree_side_effect(*_args, **_kwargs):
            # Ticket #137: `sys.platform` is mocked module-wide above so
            # Gate A can be exercised as "win32" on any host, but that same
            # mocked platform is read a SECOND time in
            # `_phase_filesystem_fallback`, which then builds a Windows
            # "\\?\"-prefixed extended path and hands it to `shutil.rmtree`.
            # On a real non-Windows host that string is nonsensical and
            # `shutil.rmtree` raises OSError, falling through to the
            # robocopy fallback (also unavailable there) and finally
            # `WorktreeDirLockedError` -- reproducing the reported CI
            # failure rather than exercising the fallback deletion this row
            # means to test. Redirect the call to a REAL recursive delete of
            # the actual `checkout_path` -- but only AFTER pinning that the
            # path argument the code under test actually passed genuinely
            # references `checkout_path` (see the assertions immediately
            # below). That way a regression in `_phase_filesystem_fallback`'s
            # own path-building (wrong variable, missing abspath, wrong
            # prefix, ...) fails loudly here instead of being silently
            # masked by this side effect deleting the right directory
            # regardless of what argument it was actually given. Tolerating
            # the exact *string form* of a nonsense-but-correctly-derived
            # extended path on a non-Windows host (forward slashes from
            # posixpath's `abspath`, etc.) is still fine -- that's what the
            # prefix-strip + `Path(...).resolve()` comparison below
            # normalizes away. Do not remove this as "redundant" -- it is
            # load-bearing for cross-platform CI determinism.
            called_path = _args[0] if _args else _kwargs.get("path")
            if platform == "win32":
                assert called_path.startswith("\\\\?\\"), (
                    f"expected a win32 extended-path prefix, got {called_path!r}"
                )
                stripped = Path(called_path[4:])
                assert os.path.normcase(str(stripped.resolve())) == os.path.normcase(
                    str(checkout_path.resolve())
                ), f"extended path {called_path!r} does not point at {checkout_path!r}"
            else:
                assert os.path.normcase(str(called_path)) == os.path.normcase(
                    str(checkout_path)
                ), f"rmtree called with {called_path!r}, expected {checkout_path!r}"
            if checkout_path.exists():
                _real_rmtree(str(checkout_path))

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=[],
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=_real_rmtree_side_effect,
            ),
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = platform
            manager._teardown(record, force=force, _lifecycle_module=mock_lifecycle)

        should_scan = (platform == "win32") and not target_absent
        if should_scan:
            mock_find.assert_called_once_with(record.path, os.getpid())
        else:
            mock_find.assert_not_called()

        if not target_absent:
            # Pin that the filesystem-fallback seam actually ran (and
            # actually deleted the real directory) on both simulated
            # platforms, rather than this row passing "by accident" because
            # nothing on disk needed cleaning up.
            assert not checkout_path.exists()

    @pytest.mark.parametrize(
        "scenario",
        ["contract_absent", "contract_unparseable", "isolation_none"],
    )
    def test_stop_hook_outcome_contract_states(self, tmp_path, scenario):
        """record.stop_hook_outcome.contract_found / contract_isolation
        reflect each of the three contract-loading states correctly."""
        manager = _make_manager(tmp_path)
        record = _make_record(f"wt-contract-{scenario}")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        patches = [patch("lib_python_worktree.core.teardown._run_git")]
        if scenario == "contract_unparseable":
            patches.append(
                patch(
                    "lib_python_worktree.core.teardown._load_contract",
                    side_effect=RuntimeError("bad yaml"),
                )
            )
        elif scenario == "isolation_none":
            patches.append(
                patch(
                    "lib_python_worktree.core.teardown._load_contract",
                    return_value=WorktreeContract(version=1, isolation="none"),
                )
            )
        # contract_absent: no patch -- real loader sees a nonexistent path.

        with patches[0] as mock_git:
            mock_git.return_value = _ok()
            if len(patches) > 1:
                with patches[1]:
                    manager._teardown(
                        record, force=False, _lifecycle_module=mock_lifecycle
                    )
            else:
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )

        outcome = record.stop_hook_outcome
        assert outcome is not None
        if scenario == "contract_unparseable":
            assert outcome.status == "failed"
        else:
            assert outcome.status == "skipped"
        if scenario == "isolation_none":
            assert outcome.contract_isolation == "none"

    def test_teardown_ran_already_true_skips_steps_on_first_reachable_pass(
        self, tmp_path
    ):
        """A record that already carries teardown_ran=True (e.g. rehydrated
        from a prior partial run) never re-executes teardown: steps, even
        on what looks like the removal's first pass in THIS call."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-already-ran", teardown_ran=True)
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        fake_contract = WorktreeContract(
            version=1, isolation="full", teardown=[Step(run="echo bye", name="bye")]
        )
        mock_runner_instance = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown._load_contract",
                return_value=fake_contract,
            ),
            patch(
                "lib_python_worktree.setup.runner.SetupRunner",
                return_value=mock_runner_instance,
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
        ):
            mock_git.return_value = _ok()
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        mock_runner_instance.run.assert_not_called()
