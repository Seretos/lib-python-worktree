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
import stat
import sys
import time

from pathlib import Path
from unittest.mock import MagicMock, patch

# Ticket #137: bind the REAL shutil.rmtree function object now, before any
# test patches "lib_python_worktree.core.teardown.shutil.rmtree" (which
# patches the attribute on this same `shutil` module, since teardown.py just
# does a plain `import shutil`). A lookup of `shutil.rmtree` performed later,
# inside a test, would resolve to the patched mock instead of the real
# function and recurse infinitely.
_real_rmtree = shutil.rmtree
# Same idiom as _real_rmtree above: this file's autouse `_no_real_settle_sleep`
# fixture patches "lib_python_worktree.core.teardown.time.sleep", which -- since
# teardown.py just does `import time` -- patches the sleep attribute on the SAME
# global `time` module this test file's own `import time` sees. A test that
# needs a genuinely real sleep (e.g. to simulate an unbounded hang) must bind
# the real function now, before any test patches it.
_real_time_sleep = time.sleep

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
from lib_python_worktree.core import teardown


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


def _win32_rmtree_side_effect(checkout_path: Path):
    """Ticket #140 fix, same idiom as ``test_gate_a_runs_only_on_windows_
    with_present_target``'s ticket #137 comment above: a row that mocks
    ``sys.platform`` to "win32" on a real (possibly non-Windows) host AND
    creates a real ``checkout_path`` on disk drives
    ``_phase_filesystem_fallback`` into building a genuine Windows
    "\\?\"-prefixed extended path and handing it to ``shutil.rmtree``. On
    real Windows that deletes the directory; on a non-Windows CI host the
    string is nonsensical, ``shutil.rmtree`` raises OSError, the robocopy
    fallback is also unavailable there, and the directory survives --
    which then makes ``_phase_final_guard`` raise ``WorktreeDirLockedError``
    for a reason that has nothing to do with the behaviour the row is
    actually pinning. Redirect the mocked call to a REAL recursive delete
    of ``checkout_path``, but only after asserting the extended-path
    argument the code under test built genuinely resolves to it -- so a
    regression in the path-building itself still fails loudly instead of
    being silently papered over. This makes the row's outcome independent
    of the host OS -- host-independent-by-construction, rather than a
    host-conditional skip.
    """

    def _side_effect(*args, **kwargs):
        called_path = args[0] if args else kwargs.get("path")
        assert called_path.startswith("\\\\?\\"), (
            f"expected a win32 extended-path prefix, got {called_path!r}"
        )
        stripped = Path(called_path[4:])
        assert os.path.normcase(str(stripped.resolve())) == os.path.normcase(
            str(checkout_path.resolve())
        ), f"extended path {called_path!r} does not point at {checkout_path!r}"
        if checkout_path.exists():
            _real_rmtree(str(checkout_path))

    return _side_effect


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

    def test_ticket_148_gate_a_blind_confirming_rescan_does_not_clear_capped_hit(
        self, tmp_path, caplog
    ):
        """#148: Gate A's confirming settle-window rescan must not treat a
        handle_scan:capped-tagged result as proof the pending foreign
        blocker went away -- that tag means the scan gave up early (the
        process-wide wedged-worker cap was hit), not that it genuinely
        looked and found nothing. Before this ticket, only the literal
        string "open_files:degraded" was special-cased here, so a
        handle_scan:capped-tagged (empty) rescan result was silently
        treated as "scan succeeded, foreign hit gone" and removal proceeded
        instead of being refused.

        Also asserts (test-critic finding) that the confirming-rescan
        warning actually NAMES the matching tag ("handle_scan:capped"),
        not just some generic "blind"/"degraded" wording -- an operator
        reading the log must be able to tell which specific condition
        fired."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-t148-capped", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=15148, name="notepad.exe", cmdline=["notepad.exe"],
            source="orphan_scan",
        )
        blind_rescan = _PartialList(
            [], skipped_passes=("handle_scan:capped",)
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                side_effect=[_PartialList([hit]), blind_rescan],
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            caplog.at_level(
                logging.WARNING, logger="lib_python_worktree.core.manager"
            ),
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError):
                manager._teardown(
                    record, force=False, kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert any(
            "handle_scan:capped" in rec.message for rec in caplog.records
        ), (
            "expected the confirming-rescan warning to name the specific "
            "blind-scan tag (handle_scan:capped), not just a generic "
            "'degraded'/'blind' message"
        )

    def test_ticket_148_gate_a_blind_confirming_rescan_does_not_clear_busy_hit(
        self, tmp_path, caplog
    ):
        """#148: same guard as above, for handle_scan:busy (a contended
        scan-level lock gave up immediately -- it never looked at all, so
        an empty result from it is no evidence the foreign hit went
        away). Also asserts the confirming-rescan warning names this tag
        specifically (test-critic finding, mirrors the capped sibling
        test above)."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-t148-busy", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=15149, name="notepad.exe", cmdline=["notepad.exe"],
            source="orphan_scan",
        )
        blind_rescan = _PartialList(
            [], skipped_passes=("handle_scan:busy",)
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                side_effect=[_PartialList([hit]), blind_rescan],
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            caplog.at_level(
                logging.WARNING, logger="lib_python_worktree.core.manager"
            ),
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError):
                manager._teardown(
                    record, force=False, kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert any(
            "handle_scan:busy" in rec.message for rec in caplog.records
        ), (
            "expected the confirming-rescan warning to name the specific "
            "blind-scan tag (handle_scan:busy), not just a generic "
            "'degraded'/'blind' message"
        )


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

        # Ticket #140: the new all-platform, warn-only orphan-scan phase
        # (index 6) ALSO calls _find_blocking_processes, on every platform,
        # whenever the target is present -- so the expected call count is
        # no longer a simple win32-only single call. win32+present now pays
        # TWO full scans (Gate A's own pre-flight, win32-only, PLUS the new
        # phase); linux+present pays exactly ONE (the new phase only, since
        # Gate A never runs there); either platform with target_absent pays
        # ZERO (both Gate A -- ticket #127 -- and the new phase skip an
        # absent target).
        if target_absent:
            expected_calls = 0
        elif platform == "win32":
            expected_calls = 2
        else:
            expected_calls = 1

        assert mock_find.call_count == expected_calls, (
            f"platform={platform} target_absent={target_absent}: "
            f"expected {expected_calls} _find_blocking_processes call(s), "
            f"got {mock_find.call_count}"
        )
        if expected_calls:
            mock_find.assert_called_with(record.path, os.getpid())
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


# ---------------------------------------------------------------------------
# Part 3: ticket #140 -- new warn-only orphan-scan phase (R1-R7, R10, R11)
# ---------------------------------------------------------------------------

class TestMatrixOrphanScan:
    """Ticket #140: the new all-platform, warn-only orphan-detection phase
    (``_phase_orphan_scan``, index 6) plus the ``killed_pids``
    merge-not-overwrite fix at ``_resolve_lock_or_raise`` and Gate A.

    Platform choice rationale (repeated per-test where it matters, per the
    plan-critic note): tests that only need the NEW phase deliberately use
    ``platform="linux"`` so ``_phase_gate_a_blocking_preflight`` (win32-only)
    never runs and cannot intercept/consume the mocked scan hit meant for
    the new phase. Tests that specifically need to exercise BOTH Gate A and
    the new phase together (R3's win32 additional coverage, R7's Gate A
    merge row, R5's two best-effort rows -- see each docstring) use
    ``platform="win32"`` on purpose, and say so.
    """

    # -- R1: warn-only report, flag unset ------------------------------------

    def test_140_orphan_hit_without_kill_flag_warns_and_removes(
        self, tmp_path, caplog
    ):
        """R1 driving test: an untracked process holding the checkout is
        reported on record.orphan_scan, never silently orphaned, and never
        refuses the removal, when kill_blocking_processes=False.

        Platform: linux, deliberately -- isolates the assertion to the new
        phase; Gate A (win32-only) never runs here so it cannot intercept
        the mocked hit meant for the new phase.
        """
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r1", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=15501, name="notepad.exe", cmdline=["notepad.exe"],
            source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([hit]),
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes"
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            with caplog.at_level(
                logging.WARNING, logger="lib_python_worktree.core.manager"
            ):
                manager._teardown(
                    record, force=False, kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert record.orphan_scan is not None
        assert len(record.orphan_scan.entries) == 1
        entry = record.orphan_scan.entries[0]
        assert entry.killed is False
        assert entry.owned is False
        assert record.killed_pids == []
        assert "scan:failed" not in record.orphan_scan.skipped_passes, (
            "a clean, non-degraded scan must never carry the failure "
            "marker -- an implementation that always appends it would "
            "wrongly pass a looser check here"
        )
        mock_kill.assert_not_called()
        # Message parity (content-bearing, not merely "some string was
        # logged"): the report's message is the exact string logged, naming
        # the worktree id and the hit count -- not just a non-empty value.
        assert record.orphan_scan.message == (
            f"_teardown: worktree '{record.id}' orphan scan found "
            f"1 untracked process(es) holding the checkout."
        )
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert len(warning_records) == 1, (
            f"expected exactly one WARNING log record for this abnormal "
            f"outcome, got {[r.message for r in warning_records]!r}"
        )
        assert warning_records[0].message == record.orphan_scan.message

    @pytest.mark.parametrize("owned", [True, False])
    def test_140_orphan_entry_owned_flag_round_trips(self, tmp_path, owned):
        """R1 additional coverage: entry.owned reflects whether the pid is
        in ctx.owned_pids (record.pids at context-build time)."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        hit_pid = 15511
        record = _make_record(
            f"wt-140-r1-owned-{owned}", path=str(checkout_path),
            pids={"main": hit_pid} if owned else {},
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=hit_pid, name="proc.exe", cmdline=["proc"], source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([hit]),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert record.orphan_scan is not None
        assert record.orphan_scan.entries[0].owned is owned

    def test_140_orphan_entry_owned_flag_is_per_pid_within_one_scan(
        self, tmp_path
    ):
        """R1 additional coverage (test-critic MAJOR finding): a single
        scan can contain BOTH an owned and an unowned hit at once. An
        implementation that computes ``bool(ctx.owned_pids)`` once and
        stamps that single value onto every entry (rather than checking
        each entry's own pid membership) would incorrectly mark the
        unowned hit as owned too here -- the two single-hit parametrized
        rows above (owned=True xor owned=False) cannot distinguish that
        bug from correct per-pid behaviour, because each of those scans
        only ever contains one hit."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        owned_pid = 15521
        unowned_pid = 15522
        record = _make_record(
            "wt-140-r1-mixed-owned",
            path=str(checkout_path),
            pids={"main": owned_pid},
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        owned_hit = KilledProcessInfo(
            pid=owned_pid, name="owned.exe", cmdline=["owned"],
            source="orphan_scan",
        )
        unowned_hit = KilledProcessInfo(
            pid=unowned_pid, name="foreign.exe", cmdline=["foreign"],
            source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([owned_hit, unowned_hit]),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert record.orphan_scan is not None
        by_pid = {e.info.pid: e for e in record.orphan_scan.entries}
        assert set(by_pid) == {owned_pid, unowned_pid}
        assert by_pid[owned_pid].owned is True
        assert by_pid[unowned_pid].owned is False

    def test_140_orphan_scan_hit_list_is_uncapped(self, tmp_path):
        """R1 additional coverage: unlike StopDetail's survivor cap, the
        orphan-scan hit list is deliberately uncapped."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r1-uncapped", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        many_hits = _PartialList(
            [
                KilledProcessInfo(
                    pid=16000 + i, name=f"p{i}.exe", cmdline=[f"p{i}"],
                    source="orphan_scan",
                )
                for i in range(25)
            ]
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=many_hits,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert record.orphan_scan is not None
        assert len(record.orphan_scan.entries) == 25

    # -- R2: all-platform scan, including POSIX ------------------------------

    def test_140_orphan_scan_runs_on_posix(self, tmp_path):
        """R2 driving test: the phase scans on POSIX, where Gate A never
        runs at all."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r2", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=15601, name="vim", cmdline=["vim"], source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([hit]),
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        mock_find.assert_called_once_with(record.path, os.getpid())
        assert record.orphan_scan is not None

    @pytest.mark.parametrize("platform", ["win32", "linux"])
    def test_140_orphan_scan_skipped_when_target_absent(self, tmp_path, platform):
        """R2 additional coverage: target_absent short-circuits the new
        phase on EITHER platform -- zero scan calls attributable to it,
        orphan_scan stays None."""
        manager = _make_manager(tmp_path)
        record = _make_record(f"wt-140-r2-absent-{platform}")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes"
            ) as mock_find,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.teardown.os.path.exists",
                return_value=False,
            ),
            patch("lib_python_worktree.core.teardown.shutil.rmtree"),
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = platform
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        mock_find.assert_not_called()
        assert record.orphan_scan is None

    # -- R3: kill_blocking_processes=True kills on POSIX ---------------------

    def test_140_kill_flag_kills_heuristic_hit_on_posix(self, tmp_path):
        """R3 driving test: kill_blocking_processes becomes meaningful on
        POSIX for the first time."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r3", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=15701, name="vim", cmdline=["vim"], source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([hit]),
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=_PartialList([hit]),
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_called_once_with(record.path)
        assert record.orphan_scan is not None
        assert len(record.orphan_scan.entries) == 1
        assert record.orphan_scan.entries[0].killed is True
        assert [info.pid for info in record.killed_pids] == [15701]
        assert record.orphan_scan.kill_attempted is True, (
            "OrphanScanReport.kill_attempted must be set when the kill was "
            "actually attempted"
        )

    def test_140_kill_flag_kills_heuristic_hit_on_win32(self, tmp_path):
        """R3 additional coverage: same behaviour on win32, where Gate A
        ALSO runs. Isolated via a 2-call side_effect so Gate A's own
        pre-flight scan (first call, empty) doesn't intercept the hit meant
        for the new phase (second call)."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r3-win32", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=15801, name="notepad.exe", cmdline=["notepad.exe"],
            source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                side_effect=[_PartialList([]), _PartialList([hit])],
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=_PartialList([hit]),
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=_win32_rmtree_side_effect(checkout_path),
            ),
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=False, kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_called_once_with(record.path)
        assert record.orphan_scan is not None
        assert record.orphan_scan.entries[0].killed is True

    def test_140_only_killed_entries_merge_into_killed_pids(self, tmp_path):
        """R3 additional coverage: a KilledProcessInfo(killed=False) (the
        flag-unset case) never appears in record.killed_pids."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r3-nokill", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=15901, name="vim", cmdline=["vim"], source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([hit]),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert record.killed_pids == []

    # -- R4: clean scan skips the kill call -----------------------------------

    def test_140_kill_flag_with_clean_scan_skips_kill_call(self, tmp_path):
        """R4 driving test: no hits => the ~5s _kill_blocking_processes call
        is skipped entirely even with the flag set."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r4", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([]),
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes"
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        mock_kill.assert_not_called()
        assert record.orphan_scan is None

    # -- R5: best-effort -- neither scan nor kill can fail a removal --------

    def test_140_scan_exception_degrades_to_scan_failed(self, tmp_path):
        """R5 row 1 driving test. Platform: win32, deliberately (plan-critic
        note) -- today (pre-#140), _find_blocking_processes is called ONLY
        by Gate A's win32-only pre-flight; on POSIX there is no call site at
        all yet, so a POSIX side_effect would never fire and this test
        would not RED for the intended reason. On win32, Gate A's own
        (pre-existing, unguarded) call to _find_blocking_processes is what
        makes the injected RuntimeError propagate today."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r5-scan", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                side_effect=RuntimeError(
                    "SystemExtendedHandleInformation buffer too big"
                ),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=_win32_rmtree_side_effect(checkout_path),
            ),
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "win32"
            # Must not raise -- the phase's own best-effort `except
            # Exception` degrades this to a "scan:failed" marker instead of
            # propagating (ticket #107's bare RuntimeError guard).
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert record.orphan_scan is not None
        assert "scan:failed" in record.orphan_scan.skipped_passes

    def test_140_kill_exception_degrades_but_keeps_scan_hits(self, tmp_path):
        """R5 row 2 driving test. Platform: win32, same rationale as row 1
        (plan-critic note) -- today (pre-#140), _kill_blocking_processes'
        only caller is Gate A's own win32-only confirmed-blocker remedy, so
        this is the platform where an injected kill exception actually
        propagates before the new best-effort phase exists. The hit's pid
        is OWNED (in record.pids) so Gate A confirms it immediately, with
        no settle-window rescan to account for."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        owned_pid = 16001
        record = _make_record(
            "wt-140-r5-kill", path=str(checkout_path), pids={"main": owned_pid},
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=owned_pid, name="daemon.exe", cmdline=["daemon"],
            source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([hit]),
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                side_effect=RuntimeError("boom"),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=_win32_rmtree_side_effect(checkout_path),
            ),
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=False, kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        assert record.orphan_scan is not None
        assert "scan:failed" in record.orphan_scan.skipped_passes
        assert any(e.info.pid == owned_pid for e in record.orphan_scan.entries)

    def test_140_skipped_passes_passthrough_on_clean_degraded_scan(self, tmp_path):
        """R5 additional coverage: a degraded-but-empty scan result
        (skipped_passes non-empty, no hits) still counts as an "abnormal
        outcome" -- record.orphan_scan is non-None and carries the tag,
        even though there is nothing to warn about by pid."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r5-passthrough", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        degraded_empty = _PartialList(
            [], skipped_passes=("open_files:degraded",)
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=degraded_empty,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert record.orphan_scan is not None
        assert "open_files:degraded" in record.orphan_scan.skipped_passes

    # -- R6: union semantics between warn scan and kill's own rescan --------

    def test_140_report_is_union_of_warn_scan_and_kill_result(self, tmp_path):
        """R6 driving test: reported set = union of the warn scan and the
        kill's own tighter rescan; a pid found by one but not the other is
        still representable with the right killed flag."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r6", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        pid_a = 16101
        pid_b = 16102
        scan_hit = KilledProcessInfo(
            pid=pid_a, name="a.exe", cmdline=["a"], source="orphan_scan",
        )
        kill_hit = KilledProcessInfo(
            pid=pid_b, name="b.exe", cmdline=["b"], source="tree",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([scan_hit]),
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=_PartialList([kill_hit]),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        assert record.orphan_scan is not None
        by_pid = {e.info.pid: e for e in record.orphan_scan.entries}
        assert set(by_pid) == {pid_a, pid_b}
        assert by_pid[pid_a].killed is False
        assert by_pid[pid_b].killed is True
        assert [info.pid for info in record.killed_pids] == [pid_b]

    # -- R7: killed_pids merges instead of overwriting -----------------------

    def test_140_resolve_lock_merge_preserves_orphan_phase_kills(self, tmp_path):
        """R7 row 1 driving test: _resolve_lock_or_raise's kill-and-retry
        remedy (the primary `git worktree remove` attempt's own lock-signal
        path) must MERGE with a pid an earlier phase already recorded
        (simulated here via direct pre-population, exactly like an earlier
        stop() call would have left record.killed_pids -- see row 2's
        docstring for why this simulates "an earlier phase already killed
        X" without needing the new phase to exist yet)."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-140-r7-resolve")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        pid_x = KilledProcessInfo(
            pid=9001, name="x.exe", cmdline=["x"], source="orphan_scan",
        )
        record.killed_pids = [pid_x]

        pid_y = KilledProcessInfo(
            pid=9002, name="y.exe", cmdline=["y"], source="orphan_scan",
        )

        call_count = [0]

        def _git_side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["worktree", "remove"]:
                call_count[0] += 1
                if call_count[0] == 1:
                    return MagicMock(
                        returncode=1, stderr="fatal: worktree is locked"
                    )
                return MagicMock(returncode=0, stderr="")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=_PartialList([pid_y]),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.teardown.os.path.exists",
                return_value=False,
            ),
        ):
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        pids = [info.pid for info in record.killed_pids]
        assert pids == [9001, 9002], (
            f"expected merged [X, Y] pids (X preserved from before the "
            f"lock-retry remedy ran), got {pids}"
        )

    def test_140_gate_a_merge_preserves_prior_killed_pids(self, tmp_path):
        """R7 row 2 driving test: Gate A's own confirmed-blocker kill must
        MERGE with a pid a prior stop() call already recorded on
        record.killed_pids, not overwrite it -- and a pid the Gate A kill
        rediscovers (already present) must not be doubled or reordered."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r7-gate-a", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        pid_z = KilledProcessInfo(
            pid=6001, name="z.exe", cmdline=["z"], source="tracked",
        )
        record.killed_pids = [pid_z]  # simulates a prior stop() call

        blocker = KilledProcessInfo(
            pid=6001, name="z.exe", cmdline=["z"], source="orphan_scan",
        )
        # The Gate A kill result rediscovers Z (pid 6001, must not double or
        # reorder it) AND finds a genuinely new pid X (7001) -- deliberately
        # ordered [X, Z] so a naive overwrite (record.killed_pids = killed)
        # would visibly reorder Z to second place, distinguishing a correct
        # first-wins merge from both "overwrite" and "naive unmerged concat".
        kill_result = _PartialList([
            KilledProcessInfo(pid=7001, name="x.exe", cmdline=["x"], source="orphan_scan"),
            KilledProcessInfo(pid=6001, name="z.exe", cmdline=["z"], source="orphan_scan"),
        ])

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([blocker]),
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=kill_result,
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=_win32_rmtree_side_effect(checkout_path),
            ),
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "win32"
            manager._teardown(
                record, force=False, kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        pids = [info.pid for info in record.killed_pids]
        assert pids == [6001, 7001], (
            f"expected merged, deduped, first-wins [Z, X] pids, got {pids}"
        )

    def test_140_seretos_exemption_retry_merge_preserves_prior_killed_pids(
        self, tmp_path
    ):
        """R7 shared-call-site row: _merge_killed_pids must ALSO be
        exercised via the .seretos/-exemption auto-force retry's OWN call
        into _resolve_lock_or_raise (ticket #100) -- the function's SECOND
        caller, distinct from row 1's primary-attempt call."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-140-r7-seretos")
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        pid_z = KilledProcessInfo(
            pid=6101, name="z.exe", cmdline=["z"], source="tracked",
        )
        record.killed_pids = [pid_z]

        pid_x = KilledProcessInfo(
            pid=6102, name="x.exe", cmdline=["x"], source="orphan_scan",
        )

        call_count = [0]

        def _git_side_effect(args, cwd=None, **kwargs):
            if args == ["worktree", "remove", record.path]:
                return MagicMock(
                    returncode=128,
                    stderr="fatal: contains modified or untracked files",
                )
            if args[:2] == ["status", "--porcelain"]:
                return MagicMock(returncode=0, stdout="?? .seretos\x00")
            if args == ["worktree", "remove", "--force", record.path]:
                call_count[0] += 1
                if call_count[0] == 1:
                    return MagicMock(
                        returncode=1, stderr="fatal: worktree is locked"
                    )
                return MagicMock(returncode=0, stderr="")
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                return_value=_PartialList([pid_x]),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "linux"
            manager._teardown(
                record, force=False, kill_blocking_processes=True,
                _lifecycle_module=mock_lifecycle,
            )

        pids = [info.pid for info in record.killed_pids]
        assert pids == [6101, 6102], (
            f"expected merged [Z, X] via the .seretos/-exemption retry's "
            f"own _resolve_lock_or_raise call, got {pids}"
        )

    # -- R10: remove() copies orphan_scan forward -----------------------------

    def test_140_remove_returns_orphan_scan_on_tracked_path(self, tmp_path):
        """R10 driving test: manager.remove()'s return value carries the
        report even though a YAML-backed state.remove() returns a freshly
        deserialized object that never carries transient fields.

        Verified via content (populated report, correct entry) rather than
        object identity: the internal WorktreeRecord _teardown() mutates is
        not reachable from outside manager.remove()'s public surface, so an
        `is`-identity assertion against it is not observable here."""
        from lib_python_worktree.core.yaml_store import YamlStateStore

        state_dir = tmp_path / "state"
        manager = WorktreeManager(
            config=ManagerConfig(store_root=tmp_path / "store"),
            state=YamlStateStore(state_dir=state_dir),
            reconcile_on_init=False,
        )
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r10", path=str(checkout_path))
        manager.state.add(record)

        hit = KilledProcessInfo(
            pid=16201, name="a.exe", cmdline=["a"], source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([hit]),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            removed = manager.remove(
                record.id, force=False, kill_blocking_processes=False,
            )

        assert removed.orphan_scan is not None
        assert len(removed.orphan_scan.entries) == 1
        assert removed.orphan_scan.entries[0].info.pid == 16201

    def test_140_remove_untracked_path_carries_orphan_scan_without_copy(
        self, tmp_path
    ):
        """R10 additional coverage: the untracked (ticket #88) removal path
        never writes to the store, so the returned record IS the same
        object _teardown() mutated -- no copy-forward line is needed there.
        Isolated via a direct patch of _resolve_removal_target rather than
        a real list_repo()/git discovery round-trip (consistent with this
        file's mocking-only style)."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r10-untracked", path=str(checkout_path))
        # Deliberately NOT added to manager.state.

        hit = KilledProcessInfo(
            pid=16301, name="b.exe", cmdline=["b"], source="orphan_scan",
        )

        with (
            patch.object(
                manager, "_resolve_removal_target", return_value=(record, False)
            ),
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([hit]),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            removed = manager.remove(checkout_path=str(checkout_path))

        assert removed is record
        assert removed.orphan_scan is not None

    # -- R11 (part a): pin the real trigger split -----------------------------

    def test_140_gate_a_confirmed_blocker_not_bypassed_by_force(self, tmp_path):
        """R11 additional coverage row (already correct today): force=True
        does not substitute for kill_blocking_processes=True at Gate A's
        confirmed-blocker refusal -- characterization pin."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        owned_pid = 16401
        record = _make_record(
            "wt-140-r11-gate-a", path=str(checkout_path),
            pids={"main": owned_pid},
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=owned_pid, name="daemon.exe", cmdline=["daemon"],
            source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([hit]),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "win32"
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record, force=True, kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert excinfo.value.kill_attempted is False

    def test_140_gate_b_dirt_not_bypassed_by_kill_flag(self, tmp_path):
        """R11 additional coverage row (already correct today):
        kill_blocking_processes=True does not substitute for force=True at
        Gate B's early dirty-tree refusal -- characterization pin."""
        manager = _make_manager(tmp_path)
        record = _make_record("wt-140-r11-gate-b")
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
                    record, force=False, kill_blocking_processes=True,
                    _lifecycle_module=mock_lifecycle,
                )

    def test_140_posix_triggers_neither_gate(self, tmp_path):
        """R11 driving test: on POSIX, neither Gate A (win32-only) nor
        Gate B (dirt-only) fires for a blocking PROCESS -- pre-#140 this
        silently orphans the process (the ticket's reported defect);
        post-#140 the new warn-only phase (R1) is what surfaces it
        instead. This is the row that falsifies the ticket's "maybe
        Linux/Mac-specific" hypothesis: POSIX genuinely triggers neither
        gate, but the new phase now reports what they'd have missed."""
        manager = _make_manager(tmp_path)
        checkout_path = tmp_path / "checkout"
        checkout_path.mkdir()
        record = _make_record("wt-140-r11-posix", path=str(checkout_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        hit = KilledProcessInfo(
            pid=16501, name="vim", cmdline=["vim"], source="orphan_scan",
        )

        with (
            patch("lib_python_worktree.core.teardown._run_git") as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                return_value=_PartialList([hit]),
            ),
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_git.return_value = _ok()
            mock_sys.platform = "linux"
            # Must not raise -- POSIX triggers neither gate.
            manager._teardown(
                record, force=False, kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        assert record.orphan_scan is not None
        assert any(e.info.pid == 16501 for e in record.orphan_scan.entries)


# ---------------------------------------------------------------------------
# #154 -- rename-is-the-oracle removal mechanism (generation-2 plan.md,
# amended by the human override note: no _phase_reclaim_staged; the
# undo-failure leg hard-raises and leaves the remnant; ctx.failure_deadline
# stays None on every DirtyWorktreeError leg, staged=True included).
# ---------------------------------------------------------------------------

def _git_side_effect_deletes_on_worktree_remove(checkout: Path, calls: list):
    """A ``_run_git`` stand-in that also mirrors real git's destructive
    effect for a ``worktree remove`` call, so the final-guard's real
    ``os.path.exists`` check sees an accurate post-removal filesystem state
    against TODAY'S code path (mocked git normally leaves the directory
    behind, which would make an unrelated WorktreeDirLockedError mask the
    assertion this test actually cares about)."""

    def _side_effect(args, cwd=None, **kwargs):
        calls.append(list(args))
        if args[:2] == ["worktree", "remove"]:
            shutil.rmtree(str(checkout), ignore_errors=True)
        return _ok()

    return _side_effect


class TestMatrixRemovalMechanism:
    """New scenarios for #154. Supersedes v0.3.12's Gate A / orphan-scan /
    `git worktree remove` / filesystem-fallback mechanism with rename ->
    rmtree -> `git worktree prune --expire=now`, per the human override note
    (`.adev/154-1/plan-final-human-override.md`) amending generation-2
    plan.md (`.adev/154-1/plan.md`)."""

    # -- R1 (AC2): the happy path never scans -------------------------------

    def test_happy_path_never_scans(self, tmp_path):
        """A clean, unheld worktree's removal performs zero systemwide
        process/handle scans and spawns no non-git subprocess. RED today:
        Gate A (win32) and the orphan-scan phase (both platforms) each call
        `_find_blocking_processes` unconditionally."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r1"
        checkout.mkdir()
        record = _make_record("wt-r1", path=str(checkout), repo_root=str(tmp_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        git_calls: list = []

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect_deletes_on_worktree_remove(
                    checkout, git_calls
                ),
            ) as mock_git,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                MagicMock(return_value=[]),
            ) as mock_scan,
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                MagicMock(return_value=[]),
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown.subprocess.run") as mock_run,
            patch("lib_python_worktree.core.teardown.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert mock_scan.call_count == 0, (
            f"_find_blocking_processes must never be called on the happy "
            f"path (AC2); was called {mock_scan.call_count} time(s)"
        )
        assert mock_kill.call_count == 0
        assert mock_run.call_count == 0, "no robocopy (no non-git subprocess)"
        assert not checkout.exists()

    # -- R4: removal is rename-based; git is only pruned ---------------------

    def test_rename_then_rmtree_then_prune(self, tmp_path):
        """Removal never calls `git worktree remove`; it stages via
        `os.rename`, deletes the staged tree, then prunes with
        `--expire=now`. RED today: `worktree remove` is called and no
        `worktree prune --expire=now` call is ever made on the happy path."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r4"
        checkout.mkdir()
        (checkout / "file.txt").write_text("hello")
        record = _make_record("wt-r4", path=str(checkout), repo_root=str(tmp_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        git_calls: list = []
        with patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=_git_side_effect_deletes_on_worktree_remove(
                checkout, git_calls
            ),
        ):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert not checkout.exists()
        assert not Path(str(checkout) + ".removing").exists()
        assert ["worktree", "remove", str(checkout)] not in git_calls
        assert ["worktree", "remove", "--force", str(checkout)] not in git_calls
        assert ["worktree", "prune", "--expire=now"] in git_calls, (
            f"expected an unconditional `git worktree prune --expire=now` "
            f"call; got calls={git_calls}"
        )

    def test_prune_failure_still_reports_removed(self, tmp_path):
        """A failed prune (stale git bookkeeping) must not turn an
        otherwise-complete filesystem removal into a raised exception --
        `_phase_git_prune` warn-and-continues. RED today: prune is never
        called on this path at all, so there is nothing to make fail this
        way -- the checkout is deleted via `git worktree remove` instead."""
        from lib_python_worktree.core._exceptions import GitTimeoutError

        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r4-prune-fail"
        checkout.mkdir()
        record = _make_record(
            "wt-r4-prune-fail", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        calls: list = []

        def _side_effect(args, cwd=None, **kwargs):
            calls.append(list(args))
            if args[:2] == ["worktree", "prune"]:
                raise GitTimeoutError(args, elapsed=5.0)
            if args[:2] == ["worktree", "remove"]:
                shutil.rmtree(str(checkout), ignore_errors=True)
            return _ok()

        with patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=_side_effect,
        ):
            # Must NOT raise despite the prune call failing.
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert any(c[:2] == ["worktree", "prune"] for c in calls), (
            f"expected an unconditional `worktree prune` call that could "
            f"then be made to fail; got calls={calls}"
        )
        assert not checkout.exists()

    # -- R6: bounded rename retry replaces the settle rescan (P1) ------------

    def test_rename_retries_within_budget(self, tmp_path):
        """A transient rename failure (AV/indexer holding a handle for a
        moment) must be absorbed by a short bounded retry loop -- never by
        going straight to a systemwide scan. RED today: `os.rename` is never
        called at all (removal goes through `git worktree remove`), so the
        retry can never be observed."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r6"
        checkout.mkdir()
        record = _make_record("wt-r6", path=str(checkout), repo_root=str(tmp_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        real_rename = os.rename
        attempts = {"n": 0}

        def _flaky_rename(src, dst, *a, **kw):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise PermissionError("simulated transient AV lock")
            return real_rename(src, dst, *a, **kw)

        with (
            patch("lib_python_worktree.core.teardown.os.rename", side_effect=_flaky_rename),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                MagicMock(return_value=[]),
            ) as mock_scan,
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                MagicMock(return_value=[]),
            ) as mock_kill,
            patch(
                "lib_python_worktree.core.teardown._run_git",
                return_value=_ok(),
            ),
        ):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert attempts["n"] == 3, (
            f"expected exactly 3 rename attempts (2 transient failures + 1 "
            f"success); got {attempts['n']}"
        )
        assert mock_scan.call_count == 0, "a transient rename failure must never escalate to a scan"
        assert mock_kill.call_count == 0

    def test_transient_retry_budget_is_exhausted_in_exact_steps(self, tmp_path):
        """Budget pinning: a rename that never succeeds burns exactly the
        transient-retry budget (2.0s / 0.1s steps = 20 sleeps, 21 attempts)
        before entering diagnosis -- not an unbounded loop, not zero
        retries. RED today: `os.rename` is never called at all."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r6-exhaust"
        checkout.mkdir()
        record = _make_record(
            "wt-r6-exhaust", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        clock = {"t": 0.0}
        sleeps: list = []

        def _fake_sleep(seconds):
            sleeps.append(seconds)
            clock["t"] += seconds

        def _fake_monotonic():
            return clock["t"]

        with (
            patch(
                "lib_python_worktree.core.teardown.os.rename",
                side_effect=PermissionError("always locked"),
            ) as mock_rename,
            patch("lib_python_worktree.core.teardown.time.sleep", side_effect=_fake_sleep),
            patch("lib_python_worktree.core.teardown.time.monotonic", side_effect=_fake_monotonic),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                MagicMock(return_value=[]),
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                MagicMock(return_value=[]),
            ),
            patch("lib_python_worktree.core.teardown._run_git", return_value=_ok()),
        ):
            try:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )
            except Exception:
                pass  # the eventual raise isn't what this test pins

        assert mock_rename.call_count == 21, (
            f"expected exactly 21 rename attempts (1 initial + 20 retries "
            f"at the 2.0s/0.1s budget); got {mock_rename.call_count}"
        )
        assert len(sleeps) == 20
        assert all(s == 0.1 for s in sleeps)


    # -- R10: the dirt gate runs without teardown steps; git is no longer
    #    the mechanism ----------------------------------------------------

    def test_dirt_gate_runs_without_teardown_steps(self, tmp_path):
        """A contract with NO `teardown:` steps must still refuse a dirty,
        force=False removal -- via the new unconditional dirt gate, not via
        git's own exit-128 refusal. RED today: without `teardown:` steps,
        Gate B is skipped entirely (it only runs `if ... ctx.contract.teardown`),
        so nothing probes dirt before the (mocked, always-succeeding)
        `git worktree remove` call, and removal succeeds instead of
        raising."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r10"
        checkout.mkdir()
        record = _make_record("wt-r10", path=str(checkout), repo_root=str(tmp_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        def _git_side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["status", "--porcelain"]:
                return MagicMock(returncode=0, stdout="M  dirty.txt\x00")
            return _ok()

        with patch(
            "lib_python_worktree.core.teardown._run_git",
            side_effect=_git_side_effect,
        ):
            with pytest.raises(DirtyWorktreeError) as excinfo:
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )
        assert excinfo.value.staged is False
        assert checkout.exists(), "a refused removal must not touch the checkout"

    # -- R11 (amended by Decision 1 + Decision 2): combined verdict, probe
    #    cost, and the undo-failure leg -------------------------------------

    def test_dirty_and_locked_raises_removal_blocked(self, tmp_path):
        """Real dirt + a forward rename-probe that fails -> the combined
        WorktreeRemovalBlockedError verdict (#103), carrying `.blockers`
        (default []) and `staged=False`. RED today: no rename probe exists
        at all -- a dirty tree with no teardown: steps and a mocked-success
        git call proceeds to raise nothing (see test_dirt_gate_runs_without_
        teardown_steps) or, once #103's existing exit-128 codepath is hit,
        raises via a completely different (git-stderr-based) mechanism."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r11-blocked"
        checkout.mkdir()
        record = _make_record(
            "wt-r11-blocked", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        def _git_side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["status", "--porcelain"]:
                return MagicMock(returncode=0, stdout="M  dirty.txt\x00")
            return _ok()

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown.os.rename",
                side_effect=PermissionError("locked"),
            ),
        ):
            with pytest.raises(WorktreeRemovalBlockedError) as excinfo:
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )
        assert excinfo.value.dirty_paths
        assert excinfo.value.blockers == []
        assert excinfo.value.staged is False

    def test_dirt_gate_probe_is_a_bare_rename_pair_not_the_retry_loop(self, tmp_path):
        """The dirt gate's forward-then-undo probe must be a bare rename
        pair -- no sleep, no retry budget -- so a dirty, held worktree is
        diagnosed in microseconds, not ~2s. RED today: neither the probe nor
        `os.rename` exists on this path at all, so `os.rename` is never
        called and this assertion fails on that alone."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r11-probe-cost"
        checkout.mkdir()
        record = _make_record(
            "wt-r11-probe-cost", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        def _git_side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["status", "--porcelain"]:
                return MagicMock(returncode=0, stdout="M  dirty.txt\x00")
            return _ok()

        rename_calls: list = []
        real_rename = os.rename

        def _tracked_rename(src, dst, *a, **kw):
            rename_calls.append((src, dst))
            return real_rename(src, dst, *a, **kw)

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown.os.rename",
                side_effect=_tracked_rename,
            ),
            patch("lib_python_worktree.core.teardown.time.sleep") as mock_sleep,
        ):
            with pytest.raises(DirtyWorktreeError) as excinfo:
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )

        assert excinfo.value.staged is False
        assert len(rename_calls) == 2, (
            f"expected exactly one forward rename and one undo rename; "
            f"got {rename_calls}"
        )
        mock_sleep.assert_not_called()
        assert checkout.exists(), "the undo must put the tree back exactly where it was"

    def test_dirt_probe_undo_failure_raises_dirty_with_staged_not_blocked(
        self, tmp_path
    ):
        """#154 Decision 1 + Decision 2 (human override): when the forward
        probe succeeds but the undo rename cannot be undone (even after a
        bounded retry), the phase must hard-raise plain DirtyWorktreeError
        with staged=True -- NOT WorktreeRemovalBlockedError (the forward
        probe succeeding already proves the directory was not locked) -- and
        leave the `.removing` remnant on disk with no restore attempt.
        Decision 2: ctx.failure_deadline stays None on this leg even though
        a bounded retry ran, because the undo-retry must not share the
        `_retry_bounded` helper's `ctx.failure_deadline`-opening contract.
        RED today: neither the probe nor the undo mechanism exists."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r11-undo-fail"
        checkout.mkdir()
        record = _make_record(
            "wt-r11-undo-fail", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        def _git_side_effect(args, cwd=None, **kwargs):
            if args[:2] == ["status", "--porcelain"]:
                return MagicMock(returncode=0, stdout="M  dirty.txt\x00")
            return _ok()

        real_rename = os.rename
        state = {"forward_done": False}

        def _rename_side_effect(src, dst, *a, **kw):
            staged = str(checkout) + ".removing"
            if not state["forward_done"] and dst == staged:
                state["forward_done"] = True
                return real_rename(src, dst, *a, **kw)
            if state["forward_done"] and src == staged:
                raise PermissionError("undo refused (AV opened the staged dir)")
            return real_rename(src, dst, *a, **kw)

        ctx_holder: list = []
        real_build_context = teardown.build_context

        def _capturing_build_context(*a, **kw):
            ctx = real_build_context(*a, **kw)
            ctx_holder.append(ctx)
            return ctx

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_git_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown.os.rename",
                side_effect=_rename_side_effect,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_process_tree",
            ) as mock_kill_tree,
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill_blocking,
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
            ) as mock_find,
            patch(
                "lib_python_worktree.core.teardown.build_context",
                side_effect=_capturing_build_context,
            ),
        ):
            with pytest.raises(DirtyWorktreeError) as excinfo:
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )

        assert type(excinfo.value) is DirtyWorktreeError, (
            "must be plain DirtyWorktreeError, never the combined "
            "WorktreeRemovalBlockedError -- the forward probe succeeding "
            "already proves the directory was not locked"
        )
        assert excinfo.value.staged is True
        mock_kill_tree.assert_not_called()
        mock_kill_blocking.assert_not_called()
        mock_find.assert_not_called()
        assert ctx_holder, "build_context must have been called via manager._teardown"
        assert ctx_holder[0].failure_deadline is None, (
            "Decision 2: ctx.failure_deadline must stay None on every "
            "DirtyWorktreeError leg, staged=True included"
        )

    def test_staged_dirty_remnant_is_never_silently_destroyed_by_a_later_removal(
        self, tmp_path
    ):
        """#154 Decision 1 (human override, rejecting generation-2 plan.md's
        _phase_reclaim_staged): once a `.removing` remnant is left behind by
        the undo-failure leg above, a LATER force=False remove() of the same
        record must NOT silently restore-then-re-raise (that was generation-2's
        rejected design) and must NOT silently destroy the remnant either --
        it must raise (reusing the existing DirtyWorktreeError(staged=True) /
        WorktreeDirLockedError vocabulary) with the remnant's content
        untouched. Only force=True may destroy it. RED today: none of this
        mechanism exists -- a plain removal of a record whose `record.path`
        is absent takes today's target-absent fast path and reports success
        with the `.removing` directory's content silently left as garbage
        (and, moreover, `_target_is_absent` doesn't even know `.removing`
        exists, so it never surfaces to the caller at all)."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r11-datasafety"
        staged = Path(str(checkout) + ".removing")
        # Simulate the post-undo-failure state directly: record.path absent,
        # non-empty remnant present with recognisable, uncommitted content.
        staged.mkdir()
        (staged / "uncommitted.txt").write_text("precious work")
        record = _make_record(
            "wt-r11-datasafety", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        with patch(
            "lib_python_worktree.core.teardown._run_git", return_value=_ok()
        ):
            with pytest.raises((DirtyWorktreeError, WorktreeDirLockedError)) as excinfo:
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )

        assert getattr(excinfo.value, "staged", None) is True
        assert not checkout.exists(), "must not silently restore the remnant either"
        assert staged.exists() and (staged / "uncommitted.txt").read_text() == (
            "precious work"
        ), "the remnant's uncommitted content must never be silently destroyed"

        # force=True clears it, per the override's unchanged force=True rule.
        with patch(
            "lib_python_worktree.core.teardown._run_git", return_value=_ok()
        ):
            manager._teardown(record, force=True, _lifecycle_module=mock_lifecycle)
        assert not staged.exists()

    # -- R9: a remnant that cannot be cleared names `.removing`, and the
    #    non-staged phrasing is unchanged -----------------------------------

    def test_locked_staged_dir_error_names_removing(self, tmp_path):
        """A `.removing` remnant that survives the retry loop AND both
        diagnosis tiers must raise WorktreeDirLockedError(staged=True) whose
        message names `.removing` and no filesystem path. RED today: the
        whole staging mechanism does not exist -- os.rename is never called,
        so nothing ever produces a staged=True error."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r9-locked"
        checkout.mkdir()
        record = _make_record(
            "wt-r9-locked", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        with (
            patch(
                "lib_python_worktree.core.teardown.os.rename",
                side_effect=PermissionError("always locked"),
            ),
            patch("lib_python_worktree.core.teardown.time.sleep"),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                MagicMock(return_value=[]),
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                MagicMock(return_value=[]),
            ),
            patch("lib_python_worktree.core.teardown._run_git", return_value=_ok()),
        ):
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )

        assert excinfo.value.staged is True
        msg = str(excinfo.value)
        assert ".removing" in msg
        assert str(checkout) not in msg
        assert excinfo.value.blockers is not None

    def test_final_guard_on_a_surviving_original_keeps_the_v0312_message(
        self, tmp_path
    ):
        """When `record.path` itself survives the whole removal and no
        `.removing` remnant exists, the final guard's message must stay
        byte-identical to v0.3.12's kill_attempted-selected text, and
        `staged` must be False."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r9-surviving"
        checkout.mkdir()
        record = _make_record(
            "wt-r9-surviving", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        # git call "succeeds" but never actually deletes anything, so the
        # final guard sees record.path still present with no remnant.
        with (
            patch("lib_python_worktree.core.teardown._run_git", return_value=_ok()),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                MagicMock(return_value=[]),
            ),
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
                MagicMock(return_value=[]),
            ),
        ):
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )

        assert excinfo.value.staged is False
        assert str(excinfo.value) == (
            f"worktree '{record.id}' directory is still locked after killing"
            f" 0 blocking process(es)."
        )

    # -- R5 (AC4): a saturated wedge cap cannot block an unheld directory ----

    def test_saturated_wedge_cap_does_not_block_unheld_dir(self, tmp_path):
        """A `handle_scan:capped`/degraded systemwide scan result must never
        turn a clean, unheld, successfully-renamed directory's removal into
        WorktreeDirLockedError -- because on the happy path the scan is
        never even consulted (AC2/R1). RED today: this fixture reproduces
        the actual bug this ticket exists to fix -- the confirming rescan
        can never clear a previously observed foreign hit, so a persistent
        (always-degraded) scan result keeps the removal permanently
        blocked."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r5"
        checkout.mkdir()
        record = _make_record("wt-r5", path=str(checkout), repo_root=str(tmp_path))
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        degraded = _PartialList([], skipped_passes=("handle_scan:capped",))

        with (
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                MagicMock(return_value=degraded),
            ) as mock_scan,
            patch("lib_python_worktree.core.teardown._run_git", return_value=_ok()),
        ):
            # Must succeed -- and, per AC2/R1, must never even call the
            # (perpetually degraded) scan on a clean, unheld target.
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        assert mock_scan.call_count == 0, (
            f"a successful rename means the scan is never consulted at all; "
            f"got {mock_scan.call_count} call(s)"
        )
        assert not checkout.exists()

    # -- R6b: the residual trigger gets the same bounded retry first, then
    #    the same diagnosis ---------------------------------------------

    def test_transient_residual_is_resolved_by_bounded_retry_without_diagnosis(
        self, tmp_path
    ):
        """A residual left behind by the delete ladder's first attempt (a
        transient AV/indexer lock on one file) must be absorbed by the same
        bounded retry loop as a failed rename -- never escalated to
        diagnosis. RED today: there is no staged path at all (`shutil.rmtree`
        is never invoked against `<path>.removing`), so the staged-path
        retry-count assertion below is unreachable at 0."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r6b-transient"
        checkout.mkdir()
        staged_str = str(checkout) + ".removing"
        record = _make_record(
            "wt-r6b-transient", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        real_rmtree = shutil.rmtree
        staged_rmtree_calls = {"n": 0}

        def _rmtree_side_effect(path, *a, **kw):
            path_str = str(path)
            if path_str == staged_str or path_str.startswith("\\?\\" + staged_str):
                staged_rmtree_calls["n"] += 1
                if staged_rmtree_calls["n"] == 1:
                    # Simulate a transient locked file surviving the first
                    # delete attempt: remove everything else, but leave the
                    # staged directory (with one file) in place.
                    for child in Path(path_str.replace("\\?\\", "")).glob("*"):
                        if child.name != "locked.tmp":
                            if child.is_dir():
                                real_rmtree(str(child), *a, **kw)
                            else:
                                child.unlink()
                    return
                real_rmtree(path_str.replace("\\?\\", ""), *a, **kw)
                return
            real_rmtree(path_str, *a, **kw)

        with (
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=_rmtree_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
            ) as mock_find,
            patch(
                "lib_python_worktree.core.teardown._kill_blocking_processes",
            ) as mock_kill,
            patch("lib_python_worktree.core.teardown._run_git", return_value=_ok()),
        ):
            manager._teardown(
                record,
                force=False,
                kill_blocking_processes=False,
                _lifecycle_module=mock_lifecycle,
            )

        mock_find.assert_not_called()
        mock_kill.assert_not_called()
        assert staged_rmtree_calls["n"] >= 2, (
            f"expected the delete ladder to retry against the staged path "
            f"at least once after a transient residual; got "
            f"{staged_rmtree_calls['n']} call(s) against {staged_str!r}"
        )
        assert not checkout.exists()
        assert not Path(staged_str).exists()

    def test_persistent_residual_enters_diagnosis_only_after_the_retry_budget(
        self, tmp_path
    ):
        """A residual that never clears on its own must still go through the
        bounded retry loop FIRST -- only once that budget is exhausted does
        the residual trigger enter tier-1/tier-2 diagnosis. RED today: the
        staged path is never created at all, so `_find_blocking_processes`
        is never invoked against it via a residual trigger (it may still be
        invoked via today's unrelated Gate A/orphan-scan mechanics, but
        never for the reason this test pins)."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r6b-persistent"
        checkout.mkdir()
        staged_str = str(checkout) + ".removing"
        record = _make_record(
            "wt-r6b-persistent", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        real_rmtree = shutil.rmtree

        def _rmtree_side_effect(path, *a, **kw):
            path_str = str(path)
            if path_str == staged_str:
                # Never actually clears -- always leaves the directory (with
                # content) behind, so the residual persists no matter how
                # many times the ladder retries.
                return
            real_rmtree(path_str, *a, **kw)

        with (
            patch(
                "lib_python_worktree.core.teardown.shutil.rmtree",
                side_effect=_rmtree_side_effect,
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                MagicMock(return_value=[]),
            ) as mock_find,
            patch("lib_python_worktree.core.teardown._run_git", return_value=_ok()),
        ):
            with pytest.raises(WorktreeDirLockedError) as excinfo:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=False,
                    _lifecycle_module=mock_lifecycle,
                )

        assert excinfo.value.staged is True
        assert mock_find.call_count >= 1, (
            "a persistent residual must eventually escalate to tier-2 "
            "diagnosis (systemwide scan) once the retry budget is spent"
        )

    # -- R7: Windows read-only files no longer defeat the delete (P4) -------

    @pytest.mark.skipif(
        sys.platform != "win32", reason="win32-only: NTFS read-only semantics"
    )
    def test_readonly_file_is_removed(self, tmp_path):
        """`git worktree remove` used to absorb a PermissionError on a
        read-only file for us; the new merged delete ladder needs its own
        chmod-and-retry rung, exercised against the STAGED path. Real
        filesystem, real read-only file.

        NOTE: v0.3.12's existing `_phase_filesystem_fallback` already
        recovers this specific case today (its robocopy rung clears
        read-only attributes), so the plain end-to-end "did it get
        deleted" outcome is NOT RED by itself -- confirmed by first running
        this assertion alone and observing it already passes. The
        RED-worthy assertion is therefore the mechanism this ticket
        actually changes: deletion must go through the rename-staged path
        (`os.rename` to `<path>.removing`), not `git worktree remove` +
        the old fallback. RED today: `os.rename` is never called at all."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r7-readonly"
        checkout.mkdir()
        ro_file = checkout / "readonly.txt"
        ro_file.write_text("cannot touch this")
        os.chmod(str(ro_file), stat.S_IREAD)
        record = _make_record(
            "wt-r7-readonly", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        staged_str = str(checkout) + ".removing"
        real_rename = os.rename
        rename_to_staged_calls = []

        def _tracked_rename(src, dst, *a, **kw):
            if str(dst) == staged_str:
                rename_to_staged_calls.append((src, dst))
            return real_rename(src, dst, *a, **kw)

        try:
            with (
                patch(
                    "lib_python_worktree.core.teardown._run_git",
                    return_value=_ok(),
                ),
                patch(
                    "lib_python_worktree.core.teardown.os.rename",
                    side_effect=_tracked_rename,
                ),
            ):
                manager._teardown(
                    record, force=False, _lifecycle_module=mock_lifecycle
                )
            assert rename_to_staged_calls, (
                "deletion must go through the rename-staged path -- "
                "os.rename to '<path>.removing' was never called"
            )
            assert not checkout.exists()
        finally:
            if ro_file.exists():
                os.chmod(str(ro_file), stat.S_IWRITE)

    # -- R8 (remaining sub-cases not superseded by Decision 1's rejection
    #    of _phase_reclaim_staged): the merged stage+delete phase's own
    #    guard for "nothing to do here" ---------------------------------

    def test_absent_tree_and_no_remnant_attempts_no_rename_and_no_delete(
        self, tmp_path
    ):
        """record.path absent, no `.removing` remnant either -> zero
        os.rename calls, zero shutil.rmtree calls, zero subprocess.run
        calls (no robocopy) -- but the removal still completes (prune runs,
        ports released). RED today: an absent target already short-circuits
        most phases via `ctx.target_absent`, but the specific claim that
        `git worktree prune --expire=now` still runs unconditionally is new
        -- today's target-absent path never prunes at all."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r8-absent"
        # Deliberately never created -- record.path does not exist.
        record = _make_record(
            "wt-r8-absent", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        calls: list = []

        def _side_effect(args, cwd=None, **kwargs):
            calls.append(list(args))
            return _ok()

        with (
            patch(
                "lib_python_worktree.core.teardown._run_git",
                side_effect=_side_effect,
            ),
            patch("lib_python_worktree.core.teardown.os.rename") as mock_rename,
            patch("lib_python_worktree.core.teardown.shutil.rmtree") as mock_rmtree,
            patch("lib_python_worktree.core.teardown.subprocess.run") as mock_run,
        ):
            manager._teardown(record, force=False, _lifecycle_module=mock_lifecycle)

        mock_rename.assert_not_called()
        mock_rmtree.assert_not_called()
        mock_run.assert_not_called()
        assert any(c[:2] == ["worktree", "prune"] for c in calls), (
            f"prune must still run unconditionally even for an "
            f"already-absent target (so a stale git registration is "
            f"cleared); got calls={calls}"
        )

    # -- R3 end-to-end row: the whole remove() stays under the 10.0s
    #    ceiling (7.0s failure budget + 3.0s slack) even when a discovery
    #    target's cwd() hangs -------------------------------------------

    def test_154_rename_failure_diagnosis_returns_within_total_budget_when_cwd_hangs(
        self, tmp_path
    ):
        """A genuinely held/locked worktree's failure-path diagnosis must
        stay bounded end-to-end, even when the systemwide scan it
        eventually escalates to hits a process whose discovery call hangs.
        This is the *held-worktree* population -- explicitly outside AC1's
        "clean, unheld" population (item 7). RED today: `_find_blocking_
        processes` is called with no deadline at all from Gate A/the
        orphan-scan phase, so a hanging discovery call blocks the whole
        `_teardown()` call for as long as the hang lasts."""
        manager = _make_manager(tmp_path)
        checkout = tmp_path / "wt-r3-e2e"
        checkout.mkdir()
        record = _make_record(
            "wt-r3-e2e", path=str(checkout), repo_root=str(tmp_path)
        )
        manager.state.add(record)
        mock_lifecycle = MagicMock()

        def _slow_scan(*a, **kw):
            _real_time_sleep(6.0)
            return []

        with (
            patch(
                "lib_python_worktree.core.teardown.os.rename",
                side_effect=PermissionError("locked"),
            ),
            patch(
                "lib_python_worktree.core.teardown._find_blocking_processes",
                side_effect=_slow_scan,
            ),
            patch(
                "lib_python_worktree.core.teardown._run_git", return_value=_ok()
            ),
        ):
            t0 = time.monotonic()
            try:
                manager._teardown(
                    record,
                    force=False,
                    kill_blocking_processes=True,
                    _lifecycle_module=mock_lifecycle,
                )
            except Exception:
                pass  # only the wall-clock budget is pinned by this test
            elapsed = time.monotonic() - t0

        assert elapsed < 10.0, (
            f"a held/locked worktree's failure-path diagnosis must "
            f"complete within the 7.0s failure budget + 3.0s slack, not "
            f"hang on an unbounded systemwide scan; took {elapsed:.2f}s"
        )
