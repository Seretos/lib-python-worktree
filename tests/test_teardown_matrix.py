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
