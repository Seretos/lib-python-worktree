"""Ticket #135 -- per-phase unit tests, phase-order structural tests, and
no-cycle/logger-name/exception-identity guards for core/teardown.py.

Covers the plan's R1, R2, R3, R4, R7 behavioural requirements:

- R1: teardown._TEARDOWN_PHASES is a tuple of eleven callables (ticket #140
  adds ``_phase_orphan_scan`` at index 6) in documented order; run_teardown
  executes exactly it, in order, aborting on the first raise.
- R2: each phase is unit-testable standalone over a hand-built
  _TeardownContext (no WorktreeManager needed).
- R3: the dirt memo's force-gating and two-phase reset semantics.
- R4: teardown._target_is_absent(record) is the single seam covering both
  remove()'s and _teardown()'s absence decisions; the long-path-fallback
  and final-guard os.path.exists checks are NOT routed through the seam.
- R7: no-cycle guard, logger name, GitCommandError identity, manager.__all__
  unchanged.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import lib_python_worktree
from lib_python_worktree.core import manager as manager_module
from lib_python_worktree.core import teardown
from lib_python_worktree.core import _exceptions
from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord


def _make_record(wt_id: str = "wt-phase", **kwargs) -> WorktreeRecord:
    defaults = dict(
        id=wt_id,
        repo_root="/fake/repo",
        branch="feature/x",
        path="/fake/store/" + wt_id,
    )
    defaults.update(kwargs)
    return WorktreeRecord(**defaults)


def _make_ctx(record=None, **overrides) -> teardown._TeardownContext:
    """Hand-build a _TeardownContext with sane defaults -- no
    WorktreeManager involved anywhere (R2)."""
    record = record or _make_record()
    fields = dict(
        record=record,
        force=False,
        kill_blocking_processes=False,
        lifecycle=MagicMock(),
        store=InMemoryStateStore(),
        allocator=MagicMock(),
        target_absent=True,
        owned_pids=set(),
        contract=None,
        contract_found=False,
        contract_isolation=None,
        contract_load_error=None,
    )
    fields.update(overrides)
    return teardown._TeardownContext(**fields)


# ---------------------------------------------------------------------------
# R1: phase tuple shape + run_teardown ordering/abort semantics
# ---------------------------------------------------------------------------

def test_teardown_phases_is_tuple_of_ten_callables():
    """#154 (human override, Decision 1): the rename/rmtree/prune rework
    deletes four phases (_phase_gate_a_blocking_preflight, _phase_orphan_scan,
    _phase_git_worktree_remove, _phase_filesystem_fallback) and adds three
    (_phase_dirt_gate, _phase_stage_and_delete, _phase_git_prune) --
    NOT four, because the override explicitly rejects _phase_reclaim_staged
    (generation-2 plan.md item 2a) in favour of the lighter fix (no new
    phase; _target_is_absent() gains one added rule instead). Net: 11 -> 10.
    """
    assert isinstance(teardown._TEARDOWN_PHASES, tuple)
    assert len(teardown._TEARDOWN_PHASES) == 10
    assert all(callable(p) for p in teardown._TEARDOWN_PHASES)


def test_teardown_phases_documented_order():
    """#154 (human override, Decision 1): the amended order. Gate A and the
    orphan scan are gone (replaced by the rename-is-the-oracle mechanism);
    `git worktree remove` + the filesystem fallback are replaced by the
    unconditional dirt gate, the merged stage+delete phase and the
    unconditional prune. `_phase_reclaim_staged` is deliberately ABSENT --
    see the override note, Decision 1.
    """
    names = [f.__name__ for f in teardown._TEARDOWN_PHASES]
    assert names == [
        "_phase_guard_primary",
        "_phase_stop_processes",
        "_phase_stop_hook",
        "_phase_gate_b_early_dirty",
        "_phase_run_teardown_steps",
        "_phase_dirt_gate",
        "_phase_stage_and_delete",
        "_phase_git_prune",
        "_phase_final_guard",
        "_phase_release_ports",
    ]
    assert "_phase_reclaim_staged" not in names, (
        "Decision 1 rejects the reclaim phase entirely -- do not reintroduce it"
    )


def test_run_teardown_executes_phases_in_order():
    calls = []
    fake_phases = tuple(
        (lambda name: (lambda ctx: calls.append(name)))(f.__name__)
        for f in teardown._TEARDOWN_PHASES
    )
    ctx = _make_ctx()
    with patch.object(teardown, "_TEARDOWN_PHASES", fake_phases):
        teardown.run_teardown(ctx)
    assert calls == [f.__name__ for f in teardown._TEARDOWN_PHASES]


def test_run_teardown_aborts_on_first_raise():
    calls = []

    def _phase1(ctx):
        calls.append("p1")

    def _phase2(ctx):
        calls.append("p2")
        raise RuntimeError("boom")

    def _phase3(ctx):
        calls.append("p3")

    ctx = _make_ctx()
    with patch.object(teardown, "_TEARDOWN_PHASES", (_phase1, _phase2, _phase3)):
        with pytest.raises(RuntimeError, match="boom"):
            teardown.run_teardown(ctx)
    assert calls == ["p1", "p2"], "phase3 must never run after phase2 raises"


# ---------------------------------------------------------------------------
# R2: each phase, standalone, over a hand-built context
# ---------------------------------------------------------------------------

def test_phase_guard_primary_raises_for_primary_backing():
    from lib_python_worktree.core._exceptions import PrimaryCheckoutError

    record = _make_record(path="/fake/repo", repo_root="/fake/repo", backing="primary")
    ctx = _make_ctx(record)
    with pytest.raises(PrimaryCheckoutError):
        teardown._phase_guard_primary(ctx)


def test_phase_guard_primary_noop_for_worktree_backing():
    ctx = _make_ctx()
    teardown._phase_guard_primary(ctx)  # must not raise


def test_phase_stop_processes_stops_every_tracked_role():
    record = _make_record(pids={"main": 111, "worker": 222})
    lifecycle = MagicMock()
    ctx = _make_ctx(record, lifecycle=lifecycle)
    teardown._phase_stop_processes(ctx)
    assert lifecycle.stop.call_count == 2


def test_phase_stop_processes_noop_when_no_pids():
    lifecycle = MagicMock()
    ctx = _make_ctx(lifecycle=lifecycle)
    teardown._phase_stop_processes(ctx)
    lifecycle.stop.assert_not_called()


def test_phase_stop_hook_sets_skipped_outcome_with_no_contract():
    ctx = _make_ctx()
    teardown._phase_stop_hook(ctx)
    assert ctx.record.stop_hook_outcome is not None
    assert ctx.record.stop_hook_outcome.status == "skipped"


def test_phase_stop_hook_sets_failed_outcome_on_load_error():
    ctx = _make_ctx(contract_load_error="disk error")
    teardown._phase_stop_hook(ctx)
    assert ctx.record.stop_hook_outcome.status == "failed"
    assert ctx.record.stop_hook_outcome.message == "disk error"


def test_phase_gate_b_raises_dirty_when_contract_teardown_and_real_dirt():
    from lib_python_worktree.contract.schema import Step, WorktreeContract
    from lib_python_worktree.core._exceptions import DirtyWorktreeError

    contract = WorktreeContract(
        version=1, isolation="full", teardown=[Step(run="echo bye", name="bye")]
    )
    ctx = _make_ctx(force=False, contract=contract)
    with patch.object(ctx, "dirt", return_value=["real_dirt.txt"]):
        with pytest.raises(DirtyWorktreeError):
            teardown._phase_gate_b_early_dirty(ctx)


def test_phase_gate_b_noop_when_forced():
    from lib_python_worktree.contract.schema import Step, WorktreeContract

    contract = WorktreeContract(
        version=1, isolation="full", teardown=[Step(run="echo bye", name="bye")]
    )
    ctx = _make_ctx(force=True, contract=contract)
    with patch.object(ctx, "dirt", return_value=["real_dirt.txt"]):
        teardown._phase_gate_b_early_dirty(ctx)  # must not raise -- force bypasses


def test_phase_run_teardown_steps_skips_when_already_ran():
    from lib_python_worktree.contract.schema import Step, WorktreeContract

    contract = WorktreeContract(
        version=1, isolation="full", teardown=[Step(run="echo bye", name="bye")]
    )
    record = _make_record(teardown_ran=True)
    ctx = _make_ctx(record, contract=contract)
    mock_runner = MagicMock()
    with patch(
        "lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner
    ):
        teardown._phase_run_teardown_steps(ctx)
    mock_runner.run.assert_not_called()


def test_phase_run_teardown_steps_runs_and_sets_marker():
    from lib_python_worktree.contract.schema import Step, WorktreeContract

    contract = WorktreeContract(
        version=1, isolation="full", teardown=[Step(run="echo bye", name="bye")]
    )
    record = _make_record()
    store = InMemoryStateStore()
    store.add(record)
    ctx = _make_ctx(record, contract=contract, store=store)
    mock_runner = MagicMock()
    with patch(
        "lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner
    ):
        teardown._phase_run_teardown_steps(ctx)
    mock_runner.run.assert_called_once()
    assert record.teardown_ran is True


def test_phase_git_prune_calls_run_git_once(tmp_path):
    """#154: `_phase_git_worktree_remove` is replaced by the unconditional
    `_phase_git_prune`."""
    record = _make_record(path=str(tmp_path / "wt-phase"))
    ctx = _make_ctx(record)
    with patch("lib_python_worktree.core.teardown._run_git") as mock_git:
        mock_git.return_value = MagicMock(returncode=0, stderr="")
        teardown._phase_git_prune(ctx)
    mock_git.assert_called_once_with(
        ["worktree", "prune", "--expire=now"], cwd=Path(record.repo_root)
    )


def test_phase_git_prune_warns_and_continues_on_failure(tmp_path, caplog):
    """#154: a prune failure must never raise -- it warn-and-continues."""
    record = _make_record(path=str(tmp_path / "wt-phase"))
    ctx = _make_ctx(record)
    with patch("lib_python_worktree.core.teardown._run_git") as mock_git:
        mock_git.return_value = MagicMock(returncode=1, stderr="fatal: something else")
        teardown._phase_git_prune(ctx)  # must not raise


def test_phase_final_guard_raises_when_still_present():
    from lib_python_worktree.core._exceptions import WorktreeDirLockedError

    ctx = _make_ctx()
    with patch(
        "lib_python_worktree.core.teardown.os.path.exists", return_value=True
    ):
        with pytest.raises(WorktreeDirLockedError):
            teardown._phase_final_guard(ctx)


def test_phase_final_guard_noop_when_absent():
    ctx = _make_ctx()
    with patch(
        "lib_python_worktree.core.teardown.os.path.exists", return_value=False
    ):
        teardown._phase_final_guard(ctx)  # must not raise


def test_phase_release_ports_calls_allocator_release():
    allocator = MagicMock()
    ctx = _make_ctx(allocator=allocator)
    teardown._phase_release_ports(ctx)
    allocator.release.assert_called_once_with(ctx.record.id)


# ---------------------------------------------------------------------------
# R3: dirt memo force-gating + two-phase reset semantics
# ---------------------------------------------------------------------------

def test_dirt_memo_force_true_never_calls_status_entries():
    ctx = _make_ctx(force=True)
    with patch.object(
        teardown, "_status_entries", side_effect=AssertionError("must not be called")
    ):
        assert ctx.dirt() == []


def test_dirt_memo_two_probes_across_a_run_with_teardown_steps():
    """Exactly 2 _status_entries calls across a run with teardown: steps
    present: one pre-teardown (Gate B), one post-teardown (re-fetched after
    invalidate_dirt()) -- matching today's semantics."""
    from lib_python_worktree.contract.schema import Step, WorktreeContract

    contract = WorktreeContract(
        version=1, isolation="full", teardown=[Step(run="echo bye", name="bye")]
    )
    record = _make_record()
    ctx = _make_ctx(record, contract=contract, force=False)

    call_count = [0]

    def _fake_status_entries(rec):
        call_count[0] += 1
        # A CONCLUSIVE (non-None) result: _real_dirt_paths only re-probes
        # internally when handed entries=None, so an inconclusive fake here
        # would inflate the count via that (separate, pre-existing) code
        # path rather than exercising the memoisation this test targets.
        return ["?? .seretos"]  # benign-only entry -> no real dirt

    mock_runner = MagicMock()
    with (
        patch.object(teardown, "_status_entries", side_effect=_fake_status_entries),
        patch(
            "lib_python_worktree.setup.runner.SetupRunner", return_value=mock_runner
        ),
    ):
        teardown._phase_gate_b_early_dirty(ctx)
        teardown._phase_run_teardown_steps(ctx)
        # A later phase-6+ consumer re-probing after invalidate_dirt():
        ctx.dirt()

    assert call_count[0] == 2, f"expected exactly 2 probes, got {call_count[0]}"


def test_invalidate_dirt_forces_reprobe():
    ctx = _make_ctx(force=False)
    # A CONCLUSIVE (non-None) fake result -- see the comment in
    # test_dirt_memo_two_probes_across_a_run_with_teardown_steps above for
    # why an inconclusive (None) one would inflate the call count via
    # _real_dirt_paths's own internal re-probe-on-None behaviour, unrelated
    # to the memoisation this test targets.
    with patch.object(
        teardown, "_status_entries", return_value=["?? .seretos"]
    ) as mock_probe:
        ctx.dirt()
        ctx.dirt()  # memoised -- should not re-probe
        assert mock_probe.call_count == 1
        ctx.invalidate_dirt()
        ctx.dirt()
        assert mock_probe.call_count == 2


# ---------------------------------------------------------------------------
# R4: _target_is_absent seam
# ---------------------------------------------------------------------------

def test_target_is_absent_true_for_nonexistent_path():
    record = _make_record(path="/definitely/not/a/real/path/xyz")
    assert teardown._target_is_absent(record) is True


def test_target_is_absent_seam_covers_both_remove_and_build_context():
    """Patching the seam to True must be observed both by remove()'s own
    probe AND by build_context()'s target_absent field (ticket #135, A3's
    module-attribute-access design)."""
    from lib_python_worktree.core.manager import WorktreeManager, ManagerConfig

    record = _make_record()
    manager = WorktreeManager(
        config=ManagerConfig(store_root=Path("/fake/store")),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )
    manager.state.add(record)

    with patch(
        "lib_python_worktree.core.teardown._target_is_absent", return_value=True
    ):
        # remove()'s own probe:
        remove_side_target_absent = manager_module._teardown_mod._target_is_absent(
            record
        )
        # build_context()'s field:
        ctx = teardown.build_context(
            record,
            force=False,
            store=InMemoryStateStore(),
            allocator=MagicMock(),
            lifecycle_module=MagicMock(),
        )

    assert remove_side_target_absent is True
    assert ctx.target_absent is True


def test_long_path_fallback_and_final_guard_not_routed_through_seam():
    """The long-path-fallback and Final-guard os.path.exists checks must
    see the REAL filesystem even when the _target_is_absent seam is
    patched -- they are deliberately not routed through it (R4)."""
    ctx = _make_ctx()
    with (
        patch(
            "lib_python_worktree.core.teardown._target_is_absent", return_value=True
        ),
        patch(
            "lib_python_worktree.core.teardown.os.path.exists", return_value=False
        ) as mock_exists,
    ):
        teardown._phase_final_guard(ctx)
    mock_exists.assert_called_once_with(ctx.record.path)


# ---------------------------------------------------------------------------
# R7: no-cycle guard, logger name, exception identity, __all__ unchanged
# ---------------------------------------------------------------------------

def test_teardown_module_never_imports_manager():
    """AST-checked: core/teardown.py must never import core/manager.py."""
    source_path = Path(inspect.getfile(teardown))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "manager" not in alias.name.split("."), (
                    f"teardown.py must not import manager: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.split(".")[-1] == "manager", (
                f"teardown.py must not import from manager: {module}"
            )
            for alias in node.names:
                assert alias.name != "manager", (
                    f"teardown.py must not import 'manager' by name"
                )


def test_teardown_logger_name_is_manager_module_name():
    assert teardown._logger.name == "lib_python_worktree.core.manager"


def test_git_command_error_identity_after_relocation():
    assert manager_module.GitCommandError is lib_python_worktree.GitCommandError
    assert manager_module.GitCommandError is _exceptions.GitCommandError


def test_manager_all_unchanged():
    assert "GitCommandError" in manager_module.__all__
    assert "WorktreeManager" in manager_module.__all__


# ---------------------------------------------------------------------------
# #154 Decision 1 (human override): _target_is_absent() gains one added rule
# for a non-empty `.removing` remnant -- the lighter fix replacing the
# rejected _phase_reclaim_staged. See end-to-end coverage in
# test_teardown_matrix.py::TestMatrixRemovalMechanism::
# test_staged_dirty_remnant_is_never_silently_destroyed_by_a_later_removal.
# ---------------------------------------------------------------------------

def test_target_is_absent_false_for_nonempty_remnant_under_no_force(tmp_path):
    """record.path absent, a non-empty `<path>.removing` remnant present,
    force=False -> NOT 'target absent' (must surface downstream rather than
    silently fast-pathing to 'nothing to remove here'). RED today:
    _target_is_absent() only ever checks `record.path` itself and has no
    `force` parameter at all."""
    checkout = tmp_path / "wt-target-absent-remnant"
    staged = Path(str(checkout) + ".removing")
    staged.mkdir()
    (staged / "dirty.txt").write_text("uncommitted")
    record = _make_record(path=str(checkout))

    assert teardown._target_is_absent(record, force=False) is False


def test_target_is_absent_true_for_nonempty_remnant_under_force(tmp_path):
    """The same remnant under force=True is still 'target absent' for
    gating purposes -- force=True's pre-clean of a stale remnant is
    unchanged and happens in _phase_stage_and_delete's opening guard, not
    here."""
    checkout = tmp_path / "wt-target-absent-remnant-forced"
    staged = Path(str(checkout) + ".removing")
    staged.mkdir()
    (staged / "dirty.txt").write_text("uncommitted")
    record = _make_record(path=str(checkout))

    assert teardown._target_is_absent(record, force=True) is True


def test_target_is_absent_true_for_truly_absent_target_default_force_false():
    """Backward-compat: no remnant at all -> unchanged True, with the new
    `force` parameter defaulting so existing single-arg call sites (e.g.
    manager.remove()'s own probe) keep working."""
    record = _make_record(path="/definitely/not/a/real/path/xyz-154")
    assert teardown._target_is_absent(record) is True


# ---------------------------------------------------------------------------
# #154 R3/R17: _diagnose_and_retry behavioural coverage (review fix round,
# reviewer finding 2). The structural existence check
# (test_tier1_diagnosis_scaffolding_exists) is superseded by these -- the
# scaffolding it pinned is now exercised directly.
# ---------------------------------------------------------------------------

def test_diagnose_and_retry_tier1_kills_confirmed_live_owned_pid_and_retries():
    """Tier 1: a pid in ctx.owned_pids that the bounded psutil.pid_exists
    probe confirms alive is killed via ctx.lifecycle._kill_process_tree
    and *retry* is attempted exactly once; a successful retry resolves the
    failure without tier 2 (the systemwide scan) ever running."""
    record = _make_record(pids={"main": 555})
    lifecycle = MagicMock()
    lifecycle._bounded_call.side_effect = lambda fn, deadline=None: (True, fn())
    ctx = _make_ctx(
        record, lifecycle=lifecycle, kill_blocking_processes=True, owned_pids={555}
    )

    retry_calls = []

    def _retry():
        retry_calls.append(1)

    with (
        patch("psutil.pid_exists", side_effect=lambda pid: pid == 555),
        patch.object(teardown, "_find_blocking_processes") as mock_tier2,
    ):
        resolved = teardown._diagnose_and_retry(ctx, trigger="rename", retry=_retry)

    assert resolved is True
    assert retry_calls == [1]
    lifecycle._kill_process_tree.assert_called_once()
    killed_infos = lifecycle._kill_process_tree.call_args[0][0]
    assert {i.pid for i in killed_infos} == {555}
    assert ctx.kill_attempted is True
    assert {b.pid for b in ctx.blockers} == {555}
    mock_tier2.assert_not_called()


def test_diagnose_and_retry_tier1_skips_pid_whose_liveness_probe_did_not_complete():
    """An owned pid whose bounded liveness probe never completes must
    never be treated as a confirmed blocker -- not killed, not reported,
    not retried on. Falls through to tier 2 instead of trusting an
    unconfirmed pid."""
    record = _make_record(pids={"main": 777})
    lifecycle = MagicMock()
    lifecycle._bounded_call.return_value = (False, None)
    ctx = _make_ctx(
        record, lifecycle=lifecycle, kill_blocking_processes=True, owned_pids={777}
    )

    with patch.object(teardown, "_find_blocking_processes", return_value=[]) as mock_tier2:
        resolved = teardown._diagnose_and_retry(ctx, trigger="rename", retry=lambda: None)

    assert resolved is False
    lifecycle._kill_process_tree.assert_not_called()
    assert ctx.blockers == []
    assert ctx.kill_attempted is False
    mock_tier2.assert_called_once()


def test_diagnose_and_retry_tier1_report_only_when_kill_flag_false():
    """kill_blocking_processes=False: a confirmed-live owned pid is still
    reported in ctx.blockers (report-only diagnosis), but never killed and
    never retried on via tier 1 -- falls through to tier 2 for the same
    report-only treatment."""
    record = _make_record(pids={"main": 888})
    lifecycle = MagicMock()
    lifecycle._bounded_call.side_effect = lambda fn, deadline=None: (True, fn())
    ctx = _make_ctx(
        record, lifecycle=lifecycle, kill_blocking_processes=False, owned_pids={888}
    )

    with (
        patch("psutil.pid_exists", return_value=True),
        patch.object(teardown, "_find_blocking_processes", return_value=[]) as mock_tier2,
    ):
        resolved = teardown._diagnose_and_retry(ctx, trigger="rename", retry=lambda: None)

    assert resolved is False
    lifecycle._kill_process_tree.assert_not_called()
    assert ctx.kill_attempted is False
    assert {b.pid for b in ctx.blockers} == {888}
    mock_tier2.assert_called_once()


def test_diagnose_and_retry_tier2_never_kills_when_nothing_found():
    """Regression (found and fixed during phase=implement, review fix
    round finding 1 confirmed this fix as correct): _kill_blocking_processes
    must never be called when tier 2's discovery finds no candidate --
    calling it anyway would pay its own internal discovery scan for no
    benefit and could blow the shared per-removal failure budget."""
    record = _make_record()
    lifecycle = MagicMock()
    ctx = _make_ctx(record, lifecycle=lifecycle, kill_blocking_processes=True, owned_pids=set())

    with (
        patch.object(teardown, "_find_blocking_processes", return_value=[]),
        patch.object(teardown, "_kill_blocking_processes") as mock_kill,
    ):
        resolved = teardown._diagnose_and_retry(ctx, trigger="residual", retry=lambda: None)

    assert resolved is False
    mock_kill.assert_not_called()
    assert ctx.kill_attempted is False


def test_diagnose_and_retry_tier2_kills_and_retries_when_found():
    """Tier 2: a systemwide-scan hit is killed via
    teardown._kill_blocking_processes and *retry* is attempted exactly
    once; a successful retry resolves the failure."""
    from lib_python_worktree.core.process_lifecycle import KilledProcessInfo

    record = _make_record()
    lifecycle = MagicMock()
    ctx = _make_ctx(record, lifecycle=lifecycle, kill_blocking_processes=True, owned_pids=set())

    tier2_hit = KilledProcessInfo(pid=999, name="node", cmdline=["node"], source="orphan_scan")
    retry_calls = []

    def _retry():
        retry_calls.append(1)

    with (
        patch.object(teardown, "_find_blocking_processes", return_value=[tier2_hit]),
        patch.object(teardown, "_kill_blocking_processes", return_value=[tier2_hit]) as mock_kill,
    ):
        resolved = teardown._diagnose_and_retry(ctx, trigger="residual", retry=_retry)

    assert resolved is True
    mock_kill.assert_called_once()
    assert retry_calls == [1]
    assert ctx.kill_attempted is True
    assert 999 in {b.pid for b in ctx.blockers}
