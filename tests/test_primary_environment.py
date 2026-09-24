"""Tests for primary-checkout environment management (ticket #84).

Covers:
- R6: remove()/_teardown() structurally refuse to delete a primary checkout,
  even with force=True.
- R7/R7b: start()/stop() addressed by checkout_path materialise/resolve a
  primary record; the target-resolution matrix.
- R8: a primary record's branch is always read live, never stored stale.
- R9: port reservations are incremental and survive stop -> restart.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import pytest

from lib_python_worktree.core._exceptions import (
    CheckoutTargetError,
    InvalidRepoError,
    PrimaryCheckoutError,
    UnknownVariantError,
    WorktreeError,
)
from lib_python_worktree.core.checkout import classify_checkout, primary_id_for
from lib_python_worktree.core.manager import (
    ManagerConfig,
    WorktreeManager,
    WorktreeNotFoundError,
)
from lib_python_worktree.core.process_lifecycle import ProcessAlreadyRunningError
from lib_python_worktree.core.state import WorktreeRecord
from lib_python_worktree.core.yaml_store import YamlStateStore, _pid_alive


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _write_contract(repo_root: Path, text: str) -> None:
    p = repo_root / ".seretos" / "worktree-setup.yml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _run_line_for_code(code: str) -> str:
    """Cross-platform ``run:`` line executing *code* via the current
    interpreter, mirroring the quoting pattern already established in
    ``tests/test_setup_runner.py``."""
    if sys.platform == "win32":
        return f"& '{sys.executable}' -c '{code}'"
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def _run_line_for_script(script_path: Path) -> str:
    """Cross-platform ``run:`` line executing the python file at
    *script_path* via the current interpreter."""
    if sys.platform == "win32":
        return f"& '{sys.executable}' '{script_path}'"
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))}"


def _step_yaml(run_line: str, *, name: "str | None" = None) -> str:
    """Render one ``start:``/``stop:`` list item as YAML text.

    Uses a literal block scalar (``|-``) for ``run:`` rather than a
    double-quoted scalar: a Windows ``run_line`` embeds raw backslashes
    (from ``sys.executable``'s path), and YAML double-quoted scalars treat
    backslash as an escape-sequence introducer (``\\A``, \\U``, ...) --
    exactly the kind of sequence a Windows path collides with. A literal
    block scalar has no escape processing at all, so the path/quoting in
    *run_line* survives byte-for-byte.
    """
    lines = ["  - run: |-", f"      {run_line}"]
    if name:
        lines.append(f"    name: {name}")
    return "\n".join(lines) + "\n"


def _wait_for_file_content(
    path: Path,
    expected: str,
    *,
    record: "WorktreeRecord | None" = None,
    timeout: float = 30.0,
    interval: float = 0.2,
) -> None:
    """Poll *path* until its stripped text content equals *expected*.

    Used to observe the side effect of a *detached* ``start:`` step (ticket
    #84's live-branch-export tests): ``mgr.start()`` returns as soon as the
    child process is spawned, well before a cold-started Python interpreter
    on a loaded CI runner has necessarily gotten around to writing its
    output file. A fixed short poll budget flakes under exactly that kind of
    scheduling pressure (GH Actions run 31888867221, windows-latest only --
    5s budget, 508 other tests green, identical commit passed on
    pull_request-event the same day), so this uses a much longer budget
    (30s) suited to a cold Windows CI runner while still failing fast on a
    genuine defect via a short poll interval.

    On timeout this raises a clear ``AssertionError`` naming what was
    expected vs. what is actually on disk (or that the file never appeared)
    instead of letting a bare ``read_text()`` raise ``FileNotFoundError`` --
    that bare exception is uninformative about *why* the step never wrote
    the file. When *record* is given and carries a ``start_log_paths``
    entry, the step's own log is read and appended to the failure message,
    since that log is what would actually explain a genuine (non-timing)
    failure. Ticket #119: ``start_log_paths`` is now role-keyed, so this
    prefers the log for whichever role *record* actually has a tracked pid
    for, falling back to any single entry in the map (e.g. a role that has
    already been stopped, whose pid is gone but whose log entry is
    deliberately retained).
    """
    import time as _time

    deadline = _time.monotonic() + timeout
    last_seen: "str | None" = None
    while True:
        if path.exists():
            last_seen = path.read_text(encoding="utf-8").strip()
            if last_seen == expected:
                return
        if _time.monotonic() >= deadline:
            break
        _time.sleep(interval)

    if not path.exists():
        detail = f"file {path} was never created"
    else:
        detail = f"file {path} contained {last_seen!r}"

    log_excerpt = ""
    log_path = None
    if record is not None:
        paths = getattr(record, "start_log_paths", None) or {}
        # Prefer the log for a role record still has a tracked pid for...
        for role in getattr(record, "pids", None) or {}:
            if role in paths:
                log_path = paths[role]
                break
        # ...falling back to any single entry (e.g. a role already stopped,
        # whose start_log_paths entry ticket #119 deliberately retains).
        if log_path is None and paths:
            log_path = next(iter(paths.values()))
    if log_path:
        log_file = Path(log_path)
        if log_file.exists():
            log_excerpt = (
                f"\n--- start step log ({log_path}) ---\n"
                + log_file.read_text(encoding="utf-8", errors="replace")
            )
        else:
            log_excerpt = f"\n--- start step log ({log_path}) does not exist ---"
    elif record is not None:
        log_excerpt = "\n--- record has no start_log_paths ---"

    raise AssertionError(
        f"timed out after {timeout}s waiting for {path} to contain "
        f"{expected!r}; {detail}{log_excerpt}"
    )


# ---------------------------------------------------------------------------
# R6 -- remove()/_teardown() refuse a primary record, even with force=True
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_remove_refuses_primary_record_even_with_force(manager, git_repo: Path):
    """remove(force=True) on a primary record must refuse before any FS op.

    Uses the isolated, function-scoped ``git_repo`` fixture (a fresh tmp_path
    checkout per test) so a RED run against the unfixed code -- which would
    proceed into _teardown() and eventually shutil.rmtree the repo -- cannot
    damage anything shared.
    """
    rec = WorktreeRecord(
        id=primary_id_for(git_repo),
        repo_root=git_repo.resolve().as_posix(),
        branch=None,
        path=git_repo.resolve().as_posix(),
        backing="primary",
    )
    manager.state.add(rec)

    with pytest.raises(PrimaryCheckoutError):
        manager.remove(rec.id, force=True)

    assert (git_repo / "README.md").exists()
    assert manager.state.get(rec.id) is not None


@pytest.mark.requires_git
def test_remove_refuses_primary_record_force_false(manager, git_repo: Path):
    """Same refusal with force=False -- identical behaviour either way."""
    rec = WorktreeRecord(
        id=primary_id_for(git_repo),
        repo_root=git_repo.resolve().as_posix(),
        branch=None,
        path=git_repo.resolve().as_posix(),
        backing="primary",
    )
    manager.state.add(rec)

    with pytest.raises(PrimaryCheckoutError):
        manager.remove(rec.id, force=False)

    assert (git_repo / "README.md").exists()


@pytest.mark.requires_git
def test_remove_refuses_mislabelled_record_whose_path_is_repo_root(
    manager, git_repo: Path
):
    """Guard 2 in isolation: even a record with backing="worktree" (i.e.
    mislabelled) is refused if its path equals repo_root."""
    rec = WorktreeRecord(
        id="mislabelled-primary",
        repo_root=git_repo.resolve().as_posix(),
        branch="main",
        path=git_repo.resolve().as_posix(),
        backing="worktree",
    )
    manager.state.add(rec)

    with pytest.raises(PrimaryCheckoutError):
        manager.remove(rec.id, force=True)

    assert (git_repo / "README.md").exists()


@pytest.mark.requires_git
def test_teardown_called_directly_refuses_primary(manager, git_repo: Path):
    """_teardown() itself refuses a primary record when called directly."""
    rec = WorktreeRecord(
        id=primary_id_for(git_repo),
        repo_root=git_repo.resolve().as_posix(),
        branch=None,
        path=git_repo.resolve().as_posix(),
        backing="primary",
    )
    with pytest.raises(PrimaryCheckoutError):
        manager._teardown(rec, force=True)

    assert (git_repo / "README.md").exists()


# ---------------------------------------------------------------------------
# R7 -- start()/stop() manage a primary checkout addressed by checkout_path;
# the record is created by the first start() only
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_first_start_materialises_primary_record_and_tracks_pid(
    yaml_manager, git_repo: Path
):
    mgr = yaml_manager()
    sleep_line = _run_line_for_code("import time; time.sleep(20)")
    _write_contract(
        git_repo,
        "version: 1\nisolation: full\nstart:\n"
        + _step_yaml(sleep_line)
        + _step_yaml(sleep_line, name="alt"),
    )

    # Read-only classification/listing must never write a record.
    classify_checkout(git_repo)
    mgr.list_repo(str(git_repo))
    assert mgr.state.list() == []

    try:
        record = mgr.start(checkout_path=str(git_repo))

        assert record.backing == "primary"
        assert record.id == primary_id_for(git_repo)
        assert record.path == git_repo.resolve().as_posix()
        assert record.branch is None
        assert "main" in record.pids
        assert _pid_alive(record.pids["main"])
        assert record.start_log_paths.get("main")

        all_records = mgr.state.list()
        assert len(all_records) == 1
        assert all_records[0].id == record.id

        # A second start() on the same role while still running refuses.
        with pytest.raises(ProcessAlreadyRunningError):
            mgr.start(checkout_path=str(git_repo))

        stopped = mgr.stop(checkout_path=str(git_repo))
        assert stopped.status == "stopped"
        assert "main" not in stopped.pids
    finally:
        rec = mgr.state.get(primary_id_for(git_repo))
        if rec is not None and rec.pids:
            try:
                mgr.stop(checkout_path=str(git_repo))
            except Exception:  # noqa: BLE001
                pass


@pytest.mark.requires_git
def test_start_variant_selects_named_step(yaml_manager, git_repo: Path):
    mgr = yaml_manager()
    sleep_line = _run_line_for_code("import time; time.sleep(20)")
    _write_contract(
        git_repo,
        "version: 1\nisolation: full\nstart:\n"
        + _step_yaml(sleep_line)
        + _step_yaml(sleep_line, name="alt"),
    )
    try:
        record = mgr.start(checkout_path=str(git_repo), variant="alt")
        assert "main" in record.pids
    finally:
        try:
            mgr.stop(checkout_path=str(git_repo))
        except Exception:  # noqa: BLE001
            pass


@pytest.mark.requires_git
def test_start_unknown_variant_raises(yaml_manager, git_repo: Path):
    mgr = yaml_manager()
    _write_contract(
        git_repo,
        "version: 1\nisolation: full\nstart:\n"
        + _step_yaml(_run_line_for_code("pass"), name="named-only"),
    )
    with pytest.raises(UnknownVariantError) as exc_info:
        mgr.start(checkout_path=str(git_repo), variant="nope")
    # Ticket #131: the lone named step's own name is listed, plus the
    # implicit "default" fallback (reachable here via the ticket #112
    # single-step-total tier), even though this specific failed call
    # requested "nope".
    assert exc_info.value.available == ["named-only", "default"]
    # No record was written by the failed attempt (variant resolution
    # happens after materialisation lookup but the failure must not corrupt
    # state -- the record, if created, simply has no process running).
    rec = mgr.state.get(primary_id_for(git_repo))
    if rec is not None:
        assert not rec.pids


@pytest.mark.requires_git
def test_start_no_start_step_is_noop_ready_and_still_materialises(
    yaml_manager, git_repo: Path
):
    """A repo with no start: step yields the no-op "ready" path and still
    materialises exactly one primary record."""
    mgr = yaml_manager()
    record = mgr.start(checkout_path=str(git_repo))
    assert record.status == "ready"
    assert record.backing == "primary"
    all_records = mgr.state.list()
    assert len(all_records) == 1
    assert all_records[0].id == record.id


@pytest.mark.requires_git
def test_start_primary_checkout_never_flags_shadowed_contract(
    yaml_manager, git_repo: Path
):
    """Ticket #100: a primary checkout has no separate checkout-local
    contract copy at all -- its own contract IS the repo-root one -- so
    `_detect_shadowed_contract` must always return None for it, regardless
    of what the contract says."""
    _write_contract(git_repo, "version: 1\nisolation: none\n")
    mgr = yaml_manager()

    record = mgr.start(checkout_path=str(git_repo))

    assert record.backing == "primary"
    assert record.shadowed_contract is None


@pytest.mark.requires_git
def test_stop_never_started_role_is_graceful_noop(yaml_manager, git_repo: Path):
    mgr = yaml_manager()
    record = mgr.start(checkout_path=str(git_repo))  # no-op ready start
    stopped = mgr.stop(checkout_path=str(git_repo))
    assert stopped.id == record.id
    assert stopped.status == "stopped"


# ---------------------------------------------------------------------------
# R7b -- target-resolution matrix for start()/stop()
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_target_resolution_neither_given_raises_missing(yaml_manager, git_repo: Path):
    mgr = yaml_manager()
    with pytest.raises(CheckoutTargetError) as exc_info:
        mgr.start()
    assert exc_info.value.reason == "missing"
    assert isinstance(exc_info.value, WorktreeError)
    assert isinstance(exc_info.value, ValueError)

    with pytest.raises(CheckoutTargetError) as exc_info2:
        mgr.stop()
    assert exc_info2.value.reason == "missing"


@pytest.mark.requires_git
def test_target_resolution_checkout_path_tracked_linked_worktree(
    yaml_manager, git_repo: Path, tmp_path: Path
):
    mgr = yaml_manager()
    wt_path = tmp_path / "tracked-wt"
    import subprocess
    subprocess.run(
        ["git", "worktree", "add", str(wt_path), "feature/alpha"],
        cwd=git_repo, check=True, capture_output=True,
    )
    rec = WorktreeRecord(
        id="tracked-linked",
        repo_root=git_repo.resolve().as_posix(),
        branch="feature/alpha",
        path=wt_path.resolve().as_posix(),
        backing="worktree",
    )
    mgr.state.add(rec)

    before_count = len(mgr.state.list())
    result = mgr.start(checkout_path=str(wt_path))
    assert result.id == "tracked-linked"
    assert len(mgr.state.list()) == before_count

    subprocess.run(
        ["git", "worktree", "remove", "--force", str(wt_path)],
        cwd=git_repo, capture_output=True,
    )


@pytest.mark.requires_git
def test_target_resolution_checkout_path_untracked_linked_worktree(
    yaml_manager, git_repo: Path, linked_worktree: Path
):
    """An untracked linked worktree raises WorktreeNotFoundError and writes
    no record (materialisation is primary-only)."""
    mgr = yaml_manager()
    before = mgr.state.list()
    with pytest.raises(WorktreeNotFoundError):
        mgr.start(checkout_path=str(linked_worktree))
    assert mgr.state.list() == before

    with pytest.raises(WorktreeNotFoundError):
        mgr.stop(checkout_path=str(linked_worktree))
    assert mgr.state.list() == before


@pytest.mark.requires_git
def test_target_resolution_subdirectory_of_tracked_linked_worktree(
    yaml_manager, git_repo: Path, tmp_path: Path
):
    """A *subdirectory* of an already-tracked linked worktree must resolve
    to that worktree's record, mirroring how the primary branch resolves
    subdirectories via primary_id_for()/repo_root (review finding, ticket
    #84 fix cycle). Resolution must not write a new record -- materialisation
    stays primary-only."""
    mgr = yaml_manager()
    wt_path = tmp_path / "tracked-wt"
    import subprocess
    subprocess.run(
        ["git", "worktree", "add", str(wt_path), "feature/alpha"],
        cwd=git_repo, check=True, capture_output=True,
    )
    rec = WorktreeRecord(
        id="tracked-linked",
        repo_root=git_repo.resolve().as_posix(),
        branch="feature/alpha",
        path=wt_path.resolve().as_posix(),
        backing="worktree",
    )
    mgr.state.add(rec)
    sub_path = wt_path / "sub"
    sub_path.mkdir()

    before_count = len(mgr.state.list())
    result = mgr.start(checkout_path=str(sub_path))
    assert result.id == "tracked-linked"
    assert len(mgr.state.list()) == before_count

    subprocess.run(
        ["git", "worktree", "remove", "--force", str(wt_path)],
        cwd=git_repo, capture_output=True,
    )


@pytest.mark.requires_git
def test_target_resolution_subdirectory_of_untracked_linked_worktree(
    yaml_manager, git_repo: Path, linked_worktree: Path
):
    """A subdirectory of an *untracked* linked worktree still raises
    WorktreeNotFoundError and writes no record."""
    mgr = yaml_manager()
    sub_path = linked_worktree / "sub"
    sub_path.mkdir()
    before = mgr.state.list()
    with pytest.raises(WorktreeNotFoundError):
        mgr.start(checkout_path=str(sub_path))
    assert mgr.state.list() == before


@pytest.mark.requires_git
def test_target_resolution_id_mismatch(yaml_manager, git_repo: Path):
    mgr = yaml_manager()
    with pytest.raises(CheckoutTargetError) as exc_info:
        mgr.start(worktree_id="totally-wrong-id", checkout_path=str(git_repo))
    assert exc_info.value.reason == "id_mismatch"
    assert exc_info.value.resolved_id == primary_id_for(git_repo)
    # No materialisation occurred on mismatch.
    assert mgr.state.get(primary_id_for(git_repo)) is None


@pytest.mark.requires_git
def test_target_resolution_matching_id_and_path_materialises(
    yaml_manager, git_repo: Path
):
    mgr = yaml_manager()
    expected_id = primary_id_for(git_repo)
    record = mgr.start(worktree_id=expected_id, checkout_path=str(git_repo))
    assert record.id == expected_id
    assert record.backing == "primary"


@pytest.mark.requires_git
def test_target_resolution_non_repo_dir_raises_before_state_access(
    yaml_manager, tmp_path: Path
):
    mgr = yaml_manager()
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    with pytest.raises(InvalidRepoError):
        mgr.start(checkout_path=str(not_a_repo))
    assert mgr.state.list() == []


def test_bare_worktree_id_for_unstarted_primary_raises_not_found(
    yaml_manager, tmp_path: Path
):
    """A bare primary id that was never started still raises
    WorktreeNotFoundError (no classification/materialisation is attempted
    for the id-only path)."""
    mgr = yaml_manager()
    with pytest.raises(WorktreeNotFoundError) as exc_info:
        mgr.start("some-primary-id-never-started")
    assert "checkout_path" in str(exc_info.value)


# ---------------------------------------------------------------------------
# R8 -- primary branch is read live, never stored stale
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_primary_start_exports_live_branch(yaml_manager, git_repo: Path):
    mgr = yaml_manager()
    out_file = git_repo.parent / "branch_out.txt"
    script = git_repo.parent / "write_branch.py"
    script.write_text(
        "import os\n"
        f"open(r'{out_file}', 'w', encoding='utf-8').write(os.environ.get('WORKTREE_BRANCH', ''))\n",
        encoding="utf-8",
    )
    _write_contract(
        git_repo,
        "version: 1\nisolation: full\nstart:\n"
        + _step_yaml(_run_line_for_script(script)),
    )

    record = mgr.start(checkout_path=str(git_repo))
    assert record.branch is None
    # Wait for the detached write-and-exit step to land its output. Uses a
    # generous, CI-suited budget (see _wait_for_file_content) rather than
    # asserting on the file directly, since a bare read_text() on a not-yet
    # -written file raises an uninformative FileNotFoundError.
    _wait_for_file_content(out_file, "main", record=record)

    mgr.stop(checkout_path=str(git_repo))

    import subprocess
    subprocess.run(["git", "checkout", "feature/alpha"], cwd=git_repo, check=True, capture_output=True)

    record2 = mgr.start(checkout_path=str(git_repo))
    assert record2.branch is None
    _wait_for_file_content(out_file, "feature/alpha", record=record2)

    try:
        mgr.stop(checkout_path=str(git_repo))
    except Exception:  # noqa: BLE001
        pass


def test_effective_branch_linked_worktree_no_git_call(monkeypatch):
    """_effective_branch for a linked worktree (non-empty branch) returns
    the stored value without spawning git."""
    from lib_python_worktree.core import manager as manager_module

    calls = {"n": 0}

    def _counting_run_git(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("git must not be called for a linked worktree")

    monkeypatch.setattr(manager_module, "_run_git", _counting_run_git)

    rec = WorktreeRecord(
        id="wt-x", repo_root="/fake/repo", branch="feature/x", path="/fake/repo-wt",
        backing="worktree",
    )
    assert manager_module._effective_branch(rec) == "feature/x"
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# R9 -- port reservations are incremental and survive stop -> restart
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
def test_ports_survive_stop_and_new_slot_allocated_incrementally(
    yaml_manager, git_repo: Path
):
    mgr = yaml_manager()
    _write_contract(
        git_repo,
        "version: 1\nisolation: full\nports:\n  - name: web\n",
    )

    record = mgr.start(checkout_path=str(git_repo))
    assert record.status == "ready"
    web_port = record.ports["web"]
    assert isinstance(web_port, int)

    stopped = mgr.stop(checkout_path=str(git_repo))
    assert stopped.ports["web"] == web_port

    from lib_python_worktree.core.yaml_store import reconcile
    reconcile(mgr.state)
    after_reconcile = mgr.state.get(primary_id_for(git_repo))
    assert after_reconcile.ports["web"] == web_port

    _write_contract(
        git_repo,
        "version: 1\nisolation: full\nports:\n  - name: web\n  - name: api\n",
    )
    record2 = mgr.start(checkout_path=str(git_repo))
    assert record2.ports["web"] == web_port
    assert "api" in record2.ports
    assert record2.ports["api"] != web_port

    all_ports = mgr.state._ports.get_all() if hasattr(mgr.state, "_ports") else None
    if all_ports is not None:
        pid = primary_id_for(git_repo)
        assert all_ports.get(f"{pid}:web") == web_port
        assert all_ports.get(f"{pid}:api") == record2.ports["api"]

    mgr.stop(checkout_path=str(git_repo))


@pytest.mark.requires_git
def test_start_no_ports_allocates_nothing_and_takes_no_lock(
    yaml_manager, git_repo: Path, monkeypatch
):
    mgr = yaml_manager()
    calls = {"n": 0}
    real_allocate = mgr._allocator.allocate

    def _counting_allocate(slots, worktree_id, **kwargs):
        calls["n"] += 1
        return real_allocate(slots, worktree_id, **kwargs)

    monkeypatch.setattr(mgr._allocator, "allocate", _counting_allocate)

    record = mgr.start(checkout_path=str(git_repo))
    assert record.ports == {}
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# Ticket #166 -- stale `ports` survive a contract change; a mislabelled
# primary heals to backing="primary" at list time
# ---------------------------------------------------------------------------
#
# NOTE on fixture choice: the `yaml_manager` fixture factory (conftest.py)
# hard-codes `reconcile_on_init=False` for reasons unrelated to this ticket
# (see its own tests elsewhere), which means a manager built through it never
# runs `reconcile()` from `list()`/`list_repo()` at all -- exactly the call
# path this ticket's list-time prune/heal lives in. The established idiom for
# exercising that reconcile-triggered path (see `TestManagerListReconciles`
# and `test_reconcile_heal_never_endangers_owned_branch_deletion` in
# test_manager.py) is either constructing a `WorktreeManager` directly with
# the default `reconcile_on_init=True`, or -- when `create()`/`start()`/
# `stop()` via `yaml_manager()` already did the real git/port-allocation
# work -- wrapping the SAME on-disk `state`/`config` in a second manager with
# that default, used only for the listing calls under test. Both patterns are
# used below.

def _reconciling_reader(mgr: WorktreeManager) -> WorktreeManager:
    """A second manager over *mgr*'s own on-disk state/config, with the
    default ``reconcile_on_init=True`` -- see the NOTE above."""
    return WorktreeManager(config=mgr.config, state=mgr.state)


@pytest.mark.requires_git
def test_list_prunes_ports_after_downgrade_to_isolation_none(
    yaml_manager, git_repo: Path
):
    """R1 driving test: a stopped primary downgraded to isolation:none lists
    ports: {} without a restart -- both list_repo() and list() reflect it,
    and the port is actually freed in ports.yaml, not just hidden."""
    mgr = yaml_manager()
    _write_contract(
        git_repo, "version: 1\nisolation: full\nports:\n  - name: web\n"
    )

    record = mgr.start(checkout_path=str(git_repo))
    web_port = record.ports["web"]
    assert isinstance(web_port, int)
    mgr.stop(checkout_path=str(git_repo))

    _write_contract(git_repo, "version: 1\nisolation: none\n")

    reader = _reconciling_reader(mgr)

    listing = reader.list_repo(str(git_repo))
    primary_entries = [e for e in listing.entries if e.record.backing == "primary"]
    assert len(primary_entries) == 1
    assert primary_entries[0].record.ports == {}

    listed = reader.list()
    assert listed[0].ports == {}

    persisted = mgr.state.get(primary_id_for(git_repo))
    assert persisted.ports == {}

    all_ports = mgr.state._ports.get_all()
    pid = primary_id_for(git_repo)
    assert f"{pid}:web" not in all_ports


@pytest.mark.requires_git
def test_list_keeps_ports_when_contract_forbids_isolation_none_with_ports(
    yaml_manager, git_repo: Path
):
    """R1 edge-case (a): a structurally-invalid contract (isolation: none
    declared alongside ports:) must not raise from list()/list_repo() --
    declared slots are unknown in this case (the load itself fails), so
    pruning is skipped and existing ports are kept -- while start() still
    raises ContractValidationError exactly as before this ticket (R5,
    existing-suite: test_contract.py::test_isolation_none_forbids_ports)."""
    from lib_python_worktree.contract.loader import ContractValidationError

    mgr = yaml_manager()
    _write_contract(
        git_repo, "version: 1\nisolation: full\nports:\n  - name: web\n"
    )
    mgr.start(checkout_path=str(git_repo))
    mgr.stop(checkout_path=str(git_repo))

    _write_contract(
        git_repo, "version: 1\nisolation: none\nports:\n  - name: web\n"
    )

    reader = _reconciling_reader(mgr)
    listing = reader.list_repo(str(git_repo))  # must not raise
    primary_entries = [e for e in listing.entries if e.record.backing == "primary"]
    assert len(primary_entries) == 1
    assert "web" in primary_entries[0].record.ports

    with pytest.raises(ContractValidationError):
        mgr.start(checkout_path=str(git_repo))


@pytest.mark.requires_git
def test_list_prunes_ports_when_contract_file_deleted(
    yaml_manager, git_repo: Path
):
    """R1 edge-case (b): a missing contract file loads as an implicit
    isolation: none contract (no ports at all) -- pruning treats it exactly
    like an explicit isolation: none downgrade."""
    mgr = yaml_manager()
    _write_contract(
        git_repo, "version: 1\nisolation: full\nports:\n  - name: web\n"
    )
    mgr.start(checkout_path=str(git_repo))
    mgr.stop(checkout_path=str(git_repo))

    (git_repo / ".seretos" / "worktree-setup.yml").unlink()

    reader = _reconciling_reader(mgr)
    listing = reader.list_repo(str(git_repo))
    primary_entries = [e for e in listing.entries if e.record.backing == "primary"]
    assert primary_entries[0].record.ports == {}


@pytest.mark.requires_git
def test_list_keeps_ports_when_contract_yaml_invalid(
    yaml_manager, git_repo: Path
):
    """R1 edge-case (c): a contract file that fails to even parse as YAML
    must not raise from listing and must not prune -- declared slots are
    genuinely unknown here (the load raised), not "none"."""
    mgr = yaml_manager()
    _write_contract(
        git_repo, "version: 1\nisolation: full\nports:\n  - name: web\n"
    )
    mgr.start(checkout_path=str(git_repo))
    mgr.stop(checkout_path=str(git_repo))

    # Same broken-YAML fixture text already established in test_manager.py's
    # test_manager_start_contract_invalid_yaml_raises_contract_error.
    _write_contract(git_repo, "version: 1\n  bad: indent: here\n")

    reader = _reconciling_reader(mgr)
    listing = reader.list_repo(str(git_repo))  # must not raise
    primary_entries = [e for e in listing.entries if e.record.backing == "primary"]
    assert "web" in primary_entries[0].record.ports


@pytest.mark.requires_git
def test_list_keeps_ports_when_role_pid_alive(tmp_path: Path, git_repo: Path):
    """R1 edge-case (e): a record with a live tracked pid must not be
    pruned -- the process may still hold the port. Pruning only applies at
    the first list/start after stop (the plan's "not pids" qualifier)."""
    store = YamlStateStore(state_dir=tmp_path / "state")
    _write_contract(git_repo, "version: 1\nisolation: none\n")
    rec = WorktreeRecord(
        id=primary_id_for(git_repo),
        repo_root=git_repo.resolve().as_posix(),
        path=git_repo.resolve().as_posix(),
        branch=None,
        backing="primary",
        ports={"web": 31234},
        pids={"main": os.getpid()},
    )
    store.add(rec)
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"), state=store
    )

    listed = mgr.list()
    assert listed[0].ports == {"web": 31234}


@pytest.mark.requires_git
def test_list_prunes_linked_worktree_by_repo_root_contract(
    yaml_manager, git_repo: Path
):
    """R1d driving test: a linked worktree is pruned against the
    REPO-ROOT contract, not any checkout-local copy -- the same contract
    start() itself reads (misread::M1 in the plan: every engine read
    composes Path(record.repo_root) / CONTRACT_FILENAME, never
    record.path). A stale checkout-local copy that still declares the
    removed slot must have zero effect."""
    mgr = yaml_manager()
    _write_contract(
        git_repo,
        "version: 1\nisolation: full\nports:\n  - name: web\n  - name: db\n",
    )
    rec = mgr.create(str(git_repo), "feature/r1d-prune", base="main", fetch=False)
    started = mgr.start(worktree_id=rec.id)
    web_port = started.ports["web"]
    assert "db" in started.ports
    mgr.stop(worktree_id=rec.id)

    # Repo root drops the db slot...
    _write_contract(
        git_repo, "version: 1\nisolation: full\nports:\n  - name: web\n"
    )
    # ...but the checkout-local copy (never read by start()/reconcile) still
    # declares both -- if pruning ever consulted THIS file instead of the
    # repo-root one, db would incorrectly survive.
    _write_contract(
        Path(rec.path),
        "version: 1\nisolation: full\nports:\n  - name: web\n  - name: db\n",
    )

    reader = _reconciling_reader(mgr)
    listing = reader.list_repo(str(git_repo))
    linked_entry = next(e for e in listing.entries if e.record.id == rec.id)
    assert linked_entry.record.ports == {"web": web_port}

    all_ports = mgr.state._ports.get_all()
    assert f"{rec.id}:db" not in all_ports
    assert all_ports.get(f"{rec.id}:web") == web_port

    # start() reaches the same verdict, from the same repo-root contract,
    # with no flap on the surviving port.
    restarted = mgr.start(worktree_id=rec.id)
    assert restarted.ports == {"web": web_port}

    mgr.stop(worktree_id=rec.id)


@pytest.mark.requires_git
def test_list_and_start_ignore_checkout_local_contract_drop(
    yaml_manager, git_repo: Path
):
    """R1d edge-case (reverse): the repo root keeps both slots; only the
    (never-read) checkout-local copy drops one. Both list_repo() and
    start() must keep the slot the repo-root contract still declares."""
    mgr = yaml_manager()
    _write_contract(
        git_repo,
        "version: 1\nisolation: full\nports:\n  - name: web\n  - name: db\n",
    )
    rec = mgr.create(str(git_repo), "feature/r1d-reverse", base="main", fetch=False)
    started = mgr.start(worktree_id=rec.id)
    web_port = started.ports["web"]
    db_port = started.ports["db"]
    mgr.stop(worktree_id=rec.id)

    # Repo root still declares both; only the checkout-local copy drops db.
    _write_contract(
        Path(rec.path), "version: 1\nisolation: full\nports:\n  - name: web\n"
    )

    reader = _reconciling_reader(mgr)
    listing = reader.list_repo(str(git_repo))
    linked_entry = next(e for e in listing.entries if e.record.id == rec.id)
    assert linked_entry.record.ports == {"web": web_port, "db": db_port}

    restarted = mgr.start(worktree_id=rec.id)
    assert restarted.ports == {"web": web_port, "db": db_port}

    mgr.stop(worktree_id=rec.id)


@pytest.mark.requires_git
def test_start_prunes_slot_removed_from_contract(yaml_manager, git_repo: Path):
    """R2 driving test: a removed slot disappears at start() -- undeclared
    slots are dropped from the record and from ports.yaml, declared ones
    are kept and their port numbers do not flap."""
    mgr = yaml_manager()
    _write_contract(
        git_repo,
        "version: 1\nisolation: full\nports:\n  - name: web\n  - name: db\n",
    )
    record = mgr.start(checkout_path=str(git_repo))
    web_port = record.ports["web"]
    assert "db" in record.ports
    mgr.stop(checkout_path=str(git_repo))

    _write_contract(
        git_repo, "version: 1\nisolation: full\nports:\n  - name: web\n"
    )

    record2 = mgr.start(checkout_path=str(git_repo))
    assert record2.ports == {"web": web_port}

    persisted = mgr.state.get(primary_id_for(git_repo))
    assert persisted.ports == {"web": web_port}

    pid = primary_id_for(git_repo)
    all_ports = mgr.state._ports.get_all()
    assert f"{pid}:db" not in all_ports
    assert all_ports.get(f"{pid}:web") == web_port

    mgr.stop(checkout_path=str(git_repo))


@pytest.mark.requires_git
def test_start_prunes_all_ports_after_isolation_none_downgrade(
    yaml_manager, git_repo: Path
):
    """R2 edge-case: after an isolation:none downgrade, start() itself
    (not just listing) returns ports == {}."""
    mgr = yaml_manager()
    _write_contract(
        git_repo, "version: 1\nisolation: full\nports:\n  - name: web\n"
    )
    mgr.start(checkout_path=str(git_repo))
    mgr.stop(checkout_path=str(git_repo))

    _write_contract(git_repo, "version: 1\nisolation: none\n")
    record2 = mgr.start(checkout_path=str(git_repo))
    assert record2.ports == {}


@pytest.mark.requires_git
def test_list_heals_mislabelled_primary_backing(tmp_path: Path, git_repo: Path):
    """R3 driving test: a record with path == repo_root but a stale
    backing="worktree" heals to "primary" at list time -- this is a live
    bad state (no write site produces it), so it is healed by reconcile()
    rather than fixed at a write site."""
    store = YamlStateStore(state_dir=tmp_path / "state")
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"), state=store
    )
    rec = WorktreeRecord(
        id=primary_id_for(git_repo),
        repo_root=git_repo.resolve().as_posix(),
        path=git_repo.resolve().as_posix(),
        branch=None,
        backing="worktree",
    )
    mgr.state.add(rec)

    listing = mgr.list_repo(str(git_repo))
    primary_entries = [
        e for e in listing.entries if e.record.id == primary_id_for(git_repo)
    ]
    assert len(primary_entries) == 1
    assert primary_entries[0].record.backing == "primary"

    persisted = mgr.state.get(primary_id_for(git_repo))
    assert persisted.backing == "primary"


@pytest.mark.requires_git
def test_list_leaves_linked_worktree_backing_unchanged(
    tmp_path: Path, git_repo: Path, linked_worktree: Path
):
    """R3 edge-case: a genuinely linked worktree record (path != repo_root)
    must not be relabelled -- the heal is scoped to path == repo_root."""
    store = YamlStateStore(state_dir=tmp_path / "state")
    mgr = WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"), state=store
    )
    rec = WorktreeRecord(
        id="linked-r3-unchanged",
        repo_root=git_repo.resolve().as_posix(),
        path=linked_worktree.resolve().as_posix(),
        branch="feature/alpha",
        backing="worktree",
    )
    mgr.state.add(rec)

    listing = mgr.list_repo(str(git_repo))
    linked_entries = [
        e for e in listing.entries if e.record.id == "linked-r3-unchanged"
    ]
    assert len(linked_entries) == 1
    assert linked_entries[0].record.backing == "worktree"
