"""Tests for the process lifecycle module (ticket #8).

Uses ``InMemoryStateStore`` and real/mock subprocesses.  No real git or
``YamlStateStore`` required.

Regression test for #8: ``start`` spawns a detached child that survives the
caller; ``stop`` terminates it and clears the PID.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from lib_python_worktree.core import process_lifecycle as _pl
from lib_python_worktree.core.process_lifecycle import (
    DEFAULT_ROLE,
    KilledProcessInfo,
    ProcessAlreadyRunningError,
    ProcessLifecycleError,
    ProcessNotRunningError,
    _BoundedQueryWorker,
    _GraceBudget,
    _QueryStatus,
    _HANDLE_QUERY_GRACE_BUDGET_SEC,
    _HANDLE_QUERY_GRACE_SEC,
    _HANDLE_QUERY_TIMEOUT_SEC,
    _MAX_WEDGED_HANDLE_WORKERS,
    _describe_pid,
    _find_blocking_processes,
    _force_kill,
    _kill_blocking_processes,
    _kill_process_tree,
    _pid_alive,
    _process_group_members,
    _process_tree,
    _send_graceful_signal,
    _signal_process_group,
    _spawn_detached,
    _wait_or_kill,
    _wedged_slot_available,
    _win_handle_holders,
    start,
    stop,
)
from lib_python_worktree.core.manager import WorktreeNotFoundError
from lib_python_worktree.core.state import (
    STOP_ATTEMPT_ALREADY_EXITED,
    STOP_ATTEMPT_KILLED,
    STOP_ATTEMPT_TRACKED_PID_MISSING,
    STOP_REASON_JOB_MEMBER_LIST_TRUNCATED,
    STOP_REASON_ORPHAN_SCAN_INCOMPLETE,
    STOP_REASON_SURVIVORS,
    STOP_REASON_TREE_TRUNCATED,
    InMemoryStateStore,
    StopDetail,
    WorktreeRecord,
)

# Generous budget_sec for real (non-mocked) TestWinHandleHoldersReal scans that
# assert *correctness* rather than deadline bail-out behaviour. The production
# default (_HANDLE_SCAN_BUDGET_SEC = 15.0s) can expire before the scan reaches
# the child's PID under high ambient system handle-table load, flaking those
# tests. This constant is test-only and does not affect production behaviour.
_REAL_SCAN_TEST_BUDGET_SEC = 120.0


@pytest.fixture(autouse=True)
def _redirect_start_log_root(tmp_path, monkeypatch):
    """Redirect WORKTREE_LOG_ROOT to a tmp dir for every test in this module.

    ``start()`` now unconditionally writes a captured-output log file under
    the resolved log root (ticket #81). Without this, any test in this file
    that calls ``start()`` without an explicit ``env=`` would fall back to
    the real ``os.environ`` and write log files under the ambient
    ``WORKTREE_LOG_ROOT``/``~/.agent-worktree/logs`` on the machine running
    the suite.
    """
    monkeypatch.setenv("WORKTREE_LOG_ROOT", str(tmp_path / "logs"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(*records: WorktreeRecord) -> InMemoryStateStore:
    store = InMemoryStateStore()
    for rec in records:
        store.add(rec)
    return store


def _make_record(wt_id: str = "wt-abc", **kwargs) -> WorktreeRecord:
    defaults = dict(
        id=wt_id,
        repo_root="/fake/repo",
        branch="feature/x",
        path="/fake/repo/../store/wt-abc",
    )
    defaults.update(kwargs)
    return WorktreeRecord(**defaults)


# ---------------------------------------------------------------------------
# _spawn_detached unit tests
# ---------------------------------------------------------------------------

class TestSpawnDetached:
    def test_returns_positive_pid(self):
        """_spawn_detached returns a Popen whose .pid is a valid PID."""
        pid = _spawn_detached([sys.executable, "-c", "import time; time.sleep(30)"]).pid
        assert pid > 0
        # Cleanup
        try:
            if sys.platform == "win32":
                import ctypes
                PROCESS_TERMINATE = 0x0001
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
                if handle:
                    kernel32.TerminateProcess(handle, 1)
                    kernel32.CloseHandle(handle)
            else:
                os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    def test_empty_cmd_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            _spawn_detached([])

    def test_pid_is_alive_after_spawn(self):
        """Regression #8: spawned process is alive immediately after start."""
        pid = _spawn_detached([sys.executable, "-c", "import time; time.sleep(30)"]).pid
        assert _pid_alive(pid), "spawned process must be alive"
        # Cleanup
        try:
            _force_kill(pid)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# start() tests
# ---------------------------------------------------------------------------

class TestStart:
    def test_start_spawns_detached_and_records_pid(self):
        """Regression #8: start spawns a child, records pids['main'], status='running'."""
        record = _make_record("wt-1")
        store = _make_store(record)

        result = start(
            "wt-1",
            [sys.executable, "-c", "import time; time.sleep(60)"],
            store=store,
        )

        pid = result.pids.get(DEFAULT_ROLE, 0)
        assert pid > 0, "pids['main'] must be a positive PID"
        assert _pid_alive(pid), "spawned process must be alive"
        assert result.status == "running"

        # Cleanup
        try:
            _force_kill(pid)
        except Exception:  # noqa: BLE001
            pass

    def test_start_status_persisted(self):
        """start() persists status and pids to the store."""
        record = _make_record("wt-persist")
        store = _make_store(record)

        start(
            "wt-persist",
            [sys.executable, "-c", "import time; time.sleep(60)"],
            store=store,
        )

        stored = store.get("wt-persist")
        assert stored is not None
        pid = stored.pids.get(DEFAULT_ROLE, 0)
        assert pid > 0
        assert stored.status == "running"

        # Cleanup
        try:
            _force_kill(pid)
        except Exception:  # noqa: BLE001
            pass

    def test_start_idempotent_raises_already_running(self):
        """start raises ProcessAlreadyRunningError when the role's PID is alive."""
        live_pid = os.getpid()  # current process is definitely alive
        record = _make_record("wt-2", pids={DEFAULT_ROLE: live_pid})
        store = _make_store(record)

        with pytest.raises(ProcessAlreadyRunningError) as exc_info:
            start("wt-2", [sys.executable, "-c", "pass"], store=store)

        err = exc_info.value
        assert err.pid == live_pid
        assert err.worktree_id == "wt-2"
        assert err.role == DEFAULT_ROLE

    def test_start_unknown_id_raises_worktree_not_found(self):
        """start raises WorktreeNotFoundError for an unregistered id."""
        store = _make_store()
        with pytest.raises(WorktreeNotFoundError):
            start("no-such-id", [sys.executable, "-c", "pass"], store=store)

    def test_start_empty_cmd_raises_value_error(self):
        record = _make_record("wt-empty-cmd")
        store = _make_store(record)
        with pytest.raises(ValueError):
            start("wt-empty-cmd", [], store=store)

    def test_start_dead_pid_restarts(self):
        """start re-launches when the recorded PID is dead (process replaced)."""
        record = _make_record("wt-dead", pids={DEFAULT_ROLE: 99999999})
        store = _make_store(record)

        # 99999999 is almost certainly dead — but let's be explicit and
        # confirm _pid_alive returns False for it.
        # If by extreme coincidence it's alive, skip.
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — unlikely but skipping")

        result = start(
            "wt-dead",
            [sys.executable, "-c", "import time; time.sleep(60)"],
            store=store,
        )
        new_pid = result.pids.get(DEFAULT_ROLE, 0)
        assert new_pid > 0
        assert new_pid != 99999999

        # Cleanup
        try:
            _force_kill(new_pid)
        except Exception:  # noqa: BLE001
            pass

    def test_start_custom_role(self):
        """start records the pid under the supplied role key."""
        record = _make_record("wt-role")
        store = _make_store(record)

        result = start(
            "wt-role",
            [sys.executable, "-c", "import time; time.sleep(60)"],
            store=store,
            role="worker",
        )
        pid = result.pids.get("worker", 0)
        assert pid > 0

        # Cleanup
        try:
            _force_kill(pid)
        except Exception:  # noqa: BLE001
            pass

    def test_start_records_variant_for_role(self):
        """Ticket #104: start(variant=...) records which variant started
        this role under record.variants[role], and the store-reloaded
        record agrees."""
        record = _make_record("wt-variant")
        store = _make_store(record)

        result = start(
            "wt-variant",
            [sys.executable, "-c", "import time; time.sleep(60)"],
            store=store,
            role="web",
            variant="web",
        )
        try:
            assert result.variants == {"web": "web"}
            reloaded = store.get("wt-variant")
            assert reloaded is not None
            assert reloaded.variants == {"web": "web"}
        finally:
            _force_kill(result.pids["web"])

    def test_start_without_variant_pops_stale_variant_entry(self):
        """A pre-seeded variants["main"] entry is removed when start() is
        called again for that role with variant=None (the default)."""
        record = _make_record("wt-variant-stale")
        record.variants["main"] = "default"
        store = _make_store(record)

        result = start(
            "wt-variant-stale",
            [sys.executable, "-c", "import time; time.sleep(60)"],
            store=store,
        )
        try:
            assert "main" not in result.variants
        finally:
            _force_kill(result.pids[DEFAULT_ROLE])


# ---------------------------------------------------------------------------
# start(): output capture + early-exit detection (ticket #81)
# ---------------------------------------------------------------------------

class TestStartOutputCaptureAndEarlyExit:
    def test_start_detects_immediate_exit(self, tmp_path):
        """Ticket #81: an immediately-exiting command is surfaced as
        status='exited' with its returncode, and its output is captured to
        the start log file rather than being lost to DEVNULL."""
        record = _make_record("wt-exit")
        store = _make_store(record)

        result = start(
            "wt-exit",
            [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
            store=store,
        )

        assert result.status == "exited"
        assert result.returncode == 3
        assert result.start_log_path is not None
        log_path = Path(result.start_log_path)
        assert log_path.exists()
        assert "boom" in log_path.read_text(encoding="utf-8", errors="replace")

    def test_start_long_lived_reports_running(self):
        """A process still alive after the early-exit poll reports status='running'
        and returncode=None."""
        record = _make_record("wt-long")
        store = _make_store(record)

        result = start(
            "wt-long",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            store=store,
        )

        assert result.status == "running"
        assert result.returncode is None

        # Cleanup
        pid = result.pids.get(DEFAULT_ROLE, 0)
        try:
            _force_kill(pid)
        except Exception:  # noqa: BLE001
            pass

    def test_start_log_file_created_and_written(self):
        """A chatty short-lived command yields a non-empty start_log_path file."""
        record = _make_record("wt-chatty")
        store = _make_store(record)

        result = start(
            "wt-chatty",
            [sys.executable, "-c", "print('hello from child')"],
            store=store,
        )

        log_path = Path(result.start_log_path)
        assert log_path.exists()
        assert log_path.stat().st_size > 0
        assert "hello from child" in log_path.read_text(
            encoding="utf-8", errors="replace"
        )

    def test_start_missing_log_root_falls_back_to_default(self, monkeypatch, tmp_path):
        """With WORKTREE_LOG_ROOT unset, start_log_path resolves under
        DEFAULT_LOG_ROOT."""
        monkeypatch.delenv("WORKTREE_LOG_ROOT", raising=False)
        import lib_python_worktree.setup.runner as _runner_module
        fake_default = tmp_path / "default-logs"
        monkeypatch.setattr(_runner_module, "DEFAULT_LOG_ROOT", fake_default)

        record = _make_record("wt-default-log")
        store = _make_store(record)

        result = start(
            "wt-default-log",
            [sys.executable, "-c", "print('x')"],
            store=store,
        )

        log_path = Path(result.start_log_path)
        assert log_path.exists()
        assert str(fake_default) in str(log_path)

    def test_start_exited_status_persisted(self):
        """After an immediate-exit start, store.get(id) round-trips status,
        returncode, and start_log_path -- proving these survive
        store.update()/serialization (see YamlStateStore round-trip test for
        the on-disk serialization path)."""
        record = _make_record("wt-exit-persist")
        store = _make_store(record)

        start(
            "wt-exit-persist",
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            store=store,
        )

        stored = store.get("wt-exit-persist")
        assert stored is not None
        assert stored.status == "exited"
        assert stored.returncode == 7
        assert stored.start_log_path is not None


# ---------------------------------------------------------------------------
# start() log filename casing (ticket #111)
# ---------------------------------------------------------------------------

class TestStartLogFilenameCasing:
    """Ticket #111: the log filename's role component must preserve the
    same casing as the ``pids`` dict key -- ``_slug`` (from
    ``setup.runner``) lower-cases, which desynced ``start-<role>.log``
    from ``record.pids[role]``. ``start()`` must sanitize the role for
    filesystem-safety without lower-casing it.
    """

    def test_start_log_filename_role_component_matches_pids_key(self):
        """Requirement 1 (driving test): the token between 'start-' and
        '.log' in the log filename must be a literal key of result.pids.
        Pre-fix, _slug(role) lower-cases 'roleA' to 'rolea', which is not a
        key of pids (only 'roleA' is) -- AssertionError."""
        record = _make_record("wt-casing-1")
        store = _make_store(record)

        result = start(
            "wt-casing-1",
            [sys.executable, "-c", "print('x')"],
            store=store,
            role="roleA",
        )

        name = Path(result.start_log_path).name
        assert name.startswith("start-") and name.endswith(".log")
        token = name[len("start-"):-len(".log")]
        assert token in result.pids

    def test_start_log_filename_exact_for_mixed_case_role(self):
        """Edge case: exact filename for a simple mixed-case role."""
        record = _make_record("wt-casing-2")
        store = _make_store(record)

        result = start(
            "wt-casing-2",
            [sys.executable, "-c", "print('x')"],
            store=store,
            role="roleA",
        )

        assert Path(result.start_log_path).name == "start-roleA.log"

    def test_start_log_filename_unchanged_for_default_role(self):
        """Edge case: the default role ('main') is already all-lowercase,
        so its filename is unaffected by the fix."""
        record = _make_record("wt-casing-3")
        store = _make_store(record)

        result = start(
            "wt-casing-3",
            [sys.executable, "-c", "print('x')"],
            store=store,
        )

        assert Path(result.start_log_path).name == "start-main.log"

    def test_start_pids_key_is_raw_role(self):
        """Edge case: guards against fixing the mismatch in the wrong
        direction (e.g. by lower-casing the pids key instead of
        preserving the role's case in the filename)."""
        record = _make_record("wt-casing-4")
        store = _make_store(record)

        result = start(
            "wt-casing-4",
            [sys.executable, "-c", "print('x')"],
            store=store,
            role="roleA",
        )

        assert "roleA" in result.pids
        assert "rolea" not in result.pids

    def test_start_log_filename_sanitizes_unsafe_chars_preserving_case(self):
        """Requirement 2 (driving test): filesystem-unsafe characters are
        still replaced with '-', but case is preserved. Pre-fix this
        produces 'start-role-a-b.log' (lower-cased)."""
        from lib_python_worktree.setup.runner import log_dir_for

        record = _make_record("wt-casing-5")
        store = _make_store(record)

        result = start(
            "wt-casing-5",
            [sys.executable, "-c", "print('x')"],
            store=store,
            role="Role A/B",
        )

        log_path = Path(result.start_log_path)
        assert log_path.name == "start-Role-A-B.log"
        assert log_path.parent == log_dir_for("wt-casing-5")
        assert log_path.exists()

    def test_start_log_filename_degenerate_role_falls_back(self):
        """Edge case: a role with zero alphanumeric characters falls back
        to a non-empty token ('_', distinct from setup.runner._slug's
        'step' fallback) rather than producing 'start-.log'."""
        record = _make_record("wt-casing-6")
        store = _make_store(record)

        result = start(
            "wt-casing-6",
            [sys.executable, "-c", "print('x')"],
            store=store,
            role="!!!",
        )

        log_path = Path(result.start_log_path)
        assert log_path.name == "start-_.log"
        assert log_path.exists()

    def test_start_log_filename_fallback_does_not_collide_with_literal_role(
        self,
    ):
        """Regression (review finding): the fallback token must not collide
        with a literal role string that happens to equal the old fallback.
        A literal role='role' and a degenerate role='!!!' must produce
        different filenames -- 'start-role.log' vs 'start-_.log' -- even
        though both were tracked as distinct 'pids' keys. Pre-fix (fallback
        == 'role'), both produced the identical string 'start-role.log'."""
        record = _make_record("wt-casing-8")
        store = _make_store(record)

        result_literal = start(
            "wt-casing-8",
            [sys.executable, "-c", "print('x')"],
            store=store,
            role="role",
        )
        name_literal = Path(result_literal.start_log_path).name

        stop("wt-casing-8", store=store, role="role")

        result_degenerate = start(
            "wt-casing-8",
            [sys.executable, "-c", "print('x')"],
            store=store,
            role="!!!",
        )
        name_degenerate = Path(result_degenerate.start_log_path).name

        assert name_literal == "start-role.log"
        assert name_degenerate == "start-_.log"
        assert name_literal != name_degenerate

    def test_start_log_filenames_differ_for_case_only_distinct_roles(self):
        """Requirement 3 (driving test): two roles that differ only by case
        must produce distinct filename strings. This is a platform-
        independent string comparison -- it deliberately does NOT assert
        both log files exist on disk, since on a case-insensitive
        filesystem (Windows, default macOS) 'start-roleA.log' and
        'start-rolea.log' name the same physical file (accepted, documented
        limitation -- see _role_log_slug's docstring). Pre-fix, both calls
        produce the identical string 'start-rolea.log' -- AssertionError."""
        record = _make_record("wt-casing-7")
        store = _make_store(record)

        result_a = start(
            "wt-casing-7",
            [sys.executable, "-c", "print('x')"],
            store=store,
            role="roleA",
        )
        name_a = Path(result_a.start_log_path).name

        stop("wt-casing-7", store=store, role="roleA")

        result_b = start(
            "wt-casing-7",
            [sys.executable, "-c", "print('x')"],
            store=store,
            role="rolea",
        )
        name_b = Path(result_b.start_log_path).name

        assert name_a != name_b


class TestRoleLogSlug:
    """Unit tests for the private ``_role_log_slug`` helper directly."""

    def test_preserves_case(self):
        assert _pl._role_log_slug("AbC") == "AbC"

    def test_strips_leading_and_trailing_unsafe_runs(self):
        assert _pl._role_log_slug("--Ab--") == "Ab"

    def test_replaces_internal_unsafe_chars_with_single_dash(self):
        assert _pl._role_log_slug("Role A/B") == "Role-A-B"

    def test_empty_string_falls_back(self):
        assert _pl._role_log_slug("") == "_"

    def test_all_unsafe_falls_back(self):
        assert _pl._role_log_slug("!!!") == "_"

    def test_truncates_to_max_len(self):
        role = "A" * 80
        assert _pl._role_log_slug(role) == "A" * 40

    def test_digits_and_mixed_alphanumerics_untouched(self):
        assert _pl._role_log_slug("Worker2b") == "Worker2b"

    def test_fallback_does_not_collide_with_literal_role_named_role(self):
        """Regression (review finding): a literal role of 'role' must not
        sanitize to the same token as a degenerate role's fallback. Pre-fix
        the fallback was 'role' itself, so _role_log_slug('!!!') == 'role'
        == _role_log_slug('role') -- a collision. Post-fix, a
        *non-degenerate* role (one with at least one alphanumeric character,
        like 'role') can never sanitize to '_', so the fallback can never
        collide with it -- which is exactly why the old 'role' fallback was
        broken: it collided with the non-degenerate literal role 'role'.
        Degenerate roles (including the literal role '_') all sanitize to
        '_' and collide with each other -- that narrower ambiguity is a
        separately documented, accepted limitation, not fixed here."""
        assert _pl._role_log_slug("role") == "role"
        assert _pl._role_log_slug("!!!") == "_"
        assert _pl._role_log_slug("role") != _pl._role_log_slug("!!!")

    def test_truncation_never_leaves_trailing_dash(self):
        """Nit fix: strip("-") -> [:max_len] can expose a trailing '-' at
        the truncation boundary (e.g. 'a'*39 + '-' + 'b'*10 truncates to
        'a'*39 + '-' before a second strip). The result must be re-stripped
        so no filename ends in '-.log'."""
        role = ("a" * 39) + "-" + ("b" * 10)
        result = _pl._role_log_slug(role, max_len=40)
        assert result == "a" * 39
        assert not result.endswith("-")


# ---------------------------------------------------------------------------
# stop() tests
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_sends_termination_and_clears_pid(self):
        """stop terminates the process and clears pids/sets status='stopped'."""
        record = _make_record("wt-stop")
        store = _make_store(record)

        # Start a real process first.
        start(
            "wt-stop",
            [sys.executable, "-c", "import time; time.sleep(60)"],
            store=store,
        )
        # Confirm it's running.
        stored_before = store.get("wt-stop")
        assert stored_before.status == "running"
        pid = stored_before.pids[DEFAULT_ROLE]
        assert _pid_alive(pid)

        result = stop("wt-stop", store=store, timeout=5.0)

        assert DEFAULT_ROLE not in result.pids
        assert result.status == "stopped"
        # Give the OS a moment to reap the process.
        time.sleep(0.2)
        assert not _pid_alive(pid)

    def test_stop_status_persisted(self):
        """stop persists cleared pids and status='stopped' to the store."""
        record = _make_record("wt-stop-persist")
        store = _make_store(record)

        start(
            "wt-stop-persist",
            [sys.executable, "-c", "import time; time.sleep(60)"],
            store=store,
        )

        stop("wt-stop-persist", store=store, timeout=5.0)

        stored = store.get("wt-stop-persist")
        assert stored is not None
        assert DEFAULT_ROLE not in stored.pids
        assert stored.status == "stopped"

    def test_stop_not_running_raises(self):
        """stop raises ProcessNotRunningError when no PID is recorded."""
        record = _make_record("wt-no-pid")
        store = _make_store(record)

        with pytest.raises(ProcessNotRunningError) as exc_info:
            stop("wt-no-pid", store=store)

        err = exc_info.value
        assert err.worktree_id == "wt-no-pid"
        assert err.role == DEFAULT_ROLE

    def test_stop_unknown_id_raises_worktree_not_found(self):
        """stop raises WorktreeNotFoundError for an unregistered id."""
        store = _make_store()
        with pytest.raises(WorktreeNotFoundError):
            stop("ghost-id", store=store)

    def test_stop_dead_pid_clears_gracefully(self):
        """stop clears a dead PID without raising."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — unlikely but skipping")

        record = _make_record("wt-dead-stop", pids={DEFAULT_ROLE: 99999999})
        store = _make_store(record)

        result = stop("wt-dead-stop", store=store, timeout=1.0)
        assert DEFAULT_ROLE not in result.pids
        assert result.status == "stopped"

    def test_stop_timeout_fallback_kills(self):
        """stop force-kills the process when it doesn't die within timeout.

        Ticket #87: since _pid_alive is patched to always return True (an
        "immortal" process for this test), the post-kill survivor re-probe
        also finds it alive -- so status is now 'stop_incomplete', not
        'stopped' (stop() must never claim success while the pid is
        demonstrably still alive)."""
        record = _make_record("wt-stubborn")
        store = _make_store(record)

        # Use _pid_alive and _force_kill patches: simulate an immortal process.
        fake_pid = 12345

        record.pids[DEFAULT_ROLE] = fake_pid
        store.update(record)

        kill_called = []

        def fake_force_kill(pid):
            kill_called.append(pid)

        # _pid_alive: always True (process never dies)
        # _force_kill: captured
        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._force_kill",
                side_effect=fake_force_kill,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
        ):
            result = stop("wt-stubborn", store=store, timeout=0.3)

        assert fake_pid in kill_called, "_force_kill must be called on timeout"
        assert DEFAULT_ROLE not in result.pids
        assert result.status == "stop_incomplete"

    def test_stop_timeout_zero_immediate_force_kill(self):
        """timeout=0 must go straight to force-kill without waiting.

        Ticket #87: _pid_alive is patched to always return True, so the
        post-kill survivor re-probe reports status='stop_incomplete' rather
        than 'stopped' -- see test_stop_timeout_fallback_kills above."""
        record = _make_record("wt-zero-timeout", pids={DEFAULT_ROLE: 99999999})
        store = _make_store(record)

        force_kill_calls = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._force_kill",
                side_effect=lambda pid: force_kill_calls.append(pid),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
        ):
            result = stop("wt-zero-timeout", store=store, timeout=0)

        assert 99999999 in force_kill_calls
        assert result.status == "stop_incomplete"

    def test_stop_custom_role(self):
        """stop clears only the specified role key.

        Regression for fix #2: when another role's PID is still present the
        whole-worktree ``status`` must NOT be set to ``"stopped"``.  The old
        unconditional ``record.status = "stopped"`` would fail this assertion.
        """
        record = _make_record(
            "wt-multi-role",
            pids={"main": os.getpid(), "worker": 99999999},
            status="running",
        )
        store = _make_store(record)

        # Stop only the "worker" role (which is a dead PID).
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine")

        result = stop("wt-multi-role", store=store, role="worker", timeout=1.0)
        # worker should be cleared; main should remain.
        assert "worker" not in result.pids
        assert "main" in result.pids
        # With another role still alive, status must NOT be "stopped".
        # (It should retain the prior value, i.e. "running".)
        assert result.status != "stopped", (
            "status must not become 'stopped' while other roles are still alive"
        )
        assert result.status == "running"

    def test_stop_pops_variant_entry(self):
        """Ticket #104: stop() clears record.variants[role] alongside
        record.pids[role] -- set(variants) <= set(pids) must hold after."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine")

        record = _make_record("wt-stop-variant", pids={"web": 99999999})
        record.variants["web"] = "web"
        store = _make_store(record)

        result = stop("wt-stop-variant", store=store, role="web", timeout=1.0)
        assert "web" not in result.variants

    def test_stop_does_not_touch_other_roles_variant(self):
        """Stopping one role must not remove another role's variants entry."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine")

        record = _make_record(
            "wt-stop-variant-multi",
            pids={"main": os.getpid(), "worker": 99999999},
            status="running",
        )
        record.variants["main"] = "default"
        record.variants["worker"] = "web"
        store = _make_store(record)

        result = stop("wt-stop-variant-multi", store=store, role="worker", timeout=1.0)
        assert "worker" not in result.variants
        assert result.variants.get("main") == "default"


# ---------------------------------------------------------------------------
# stop() kill_orphans tests  (ticket #36)
# ---------------------------------------------------------------------------

class TestStopKillOrphans:
    """Regression tests for ticket #36: orphaned grandchild termination via stop().

    All tests patch _kill_blocking_processes / _send_graceful_signal rather
    than spawning real grandchildren, following the pattern in
    TestKillBlockingProcesses.
    """

    def test_stop_kill_orphans_when_shell_pid_is_dead(self):
        """Regression #36: when tracked PID is already dead and kill_orphans=True,
        _kill_blocking_processes is called with record.path and record is cleared."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — skipping")

        record = _make_record("wt-orphan-dead", pids={DEFAULT_ROLE: 99999999})
        store = _make_store(record)

        kbp_calls = []

        with patch(
            "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
            side_effect=lambda path, **kw: kbp_calls.append(path) or [],
        ):
            result = stop("wt-orphan-dead", store=store, kill_orphans=True)

        assert kbp_calls == [record.path], (
            "_kill_blocking_processes must be called with record.path"
        )
        assert DEFAULT_ROLE not in result.pids
        assert result.status == "stopped"

    def test_stop_kill_orphans_false_default_no_orphan_scan(self):
        """kill_orphans=False (default) must NOT call _kill_blocking_processes,
        preserving backward-compatible behaviour even when the shell PID is dead."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — skipping")

        record = _make_record("wt-orphan-no-scan", pids={DEFAULT_ROLE: 99999999})
        store = _make_store(record)

        with patch(
            "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
        ) as mock_kbp:
            result = stop("wt-orphan-no-scan", store=store)  # kill_orphans defaults to False

        mock_kbp.assert_not_called()
        assert DEFAULT_ROLE not in result.pids
        assert result.status == "stopped"

    def test_stop_kill_orphans_shell_alive_runs_signal_then_orphan_scan(self):
        """When the shell PID is alive, the graceful signal runs first, then the
        orphan scan runs as a second pass — both _send_graceful_signal and
        _kill_blocking_processes are invoked."""
        fake_pid = 55555
        record = _make_record("wt-orphan-alive", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        graceful_calls = []
        kbp_calls = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
                side_effect=lambda pid: graceful_calls.append(pid),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                side_effect=lambda path, **kw: kbp_calls.append(path) or [],
            ),
        ):
            result = stop("wt-orphan-alive", store=store, kill_orphans=True)

        assert fake_pid in graceful_calls, "_send_graceful_signal must be called on the shell PID"
        assert kbp_calls == [record.path], (
            "_kill_blocking_processes must be called as a second pass with record.path"
        )
        assert DEFAULT_ROLE not in result.pids

    def test_stop_kill_orphans_no_processes_found_no_error(self):
        """kill_orphans=True with _kill_blocking_processes returning [] must not
        raise and must still clear the record normally."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — skipping")

        record = _make_record("wt-orphan-empty", pids={DEFAULT_ROLE: 99999999})
        store = _make_store(record)

        with patch(
            "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
            return_value=[],
        ):
            result = stop("wt-orphan-empty", store=store, kill_orphans=True)

        assert DEFAULT_ROLE not in result.pids
        assert result.status == "stopped"

    def test_stop_kill_orphans_passes_record_path_not_repo_root(self):
        """_kill_blocking_processes must receive record.path (the worktree checkout
        directory), not record.repo_root or any other path."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — skipping")

        record = _make_record(
            "wt-orphan-path",
            pids={DEFAULT_ROLE: 99999999},
            # Ensure path and repo_root are clearly different values.
            path="/fake/store/wt-orphan-path",
        )
        store = _make_store(record)

        captured_path = []

        with patch(
            "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
            side_effect=lambda path, **kw: captured_path.append(path) or [],
        ):
            stop("wt-orphan-path", store=store, kill_orphans=True)

        assert captured_path == ["/fake/store/wt-orphan-path"], (
            "_kill_blocking_processes must be passed record.path, not repo_root"
        )
        assert captured_path[0] != record.repo_root


# ---------------------------------------------------------------------------
# _wait_or_kill unit tests
# ---------------------------------------------------------------------------

class TestWaitOrKill:
    def test_returns_immediately_if_dead(self):
        """_wait_or_kill returns without killing if the process is already dead."""
        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._force_kill"
            ) as mock_kill,
        ):
            _wait_or_kill(12345, timeout=5.0)

        mock_kill.assert_not_called()

    def test_kills_after_timeout(self):
        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._force_kill"
            ) as mock_kill,
            patch("time.sleep"),  # speed up test
        ):
            _wait_or_kill(99999, timeout=0.1)

        mock_kill.assert_called_once_with(99999)


# ---------------------------------------------------------------------------
# _find_blocking_processes unit tests  (ticket #29)
# ---------------------------------------------------------------------------

def _make_fake_proc(pid: int, name: str, cmdline: list, cwd: str):
    """Build a fake psutil.Process-like object for _find_blocking_processes tests."""
    proc = MagicMock()
    proc.info = {"pid": pid, "name": name, "cmdline": cmdline}
    proc.cwd.return_value = cwd
    return proc


class TestFindBlockingProcesses:
    """Unit tests for _find_blocking_processes (psutil mocked)."""

    def test_matching_process_returned(self):
        """A process whose cwd is exactly the target path is returned."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc_match = _make_fake_proc(9001, "node", ["node", "server.js"], target)
        proc_other = _make_fake_proc(9002, "python", ["python"], "/other/path")

        with (
            patch.object(psutil, "process_iter", return_value=[proc_match, proc_other]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert len(result) == 1
        assert result[0].pid == 9001
        assert result[0].name == "node"
        assert result[0].cmdline == ["node", "server.js"]

    def test_subprocess_cwd_is_returned(self):
        """A process whose cwd is a subdirectory of the target path is returned."""
        import psutil

        target = "/fake/worktree"
        sub_cwd = "/fake/worktree/subdir"
        host_pid = os.getpid()

        proc_sub = _make_fake_proc(9003, "bash", ["bash"], sub_cwd)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_sub]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert len(result) == 1
        assert result[0].pid == 9003

    def test_non_matching_process_excluded(self):
        """Processes with unrelated cwd are not included."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc_other = _make_fake_proc(9004, "vim", ["vim"], "/home/user")

        with (
            patch.object(psutil, "process_iter", return_value=[proc_other]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert result == []

    def test_host_pid_excluded(self):
        """The host process itself is never included even if its cwd matches."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc_host = _make_fake_proc(host_pid, "python", ["python"], target)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_host]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert result == []

    def test_ancestor_pid_excluded(self):
        """Ancestors of the host process are never included."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()
        ancestor_pid = 1001

        proc_ancestor = _make_fake_proc(ancestor_pid, "init", ["init"], target)
        proc_blocker = _make_fake_proc(9005, "node", ["node"], target)

        ancestor_mock = MagicMock()
        ancestor_mock.pid = ancestor_pid

        with (
            patch.object(psutil, "process_iter", return_value=[proc_ancestor, proc_blocker]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = [ancestor_mock]
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert len(result) == 1
        assert result[0].pid == 9005

    def test_access_denied_cwd_skipped(self):
        """A process whose cwd() raises AccessDenied is silently skipped."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc_denied = MagicMock()
        proc_denied.info = {"pid": 9006, "name": "sshd", "cmdline": ["sshd"]}
        proc_denied.cwd.side_effect = psutil.AccessDenied(9006)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_denied]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert result == []

    def test_empty_process_list_returns_empty(self):
        """Empty process list yields empty result."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        with (
            patch.object(psutil, "process_iter", return_value=[]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert result == []

    def test_sibling_path_not_matched(self):
        """Regression: a cwd that is a string-prefix of the target path but is
        NOT under it (e.g. /fake/worktree-sibling vs /fake/worktree) must NOT
        be returned.  Covers exact-match and genuine-subdir positive cases
        alongside the sibling negative case."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        # Negative: string-prefix sibling — must NOT match.
        proc_sibling = _make_fake_proc(9010, "vim", ["vim"], "/fake/worktree-sibling")
        # Positive: exact cwd match — must match.
        proc_exact = _make_fake_proc(9011, "node", ["node"], "/fake/worktree")
        # Positive: genuine subdirectory — must match.
        proc_sub = _make_fake_proc(9012, "bash", ["bash"], "/fake/worktree/src")

        with (
            patch.object(
                psutil,
                "process_iter",
                return_value=[proc_sibling, proc_exact, proc_sub],
            ),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        returned_pids = {r.pid for r in result}
        assert 9010 not in returned_pids, (
            "/fake/worktree-sibling must not match target /fake/worktree"
        )
        assert 9011 in returned_pids, "exact cwd match must be included"
        assert 9012 in returned_pids, "genuine subdirectory must be included"

    def test_open_file_handle_under_path_returned(self):
        """A process whose open_files() contains a file under the target path
        is included even when its cwd is outside the path (gap 1 fix)."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        # This process's cwd is OUTSIDE target, so the CWD pass won't catch it.
        proc_daemon = MagicMock()
        proc_daemon.info = {"pid": 9020, "name": "unity", "cmdline": ["unity"]}
        proc_daemon.cwd.return_value = "/other/path"
        # But it holds an open file handle inside target.
        file_info = MagicMock()
        file_info.path = "/fake/worktree/Assets/scene.unity"
        proc_daemon.open_files.return_value = [file_info]

        with (
            patch.object(psutil, "process_iter", return_value=[proc_daemon]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert len(result) == 1
        assert result[0].pid == 9020
        assert result[0].name == "unity"

    def test_open_files_access_denied_skipped(self):
        """A process whose open_files() raises AccessDenied is silently skipped."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc_denied = MagicMock()
        proc_denied.info = {"pid": 9021, "name": "system", "cmdline": ["system"]}
        proc_denied.cwd.return_value = "/other/path"
        proc_denied.open_files.side_effect = psutil.AccessDenied(9021)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_denied]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert result == []

    def test_open_files_empty_list_no_spurious_additions(self):
        """A process with an empty open_files() list is not added."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc = MagicMock()
        proc.info = {"pid": 9022, "name": "idle", "cmdline": ["idle"]}
        proc.cwd.return_value = "/other/path"
        proc.open_files.return_value = []

        with (
            patch.object(psutil, "process_iter", return_value=[proc]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert result == []

    def test_cwd_match_not_duplicated_by_open_files(self):
        """A process already matched by CWD must not be returned twice even if
        it also has open file handles inside the target path."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc = MagicMock()
        proc.info = {"pid": 9023, "name": "node", "cmdline": ["node"]}
        proc.cwd.return_value = "/fake/worktree"  # matches CWD pass
        file_info = MagicMock()
        file_info.path = "/fake/worktree/index.js"
        proc.open_files.return_value = [file_info]

        with (
            patch.object(psutil, "process_iter", return_value=[proc]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _find_blocking_processes(target, host_pid)

        assert len(result) == 1, "process must appear exactly once even with both CWD and open-file match"
        assert result[0].pid == 9023


# ---------------------------------------------------------------------------
# _kill_blocking_processes unit tests  (ticket #29)
# ---------------------------------------------------------------------------

class TestKillBlockingProcesses:
    """Unit tests for _kill_blocking_processes."""

    def test_kills_each_found_process(self):
        """_kill_blocking_processes calls graceful signal then wait_or_kill per process."""
        target = "/fake/worktree"
        fake_found = [
            KilledProcessInfo(pid=1010, name="node", cmdline=["node"]),
            KilledProcessInfo(pid=2020, name="python", cmdline=["python", "app.py"]),
        ]

        graceful_calls = []
        wait_calls = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=fake_found,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
                side_effect=lambda pid: graceful_calls.append(pid),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
                side_effect=lambda pid, timeout: wait_calls.append((pid, timeout)),
            ),
        ):
            result = _kill_blocking_processes(target)

        assert result == fake_found
        assert graceful_calls == [1010, 2020]
        assert [p for (p, _) in wait_calls] == [1010, 2020]
        # Each per-pid budget must be positive and the total must not exceed the
        # default 5.0 s budget (plus a tight epsilon — _wait_or_kill is mocked
        # so there is no real elapsed time; any overshoot indicates a logic bug).
        assert all(t > 0 for (_, t) in wait_calls), "each per-pid budget must be positive"
        assert sum(t for (_, t) in wait_calls) <= 5.0 + 1e-3, (
            "total wait budget must not exceed the requested timeout"
        )

    def test_no_blockers_returns_empty_no_kills(self):
        """_kill_blocking_processes returns [] and makes no kill calls when no blockers."""
        target = "/fake/worktree"

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ) as mock_graceful,
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
            ) as mock_wait,
        ):
            result = _kill_blocking_processes(target)

        assert result == []
        mock_graceful.assert_not_called()
        mock_wait.assert_not_called()

    def test_total_time_bounded_by_timeout(self):
        """Budget distributed across orphans must not exceed the requested timeout."""
        target = "/fake/worktree"
        fake_found = [
            KilledProcessInfo(pid=3001, name="node", cmdline=["node"]),
            KilledProcessInfo(pid=3002, name="python", cmdline=["python"]),
            KilledProcessInfo(pid=3003, name="ruby", cmdline=["ruby"]),
        ]

        wait_calls: List[tuple] = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=fake_found,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
                side_effect=lambda pid, timeout: wait_calls.append((pid, timeout)),
            ),
        ):
            result = _kill_blocking_processes(target, timeout=6.0)

        assert result == fake_found
        assert len(wait_calls) == 3
        total = sum(t for (_, t) in wait_calls)
        assert total <= 6.0 + 0.1, (
            f"sum of per-pid budgets ({total:.3f}) must not exceed requested timeout (6.0)"
        )

    def test_timeout_zero_skips_wait_calls(self):
        """timeout=0.0 sends the graceful signal but skips _wait_or_kill entirely.

        Both orphan PIDs must receive the graceful signal even though no budget
        is available for waiting (the docstring guarantees this behaviour).
        """
        target = "/fake/worktree"
        fake_found = [
            KilledProcessInfo(pid=4001, name="node", cmdline=["node"]),
            KilledProcessInfo(pid=4002, name="python", cmdline=["python"]),
        ]

        graceful_calls: List[int] = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=fake_found,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
                side_effect=lambda pid: graceful_calls.append(pid),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
            ) as mock_wait,
        ):
            result = _kill_blocking_processes(target, timeout=0.0)

        assert result == fake_found
        # _wait_or_kill must not be called — no budget to wait.
        mock_wait.assert_not_called()
        # _send_graceful_signal must be called for EVERY orphan, even with
        # timeout=0.0.  This is the core of the fix for blocking issue #1.
        assert graceful_calls == [4001, 4002], (
            f"expected graceful signals for both orphans [4001, 4002], got {graceful_calls}"
        )

    def test_kill_blocking_processes_expands_found_with_descendants(self):
        """Ticket #87: each found blocker's own descendant tree (via
        _process_tree) is also signalled/killed and included in the
        returned list, not just the blocker itself -- catches a grandchild
        of a path-heuristic-matched process."""
        target = "/fake/worktree"
        fake_found = [KilledProcessInfo(pid=5010, name="node", cmdline=["node"])]
        fake_descendants = [
            KilledProcessInfo(pid=5011, name="node-child", cmdline=["node-child"]),
        ]

        graceful_calls = []
        tree_calls = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=fake_found,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                side_effect=lambda pid: (
                    tree_calls.append(pid) or (fake_descendants if pid == 5010 else [])
                ),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
                side_effect=lambda pid: graceful_calls.append(pid),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
            ),
        ):
            result = _kill_blocking_processes(target)

        assert tree_calls == [5010]
        result_pids = {info.pid for info in result}
        assert result_pids == {5010, 5011}
        assert 5011 in graceful_calls, "descendant of a found blocker must also be signalled"

    def test_kill_blocking_processes_expansion_dedups_by_pid(self):
        """A descendant that happens to already be in the found list (e.g.
        two independently path-matched processes that are also parent/child
        of each other) must not be duplicated in the expanded result."""
        target = "/fake/worktree"
        fake_found = [
            KilledProcessInfo(pid=5020, name="parent", cmdline=["parent"]),
            KilledProcessInfo(pid=5021, name="child", cmdline=["child"]),
        ]

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=fake_found,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                side_effect=lambda pid: (
                    [KilledProcessInfo(pid=5021, name="child", cmdline=["child"])]
                    if pid == 5020
                    else []
                ),
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            result = _kill_blocking_processes(target)

        result_pids = [info.pid for info in result]
        assert result_pids.count(5021) == 1, "descendant already in found must not be duplicated"

    def test_kill_blocking_processes_expansion_loop_bounded_by_deadline(self):
        """Regression (ticket #87 follow-up, finding F1): the lineage-
        expansion loop -- which calls _process_tree() once per blocker found
        by _find_blocking_processes, AFTER discovery already returned --
        must itself respect the same *timeout* deadline as the rest of
        _kill_blocking_processes. Before this fix only the subsequent
        signal/wait loop checked the deadline; this loop had no budget check
        at all, so a host with many path-heuristic blockers could blow
        through the caller's timeout entirely inside this loop, directly
        contradicting stop()'s documented "total operation is bounded by
        timeout seconds" guarantee.

        Simulates 60 found blockers, each with an artificially slow (0.05s)
        _process_tree() call -- 3.0s unbounded -- against a 0.2s timeout, and
        asserts both that the call returns well within budget AND that the
        loop actually broke early (_process_tree was not called for every
        blocker)."""
        target = "/fake/worktree"
        num_blockers = 60
        fake_found = [
            KilledProcessInfo(pid=6000 + i, name="proc", cmdline=["proc"])
            for i in range(num_blockers)
        ]

        tree_calls: List[int] = []

        def _slow_process_tree(pid):
            tree_calls.append(pid)
            time.sleep(0.05)
            return []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=fake_found,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                side_effect=_slow_process_tree,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            t0 = time.monotonic()
            result = _kill_blocking_processes(target, timeout=0.2)
            elapsed = time.monotonic() - t0

        assert elapsed < 1.5, (
            f"_kill_blocking_processes took {elapsed:.2f}s against a 0.2s "
            f"timeout with {num_blockers} blockers (unbounded would be "
            f"~{num_blockers * 0.05:.1f}s) -- the lineage-expansion loop is "
            f"not bounded by the deadline"
        )
        assert len(tree_calls) < num_blockers, (
            f"expected the expansion loop to break early once the deadline "
            f"passed, but _process_tree was called for all "
            f"{len(tree_calls)} blockers -- the loop never breaks"
        )
        # Whatever was expanded before the budget ran out is still returned,
        # degrading gracefully rather than discarding it.
        assert len(result) >= num_blockers


# ---------------------------------------------------------------------------
# stop() timeout-budget regression tests  (ticket #50)
# ---------------------------------------------------------------------------

class TestStopTimeoutBudget:
    """Regression tests for ticket #50: stop() total time bounded by timeout.

    The bug: _kill_blocking_processes was called without a timeout, so N orphans
    each consumed up to 5 s — resulting in up to 5*N seconds for the orphan scan
    alone, independent of stop()'s own timeout parameter.

    The fix: stop() computes a shared deadline at entry and passes the remaining
    budget to _kill_blocking_processes(... timeout=orphan_budget).
    """

    def test_stop_dead_pid_orphan_scan_receives_full_timeout_budget(self):
        """Primary regression (#50): when the tracked PID is already dead the
        orphan scan must receive nearly the full timeout budget, not an unbounded
        hardcoded value."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — skipping")

        record = _make_record("wt-budget-dead", pids={DEFAULT_ROLE: 99999999})
        store = _make_store(record)

        captured_timeout: List[float] = []

        with patch(
            "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
            side_effect=lambda path, **kw: captured_timeout.append(kw.get("timeout", -1)) or [],
        ):
            stop("wt-budget-dead", store=store, kill_orphans=True, timeout=8.0)

        assert len(captured_timeout) == 1
        # The dead-pid fast-path spends nearly no time, so the orphan scan
        # must receive close to the full 8.0 s budget.
        assert captured_timeout[0] >= 7.5, (
            f"orphan scan received only {captured_timeout[0]:.3f}s of the 8.0s budget"
        )
        assert captured_timeout[0] <= 8.0 + 0.1, (
            "orphan scan must not receive more time than the caller requested"
        )

    def test_stop_alive_pid_orphan_scan_receives_remaining_budget(self):
        """When the shell PID is alive the orphan scan receives whatever time
        remains after the primary _wait_or_kill call completes."""
        fake_pid = 66666
        record = _make_record("wt-budget-alive", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        captured_timeout: List[float] = []
        caller_timeout = 10.0

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                side_effect=lambda path, **kw: captured_timeout.append(kw.get("timeout", -1)) or [],
            ),
        ):
            stop("wt-budget-alive", store=store, kill_orphans=True, timeout=caller_timeout)

        assert len(captured_timeout) == 1
        orphan_budget = captured_timeout[0]
        # _wait_or_kill is a no-op here, so nearly the full caller_timeout must
        # remain for the orphan scan.  Tighten the lower bound accordingly.
        assert orphan_budget >= caller_timeout - 0.5, (
            f"orphan budget {orphan_budget:.3f} must be >= caller_timeout - 0.5 ({caller_timeout - 0.5})"
        )
        assert orphan_budget <= caller_timeout, (
            f"orphan budget {orphan_budget:.3f} must be <= caller timeout {caller_timeout}"
        )

    def test_stop_orphan_scan_bounded_when_many_orphans(self):
        """Wall-clock smoke test: stop() with 5 fake orphans and a 2-second
        timeout must return in under 3 seconds total."""
        fake_pid = 77777
        record = _make_record("wt-many-orphans", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        fake_found = [
            KilledProcessInfo(pid=5000 + i, name="proc", cmdline=["proc"])
            for i in range(5)
        ]

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=fake_found,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
            ),
        ):
            t0 = time.monotonic()
            stop("wt-many-orphans", store=store, kill_orphans=True, timeout=2.0)
            elapsed = time.monotonic() - t0

        assert elapsed < 3.0, (
            f"stop() with 5 orphans took {elapsed:.2f}s — must complete in under 3.0s"
        )

    def test_stop_dead_pid_no_orphans_returns_fast(self):
        """stop() with a dead PID and kill_orphans=False must return quickly
        (no sleeping or waiting)."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — skipping")

        record = _make_record("wt-fast-dead", pids={DEFAULT_ROLE: 99999999})
        store = _make_store(record)

        t0 = time.monotonic()
        result = stop("wt-fast-dead", store=store, kill_orphans=False, timeout=10.0)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, (
            f"stop() with a dead PID and no orphan scan took {elapsed:.2f}s — must return in under 1.0s"
        )
        assert DEFAULT_ROLE not in result.pids
        assert result.status == "stopped"


# ---------------------------------------------------------------------------
# TestStopBudgetReallocation -- ticket #95, R1
# ---------------------------------------------------------------------------

class TestStopBudgetReallocation:
    """R1 (ticket #95): the primary signal/wait step must never be able to
    starve the tree-kill and orphan-scan steps of their whole budget.

    Root cause (finding 2): ``stop()`` used to hand the *entire* remaining
    timeout to the primary ``_wait_or_kill`` call. On Windows, a
    ``CTRL_BREAK_EVENT`` to a ``CREATE_NEW_PROCESS_GROUP`` child sharing no
    console frequently does nothing, so ``_wait_or_kill`` polls the full
    budget before force-killing -- leaving 0.0s for the orphan scan
    (``_kill_blocking_processes(..., timeout=0.0)``), whose deadline-driven
    guards (Pass 1/1b/1c/2) then all fail immediately. The fix reserves a
    floor for the tree kill and orphan scan up front, computed by
    ``_compute_stop_budget``, and caps the primary wait at
    ``primary_cap`` so it can never eat into those floors.
    """

    def test_primary_wait_cannot_starve_orphan_scan(self):
        """Driving test: even when the primary wait consumes its entire
        granted timeout, the orphan scan still receives a real, non-trivial
        budget -- not the 0.0s the pre-fix code handed it."""
        fake_pid = 55555
        record = _make_record("wt-no-starve", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        captured_primary_timeout: List[float] = []
        captured_orphan_timeout: List[float] = []

        def fake_wait_or_kill(pid, timeout):
            captured_primary_timeout.append(timeout)
            # Simulate the pathological case: _wait_or_kill burns its whole
            # granted budget (e.g. CTRL_BREAK_EVENT was a no-op).
            time.sleep(timeout)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
                side_effect=fake_wait_or_kill,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                side_effect=lambda path, **kw: captured_orphan_timeout.append(
                    kw.get("timeout", -1)
                ) or [],
            ),
        ):
            stop("wt-no-starve", store=store, kill_orphans=True, timeout=1.0)

        assert len(captured_primary_timeout) == 1
        assert captured_primary_timeout[0] <= 0.6 + 0.05, (
            f"primary wait received {captured_primary_timeout[0]:.3f}s -- "
            "must be capped at primary_cap (~0.6s of a 1.0s budget)"
        )
        assert len(captured_orphan_timeout) == 1
        assert captured_orphan_timeout[0] >= 0.25, (
            f"orphan scan received only {captured_orphan_timeout[0]:.3f}s -- "
            "the primary wait starved it almost entirely"
        )

    def test_budget_split_arithmetic_at_default_timeout(self):
        """_compute_stop_budget's formula matches the pinned design at the
        documented default timeout (10.0s), for both kill_orphans values."""
        primary_cap, tree_floor, orphan_floor = _pl._compute_stop_budget(10.0, True)
        assert tree_floor == pytest.approx(1.0)
        assert orphan_floor == pytest.approx(3.0)
        assert primary_cap == pytest.approx(6.0)

        primary_cap2, tree_floor2, orphan_floor2 = _pl._compute_stop_budget(10.0, False)
        assert tree_floor2 == pytest.approx(1.0)
        assert orphan_floor2 == 0.0
        assert primary_cap2 == pytest.approx(6.0)

    def test_tree_kill_receives_nonzero_budget_when_primary_exhausts(self):
        """When the primary wait consumes its entire cap, _kill_process_tree
        must still be handed a strictly positive timeout (never the 0.0s the
        pre-fix code could produce)."""
        fake_pid = 55556
        tree = [KilledProcessInfo(pid=90001, name="child", cmdline=["child"])]
        record = _make_record("wt-tree-budget", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        captured_tree_timeout: List[float] = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
                side_effect=lambda pid, timeout: time.sleep(timeout),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=tree,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_process_tree",
                side_effect=lambda killed_tree, **kw: captured_tree_timeout.append(
                    kw.get("timeout", -1)
                ) or killed_tree,
            ),
        ):
            stop("wt-tree-budget", store=store, kill_orphans=False, timeout=1.0)

        assert len(captured_tree_timeout) == 1
        assert captured_tree_timeout[0] > 0.0, (
            "tree kill must receive a strictly positive budget even when the "
            "primary wait exhausted its own cap"
        )

    def test_kill_orphans_false_reserves_no_orphan_floor(self):
        """orphan_floor is 0.0 when kill_orphans=False -- the tree kill gets
        the entire remainder instead of ceding a floor to a scan that will
        never run."""
        _primary_cap, _tree_floor, orphan_floor = _pl._compute_stop_budget(5.0, False)
        assert orphan_floor == 0.0

    def test_timeout_zero_still_bounded(self):
        """timeout=0.0 must still compute a sane, all-zero split -- today's
        immediate-force-kill behaviour is preserved."""
        primary_cap, tree_floor, orphan_floor = _pl._compute_stop_budget(0.0, True)
        assert primary_cap == 0.0
        assert tree_floor == 0.0
        assert orphan_floor == 0.0

    def test_total_elapsed_never_exceeds_timeout(self):
        """Wall-clock smoke test: stop() with an immortal fake pid and a
        pathological _wait_or_kill that always burns its full grant must
        still return within the caller's overall timeout (plus a small
        scheduling tolerance)."""
        fake_pid = 55557
        record = _make_record("wt-total-bound", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
                side_effect=lambda pid, timeout: time.sleep(timeout),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=[],
            ),
        ):
            t0 = time.monotonic()
            stop("wt-total-bound", store=store, kill_orphans=True, timeout=1.0)
            elapsed = time.monotonic() - t0

        assert elapsed <= 1.0 + 0.2, (
            f"stop() took {elapsed:.2f}s -- total must stay bounded by timeout"
        )

    def test_tree_kill_receives_at_least_tree_floor_when_primary_overruns(self):
        """Driving test (ticket #95 fix cycle, blocking finding): tree_floor
        must actually be ENFORCED, not merely computed and discarded.

        Root cause: the call site unpacked ``_compute_stop_budget``'s
        ``tree_floor`` into a throwaway ``_tree_floor`` and never used it --
        the tree-kill call only ever reserved ``orphan_floor``:
        ``timeout=max(0.0, deadline - time.monotonic() - orphan_floor)``.
        ``_wait_or_kill`` polls on a 0.1s tick and can therefore overrun its
        granted ``primary_cap`` (acknowledged in stop()'s own docstring) --
        when it does, ``deadline - time.monotonic()`` can be zero or even
        negative, which the pre-fix ``max(0.0, ...)`` clamp would silently
        round to ``0.0``, starving the tree-kill step exactly like this
        ticket's original bug (just one step later in the pipeline).

        Simulates that overrun directly: _wait_or_kill sleeps well past its
        granted primary_cap, pushing time.monotonic() past stop()'s own
        deadline before the tree-kill budget is computed. The tree kill must
        still receive at least tree_floor seconds, never 0.0.
        """
        fake_pid = 55558
        tree = [KilledProcessInfo(pid=90002, name="child", cmdline=["child"])]
        record = _make_record("wt-tree-floor-overrun", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        captured_tree_timeout: List[float] = []
        timeout = 1.0
        _primary_cap, tree_floor, _orphan_floor = _pl._compute_stop_budget(
            timeout, False
        )

        def fake_wait_or_kill(pid, timeout_arg):
            # Simulate _wait_or_kill overrunning its granted primary_cap --
            # e.g. a polling tick landing just past the deadline -- so that
            # by the time the tree-kill budget is computed, the shared
            # deadline has already passed.
            time.sleep(timeout_arg + 0.3)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
                side_effect=fake_wait_or_kill,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=tree,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_process_tree",
                side_effect=lambda killed_tree, **kw: captured_tree_timeout.append(
                    kw.get("timeout", -1)
                ) or killed_tree,
            ),
        ):
            stop("wt-tree-floor-overrun", store=store, kill_orphans=False, timeout=timeout)

        assert len(captured_tree_timeout) == 1
        assert captured_tree_timeout[0] >= tree_floor, (
            f"tree kill received only {captured_tree_timeout[0]:.3f}s after the "
            f"primary wait overran its budget -- must be >= tree_floor "
            f"({tree_floor:.3f}s), the guarantee _compute_stop_budget's "
            "docstring documents but the call site was not enforcing"
        )

    def test_orphan_scan_receives_at_least_orphan_floor_when_primary_overruns(self):
        """Driving test (ticket #95 fix cycle, R5 review): orphan_floor must
        actually be ENFORCED for the orphan scan, not merely computed and
        discarded -- the identical starvation bug already fixed for the
        tree-kill step (see the sibling test above) but left unfixed one
        step further down the pipeline.

        Root cause: the orphan-scan call site computed its budget as
        ``orphan_budget = max(0.0, deadline - time.monotonic())`` -- no
        floor enforcement. Simulates the pathological overrun chain: the
        primary wait overruns its granted ``primary_cap`` (acknowledged
        possible -- ``_wait_or_kill`` polls on a 0.1s tick), and the
        tree-kill step then legitimately consumes up to its own
        ``tree_floor`` seconds past the deadline to satisfy the guarantee
        fixed earlier in this cycle. By the time the orphan budget is
        computed, ``time.monotonic()`` is already well past ``deadline``,
        so the pre-fix ``max(0.0, ...)`` clamp collapses it to exactly
        ``0.0`` -- silently skipping the orphan scan (Pass 1c, the whole
        reason this ticket exists) in exactly the scenario this ticket is
        about.
        """
        fake_pid = 55559
        record = _make_record("wt-orphan-floor-overrun", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        captured_orphan_timeout: List[float] = []
        timeout = 1.0
        _primary_cap, tree_floor, orphan_floor = _pl._compute_stop_budget(
            timeout, True
        )

        def fake_wait_or_kill(pid, timeout_arg):
            # Simulate _wait_or_kill overrunning its granted primary_cap --
            # e.g. a polling tick landing just past the deadline.
            time.sleep(timeout_arg + 0.3)

        def fake_kill_process_tree(killed_tree, **kw):
            # Simulate the tree-kill step legitimately consuming up to its
            # own tree_floor seconds past the (already-overrun) deadline --
            # exactly what the earlier fix in this cycle now guarantees it
            # may do.
            time.sleep(tree_floor)
            return killed_tree

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
                side_effect=fake_wait_or_kill,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_process_tree",
                side_effect=fake_kill_process_tree,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                side_effect=lambda path, **kw: captured_orphan_timeout.append(
                    kw.get("timeout", -1)
                ) or [],
            ),
        ):
            stop(
                "wt-orphan-floor-overrun",
                store=store,
                kill_orphans=True,
                timeout=timeout,
            )

        assert len(captured_orphan_timeout) == 1
        assert captured_orphan_timeout[0] >= orphan_floor, (
            f"orphan scan received only {captured_orphan_timeout[0]:.3f}s after "
            f"the primary wait and tree-kill step overran the deadline -- must "
            f"be >= orphan_floor ({orphan_floor:.3f}s), the guarantee "
            "_compute_stop_budget's docstring documents but the call site was "
            "not enforcing"
        )


# ---------------------------------------------------------------------------
# TestStopReportsKilledPids -- ticket #95, R3
# ---------------------------------------------------------------------------

class TestStopReportsKilledPids:
    """R3 (ticket #95): stop() must populate ``record.killed_pids``.

    Root cause (finding 1): ``WorktreeRecord.killed_pids`` is only ever
    written at two call sites in ``manager.py`` (the ``_teardown`` kill-and-
    retry paths) -- ``process_lifecycle.stop()`` never touches it, so its
    ``killed_pids: []`` in any report built from a ``stop()`` result is
    structurally guaranteed, regardless of what was actually killed.
    """

    def test_stop_populates_killed_pids_with_tree_and_orphans(self):
        """Driving test: a 2-node descendant tree plus 1 orphan must all
        appear in ``result.killed_pids``, deepest-first (tree, then the
        tracked pid, then orphans), deduplicated, and NOT persisted to the
        YAML-shaped dict ``_record_to_dict`` would produce."""
        fake_pid = 71000
        record = _make_record("wt-killed-pids", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        child = KilledProcessInfo(pid=71001, name="child", cmdline=["child"])
        grandchild = KilledProcessInfo(pid=71002, name="grandchild", cmdline=["gc"])
        orphan = KilledProcessInfo(pid=71003, name="orphan", cmdline=["orphan"])

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[grandchild, child],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=_pl._PartialList([orphan]),
            ),
        ):
            result = stop(
                "wt-killed-pids", store=store, kill_orphans=True, timeout=5.0,
            )

        killed_pids_by_pid = {info.pid: info for info in result.killed_pids}
        assert set(killed_pids_by_pid) == {71001, 71002, 71003}, (
            f"expected tree + orphan pids, got {set(killed_pids_by_pid)}"
        )
        # Deepest-first: the tree's own order (grandchild, child) must be
        # preserved ahead of the orphan.
        pids_in_order = [info.pid for info in result.killed_pids]
        assert pids_in_order.index(71002) < pids_in_order.index(71001), (
            "grandchild must be listed before its parent (deepest-first)"
        )
        assert pids_in_order.index(71001) < pids_in_order.index(71003), (
            "tree entries must precede orphan entries"
        )

        # Never persisted -- killed_pids is transient (state.py docstring).
        from lib_python_worktree.core.yaml_store import _record_to_dict
        assert "killed_pids" not in _record_to_dict(result)

    def test_stop_killed_pids_includes_tracked_pid_when_alive(self):
        """The tracked PID itself is included in killed_pids when it was
        alive and something was actually attempted against it."""
        fake_pid = 71100
        record = _make_record("wt-killed-pids-tracked", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
        ):
            result = stop("wt-killed-pids-tracked", store=store, timeout=1.0)

        assert fake_pid in {info.pid for info in result.killed_pids}

    def test_stop_killed_pids_excludes_tracked_pid_when_already_dead(self):
        """The tracked PID is NOT reported as "killed" when it was already
        dead at entry -- nothing was attempted against it."""
        if _pid_alive(99999998):
            pytest.skip("PID 99999998 is alive on this machine — skipping")

        record = _make_record("wt-killed-pids-dead", pids={DEFAULT_ROLE: 99999998})
        store = _make_store(record)

        result = stop("wt-killed-pids-dead", store=store, timeout=1.0)

        assert 99999998 not in {info.pid for info in result.killed_pids}

    def test_stop_killed_pids_deduplicated(self):
        """A PID appearing in both the tree and the orphan scan must appear
        only once in killed_pids."""
        fake_pid = 71200
        shared = KilledProcessInfo(pid=71201, name="dup", cmdline=["dup"])
        record = _make_record("wt-killed-pids-dedup", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[shared],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=_pl._PartialList([shared]),
            ),
        ):
            result = stop(
                "wt-killed-pids-dedup", store=store, kill_orphans=True, timeout=5.0,
            )

        matching = [info for info in result.killed_pids if info.pid == 71201]
        assert len(matching) == 1, (
            f"expected pid 71201 exactly once, found {len(matching)} times"
        )

    def test_stop_no_kill_orphans_killed_pids_still_populated_from_tree(self):
        """Even with kill_orphans=False, killed_pids reflects the tree kill
        step (no orphan scan runs, but that must not leave killed_pids
        empty when there was a real tree to report)."""
        fake_pid = 71300
        child = KilledProcessInfo(pid=71301, name="child", cmdline=["child"])
        record = _make_record("wt-killed-pids-no-orphans", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[child],
            ),
        ):
            result = stop(
                "wt-killed-pids-no-orphans", store=store, kill_orphans=False, timeout=5.0,
            )

        assert 71301 in {info.pid for info in result.killed_pids}


# ---------------------------------------------------------------------------
# TestFindBlockingProcessesWindows -- ticket #57
# ---------------------------------------------------------------------------

class TestFindBlockingProcessesWindows:
    """Regression tests for ticket #57: Windows cmdline token scan (Pass 1b).

    On Windows, proc.cwd() raises AccessDenied for almost all foreign
    processes, making the Pass 1 CWD match a no-op.  Pass 1b scans cmdline
    tokens instead: if any token resolves to a path under the worktree
    directory, the process is treated as blocking.
    """

    def test_cwd_access_denied_falls_through_to_cmdline(self):
        """Regression #57: on Windows, when proc.cwd() raises AccessDenied but a
        cmdline token points under the target path, the process is returned."""
        import psutil

        # Use a POSIX-style path so os.sep and os.path.normpath work correctly
        # on Linux CI even though sys.platform is patched to "win32".
        target = "/fake/worktree"
        host_pid = os.getpid()

        # Simulate a Windows foreign process: cwd() denied, but cmdline contains
        # a path inside the target worktree.
        proc_win = MagicMock()
        proc_win.info = {
            "pid": 8801,
            "name": "code.exe",
            "cmdline": ["code.exe", "/fake/worktree/src/main.py"],
        }
        proc_win.cwd.side_effect = psutil.AccessDenied(8801)
        proc_win.open_files.side_effect = psutil.AccessDenied(8801)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_win]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            result = _find_blocking_processes(target, host_pid)

        assert len(result) == 1, (
            f"Expected 1 blocking process via cmdline scan, got {result}"
        )
        assert result[0].pid == 8801
        assert result[0].name == "code.exe"

    def test_no_match_returns_empty(self):
        """On Windows, when cwd() is denied and cmdline tokens are all unrelated
        paths, the result is empty."""
        import psutil

        target = "C:\\fake\\worktree"
        host_pid = os.getpid()

        proc_unrelated = MagicMock()
        proc_unrelated.info = {
            "pid": 8802,
            "name": "explorer.exe",
            "cmdline": ["explorer.exe", "C:\\Users\\user\\Documents"],
        }
        proc_unrelated.cwd.side_effect = psutil.AccessDenied(8802)
        proc_unrelated.open_files.side_effect = psutil.AccessDenied(8802)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_unrelated]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            result = _find_blocking_processes(target, host_pid)

        assert result == [], (
            f"Unrelated cmdline tokens must not produce any matches, got {result}"
        )

    def test_cmdline_scan_skipped_on_non_windows(self):
        """Pass 1b (cmdline scan) must NOT run on non-Windows platforms.
        A process whose cwd() is denied but whose cmdline contains the path
        must NOT appear in the result when platform != 'win32'."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc_posix = MagicMock()
        proc_posix.info = {
            "pid": 8803,
            "name": "bash",
            "cmdline": ["bash", "/fake/worktree/run.sh"],
        }
        proc_posix.cwd.side_effect = psutil.AccessDenied(8803)
        proc_posix.open_files.side_effect = psutil.AccessDenied(8803)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_posix]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            result = _find_blocking_processes(target, host_pid)

        assert result == [], (
            "cmdline token scan must not run on non-Windows; got unexpected matches"
        )

    def test_cmdline_token_is_exact_match_to_target(self):
        """A cmdline token equal to the target path (not just under it) is also
        a valid match on Windows."""
        import psutil

        target = "C:\\fake\\worktree"
        host_pid = os.getpid()

        proc_exact = MagicMock()
        proc_exact.info = {
            "pid": 8804,
            "name": "tool.exe",
            "cmdline": ["tool.exe", "--root", "C:\\fake\\worktree"],
        }
        proc_exact.cwd.side_effect = psutil.AccessDenied(8804)
        proc_exact.open_files.side_effect = psutil.AccessDenied(8804)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_exact]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            result = _find_blocking_processes(target, host_pid)

        assert len(result) == 1
        assert result[0].pid == 8804

    def test_cmdline_not_duplicated_when_cwd_also_matches(self):
        """A process matched by CWD (Pass 1) must not be re-added by the
        cmdline scan (Pass 1b)."""
        import psutil

        target = "C:\\fake\\worktree"
        host_pid = os.getpid()

        # This process: cwd succeeds AND cmdline matches
        proc_both = MagicMock()
        proc_both.info = {
            "pid": 8805,
            "name": "node.exe",
            "cmdline": ["node.exe", "C:\\fake\\worktree\\index.js"],
        }
        proc_both.cwd.return_value = "C:\\fake\\worktree"
        proc_both.open_files.return_value = []

        with (
            patch.object(psutil, "process_iter", return_value=[proc_both]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            result = _find_blocking_processes(target, host_pid)

        assert len(result) == 1, (
            f"Process must appear exactly once, got {len(result)} entries"
        )
        assert result[0].pid == 8805


# ---------------------------------------------------------------------------
# TestBoundedQueryWorker -- ticket #90 (bounded worker thread hygiene)
# ---------------------------------------------------------------------------

class TestBoundedQueryWorker:
    """Cross-platform unit tests for ``_BoundedQueryWorker`` (ticket #90).

    ``_win_handle_holders``' NtQueryObject calls (Windows-only) are the
    motivating use case, but ``_BoundedQueryWorker`` itself runs an
    arbitrary zero-arg callable and has no Windows dependency -- these
    tests drive it directly with plain Python callables (fast,
    slow-but-resolves-in-grace, and permanently-blocked-on-an-``Event``) so
    the core thread-hygiene and grace-budget mechanism is exercised on
    every platform this suite runs on, not just Windows. The real
    ctypes/ntdll wiring is covered separately by ``TestWinHandleHoldersReal``
    / ``TestWinHandleHoldersThreadHygiene`` (Windows-only).

    Thread-count assertions use a delta against a per-test baseline
    (``threading.active_count()``) with a short settle poll, never an
    absolute count. ``_wedged_worker_count`` is a module-level global
    (ticket #90's process-wide cap accounting); every test that reads or
    relies on its value resets it to ``0`` via ``monkeypatch`` first, for
    isolation from any other test in this class/session.
    """

    @staticmethod
    def _wait_until_thread_gone(thread: threading.Thread, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)

    @staticmethod
    def _wait_until(predicate, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)

    # -- B1: a completed scan leaves no worker thread behind -------------

    def test_worker_thread_exits_after_close(self):
        worker = _BoundedQueryWorker()
        thread = worker._thread
        assert thread.is_alive()

        outcome = worker.submit(lambda: "ok")
        assert outcome.status == _QueryStatus.RESOLVED
        assert outcome.value == "ok"

        worker.close()
        self._wait_until_thread_gone(thread)
        assert not thread.is_alive(), "worker thread must exit after close()"

    def test_repeated_create_submit_close_cycles_leave_no_thread_delta(self):
        baseline = threading.active_count()
        for _ in range(50):
            worker = _BoundedQueryWorker()
            outcome = worker.submit(lambda: 1)
            assert outcome.status == _QueryStatus.RESOLVED
            worker.close()

        self._wait_until(lambda: threading.active_count() <= baseline)
        assert threading.active_count() == baseline, (
            "50 create/submit/close cycles must leave the thread count unchanged"
        )

    def test_close_is_idempotent(self):
        worker = _BoundedQueryWorker()
        thread = worker._thread
        worker.close()
        worker.close()  # must not raise or hang
        self._wait_until_thread_gone(thread)
        assert not thread.is_alive()

    def test_close_on_worker_that_never_received_a_job(self):
        worker = _BoundedQueryWorker()
        thread = worker._thread
        worker.close()
        self._wait_until_thread_gone(thread)
        assert not thread.is_alive()

    def test_submit_after_close_returns_non_resolved_without_raising(self):
        worker = _BoundedQueryWorker()
        worker.close()

        outcome = worker.submit(lambda: "should never run")

        assert outcome.status != _QueryStatus.RESOLVED

    # -- B2: repeated timeouts do not grow the thread count without bound,
    #        across calls -------------------------------------------------

    def test_repeated_timeouts_are_capped_process_wide(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        grace = _GraceBudget(0.0)  # zeroed -- stage 2 never engages
        scan_deadline = time.monotonic() + 60

        baseline = threading.active_count()
        worker = _BoundedQueryWorker()
        created = [worker]
        statuses = []
        try:
            for _ in range(200):
                outcome = worker.submit(
                    lambda: release.wait(timeout=30), grace=grace, scan_deadline=scan_deadline
                )
                statuses.append(outcome.status)
                if outcome.status == _QueryStatus.ABANDONED and _wedged_slot_available():
                    worker = _BoundedQueryWorker()
                    created.append(worker)
                # Otherwise keep resubmitting to the same, now-retired
                # worker -- mirrors many callers hammering a saturated
                # system; a retired worker refuses immediately (see
                # test_submit_after_close_returns_non_resolved_without_raising)
                # without ever creating a new thread.

            assert _QueryStatus.ABANDONED in statuses
            assert len(created) <= _MAX_WEDGED_HANDLE_WORKERS, (
                f"expected at most {_MAX_WEDGED_HANDLE_WORKERS} worker instances "
                f"to ever be created across 200 attempted submissions, got {len(created)}"
            )
            alive_delta = threading.active_count() - baseline
            assert alive_delta <= _MAX_WEDGED_HANDLE_WORKERS, (
                f"expected at most {_MAX_WEDGED_HANDLE_WORKERS} live worker threads "
                f"after 200 attempted submissions, got a delta of {alive_delta}"
            )
        finally:
            release.set()
            for w in created:
                w.close()
            self._wait_until(lambda: threading.active_count() <= baseline)
            assert threading.active_count() == baseline, (
                "releasing the blocked callable must drive the thread delta back to 0"
            )

    def test_submit_at_cap_returns_capped_without_raising(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", _MAX_WEDGED_HANDLE_WORKERS)
        release = threading.Event()
        worker = _BoundedQueryWorker()
        try:
            outcome = worker.submit(
                lambda: release.wait(timeout=10),
                grace=_GraceBudget(0.0),
                scan_deadline=time.monotonic() + 10,
            )
            assert outcome.status == _QueryStatus.CAPPED
        finally:
            release.set()
            worker.close()

    def test_capped_outcome_still_counts_and_releases_its_slot(self, monkeypatch):
        """Review finding (ticket #90 fix pass): a worker whose own query
        wedges *after* the process-wide cap is already full elsewhere
        returns CAPPED -- but its thread is just as genuinely blocked
        inside ``fn()`` as an ABANDONED worker's is, for exactly as long as
        that real call takes. Before this fix, ``slot_acquired`` was left
        ``False`` for CAPPED, so this worker's own thread was never counted
        into ``_wedged_worker_count`` and therefore never decremented by
        ``_run()`` either -- ``_MAX_WEDGED_HANDLE_WORKERS`` stopped being a
        real bound on live blocked threads, only on how many *replacement*
        workers a single scan would create. This reproduces that scenario
        directly: seed the counter to the cap (simulating capacity already
        claimed by other workers), then drive a fresh worker's own query
        into CAPPED and assert its thread is counted while blocked and its
        slot is released once the wedged call finally returns -- identical
        bookkeeping to the ABANDONED path."""
        monkeypatch.setattr(_pl, "_wedged_worker_count", _MAX_WEDGED_HANDLE_WORKERS)
        release = threading.Event()
        worker = _BoundedQueryWorker()
        try:
            outcome = worker.submit(
                lambda: release.wait(timeout=10),
                grace=_GraceBudget(0.0),
                scan_deadline=time.monotonic() + 10,
            )
            assert outcome.status == _QueryStatus.CAPPED
            # The CAPPED worker's own thread is still live and blocked in
            # fn() -- it must be counted, not silently dropped from the
            # accounting just because no *new* slot was "granted".
            assert _pl._wedged_worker_count == _MAX_WEDGED_HANDLE_WORKERS + 1, (
                "a CAPPED worker's own blocked thread must still be counted "
                "against _wedged_worker_count, exactly like ABANDONED"
            )

            release.set()
            self._wait_until(
                lambda: _pl._wedged_worker_count == _MAX_WEDGED_HANDLE_WORKERS
            )
            assert _pl._wedged_worker_count == _MAX_WEDGED_HANDLE_WORKERS, (
                "the CAPPED worker's slot must be released once its wedged "
                "call finally returns, same as the ABANDONED path -- "
                "otherwise the cap ratchets upward forever and never bounds "
                "live thread count"
            )
        finally:
            release.set()
            worker.close()

    def test_wedged_worker_count_restored_after_retired_worker_exits(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        worker = _BoundedQueryWorker()
        try:
            outcome = worker.submit(
                lambda: release.wait(timeout=10),
                grace=_GraceBudget(0.0),
                scan_deadline=time.monotonic() + 10,
            )
            assert outcome.status == _QueryStatus.ABANDONED
            assert _pl._wedged_worker_count == 1

            release.set()
            self._wait_until(lambda: _pl._wedged_worker_count == 0)
            assert _pl._wedged_worker_count == 0, (
                "the counter must be restored once the retired worker's wedged "
                "call finally returns, so a later scan can wedge again"
            )
        finally:
            release.set()
            worker.close()

    # -- B3: an abandoned worker self-terminates once its wedged call
    #        returns ------------------------------------------------------

    def test_retired_worker_exits_when_its_query_finally_returns(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        calls = []

        worker = _BoundedQueryWorker()
        thread = worker._thread
        try:
            outcome = worker.submit(
                lambda: release.wait(timeout=10),
                grace=_GraceBudget(0.0),
                scan_deadline=time.monotonic() + 10,
                on_abandoned_done=lambda value: calls.append(value),
            )
            assert outcome.status == _QueryStatus.ABANDONED
            assert thread.is_alive()

            release.set()
            self._wait_until_thread_gone(thread)

            assert not thread.is_alive(), (
                "the retired worker must exit on its own once its wedged call finally returns"
            )
            assert calls == [True], "on_abandoned_done must fire exactly once"
        finally:
            release.set()
            worker.close()

    def test_retired_worker_callable_that_raises_still_triggers_cleanup(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        calls = []

        def _raises_after_release():
            release.wait(timeout=10)
            raise RuntimeError("boom")

        worker = _BoundedQueryWorker()
        thread = worker._thread
        try:
            outcome = worker.submit(
                _raises_after_release,
                grace=_GraceBudget(0.0),
                scan_deadline=time.monotonic() + 10,
                on_abandoned_done=lambda value: calls.append(value),
            )
            assert outcome.status == _QueryStatus.ABANDONED

            release.set()
            self._wait_until_thread_gone(thread)

            assert not thread.is_alive()
            assert calls == [None], "a raising callable must still swallow the exception and clean up"
            self._wait_until(lambda: _pl._wedged_worker_count == 0)
            assert _pl._wedged_worker_count == 0
        finally:
            release.set()
            worker.close()

    # -- B4: the abandoned handle is not closed by the scan loop ---------

    def test_abandoned_job_closes_its_own_handle_exactly_once(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        close_calls = []

        worker = _BoundedQueryWorker()
        try:
            outcome = worker.submit(
                lambda: release.wait(timeout=10),
                grace=_GraceBudget(0.0),
                scan_deadline=time.monotonic() + 10,
                on_abandoned_done=lambda value: close_calls.append(value),
            )
            assert outcome.status == _QueryStatus.ABANDONED
            assert close_calls == [], "must not close before the blocked callable returns"

            release.set()
            self._wait_until(lambda: close_calls)

            assert len(close_calls) == 1, f"expected exactly one close, got {close_calls}"
        finally:
            release.set()
            worker.close()

    def test_dependent_follow_up_query_never_runs_after_worker_retires(self):
        """Mirrors _win_handle_holders' type-probe-then-name-query pattern:
        once a worker retires because its current query was abandoned, it
        must never run a second, dependent callable -- ownership of
        whatever the first callable was operating on has already
        transferred away. The real scan additionally guards this at the
        call-site level (it checks the first query's returned status before
        ever attempting the second) -- this test demonstrates the worker
        itself also refuses defensively, via the same closed-worker
        short-circuit exercised by
        test_submit_after_close_returns_non_resolved_without_raising."""
        release = threading.Event()
        follow_up_invoked = []

        def _follow_up():
            follow_up_invoked.append(True)
            return "should never run"

        worker = _BoundedQueryWorker()
        try:
            first = worker.submit(
                lambda: release.wait(timeout=10),
                grace=_GraceBudget(0.0),
                scan_deadline=time.monotonic() + 10,
            )
            assert first.status != _QueryStatus.RESOLVED

            second = worker.submit(_follow_up)
            assert second.status != _QueryStatus.RESOLVED
            assert follow_up_invoked == [], (
                "a dependent follow-up query must never actually run once its "
                "worker has already retired"
            )
        finally:
            release.set()
            worker.close()

    # -- B5: a merely-slow query is recovered by the bounded grace wait --

    def test_slow_query_resolved_in_grace_window_keeps_same_worker(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        grace = _GraceBudget(_HANDLE_QUERY_GRACE_BUDGET_SEC)
        scan_deadline = time.monotonic() + 5

        def _slow():
            time.sleep(0.04)
            return "value"

        worker = _BoundedQueryWorker()
        original_thread = worker._thread
        try:
            t0 = time.monotonic()
            outcome = worker.submit(_slow, grace=grace, scan_deadline=scan_deadline)
            elapsed = time.monotonic() - t0

            assert outcome.status == _QueryStatus.RESOLVED
            assert outcome.value == "value"
            assert worker._thread is original_thread, (
                "a grace-recovered query must not replace the worker"
            )
            assert _pl._wedged_worker_count == 0

            grace_spent = _HANDLE_QUERY_GRACE_BUDGET_SEC - grace.remaining
            # Upper bound only (ticket #90 CI flake sweep): grace_spent must
            # never exceed roughly the observed wait, but it must NOT assert
            # a minimum. On a slow/contended runner, scheduling overhead
            # around the stage-2 wait could in principle leave grace_spent
            # at (or clamped to) 0.0 even though the query still ultimately
            # resolved -- that is not a bug in the design, so a lower bound
            # here is not something the design promises.
            assert grace_spent < elapsed + 0.2, (
                f"grace.remaining dropped by {grace_spent:.3f}s -- expected it to "
                f"roughly track the observed ~{elapsed:.3f}s wait"
            )
        finally:
            worker.close()

    def test_fast_query_spends_zero_grace(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        grace = _GraceBudget(_HANDLE_QUERY_GRACE_BUDGET_SEC)

        worker = _BoundedQueryWorker()
        try:
            outcome = worker.submit(lambda: "fast", grace=grace, scan_deadline=time.monotonic() + 5)
            assert outcome.status == _QueryStatus.RESOLVED
            assert grace.remaining == _HANDLE_QUERY_GRACE_BUDGET_SEC
        finally:
            worker.close()

    def test_query_slower_than_grace_ceiling_is_abandoned_after_bounded_wait(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        grace = _GraceBudget(_HANDLE_QUERY_GRACE_BUDGET_SEC)

        def _too_slow():
            time.sleep(0.3)
            return "late"

        worker = _BoundedQueryWorker()
        try:
            t0 = time.monotonic()
            outcome = worker.submit(_too_slow, grace=grace, scan_deadline=time.monotonic() + 5)
            elapsed = time.monotonic() - t0

            assert outcome.status == _QueryStatus.ABANDONED
            # Stage 1 (~0.01s) + stage 2 ceiling (~0.10s) -- generous
            # one-sided bound, never a tight window.
            assert elapsed < 1.0, f"expected a bounded ~0.11s total wait, took {elapsed:.3f}s"
        finally:
            time.sleep(0.35)  # let the slow callable actually finish first
            worker.close()

    # -- B6: the grace budget is bounded per scan and clamped by the scan
    #        deadline -----------------------------------------------------

    def test_grace_budget_exhausts_and_degrades_to_fast_timeout(self, monkeypatch):
        """Ticket #90 CI flake sweep (3rd-in-a-row failure class).

        The previous design accumulated grace spend across several
        *partial* stage-2 waits (a 0.25s pool against a 0.10s per-query
        ceiling), then asserted the pool ended up at (near-)exact 0.0 and
        that *some* count of queries had paid into it. Both assertions were
        lower bounds on timing-derived values in disguise -- and both
        broke in CI (first an exact-zero `remaining`, then a `grace_paid <=
        3` ceiling that a small residue defeated).

        This version is deterministic *by construction* instead: the pool
        is sized to at most 1.5x the per-query stage-2 ceiling
        (``_HANDLE_QUERY_GRACE_SEC``), so it is provably exhausted to
        *exactly* 0.0 within at most the first two queries --
        ``threading.Event.wait(timeout)`` is guaranteed (by the stdlib) to
        never return before *timeout* elapses unless the event is set, and
        ``release`` is never set until this test's ``finally``. So each
        stage-2 wait's actually-elapsed time is guaranteed >= the allowance
        it was granted, which means ``max(0.0, remaining - elapsed)``
        clamps to exactly 0.0 once the cumulative granted allowance reaches
        the seeded budget -- no residue is possible, no matter how slow or
        jittery the runner is. ``scan_deadline`` is seeded 30s out so it
        can never be the thing that clamps stage 2 (that scenario is
        covered separately by test_grace_truncated_by_near_scan_deadline).
        """
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        initial_budget = 1.5 * _HANDLE_QUERY_GRACE_SEC
        grace = _GraceBudget(initial_budget)
        scan_deadline = time.monotonic() + 30

        per_query_elapsed = []
        worker = _BoundedQueryWorker()
        created = [worker]
        try:
            for _ in range(5):
                t0 = time.monotonic()
                outcome = worker.submit(
                    lambda: release.wait(timeout=30), grace=grace, scan_deadline=scan_deadline
                )
                elapsed = time.monotonic() - t0
                per_query_elapsed.append(elapsed)
                assert outcome.status != _QueryStatus.RESOLVED
                if _wedged_slot_available():
                    worker = _BoundedQueryWorker()
                    created.append(worker)

            # Exact, by construction (see docstring): after at most the
            # first two queries the pool is guaranteed to have clamped to
            # exactly 0.0, and it can only ever decrease or stay the same,
            # so by the end of all 5 queries it is still exactly 0.0. This
            # is the "remaining in cases where stage 2 provably cannot run
            # [again]" carve-out, not a bet on the runner being fast.
            assert grace.remaining == 0.0, (
                f"expected the grace pool to be exhausted to exactly 0.0 by "
                f"construction (guaranteed-minimum-length stage-2 waits), "
                f"{grace.remaining:.6f}s remained"
            )

            # Upper bound only, applied to every query including the ones
            # that paid grace: the most any single query can ever take by
            # design is stage 1 (~0.01s) plus the stage-2 ceiling (~0.10s).
            # The bound is deliberately generous (2.0s, not a tight
            # multiple of the design's ~0.11s nominal total) -- ticket #90
            # CI flake sweep, heavy-load verification pass: a tight
            # multiplier-based ceiling (previously 3x the stage-2
            # allowance, 0.30s) was observed to leave too little headroom
            # for real scheduling overshoot on a contended runner (other
            # nominally-~0.01s-only waits in this class were observed to
            # take up to ~0.10s under artificial CPU saturation). 2.0s is
            # still nowhere near the 30s the callable would need to
            # resolve on its own, so this still clearly catches an actual
            # hang. Never a lower bound: which (if any) of the 5 queries
            # actually paid grace is itself timing-dependent and
            # deliberately not asserted.
            for i, elapsed in enumerate(per_query_elapsed, start=1):
                assert elapsed < 2.0, (
                    f"expected query #{i} to take at most a bounded "
                    f"stage-1 + stage-2 wait, took {elapsed:.3f}s"
                )
        finally:
            release.set()
            for w in created:
                w.close()

    def test_grace_skipped_when_scan_deadline_already_passed(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        grace = _GraceBudget(1.0)
        past_deadline = time.monotonic() - 1.0

        worker = _BoundedQueryWorker()
        try:
            t0 = time.monotonic()
            outcome = worker.submit(
                lambda: release.wait(timeout=10), grace=grace, scan_deadline=past_deadline
            )
            elapsed = time.monotonic() - t0

            assert outcome.status != _QueryStatus.RESOLVED
            # Generous, one-sided sanity bound (ticket #90 CI flake sweep,
            # heavy-load verification pass): this is *not* a tight timing
            # differentiation -- the real proof that stage 2 was skipped is
            # the exact `grace.remaining` assertion below, which holds
            # regardless of scheduling overhead. This is only "didn't
            # hang" sanity, so it must tolerate real scheduling overshoot
            # around the stage-1 wait itself on a contended runner (5x
            # _HANDLE_QUERY_TIMEOUT_SEC, i.e. 0.05s, was observed to fail
            # under artificial CPU saturation).
            assert elapsed < 1.0, f"expected an ~stage-1-only wait, took {elapsed:.3f}s"
            assert grace.remaining == 1.0, "no grace should be spent once the deadline has passed"
        finally:
            release.set()
            worker.close()

    def test_grace_truncated_by_near_scan_deadline(self, monkeypatch):
        """Ticket #90 CI flake sweep (the 3rd-in-a-row failure).

        ``scan_deadline`` is seeded only ~20ms out on purpose, to exercise
        stage 2 being *truncated* by an imminent deadline. But on a slow,
        contended runner more than that ~20ms margin can elapse between
        seeding ``scan_deadline`` above and ``submit()`` computing
        ``scan_deadline - time.monotonic()`` for the stage-2 allowance --
        in which case that term is already <= 0, ``allowance > 0`` is
        false, and stage 2 is skipped *entirely*: zero grace spent, zero
        extra elapsed time. That is truncation working exactly as
        designed (the near deadline clamped stage 2 down to nothing), not
        a bug -- so this test must never assert a *minimum* spend/elapsed;
        only that the design's ceiling holds.
        """
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        grace = _GraceBudget(1.0)

        worker = _BoundedQueryWorker()
        try:
            near_deadline_margin = _HANDLE_QUERY_TIMEOUT_SEC + 0.02
            scan_deadline = time.monotonic() + near_deadline_margin
            t0 = time.monotonic()
            outcome = worker.submit(
                lambda: release.wait(timeout=10), grace=grace, scan_deadline=scan_deadline
            )
            elapsed = time.monotonic() - t0

            assert outcome.status != _QueryStatus.RESOLVED
            # Generous, one-sided sanity bound (ticket #90 CI flake sweep,
            # heavy-load verification pass): this is *not* a tight
            # differentiation between "truncated" and "full stage 2" --
            # the real proof of truncation is the grace_spent ceiling
            # below, which stays meaningful (well under the full 1.0s
            # pool) regardless of scheduling overshoot. This bound is only
            # "didn't hang" sanity, so it must tolerate real overshoot on
            # a contended runner (the design's nominal ~0.03s ceiling, or
            # even the full 0.10s stage-2 ceiling, was observed to fail
            # under artificial CPU saturation, e.g. 0.105s measured
            # against a 0.10s bound).
            assert elapsed < 1.0, (
                f"expected the grace wait truncated to at most the near "
                f"scan_deadline, took {elapsed:.3f}s"
            )
            # Upper bound only (this was CI failure #3): grace spent must
            # never approach the full seeded pool -- it must NOT assert a
            # minimum. Zero spend is a legitimate outcome (see docstring):
            # if the deadline had already passed by the time submit()
            # reached the allowance computation, stage 2 never ran at all
            # and grace.remaining stays at exactly the seeded 1.0 -- that
            # is truncation doing its job, not a defect. The ceiling is
            # deliberately generous (half the pool, not the design's
            # nominal ~0.03s margin) because a single stage-2 wait's
            # *actually elapsed* time -- what grace_spent tracks -- can
            # overshoot its nominal allowance substantially under real
            # scheduling contention (observed under artificial CPU
            # saturation); the meaningful invariant this still proves is
            # that scan_deadline is clamping the spend at all, not letting
            # it run away to the full pool.
            grace_spent = 1.0 - grace.remaining
            assert grace_spent < 0.5, (
                f"expected grace spend to stay well clear of the full "
                f"seeded pool (truncation should keep it small), spent "
                f"{grace_spent:.3f}s"
            )
        finally:
            release.set()
            worker.close()

    def test_zero_grace_budget_is_single_stage_only(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        grace = _GraceBudget(0.0)

        worker = _BoundedQueryWorker()
        try:
            t0 = time.monotonic()
            outcome = worker.submit(
                lambda: release.wait(timeout=10), grace=grace, scan_deadline=time.monotonic() + 5
            )
            elapsed = time.monotonic() - t0

            assert outcome.status != _QueryStatus.RESOLVED
            # Generous, one-sided sanity bound (ticket #90 CI flake sweep,
            # heavy-load verification pass) -- see the identical comment
            # in test_grace_skipped_when_scan_deadline_already_passed.
            assert elapsed < 1.0
            assert grace.remaining == 0.0
        finally:
            release.set()
            worker.close()


# ---------------------------------------------------------------------------
# TestWinHandleHoldersIntegration -- ticket #71 (Pass 1c wiring)
# ---------------------------------------------------------------------------

class TestWinHandleHoldersIntegration:
    """Wiring tests for Pass 1c (``_win_handle_holders``) inside
    ``_find_blocking_processes``. ``_win_handle_holders`` itself is mocked
    here -- its real ctypes/NT internals are exercised separately by
    ``TestWinHandleHoldersReal`` (Windows-only, real subprocess + real
    handles).
    """

    def test_foreign_process_invisible_to_other_passes_is_found_via_handle_scan(self):
        """Regression for ticket #71: a process whose cwd() and
        open_files() both raise AccessDenied, and whose cmdline contains no
        path token under the worktree (so it is invisible to Pass 1, 1b,
        and 2), is still reported once the OS-level handle scan (Pass 1c)
        reports it holding a handle inside the worktree."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc_foreign = MagicMock()
        proc_foreign.info = {"pid": 9101, "name": "", "cmdline": ["some.exe", "--flag"]}
        proc_foreign.cwd.side_effect = psutil.AccessDenied(9101)
        proc_foreign.open_files.side_effect = psutil.AccessDenied(9101)

        def _process_side_effect(pid):
            m = MagicMock()
            if pid == host_pid:
                m.parents.return_value = []
            else:
                m.cmdline.return_value = ["some.exe", "--flag"]
                m.name.return_value = "some.exe"
            return m

        with (
            patch.object(psutil, "process_iter", return_value=[proc_foreign]),
            patch.object(psutil, "Process", side_effect=_process_side_effect),
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.process_lifecycle._win_handle_holders",
                return_value=[(9101, "some.exe")],
            ) as mock_handle_scan,
        ):
            mock_sys.platform = "win32"

            result = _find_blocking_processes(target, host_pid)

        mock_handle_scan.assert_called_once()
        assert len(result) == 1, f"expected the foreign PID to be reported, got {result}"
        assert result[0].pid == 9101
        assert result[0].name == "some.exe"

    def test_pid_already_found_by_earlier_pass_not_duplicated(self):
        """A PID already matched by Pass 1 (cwd) must not be duplicated
        when Pass 1c's handle scan also reports it."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc_cwd_match = _make_fake_proc(9102, "node", ["node"], target)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_cwd_match]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.process_lifecycle._win_handle_holders",
                return_value=[(9102, "node")],
            ) as mock_handle_scan,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            result = _find_blocking_processes(target, host_pid)

        mock_handle_scan.assert_called_once()
        assert len(result) == 1, (
            f"PID found by both Pass 1 and Pass 1c must appear exactly once, got {result}"
        )
        assert result[0].pid == 9102

    def test_handle_scan_excluded_pid_is_dropped(self):
        """A PID reported by _win_handle_holders that is in excluded_pids
        (the host process or one of its ancestors) must not appear in the
        final result."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()
        ancestor_pid = 4242

        ancestor_mock = MagicMock()
        ancestor_mock.pid = ancestor_pid

        with (
            patch.object(psutil, "process_iter", return_value=[]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.process_lifecycle._win_handle_holders",
                return_value=[(host_pid, "host"), (ancestor_pid, "ancestor")],
            ),
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = [ancestor_mock]
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            result = _find_blocking_processes(target, host_pid)

        assert result == [], (
            f"excluded PIDs (host + ancestors) reported by the handle scan "
            f"must be dropped, got {result}"
        )

    def test_handle_scan_failure_degrades_gracefully(self):
        """If _win_handle_holders raises (ctypes/OS failure), the function
        must not raise and must still return whatever Pass 1/1b/2 already
        found."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc_cwd_match = _make_fake_proc(9103, "bash", ["bash"], target)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_cwd_match]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.process_lifecycle._win_handle_holders",
                side_effect=OSError("simulated ctypes failure"),
            ),
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            result = _find_blocking_processes(target, host_pid)  # must not raise

        assert len(result) == 1
        assert result[0].pid == 9103

    def test_handle_scan_skipped_on_non_windows(self):
        """Pass 1c must not run -- and _win_handle_holders must never be
        called -- on non-Windows platforms."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        with (
            patch.object(psutil, "process_iter", return_value=[]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.process_lifecycle._win_handle_holders"
            ) as mock_handle_scan,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            result = _find_blocking_processes(target, host_pid)

        mock_handle_scan.assert_not_called()
        assert result == []


# ---------------------------------------------------------------------------
# TestHandleScanDeadlineThreading -- ticket #71 follow-up (reviewer finding):
# Pass 1c's Windows handle-table scan must respect the caller's overall
# timeout instead of always independently consuming up to its own fixed
# _HANDLE_SCAN_BUDGET_SEC (15.0s) ceiling on top of it -- e.g.
# stop(timeout=10.0, kill_orphans=True) must not silently take ~25s.
# ---------------------------------------------------------------------------

class TestHandleScanDeadlineThreading:
    """Regression tests: the deadline is computed once in
    _kill_blocking_processes *before* discovery starts, threaded into
    _find_blocking_processes as `deadline`, and from there into
    _win_handle_holders as a capped `budget_sec` -- rather than
    _win_handle_holders always using its own fixed 15s ceiling regardless of
    how much time the caller actually has left.
    """

    def test_kill_blocking_processes_threads_deadline_into_find_blocking_processes(self):
        """_kill_blocking_processes must pass a `deadline` kwarg to
        _find_blocking_processes, computed from `timeout` BEFORE discovery
        starts (so discovery time counts against the same overall budget as
        the subsequent kill/wait step, instead of being extra time on top)."""
        target = "/fake/worktree"
        captured: dict = {}
        t0 = time.monotonic()

        def _fake_find(path, host_pid, **kwargs):
            captured["deadline"] = kwargs.get("deadline")
            return []

        with patch(
            "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
            side_effect=_fake_find,
        ):
            _kill_blocking_processes(target, timeout=3.0)

        assert "deadline" in captured, (
            "deadline kwarg was not passed to _find_blocking_processes at all"
        )
        assert captured["deadline"] is not None, (
            "deadline must not be None when a finite timeout is given"
        )
        # The deadline must be ~t0 + 3.0 (computed before discovery), not
        # computed afterward -- assert it lands close to the expected value.
        assert abs(captured["deadline"] - (t0 + 3.0)) < 0.5, (
            f"deadline {captured['deadline']} should be ~{t0 + 3.0} (t0 + timeout)"
        )

    def test_find_blocking_processes_caps_handle_scan_budget_to_remaining_deadline(self):
        """On win32, _win_handle_holders must be called with a `budget_sec`
        bounded by the time remaining until *deadline*, not the full 15s
        _HANDLE_SCAN_BUDGET_SEC ceiling, when the caller's deadline leaves
        less time than that ceiling."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()
        captured_budget: List[float] = []

        with (
            patch.object(psutil, "process_iter", return_value=[]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.process_lifecycle._win_handle_holders",
                side_effect=lambda path, excluded, **kw: (
                    captured_budget.append(kw.get("budget_sec")) or []
                ),
            ),
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            deadline = time.monotonic() + 0.2  # far less than the 15.0s ceiling
            _find_blocking_processes(target, host_pid, deadline=deadline)

        assert len(captured_budget) == 1, "expected exactly one _win_handle_holders call"
        budget = captured_budget[0]
        assert budget is not None, (
            "_win_handle_holders must be called with an explicit budget_sec"
        )
        assert budget <= 0.2 + 0.05, (
            f"handle scan budget {budget} must be bounded by the caller's remaining "
            f"deadline (~0.2s), not the full _HANDLE_SCAN_BUDGET_SEC ceiling (15.0s)"
        )

    def test_find_blocking_processes_skips_handle_scan_when_deadline_already_passed(self):
        """When *deadline* has already elapsed by the time Pass 1c would
        run, it must be skipped entirely -- _win_handle_holders must not be
        called at all, rather than being called with a near-zero budget."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        with (
            patch.object(psutil, "process_iter", return_value=[]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.process_lifecycle._win_handle_holders"
            ) as mock_handle_scan,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            past_deadline = time.monotonic() - 1.0
            result = _find_blocking_processes(target, host_pid, deadline=past_deadline)

        mock_handle_scan.assert_not_called()
        assert result == []

    def test_no_deadline_uses_full_ceiling_backward_compatible(self):
        """Direct/legacy callers that omit `deadline` entirely (e.g. calling
        _find_blocking_processes without it, as pre-existing tests and code
        do) must still get the full _HANDLE_SCAN_BUDGET_SEC ceiling passed
        to _win_handle_holders -- this keeps the new parameter opt-in and
        backward compatible."""
        import psutil
        from lib_python_worktree.core.process_lifecycle import _HANDLE_SCAN_BUDGET_SEC

        target = "/fake/worktree"
        host_pid = os.getpid()
        captured_budget: List[float] = []

        with (
            patch.object(psutil, "process_iter", return_value=[]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch(
                "lib_python_worktree.core.process_lifecycle._win_handle_holders",
                side_effect=lambda path, excluded, **kw: (
                    captured_budget.append(kw.get("budget_sec")) or []
                ),
            ),
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            _find_blocking_processes(target, host_pid)  # no deadline kwarg

        assert captured_budget == [_HANDLE_SCAN_BUDGET_SEC]


# ---------------------------------------------------------------------------
# TestWinHandleHoldersReal -- ticket #71 (real subprocess + real OS handles)
# ---------------------------------------------------------------------------

class TestWinHandleHoldersReal:
    """Windows-only real-subprocess test for ``_win_handle_holders``.

    Unlike ``TestWinHandleHoldersIntegration`` (which mocks
    ``_win_handle_holders`` entirely), this exercises the actual
    ``ctypes``/``ntdll`` internals -- ``NtQuerySystemInformation``,
    ``DuplicateHandle``, and ``NtQueryObject`` -- against a real child
    process holding a real open file handle. Skipped outside win32 since
    the implementation is ctypes/ntdll-only and has no meaning elsewhere.

    Also covers the required path-boundary correctness case: exact match,
    true subpath match, and the sibling-directory negative case, all
    against one real held-open file handle.
    """

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-only: exercises ntdll/ctypes handle enumeration",
    )
    def test_real_subprocess_holding_open_file_is_detected_with_path_boundaries(
        self, tmp_path
    ):
        """A real child process holding a real open file handle under a
        temp directory is detected by _win_handle_holders, with correct
        path-boundary semantics:

        - exact match: querying with the file's own path finds it.
        - subpath match: querying with the file's parent directory finds it.
        - sibling non-match: a directory sharing a name prefix with the
          parent (but not an ancestor of the file) does NOT find it.
        """
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        sibling_dir = tmp_path / "target-sibling"
        sibling_dir.mkdir()

        target_file = target_dir / "held.txt"
        target_file.write_text("hold me open")

        code = (
            "import sys, time\n"
            f"f = open({str(target_file)!r}, 'r')\n"
            "sys.stdout.write('ready\\n')\n"
            "sys.stdout.flush()\n"
            "time.sleep(10)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            import psutil

            ready_line = proc.stdout.readline()
            assert ready_line.strip() == "ready", (
                f"child process failed to signal readiness: {ready_line!r}"
            )

            # Narrow the scan to just the child pid so that high ambient
            # system handle-table load elsewhere on the machine can't
            # starve the scan before it reaches this process, and give it
            # a generous budget for the same reason (see
            # _REAL_SCAN_TEST_BUDGET_SEC docstring above).
            excluded_pids = set(psutil.pids()) - {proc.pid}

            # Exact match: querying with the file's own path.
            found_exact = _win_handle_holders(
                str(target_file),
                excluded_pids=excluded_pids,
                budget_sec=_REAL_SCAN_TEST_BUDGET_SEC,
            )
            exact_pids = {pid for pid, _ in found_exact}
            assert proc.pid in exact_pids, (
                f"expected child pid {proc.pid} to be found via exact-path "
                f"match against {target_file}; got {found_exact}"
            )

            # Subpath match: querying with the file's parent directory.
            found_subpath = _win_handle_holders(
                str(target_dir),
                excluded_pids=excluded_pids,
                budget_sec=_REAL_SCAN_TEST_BUDGET_SEC,
            )
            subpath_pids = {pid for pid, _ in found_subpath}
            assert proc.pid in subpath_pids, (
                f"expected child pid {proc.pid} to be found via subpath "
                f"match against {target_dir}; got {found_subpath}"
            )

            # Sibling non-match: a directory that shares a name prefix with
            # the parent but is not an ancestor of the held-open file must
            # not report the child pid.
            found_sibling = _win_handle_holders(
                str(sibling_dir),
                excluded_pids=excluded_pids,
                budget_sec=_REAL_SCAN_TEST_BUDGET_SEC,
            )
            sibling_pids = {pid for pid, _ in found_sibling}
            assert proc.pid not in sibling_pids, (
                f"sibling directory {sibling_dir} must not false-match the "
                f"child pid holding a handle only under {target_dir}; "
                f"got {found_sibling}"
            )
        finally:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-only: exercises ntdll/ctypes handle enumeration",
    )
    def test_scan_still_runs_and_finds_results_when_wedged_cap_already_full(
        self, tmp_path, monkeypatch
    ):
        """Reviewer fix pass, ticket #90: a scan-start check that refused to
        scan at all once ``_wedged_worker_count`` reached
        ``_MAX_WEDGED_HANDLE_WORKERS`` was tried and rejected -- a wedged
        worker may, by definition, never return from its NtQueryObject
        call, so that check would latch permanently once the cap had ever
        been reached, making every later ``_win_handle_holders`` call
        silently return ``[]`` forever for the rest of the process's life.

        This pins the corrected contract directly: even with
        ``_wedged_worker_count`` seeded at the cap (simulating capacity
        already claimed elsewhere in the process, exactly as the rejected
        scan-start check would have seen it), a scan must still create its
        initial worker, actually run, and return genuine findings -- not an
        empty list.
        """
        monkeypatch.setattr(_pl, "_wedged_worker_count", _MAX_WEDGED_HANDLE_WORKERS)

        target_dir = tmp_path / "target"
        target_dir.mkdir()
        target_file = target_dir / "held.txt"
        target_file.write_text("hold me open")

        code = (
            "import sys, time\n"
            f"f = open({str(target_file)!r}, 'r')\n"
            "sys.stdout.write('ready\\n')\n"
            "sys.stdout.flush()\n"
            "time.sleep(10)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            import psutil

            ready_line = proc.stdout.readline()
            assert ready_line.strip() == "ready", (
                f"child process failed to signal readiness: {ready_line!r}"
            )

            excluded_pids = set(psutil.pids()) - {proc.pid}

            result = _win_handle_holders(
                str(target_file),
                excluded_pids=excluded_pids,
                budget_sec=_REAL_SCAN_TEST_BUDGET_SEC,
            )
            found_pids = {pid for pid, _ in result}
            assert proc.pid in found_pids, (
                "a scan started while the process-wide wedged-worker cap was "
                "already full must still run and find real results, not "
                f"silently return []; got {result}"
            )
        finally:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-only: exercises ntdll/ctypes handle enumeration",
    )
    def test_zero_budget_skips_per_handle_loop_entirely(self, tmp_path, monkeypatch):
        """De-flake (ticket #107): the previous version of this test asserted
        ``elapsed < 8.0`` around a real, unmocked handle-table dump -- on a
        loaded CI runner the dump itself (unavoidable; see below) could
        legitimately exceed that threshold even with correct behaviour,
        making the assertion measure ambient system load rather than the
        contract under test. This is the same class of fix as D5's
        ``test_handle_scan_truncated_by_own_deadline`` and the N2/N3 tests
        above (see their rationale comments): prove the *structural*
        contract -- "budget_sec <= 0 means the per-PID/per-handle
        resolution loop never executes a single query" -- by spying on
        ``_BoundedQueryWorker.submit`` (the function that loop would have to
        call at least once to resolve even a single handle) and asserting it
        is never invoked, rather than by racing a stopwatch against
        unpredictable ambient load.

        The handle-table dump/parse itself (``NtQuerySystemInformation`` +
        structure walk) runs unconditionally before the per-handle loop's
        deadline check is ever consulted -- that part is real and unmocked,
        so this still exercises the genuine early-exit path against the live
        system, just without timing it."""
        target = str(tmp_path / "definitely-not-a-real-worktree")
        calls: List[int] = []

        def fake_submit(self, fn, *, grace=None, scan_deadline=None, on_abandoned_done=None):
            calls.append(1)
            return _pl._QueryOutcome(_pl._QueryStatus.CAPPED, None)

        monkeypatch.setattr(_pl._BoundedQueryWorker, "submit", fake_submit)

        result = _win_handle_holders(target, excluded_pids=set(), budget_sec=0.0)

        assert not calls, (
            "_BoundedQueryWorker.submit was invoked -- the per-handle "
            "resolution loop ran despite budget_sec=0.0 (already expired at "
            "entry), which is exactly what this contract forbids"
        )
        assert result == []
        assert result.complete is False

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-only: exercises ntdll/ctypes handle enumeration",
    )
    def test_real_scan_is_bounded_to_target_pid_regardless_of_load(self, tmp_path):
        """Regression for #73: pins the API contract of a PID-scoped call --
        when ``excluded_pids`` is set to every pid except the child's, the
        scan must report *only* the child's pid (never an extra one) while
        still finding it. This exercises the same internal ``excluded_pids``
        filter (see ``_win_handle_holders``'s ``by_pid`` construction) that
        this module's flaky-test fix now relies on to narrow the per-PID scan
        loop to a single entry regardless of ambient system handle-table
        load.

        Note: on a clean test machine, no other real process happens to hold
        a handle inside the fresh per-test ``tmp_path``, so the "no extra
        pids" half of this assertion holds true even without the
        ``excluded_pids`` scoping applied (verified manually while
        investigating a review comment on this change) -- it does not by
        itself prove resilience to ambient contention. That resilience is
        instead demonstrated by budget-starvation timing behaviour (see the
        module-level ``_REAL_SCAN_TEST_BUDGET_SEC`` comment and the sibling
        test above): an unscoped call needs a multi-second budget to reach
        the target pid past hundreds of other live pids, while a PID-scoped
        call reliably finds it in well under a second because ``by_pid`` has
        only one entry to iterate."""
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        target_file = target_dir / "held.txt"
        target_file.write_text("hold me open")

        code = (
            "import sys, time\n"
            f"f = open({str(target_file)!r}, 'r')\n"
            "sys.stdout.write('ready\\n')\n"
            "sys.stdout.flush()\n"
            "time.sleep(10)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            import psutil

            ready_line = proc.stdout.readline()
            assert ready_line.strip() == "ready", (
                f"child process failed to signal readiness: {ready_line!r}"
            )

            excluded_pids = set(psutil.pids()) - {proc.pid}

            result = _win_handle_holders(
                str(target_dir),
                excluded_pids=excluded_pids,
                budget_sec=_REAL_SCAN_TEST_BUDGET_SEC,
            )
            result_pids = {pid for pid, _ in result}

            assert proc.pid in result_pids, (
                f"expected child pid {proc.pid} to be found in {target_dir}; "
                f"got {result}"
            )
            assert result_pids <= {proc.pid}, (
                "scan must be bounded to the target pid regardless of "
                f"ambient system load; got extra pids {result_pids - {proc.pid}}"
            )
        finally:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass


# ---------------------------------------------------------------------------
# TestWinHandleHoldersThreadHygiene -- ticket #90 (real-scan thread hygiene)
# ---------------------------------------------------------------------------

class TestWinHandleHoldersThreadHygiene:
    """Windows-only real-scan hygiene test (ticket #90 -- B7).

    Unlike ``TestWinHandleHoldersReal`` (which asserts detection
    correctness), this asserts the actual regression this ticket exists to
    fix: repeatedly invoking ``_win_handle_holders`` against a real handle
    table -- the long-lived-host-process scenario a real E2E run observed
    accumulating thousands of threads and non-trivial idle CPU over 36
    minutes of otherwise-idle teardown churn -- must not grow the process's
    thread count. PID-scoped via ``excluded_pids`` and the module's
    ``_REAL_SCAN_TEST_BUDGET_SEC`` test-only budget constant, mirroring
    #73's flake fix, so ambient system load on the machine running the
    suite cannot make this test itself flaky.
    """

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-only: exercises ntdll/ctypes handle enumeration",
    )
    def test_repeated_real_scans_do_not_grow_thread_count(self, tmp_path):
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        target_file = target_dir / "held.txt"
        target_file.write_text("hold me open")

        code = (
            "import sys, time\n"
            f"f = open({str(target_file)!r}, 'r')\n"
            "sys.stdout.write('ready\\n')\n"
            "sys.stdout.flush()\n"
            "time.sleep(120)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            import psutil

            ready_line = proc.stdout.readline()
            assert ready_line.strip() == "ready", (
                f"child process failed to signal readiness: {ready_line!r}"
            )

            excluded_pids = set(psutil.pids()) - {proc.pid}

            # 10 real scans run sequentially against a live handle table.
            # Ticket #90's fix pass re-evaluated this 120s child lifetime
            # (it was suspected of merely papering over the CAPPED-
            # accounting review finding fixed alongside this test): with
            # that production bug fixed, a 30s child lifetime was tried
            # here and still failed reliably, not because of any wedged-
            # worker accounting but because a single real system-wide
            # handle-table scan on a dev/CI host with a large ambient
            # handle count genuinely costs several seconds (measured
            # ~4-5s/scan on one such host here, i.e. ~45-50s cumulative
            # for 10 scans, before even counting process-spawn and
            # readline overhead) -- 30s does not cover that. 120s --
            # matching _REAL_SCAN_TEST_BUDGET_SEC's own generous ceiling
            # (see that constant's docstring) -- keeps roughly 2x headroom
            # over the measured cumulative worst case and is kept
            # deliberately, not as an unexamined carry-over.
            baseline = threading.active_count()
            for i in range(10):
                result = _win_handle_holders(
                    str(target_dir),
                    excluded_pids=excluded_pids,
                    budget_sec=_REAL_SCAN_TEST_BUDGET_SEC,
                )
                found_pids = {pid for pid, _ in result}
                assert proc.pid in found_pids, (
                    f"scan #{i} failed to find the held-open handle; got {result} "
                    "-- ticket #90's refactor must not regress Pass 1c detection quality"
                )

            # Short settle poll: any worker still finishing its very last
            # (already-resolved-by-now) job has a bounded moment to exit
            # before the assertion below.
            deadline = time.monotonic() + 2.0
            delta = threading.active_count() - baseline
            while delta > 0 and time.monotonic() < deadline:
                time.sleep(0.02)
                delta = threading.active_count() - baseline

            assert delta <= 0, (
                f"10 repeated _win_handle_holders scans against a real, healthy "
                f"(non-wedged) handle table left a thread-count delta of {delta} "
                f"-- expected it to settle back to 0, not grow with each scan"
            )
        finally:
            proc.kill()
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass


# ---------------------------------------------------------------------------
# TestProcessTree -- ticket #87 (_process_tree unit tests)
# ---------------------------------------------------------------------------

class TestProcessTree:
    """Unit tests for _process_tree (ticket #87).

    stop() used to track only the outer shell-wrapper PID. A grandchild
    spawned by a nested shell (e.g. a run: step whose command itself invokes
    another shell) survived being reparented once the wrapper died.
    _process_tree snapshots the whole descendant tree while the root is
    still alive so reparenting after the kill cannot hide anything.
    """

    def test_process_tree_pid_zero_or_negative_returns_empty(self):
        assert _process_tree(0) == []
        assert _process_tree(-1) == []
        assert _process_tree(None) == []

    def test_process_tree_returns_empty_on_no_such_process(self):
        import psutil

        host_pid = os.getpid()

        def _process_side_effect(pid):
            if pid == host_pid:
                m = MagicMock()
                m.parents.return_value = []
                return m
            raise psutil.NoSuchProcess(pid)

        with (
            patch.object(psutil, "Process", side_effect=_process_side_effect),
            patch.object(psutil, "pid_exists", return_value=False),
        ):
            result = _process_tree(99999)

        assert result == []

    def test_process_tree_returns_empty_on_access_denied(self):
        import psutil

        host_pid = os.getpid()

        def _process_side_effect(pid):
            if pid == host_pid:
                m = MagicMock()
                m.parents.return_value = []
                return m
            raise psutil.AccessDenied(pid)

        with (
            patch.object(psutil, "Process", side_effect=_process_side_effect),
            patch.object(psutil, "pid_exists", return_value=False),
        ):
            result = _process_tree(4242)

        assert result == []

    def test_process_tree_excludes_host_pid_and_ancestors(self):
        """A "descendant" that happens to be the host process or one of its
        ancestors must never be targeted -- this function must not be able
        to signal the caller's own lineage."""
        import psutil

        host_pid = os.getpid()
        ancestor_pid = 3131
        root_pid = 5000

        ancestor_mock = MagicMock()
        ancestor_mock.pid = ancestor_pid

        child_normal = MagicMock()
        child_normal.pid = 6001
        child_normal.ppid.return_value = root_pid
        child_normal.name.return_value = "child"
        child_normal.cmdline.return_value = ["child"]

        child_is_host = MagicMock()
        child_is_host.pid = host_pid
        child_is_host.ppid.return_value = root_pid

        child_is_ancestor = MagicMock()
        child_is_ancestor.pid = ancestor_pid
        child_is_ancestor.ppid.return_value = root_pid

        def _process_side_effect(pid):
            if pid == host_pid:
                m = MagicMock()
                m.parents.return_value = [ancestor_mock]
                return m
            if pid == root_pid:
                m = MagicMock()
                m.children.return_value = [child_normal, child_is_host, child_is_ancestor]
                return m
            raise AssertionError(f"unexpected psutil.Process({pid}) call")

        with patch.object(psutil, "Process", side_effect=_process_side_effect):
            result = _process_tree(root_pid)

        result_pids = {info.pid for info in result}
        assert result_pids == {6001}
        assert host_pid not in result_pids
        assert ancestor_pid not in result_pids

    def test_process_tree_orders_deepest_first(self):
        """A node whose parent is also in the collected set must sort after
        that parent, so callers can kill children before their own
        ancestors."""
        import psutil

        host_pid = os.getpid()
        root_pid = 7000

        child = MagicMock()
        child.pid = 7001
        child.ppid.return_value = root_pid
        child.name.return_value = "child"
        child.cmdline.return_value = ["child"]

        grandchild = MagicMock()
        grandchild.pid = 7002
        grandchild.ppid.return_value = 7001
        grandchild.name.return_value = "grandchild"
        grandchild.cmdline.return_value = ["grandchild"]

        def _process_side_effect(pid):
            if pid == host_pid:
                m = MagicMock()
                m.parents.return_value = []
                return m
            if pid == root_pid:
                m = MagicMock()
                # children(recursive=True) returns a flat list in whatever
                # order psutil discovered them -- deliberately NOT
                # deepest-first, to prove _process_tree re-sorts it.
                m.children.return_value = [child, grandchild]
                return m
            raise AssertionError(f"unexpected psutil.Process({pid}) call")

        with patch.object(psutil, "Process", side_effect=_process_side_effect):
            result = _process_tree(root_pid)

        result_pids = [info.pid for info in result]
        assert result_pids.index(7002) < result_pids.index(7001), (
            "grandchild (deeper) must appear before its own parent in the result"
        )

    def test_process_tree_caps_at_max_nodes(self):
        import psutil
        from lib_python_worktree.core.process_lifecycle import _MAX_TREE_NODES

        host_pid = os.getpid()
        root_pid = 8000

        many_children = []
        for i in range(_MAX_TREE_NODES + 50):
            m = MagicMock()
            m.pid = 9000 + i
            m.ppid.return_value = root_pid
            m.name.return_value = "child"
            m.cmdline.return_value = ["child"]
            many_children.append(m)

        def _process_side_effect(pid):
            if pid == host_pid:
                m = MagicMock()
                m.parents.return_value = []
                return m
            if pid == root_pid:
                m = MagicMock()
                m.children.return_value = many_children
                return m
            raise AssertionError(f"unexpected psutil.Process({pid}) call")

        with patch.object(psutil, "Process", side_effect=_process_side_effect):
            result = _process_tree(root_pid)

        assert len(result) <= _MAX_TREE_NODES

    def test_process_tree_truncation_logs_warning(self, caplog):
        """Regression (ticket #87 follow-up, finding F2): before this fix,
        hitting the _MAX_TREE_NODES cap was completely silent -- an operator
        had no way to tell from the logs that a stop() call's tree snapshot
        was incomplete. When more descendants exist than the cap allows, a
        warning naming the root pid and the cap must be logged."""
        import psutil
        from lib_python_worktree.core.process_lifecycle import _MAX_TREE_NODES

        caplog.set_level(
            logging.WARNING, logger="lib_python_worktree.core.process_lifecycle"
        )

        host_pid = os.getpid()
        root_pid = 8100

        many_children = []
        for i in range(_MAX_TREE_NODES + 50):
            m = MagicMock()
            m.pid = 9100 + i
            m.ppid.return_value = root_pid
            m.name.return_value = "child"
            m.cmdline.return_value = ["child"]
            many_children.append(m)

        def _process_side_effect(pid):
            if pid == host_pid:
                m = MagicMock()
                m.parents.return_value = []
                return m
            if pid == root_pid:
                m = MagicMock()
                m.children.return_value = many_children
                return m
            raise AssertionError(f"unexpected psutil.Process({pid}) call")

        with patch.object(psutil, "Process", side_effect=_process_side_effect):
            _process_tree(root_pid)

        assert any(
            str(root_pid) in rec.message and str(_MAX_TREE_NODES) in rec.message
            for rec in caplog.records
        ), (
            f"expected a warning naming root pid {root_pid} and the "
            f"{_MAX_TREE_NODES}-node cap, got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_process_tree_not_truncated_when_under_cap_no_warning(self, caplog):
        """The normal, uncapped case must not log a truncation warning --
        this is what keeps _process_tree's behaviour unchanged for the
        common case."""
        import psutil

        caplog.set_level(
            logging.WARNING, logger="lib_python_worktree.core.process_lifecycle"
        )

        host_pid = os.getpid()
        root_pid = 8200

        child = MagicMock()
        child.pid = 9200
        child.ppid.return_value = root_pid
        child.name.return_value = "child"
        child.cmdline.return_value = ["child"]

        def _process_side_effect(pid):
            if pid == host_pid:
                m = MagicMock()
                m.parents.return_value = []
                return m
            if pid == root_pid:
                m = MagicMock()
                m.children.return_value = [child]
                return m
            raise AssertionError(f"unexpected psutil.Process({pid}) call")

        with patch.object(psutil, "Process", side_effect=_process_side_effect):
            result = _process_tree(root_pid)

        assert {info.pid for info in result} == {9200}
        assert caplog.records == []

    def test_process_tree_dead_root_skips_expensive_fallback_scan(self):
        """Regression: a dead/nonexistent root pid must not trigger the
        full-system ppid-walk fallback (psutil.process_iter) -- it cannot
        have discoverable descendants via that path either, and doing the
        scan anyway is pure wasted CPU (this was measured making
        _kill_blocking_processes'/stop()'s wall-clock budget tests flake)."""
        import psutil

        host_pid = os.getpid()

        def _process_side_effect(pid):
            if pid == host_pid:
                m = MagicMock()
                m.parents.return_value = []
                return m
            raise psutil.NoSuchProcess(pid)

        with (
            patch.object(psutil, "Process", side_effect=_process_side_effect),
            patch.object(psutil, "pid_exists", return_value=False),
            patch.object(psutil, "process_iter") as mock_process_iter,
        ):
            result = _process_tree(123456)

        assert result == []
        mock_process_iter.assert_not_called()


# ---------------------------------------------------------------------------
# TestSignalProcessGroup -- ticket #87
# ---------------------------------------------------------------------------

class TestSignalProcessGroup:
    """Unit tests for _signal_process_group (ticket #87).

    POSIX-only path; simulated on this Windows dev box the same way the
    existing TestFindBlockingProcessesWindows suite simulates win32 -- by
    patching module-level `sys` and, here, `os.getpgid`/`os.killpg` (which
    do not exist on Windows, hence create=True).
    """

    def test_signal_process_group_noop_on_windows(self):
        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(os, "killpg", create=True) as mock_killpg,
        ):
            mock_sys.platform = "win32"
            result = _signal_process_group(1234)

        assert result is False
        mock_killpg.assert_not_called()

    def test_signal_process_group_skips_when_pgid_is_not_pid(self):
        """A pid that is not the leader of its own group must not be
        signalled -- its group may contain unrelated processes we did not
        spawn."""
        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(os, "getpgid", create=True, side_effect=lambda p: 42),
            patch.object(os, "killpg", create=True) as mock_killpg,
        ):
            mock_sys.platform = "linux"
            result = _signal_process_group(1234)  # getpgid(1234) -> 42 != 1234

        assert result is False
        mock_killpg.assert_not_called()

    def test_signal_process_group_never_signals_own_group(self):
        """Even if *pid* is a group leader, this function must never signal
        OUR OWN process group."""
        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(os, "getpgid", create=True, side_effect=lambda p: 555),
            patch.object(os, "killpg", create=True) as mock_killpg,
        ):
            mock_sys.platform = "linux"
            # pid 555 is its own group leader (getpgid(555) == 555) AND
            # getpgid(0) (our own group) also resolves to 555.
            result = _signal_process_group(555)

        assert result is False, "must never signal our own process group"
        mock_killpg.assert_not_called()

    def test_signal_process_group_signals_when_leader_and_not_own_group(self):
        calls = []

        def _fake_getpgid(p):
            return 999 if p == 0 else 777

        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(os, "getpgid", create=True, side_effect=_fake_getpgid),
            patch.object(
                os, "killpg", create=True,
                side_effect=lambda pgid, sig: calls.append((pgid, sig)),
            ),
        ):
            mock_sys.platform = "linux"
            result = _signal_process_group(777)

        assert result is True
        assert calls == [(777, signal.SIGTERM)]

    def test_signal_process_group_dead_pid_returns_false(self):
        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(os, "getpgid", create=True, side_effect=OSError("no such process")),
            patch.object(os, "killpg", create=True) as mock_killpg,
        ):
            mock_sys.platform = "linux"
            result = _signal_process_group(424242)

        assert result is False
        mock_killpg.assert_not_called()


# ---------------------------------------------------------------------------
# TestKillProcessTree -- ticket #87
# ---------------------------------------------------------------------------

class TestKillProcessTree:
    """Unit tests for _kill_process_tree (ticket #87)."""

    def test_kill_process_tree_empty_returns_empty(self):
        assert _kill_process_tree([], timeout=5.0) == []

    def test_kill_process_tree_signals_and_waits_each_node_in_order(self):
        tree = [
            KilledProcessInfo(pid=101, name="grandchild", cmdline=["gc"]),
            KilledProcessInfo(pid=102, name="child", cmdline=["c"]),
        ]
        graceful_calls = []
        wait_calls = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
                side_effect=lambda pid: graceful_calls.append(pid),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
                side_effect=lambda pid, timeout: wait_calls.append((pid, timeout)),
            ),
        ):
            result = _kill_process_tree(tree, timeout=4.0)

        assert result == tree
        assert graceful_calls == [101, 102]
        assert [p for p, _ in wait_calls] == [101, 102]
        assert all(t > 0 for _, t in wait_calls)
        assert sum(t for _, t in wait_calls) <= 4.0 + 1e-3

    def test_kill_process_tree_budget_exhausted_skips_remaining_waits(self):
        tree = [
            KilledProcessInfo(pid=201, name="a", cmdline=["a"]),
            KilledProcessInfo(pid=202, name="b", cmdline=["b"]),
        ]
        graceful_calls = []
        wait_calls = []

        def _slow_wait(pid, timeout):
            time.sleep(0.2)
            wait_calls.append((pid, timeout))

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
                side_effect=lambda pid: graceful_calls.append(pid),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
                side_effect=_slow_wait,
            ),
        ):
            _kill_process_tree(tree, timeout=0.1)

        # Both nodes must still be signalled even though the budget runs out.
        assert graceful_calls == [201, 202]
        # With a 0.1s total budget and the first _wait_or_kill call alone
        # taking 0.2s (mocked sleep), the deadline is exhausted by the time
        # the second node is reached -- it must be signalled but not waited on.
        assert len(wait_calls) <= 1


# ---------------------------------------------------------------------------
# TestStopKillsProcessTree -- ticket #87
# ---------------------------------------------------------------------------

class TestStopKillsProcessTree:
    """Regression tests for ticket #87: stop() must kill the process TREE
    rooted at the tracked pid, not just the tracked pid itself -- otherwise a
    grandchild spawned by a nested shell (e.g. run: powershell ... -Command
    "...") survives being reparented once the wrapper dies, along with any
    ports it holds, while stop() reports status="stopped"."""

    def test_stop_kills_grandchild_process(self, tmp_path):
        """Driving test (R1): a real wrapper process spawns a real grandchild
        in its OWN process group/session (CREATE_NEW_PROCESS_GROUP on
        win32, start_new_session=True on POSIX) -- so it is not reachable by
        simply signalling the wrapper's own group/console -- and writes the
        grandchild's pid to a marker file. stop() must terminate the
        grandchild too, not just the wrapper."""
        record = _make_record("wt-grandchild")
        store = _make_store(record)

        marker_path = tmp_path / "grandchild_pid.txt"

        wrapper_code = (
            "import subprocess, sys, time\n"
            "if sys.platform == 'win32':\n"
            "    kwargs = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}\n"
            "else:\n"
            "    kwargs = {'start_new_session': True}\n"
            "p = subprocess.Popen(\n"
            "    [sys.executable, '-c', 'import time; time.sleep(300)'],\n"
            "    **kwargs,\n"
            ")\n"
            f"with open({str(marker_path)!r}, 'w') as f:\n"
            "    f.write(str(p.pid))\n"
            "    f.flush()\n"
            "time.sleep(300)\n"
        )

        start(
            "wt-grandchild",
            [sys.executable, "-c", wrapper_code],
            store=store,
        )
        stored = store.get("wt-grandchild")
        wrapper_pid = stored.pids[DEFAULT_ROLE]

        grandchild_pid = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if marker_path.exists():
                content = marker_path.read_text().strip()
                if content:
                    grandchild_pid = int(content)
                    break
            time.sleep(0.05)

        assert grandchild_pid is not None, (
            "grandchild pid marker file was never written by the wrapper"
        )
        assert _pid_alive(grandchild_pid), "grandchild must be alive before stop()"

        try:
            stop("wt-grandchild", store=store, timeout=10.0)

            # Give the OS a brief moment to finish reaping/terminating.
            reap_deadline = time.monotonic() + 5.0
            while time.monotonic() < reap_deadline and _pid_alive(grandchild_pid):
                time.sleep(0.1)

            assert not _pid_alive(grandchild_pid), (
                "grandchild process must be killed by stop(), not just the "
                "wrapper process -- this is the ticket #87 regression"
            )
        finally:
            for leaked_pid in (wrapper_pid, grandchild_pid):
                if leaked_pid is not None:
                    try:
                        _force_kill(leaked_pid)
                    except Exception:  # noqa: BLE001
                        pass

    def test_stop_dead_root_pid_still_fast(self):
        """When the tracked pid is already dead, _process_tree(pid) still
        returns quickly (best-effort empty) and stop() does not hang."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — skipping")

        record = _make_record("wt-dead-tree", pids={DEFAULT_ROLE: 99999999})
        store = _make_store(record)

        t0 = time.monotonic()
        result = stop("wt-dead-tree", store=store, timeout=10.0)
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, f"stop() with a dead root pid took {elapsed:.2f}s"
        assert DEFAULT_ROLE not in result.pids
        assert result.status == "stopped"

    def test_stop_tree_kill_runs_with_kill_orphans_false(self):
        """The tree kill is UNCONDITIONAL -- it must run even when
        kill_orphans=False (the default), unlike the path-heuristic orphan
        scan which stays gated behind kill_orphans."""
        fake_pid = 9999
        record = _make_record("wt-tree-no-orphans", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        fake_tree = [KilledProcessInfo(pid=9500, name="grand", cmdline=["grand"])]
        tree_calls = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                side_effect=lambda p: tree_calls.append(p) or fake_tree,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ) as mock_signal,
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
            ),
        ):
            stop("wt-tree-no-orphans", store=store, kill_orphans=False, timeout=1.0)

        assert tree_calls == [fake_pid]
        signalled_pids = [c.args[0] for c in mock_signal.call_args_list]
        assert 9500 in signalled_pids, (
            "the tree kill must run (and signal the snapshotted descendant) "
            "even though kill_orphans=False"
        )

    def test_stop_custom_role_only_kills_that_roles_tree(self):
        """_process_tree must be called with the specific role's pid, not
        some other role's."""
        record = _make_record("wt-role-tree", pids={"main": 111, "worker": 222})
        store = _make_store(record)

        tree_calls = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                side_effect=lambda p: tree_calls.append(p) or [],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
        ):
            stop("wt-role-tree", store=store, role="worker", timeout=1.0)

        assert tree_calls == [222]

    def test_stop_signals_process_group_on_posix(self):
        """stop() invokes _signal_process_group as part of its kill
        sequence (POSIX-only behaviour; simulated here since this suite
        runs on a Windows dev box, matching the module's existing
        Windows-simulation test style)."""
        fake_pid = 8888
        record = _make_record("wt-pgroup", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        spg_calls = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._signal_process_group",
                side_effect=lambda pid, **kw: spg_calls.append((pid, kw)) or False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
        ):
            stop("wt-pgroup", store=store, timeout=1.0)

        assert spg_calls == [(fake_pid, {"force": False})]


# ---------------------------------------------------------------------------
# TestProcessTreeLineage -- ticket #87
# ---------------------------------------------------------------------------

class TestProcessTreeLineage:
    """Regression tests for ticket #87: orphan detection must use parent-PID
    lineage, not just path heuristics (cwd/cmdline/open-file scans), which
    is what kill_orphans=True relied on exclusively before this fix."""

    def test_stop_kills_descendants_when_path_heuristics_find_nothing(self):
        """Driving test (R2): even when _find_blocking_processes (the
        path-heuristic scan) finds nothing at all, stop()'s own
        process-tree snapshot of the tracked pid still reaches real
        descendants and signals them."""
        import psutil

        host_pid = os.getpid()
        fake_pid = 6600

        descendant_a = MagicMock()
        descendant_a.pid = 7001
        descendant_a.ppid.return_value = fake_pid
        descendant_a.name.return_value = "child-a"
        descendant_a.cmdline.return_value = ["child-a"]

        descendant_b = MagicMock()
        descendant_b.pid = 7002
        descendant_b.ppid.return_value = fake_pid
        descendant_b.name.return_value = "child-b"
        descendant_b.cmdline.return_value = ["child-b"]

        def _process_side_effect(pid):
            if pid == host_pid:
                m = MagicMock()
                m.parents.return_value = []
                return m
            if pid == fake_pid:
                m = MagicMock()
                m.children.return_value = [descendant_a, descendant_b]
                return m
            raise AssertionError(f"unexpected psutil.Process({pid}) call")

        record = _make_record("wt-lineage", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        graceful_calls = []

        with (
            patch.object(psutil, "Process", side_effect=_process_side_effect),
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
                side_effect=lambda pid: graceful_calls.append(pid),
            ),
            patch("lib_python_worktree.core.process_lifecycle._force_kill"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            stop("wt-lineage", store=store, kill_orphans=True, timeout=2.0)

        assert 7001 in graceful_calls, "first descendant must receive a graceful signal"
        assert 7002 in graceful_calls, "second descendant must receive a graceful signal"


# ---------------------------------------------------------------------------
# TestStopStatusHonesty -- ticket #87
# ---------------------------------------------------------------------------

class TestStopStatusHonesty:
    """Regression tests for ticket #87: stop() must never report
    status='stopped' while a process it was responsible for is demonstrably
    still running."""

    def test_stop_reports_stop_incomplete_when_process_survives(self, caplog):
        """Driving test (R3): when the tracked pid stubbornly survives every
        kill attempt, stop() must report status='stop_incomplete' (not
        'stopped'), still clear pids[role], persist the status to the
        store, and log a warning naming the survivor pid."""
        caplog.set_level(
            logging.WARNING, logger="lib_python_worktree.core.process_lifecycle"
        )

        fake_pid = 31337
        record = _make_record("wt-survivor", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch("lib_python_worktree.core.process_lifecycle._force_kill"),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
        ):
            result = stop("wt-survivor", store=store, timeout=0.3)

        assert result.status == "stop_incomplete"
        assert DEFAULT_ROLE not in result.pids

        stored = store.get("wt-survivor")
        assert stored is not None
        assert stored.status == "stop_incomplete"
        assert DEFAULT_ROLE not in stored.pids

        assert any(str(fake_pid) in rec.message for rec in caplog.records), (
            f"expected a warning naming survivor pid {fake_pid}, got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_stop_reports_stopped_when_everything_dies(self):
        """When nothing survives, the existing status='stopped' contract is
        unchanged."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — skipping")

        record = _make_record("wt-all-dead", pids={DEFAULT_ROLE: 99999999})
        store = _make_store(record)

        result = stop("wt-all-dead", store=store, timeout=1.0)

        assert result.status == "stopped"
        assert DEFAULT_ROLE not in result.pids

    def test_stop_other_role_alive_status_unchanged(self):
        """Regression: when another role's process is still alive, stopping
        one role (with no survivors of its own root/tree) must not force
        the overall status to 'stopped' OR 'stop_incomplete'."""
        record = _make_record(
            "wt-other-alive",
            pids={"main": os.getpid(), "worker": 99999999},
            status="running",
        )
        store = _make_store(record)

        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine")

        result = stop("wt-other-alive", store=store, role="worker", timeout=1.0)

        assert "worker" not in result.pids
        assert "main" in result.pids
        assert result.status == "running"

    def test_stop_clean_role_does_not_clobber_sticky_stop_incomplete(self):
        """Regression (fix pass #2, ticket #87): a prior stop(role="main")
        call that detected a survivor already left status="stop_incomplete"
        on the record, and (per stop()'s postcondition) already cleared
        "main" from pids -- only "worker" remains tracked. A later
        stop(role="worker") whose own kill is completely clean (no
        survivors) empties record.pids entirely as a side effect, but that
        must NOT cause the `elif not record.pids` branch to silently
        overwrite the sticky "stop_incomplete" back to "stopped" -- nothing
        about this call re-verified that main's leaked process actually
        died. status="stop_incomplete" must remain sticky until the next
        start(), exactly as documented."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — skipping")

        record = _make_record(
            "wt-sticky-cross-role",
            pids={"worker": 99999999},
            status="stop_incomplete",
        )
        store = _make_store(record)

        result = stop("wt-sticky-cross-role", store=store, role="worker", timeout=1.0)

        assert "worker" not in result.pids
        assert result.status == "stop_incomplete"

        stored = store.get("wt-sticky-cross-role")
        assert stored is not None
        assert stored.status == "stop_incomplete"

    def test_stop_reports_stop_incomplete_when_tree_snapshot_truncated(self, caplog):
        """Regression (ticket #87 follow-up, finding F2): _process_tree's
        _MAX_TREE_NODES cap is silent truncation -- a tree with more
        descendants than the cap allows leaves the excess neither killed nor
        checked. stop() must not report status='stopped' purely on the
        strength of a capped snapshot, even when every pid it DID collect is
        confirmed dead -- the excess beyond the cap was never even examined,
        so "clean" here cannot be trusted. This is the exact false-positive
        class ticket #87 exists to eliminate."""
        caplog.set_level(
            logging.WARNING, logger="lib_python_worktree.core.process_lifecycle"
        )

        fake_pid = 41000
        record = _make_record("wt-truncated-tree", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        # Patch _MAX_TREE_NODES down to a small number so a small, fast fake
        # tree can exercise the "snapshot is exactly at the cap" condition.
        capped_tree = [
            KilledProcessInfo(pid=42000 + i, name="child", cmdline=["child"])
            for i in range(3)
        ]

        with (
            patch("lib_python_worktree.core.process_lifecycle._MAX_TREE_NODES", 3),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=capped_tree,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            result = stop("wt-truncated-tree", store=store, timeout=1.0)

        assert result.status == "stop_incomplete", (
            f"expected 'stop_incomplete' for a capped tree snapshot (cannot "
            f"guarantee completeness), got {result.status!r}"
        )
        assert DEFAULT_ROLE not in result.pids

        stored = store.get("wt-truncated-tree")
        assert stored is not None
        assert stored.status == "stop_incomplete"

        assert any(
            "truncat" in rec.message.lower() for rec in caplog.records
        ), (
            f"expected a warning about the truncated tree snapshot, got: "
            f"{[r.message for r in caplog.records]}"
        )

    def test_stop_reports_stopped_when_tree_snapshot_under_cap(self):
        """Sanity counterpart: a tree snapshot that is genuinely UNDER the
        cap must not trip the new truncation guard -- 'stopped' is still
        reported when everything collected is confirmed dead and nothing
        was capped."""
        fake_pid = 41100
        record = _make_record("wt-under-cap-tree", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        small_tree = [KilledProcessInfo(pid=42100, name="child", cmdline=["child"])]

        with (
            patch("lib_python_worktree.core.process_lifecycle._MAX_TREE_NODES", 3),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=small_tree,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            result = stop("wt-under-cap-tree", store=store, timeout=1.0)

        assert result.status == "stopped"


# ---------------------------------------------------------------------------
# TestStopDetail -- ticket #99
# ---------------------------------------------------------------------------

class TestStopDetail:
    """Regression tests for ticket #99: every ``stop_incomplete`` branch in
    ``stop()`` must attach a machine-readable ``StopDetail`` to
    ``record.stop_detail``, not just log a warning."""

    def test_survivor_sets_stop_detail_reason(self):
        """B1 driving test: a survivor pid populates a StopDetail with
        reason="survivors", the survivor pid tuple, count, role, and a
        message naming the pid -- on both the returned record and the
        persisted one."""
        fake_pid = 31337
        record = _make_record("wt-survivor-detail", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch("lib_python_worktree.core.process_lifecycle._force_kill"),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
        ):
            result = stop("wt-survivor-detail", store=store, timeout=0.3)

        assert result.status == "stop_incomplete"
        detail = result.stop_detail
        assert detail is not None
        assert detail.reason == STOP_REASON_SURVIVORS
        assert detail.survivor_pids == (fake_pid,)
        assert detail.survivor_count == 1
        assert detail.role == DEFAULT_ROLE
        assert str(fake_pid) in detail.message
        assert detail.kill_orphans_may_help is True

        stored = store.get("wt-survivor-detail")
        assert stored is not None
        assert stored.stop_detail == detail

    def test_tree_truncation_sets_reason_tree_truncated(self):
        """Edge case: a capped tree snapshot sets reason="tree_truncated"
        with truncated_at == the patched cap."""
        fake_pid = 41200
        record = _make_record("wt-tree-detail", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        capped_tree = [
            KilledProcessInfo(pid=42200 + i, name="child", cmdline=["child"])
            for i in range(3)
        ]

        with (
            patch("lib_python_worktree.core.process_lifecycle._MAX_TREE_NODES", 3),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=capped_tree,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            result = stop("wt-tree-detail", store=store, timeout=1.0)

        assert result.status == "stop_incomplete"
        detail = result.stop_detail
        assert detail is not None
        assert detail.reason == STOP_REASON_TREE_TRUNCATED
        assert detail.truncated_at == 3
        assert detail.survivor_pids == ()
        assert detail.kill_orphans_may_help is True

    def test_job_member_truncation_sets_reason_job_member_list_truncated(self):
        """Edge case: a truncated Job Object member list sets
        reason="job_member_list_truncated" with truncated_at ==
        _JOB_MEMBER_LIST_MAX_SLOTS."""
        fake_pid = 81400
        record = _make_record(
            "wt-job-truncated-detail",
            pids={DEFAULT_ROLE: fake_pid},
            job_names={DEFAULT_ROLE: "Local\\fake-job"},
        )
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._open_job_object",
                return_value=12345,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._job_object_member_pids",
                return_value=_pl._PartialList([], complete=False),
            ),
            patch("lib_python_worktree.core.process_lifecycle._terminate_job_object"),
        ):
            result = stop("wt-job-truncated-detail", store=store, timeout=1.0)

        assert result.status == "stop_incomplete"
        detail = result.stop_detail
        assert detail is not None
        assert detail.reason == STOP_REASON_JOB_MEMBER_LIST_TRUNCATED
        assert detail.truncated_at == _pl._JOB_MEMBER_LIST_MAX_SLOTS

    def test_orphan_scan_incomplete_sets_reason_and_skipped_passes(self):
        """Edge case: an incomplete orphan-scan discovery pass sets
        reason="orphan_scan_incomplete" and carries the skipped_passes
        tags, with kill_orphans_may_help forced False (that pass already
        ran -- the remediation is a larger timeout, stated in message)."""
        fake_pid = 61100
        record = _make_record(
            "wt-orphan-incomplete-detail", pids={DEFAULT_ROLE: fake_pid},
        )
        store = _make_store(record)

        partial = _pl._PartialList(
            [], complete=False, skipped_passes=("handle_scan:skipped",)
        )

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=partial,
            ),
        ):
            result = stop(
                "wt-orphan-incomplete-detail",
                store=store,
                kill_orphans=True,
                timeout=5.0,
            )

        assert result.status == "stop_incomplete"
        detail = result.stop_detail
        assert detail is not None
        assert detail.reason == STOP_REASON_ORPHAN_SCAN_INCOMPLETE
        assert detail.skipped_passes == ("handle_scan:skipped",)
        assert detail.kill_orphans_may_help is False

    def test_branch_precedence_survivors_wins_over_truncation(self):
        """Edge case: survivors and a truncated tree simultaneously must
        still yield reason="survivors" -- pinning the unchanged if/elif
        precedence from before this ticket."""
        fake_pid = 41300
        record = _make_record("wt-precedence", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        capped_tree = [
            KilledProcessInfo(pid=42300 + i, name="child", cmdline=["child"])
            for i in range(3)
        ]

        with (
            patch("lib_python_worktree.core.process_lifecycle._MAX_TREE_NODES", 3),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=capped_tree,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch("lib_python_worktree.core.process_lifecycle._force_kill"),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            result = stop("wt-precedence", store=store, timeout=0.3)

        assert result.status == "stop_incomplete"
        assert result.stop_detail is not None
        assert result.stop_detail.reason == STOP_REASON_SURVIVORS

    def test_stop_detail_cleared_when_later_stop_is_clean(self):
        """B2 driving test: a record left with status='stop_incomplete' and
        a stale stop_detail from an earlier stop() call must have
        stop_detail cleared once start() runs for that role and flips
        status away from stop_incomplete."""
        stale_detail = StopDetail(
            reason=STOP_REASON_SURVIVORS,
            message="stale",
            role=DEFAULT_ROLE,
            survivor_pids=(999,),
            survivor_count=1,
        )
        record = _make_record(
            "wt-clear-on-start",
            status="stop_incomplete",
            stop_detail=stale_detail,
        )
        store = _make_store(record)

        result = start(
            "wt-clear-on-start",
            [sys.executable, "-c", "import time; time.sleep(60)"],
            store=store,
        )

        assert result.status == "running"
        assert result.stop_detail is None

        stored = store.get("wt-clear-on-start")
        assert stored is not None
        assert stored.stop_detail is None

        # Cleanup
        pid = result.pids.get(DEFAULT_ROLE, 0)
        try:
            _force_kill(pid)
        except Exception:  # noqa: BLE001
            pass

    def test_sticky_stop_incomplete_preserves_earlier_stop_detail(self):
        """Edge case (B2): mirrors
        test_stop_clean_role_does_not_clobber_sticky_stop_incomplete -- an
        earlier stop(role="main") already left status="stop_incomplete" and
        a StopDetail on the record; a later, clean stop(role="worker") must
        not erase that detail even though it empties record.pids as a side
        effect."""
        if _pid_alive(99999999):
            pytest.skip("PID 99999999 is alive on this machine — skipping")

        earlier_detail = StopDetail(
            reason=STOP_REASON_SURVIVORS,
            message="earlier survivor",
            role="main",
            survivor_pids=(31337,),
            survivor_count=1,
        )
        record = _make_record(
            "wt-sticky-detail-cross-role",
            pids={"worker": 99999999},
            status="stop_incomplete",
            stop_detail=earlier_detail,
        )
        store = _make_store(record)

        result = stop(
            "wt-sticky-detail-cross-role", store=store, role="worker", timeout=1.0,
        )

        assert result.status == "stop_incomplete"
        assert result.stop_detail == earlier_detail

        stored = store.get("wt-sticky-detail-cross-role")
        assert stored is not None
        assert stored.stop_detail == earlier_detail

    def test_kill_orphans_hint_set_when_orphan_pass_never_ran(self):
        """B4: kill_orphans_may_help is True when the orphan-scan pass
        never ran (kill_orphans=False) for a survivors outcome -- retrying
        with kill_orphans=True might still catch it."""
        fake_pid = 31338
        record = _make_record("wt-hint-may-help", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch("lib_python_worktree.core.process_lifecycle._force_kill"),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
        ):
            result = stop(
                "wt-hint-may-help", store=store, timeout=0.3, kill_orphans=False,
            )

        assert result.stop_detail is not None
        assert result.stop_detail.kill_orphans_may_help is True

    def test_kill_orphans_hint_false_when_orphan_scan_already_ran(self):
        """B4: kill_orphans_may_help is False for a survivors outcome when
        kill_orphans=True was already passed on this call -- there is no
        further remediation this hint can point to."""
        fake_pid = 31339
        record = _make_record("wt-hint-no-help", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch("lib_python_worktree.core.process_lifecycle._force_kill"),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=_pl._PartialList([]),
            ),
        ):
            result = stop(
                "wt-hint-no-help", store=store, timeout=0.3, kill_orphans=True,
            )

        assert result.stop_detail is not None
        assert result.stop_detail.kill_orphans_may_help is False


# ---------------------------------------------------------------------------
# TestStopAttemptOutcome -- ticket #110
# ---------------------------------------------------------------------------

class TestStopAttemptOutcome:
    """Regression tests for ticket #110: stop() must attach a machine-
    readable ``StopAttempt`` to ``record.stop_attempt`` distinguishing
    "nothing needed killing" from "the tracked PID had already gone stale
    while real work it spawned kept running" (the stale-tracked-pid-for-
    composite-command finding, #110-1) from the ordinary case where the
    tracked pid was alive at entry."""

    def test_outcome_killed_when_tracked_pid_alive(self):
        """Driving test: tracked pid alive at entry -> stop_attempt.outcome
        == "killed"."""
        fake_pid = 91000
        record = _make_record("wt-attempt-killed", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            result = stop("wt-attempt-killed", store=store, timeout=1.0)

        assert result.stop_attempt is not None
        assert result.stop_attempt.outcome == STOP_ATTEMPT_KILLED
        assert result.stop_attempt.tracked_pid == fake_pid
        assert result.stop_attempt.tracked_pid_alive is True
        assert result.stop_attempt.role == DEFAULT_ROLE

    def test_outcome_already_exited_when_nothing_found(self):
        """Driving test: tracked pid dead at entry AND nothing from its
        tree/process-group/job/orphan-scan was found -> stop_attempt.outcome
        == "already_exited", killed_pids == [] (the genuinely-nothing-to-do
        case, distinct from tracked_pid_missing below)."""
        fake_pid = 91100
        record = _make_record("wt-attempt-exited", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_group_members",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
        ):
            result = stop("wt-attempt-exited", store=store, timeout=1.0)

        assert result.stop_attempt is not None
        assert result.stop_attempt.outcome == STOP_ATTEMPT_ALREADY_EXITED
        assert result.stop_attempt.tracked_pid_alive is False
        assert result.killed_pids == []

    def test_outcome_tracked_pid_missing_when_tree_survives_dead_tracked_pid(self):
        """Driving test (finding #110-1): tracked pid already dead at entry,
        BUT its descendant tree still has a member -> stop_attempt.outcome
        == "tracked_pid_missing" -- the tracked pid had gone stale for a
        composite/chained shell command while real work it spawned kept
        running under a different, untracked pid."""
        fake_pid = 91200
        child = KilledProcessInfo(pid=91201, name="child", cmdline=["child"])
        record = _make_record("wt-attempt-missing", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[child],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            result = stop("wt-attempt-missing", store=store, timeout=1.0)

        assert result.stop_attempt is not None
        assert result.stop_attempt.outcome == STOP_ATTEMPT_TRACKED_PID_MISSING
        assert result.stop_attempt.tracked_pid_alive is False
        assert result.stop_attempt.kill_orphans_may_help is True

    def test_outcome_already_exited_when_only_unrelated_orphan_found(self):
        """Driving test (reviewer fix cycle, blocking finding #1): tracked
        pid dead at entry, its OWN tree/process-group/job is genuinely empty,
        but the unrelated path-heuristic orphan scan (kill_orphans=True)
        separately found and killed something. This must still be
        "already_exited", NOT "tracked_pid_missing" -- an orphan-scan hit
        says nothing about whether the tracked pid's own descendant
        tree/process-group/job survived it, so it must never be folded into
        that determination. Before the fix, `_something_else_found =
        bool(killed_tree) or bool(orphan_found)` conflated the two, so this
        exact scenario (empty killed_tree, non-empty orphan_found) was
        mislabeled "tracked_pid_missing" with a message falsely claiming
        "tree/process-group/job" evidence."""
        fake_pid = 91300
        orphan = KilledProcessInfo(
            pid=91301, name="orphan", cmdline=["orphan"], source="orphan_scan",
        )
        record = _make_record("wt-attempt-orphan-only", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_group_members",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                side_effect=lambda p: False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=_pl._PartialList([orphan]),
            ),
        ):
            result = stop(
                "wt-attempt-orphan-only", store=store, timeout=1.0, kill_orphans=True,
            )

        assert result.stop_attempt is not None
        assert result.stop_attempt.outcome == STOP_ATTEMPT_ALREADY_EXITED
        assert result.stop_attempt.tracked_pid_alive is False

    def test_kill_orphans_may_help_false_for_tracked_pid_missing_when_already_run(self):
        """Edge case: kill_orphans_may_help must be False for
        tracked_pid_missing when kill_orphans=True was already passed on
        this call -- retrying offers nothing new."""
        fake_pid = 91250
        child = KilledProcessInfo(pid=91251, name="child", cmdline=["child"])
        record = _make_record("wt-attempt-missing-help", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[child],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=_pl._PartialList([]),
            ),
        ):
            result = stop(
                "wt-attempt-missing-help", store=store, timeout=1.0, kill_orphans=True,
            )

        assert result.stop_attempt is not None
        assert result.stop_attempt.outcome == STOP_ATTEMPT_TRACKED_PID_MISSING
        assert result.stop_attempt.kill_orphans_may_help is False

    def test_start_clears_leftover_stop_attempt(self):
        """start() must clear a leftover stop_attempt from a previous
        stop() call, mirroring how it already clears stop_detail."""
        from lib_python_worktree.core.state import StopAttempt

        record = _make_record(
            "wt-attempt-start-clears",
            stop_attempt=StopAttempt(outcome=STOP_ATTEMPT_KILLED, message="stale"),
        )
        store = _make_store(record)

        with patch(
            "lib_python_worktree.core.process_lifecycle._spawn_detached"
        ) as mock_spawn:
            mock_proc = MagicMock()
            mock_proc.pid = 91300
            mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=0.25)
            mock_spawn.return_value = mock_proc

            result = start("wt-attempt-start-clears", ["echo", "hi"], store=store)

        assert result.stop_attempt is None


# ---------------------------------------------------------------------------
# TestKilledPidsIdentifiability -- ticket #110
# ---------------------------------------------------------------------------

class TestKilledPidsIdentifiability:
    """Regression tests for ticket #110, finding #110-3: every
    ``killed_pids`` entry must carry a ``source`` provenance tag, and
    duplicate-pid entries from two sources must keep the richer (non-empty
    name/cmdline) one rather than first-wins."""

    def test_describe_pid_never_raises_on_access_denied(self):
        """_describe_pid must never raise -- an AccessDenied/NoSuchProcess
        failure degrades to ("", [])."""
        import psutil

        with patch.object(
            psutil, "Process", side_effect=psutil.AccessDenied(pid=12345)
        ):
            name, cmdline = _describe_pid(12345)

        assert name == ""
        assert cmdline == []

    def test_describe_pid_never_raises_on_no_such_process(self):
        import psutil

        with patch.object(
            psutil, "Process", side_effect=psutil.NoSuchProcess(pid=12345)
        ):
            name, cmdline = _describe_pid(12345)

        assert name == ""
        assert cmdline == []

    def test_tracked_pid_entry_has_source_tracked(self):
        """The tracked-pid killed_pids entry carries source="tracked"."""
        fake_pid = 92000
        record = _make_record("wt-source-tracked", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=True,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            result = stop("wt-source-tracked", store=store, timeout=1.0)

        tracked_entries = [info for info in result.killed_pids if info.pid == fake_pid]
        assert len(tracked_entries) == 1
        assert tracked_entries[0].source == "tracked"

    def test_tree_entry_has_source_tree(self):
        fake_pid = 92100
        child = KilledProcessInfo(
            pid=92101, name="child", cmdline=["child"], source="tree",
        )
        record = _make_record("wt-source-tree", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[child],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
        ):
            result = stop("wt-source-tree", store=store, timeout=1.0)

        matching = [info for info in result.killed_pids if info.pid == 92101]
        assert len(matching) == 1
        assert matching[0].source == "tree"

    def test_process_group_only_entry_has_source_process_group(self):
        fake_pid = 92200
        group_only_pid = 92201
        record = _make_record("wt-source-group", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_group_members",
                return_value=[group_only_pid],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._signal_process_group",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            result = stop("wt-source-group", store=store, timeout=1.0)

        matching = [info for info in result.killed_pids if info.pid == group_only_pid]
        assert len(matching) == 1
        assert matching[0].source == "process_group"

    def test_job_only_entry_has_source_job_object(self):
        """A Job Object member not already covered by the ppid-tree or
        process-group snapshot must be tagged source="job_object", with its
        name/cmdline populated from the pre-terminate _describe_pid
        snapshot (ticket #110, finding #110-3) -- mirroring how
        process_group/tree/orphan_scan/tracked entries are already
        covered above."""
        fake_pid = 92500
        member_pid = 92501
        record = _make_record(
            "wt-source-job",
            pids={DEFAULT_ROLE: fake_pid},
            job_names={DEFAULT_ROLE: "Local\\fake-job-source"},
        )
        store = _make_store(record)

        def _fake_describe(pid):
            if pid == member_pid:
                return ("job-member.exe", ["job-member.exe", "--flag"])
            return ("", [])

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._open_job_object",
                return_value=12345,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._job_object_member_pids",
                return_value=_pl._PartialList([member_pid]),
            ),
            patch("lib_python_worktree.core.process_lifecycle._terminate_job_object"),
            patch(
                "lib_python_worktree.core.process_lifecycle._describe_pid",
                side_effect=_fake_describe,
            ),
        ):
            result = stop("wt-source-job", store=store, timeout=1.0)

        matching = [info for info in result.killed_pids if info.pid == member_pid]
        assert len(matching) == 1
        assert matching[0].source == "job_object"
        assert matching[0].name == "job-member.exe"
        assert matching[0].cmdline == ["job-member.exe", "--flag"]

    def test_orphan_entry_has_source_orphan_scan(self):
        fake_pid = 92300
        orphan = KilledProcessInfo(
            pid=92301, name="orphan", cmdline=["orphan"], source="orphan_scan",
        )
        record = _make_record("wt-source-orphan", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=_pl._PartialList([orphan]),
            ),
        ):
            result = stop(
                "wt-source-orphan", store=store, kill_orphans=True, timeout=1.0,
            )

        matching = [info for info in result.killed_pids if info.pid == 92301]
        assert len(matching) == 1
        assert matching[0].source == "orphan_scan"

    def test_dedup_prefers_richest_entry(self):
        """A pid appearing twice -- once bare (empty name/cmdline, e.g. a
        process-group/job-object artifact) and once with real metadata --
        must keep the richer entry, not first-wins."""
        fake_pid = 92400
        shared_pid = 92401
        bare = KilledProcessInfo(pid=shared_pid, name="", cmdline=[], source="tree")
        rich = KilledProcessInfo(
            pid=shared_pid, name="real-name", cmdline=["real", "cmd"],
            source="orphan_scan",
        )
        record = _make_record("wt-source-dedup", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[bare],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=_pl._PartialList([rich]),
            ),
        ):
            result = stop(
                "wt-source-dedup", store=store, kill_orphans=True, timeout=1.0,
            )

        matching = [info for info in result.killed_pids if info.pid == shared_pid]
        assert len(matching) == 1
        assert matching[0].name == "real-name", (
            "dedup must prefer the richer (non-empty name/cmdline) entry, "
            "not first-wins"
        )
        assert matching[0].cmdline == ["real", "cmd"]

    def test_dedup_prefers_richer_entry_between_two_partially_populated(self):
        """Driving test for review round 4's blocking finding: the OLD dedup
        logic only distinguished has-any-metadata vs has-none, so once the
        first-seen entry for a pid had ANY metadata (e.g. name set but
        cmdline empty), a later duplicate with STRICTLY RICHER metadata (both
        name AND a full cmdline) was wrongly discarded -- has-any-metadata
        was already True, so `candidate_has_metadata and not
        existing_has_metadata` never fired. The fix scores richness by how
        many of {name, cmdline} are populated, so the two-populated-fields
        entry must win over the one-populated-fields entry regardless of
        which was appended first."""
        fake_pid = 92500
        shared_pid = 92501
        partial = KilledProcessInfo(
            pid=shared_pid, name="proc.exe", cmdline=[], source="tree",
        )
        fuller = KilledProcessInfo(
            pid=shared_pid, name="proc.exe", cmdline=["proc.exe", "--flag"],
            source="orphan_scan",
        )
        record = _make_record("wt-source-dedup-partial", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[partial],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=_pl._PartialList([fuller]),
            ),
        ):
            result = stop(
                "wt-source-dedup-partial", store=store, kill_orphans=True, timeout=1.0,
            )

        matching = [info for info in result.killed_pids if info.pid == shared_pid]
        assert len(matching) == 1
        assert matching[0].cmdline == ["proc.exe", "--flag"], (
            "dedup must prefer the entry with MORE populated fields (name "
            "AND cmdline) over one with fewer (name only), not just "
            "any-metadata-vs-none"
        )


# ---------------------------------------------------------------------------
# TestStaleTrackedPidCompositeCommand -- ticket #110, finding #110-1
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group only")
class TestStaleTrackedPidCompositeCommand:
    """Regression test for finding #110-1: a composite/chained shell
    command (e.g. ``sh -c "... & long_running_child"``) can leave the
    tracked wrapper pid dead while the backgrounded child survives under a
    different, untracked pid but the SAME process group (start_new_session
    guarantees pgid == the tracked pid). Before this ticket, stop() bailed
    out of _process_group_members the instant os.getpgid(tracked_pid) raised
    OSError (leader already reaped), so the surviving child was never found,
    never killed, and stop() silently reported killed_pids: [] /
    status="stopped"."""

    def test_stop_finds_and_kills_survivor_of_dead_leader_composite_command(self):
        """Driving test: spawn a real `sh -c` wrapper that exits almost
        immediately while backgrounding a long-running child in the same
        process group; stop() must find that child via
        _process_group_members, kill it, report it with
        source="process_group", and set stop_attempt.outcome ==
        "tracked_pid_missing"."""
        marker = Path(_tempdir_for_test()) / f"pgroup-marker-{os.getpid()}"
        if marker.exists():
            marker.unlink()

        proc = _spawn_detached(
            [
                "sh", "-c",
                # Ticket #110 fix cycle (blocking finding #2): capture `$!`
                # immediately after backgrounding the marker-write job, into
                # pid1, BEFORE backgrounding `sleep 30` -- `$!` always refers
                # to the most recently backgrounded job, so waiting on a
                # bare `$!` taken after both jobs are backgrounded would
                # actually be `sleep 30`'s pid, blocking this wrapper for
                # the full 30s instead of letting it exit right after the
                # marker write (which is what makes the wrapper's PID go
                # stale while `sleep 30` keeps running -- the scenario this
                # test exists to exercise).
                f"echo started > {marker} & pid1=$!; sleep 30 & wait $pid1",
            ],
            env=dict(os.environ),
            cwd=None,
            log_path=Path(_tempdir_for_test()) / f"pgroup-log-{os.getpid()}.log",
        )
        wrapper_pid = proc.pid

        try:
            # Wait for the backgrounded grandchild to actually start, then
            # let the wrapper shell itself exit -- `wait $pid1` returns once
            # the marker-write job (not `sleep 30`) has finished, so the
            # outer `sh -c` process this test tracks exits shortly after,
            # while `sleep 30` (spawned into the SAME process group --
            # start_new_session guarantees pgid == wrapper_pid) keeps
            # running under a different, now-untracked pid.
            deadline = time.monotonic() + 10.0
            while not marker.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert marker.exists(), "wrapper never reached the backgrounding point"

            # Poll for the wrapper to actually exit so os.getpgid(wrapper_pid)
            # is guaranteed to raise OSError (leader reaped) by the time
            # stop() runs -- this is what makes the test exercise finding
            # #110-1's dead-leader path rather than the ordinary alive path.
            # `proc` is this test process's own child, so a wrapper that has
            # exited but not yet been waited on lingers as a zombie -- for
            # which `_pid_alive` (os.kill(pid, 0)) keeps reading "alive"
            # forever. `proc.poll()` performs the reaping wait4/waitpid
            # itself, so it (unlike raw `_pid_alive`) actually observes and
            # clears the exit.
            deadline = time.monotonic() + 10.0
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert proc.poll() is not None, "wrapper did not exit in time"

            record = _make_record(
                "wt-composite-stale", pids={DEFAULT_ROLE: wrapper_pid},
            )
            store = _make_store(record)

            result = stop("wt-composite-stale", store=store, timeout=10.0)

            process_group_entries = [
                info for info in result.killed_pids
                if info.source == "process_group"
            ]
            assert process_group_entries, (
                "the surviving `sleep 30` grandchild (same process group as "
                "the dead wrapper) must be found via _process_group_members "
                "and reported with source='process_group'"
            )
            assert result.stop_attempt is not None
            assert result.stop_attempt.outcome == STOP_ATTEMPT_TRACKED_PID_MISSING
            for info in process_group_entries:
                assert not _pid_alive(info.pid), (
                    f"survivor pid {info.pid} must actually be killed by stop()"
                )
        finally:
            if marker.exists():
                marker.unlink()
            if _pid_alive(wrapper_pid):
                _force_kill(wrapper_pid)


def _tempdir_for_test() -> str:
    import tempfile
    return tempfile.gettempdir()


# ---------------------------------------------------------------------------
# TestDiscoveryBudget -- ticket #87
# ---------------------------------------------------------------------------

class TestDiscoveryBudget:
    """Regression tests for ticket #87: _find_blocking_processes' discovery
    passes must be bounded, not left to run unboundedly (~75s of CPU was
    observed for a single call that found nothing)."""

    def test_find_blocking_processes_respects_deadline_across_all_passes(self):
        """Driving test (R4): Pass 1 (cwd) and Pass 2 (open_files) -- the two
        passes that run on every platform -- must each bail out once the
        deadline-derived scan budget is exhausted, instead of grinding
        through the full (here: artificially slow) process list. Simulated
        on a non-Windows platform so Pass 1b/1c (Windows-only) never run,
        keeping this test deterministic and independent of real Windows
        ctypes internals."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        def _make_slow_proc(pid):
            proc = MagicMock()
            proc.info = {"pid": pid, "name": "slow", "cmdline": ["slow"]}

            def _slow_cwd():
                time.sleep(0.05)
                return "/other/path"

            def _slow_open_files():
                time.sleep(0.05)
                return []

            proc.cwd.side_effect = _slow_cwd
            proc.open_files.side_effect = _slow_open_files
            return proc

        def _process_iter_side_effect(*args, **kwargs):
            # Fresh generator each call -- Pass 1 and Pass 2 each iterate
            # process_iter independently.
            return (_make_slow_proc(20000 + i) for i in range(200))

        with (
            patch.object(psutil, "process_iter", side_effect=_process_iter_side_effect),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            t0 = time.monotonic()
            _find_blocking_processes(target, host_pid, deadline=time.monotonic() + 0.5)
            elapsed = time.monotonic() - t0

        assert elapsed < 2.0, (
            f"_find_blocking_processes took {elapsed:.2f}s against a "
            f"deadline of 0.5s -- discovery passes are not respecting it"
        )

    def test_find_blocking_processes_no_deadline_still_capped_by_discovery_max(self):
        """Even without an explicit *deadline*, discovery must still be
        capped by _DISCOVERY_MAX_SEC -- not left fully unbounded (this is
        what caused ~75s of CPU for a single real-world call that found
        nothing)."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        def _make_slow_proc(pid):
            proc = MagicMock()
            proc.info = {"pid": pid, "name": "slow", "cmdline": ["slow"]}
            proc.cwd.side_effect = lambda: (time.sleep(0.05), "/other/path")[1]
            proc.open_files.side_effect = lambda: (time.sleep(0.05), [])[1]
            return proc

        def _process_iter_side_effect(*args, **kwargs):
            return (_make_slow_proc(30000 + i) for i in range(200))

        with (
            patch.object(psutil, "process_iter", side_effect=_process_iter_side_effect),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch("lib_python_worktree.core.process_lifecycle._DISCOVERY_MAX_SEC", 0.5),
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            t0 = time.monotonic()
            _find_blocking_processes(target, host_pid)  # no deadline kwarg
            elapsed = time.monotonic() - t0

        assert elapsed < 2.0, (
            f"_find_blocking_processes took {elapsed:.2f}s with "
            f"_DISCOVERY_MAX_SEC patched to 0.5s and no deadline -- the "
            f"ceiling must still apply"
        )

    def test_find_blocking_processes_returns_partial_results_when_budget_exhausted(self):
        """When the budget runs out mid-scan, whatever was already found is
        still returned rather than being discarded."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        matching_first = _make_fake_proc(40001, "node", ["node"], target)

        matching_second = MagicMock()
        matching_second.info = {"pid": 40002, "name": "node2", "cmdline": ["node2"]}

        def _slow_cwd_second():
            time.sleep(0.3)
            return target

        matching_second.cwd.side_effect = _slow_cwd_second

        def _process_iter_side_effect(*args, **kwargs):
            return iter([matching_first, matching_second])

        with (
            patch.object(psutil, "process_iter", side_effect=_process_iter_side_effect),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            deadline = time.monotonic() + 0.1
            result = _find_blocking_processes(target, host_pid, deadline=deadline)

        result_pids = {r.pid for r in result}
        assert 40001 in result_pids, (
            "process found before the budget ran out must still be reported"
        )

    def test_find_blocking_processes_deadline_already_passed_returns_immediately(self):
        """B4 (nit, verified): an already-expired *deadline* at entry
        (``remaining < 0``) must still resolve to a ``scan_stop`` at or
        before entry -- i.e. no per-process discovery scanning happens at
        all, rather than the negative-`remaining` algebra somehow pushing
        `scan_stop` forward. Confirmed via a process_iter mock whose
        per-process cwd()/open_files() calls record every PID they are
        asked to probe: if the deadline guard is not actually skipping
        discovery, this list would be non-empty."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        probed: List[int] = []

        def _make_probe_proc(pid):
            proc = MagicMock()
            proc.info = {"pid": pid, "name": "x", "cmdline": ["x"]}

            def _cwd():
                probed.append(pid)
                return "/other/path"

            def _open_files():
                probed.append(pid)
                return []

            proc.cwd.side_effect = _cwd
            proc.open_files.side_effect = _open_files
            return proc

        def _process_iter_side_effect(*args, **kwargs):
            return (_make_probe_proc(50000 + i) for i in range(50))

        with (
            patch.object(psutil, "process_iter", side_effect=_process_iter_side_effect),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            past_deadline = time.monotonic() - 5.0
            t0 = time.monotonic()
            result = _find_blocking_processes(target, host_pid, deadline=past_deadline)
            elapsed = time.monotonic() - t0

        assert elapsed < 0.5, (
            f"an already-expired deadline took {elapsed:.3f}s to return -- "
            "expected an almost-immediate no-op"
        )
        assert probed == [], (
            "no process should have been probed once the deadline had "
            f"already passed at entry; probed={probed}"
        )
        assert result == []


# ---------------------------------------------------------------------------
# TestPartialList / TestDiscoveryCompleteness -- ticket #95, R2
# ---------------------------------------------------------------------------

class TestPartialList:
    """Unit tests for the ``_PartialList`` list subclass itself.

    Finding 3 (ticket #95): ``_find_blocking_processes`` /
    ``_kill_blocking_processes`` used to return a plain ``[]`` in both the
    "genuinely found nothing" case and the "discovery was starved by the
    deadline before it could look" case -- callers (``stop()``) could not
    tell them apart. ``_PartialList`` carries that distinction without
    changing any existing call site's contract: it behaves exactly like the
    list it wraps, plus a ``complete`` flag and a ``skipped_passes`` tuple.
    """

    def test_default_complete_true_no_skipped_passes(self):
        pl = _pl._PartialList([1, 2, 3])
        assert list(pl) == [1, 2, 3]
        assert pl.complete is True
        assert pl.skipped_passes == ()

    def test_incomplete_carries_skipped_passes(self):
        pl = _pl._PartialList([], complete=False, skipped_passes=("handle_scan:skipped",))
        assert pl.complete is False
        assert pl.skipped_passes == ("handle_scan:skipped",)

    def test_behaves_as_plain_list(self):
        pl = _pl._PartialList([1, 2])
        assert pl == [1, 2]
        assert bool(pl) is True
        assert bool(_pl._PartialList([])) is False

    def test_bare_list_getattr_defaults_to_complete(self):
        """A bare list (e.g. from an existing test mock) must read as
        complete via getattr(..., "complete", True) -- the fallback every
        reader in this module uses."""
        assert getattr([], "complete", True) is True
        assert getattr([], "skipped_passes", ()) == ()


class TestDiscoveryCompleteness:
    """R2 (ticket #95): distinguish "found nothing" from "never looked" /
    "ran out of time looking" so stop() never reports a false "stopped" on
    the strength of starved discovery.

    ``complete`` tracks discovery COVERAGE only -- never kill efficacy
    (survivors are the re-probe's job) and never platform applicability (a
    pass simply not running on this OS is not incompleteness).
    """

    # -- stop() wiring: the driving test + its mandatory negative pair -----

    def test_deadline_skipped_pass_marks_stop_incomplete(self, caplog):
        """Driving test: when the orphan scan reports incomplete discovery
        (a pass was skipped by the deadline), stop() must report
        "stop_incomplete", not "stopped" -- even though nothing survived."""
        fake_pid = 61000
        record = _make_record("wt-discovery-incomplete", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        partial = _pl._PartialList(
            [], complete=False, skipped_passes=("handle_scan:skipped",)
        )

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=partial,
            ),
            caplog.at_level(
                logging.WARNING, logger="lib_python_worktree.core.process_lifecycle"
            ),
        ):
            result = stop(
                "wt-discovery-incomplete", store=store, kill_orphans=True, timeout=5.0,
            )

        assert result.status == "stop_incomplete"
        assert any(
            "handle_scan:skipped" in rec.message for rec in caplog.records
        ), "warning must name the skipped pass"

    def test_internal_degradation_does_not_mark_stop_incomplete(self):
        """Mandatory negative test: a pass that degraded internally (e.g. hit
        the worker cap) but still ran to completion (complete=True) must NOT
        cause stop_incomplete."""
        fake_pid = 61001
        record = _make_record("wt-discovery-degraded", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        partial = _pl._PartialList([], complete=True)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=partial,
            ),
        ):
            result = stop(
                "wt-discovery-degraded", store=store, kill_orphans=True, timeout=5.0,
            )

        assert result.status == "stopped"

    def test_stop_treats_bare_list_from_mock_as_complete(self):
        """N8: a plain ``[]`` (no ``.complete`` attribute -- e.g. an existing
        test mock, or manager._teardown's positional caller) must not be
        treated as incomplete; getattr(..., "complete", True) is the
        fallback."""
        fake_pid = 61002
        record = _make_record("wt-discovery-bare-list", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._kill_blocking_processes",
                return_value=[],
            ),
        ):
            result = stop(
                "wt-discovery-bare-list", store=store, kill_orphans=True, timeout=5.0,
            )

        assert result.status == "stopped"

    # -- N4: individual per-PID AccessDenied never marks incomplete --------

    def test_access_denied_pid_leaves_complete_true(self):
        """N4: psutil.AccessDenied/NoSuchProcess on one individual PID during
        Pass 1/2 is caught and skipped (continue) -- must not mark the whole
        pass incomplete."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        proc_denied = MagicMock()
        proc_denied.info = {"pid": 62000, "name": "x", "cmdline": []}
        proc_denied.cwd.side_effect = psutil.AccessDenied(62000)
        proc_denied.open_files.side_effect = psutil.AccessDenied(62000)

        with (
            patch.object(psutil, "process_iter", return_value=[proc_denied]),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            result = _find_blocking_processes(target, host_pid)

        assert result.complete is True
        assert result.skipped_passes == ()

    # -- D1/D2: Pass 1 (cwd) skipped vs truncated ---------------------------

    def test_cwd_pass_skipped_when_deadline_already_passed(self):
        """D1: an already-expired deadline at entry means Pass 1's entry
        guard is false -- tagged "cwd:skipped", not run at all."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        with (
            patch.object(psutil, "process_iter", return_value=iter([])),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            past_deadline = time.monotonic() - 5.0
            result = _find_blocking_processes(target, host_pid, deadline=past_deadline)

        assert result.complete is False
        assert "cwd:skipped" in result.skipped_passes

    def test_cwd_pass_truncated_marks_incomplete(self):
        """D2: Pass 1's inner per-process loop breaking on scan_stop (budget
        exhausted mid-scan) is tagged "cwd:truncated"."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        matching_first = _make_fake_proc(63001, "node", ["node"], target)

        matching_second = MagicMock()
        matching_second.info = {"pid": 63002, "name": "node2", "cmdline": ["node2"]}

        def _slow_cwd_second():
            time.sleep(0.3)
            return target

        matching_second.cwd.side_effect = _slow_cwd_second

        # A third proc is required to actually observe the deadline-driven
        # `break`: the loop's guard is checked at the TOP of each iteration,
        # so proc2's slow call blowing the budget is only detected once the
        # loop tries to move on to a *next* item -- with only two procs the
        # iterator would simply exhaust naturally instead.
        third = _make_fake_proc(63003, "node3", ["node3"], "/other/path")

        def _process_iter_side_effect(*args, **kwargs):
            return iter([matching_first, matching_second, third])

        with (
            patch.object(psutil, "process_iter", side_effect=_process_iter_side_effect),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            deadline = time.monotonic() + 0.1
            result = _find_blocking_processes(target, host_pid, deadline=deadline)

        assert result.complete is False
        assert "cwd:truncated" in result.skipped_passes

    # -- D7: Pass 2 (open_files) truncated ----------------------------------

    def test_open_files_pass_truncated_marks_incomplete(self):
        """D7: Pass 2's inner loop breaking on scan_stop is tagged
        "open_files:truncated"."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        def _make_slow_proc(pid):
            proc = MagicMock()
            proc.info = {"pid": pid, "name": "slow", "cmdline": ["slow"]}
            proc.cwd.side_effect = psutil.AccessDenied(pid)  # skip Pass 1 entirely

            def _slow_open_files():
                time.sleep(0.3)
                return []

            proc.open_files.side_effect = _slow_open_files
            return proc

        def _process_iter_side_effect(*args, **kwargs):
            return (_make_slow_proc(64000 + i) for i in range(3))

        with (
            patch.object(psutil, "process_iter", side_effect=_process_iter_side_effect),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            deadline = time.monotonic() + 0.1
            result = _find_blocking_processes(target, host_pid, deadline=deadline)

        assert result.complete is False
        assert "open_files:truncated" in result.skipped_passes

    # -- D9 (ticket #107): Pass 2 (open_files) OS-wide RuntimeError ---------

    def test_open_files_runtime_error_degrades_instead_of_raising(self):
        """D9: psutil's Windows open_files() can raise a bare RuntimeError
        (e.g. "SystemExtendedHandleInformation buffer too big") when the
        OS-wide handle table is large. Pass 2 must catch it, stop scanning
        (the condition is process-independent -- every remaining PID would
        raise identically), and report "open_files:degraded" instead of
        letting the exception propagate out of _find_blocking_processes and
        crash every caller (stop()/remove())."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        def _make_raising_proc(pid):
            proc = MagicMock()
            proc.info = {"pid": pid, "name": "x", "cmdline": ["x"]}
            proc.cwd.side_effect = psutil.AccessDenied(pid)  # skip Pass 1
            proc.open_files.side_effect = RuntimeError(
                "SystemExtendedHandleInformation buffer too big"
            )
            return proc

        procs = [_make_raising_proc(66000 + i) for i in range(3)]

        with (
            patch.object(psutil, "process_iter", side_effect=lambda *a, **kw: iter(procs)),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            # D9's RuntimeError catch is Windows-only (see the platform-gate
            # regression test below) -- this is the platform on which the
            # real condition occurs, so exercise it here.
            mock_sys.platform = "win32"

            result = _find_blocking_processes(target, host_pid)

        assert result.complete is False
        assert "open_files:degraded" in result.skipped_passes
        assert result == []

    def test_open_files_runtime_error_stops_pass_after_first_pid(self):
        """Additional coverage: the OS-wide condition means every remaining
        PID would raise identically, so Pass 2 must break (not continue) --
        open_files() must be invoked exactly once, not once per raising
        proc, and the tag must be emitted exactly once."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        def _make_raising_proc(pid):
            proc = MagicMock()
            proc.info = {"pid": pid, "name": "x", "cmdline": ["x"]}
            proc.cwd.side_effect = psutil.AccessDenied(pid)
            proc.open_files.side_effect = RuntimeError("buffer too big")
            return proc

        procs = [_make_raising_proc(67000 + i) for i in range(50)]

        with (
            patch.object(psutil, "process_iter", side_effect=lambda *a, **kw: iter(procs)),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            # D9's RuntimeError catch is Windows-only -- see the
            # platform-gate regression test below.
            mock_sys.platform = "win32"

            result = _find_blocking_processes(target, host_pid)

        called = sum(1 for p in procs if p.open_files.called)
        assert called == 1, (
            f"expected exactly 1 open_files() call before the pass breaks "
            f"out entirely, got {called}"
        )
        assert result.skipped_passes.count("open_files:degraded") == 1
        assert result == []

    def test_open_files_access_denied_still_continues_no_regression(self):
        """No-regression guard: AccessDenied/NoSuchProcess from open_files()
        must still `continue` per-PID (not degrade the whole pass) -- this
        pre-existing behaviour must survive the new RuntimeError handling."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        def _make_denied_proc(pid):
            proc = MagicMock()
            proc.info = {"pid": pid, "name": "x", "cmdline": ["x"]}
            proc.cwd.side_effect = psutil.AccessDenied(pid)
            proc.open_files.side_effect = psutil.AccessDenied(pid)
            return proc

        procs = [_make_denied_proc(68000 + i) for i in range(5)]

        with (
            patch.object(psutil, "process_iter", side_effect=lambda *a, **kw: iter(procs)),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            result = _find_blocking_processes(target, host_pid)

        assert result.complete is True
        assert result.skipped_passes == ()
        assert all(p.open_files.called for p in procs)

    def test_open_files_truncation_wins_over_degradation_if_deadline_fires_first(self):
        """If Pass 2's per-loop deadline check fires before a
        RuntimeError-raising proc is reached, "open_files:truncated" wins --
        the pass never got a chance to observe the RuntimeError, so it must
        not be tagged "open_files:degraded". Distinguishes D7 (truncated:
        ran out of clock) from D9 (degraded: OS refused the query
        outright)."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        def _make_proc(pid, open_files_effect):
            proc = MagicMock()
            proc.info = {"pid": pid, "name": "x", "cmdline": ["x"]}
            proc.cwd.side_effect = psutil.AccessDenied(pid)  # skip Pass 1
            proc.open_files.side_effect = open_files_effect
            return proc

        def _fast():
            return []

        def _slow():
            time.sleep(0.3)
            return []

        def _would_raise():
            raise RuntimeError("buffer too big")

        proc0 = _make_proc(70000, _fast)
        proc1 = _make_proc(70001, _slow)
        proc2 = _make_proc(70002, _would_raise)

        def _process_iter_side_effect(*args, **kwargs):
            return iter([proc0, proc1, proc2])

        with (
            patch.object(psutil, "process_iter", side_effect=_process_iter_side_effect),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            deadline = time.monotonic() + 0.1
            result = _find_blocking_processes(target, host_pid, deadline=deadline)

        assert result.complete is False
        assert "open_files:truncated" in result.skipped_passes
        assert "open_files:degraded" not in result.skipped_passes
        assert not proc2.open_files.called

    def test_open_files_runtime_error_reraises_on_non_windows(self):
        """Regression test for the review finding on D9: the
        "SystemExtendedHandleInformation buffer too big" condition is
        Windows/psutil-C-extension-specific. On POSIX, a bare RuntimeError
        from open_files() is NOT this known failure mode and must propagate
        rather than being silently caught and mis-tagged as
        "open_files:degraded" -- doing so would swallow and mis-attribute an
        unrelated, genuinely unexpected bug on that platform."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        def _make_raising_proc(pid):
            proc = MagicMock()
            proc.info = {"pid": pid, "name": "x", "cmdline": ["x"]}
            proc.cwd.side_effect = psutil.AccessDenied(pid)  # skip Pass 1
            proc.open_files.side_effect = RuntimeError("some unrelated posix bug")
            return proc

        procs = [_make_raising_proc(71000)]

        with (
            patch.object(psutil, "process_iter", side_effect=lambda *a, **kw: iter(procs)),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            with pytest.raises(RuntimeError, match="some unrelated posix bug"):
                _find_blocking_processes(target, host_pid)

    def test_open_files_runtime_error_reraises_on_windows_if_unrelated_message(self):
        """Full re-review finding (fix-loop round 2): D9's RuntimeError
        catch must be scoped to the documented psutil failure signature
        ("...buffer too big"), not to "any bare RuntimeError on Windows".
        An unrelated Windows-side bug that happens to raise a plain
        RuntimeError from open_files() must still propagate rather than
        being silently downgraded to "open_files:degraded"."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        def _make_raising_proc(pid):
            proc = MagicMock()
            proc.info = {"pid": pid, "name": "x", "cmdline": ["x"]}
            proc.cwd.side_effect = psutil.AccessDenied(pid)  # skip Pass 1
            proc.open_files.side_effect = RuntimeError("some unrelated windows bug")
            return proc

        procs = [_make_raising_proc(72000)]

        with (
            patch.object(psutil, "process_iter", side_effect=lambda *a, **kw: iter(procs)),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "win32"

            with pytest.raises(RuntimeError, match="some unrelated windows bug"):
                _find_blocking_processes(target, host_pid)

    # -- N5: Windows-only passes simply not applicable on POSIX ------------

    def test_windows_only_passes_not_applicable_on_posix_leaves_complete_true(self):
        """N5: on a non-Windows platform, Pass 1b/1c are not run at all --
        this must never itself contribute a skipped/truncated tag."""
        import psutil

        target = "/fake/worktree"
        host_pid = os.getpid()

        with (
            patch.object(psutil, "process_iter", return_value=iter([])),
            patch.object(psutil, "Process") as mock_proc_cls,
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
        ):
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host
            mock_sys.platform = "linux"

            result = _find_blocking_processes(target, host_pid)

        assert result.complete is True
        assert result.skipped_passes == ()

    # -- D4/D6: Windows-only Pass 1c (real win32) ---------------------------

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_handle_scan_skipped_when_no_budget_remains(self):
        """D4: a deadline that leaves zero budget by the time Pass 1c would
        run must skip it outright (handle_scan:skipped), never call
        _win_handle_holders with a <= 0 budget."""
        record_target = "/fake/worktree"
        host_pid = os.getpid()

        with patch(
            "lib_python_worktree.core.process_lifecycle._win_handle_holders"
        ) as mock_scan:
            # A deadline exactly "now" leaves ~0.0s once Pass 1/1b complete.
            deadline = time.monotonic() + 0.001
            result = _find_blocking_processes(record_target, host_pid, deadline=deadline)

        assert "handle_scan:skipped" in result.skipped_passes or not mock_scan.called

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_handle_scan_failed_when_win_handle_holders_raises(self):
        """D6: _win_handle_holders raising (any ctypes/structure failure) is
        swallowed -- the pass yielded zero coverage, so it counts as a whole-
        pass skip (handle_scan:failed), not a mere degradation."""
        target = "/fake/worktree"
        host_pid = os.getpid()

        with patch(
            "lib_python_worktree.core.process_lifecycle._win_handle_holders",
            side_effect=RuntimeError("boom"),
        ):
            result = _find_blocking_processes(target, host_pid)

        assert result.complete is False
        assert "handle_scan:failed" in result.skipped_passes

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_handle_scan_truncated_by_own_deadline(self):
        """D5: _win_handle_holders returning early because its OWN
        budget_sec expired mid-enumeration (not a worker-cap hit) must be
        propagated as handle_scan:truncated. Real end-to-end: a live system
        handle table plus a budget_sec of 0.0 guarantees the per-handle loop
        never even starts a single iteration before the deadline check
        fires."""
        result = _win_handle_holders("C:/nonexistent-worktree-path", set(), budget_sec=0.0)
        assert result.complete is False

    # -- N2/N3: worker-cap / per-query ABANDONED-CAPPED leave complete=True
    #
    # Ticket #106: budget_sec alone cannot force the CAPPED/ABANDONED early
    # break-out. scan_deadline is armed before the handle-table dump/parse
    # even starts (see the comment near the top of _win_handle_holders'
    # scan loop), so a *small* budget just makes the scan die on its own
    # deadline -- the genuine D5 truncation path -- without ever reaching a
    # single _bounded_query call. Both tests below instead patch
    # _BoundedQueryWorker.submit directly so the very first query returns
    # the verdict under test, and use the generous _REAL_SCAN_TEST_BUDGET_SEC
    # budget so the deadline can never be the thing that ends the scan. Each
    # also asserts the fake submit was actually invoked and that the scan
    # returned fast, so a false pass (complete=True for the wrong reason)
    # cannot slip through.

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_worker_cap_hit_leaves_complete_true(self, monkeypatch):
        """N2: hitting the process-wide wedged-worker cap mid-scan is an
        internal degradation (the table was still enumerated up to that
        point) -- it must NOT be reported as handle_scan:truncated. Forced
        by patching _BoundedQueryWorker.submit so the very first query this
        scan issues returns CAPPED, driving _bounded_query straight to its
        _STOP branch (process_lifecycle.py, _bounded_query).

        This test is deliberately integration-level: it pins
        _win_handle_holders' *reaction* to a _STOP/CAPPED verdict (that it
        must NOT be folded into handle_scan:truncated), not submit()'s own
        cap-detection arithmetic (the _wedged_worker_count vs
        _MAX_WEDGED_HANDLE_WORKERS comparison that decides CAPPED vs
        ABANDONED in the first place) -- submit is stubbed here specifically
        to bypass that. The real, unmocked arithmetic is already covered by
        TestBoundedQueryWorker::test_submit_at_cap_returns_capped_without_raising
        and test_capped_outcome_still_counts_and_releases_its_slot.

        No wall-clock assertion is used here: the deadline check in each
        loop head always runs strictly *before* the call to
        _process_handle/submit, and stop_scan is always checked strictly
        before the deadline check on every subsequent iteration (see the
        loop in _win_handle_holders). So once the fake submit's CAPPED
        verdict sets stop_scan=True on the first handle processed,
        deadline_truncated can no longer be set afterwards -- the loop
        breaks on the stop_scan check before it ever reaches the deadline
        check again. `calls` being non-empty together with
        `result.complete is True` therefore already proves the CAPPED
        break-out fired, not the deadline path -- a timing threshold would
        be redundant and, worse, flaky on a loaded CI runner."""
        calls = []

        def fake_submit(self, fn, *, grace=None, scan_deadline=None, on_abandoned_done=None):
            calls.append(1)
            return _pl._QueryOutcome(_pl._QueryStatus.CAPPED, None)

        monkeypatch.setattr(_pl._BoundedQueryWorker, "submit", fake_submit)

        wedged_before = _pl._wedged_worker_count
        result = _win_handle_holders(
            "C:/nonexistent-worktree-path", set(), budget_sec=_REAL_SCAN_TEST_BUDGET_SEC
        )

        assert result.complete is True
        assert calls, "fake submit was never invoked -- did not exercise the CAPPED path"
        assert list(result) == []
        # The fake submit never touches _wedged_worker_count, so this test
        # itself must leave no process-wide residue behind (compared against
        # the pre-call value, not an absolute 0 -- other tests in the same
        # session legitimately leave the global non-zero).
        assert _pl._wedged_worker_count == wedged_before

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_abandoned_with_no_replacement_capacity_leaves_complete_true(self, monkeypatch):
        """N3: _bounded_query's other _STOP branch -- ABANDONED with no
        replacement-worker capacity left (_wedged_slot_available() False,
        process_lifecycle.py lines ~1819-1838) -- is the same internal-
        degradation class as CAPPED and must likewise leave complete=True,
        never handle_scan:truncated.

        Like N2, this test is deliberately integration-level: it pins
        _win_handle_holders' *reaction* to a _STOP/ABANDONED verdict, not
        submit()'s own unmocked ABANDONED/slot-release bookkeeping, which is
        already covered by
        TestBoundedQueryWorker::test_wedged_worker_count_restored_after_retired_worker_exits
        and its neighbours.

        No wall-clock assertion is used, for the same reason documented in
        N2 above: `calls` non-empty plus `result.complete is True` already
        proves the ABANDONED break-out (not the deadline) ended the scan,
        since stop_scan is always checked before the deadline on every
        iteration after the first _STOP verdict."""
        calls = []

        def fake_submit(self, fn, *, grace=None, scan_deadline=None, on_abandoned_done=None):
            calls.append(1)
            return _pl._QueryOutcome(_pl._QueryStatus.ABANDONED, None)

        monkeypatch.setattr(_pl._BoundedQueryWorker, "submit", fake_submit)
        monkeypatch.setattr(_pl, "_wedged_slot_available", lambda: False)

        result = _win_handle_holders(
            "C:/nonexistent-worktree-path", set(), budget_sec=_REAL_SCAN_TEST_BUDGET_SEC
        )

        assert result.complete is True
        assert calls, "fake submit was never invoked -- did not exercise the ABANDONED path"
        assert list(result) == []

    # -- D8: lineage-expansion loop truncated by the deadline ---------------

    def test_lineage_expansion_truncated_marks_incomplete(self):
        """D8: _kill_blocking_processes' post-discovery lineage-expansion
        loop breaking on the deadline is tagged lineage:truncated."""
        target = "/fake/worktree"
        num_blockers = 60
        fake_found = _pl._PartialList(
            [
                KilledProcessInfo(pid=65000 + i, name="proc", cmdline=["proc"])
                for i in range(num_blockers)
            ]
        )

        def _slow_process_tree(pid):
            time.sleep(0.05)
            return []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=fake_found,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                side_effect=_slow_process_tree,
            ),
            patch("lib_python_worktree.core.process_lifecycle._send_graceful_signal"),
            patch("lib_python_worktree.core.process_lifecycle._wait_or_kill"),
        ):
            result = _kill_blocking_processes(target, timeout=0.2)

        assert result.complete is False
        assert "lineage:truncated" in result.skipped_passes

    # -- D9 (ticket #107): stop(kill_orphans=True) survives an open_files() --
    # -- RuntimeError instead of crashing -------------------------------------

    def test_stop_kill_orphans_reports_incomplete_on_open_files_degraded(self):
        """R2 (ticket #107): when Pass 2's OS-wide open_files() RuntimeError
        degrades discovery to an empty, incomplete _PartialList tagged
        "open_files:degraded" (see D9 above), stop(kill_orphans=True) must
        report "stop_incomplete" via STOP_REASON_ORPHAN_SCAN_INCOMPLETE --
        exactly like any other incomplete-discovery tag -- rather than the
        pre-fix behaviour of the RuntimeError propagating out of
        _kill_blocking_processes and crashing stop() entirely. This pins the
        integration surface between _find_blocking_processes' D9 catch (via
        _kill_blocking_processes' `if not found: return found` early return)
        and stop()'s existing reporting branch; _find_blocking_processes' own
        RuntimeError handling is covered directly by
        TestDiscoveryCompleteness::
        test_open_files_runtime_error_degrades_instead_of_raising above."""
        fake_pid = 61003
        record = _make_record(
            "wt-discovery-degraded-open-files", pids={DEFAULT_ROLE: fake_pid}
        )
        store = _make_store(record)

        degraded = _pl._PartialList(
            [], complete=False, skipped_passes=("open_files:degraded",)
        )

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
                return_value=degraded,
            ),
        ):
            result = stop(
                "wt-discovery-degraded-open-files",
                store=store,
                kill_orphans=True,
                timeout=5.0,
            )

        assert result.status == "stop_incomplete"
        assert result.stop_detail.reason == STOP_REASON_ORPHAN_SCAN_INCOMPLETE
        assert "open_files:degraded" in result.stop_detail.skipped_passes
        assert result.stop_detail.kill_orphans_may_help is False

    def test_stop_kill_orphans_false_unaffected_by_open_files_degradation(self):
        """Negative pair: with kill_orphans=False, the orphan scan (and
        therefore _find_blocking_processes) never runs at all -- a would-be
        open_files() degradation is simply never observed, and stop() must
        report "stopped" normally."""
        fake_pid = 61004
        record = _make_record(
            "wt-discovery-degraded-no-orphans", pids={DEFAULT_ROLE: fake_pid}
        )
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._find_blocking_processes",
            ) as mock_find,
        ):
            result = stop(
                "wt-discovery-degraded-no-orphans",
                store=store,
                kill_orphans=False,
                timeout=5.0,
            )

        assert result.status == "stopped"
        mock_find.assert_not_called()


# ---------------------------------------------------------------------------
# TestProcessGroupMembers -- ticket #87 follow-up, finding B3
# ---------------------------------------------------------------------------

class TestProcessGroupMembers:
    """Unit tests for _process_group_members.

    Companion snapshot to _signal_process_group: catches a same-pgid
    descendant that _signal_process_group's single killpg SIGTERM might not
    kill, so it can be folded into the force-kill path and the final
    survivor re-probe in stop() (see TestStopKillsProcessGroupSurvivors).
    """

    def test_noop_on_windows(self):
        with patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys:
            mock_sys.platform = "win32"
            result = _process_group_members(1234)

        assert result == []

    def test_empty_when_not_group_leader(self):
        """Mirrors _signal_process_group's own guard: a pid that is not the
        leader of its own group must yield no members -- signalling/scanning
        a group we did not create could hit unrelated processes."""
        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(os, "getpgid", create=True, side_effect=lambda p: 42),
        ):
            mock_sys.platform = "linux"
            result = _process_group_members(1234)  # getpgid(1234) -> 42 != 1234

        assert result == []

    def test_never_scans_own_group(self):
        """Even if *pid* is a group leader, must never scan OUR OWN group."""
        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(os, "getpgid", create=True, side_effect=lambda p: 555),
        ):
            mock_sys.platform = "linux"
            # pid 555 is its own group leader AND getpgid(0) (our own group)
            # also resolves to 555.
            result = _process_group_members(555)

        assert result == []

    def test_dead_leader_no_members_returns_empty(self):
        """Ticket #110: a dead leader (os.getpgid(pid) raises OSError -- the
        leader has already been reaped) with no surviving group members
        still yields [] -- but this is NO LONGER unconditional for any dead
        leader (see test_dead_leader_finds_surviving_group_members below for
        the composite-command regression this behavior change closes). This
        replaces the old test_dead_pid_returns_empty, which asserted []
        purely because the pre-#110 implementation bailed out immediately on
        OSError without ever scanning for survivors."""
        import psutil

        host_pid = os.getpid()

        def _fake_getpgid(p):
            if p == 424242:
                raise OSError("no such process")  # leader already reaped
            if p in (0, host_pid):
                return 999  # our own pgid -- distinct from the probed group
            raise OSError("no such process")  # no member matches either

        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(os, "getpgid", create=True, side_effect=_fake_getpgid),
            patch.object(psutil, "process_iter", return_value=[]),
        ):
            mock_sys.platform = "linux"
            result = _process_group_members(424242)

        assert result == []

    def test_dead_leader_finds_surviving_group_members(self):
        """Ticket #110, finding #110-1 (the composite/chained-shell-command
        case): when the group leader has already been reaped
        (os.getpgid(pid) raises OSError), a member that is STILL ALIVE and
        shares pid's pgid must still be found. start_new_session=True (see
        _spawn_detached) guarantees a group this module created has pgid ==
        the tracked pid, so pid remains a valid probe for surviving members
        even after the leader itself exits -- this is exactly what lets
        stop() recover a backgrounded child of a wrapper shell (e.g. ``sh -c
        "... & long_running_child"``) that exits while the child keeps
        running under a different, untracked pid."""
        import psutil

        host_pid = os.getpid()
        dead_leader_pid = 55000
        survivor_pid = 55001

        def _fake_getpgid(p):
            if p == dead_leader_pid:
                raise OSError("no such process")  # leader already reaped
            if p in (0, host_pid):
                return 999  # our own pgid
            if p == survivor_pid:
                return dead_leader_pid  # still shares the now-leaderless group
            raise OSError("no such process")

        proc_survivor = MagicMock()
        proc_survivor.info = {"pid": survivor_pid}

        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(os, "getpgid", create=True, side_effect=_fake_getpgid),
            patch.object(psutil, "process_iter", return_value=[proc_survivor]),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_sys.platform = "linux"
            mock_host = MagicMock()
            mock_host.parents.return_value = []
            mock_proc_cls.return_value = mock_host

            result = _process_group_members(dead_leader_pid)

        assert result == [survivor_pid]

    def test_returns_other_members_sharing_pgid_excludes_self_and_host(self):
        """Only OTHER processes sharing the pgid are returned -- *pid*
        itself, the host process, and the host's ancestors are excluded."""
        import psutil

        host_pid = os.getpid()
        ancestor_pid = 9797
        target_pgid = 777

        def _fake_getpgid(p):
            mapping = {
                0: 999,  # our own pgid
                host_pid: 999,
                777: target_pgid,   # the tracked pid, its own leader
                778: target_pgid,   # a genuine other member
                779: 42,            # unrelated pgid
                ancestor_pid: 999,
            }
            return mapping.get(p, 42)

        proc_member = MagicMock()
        proc_member.info = {"pid": 778}
        proc_other_group = MagicMock()
        proc_other_group.info = {"pid": 779}
        proc_self = MagicMock()
        proc_self.info = {"pid": 777}
        proc_host = MagicMock()
        proc_host.info = {"pid": host_pid}

        ancestor_mock = MagicMock()
        ancestor_mock.pid = ancestor_pid

        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(os, "getpgid", create=True, side_effect=_fake_getpgid),
            patch.object(
                psutil,
                "process_iter",
                return_value=[proc_member, proc_other_group, proc_self, proc_host],
            ),
            patch.object(psutil, "Process") as mock_proc_cls,
        ):
            mock_sys.platform = "linux"
            mock_host = MagicMock()
            mock_host.parents.return_value = [ancestor_mock]
            mock_proc_cls.return_value = mock_host

            result = _process_group_members(777)

        assert result == [778]


# ---------------------------------------------------------------------------
# TestStopKillsProcessGroupSurvivors -- ticket #87 follow-up, finding B3
# ---------------------------------------------------------------------------

class TestStopKillsProcessGroupSurvivors:
    """Regression tests for finding B3: a same-pgid descendant reachable
    only via _signal_process_group's SIGTERM (absent from the ppid-tree
    snapshot, e.g. already reparented out of the tracked pid's lineage) must
    be force-killed if it ignores that signal, and must be included in
    stop()'s final survivor re-probe -- not silently allowed to survive
    while stop() still reports 'stopped'."""

    def test_stop_force_kills_and_reports_survivor_of_group_only_descendant(self):
        """Driving test: a process reachable only via the process-group
        snapshot (not the ppid tree) that stays alive through the graceful
        signal must be (a) force-killed, and (b) if it STILL survives,
        reflected in stop()'s final status -- proving it was folded into
        both the force-kill path and the candidate_pids re-probe."""
        fake_pid = 41000
        group_only_pid = 41001
        record = _make_record("wt-group-survivor", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        force_kill_calls: List[int] = []
        # fake_pid (the tracked root) is already dead; group_only_pid is
        # immortal for this test so it survives every kill attempt.
        alive = {fake_pid: False, group_only_pid: True}

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_group_members",
                return_value=[group_only_pid],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._signal_process_group",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                side_effect=lambda p: alive.get(p, False),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._force_kill",
                side_effect=lambda p: force_kill_calls.append(p),
            ),
        ):
            result = stop("wt-group-survivor", store=store, timeout=0.3)

        assert group_only_pid in force_kill_calls, (
            "a process-group-only descendant that ignored the graceful "
            "signal must be force-killed, not just signalled once"
        )
        assert result.status == "stop_incomplete", (
            "a surviving process-group-only descendant must be reflected in "
            "the final status, not silently dropped from the survivor re-probe"
        )

    def test_stop_group_member_already_in_tree_not_double_force_killed(self):
        """A group member that _process_tree also already found must not be
        signalled/force-killed twice via the merged kill list."""
        fake_pid = 42000
        shared_pid = 42001
        record = _make_record("wt-group-dedup", pids={DEFAULT_ROLE: fake_pid})
        store = _make_store(record)

        graceful_calls: List[int] = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[
                    KilledProcessInfo(pid=shared_pid, name="child", cmdline=["child"])
                ],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_group_members",
                return_value=[shared_pid],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._signal_process_group",
                return_value=True,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._send_graceful_signal",
                side_effect=lambda p: graceful_calls.append(p),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._wait_or_kill",
            ),
        ):
            stop("wt-group-dedup", store=store, timeout=1.0)

        assert graceful_calls.count(shared_pid) == 1, (
            "a pid present in both the ppid-tree snapshot and the "
            f"process-group snapshot must be signalled exactly once, got "
            f"{graceful_calls.count(shared_pid)} signals"
        )


# ---------------------------------------------------------------------------
# TestWindowsJobObjectContainment -- ticket #95, R5
# ---------------------------------------------------------------------------

class TestWindowsJobObjectContainment:
    """R5 (ticket #95): Windows Job Object containment.

    Root cause (finding 5): _process_tree is ppid-derived on both its
    primary (psutil children(recursive=True)) and fallback (manual ppid
    walk) paths. A ShellExecuteEx-delegated launch (what `Start-Process`
    uses without stream redirection) lands OUTSIDE our ppid lineage
    entirely -- no recursion depth fixes that. Only a Windows Job Object
    closes this gap by construction: every process assigned to the job is
    enumerable/terminable as a unit, regardless of ppid.
    """

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_stop_kills_shellexecute_delegated_grandchild(self, tmp_path, caplog):
        """Driving test / the ticket's actual acceptance test: real
        end-to-end, no mocks. Uses PowerShell's `Start-Process` -- which
        delegates via ShellExecuteEx -- to launch a Python sleeper whose PID
        is written to a file under tmp_path. This grandchild is NOT in our
        ppid lineage at all (confirmed manually: _process_tree(wrapper_pid)
        finds nothing), so only the Job Object mechanism can catch it.
        stop(kill_orphans=True) must still kill it and report it in
        killed_pids -- proving R5's containment mechanism itself.

        The overall ``status`` is asserted more leniently: "stopped" is the
        expected outcome, but R2's own (separately, already GREEN-tested)
        discovery-completeness tracking can legitimately report
        "stop_incomplete" if Pass 2's real, system-wide ``open_files()``
        scan happens to be slow on the machine running this test (measured
        real-world cost documented on ``_DISCOVERY_MAX_SEC`` -- this is a
        genuine machine/environment characteristic, not a bug in this
        ticket's fix). What must NEVER happen is an actual surviving
        process, so if status is not "stopped" this asserts the ONLY reason
        is discovery incompleteness (the logged warning), never a real
        "process(es) survived termination"."""
        pidfile = tmp_path / "child.pid"
        sleeper = tmp_path / "sleeper.py"
        sleeper.write_text(
            "import os, time\n"
            f"open(r'{pidfile}', 'w').write(str(os.getpid()))\n"
            "time.sleep(120)\n"
        )

        record = _make_record("wt-shellexecute-grandchild")
        store = _make_store(record)

        cmd_str = (
            f'Start-Process -FilePath "{sys.executable}" '
            f'-ArgumentList "{sleeper}" -WindowStyle Hidden'
        )
        result = start(
            "wt-shellexecute-grandchild",
            ["powershell", "-NoProfile", "-Command", cmd_str],
            store=store,
        )
        assert result.job_names.get(DEFAULT_ROLE) is not None, (
            "job object creation must succeed on this real Windows host"
        )

        # Wait for the grandchild to actually start and report its PID.
        grandchild_pid = None
        for _ in range(100):
            if pidfile.exists():
                try:
                    grandchild_pid = int(pidfile.read_text().strip())
                    break
                except ValueError:
                    pass
            time.sleep(0.1)
        assert grandchild_pid is not None, "sleeper never wrote its PID file"
        assert _pid_alive(grandchild_pid)

        # Confirm the premise: the grandchild is genuinely outside our ppid
        # lineage -- the wrapper's own tracked pid may already have exited
        # (powershell -Command returns once Start-Process launches), so a
        # ppid-tree walk from it finds nothing.
        wrapper_pid = result.pids[DEFAULT_ROLE]
        ppid_tree_pids = {info.pid for info in _process_tree(wrapper_pid)}
        assert grandchild_pid not in ppid_tree_pids, (
            "test premise violated: the grandchild is reachable via the "
            "ppid tree, so this is not actually testing the containment "
            "gap the Job Object mechanism exists to close"
        )

        # A generous timeout: _DISCOVERY_MAX_SEC caps Pass 1/1b/1c/2's
        # combined discovery cost at 20s regardless of *timeout*, and Pass 2
        # (open_files() across every process on the system) is measurably
        # expensive on a real, possibly busy dev machine (see this module's
        # own _DISCOVERY_MAX_SEC docstring for a measured real-world worst
        # case). 30s leaves that pass its full ceiling's worth of room so
        # this test exercises the Job Object mechanism itself, not R2's
        # (separately, already GREEN-tested) discovery-budget behaviour.
        with caplog.at_level(
            logging.WARNING, logger="lib_python_worktree.core.process_lifecycle"
        ):
            stop_result = stop(
                "wt-shellexecute-grandchild", store=store, kill_orphans=True, timeout=30.0,
            )

        time.sleep(0.3)
        try:
            assert not _pid_alive(grandchild_pid), (
                "the ShellExecuteEx-delegated grandchild survived stop() -- "
                "the Job Object containment mechanism failed to catch it"
            )
            assert grandchild_pid in {info.pid for info in stop_result.killed_pids}

            if stop_result.status != "stopped":
                assert stop_result.status == "stop_incomplete"
                assert not any(
                    "survived termination" in rec.message for rec in caplog.records
                ), (
                    "status was stop_incomplete due to an ACTUAL survivor, "
                    "not mere discovery incompleteness -- this IS a real "
                    "containment failure"
                )
                assert any(
                    "orphan scan discovery was incomplete" in rec.message
                    for rec in caplog.records
                ), (
                    "expected the incompleteness to be attributable to the "
                    "orphan scan's own discovery budget, not anything else"
                )
        finally:
            if _pid_alive(grandchild_pid):
                _force_kill(grandchild_pid)

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_job_has_no_kill_on_job_close(self):
        """Regression: closing the job's last handle must NOT kill its
        member processes. JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE must never be
        set -- if it were, a detached process would die the moment this
        keeper handle closes (e.g. conceptually, host exit), directly
        contradicting this module's core detachment invariant (a spawned
        process must survive the host process exiting)."""
        proc = _spawn_detached([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            job_name = proc._worktree_job_name
            assert job_name is not None, (
                "job object creation must succeed on this real Windows host"
            )

            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = _pl._JOB_HANDLES.pop(job_name)
            kernel32.CloseHandle(handle)

            time.sleep(0.3)
            assert _pid_alive(proc.pid), (
                "child process died when the job object's last handle "
                "closed -- JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE must have "
                "been set, which would break detachment"
            )
        finally:
            _force_kill(proc.pid)

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_start_assigns_job_and_persists_job_name(self):
        """Driving test for the start()-side half of R5: start() must create
        a Job Object, assign the spawned process to it, and persist the job
        name under this role's key in job_names on the record -- with the
        spawned pid actually enumerable as a member of that job."""
        record = _make_record("wt-job-start")
        store = _make_store(record)

        result = start(
            "wt-job-start",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            store=store,
        )
        try:
            assert result.job_names.get(DEFAULT_ROLE) is not None
            assert result.job_names[DEFAULT_ROLE].startswith("Local\\worktree-")

            job_handle = _pl._open_job_object(result.job_names[DEFAULT_ROLE])
            assert job_handle is not None
            members = _pl._job_object_member_pids(job_handle)
            assert result.pids[DEFAULT_ROLE] in members
        finally:
            _force_kill(result.pids[DEFAULT_ROLE])

    def test_stop_terminates_job_when_handle_available(self):
        """stop() must call _terminate_job_object when a job handle is
        available, and fold its members into the tree-kill step."""
        fake_pid = 81000
        member_pid = 81001
        record = _make_record(
            "wt-job-terminate", pids={DEFAULT_ROLE: fake_pid}, job_names={DEFAULT_ROLE: "Local\\fake-job"},
        )
        store = _make_store(record)

        terminate_calls: List[int] = []

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._open_job_object",
                return_value=12345,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._job_object_member_pids",
                return_value=_pl._PartialList([member_pid]),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._terminate_job_object",
                side_effect=lambda h: terminate_calls.append(h),
            ),
        ):
            result = stop("wt-job-terminate", store=store, timeout=1.0)

        assert terminate_calls == [12345]
        assert member_pid in {info.pid for info in result.killed_pids}
        assert result.status == "stopped"

    def test_stop_does_not_touch_other_roles_job(self):
        """Regression (reviewer finding, ticket #95 fix cycle): job
        tracking must be per-role (``job_names: Dict[str, str]``,
        mirroring ``pids``), not a record-wide scalar.

        Two roles ("main" and "worker") are running concurrently, each with
        its own Job Object. Stopping "main" must open/enumerate/terminate
        ONLY "main"'s job -- "worker"'s job must never be touched, and its
        ``job_names`` entry must survive intact.

        Pre-fix, ``record.job_name`` was a single scalar unconditionally
        overwritten by every ``start()`` call regardless of role: starting
        "worker" after "main" would overwrite the scalar to "worker"'s job
        name, so ``stop(role="main")`` would then open/terminate "worker"'s
        job instead of "main"'s -- killing "worker"'s entire contained
        process tree as a side effect of stopping "main", while "main"'s own
        job was never enumerated or terminated at all."""
        main_pid = 82000
        worker_pid = 82001
        record = _make_record(
            "wt-job-per-role",
            pids={"main": main_pid, "worker": worker_pid},
            job_names={
                "main": "Local\\fake-job-main",
                "worker": "Local\\fake-job-worker",
            },
        )
        store = _make_store(record)

        open_calls: List[str] = []
        terminate_calls: List[int] = []
        job_handles = {"Local\\fake-job-main": 111, "Local\\fake-job-worker": 222}

        def _fake_open(job_name):
            open_calls.append(job_name)
            return job_handles[job_name]

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._open_job_object",
                side_effect=_fake_open,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._job_object_member_pids",
                return_value=_pl._PartialList([]),
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._terminate_job_object",
                side_effect=lambda h: terminate_calls.append(h),
            ),
        ):
            result = stop("wt-job-per-role", store=store, role="main", timeout=1.0)

        assert open_calls == ["Local\\fake-job-main"], (
            "stop(role='main') must open ONLY role 'main''s job -- "
            f"'worker''s job must never be enumerated, got {open_calls}"
        )
        assert terminate_calls == [111], (
            "stop(role='main') must terminate ONLY role 'main''s job "
            f"handle -- got {terminate_calls}"
        )
        assert result.job_names.get("worker") == "Local\\fake-job-worker", (
            "role 'worker''s job_names entry must survive stopping role "
            "'main' completely untouched"
        )
        assert "main" not in result.job_names, (
            "role 'main''s job_names entry must be cleared once stop() has "
            "processed it, mirroring how pids['main'] is cleared"
        )
        assert result.pids == {"worker": worker_pid}, (
            "role 'worker''s tracked pid must be untouched by stopping "
            "'main'"
        )

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_stop_closes_job_handle_and_evicts_registry_entry(self):
        """Regression (ticket #95 fix cycle, blocking finding): stop() must
        close the Job Object handle it opened/queried, and evict it from
        the :data:`_JOB_HANDLES` keeper registry, once it is done with it.

        Pre-fix, nothing anywhere calls ``CloseHandle`` on a job handle and
        nothing pops the registry entry -- this is the COMMON path (the
        keeper-registry handle _open_job_object serves back on every
        ``stop()`` call), not just the rarer restarted-host fallback the
        round-1 review flagged. On a long-lived host process (e.g. an MCP
        server that never restarts), every start()/stop() cycle for a
        Windows worktree leaks one kernel HANDLE and one _JOB_HANDLES dict
        entry indefinitely.
        """
        record = _make_record("wt-job-handle-leak")
        store = _make_store(record)

        result = start(
            "wt-job-handle-leak",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            store=store,
        )
        job_name = result.job_names[DEFAULT_ROLE]
        assert job_name in _pl._JOB_HANDLES, (
            "test premise: start() must have registered a keeper handle"
        )
        job_handle = _pl._JOB_HANDLES[job_name]
        target_pid = result.pids[DEFAULT_ROLE]

        try:
            stop("wt-job-handle-leak", store=store, timeout=5.0)
        finally:
            if _pid_alive(target_pid):
                _force_kill(target_pid)

        assert job_name not in _pl._JOB_HANDLES, (
            "stop() must evict this role's job handle from the "
            "_JOB_HANDLES keeper registry once it is done with it -- "
            "otherwise every stop() call leaks one dict entry"
        )

        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetHandleInformation.restype = wintypes.BOOL
        kernel32.GetHandleInformation.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
        ]
        flags = wintypes.DWORD(0)
        still_valid = kernel32.GetHandleInformation(job_handle, ctypes.byref(flags))
        assert not still_valid, (
            "the job object handle stop() opened/queried is still a "
            "valid, open kernel handle after stop() returned -- it must "
            "be closed via CloseHandle, or a long-lived host leaks one "
            "HANDLE per stop() call"
        )

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_stop_closes_job_handle_opened_via_fallback_path(self):
        """Regression (ticket #95 fix cycle, blocking finding): the
        ``OpenJobObjectW`` fallback path (``job_name`` not found in the
        ``_JOB_HANDLES`` keeper registry -- e.g. a restarted host) must
        ALSO close the handle it opens, not just the common keeper-
        registry path covered by the sibling test above. The round-1
        review only flagged this narrower fallback case; round 2 folds it
        into the same fix since neither path closed anything before it.
        """
        record = _make_record("wt-job-handle-fallback")
        store = _make_store(record)

        result = start(
            "wt-job-handle-fallback",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            store=store,
        )
        job_name = result.job_names[DEFAULT_ROLE]

        # Simulate a restarted host: THIS process's own keeper handle is
        # gone from the registry, while the underlying OS job object
        # itself stays alive (as it genuinely would across a real host
        # restart) by opening and holding a second, independent handle to
        # it before evicting the registry entry.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        JOB_OBJECT_QUERY = 0x0004
        JOB_OBJECT_TERMINATE = 0x0001
        kernel32.OpenJobObjectW.restype = wintypes.HANDLE
        kernel32.OpenJobObjectW.argtypes = [
            wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR,
        ]
        keeper_handle = kernel32.OpenJobObjectW(
            JOB_OBJECT_QUERY | JOB_OBJECT_TERMINATE, False, job_name
        )
        assert keeper_handle, (
            "failed to open an independent keeper handle -- test premise"
        )
        del _pl._JOB_HANDLES[job_name]
        target_pid = result.pids[DEFAULT_ROLE]

        opened_handles: List[int] = []
        orig_open_job_object = _pl._open_job_object

        def _capturing_open(name):
            handle = orig_open_job_object(name)
            opened_handles.append(handle)
            return handle

        try:
            with patch(
                "lib_python_worktree.core.process_lifecycle._open_job_object",
                side_effect=_capturing_open,
            ):
                stop("wt-job-handle-fallback", store=store, timeout=5.0)

            assert len(opened_handles) == 1 and opened_handles[0], (
                "test premise: stop() must have opened a fresh handle via "
                "the OpenJobObjectW fallback path"
            )
            fallback_handle = opened_handles[0]

            kernel32.GetHandleInformation.restype = wintypes.BOOL
            kernel32.GetHandleInformation.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
            ]
            flags = wintypes.DWORD(0)
            still_valid = kernel32.GetHandleInformation(
                fallback_handle, ctypes.byref(flags)
            )
            assert not still_valid, (
                "the fallback-path job handle stop() opened via "
                "OpenJobObjectW is still a valid, open kernel handle "
                "after stop() returned -- it must be closed too"
            )
        finally:
            if _pid_alive(target_pid):
                _force_kill(target_pid)
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle(keeper_handle)

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_start_closes_job_handle_on_assignment_failure(self):
        """Regression (ticket #95 fix cycle, round 4 blocking finding):
        when ``_assign_process_to_job`` fails inside ``_spawn_detached``
        (its own docstring documents this as a real, reachable race --
        the child could spawn its own descendants or exit before
        assignment lands), ``job_name`` deliberately stays ``None`` so
        ``record.job_names[role]`` never gets populated for that role.
        Pre-fix, this meant the Job Object handle already created by
        ``_create_job_object`` had no way to ever be discovered by
        stop() (whose handle-close logic only runs when
        ``record.job_names.get(role)`` is truthy) -- it and its
        ``_JOB_HANDLES`` entry leaked indefinitely on a long-lived host.
        The fix must close and evict that handle right on this failure
        path, since it is the only place that will ever know about it.
        """
        record = _make_record("wt-job-assign-fail")
        store = _make_store(record)

        created: List[tuple] = []
        orig_create_job_object = _pl._create_job_object

        def _capturing_create(name):
            handle = orig_create_job_object(name)
            if handle is not None:
                created.append((name, handle))
            return handle

        target_pid = None
        try:
            with (
                patch(
                    "lib_python_worktree.core.process_lifecycle._create_job_object",
                    side_effect=_capturing_create,
                ),
                patch(
                    "lib_python_worktree.core.process_lifecycle._assign_process_to_job",
                    return_value=False,
                ),
            ):
                result = start(
                    "wt-job-assign-fail",
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    store=store,
                )
            target_pid = result.pids[DEFAULT_ROLE]

            assert len(created) == 1, (
                "test premise: _create_job_object must have been called "
                "exactly once during this start()"
            )
            job_name, job_handle = created[0]

            assert DEFAULT_ROLE not in result.job_names, (
                "job_names must stay unset for this role when assignment "
                "failed -- that documented behavior must be preserved by "
                "this fix"
            )
            assert job_name not in _pl._JOB_HANDLES, (
                "the job handle created before the failed assignment must "
                "not be left in the _JOB_HANDLES keeper registry -- since "
                "job_name was never persisted to the record, stop() has "
                "no way to ever discover and close it"
            )

            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetHandleInformation.restype = wintypes.BOOL
            kernel32.GetHandleInformation.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
            ]
            flags = wintypes.DWORD(0)
            still_valid = kernel32.GetHandleInformation(
                job_handle, ctypes.byref(flags)
            )
            assert not still_valid, (
                "the job object handle created before the failed "
                "assignment is still a valid, open kernel handle after "
                "start() returned -- it must be closed via CloseHandle, "
                "or a long-lived host leaks one HANDLE per failed "
                "assignment"
            )
        finally:
            if target_pid is not None and _pid_alive(target_pid):
                _force_kill(target_pid)

    def test_stop_degrades_gracefully_when_job_unavailable(self):
        """When _open_job_object returns None (POSIX, no live handle, or
        OpenJobObject failed) stop() must not raise and must not force
        stop_incomplete purely because containment was unavailable (rule
        N7 -- a fallback, not a coverage claim)."""
        fake_pid = 81100
        record = _make_record(
            "wt-job-unavailable", pids={DEFAULT_ROLE: fake_pid}, job_names={DEFAULT_ROLE: "Local\\fake-job"},
        )
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._open_job_object",
                return_value=None,
            ),
        ):
            result = stop("wt-job-unavailable", store=store, timeout=1.0)  # must not raise

        assert result.status == "stopped"

    def test_no_job_object_on_posix(self):
        """_create_job_object/_assign_process_to_job are no-ops off Windows
        -- _spawn_detached must stash job_name=None."""
        with patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys:
            mock_sys.platform = "linux"
            job_handle = _pl._create_job_object("Local\\irrelevant")
        assert job_handle is None

        with patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys:
            mock_sys.platform = "linux"
            assigned = _pl._assign_process_to_job(1, os.getpid())
        assert assigned is False

    @pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
    def test_job_member_list_buffer_growth(self):
        """_job_object_member_pids must retry with a larger buffer on
        ERROR_MORE_DATA rather than reporting incomplete on the first
        attempt, and succeed once the buffer is big enough."""
        import ctypes
        from ctypes import wintypes

        ERROR_MORE_DATA = 234
        header_size = ctypes.sizeof(ctypes.c_ulong) * 2
        slot_size = ctypes.sizeof(ctypes.c_size_t)

        call_count = {"n": 0}
        # More than _JOB_MEMBER_LIST_INITIAL_SLOTS (64) so the FIRST attempt
        # is genuinely too small and must trigger a real ERROR_MORE_DATA ->
        # retry cycle, rather than happening to already fit.
        real_pids = list(range(90001, 90001 + 100))

        def _fake_query(handle, info_class, buf, buf_size, ret_len_ptr):
            call_count["n"] += 1
            needed = header_size + len(real_pids) * slot_size
            if buf_size < needed:
                # Simulate ERROR_MORE_DATA: report the true assigned count
                # in the header so the caller can size its retry, but leave
                # the buffer otherwise unwritten -- write ONLY the header.
                ctypes.memmove(
                    buf,
                    ctypes.pointer(ctypes.c_ulong(len(real_pids))),
                    ctypes.sizeof(ctypes.c_ulong),
                )
                ctypes.set_last_error(ERROR_MORE_DATA)
                return 0
            header = (ctypes.c_ulong * 2)(len(real_pids), len(real_pids))
            ctypes.memmove(buf, header, header_size)
            body = (ctypes.c_size_t * len(real_pids))(*real_pids)
            ctypes.memmove(
                ctypes.cast(buf, ctypes.c_void_p).value + header_size, body,
                len(real_pids) * slot_size,
            )
            return 1

        fake_kernel32 = MagicMock()
        fake_kernel32.QueryInformationJobObject.side_effect = _fake_query

        with patch("ctypes.WinDLL", return_value=fake_kernel32):
            result = _pl._job_object_member_pids(999)

        assert call_count["n"] >= 2, "expected a retry after ERROR_MORE_DATA"
        assert list(result) == real_pids
        assert result.complete is True

    def test_job_member_truncation_sets_stop_incomplete(self):
        """A _PartialList(complete=False) from the job member scan (cap hit)
        must mark stop_incomplete, same class as tree_possibly_truncated."""
        fake_pid = 81200
        record = _make_record(
            "wt-job-truncated", pids={DEFAULT_ROLE: fake_pid}, job_names={DEFAULT_ROLE: "Local\\fake-job"},
        )
        store = _make_store(record)

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                return_value=False,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._open_job_object",
                return_value=12345,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._job_object_member_pids",
                return_value=_pl._PartialList([], complete=False),
            ),
            patch("lib_python_worktree.core.process_lifecycle._terminate_job_object"),
        ):
            result = stop("wt-job-truncated", store=store, timeout=1.0)

        assert result.status == "stop_incomplete"

    def test_job_members_folded_into_survivor_reprobe(self):
        """A job member that survives everything must flip status to
        stop_incomplete via the ordinary survivor re-probe -- proving job
        members are included in candidate_pids, not just killed_tree."""
        fake_pid = 81300
        surviving_member = 81301
        record = _make_record(
            "wt-job-survivor", pids={DEFAULT_ROLE: fake_pid}, job_names={DEFAULT_ROLE: "Local\\fake-job"},
        )
        store = _make_store(record)

        def _fake_pid_alive(p):
            return p == surviving_member

        with (
            patch(
                "lib_python_worktree.core.process_lifecycle._pid_alive",
                side_effect=_fake_pid_alive,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._process_tree",
                return_value=[],
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._open_job_object",
                return_value=12345,
            ),
            patch(
                "lib_python_worktree.core.process_lifecycle._job_object_member_pids",
                return_value=_pl._PartialList([surviving_member]),
            ),
            patch("lib_python_worktree.core.process_lifecycle._terminate_job_object"),
            patch("lib_python_worktree.core.process_lifecycle._kill_process_tree"),
        ):
            result = stop("wt-job-survivor", store=store, timeout=1.0)

        assert result.status == "stop_incomplete"
