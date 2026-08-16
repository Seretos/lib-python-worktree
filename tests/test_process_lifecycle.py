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
from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord

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
            # Generous, one-sided: roughly the observed wait beyond stage 1,
            # never a tight window.
            assert 0.0 < grace_spent < elapsed + 0.2, (
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
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        grace = _GraceBudget(0.25)
        scan_deadline = time.monotonic() + 30

        grace_paid = 0
        worker = _BoundedQueryWorker()
        created = [worker]
        try:
            for _ in range(5):
                t0 = time.monotonic()
                outcome = worker.submit(
                    lambda: release.wait(timeout=30), grace=grace, scan_deadline=scan_deadline
                )
                elapsed = time.monotonic() - t0
                if elapsed > _HANDLE_QUERY_TIMEOUT_SEC * 3:
                    grace_paid += 1
                assert outcome.status != _QueryStatus.RESOLVED
                if _wedged_slot_available():
                    worker = _BoundedQueryWorker()
                    created.append(worker)

            assert grace_paid <= 3, f"expected at most 3 queries to pay a grace wait, got {grace_paid}"
            # Each stage-2 wait deducts the *actually elapsed* wall-clock
            # time (clamped at 0.0), so real timing jitter can leave a tiny
            # positive residue instead of landing on exact 0.0 -- assert
            # "effectively exhausted" rather than bit-exact zero. The 0.01s
            # tolerance is one tenth of _HANDLE_QUERY_GRACE_SEC (0.10s), far
            # too small to fund another grace wait, so this still proves
            # exhaustion rather than masking a real bug.
            assert grace.remaining < 0.01, (
                f"expected the grace budget to be effectively exhausted, "
                f"{grace.remaining:.6f}s remained"
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
            assert elapsed < _HANDLE_QUERY_TIMEOUT_SEC * 5, (
                f"expected an ~stage-1-only wait, took {elapsed:.3f}s"
            )
            assert grace.remaining == 1.0, "no grace should be spent once the deadline has passed"
        finally:
            release.set()
            worker.close()

    def test_grace_truncated_by_near_scan_deadline(self, monkeypatch):
        monkeypatch.setattr(_pl, "_wedged_worker_count", 0)
        release = threading.Event()
        grace = _GraceBudget(1.0)

        worker = _BoundedQueryWorker()
        try:
            scan_deadline = time.monotonic() + _HANDLE_QUERY_TIMEOUT_SEC + 0.02
            t0 = time.monotonic()
            outcome = worker.submit(
                lambda: release.wait(timeout=10), grace=grace, scan_deadline=scan_deadline
            )
            elapsed = time.monotonic() - t0

            assert outcome.status != _QueryStatus.RESOLVED
            # stage 1 (~0.01s) + truncated stage 2 (~0.02s) -- well under
            # the full 0.10s grace ceiling.
            assert elapsed < _HANDLE_QUERY_GRACE_SEC, (
                f"expected the grace wait truncated to the near scan_deadline, took {elapsed:.3f}s"
            )
            assert grace.remaining < 1.0, "some (small) grace must have been spent"
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
            assert elapsed < _HANDLE_QUERY_TIMEOUT_SEC * 5
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
    def test_budget_sec_bounds_real_scan_wall_clock(self, tmp_path):
        """Regression for the deadline-threading fix: passing a near-zero
        ``budget_sec`` must make the per-handle resolution loop bail out
        almost immediately against the real, full system handle table,
        rather than spending up to the full 15s _HANDLE_SCAN_BUDGET_SEC
        ceiling. This is the real (non-mocked) mechanism that
        _find_blocking_processes relies on to keep Pass 1c bounded by
        whatever remains of a caller's overall timeout."""
        target = str(tmp_path / "definitely-not-a-real-worktree")

        t0 = time.monotonic()
        result = _win_handle_holders(target, excluded_pids=set(), budget_sec=0.0)
        elapsed = time.monotonic() - t0

        assert result == []
        # Well under the 15s ceiling -- the handle-table dump itself is
        # unavoidable, but the per-handle resolution loop must not run once
        # the (already-expired) budget is exhausted.
        assert elapsed < 8.0, (
            f"_win_handle_holders(budget_sec=0.0) took {elapsed:.2f}s -- expected "
            "it to bail out of the per-handle loop almost immediately instead of "
            "spending time comparable to the full _HANDLE_SCAN_BUDGET_SEC ceiling"
        )

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

    def test_dead_pid_returns_empty(self):
        with (
            patch("lib_python_worktree.core.process_lifecycle.sys") as mock_sys,
            patch.object(
                os, "getpgid", create=True, side_effect=OSError("no such process")
            ),
        ):
            mock_sys.platform = "linux"
            result = _process_group_members(424242)

        assert result == []

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
