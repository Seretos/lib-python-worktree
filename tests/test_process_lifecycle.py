"""Tests for the process lifecycle module (ticket #8).

Uses ``InMemoryStateStore`` and real/mock subprocesses.  No real git or
``YamlStateStore`` required.

Regression test for #8: ``start`` spawns a detached child that survives the
caller; ``stop`` terminates it and clears the PID.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

from lib_python_worktree.core.process_lifecycle import (
    DEFAULT_ROLE,
    KilledProcessInfo,
    ProcessAlreadyRunningError,
    ProcessLifecycleError,
    ProcessNotRunningError,
    _find_blocking_processes,
    _force_kill,
    _kill_blocking_processes,
    _pid_alive,
    _send_graceful_signal,
    _spawn_detached,
    _wait_or_kill,
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
        """_spawn_detached returns a valid PID for a short-lived process."""
        pid = _spawn_detached([sys.executable, "-c", "import time; time.sleep(30)"])
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
        pid = _spawn_detached([sys.executable, "-c", "import time; time.sleep(30)"])
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
        """stop force-kills the process when it doesn't die within timeout."""
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
        assert result.status == "stopped"

    def test_stop_timeout_zero_immediate_force_kill(self):
        """timeout=0 must go straight to force-kill without waiting."""
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
        assert result.status == "stopped"

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
