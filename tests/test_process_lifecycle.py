"""Tests for the process lifecycle module (ticket #8).

Uses ``InMemoryStateStore`` and real/mock subprocesses.  No real git or
``YamlStateStore`` required.

Regression test for #8: ``start`` spawns a detached child that survives the
caller; ``stop`` terminates it and clears the PID.
"""

from __future__ import annotations

import os
import signal
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
    start,
    stop,
)
from lib_python_worktree.core.manager import WorktreeNotFoundError
from lib_python_worktree.core.state import InMemoryStateStore, WorktreeRecord


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
