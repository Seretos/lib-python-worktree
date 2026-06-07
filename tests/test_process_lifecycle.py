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
        assert all(t == 5.0 for (_, t) in wait_calls)
        assert [p for (p, _) in wait_calls] == [1010, 2020]

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
