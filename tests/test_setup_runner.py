"""Tests for the W5 setup-script runner."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pytest

from lib_python_worktree.setup.runner import (
    DEFAULT_LOG_ROOT,
    LOG_ROOT_ENV,
    SetupFailedError,
    SetupRunner,
    _PlainStep,
    _resolve_setup_timeout,
    _resolve_shell,
    log_dir_for,
)


@dataclass
class _FakeProc:
    """Popen-shaped fake: exposes ``.communicate()`` / ``.kill()``.

    ``self._runner`` is now a ``(cmd, *, cwd, env) -> Popen-like`` seam (see
    runner.py's ``_invoke``), so fakes must return an object shaped like a
    ``Popen`` rather than a ``subprocess.run``-style ``CompletedProcess``.
    """

    returncode: int
    stdout: str = ""
    stderr: str = ""
    kill_called: bool = False

    def communicate(self, timeout=None):
        return (self.stdout, self.stderr)

    def kill(self):
        self.kill_called = True


def _real_runner(*args, **kwargs):
    return subprocess.run(*args, **kwargs)


def _native_echo(message: str) -> str:
    """Return a shell-agnostic echo command line.

    PowerShell, bash, and sh all interpret ``echo <msg>`` compatibly enough
    for the test message we use here (no special chars).
    """

    return f"echo {message}"


def _native_false() -> str:
    """Return a shell command that exits non-zero on both PowerShell and Bash."""

    if sys.platform == "win32":
        return "exit 1"
    return "exit 1"


def test_runner_skips_when_isolation_none(tmp_path: Path):
    runner = SetupRunner(log_root=tmp_path / "logs")
    res = runner.run(
        setup=[_PlainStep(run=_native_echo("nope"))],
        worktree_id="wt-x",
        worktree_path=tmp_path / "wt",
        branch="main",
        isolation="none",
    )
    assert res.ok
    assert res.steps == []
    assert not (tmp_path / "logs" / "wt-x").exists()


def test_runner_skips_when_no_steps(tmp_path: Path):
    runner = SetupRunner(log_root=tmp_path / "logs")
    res = runner.run(
        setup=[],
        worktree_id="wt-x",
        worktree_path=tmp_path / "wt",
        branch="main",
    )
    assert res.ok
    assert res.steps == []


def test_successful_multistep_run(tmp_path: Path):
    wt = tmp_path / "wt"
    wt.mkdir()
    runner = SetupRunner(log_root=tmp_path / "logs")
    res = runner.run(
        setup=[
            _PlainStep(run=_native_echo("first"), name="hello"),
            _PlainStep(run=_native_echo("second")),
        ],
        worktree_id="wt-success",
        worktree_path=wt,
        branch="main",
    )
    assert res.ok
    assert len(res.steps) == 2
    assert res.steps[0].name == "hello"
    assert res.steps[1].name == "step-1"
    assert all(s.returncode == 0 for s in res.steps)
    for step in res.steps:
        assert step.log_path.exists()
        text = step.log_path.read_text(encoding="utf-8")
        assert "# returncode: 0" in text
        assert "# ---- stdout ----" in text
        assert "# ---- stderr ----" in text


def test_failed_step_aborts_chain_and_raises(tmp_path: Path):
    wt = tmp_path / "wt"
    wt.mkdir()
    runner = SetupRunner(log_root=tmp_path / "logs")
    with pytest.raises(SetupFailedError) as exc_info:
        runner.run(
            setup=[
                _PlainStep(run=_native_echo("ok-1"), name="ok-1"),
                _PlainStep(run=_native_false(), name="boom"),
                _PlainStep(run=_native_echo("not-reached"), name="ok-3"),
            ],
            worktree_id="wt-fail",
            worktree_path=wt,
            branch="main",
        )
    err = exc_info.value
    assert err.step_index == 1
    assert err.step_name == "boom"
    assert err.returncode != 0
    assert err.log_path.exists()
    # Third step must NOT have been logged.
    log_dir = tmp_path / "logs" / "wt-fail"
    files = sorted(log_dir.iterdir())
    assert len(files) == 2


def test_env_vars_injected(tmp_path: Path, monkeypatch):
    wt = tmp_path / "wt"
    wt.mkdir()
    calls: List[List[str]] = []
    seen_env: List[dict] = []

    def fake_run(cmd, *, cwd, env):
        calls.append(list(cmd))
        seen_env.append(dict(env))
        return _FakeProc(returncode=0, stdout="ok", stderr="")

    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
    runner.run(
        setup=[_PlainStep(run="anything", name="probe")],
        worktree_id="wt-env",
        worktree_path=wt,
        branch="feature/foo",
        port_mapping={"app": 31000, "db": 31001},
    )
    env = seen_env[0]
    assert env["WORKTREE_ID"] == "wt-env"
    assert env["WORKTREE_PATH"] == str(wt)
    assert env["WORKTREE_BRANCH"] == "feature/foo"
    assert env["WORKTREE_PORT_APP"] == "31000"
    assert env["WORKTREE_PORT_DB"] == "31001"


def test_shell_override_pwsh(monkeypatch):
    assert _resolve_shell("pwsh") == ["pwsh", "-NoProfile", "-Command"]


def test_shell_override_bash():
    assert _resolve_shell("bash") == ["bash", "-c"]


def test_shell_override_sh():
    assert _resolve_shell("sh") == ["sh", "-c"]


def test_shell_override_powershell():
    assert _resolve_shell("powershell") == [
        "powershell.exe",
        "-NoProfile",
        "-Command",
    ]


def test_shell_override_invalid_raises():
    with pytest.raises(ValueError):
        _resolve_shell("zsh")


def test_shell_auto_detect_uses_platform_default(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert _resolve_shell(None)[0] == "powershell.exe"
    monkeypatch.setattr(sys, "platform", "linux")
    assert _resolve_shell(None) == ["bash", "-c"]


def test_shell_override_used_for_step(tmp_path: Path):
    wt = tmp_path / "wt"
    wt.mkdir()
    captured: List[List[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        return _FakeProc(returncode=0)

    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
    runner.run(
        setup=[_PlainStep(run="echo hi", shell="bash")],
        worktree_id="wt-shell",
        worktree_path=wt,
        branch="main",
    )
    assert captured[0][:2] == ["bash", "-c"]
    assert captured[0][-1] == "echo hi"


def test_log_dir_for_default():
    p = log_dir_for("abc", env={})
    assert p == DEFAULT_LOG_ROOT / "abc"


def test_log_dir_for_env_override(tmp_path: Path):
    p = log_dir_for("abc", env={LOG_ROOT_ENV: str(tmp_path / "logs")})
    assert p == tmp_path / "logs" / "abc"


def test_failed_step_marks_aborted_at(tmp_path: Path):
    wt = tmp_path / "wt"
    wt.mkdir()

    def fake_run(cmd, **kwargs):
        # Step 0 succeeds, step 1 fails.
        return _FakeProc(returncode=0 if "ok" in cmd[-1] else 7)

    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
    with pytest.raises(SetupFailedError) as exc_info:
        runner.run(
            setup=[
                _PlainStep(run="ok-1"),
                _PlainStep(run="bad-step"),
            ],
            worktree_id="wt-7",
            worktree_path=wt,
            branch="main",
        )
    assert exc_info.value.returncode == 7


# ---------------------------------------------------------------------------
# Ticket #49: SetupRunner._build_env uses _get_user_profile_env as its base
# ---------------------------------------------------------------------------

from unittest.mock import patch  # noqa: E402


def test_build_env_uses_get_user_profile_env_as_base(tmp_path: Path):
    """_build_env() starts from _get_user_profile_env(), not raw dict(self._env).

    Patches ``lib_python_worktree.setup.runner._get_user_profile_env`` with a
    sentinel dict and confirms the sentinel key is present in the env dict
    actually passed to the subprocess runner invocation.  A regression where
    _build_env reverts to ``dict(self._env)`` would omit the sentinel key.
    """
    wt = tmp_path / "wt"
    wt.mkdir()

    seen_env: List[dict] = []

    def fake_run(cmd, *, cwd, env):
        seen_env.append(dict(env))
        return _FakeProc(returncode=0, stdout="", stderr="")

    sentinel_base = {"SENTINEL_PROFILE_VAR": "from_profile_env"}

    import lib_python_worktree.setup.runner as _runner_module  # noqa: PLC0415

    with patch.object(_runner_module, "_get_user_profile_env", return_value=dict(sentinel_base)):
        runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
        runner.run(
            setup=[_PlainStep(run="probe", name="probe")],
            worktree_id="wt-sentinel",
            worktree_path=wt,
            branch="main",
        )

    assert len(seen_env) == 1, "fake_run must have been called exactly once"
    env = seen_env[0]
    assert "SENTINEL_PROFILE_VAR" in env, (
        "_get_user_profile_env() sentinel key must appear in the env passed to the subprocess"
    )
    assert env["SENTINEL_PROFILE_VAR"] == "from_profile_env"
    # Worktree identity vars are still injected on top.
    assert env["WORKTREE_ID"] == "wt-sentinel"
    assert env["WORKTREE_BRANCH"] == "main"


def test_build_env_self_env_overlays_profile_base(tmp_path: Path):
    """self._env overlays _get_user_profile_env() so the test-injection seam still works.

    When SetupRunner is constructed with env={"FOO": "from_self_env"} and
    _get_user_profile_env returns {"FOO": "from_profile"}, the subprocess must
    see FOO="from_self_env" (self._env wins over the profile base).
    """
    wt = tmp_path / "wt"
    wt.mkdir()

    seen_env: List[dict] = []

    def fake_run(cmd, *, cwd, env):
        seen_env.append(dict(env))
        return _FakeProc(returncode=0, stdout="", stderr="")

    profile_base = {"FOO": "from_profile", "ONLY_IN_PROFILE": "yes"}
    injection_env = {"FOO": "from_self_env"}

    import lib_python_worktree.setup.runner as _runner_module  # noqa: PLC0415

    with patch.object(_runner_module, "_get_user_profile_env", return_value=dict(profile_base)):
        runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run, env=injection_env)
        runner.run(
            setup=[_PlainStep(run="probe", name="probe")],
            worktree_id="wt-overlay",
            worktree_path=wt,
            branch="main",
        )

    assert len(seen_env) == 1
    env = seen_env[0]
    assert env["FOO"] == "from_self_env", (
        "self._env must overlay _get_user_profile_env() (test-injection seam)"
    )
    assert env["ONLY_IN_PROFILE"] == "yes", (
        "keys only in profile base must still appear when self._env does not override them"
    )


# ---------------------------------------------------------------------------
# Ticket #66: SetupRunner._invoke has no subprocess timeout -- a wedged
# setup/teardown step must be killed rather than hanging forever.
# ---------------------------------------------------------------------------


class _TimeoutThenDrainProc:
    """Fake Popen whose first ``communicate()`` call always times out.

    Records whether ``kill()`` was called and how many times ``communicate()``
    was invoked (so tests can assert the bounded post-kill drain happened).
    """

    def __init__(self) -> None:
        self.returncode = -1
        self.kill_called = False
        self.communicate_calls: List[Optional[float]] = []

    def communicate(self, timeout=None):
        self.communicate_calls.append(timeout)
        raise subprocess.TimeoutExpired(cmd=["fake"], timeout=timeout)

    def kill(self):
        self.kill_called = True


def test_invoke_timeout_raises_setup_failed_error_and_kills_process(tmp_path: Path):
    """Regression test for ticket #66.

    A step whose ``communicate(timeout=...)`` raises ``TimeoutExpired`` must
    be killed, have a bounded post-kill drain attempted, get a log file
    written with a synthetic ``returncode == -1``, and surface as a
    ``SetupFailedError`` (not an unhandled hang) with a "timed out after"
    message and the correct step identity.
    """
    wt = tmp_path / "wt"
    wt.mkdir()

    procs: List[_TimeoutThenDrainProc] = []

    def fake_runner(cmd, *, cwd, env):
        proc = _TimeoutThenDrainProc()
        procs.append(proc)
        return proc

    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_runner)
    with pytest.raises(SetupFailedError) as exc_info:
        runner.run(
            setup=[_PlainStep(run=_native_echo("wedged"), name="wedged-step")],
            worktree_id="wt-timeout",
            worktree_path=wt,
            branch="main",
            timeout=0.01,
        )

    err = exc_info.value
    assert err.returncode == -1
    assert err.step_index == 0
    assert err.step_name == "wedged-step"
    assert err.timeout == 0.01
    assert "timed out after" in str(err)
    assert err.log_path.exists()

    assert len(procs) == 1
    proc = procs[0]
    assert proc.kill_called, "the wedged process must be killed"
    # First call is the real (short) timeout; second is the bounded 5s drain.
    assert len(proc.communicate_calls) == 2
    assert proc.communicate_calls[0] == 0.01
    assert proc.communicate_calls[1] == 5


def test_invoke_timeout_none_disables_timeout_and_never_kills(tmp_path: Path, monkeypatch):
    """``timeout=None`` with empty-string env fully disables the timeout.

    Confirms the seam's ``communicate()`` is invoked with ``timeout=None``
    (nothing passed into the seam) and ``kill()`` is never called for a step
    that completes normally, when ``WORKTREE_SETUP_TIMEOUT_SEC=""``.
    """
    wt = tmp_path / "wt"
    wt.mkdir()

    seen_timeouts: List[Optional[float]] = []
    procs: List[_FakeProc] = []

    def fake_run(cmd, *, cwd, env):
        proc = _FakeProc(returncode=0, stdout="ok", stderr="")
        original_communicate = proc.communicate

        def communicate(timeout=None):
            seen_timeouts.append(timeout)
            return original_communicate(timeout=timeout)

        proc.communicate = communicate  # type: ignore[method-assign]
        procs.append(proc)
        return proc

    monkeypatch.setenv("WORKTREE_SETUP_TIMEOUT_SEC", "")
    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
    res = runner.run(
        setup=[_PlainStep(run=_native_echo("no-timeout"), name="probe")],
        worktree_id="wt-no-timeout",
        worktree_path=wt,
        branch="main",
        timeout=None,
    )
    assert res.ok
    assert seen_timeouts == [None]
    assert procs[0].kill_called is False


def test_real_timeout_kills_wedged_process(tmp_path: Path):
    """End-to-end: a step that actually sleeps well past its timeout.

    Runs a real subprocess (via the runner's normal shell-resolution path,
    with the default ``_default_popen`` seam) that sleeps 30s, with
    ``timeout=0.5``. Asserts control returns promptly (proving the kill path
    works) rather than after the full 30s sleep.
    """
    wt = tmp_path / "wt"
    wt.mkdir()

    code = "import time; time.sleep(30)"
    if sys.platform == "win32":
        run_line = f"& '{sys.executable}' -c '{code}'"
    else:
        import shlex  # noqa: PLC0415

        run_line = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    runner = SetupRunner(log_root=tmp_path / "logs")

    import time as _time  # noqa: PLC0415

    start = _time.monotonic()
    with pytest.raises(SetupFailedError) as exc_info:
        runner.run(
            setup=[_PlainStep(run=run_line, name="sleeper")],
            worktree_id="wt-real-timeout",
            worktree_path=wt,
            branch="main",
            timeout=0.5,
        )
    elapsed = _time.monotonic() - start

    assert exc_info.value.returncode == -1
    assert exc_info.value.timeout == 0.5
    # Must return well before the 30s sleep would complete.
    assert elapsed < 15


# ---------------------------------------------------------------------------
# _resolve_setup_timeout precedence (mirrors _resolve_git_timeout /
# _resolve_install_timeout)
# ---------------------------------------------------------------------------


def test_resolve_setup_timeout_explicit_wins(monkeypatch):
    monkeypatch.setenv("WORKTREE_SETUP_TIMEOUT_SEC", "5")
    assert _resolve_setup_timeout(12.5) == 12.5


def test_resolve_setup_timeout_env_used_when_no_explicit(monkeypatch):
    monkeypatch.setenv("WORKTREE_SETUP_TIMEOUT_SEC", "5")
    assert _resolve_setup_timeout(None) == 5.0


def test_resolve_setup_timeout_empty_env_disables(monkeypatch):
    monkeypatch.setenv("WORKTREE_SETUP_TIMEOUT_SEC", "")
    assert _resolve_setup_timeout(None) is None


def test_resolve_setup_timeout_garbage_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("WORKTREE_SETUP_TIMEOUT_SEC", "not-a-number")
    assert _resolve_setup_timeout(None) == 300.0


def test_resolve_setup_timeout_no_env_uses_default(monkeypatch):
    monkeypatch.delenv("WORKTREE_SETUP_TIMEOUT_SEC", raising=False)
    assert _resolve_setup_timeout(None) == 300.0


def test_resolve_setup_timeout_zero_or_negative_env_disables(monkeypatch):
    monkeypatch.setenv("WORKTREE_SETUP_TIMEOUT_SEC", "0")
    assert _resolve_setup_timeout(None) is None
    monkeypatch.setenv("WORKTREE_SETUP_TIMEOUT_SEC", "-1")
    assert _resolve_setup_timeout(None) is None


def test_run_timeout_kwarg_overrides_instance_timeout(tmp_path: Path):
    """``run(timeout=...)`` overrides the instance-level ``self.timeout``."""
    wt = tmp_path / "wt"
    wt.mkdir()

    seen_timeouts: List[Optional[float]] = []

    def fake_run(cmd, *, cwd, env):
        proc = _FakeProc(returncode=0, stdout="", stderr="")
        original_communicate = proc.communicate

        def communicate(timeout=None):
            seen_timeouts.append(timeout)
            return original_communicate(timeout=timeout)

        proc.communicate = communicate  # type: ignore[method-assign]
        return proc

    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run, timeout=99.0)
    runner.run(
        setup=[_PlainStep(run="probe", name="probe")],
        worktree_id="wt-override",
        worktree_path=wt,
        branch="main",
        timeout=7.0,
    )
    assert seen_timeouts == [7.0]


def test_successful_step_still_logs_returncode_zero_with_timeout_plumbing(tmp_path: Path):
    """Guard: threading timeout support through _invoke must not regress the
    happy path -- a successful step still logs ``returncode: 0`` and returns
    normally."""
    wt = tmp_path / "wt"
    wt.mkdir()
    runner = SetupRunner(log_root=tmp_path / "logs")
    res = runner.run(
        setup=[_PlainStep(run=_native_echo("still-fine"), name="probe")],
        worktree_id="wt-happy",
        worktree_path=wt,
        branch="main",
        timeout=30,
    )
    assert res.ok
    assert res.steps[0].returncode == 0
    text = res.steps[0].log_path.read_text(encoding="utf-8")
    assert "# returncode: 0" in text
