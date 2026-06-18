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
    _resolve_shell,
    log_dir_for,
)


@dataclass
class _FakeProc:
    returncode: int
    stdout: str = ""
    stderr: str = ""


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

    def fake_run(cmd, *, cwd, env, capture_output, text, check):
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

    def fake_run(cmd, *, cwd, env, capture_output, text, check):
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

    def fake_run(cmd, *, cwd, env, capture_output, text, check):
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
