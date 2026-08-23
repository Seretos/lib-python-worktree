"""Tests for the W5 setup-script runner."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pytest

import base64

from lib_python_worktree.setup.runner import (
    DEFAULT_LOG_ROOT,
    LOG_ROOT_ENV,
    SetupFailedError,
    SetupRunner,
    _default_popen,
    _PlainStep,
    _resolve_lower_priority,
    _resolve_setup_timeout,
    _resolve_shell,
    log_dir_for,
)

# NOTE: `_build_step_command` (ticket #109) is imported locally inside the
# handful of test functions that call it directly, rather than at module
# level here. It is a brand-new helper introduced by this fix, so importing
# it at module level would make the *entire* test module fail to collect
# against pre-fix code -- masking the genuine behavioural RED signal (wrong
# argv shape / wrong log content) that the other tests in this file exist to
# demonstrate. Keeping it function-local preserves collection of the rest of
# the suite so those tests can RED for the right reason.


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
    assert _resolve_shell("pwsh") == ["pwsh", "-NoProfile", "-NonInteractive", "-Command"]


def test_shell_override_bash():
    assert _resolve_shell("bash") == ["bash", "-c"]


def test_shell_override_sh():
    assert _resolve_shell("sh") == ["sh", "-c"]


def test_shell_override_powershell():
    assert _resolve_shell("powershell") == [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
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


# ---------------------------------------------------------------------------
# Ticket #109: a self-wrapped `powershell -Command "..."` run: value fails
# silently (returncode 1, empty stdout/stderr) because raw `-Command <text>`
# round-trips through subprocess.list2cmdline re-quoting AND PowerShell's own
# re-parsing, mangling the original quote structure. Fix: transport the run
# line via -EncodedCommand (base64 utf-16-le) instead.
# ---------------------------------------------------------------------------


def test_powershell_step_argv_uses_encoded_command(tmp_path: Path, monkeypatch):
    """The argv built for a PowerShell-routed step uses -EncodedCommand, not
    a raw -Command <text> argument -- the raw run line must not appear
    verbatim in any argv element, and the blob must decode back to it."""
    monkeypatch.setattr(sys, "platform", "win32")
    # `sys` is a process-wide singleton, so the patch above also makes
    # _env_utils._get_user_profile_env() (called by SetupRunner._build_env
    # during runner.run() below) believe it is on Windows and try a real
    # `import winreg`, which does not exist on non-Windows CI runners. Stub
    # it out -- this test only asserts on the argv shape, not on
    # environment-variable sourcing.
    monkeypatch.setattr(
        "lib_python_worktree.setup.runner._get_user_profile_env",
        lambda: dict(os.environ),
    )
    wt = tmp_path / "wt"
    wt.mkdir()
    captured: List[List[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        return _FakeProc(returncode=0)

    run_line = 'powershell -NoProfile -NonInteractive -Command "Write-Output \'hello world\'"'
    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
    runner.run(
        setup=[_PlainStep(run=run_line, name="probe")],
        worktree_id="wt-encoded",
        worktree_path=wt,
        branch="main",
    )

    argv = captured[0]
    blob = argv[-1]
    assert argv == ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", blob]
    assert all(part != run_line for part in argv)
    # ticket #134: the blob now carries run_line + an exit-code epilogue, so
    # it starts with (not equals) the original run line.
    assert base64.b64decode(blob).decode("utf-16-le").startswith(run_line)


@pytest.mark.skipif(sys.platform != "win32", reason="win32-only")
def test_self_wrapped_powershell_step_runs_to_completion(tmp_path: Path):
    """End-to-end regression coverage for the ticket #109 shape: a run: value
    that self-wraps `powershell -Command "..."` (with a quoted, spaced inner
    argument), using the real (non-faked) _default_popen runner, must execute
    the inner command and produce its output.

    Honesty note: on this machine's Windows PowerShell build (5.1.26100),
    this exact quote shape was verified to already return 0 with correct
    output *before* the -EncodedCommand fix too -- extensive local probing
    (varying quote style/nesting and trailing-backslash counts) did not
    reproduce a case where `subprocess.list2cmdline`'s round-trip through
    this specific PowerShell host actually corrupted the argv, so this test
    does not demonstrate a RED->GREEN transition here. It is retained as
    real end-to-end coverage that the fix does not regress normal execution
    of this shape, and as a regression guard in case the underlying
    mangling *does* reproduce on a different PowerShell build/edition.
    `test_powershell_step_argv_uses_encoded_command` below is the test that
    demonstrates the actual behavioural change (the argv transport itself)
    with a genuine RED->GREEN transition."""
    wt = tmp_path / "wt"
    wt.mkdir()
    sentinel = "WORKTREE-109-SENTINEL"
    run_line = (
        "powershell -NoProfile -NonInteractive -Command "
        f"\"Write-Output '{sentinel} with spaces'\""
    )

    runner = SetupRunner(log_root=tmp_path / "logs")
    res = runner.run(
        setup=[_PlainStep(run=run_line, name="self-wrapped")],
        worktree_id="wt-self-wrapped",
        worktree_path=wt,
        branch="main",
    )

    assert res.ok
    assert res.steps[0].returncode == 0
    text = res.steps[0].log_path.read_text(encoding="utf-8")
    assert sentinel in text


def test_silent_nonzero_step_logs_synthetic_diagnostic(tmp_path: Path):
    """A step that exits non-zero with completely empty stdout/stderr gets a
    synthetic diagnostic note appended to the log's stderr section, naming
    the exit code and effective argv -- so it can never again fail as an
    opaque `returncode: 1` with two empty sections."""
    wt = tmp_path / "wt"
    wt.mkdir()

    def fake_run(cmd, **kwargs):
        return _FakeProc(returncode=1, stdout="", stderr="")

    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
    with pytest.raises(SetupFailedError) as exc_info:
        runner.run(
            setup=[_PlainStep(run="mystery-failure", name="probe", shell="bash")],
            worktree_id="wt-silent",
            worktree_path=wt,
            branch="main",
        )

    text = exc_info.value.log_path.read_text(encoding="utf-8")
    assert "mystery-failure" in text
    assert "['bash', '-c', 'mystery-failure']" in text
    assert "# returncode: 1" in text
    stderr_section = text.split("# ---- stderr ----\n", 1)[1]
    assert stderr_section.strip() != ""


def test_whitespace_only_output_also_triggers_synthetic_note(tmp_path: Path):
    """Whitespace-only stdout/stderr counts as "no output" for the synthetic
    diagnostic note -- not just a literally empty string."""
    wt = tmp_path / "wt"
    wt.mkdir()

    def fake_run(cmd, **kwargs):
        return _FakeProc(returncode=3, stdout="   \n", stderr="\t\n")

    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
    with pytest.raises(SetupFailedError) as exc_info:
        runner.run(
            setup=[_PlainStep(run="whitespace-step", name="probe", shell="bash")],
            worktree_id="wt-whitespace",
            worktree_path=wt,
            branch="main",
        )

    text = exc_info.value.log_path.read_text(encoding="utf-8")
    stderr_section = text.split("# ---- stderr ----\n", 1)[1]
    assert stderr_section.strip() != ""


def test_nonzero_step_with_real_stderr_gets_no_synthetic_note(tmp_path: Path):
    """A non-zero step that DID produce real stderr keeps that stderr
    verbatim -- no synthetic note is appended on top of it."""
    wt = tmp_path / "wt"
    wt.mkdir()

    def fake_run(cmd, **kwargs):
        return _FakeProc(returncode=1, stdout="", stderr="real error text\n")

    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
    with pytest.raises(SetupFailedError) as exc_info:
        runner.run(
            setup=[_PlainStep(run="real-stderr-step", name="probe", shell="bash")],
            worktree_id="wt-real-stderr",
            worktree_path=wt,
            branch="main",
        )

    text = exc_info.value.log_path.read_text(encoding="utf-8")
    stderr_section = text.split("# ---- stderr ----\n", 1)[1]
    assert stderr_section == "real error text\n"


def test_zero_exit_with_empty_output_gets_no_synthetic_note(tmp_path: Path):
    """A successful (zero-exit) step with empty output is normal and quiet
    -- it must NOT get the synthetic diagnostic note."""
    wt = tmp_path / "wt"
    wt.mkdir()

    def fake_run(cmd, **kwargs):
        return _FakeProc(returncode=0, stdout="", stderr="")

    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
    res = runner.run(
        setup=[_PlainStep(run="quiet-success", name="probe", shell="bash")],
        worktree_id="wt-quiet",
        worktree_path=wt,
        branch="main",
    )
    text = res.steps[0].log_path.read_text(encoding="utf-8")
    stderr_section = text.split("# ---- stderr ----\n", 1)[1]
    assert stderr_section == ""


def test_build_step_command_bash_unchanged():
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    assert _build_step_command(["bash", "-c"], "echo hi") == ["bash", "-c", "echo hi"]


def test_build_step_command_sh_unchanged():
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    assert _build_step_command(["sh", "-c"], "echo hi") == ["sh", "-c", "echo hi"]


def test_build_step_command_pwsh_matches_powershell_encoding():
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    run_line = "Get-ChildItem"
    ps_shell = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    pwsh_shell = ["pwsh", "-NoProfile", "-NonInteractive", "-Command"]
    ps_cmd = _build_step_command(ps_shell, run_line)
    pwsh_cmd = _build_step_command(pwsh_shell, run_line)
    assert ps_cmd == [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        ps_cmd[-1],
    ]
    assert pwsh_cmd == ["pwsh", "-NoProfile", "-NonInteractive", "-EncodedCommand", pwsh_cmd[-1]]
    # ticket #134: the blob now carries run_line + an exit-code epilogue, so
    # it starts with (not equals) the original run line.
    assert base64.b64decode(ps_cmd[-1]).decode("utf-16-le").startswith(run_line)
    assert base64.b64decode(pwsh_cmd[-1]).decode("utf-16-le").startswith(run_line)


def test_build_step_command_unknown_shell_raises_via_resolve_shell():
    with pytest.raises(ValueError):
        _resolve_shell("zsh")


@pytest.mark.parametrize(
    "run_line",
    [
        'echo "double quoted"',
        "echo 'single quoted'",
        "echo back\\slash\\path",
        "echo `backtick`",
        "echo $env:VAR and $variable",
        "line one\nline two",
        "echo café accented and 中文 characters",
    ],
)
def test_build_step_command_powershell_round_trips_special_characters(run_line):
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    shell_cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    cmd = _build_step_command(shell_cmd, run_line)
    blob = cmd[-1]
    # ticket #134: the blob now carries run_line + an exit-code epilogue, so
    # it starts with (not equals) the original run line.
    assert base64.b64decode(blob).decode("utf-16-le").startswith(run_line)


def test_build_step_command_posix_argv_shape_pinned():
    """Regression pin (ticket #109): bash -c / sh -c argv assembly is
    byte-identical to before this fix -- no -EncodedCommand involved for
    POSIX shells, ever."""
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    assert _build_step_command(["bash", "-c"], "npm install") == [
        "bash",
        "-c",
        "npm install",
    ]
    assert _build_step_command(["sh", "-c"], "npm install") == [
        "sh",
        "-c",
        "npm install",
    ]


# ---------------------------------------------------------------------------
# Ticket #134: a setup/teardown/stop/start step's real process exit code is
# collapsed to a bare 1 on Windows. Root cause: a -Command/-EncodedCommand
# script that ends without its own `exit` statement gets its *host* exit
# code derived from `$?` (0/1), discarding `$LASTEXITCODE` -- so a native
# child's real exit code (e.g. `cmd /c "... & exit 3"`, or a self-wrapped
# inner `powershell ... -Command "exit 3"`) is silently collapsed. Fix: append
# an exit-code-propagating epilogue to the run line before it is encoded.
# ---------------------------------------------------------------------------


def test_powershell_step_argv_appends_exit_code_epilogue():
    """The -EncodedCommand blob built for a PowerShell-routed step must carry
    the original run line followed by *something* extra (the exit-code
    epilogue) -- not the run line alone. This is the driving test: it uses
    only pre-existing symbols (``_build_step_command``) so that against
    pre-fix code it fails with a genuine AssertionError (decoded == run_line,
    nothing appended), not an ImportError for a not-yet-existing constant --
    see ``test_powershell_step_argv_epilogue_matches_constant`` below for the
    exact-content follow-up check."""
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    run_line = 'cmd /c "echo before-exit & exit 3"'
    shell_cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    cmd = _build_step_command(shell_cmd, run_line)
    decoded = base64.b64decode(cmd[-1]).decode("utf-16-le")
    assert decoded.startswith(run_line)
    assert decoded != run_line, "expected an exit-code epilogue appended after the run line"


def test_powershell_step_argv_epilogue_matches_constant():
    """Follow-up exact-content check: the appended text is precisely the
    module's ``_PS_EXIT_CODE_EPILOGUE`` constant."""
    from lib_python_worktree.setup.runner import (  # noqa: PLC0415
        _build_step_command,
        _PS_EXIT_CODE_EPILOGUE,
    )

    run_line = 'cmd /c "echo before-exit & exit 3"'
    shell_cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    cmd = _build_step_command(shell_cmd, run_line)
    decoded = base64.b64decode(cmd[-1]).decode("utf-16-le")
    assert decoded.endswith(_PS_EXIT_CODE_EPILOGUE)
    assert decoded == run_line + _PS_EXIT_CODE_EPILOGUE


@pytest.mark.parametrize(
    "run_line",
    [
        "Write-Output 'hi'  # a trailing comment",
        "Write-Output 'hi'\n",
        "",
    ],
)
def test_powershell_step_argv_epilogue_survives_edge_case_run_lines(run_line):
    """A run line ending in a `#` comment, a trailing newline, or an empty
    run line must not swallow or corrupt the appended epilogue -- the
    epilogue's leading `\\n` guards exactly this."""
    from lib_python_worktree.setup.runner import (  # noqa: PLC0415
        _build_step_command,
        _PS_EXIT_CODE_EPILOGUE,
    )

    shell_cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    cmd = _build_step_command(shell_cmd, run_line)
    decoded = base64.b64decode(cmd[-1]).decode("utf-16-le")
    assert decoded == run_line + _PS_EXIT_CODE_EPILOGUE


def test_build_step_command_bash_unaffected_by_epilogue():
    """bash/sh argv must stay byte-identical to today -- the epilogue is
    PowerShell/pwsh-only, POSIX shells already propagate exit status via
    their own return code."""
    from lib_python_worktree.setup.runner import _build_step_command  # noqa: PLC0415

    assert _build_step_command(["bash", "-c"], "exit 3") == ["bash", "-c", "exit 3"]
    assert _build_step_command(["sh", "-c"], "exit 3") == ["sh", "-c", "exit 3"]


def _run_single_step_returncode(
    tmp_path: Path, run_line: str, *, worktree_id: str, shell: Optional[str] = None
) -> int:
    """Run one real (non-faked) step and return its returncode, whether the
    step succeeded or raised ``SetupFailedError``."""
    wt = tmp_path / "wt"
    wt.mkdir(exist_ok=True)
    runner = SetupRunner(log_root=tmp_path / "logs")
    try:
        res = runner.run(
            setup=[_PlainStep(run=run_line, name="probe", shell=shell)],
            worktree_id=worktree_id,
            worktree_path=wt,
            branch="main",
        )
        return res.steps[0].returncode
    except SetupFailedError as exc:
        return exc.returncode


@pytest.mark.skipif(sys.platform != "win32", reason="win32-only regression")
def test_native_command_exit_code_is_propagated(tmp_path: Path):
    """The driving test for ticket #134: a native child's real exit code
    (here 3, via `cmd /c "... & exit 3"`) must survive a PowerShell-routed
    step -- not collapse to a bare 1 -- and the log must show the real code
    plus the stdout produced before the exit."""
    wt = tmp_path / "wt"
    wt.mkdir()
    runner = SetupRunner(log_root=tmp_path / "logs")
    with pytest.raises(SetupFailedError) as exc_info:
        runner.run(
            setup=[_PlainStep(run='cmd /c "echo before-exit & exit 3"', name="native")],
            worktree_id="wt-native-exit",
            worktree_path=wt,
            branch="main",
        )
    assert exc_info.value.returncode == 3
    text = exc_info.value.log_path.read_text(encoding="utf-8")
    assert "# returncode: 3" in text
    assert "before-exit" in text


@pytest.mark.skipif(sys.platform != "win32", reason="win32-only regression")
@pytest.mark.parametrize(
    "run_line,expected_rc",
    [
        # ticket #109 shape: a self-wrapped inner `powershell ... -Command
        # "exit 3"` is a native child of the outer PowerShell host.
        ('powershell -NoProfile -NonInteractive -Command "exit 3"', 3),
        # A bare `exit N` with no native command anywhere already worked
        # before this fix (confirmed by the plan's verification spike) --
        # pinned here so the epilogue doesn't regress it.
        ("if ($true) { exit 3 } else { exit 0 }", 3),
        ("exit 0", 0),
        ("Write-Output 'ok'", 0),
        # A failing cmdlet (non-native failure path) still surfaces non-zero.
        ('Get-Item -Path "C:\\this\\does\\not\\exist\\at\\all.xyz"', 1),
        # Exit-code boundary coverage: 255 and an above-255 value.
        ("exit 255", 255),
        ("exit 500", 500),
    ],
)
def test_powershell_step_exit_codes_propagate(tmp_path: Path, run_line, expected_rc):
    rc = _run_single_step_returncode(
        tmp_path, run_line, worktree_id=f"wt-rc-{expected_rc}-{abs(hash(run_line))}"
    )
    assert rc == expected_rc


@pytest.mark.skipif(sys.platform == "win32", reason="posix-only pin")
def test_posix_bash_exit_code_propagates_unaffected_by_epilogue(tmp_path: Path):
    """POSIX shells were never touched by the epilogue fix -- pin that a
    real `bash -c 'exit 3'` step still reports 3, exactly as before."""
    rc = _run_single_step_returncode(tmp_path, "exit 3", worktree_id="wt-posix-rc", shell="bash")
    assert rc == 3


# ---------------------------------------------------------------------------
# Ticket #134, behavioural requirement 3: shared-call-site coverage -- the
# fix lands in the single _build_step_command seam shared by SetupRunner
# (setup:/teardown:) and WorktreeManager.start (start:).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="win32-only regression")
def test_teardown_style_step_reports_faithful_nonzero_exit_code(tmp_path: Path):
    """A teardown: step (modeled here as any SetupRunner.run() call, since
    teardown steps funnel through the same run()/_invoke() path as setup:)
    with a non-1 exit code must report that exact code, not a collapsed 1."""
    wt = tmp_path / "wt"
    wt.mkdir()
    runner = SetupRunner(log_root=tmp_path / "logs")
    with pytest.raises(SetupFailedError) as exc_info:
        runner.run(
            setup=[_PlainStep(run='cmd /c "exit 7"', name="teardown-step")],
            worktree_id="wt-teardown-exit",
            worktree_path=wt,
            branch="main",
        )
    assert exc_info.value.returncode == 7


# ---------------------------------------------------------------------------
# Ticket #134 review fix-loop: known, accepted limitation around a stale
# *global* $LASTEXITCODE misattributed to the wrong statement's failure in a
# mixed native+cmdlet script. See _build_step_command's docstring for the
# full writeup. This is a pinning test for the CURRENT (accepted) behavior,
# not an assertion that the reported code is "correct" -- resolution option
# (b) from the review (document + pin), not option (a) (correlate the
# failure to $LASTEXITCODE, which is hard in general and out of scope for a
# fix loop).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="win32-only regression")
def test_powershell_step_stale_lastexitcode_from_earlier_statement_is_a_known_limitation(
    tmp_path: Path,
):
    # NOTE: this pins a documented KNOWN LIMITATION (see _build_step_command's
    # docstring, "Known limitation -- stale global $LASTEXITCODE"), not
    # desired behavior -- do not "fix" this test by changing the expected
    # value if the underlying epilogue logic changes; update the docstring
    # and this comment together with any such change instead.
    rc = _run_single_step_returncode(
        tmp_path,
        'cmd /c "exit 7"; Get-Item C:\\this\\does\\not\\exist.xyz',
        worktree_id="wt-stale-lastexitcode",
    )
    # The actual failure is the Get-Item cmdlet (no numeric exit code of its
    # own, would report a generic 1 under bare-$? semantics), not the
    # already-completed `cmd /c "exit 7"` earlier in the script -- but
    # $LASTEXITCODE is a global, so the stale 7 is what gets reported.
    assert rc == 7


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
        #
        # shell="bash" pins the argv shape to `[..., "-c", run_line]`
        # regardless of host platform/default shell, so `cmd[-1]` is always
        # the raw run text here -- otherwise, on win32, the default
        # PowerShell shell would route the run text through
        # `_build_step_command`'s `-EncodedCommand` transport (ticket #109)
        # and `cmd[-1]` would be an opaque base64 blob instead.
        return _FakeProc(returncode=0 if "ok" in cmd[-1] else 7)

    runner = SetupRunner(log_root=tmp_path / "logs", runner=fake_run)
    with pytest.raises(SetupFailedError) as exc_info:
        runner.run(
            setup=[
                _PlainStep(run="ok-1", shell="bash"),
                _PlainStep(run="bad-step", shell="bash"),
            ],
            worktree_id="wt-7",
            worktree_path=wt,
            branch="main",
        )
    assert exc_info.value.step_index == 1
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


# ---------------------------------------------------------------------------
# Ticket #68: SetupRunner._default_popen lowers OS scheduling/IO priority of
# setup-step subprocesses so a heavy step (e.g. `npm install`) doesn't starve
# unrelated concurrent work in the calling application.
# ---------------------------------------------------------------------------

import lib_python_worktree.setup.runner as _runner_module  # noqa: E402


class _RecordingPopen:
    """Stand-in for ``subprocess.Popen`` that records call args without
    actually spawning anything."""

    def __init__(self, cmd, **kwargs):
        self.cmd = cmd
        self.kwargs = kwargs


def _patch_windows_priority_constants(monkeypatch):
    """Ensure the Windows-only priority constants exist regardless of the OS
    actually running the test (CI runs this suite on both windows-latest and
    ubuntu-22.04; these constants only exist natively on the win32 build of
    ``subprocess``)."""
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    monkeypatch.setattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000, raising=False)


def test_default_popen_windows_lowers_priority_and_keeps_create_no_window(
    tmp_path: Path, monkeypatch
):
    """Toggle enabled (default) on Windows: BELOW_NORMAL_PRIORITY_CLASS is
    OR'd into creationflags without dropping CREATE_NO_WINDOW (PR #66)."""
    _patch_windows_priority_constants(monkeypatch)
    monkeypatch.setattr(_runner_module.sys, "platform", "win32")
    monkeypatch.delenv("WORKTREE_SETUP_LOWER_PRIORITY", raising=False)
    monkeypatch.setattr(_runner_module.subprocess, "Popen", _RecordingPopen)

    proc = _default_popen(["echo", "hi"], cwd=str(tmp_path), env={})
    flags = proc.kwargs["creationflags"]
    assert flags & subprocess.CREATE_NO_WINDOW
    assert flags & subprocess.BELOW_NORMAL_PRIORITY_CLASS


def test_default_popen_windows_disabled_toggle_keeps_create_no_window_only(
    tmp_path: Path, monkeypatch
):
    """Toggle disabled on Windows: CREATE_NO_WINDOW alone, no priority flag."""
    _patch_windows_priority_constants(monkeypatch)
    monkeypatch.setattr(_runner_module.sys, "platform", "win32")
    monkeypatch.setenv("WORKTREE_SETUP_LOWER_PRIORITY", "0")
    monkeypatch.setattr(_runner_module.subprocess, "Popen", _RecordingPopen)

    proc = _default_popen(["echo", "hi"], cwd=str(tmp_path), env={})
    flags = proc.kwargs["creationflags"]
    assert flags & subprocess.CREATE_NO_WINDOW
    assert not (flags & subprocess.BELOW_NORMAL_PRIORITY_CLASS)


def test_default_popen_posix_nice_and_ionice_prefix(tmp_path: Path, monkeypatch):
    """Toggle enabled (default) on POSIX with both nice and ionice present:
    argv is prefixed with ionice -c 3 nice -n 10."""
    monkeypatch.setattr(_runner_module.sys, "platform", "linux")
    monkeypatch.delenv("WORKTREE_SETUP_LOWER_PRIORITY", raising=False)

    def fake_which(name):
        return {"nice": "/usr/bin/nice", "ionice": "/usr/bin/ionice"}.get(name)

    monkeypatch.setattr(_runner_module.shutil, "which", fake_which)
    monkeypatch.setattr(_runner_module.subprocess, "Popen", _RecordingPopen)

    proc = _default_popen(["npm", "install"], cwd=str(tmp_path), env={})
    assert proc.cmd == [
        "/usr/bin/ionice",
        "-c",
        "3",
        "/usr/bin/nice",
        "-n",
        "10",
        "npm",
        "install",
    ]
    assert "creationflags" not in proc.kwargs


def test_default_popen_posix_nice_only_when_ionice_absent(tmp_path: Path, monkeypatch):
    """ionice absent, nice present: argv degrades to nice-only prefix."""
    monkeypatch.setattr(_runner_module.sys, "platform", "linux")
    monkeypatch.delenv("WORKTREE_SETUP_LOWER_PRIORITY", raising=False)

    def fake_which(name):
        return {"nice": "/usr/bin/nice"}.get(name)

    monkeypatch.setattr(_runner_module.shutil, "which", fake_which)
    monkeypatch.setattr(_runner_module.subprocess, "Popen", _RecordingPopen)

    proc = _default_popen(["npm", "install"], cwd=str(tmp_path), env={})
    assert proc.cmd == ["/usr/bin/nice", "-n", "10", "npm", "install"]


def test_default_popen_posix_no_nice_or_ionice_falls_back_to_plain_cmd(
    tmp_path: Path, monkeypatch
):
    """Neither nice nor ionice available: best-effort fallback to a plain
    spawn -- must never raise FileNotFoundError or fail the setup step."""
    monkeypatch.setattr(_runner_module.sys, "platform", "linux")
    monkeypatch.delenv("WORKTREE_SETUP_LOWER_PRIORITY", raising=False)
    monkeypatch.setattr(_runner_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(_runner_module.subprocess, "Popen", _RecordingPopen)

    proc = _default_popen(["npm", "install"], cwd=str(tmp_path), env={})
    assert proc.cmd == ["npm", "install"]


def test_default_popen_posix_toggle_disabled_skips_nice_prefix(tmp_path: Path, monkeypatch):
    """Toggle disabled on POSIX: no nice/ionice prefix even when both binaries
    are available."""
    monkeypatch.setattr(_runner_module.sys, "platform", "linux")
    monkeypatch.setenv("WORKTREE_SETUP_LOWER_PRIORITY", "false")

    def fake_which(name):
        return {"nice": "/usr/bin/nice", "ionice": "/usr/bin/ionice"}.get(name)

    monkeypatch.setattr(_runner_module.shutil, "which", fake_which)
    monkeypatch.setattr(_runner_module.subprocess, "Popen", _RecordingPopen)

    proc = _default_popen(["npm", "install"], cwd=str(tmp_path), env={})
    assert proc.cmd == ["npm", "install"]


def test_real_subprocess_has_lowered_priority(tmp_path: Path):
    """Real-subprocess regression (ticket #68): a real, briefly-sleeping child
    spawned through the actual (non-faked) ``_default_popen`` path really
    runs at a lowered OS scheduling priority -- not just that the kwargs/argv
    were assembled correctly (covered by the unit tests above).

    Mirrors ``test_real_timeout_kills_wedged_process`` as the model for a
    real-subprocess test in this file. Uses ``psutil`` (an existing project
    dependency, see ``core/process_lifecycle.py``) rather than adding a new
    dev dependency.

    Polls rather than reading niceness exactly once: on POSIX, ``proc.pid``
    is shared across the whole ``ionice -> nice -> cmd`` exec chain (exec
    preserves the pid, there is no fork), and ``nice`` only calls
    ``setpriority()`` partway through that chain. Reading immediately after
    ``Popen()`` races the child's own startup and can observe the
    still-inherited (unlowered) niceness even though the mechanism is
    working correctly -- confirmed nondeterministic in practice (repeated
    immediate reads on Linux gave e.g. ``0, 10, 0, 0, 0``). Polling for a
    short window lets the assertion reflect the settled value instead of a
    startup snapshot, while still failing (via timeout) if the priority is
    genuinely never lowered.
    """
    import psutil  # noqa: PLC0415

    if sys.platform != "win32" and shutil.which("nice") is None:
        pytest.skip("nice(1) not available on this system")

    code = "import time; time.sleep(5)"
    cmd = [sys.executable, "-c", code]

    proc = _default_popen(cmd, cwd=str(tmp_path), env=dict(os.environ))
    try:
        ps_proc = psutil.Process(proc.pid)

        if sys.platform == "win32":
            # Windows sets the priority class atomically at CreateProcess
            # time (no exec-chain race), but poll defensively for symmetry
            # and robustness against CI scheduling jitter.
            expected = psutil.BELOW_NORMAL_PRIORITY_CLASS
            is_lowered = lambda value: value == expected  # noqa: E731
        else:
            baseline = os.getpriority(os.PRIO_PROCESS, 0)
            is_lowered = lambda value: value > baseline  # noqa: E731

        deadline = time.monotonic() + 2.0
        nice_value = ps_proc.nice()
        while not is_lowered(nice_value) and time.monotonic() < deadline:
            time.sleep(0.02)
            try:
                nice_value = ps_proc.nice()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Transient during the exec-chain transition; keep polling
                # until the deadline instead of failing the test outright.
                continue

        assert is_lowered(nice_value), (
            f"priority was never lowered within the poll window "
            f"(last observed nice_value={nice_value!r})"
        )
    finally:
        proc.kill()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass


# ---------------------------------------------------------------------------
# _resolve_lower_priority precedence (mirrors _resolve_setup_timeout's
# malformed/empty-value coverage, but for the boolean env convention this
# resolver defines)
# ---------------------------------------------------------------------------


def test_resolve_lower_priority_unset_defaults_enabled(monkeypatch):
    monkeypatch.delenv("WORKTREE_SETUP_LOWER_PRIORITY", raising=False)
    assert _resolve_lower_priority() is True


@pytest.mark.parametrize(
    "value", ["0", "false", "False", "FALSE", "no", "off", "  off  ", ""]
)
def test_resolve_lower_priority_disabled_values(monkeypatch, value):
    monkeypatch.setenv("WORKTREE_SETUP_LOWER_PRIORITY", value)
    assert _resolve_lower_priority() is False


@pytest.mark.parametrize(
    "value", ["1", "true", "TRUE", "yes", "on", "anything-else", "  1  "]
)
def test_resolve_lower_priority_enabled_values(monkeypatch, value):
    monkeypatch.setenv("WORKTREE_SETUP_LOWER_PRIORITY", value)
    assert _resolve_lower_priority() is True
