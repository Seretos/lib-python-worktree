"""Setup-script runner (W5).

Executes the ``setup:`` steps from the worktree contract
(``<repo-root>/.seretos/worktree-setup.yml``) right after
``worktree_create`` succeeds. Sequential, abort-on-error, with injected
``WORKTREE_*`` env vars and structured per-step logs.

Decisions from the plan-comment:
- D1 (Option B): Auto-detect shell — PowerShell on Windows, Bash elsewhere,
  with an optional per-step ``shell:`` override (``bash`` | ``pwsh`` | ``sh``
  | ``powershell``).
- D2 (Option A): stdout/stderr written to per-step log files only; the runner
  returns a summary at the end.

The runner accepts a duck-typed ``setup`` list of step objects so it works
both with W3's ``WorktreeContract.setup`` (once that PR lands) and any plain
object that exposes ``.run`` / ``.name`` / ``.shell``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..core._env_utils import _get_user_profile_env


# ---- duck-typed step interface ----------------------------------------------


class SetupStep(Protocol):
    """Minimal interface W5 reads off a step object.

    W3's ``contract.schema.Step`` satisfies this; tests below use a lightweight
    dataclass for independence.
    """

    run: str
    name: Optional[str]
    shell: Optional[str]


@dataclass
class _PlainStep:
    """Fallback dataclass used by tests and by ad-hoc callers."""

    run: str
    name: Optional[str] = None
    shell: Optional[str] = None


# ---- result + error types ----------------------------------------------------


@dataclass
class SetupStepResult:
    index: int
    name: str
    returncode: int
    log_path: Path


@dataclass
class SetupResult:
    worktree_id: str
    steps: List[SetupStepResult] = field(default_factory=list)
    aborted_at: Optional[int] = None  # step index that failed, or None on success

    @property
    def ok(self) -> bool:
        return self.aborted_at is None


class SetupFailedError(RuntimeError):
    def __init__(
        self,
        *,
        worktree_id: str,
        step_index: int,
        step_name: str,
        log_path: Path,
        returncode: int,
        timeout: Optional[float] = None,
    ) -> None:
        if timeout is not None:
            message = (
                f"setup step {step_index} ({step_name!r}) for worktree "
                f"{worktree_id!r} timed out after {timeout}s. "
                f"See log: {log_path}"
            )
        else:
            message = (
                f"setup step {step_index} ({step_name!r}) for worktree "
                f"{worktree_id!r} failed with exit code {returncode}. "
                f"See log: {log_path}"
            )
        super().__init__(message)
        self.worktree_id = worktree_id
        self.step_index = step_index
        self.step_name = step_name
        self.log_path = log_path
        self.returncode = returncode
        self.timeout = timeout


# ---- path + shell helpers ----------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")
LOG_ROOT_ENV = "WORKTREE_LOG_ROOT"
DEFAULT_LOG_ROOT = Path("~/.agent-worktree/logs").expanduser()

_SETUP_TIMEOUT_ENV = "WORKTREE_SETUP_TIMEOUT_SEC"
_SETUP_TIMEOUT_DEFAULT = 300.0


def _resolve_setup_timeout(explicit: Optional[float]) -> Optional[float]:
    """Resolve the timeout for a single setup/teardown step invocation.

    Precedence: explicit kwarg > ``WORKTREE_SETUP_TIMEOUT_SEC`` env > built-in
    default of 300.0 s.  ``None`` (either as kwarg or env value ``""``)
    disables the timeout entirely.  Env is read on every call so test
    fixtures can change it without re-importing the module.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get(_SETUP_TIMEOUT_ENV)
    if raw is None:
        return _SETUP_TIMEOUT_DEFAULT
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return _SETUP_TIMEOUT_DEFAULT
    return value if value > 0 else None


def _slug(value: str, max_len: int = 40) -> str:
    s = _SLUG_RE.sub("-", value.lower()).strip("-")
    if not s:
        s = "step"
    return s[:max_len]


def log_dir_for(worktree_id: str, env: Optional[Dict[str, str]] = None) -> Path:
    """Return the directory where per-step logs for ``worktree_id`` go.

    Honors ``WORKTREE_LOG_ROOT`` so tests (and W7 once it owns state) can
    redirect log output without touching the home dir.
    """

    environ = env if env is not None else os.environ
    raw = environ.get(LOG_ROOT_ENV)
    root = Path(raw).expanduser() if raw else DEFAULT_LOG_ROOT
    return root / worktree_id


def _resolve_shell(step_shell: Optional[str]) -> List[str]:
    """Return the ``[shell, "-c"-equivalent]`` prefix for a step.

    Override values map as:
    - ``bash`` / ``sh``  → ``["<name>", "-c"]``
    - ``pwsh``           → ``["pwsh", "-NoProfile", "-Command"]``
    - ``powershell``     → ``["powershell.exe", "-NoProfile", "-Command"]``

    With no override, picks ``powershell.exe`` on Windows and ``bash`` elsewhere.
    """

    if step_shell:
        name = step_shell.lower()
        if name == "bash":
            return ["bash", "-c"]
        if name == "sh":
            return ["sh", "-c"]
        if name == "pwsh":
            return ["pwsh", "-NoProfile", "-Command"]
        if name == "powershell":
            return ["powershell.exe", "-NoProfile", "-Command"]
        raise ValueError(f"unknown step shell: {step_shell!r}")

    if sys.platform == "win32":
        return ["powershell.exe", "-NoProfile", "-Command"]
    return ["bash", "-c"]


# ---- runner ------------------------------------------------------------------


def _default_popen(cmd: List[str], *, cwd: str, env: Dict[str, str]) -> subprocess.Popen:
    """Default ``self._runner`` implementation: a hardened ``subprocess.Popen``.

    Mirrors ``core/_git_utils._run_git``'s hardening: ``stdin=DEVNULL`` so the
    child can never inherit our stdin and wedge waiting on input, explicit
    ``stdout=PIPE``/``stderr=PIPE`` (rather than ``capture_output``) because
    ``_invoke`` drives ``communicate()``/kill itself, and
    ``creationflags=CREATE_NO_WINDOW`` on Windows to avoid a console flash.
    """

    popen_kwargs: dict = {
        "cwd": cwd,
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    return subprocess.Popen(cmd, **popen_kwargs)


class SetupRunner:
    """Executes a contract's ``setup`` steps in a worktree."""

    def __init__(
        self,
        *,
        log_root: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        runner: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """``runner`` is an injection seam used by tests.

        Defaults to a ``subprocess.Popen``-backed implementation
        (:func:`_default_popen`). It must be a callable shaped
        ``(cmd, *, cwd, env) -> Popen-like object`` -- an object exposing
        ``.communicate(timeout=...)`` and ``.kill()`` -- so ``_invoke`` can
        drive the timeout/kill sequence itself.

        ``timeout`` is the instance-level default step timeout (seconds);
        ``None`` means "fall through to ``WORKTREE_SETUP_TIMEOUT_SEC`` / the
        built-in default" (resolved per-call by ``_resolve_setup_timeout``).
        It can be overridden per call via ``run(timeout=...)``.
        """

        self._log_root = log_root
        self._env = env if env is not None else os.environ
        self._runner = runner or _default_popen
        self.timeout = timeout

    def run(
        self,
        *,
        setup: Sequence[SetupStep],
        worktree_id: str,
        worktree_path: Path,
        branch: str,
        port_mapping: Optional[Dict[str, int]] = None,
        isolation: str = "full",
        timeout: Optional[float] = None,
    ) -> SetupResult:
        """Run all ``setup`` steps in order. Returns a structured result.

        On the first non-zero exit code, raises ``SetupFailedError`` and
        attaches the partial run via ``error.<...>``. The caller (W2's
        ``worktree_create``) is responsible for setting state to
        ``setup_failed`` and leaving the worktree intact for user inspection.

        A step that overruns its timeout also raises ``SetupFailedError``
        (with ``.timeout`` set) after killing the wedged process -- see
        ``_invoke``.

        ``timeout``, when not ``None``, overrides ``self.timeout`` for this
        call. Either way the value is resolved through
        ``_resolve_setup_timeout`` at the point of use, so the
        ``WORKTREE_SETUP_TIMEOUT_SEC`` env default applies automatically even
        when nobody opts in explicitly.
        """

        result = SetupResult(worktree_id=worktree_id)
        if isolation == "none" or not setup:
            return result

        log_dir = (
            self._log_root / worktree_id
            if self._log_root is not None
            else log_dir_for(worktree_id, env=dict(self._env))
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        injected_env = self._build_env(
            worktree_id=worktree_id,
            worktree_path=worktree_path,
            branch=branch,
            port_mapping=port_mapping or {},
        )

        requested_timeout = timeout if timeout is not None else self.timeout

        for index, step in enumerate(setup):
            step_name = step.name or f"step-{index}"
            log_path = log_dir / f"setup-{index:02d}-{_slug(step_name)}.log"
            shell_cmd = _resolve_shell(getattr(step, "shell", None))

            rc = self._invoke(
                shell_cmd=shell_cmd,
                run_line=step.run,
                cwd=worktree_path,
                env=injected_env,
                log_path=log_path,
                step_index=index,
                step_name=step_name,
                worktree_id=worktree_id,
                timeout=requested_timeout,
            )
            step_result = SetupStepResult(
                index=index, name=step_name, returncode=rc, log_path=log_path
            )
            result.steps.append(step_result)
            if rc != 0:
                result.aborted_at = index
                raise SetupFailedError(
                    worktree_id=worktree_id,
                    step_index=index,
                    step_name=step_name,
                    log_path=log_path,
                    returncode=rc,
                )

        return result

    # ---- internals ----

    def _build_env(
        self,
        *,
        worktree_id: str,
        worktree_path: Path,
        branch: str,
        port_mapping: Dict[str, int],
    ) -> Dict[str, str]:
        # Start from a complete user-profile environment (registry-sourced on
        # Windows) so that setup steps inherit APPDATA, LOCALAPPDATA, etc.
        # Then overlay self._env (the test-injection seam) so that callers can
        # supply a custom base environment (e.g. in unit tests) and it still
        # wins over the OS-derived base.
        env = _get_user_profile_env()
        env.update(self._env)  # test-injection seam: self._env overlays the base
        env["WORKTREE_ID"] = worktree_id
        env["WORKTREE_PATH"] = str(worktree_path)
        env["WORKTREE_BRANCH"] = branch
        for slot, port in port_mapping.items():
            env[f"WORKTREE_PORT_{slot.upper()}"] = str(port)
        return env

    def _invoke(
        self,
        *,
        shell_cmd: List[str],
        run_line: str,
        cwd: Path,
        env: Dict[str, str],
        log_path: Path,
        step_index: int,
        step_name: str,
        worktree_id: str,
        timeout: Optional[float] = None,
    ) -> int:
        """Run one step's process and drive its timeout/kill sequence.

        Mirrors ``core/_git_utils._run_git``'s hardened pattern: the
        ``self._runner`` seam returns a Popen-like object and this method
        drives ``communicate(timeout=...)`` itself. On overrun: kill the
        process, attempt a bounded (5s) post-kill drain (swallowing a second
        ``TimeoutExpired`` from that drain so it can never itself hang), write
        a synthetic ``returncode=-1`` log entry noting the timeout, then raise
        ``SetupFailedError`` (with ``.timeout`` set) -- no new exception type.

        An effective timeout of ``None`` disables the timeout entirely
        (block-forever opt-out), matching how the git/plugin-install timeout
        subsystems behave when disabled via empty-string env.
        """
        cmd = [*shell_cmd, run_line]
        proc = self._runner(cmd, cwd=str(cwd), env=env)
        effective_timeout = _resolve_setup_timeout(timeout)

        try:
            stdout, stderr = proc.communicate(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            # Drain the pipes after kill so the child fully reaps; bound this
            # too so a stuck drain can never itself hang the runner.
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            header = (
                f"# setup step {step_index} ({step_name})\n"
                f"# cmd: {shell_cmd[0]} ... -c {run_line!r}\n"
                f"# returncode: -1 (timed out after {effective_timeout}s)\n"
                f"# ---- stdout ----\n"
            )
            with log_path.open("w", encoding="utf-8") as fh:
                fh.write(header)
                fh.write("\n# ---- stderr ----\n")
                fh.write("setup step timed out and was killed\n")
            raise SetupFailedError(
                worktree_id=worktree_id,
                step_index=step_index,
                step_name=step_name,
                log_path=log_path,
                returncode=-1,
                timeout=effective_timeout,
            ) from None

        header = (
            f"# setup step {step_index} ({step_name})\n"
            f"# cmd: {shell_cmd[0]} ... -c {run_line!r}\n"
            f"# returncode: {proc.returncode}\n"
            f"# ---- stdout ----\n"
        )
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            fh.write(stdout or "")
            fh.write("\n# ---- stderr ----\n")
            fh.write(stderr or "")
        return int(proc.returncode)


__all__ = (
    "DEFAULT_LOG_ROOT",
    "LOG_ROOT_ENV",
    "SetupFailedError",
    "SetupResult",
    "SetupRunner",
    "SetupStep",
    "SetupStepResult",
    "_PlainStep",
    "_resolve_setup_timeout",
    "log_dir_for",
)
