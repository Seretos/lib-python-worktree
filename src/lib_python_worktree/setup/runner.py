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
    ) -> None:
        super().__init__(
            f"setup step {step_index} ({step_name!r}) for worktree "
            f"{worktree_id!r} failed with exit code {returncode}. "
            f"See log: {log_path}"
        )
        self.worktree_id = worktree_id
        self.step_index = step_index
        self.step_name = step_name
        self.log_path = log_path
        self.returncode = returncode


# ---- path + shell helpers ----------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")
LOG_ROOT_ENV = "WORKTREE_LOG_ROOT"
DEFAULT_LOG_ROOT = Path("~/.agent-worktree/logs").expanduser()


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


class SetupRunner:
    """Executes a contract's ``setup`` steps in a worktree."""

    def __init__(
        self,
        *,
        log_root: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        runner: Optional[Any] = None,
    ) -> None:
        """``runner`` is an injection seam used by tests (defaults to
        ``subprocess.run``)."""

        self._log_root = log_root
        self._env = env if env is not None else os.environ
        self._runner = runner or subprocess.run

    def run(
        self,
        *,
        setup: Sequence[SetupStep],
        worktree_id: str,
        worktree_path: Path,
        branch: str,
        port_mapping: Optional[Dict[str, int]] = None,
        isolation: str = "full",
    ) -> SetupResult:
        """Run all ``setup`` steps in order. Returns a structured result.

        On the first non-zero exit code, raises ``SetupFailedError`` and
        attaches the partial run via ``error.<...>``. The caller (W2's
        ``worktree_create``) is responsible for setting state to
        ``setup_failed`` and leaving the worktree intact for user inspection.
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
    ) -> int:
        proc = self._runner(
            [*shell_cmd, run_line],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        header = (
            f"# setup step {step_index} ({step_name})\n"
            f"# cmd: {shell_cmd[0]} ... -c {run_line!r}\n"
            f"# returncode: {proc.returncode}\n"
            f"# ---- stdout ----\n"
        )
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            fh.write(proc.stdout or "")
            fh.write("\n# ---- stderr ----\n")
            fh.write(proc.stderr or "")
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
    "log_dir_for",
)
