"""Shared git-subprocess utilities for the worktree engine.

This module contains the single canonical ``_run_git`` implementation that
both ``manager.py`` and ``yaml_store.py`` use.  It exists to eliminate the
duplication that previously caused ``yaml_store._run_git`` to lack timeout
and kill hardening (the indefinite-hang bug from tickets #8/#19).

``GitTimeoutError`` is defined in ``_exceptions.py`` so that it is a
``WorktreeError`` subclass while ``_git_utils`` stays free of circular
imports (it does not need to import anything from ``manager``).

Public surface:
* ``_run_git(args, cwd, *, timeout)`` — the hardened Popen runner.
* ``_resolve_git_timeout(explicit)`` — resolves the effective timeout.
* ``GitTimeoutError`` — re-exported from ``_exceptions`` for callers that
  import it from here.

Both ``GitTimeoutError`` and ``_run_git`` are also re-exported from
``manager.py`` for backward compatibility with existing callers and tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from ._exceptions import GitTimeoutError  # noqa: F401 — re-exported

_GIT_TIMEOUT_ENV = "WORKTREE_GIT_TIMEOUT_SEC"
_GIT_TIMEOUT_DEFAULT = 30.0


# ---------------------------------------------------------------------------
# _resolve_git_timeout
# ---------------------------------------------------------------------------

def _resolve_git_timeout(explicit: Optional[float]) -> Optional[float]:
    """Resolve the timeout for a single ``_run_git`` call.

    Precedence: explicit kwarg > ``WORKTREE_GIT_TIMEOUT_SEC`` env > built-in
    default of 30.0 s.  ``None`` (either as kwarg or env value ``""``) disables
    the timeout entirely; that path exists for diagnostics, not normal use.

    Env is read on every call so that test fixtures and operators can change
    the value without re-importing the module.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get(_GIT_TIMEOUT_ENV)
    if raw is None:
        return _GIT_TIMEOUT_DEFAULT
    raw = raw.strip()
    if not raw:
        # Empty string is "no timeout", matching the explicit-None semantics.
        return None
    try:
        value = float(raw)
    except ValueError:
        return _GIT_TIMEOUT_DEFAULT
    return value if value > 0 else None


# ---------------------------------------------------------------------------
# _run_git
# ---------------------------------------------------------------------------

def _run_git(
    args: List[str],
    cwd: Optional[Path] = None,
    *,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Run ``git <args>`` and return a ``CompletedProcess``.

    Ticket #19 hardenings:
    * ``stdin=DEVNULL`` so ``git`` can never inherit the MCP client's stdin
      pipe and wedge waiting on input -- this was the Windows-exe hang root
      cause.
    * Explicit ``stdout=PIPE, stderr=PIPE`` (rather than ``capture_output``)
      because we now drive a ``Popen`` directly to keep a clean kill path.
    * On Windows: ``creationflags=CREATE_NO_WINDOW`` so packaged-exe runs
      don't briefly flash a console window per git call.
    * ``timeout`` defaults from ``WORKTREE_GIT_TIMEOUT_SEC`` (30 s if unset);
      on overrun the process is killed and ``GitTimeoutError`` is raised.
    """
    effective_timeout = _resolve_git_timeout(timeout)

    popen_kwargs: dict = {
        "cwd": str(cwd) if cwd else None,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if sys.platform == "win32":
        # Suppress the brief console-window flash when the packaged worktree.exe
        # spawns git from a GUI MCP host.
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    cmd = ["git", *args]
    start = time.monotonic()
    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=effective_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        # Drain the pipes after kill so the child fully reaps; ignore output.
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        elapsed = time.monotonic() - start
        raise GitTimeoutError(cmd, elapsed) from None

    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr
    )


__all__ = [
    "GitTimeoutError",
    "_resolve_git_timeout",
    "_run_git",
]
