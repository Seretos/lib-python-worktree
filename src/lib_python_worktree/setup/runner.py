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

import base64
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..core._env_utils import _get_user_profile_env
from ..core._exceptions import RunLineExpansionError


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


_LOWER_PRIORITY_ENV = "WORKTREE_SETUP_LOWER_PRIORITY"
_LOWER_PRIORITY_DEFAULT = True
_LOWER_PRIORITY_DISABLED_VALUES = {"", "0", "false", "no", "off"}


def _resolve_lower_priority() -> bool:
    """Resolve whether setup-step subprocesses spawn at lowered OS priority.

    Precedence: ``WORKTREE_SETUP_LOWER_PRIORITY`` env var; unset -> enabled
    (built-in default ``True``). When set, the value is ``.strip().lower()``-ed
    and treated as *disabled* (``False``) for ``""``, ``"0"``, ``"false"``,
    ``"no"``, or ``"off"``; any other value is *enabled* (``True``). Unlike
    ``_resolve_setup_timeout`` (float-based), this repo has no existing
    boolean-env convention, so this resolver defines its own. Env is read on
    every call so test fixtures can flip it via monkeypatch without
    re-importing the module.
    """
    raw = os.environ.get(_LOWER_PRIORITY_ENV)
    if raw is None:
        return _LOWER_PRIORITY_DEFAULT
    return raw.strip().lower() not in _LOWER_PRIORITY_DISABLED_VALUES


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


# Interpreter names for which _build_step_command switches from a raw
# "-Command <text>" argument to "-EncodedCommand <base64 blob>" (ticket #109).
_POWERSHELL_INTERPRETERS = ("powershell.exe", "pwsh")

# Appended (ticket #134) to every PowerShell/pwsh run line before it is
# base64-encoded. When a -Command/-EncodedCommand script ends without
# executing an `exit` statement of its own, the PowerShell host derives its
# *own* process exit code from `$?` (0 when true, 1 when false) and discards
# `$LASTEXITCODE` entirely -- so a native child's real exit code (e.g. `cmd
# /c "... & exit 3"`, or a self-wrapped `powershell ... -Command "exit 3"`)
# is silently collapsed to a bare 1. This epilogue recovers the real code on
# failure while leaving PowerShell's own pass/fail verdict (`$?`) as the
# authoritative signal -- see `_build_step_command`'s docstring for the full
# rationale and the design decisions already made for this shape.
_PS_EXIT_CODE_EPILOGUE = (
    "\n$__wt_ok = $?\n"
    "$__wt_rc = Get-Variable -Name LASTEXITCODE -Scope Global -ValueOnly "
    "-ErrorAction SilentlyContinue\n"
    "if ($__wt_ok) { exit 0 }\n"
    "if ($null -ne $__wt_rc -and $__wt_rc -ne 0) { exit $__wt_rc }\n"
    "exit 1\n"
)


# Anchors the *first* token of a run line as a nested powershell/pwsh
# invocation -- optionally quoted (in which case the path prefix may contain
# spaces, since the quotes are what let a real run line spell a path like
# ``"C:\Program Files\PowerShell\7\pwsh.exe"``), optionally prefixed by a
# directory path ending in a path separator, optionally suffixed with
# ``.exe``. Anything else (``cmd``, ``npm``, ``Get-Item``, ``echo``, ...)
# does not match, so _ps_double_evaluated_tokens short-circuits to `[]`
# immediately for it.
_PS_NESTED_INTERPRETER_RE = re.compile(
    r'^\s*(?:'
    r'"(?:[^"]*[\\/])?(?:powershell|pwsh)(?:\.exe)?"'
    r'|(?:[^\s"]*[\\/])?(?:powershell|pwsh)(?:\.exe)?'
    r')(?=\s|$)',
    re.IGNORECASE,
)

# `$`-expandable token, matched at a fixed start position (via `.match(s,
# pos)`) once the caller's own quote-tracking walk has already established
# that the position is inside an unescaped double-quoted region: `$env:X`,
# `$true`, `$_`, `$(...)` (subexpression), `${x}` (braced). Backtick-escape
# parity (escape #2) is handled by the walk itself, not by this regex.
_PS_EXPANDABLE_TOKEN_RE = re.compile(r"\$(?:\{[^}]*\}|\(|[A-Za-z_?^$][\w:]*)")


def _ps_double_evaluated_tokens(run_line: str) -> List[str]:
    """Detect a PowerShell-routed ``run:`` line whose first token is a nested
    ``powershell``/``pwsh`` invocation carrying a double-quoted argument that
    contains a ``$``-expandable token (ticket #158).

    The corruption this guards against: when such a run line is executed by
    the *outer* PowerShell host (the interpreter ``_resolve_shell`` selects),
    that host's own double-quoted-string interpolation expands ``$env:X``,
    ``$true``, etc. *before* the nested ``powershell``/``pwsh`` child process
    ever sees its ``-Command`` argument -- silently mangling it into a syntax
    error the inner shell then fails on (or, in the ticket's reported case,
    into something that still parses but does nothing useful), while the
    caller (``start()``/``SetupRunner.run()``) is none the wiser.

    Detection is deliberately narrow -- no general-purpose PowerShell parser:

    1. ``run_line`` must start with a nested ``powershell``/``pwsh``
       interpreter token (optionally quoted, optionally a path containing
       spaces when quoted). Anything else returns ``[]`` immediately.
    2. The remainder is walked **once**, left to right, tracking a single
       piece of state -- "currently outside any quotes", "inside a
       single-quoted region", or "inside a double-quoted region" -- rather
       than scanning with independent regexes that cannot see each other's
       context. This is what makes the three escapes compose correctly
       instead of each being evaluated as if it were the only one in play:
       - **Escape #1 (single-quoted):** while inside a single-quoted region,
         nothing is inspected at all -- not even a literal ``"..."`` or
         ``$...`` appearing inside it, since the outer host never
         interpolates inside a single-quoted string. A double-quoted-looking
         segment nested inside an outer single-quoted argument (e.g.
         ``'Write-Output "$env:X"'``) is therefore correctly never scanned.
       - **Escape #2 (backtick-escape):** while inside a double-quoted
         region, a run of consecutive backticks immediately before a ``$``
         (or a literal ``"``) is counted; an **odd** count means the last
         backtick escapes that following character (a ``$`` is left as
         inert literal text, a ``"`` does not end the segment) and an
         **even** count means the backticks escape each other in pairs,
         leaving the following character with its ordinary meaning (a ``$``
         is genuinely expandable and IS collected, a ``"`` DOES end the
         segment) -- a single-character lookbehind cannot express this
         parity rule, only a walk that counts the run can.
       - **Escape #3 (``--%`` stop-parsing token):** a standalone ``--%``
         token is only recognised while the walk is *outside* any quoted
         region -- literal text that merely looks like ``--%`` sitting
         inside an earlier single- or double-quoted argument is just text
         and never trips this escape. Hitting a genuine standalone ``--%``
         stops the walk immediately (everything after it is passed through
         PowerShell's parser verbatim, with no interpolation at all, so
         nothing past it can be double-evaluated) but does not discard
         ``$``-tokens already collected from segments seen before it.

    Returns the list of offending ``$``-token strings found (empty when the
    line does not match the dangerous shape).
    """
    match = _PS_NESTED_INTERPRETER_RE.match(run_line)
    if not match:
        return []

    remainder = run_line[match.end():]
    n = len(remainder)
    tokens: List[str] = []
    state: Optional[str] = None  # None | "single" | "double"
    i = 0
    while i < n:
        ch = remainder[i]
        if state == "double":
            if ch == "`":
                j = i
                while j < n and remainder[j] == "`":
                    j += 1
                backtick_count = j - i
                escaped = backtick_count % 2 == 1
                i = j
                if i < n and remainder[i] in ('"', "$"):
                    if escaped:
                        # the trailing backtick escapes the next char: treat
                        # it as an inert literal and step past it.
                        i += 1
                    elif remainder[i] == '"':
                        state = None
                        i += 1
                    else:  # genuinely-expandable '$' (even backtick count)
                        m = _PS_EXPANDABLE_TOKEN_RE.match(remainder, i)
                        if m:
                            tokens.append(m.group(0))
                            i = m.end()
                        else:
                            i += 1
                continue
            if ch == '"':
                state = None
                i += 1
                continue
            if ch == "$":
                m = _PS_EXPANDABLE_TOKEN_RE.match(remainder, i)
                if m:
                    tokens.append(m.group(0))
                    i = m.end()
                else:
                    i += 1
                continue
            i += 1
            continue
        if state == "single":
            if ch == "'":
                state = None
            i += 1
            continue
        # state is None: outside any quoted region
        if ch == "'":
            state = "single"
            i += 1
            continue
        if ch == '"':
            state = "double"
            i += 1
            continue
        if remainder.startswith("--%", i):
            before_ok = i == 0 or remainder[i - 1].isspace()
            after_idx = i + 3
            after_ok = after_idx >= n or remainder[after_idx].isspace()
            if before_ok and after_ok:
                break  # genuine stop-parsing token: nothing after it parses
        i += 1
    return tokens


def _resolve_shell(step_shell: Optional[str]) -> List[str]:
    """Return the ``[shell, "-c"/"-Command"-equivalent]`` prefix for a step.

    This picks the interpreter and its flags only; joining the prefix with
    the run line is done later by ``_build_step_command`` (the sole seam
    that appends a run line to a resolved shell prefix).

    Override values map as:
    - ``bash`` / ``sh``  → ``["<name>", "-c"]``
    - ``pwsh``           → ``["pwsh", "-NoProfile", "-NonInteractive", "-Command"]``
    - ``powershell``     → ``["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]``

    With no override, picks ``powershell.exe`` on Windows and ``bash`` elsewhere.
    ``-NonInteractive`` is included for both PowerShell variants so a step
    that would otherwise prompt fails loudly (``_default_popen`` already sets
    ``stdin=DEVNULL``) instead of hanging or exiting obscurely.

    The trailing ``-Command`` entry on the PowerShell/pwsh prefixes is a
    placeholder consumed by ``_build_step_command``, which replaces it with
    ``-EncodedCommand <blob>`` rather than appending the run text after it
    verbatim (ticket #109: raw ``-Command <text>`` round-trips through both
    ``subprocess.list2cmdline`` re-quoting and PowerShell's own re-parsing,
    which mangles a self-wrapped ``run:`` value's quote structure).
    ``_build_step_command`` also appends an exit-code-propagating epilogue
    to the encoded blob for these two interpreters only (ticket #134) --
    see its docstring, including the **known limitation** around a stale
    global ``$LASTEXITCODE`` documented there.
    """

    if step_shell:
        name = step_shell.lower()
        if name == "bash":
            return ["bash", "-c"]
        if name == "sh":
            return ["sh", "-c"]
        if name == "pwsh":
            return ["pwsh", "-NoProfile", "-NonInteractive", "-Command"]
        if name == "powershell":
            return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
        raise ValueError(f"unknown step shell: {step_shell!r}")

    if sys.platform == "win32":
        return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    return ["bash", "-c"]


def _build_step_command(shell_cmd: List[str], run_line: str) -> List[str]:
    """Build the final argv for a step from a resolved shell prefix + run line.

    This is the *only* place a shell prefix (from ``_resolve_shell``) and a
    run line are joined into an executable argv. Both ``SetupRunner._invoke``
    and ``WorktreeManager.start`` funnel through this function so the fix
    applies identically to ``setup:`` and ``start:`` steps.

    For the ``powershell.exe`` / ``pwsh`` interpreters, the run line is
    transported as ``-EncodedCommand <base64(utf-16-le)>`` instead of a raw
    ``-Command <text>`` argument (ticket #109): the encoded blob contains no
    spaces or quotes, so neither ``subprocess.list2cmdline`` (Windows argv
    re-quoting) nor PowerShell's own command-line re-parsing (backslash is
    not an escape character there) can mangle a self-wrapped ``run:`` value's
    quote structure, e.g. ``powershell -NoProfile -Command "..."``.

    **Exit-code caveat (ticket #134):** unlike a plain native command, a
    ``-Command``/``-EncodedCommand`` script that ends without its own
    ``exit`` statement gets its *host* exit code derived from ``$?`` (0/1),
    discarding ``$LASTEXITCODE`` -- so a native child's real exit code (e.g.
    ``cmd /c "... & exit 3"``, or a self-wrapped inner ``powershell ...
    -Command "exit 3"``) silently collapses to a bare ``1``. To recover the
    real code, ``_PS_EXIT_CODE_EPILOGUE`` is appended to ``run_line`` before
    it is encoded, for the PowerShell/pwsh interpreters only -- it is never
    appended for ``bash -c``/``sh -c``, since POSIX shells already propagate
    the last command's status via their own exit code. The epilogue emits
    nothing on stdout/stderr and keeps ``$?`` as the authoritative
    pass/fail verdict, only using ``$LASTEXITCODE`` to refine the numeric
    value on failure (so it can never turn an existing pass into a fail or
    vice versa). A run line that itself ends in an ``exit`` statement (e.g.
    the bare ``if (...) { exit 3 } else { exit 0 }`` shape) already
    propagates correctly without the epilogue's help, since ``exit``
    terminates the host directly.

    **Known limitation -- stale global ``$LASTEXITCODE`` (ticket #134
    review):** ``$__wt_rc`` is read from ``$LASTEXITCODE``, which is a
    *global* that any earlier native command in the script can have set,
    regardless of whether that earlier command is the actual cause of the
    script's final failure. The common/reported case -- the failing
    statement IS the one that set ``$LASTEXITCODE`` (the script ends in
    ``exit N``, or the last statement is the failing native command) -- is
    fixed correctly by this epilogue. A script that chains an earlier
    *successful-but-nonzero-setting* native call with a later, unrelated
    failing statement (typically a cmdlet, which has no numeric exit code
    of its own) is a known edge case where the reported code can be
    misleading: e.g. ``run_line = 'cmd /c "exit 7"; Get-Item C:\\does\\not\\
    exist.xyz'`` reports ``7`` (the stale code left over from the earlier
    ``cmd /c`` call) even though the actual failure is the ``Get-Item``
    cmdlet, which would report a generic ``1`` under bare-``$?`` semantics.
    This is an accepted limitation, not a regression of the epilogue's own
    protected invariant (a native failure followed by a recovering
    non-failing statement, e.g. ``cmd /c exit 3; Write-Host ok``, still
    reports success/``0``) -- detecting whether the last statement's
    failure genuinely correlates with ``$LASTEXITCODE`` is hard in general
    and is deliberately out of scope here. Pinned by
    ``test_powershell_step_stale_lastexitcode_from_earlier_statement_is_a_
    known_limitation`` in ``tests/test_setup_runner.py``.

    Any other shell prefix (``bash -c`` / ``sh -c`` / an unrecognised
    prefix) is appended to verbatim, unchanged from before this fix.

    **Ticket #158:** before any of the above, a PowerShell/pwsh run line is
    scanned by ``_ps_double_evaluated_tokens`` for the double-evaluation
    shape (a nested ``powershell``/``pwsh`` invocation carrying a
    double-quoted argument with a ``$``-expandable token). A non-empty
    result raises ``RunLineExpansionError`` here, before ``run_line`` is
    ever base64-encoded and before any subprocess is spawned -- this is the
    only place that call happens; the ``bash -c``/``sh -c`` branch is never
    scanned at all.
    """

    if shell_cmd and shell_cmd[0] in _POWERSHELL_INTERPRETERS:
        tokens = _ps_double_evaluated_tokens(run_line)
        if tokens:
            raise RunLineExpansionError(run_line=run_line, tokens=tokens)
        prefix = shell_cmd[:-1] if shell_cmd[-1] == "-Command" else shell_cmd
        full_line = run_line + _PS_EXIT_CODE_EPILOGUE
        blob = base64.b64encode(full_line.encode("utf-16-le")).decode("ascii")
        return [*prefix, "-EncodedCommand", blob]
    return [*shell_cmd, run_line]


# ---- runner ------------------------------------------------------------------


def _default_popen(cmd: List[str], *, cwd: str, env: Dict[str, str]) -> subprocess.Popen:
    """Default ``self._runner`` implementation: a hardened ``subprocess.Popen``.

    Mirrors ``core/_git_utils._run_git``'s hardening: ``stdin=DEVNULL`` so the
    child can never inherit our stdin and wedge waiting on input, explicit
    ``stdout=PIPE``/``stderr=PIPE`` (rather than ``capture_output``) because
    ``_invoke`` drives ``communicate()``/kill itself, and
    ``creationflags=CREATE_NO_WINDOW`` on Windows to avoid a console flash.

    Priority lowering (ticket #68): a heavy setup step (e.g. ``npm install``)
    can starve unrelated concurrent work in the calling application via
    OS-level scheduling/IO contention. When ``_resolve_lower_priority()``
    resolves truthy (default: enabled, toggled via
    ``WORKTREE_SETUP_LOWER_PRIORITY``), the spawned subprocess is given a
    lowered OS scheduling/IO priority:

    - **Windows**: ``subprocess.BELOW_NORMAL_PRIORITY_CLASS`` is OR'd into
      ``creationflags`` alongside the existing ``CREATE_NO_WINDOW`` flag.
      This is deliberately *not* ``PROCESS_MODE_BACKGROUND_BEGIN`` -- that
      value is a self-only argument to ``SetPriorityClass`` (a process lowers
      its own priority with it) and is not a valid ``Popen`` creation flag,
      so it cannot be reliably applied to a spawned child.
    - **POSIX**: the argv is prefixed with ``nice -n 10`` (resolved via
      ``shutil.which("nice")``), further prefixed with ``ionice -c 3`` (idle
      I/O class) when ``shutil.which("ionice")`` also resolves. This is
      deliberately *not* ``preexec_fn=os.nice`` -- the stdlib documents
      ``preexec_fn`` as unsafe in a multithreaded process, which is exactly
      this library's environment (concurrent worktree operations run from a
      thread pool).

    The mechanism is best-effort: if the toggle is disabled, or ``nice``/
    ``ionice`` are not found on POSIX, the subprocess is spawned normally --
    this function never raises over an unavailable priority mechanism.
    """

    popen_kwargs: dict = {
        "cwd": cwd,
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }

    lower_priority = _resolve_lower_priority()

    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        if lower_priority:
            creationflags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS  # type: ignore[attr-defined]
        popen_kwargs["creationflags"] = creationflags
    elif lower_priority:
        nice_path = shutil.which("nice")
        if nice_path is not None:
            prefix = [nice_path, "-n", "10"]
            ionice_path = shutil.which("ionice")
            if ionice_path is not None:
                prefix = [ionice_path, "-c", "3", *prefix]
            cmd = [*prefix, *cmd]

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

        The log header always shows an accurate effective-argv line: for the
        ``-EncodedCommand`` transport (PowerShell/pwsh), the opaque base64
        blob is swapped back for the human-readable ``run_line`` so the log
        stays legible (ticket #109). When a step exits non-zero and produced
        no stdout/stderr at all (or only whitespace), a synthetic diagnostic
        note is appended to the stderr section naming the exit code,
        interpreter, and effective argv -- so a step can never again fail
        with `returncode: 1` and two empty sections and no clue why.
        """
        interpreter = shell_cmd[0] if shell_cmd else ""
        cmd = _build_step_command(shell_cmd, run_line)
        log_argv = list(cmd)
        if shell_cmd and shell_cmd[0] in _POWERSHELL_INTERPRETERS and log_argv:
            log_argv[-1] = run_line  # swap the base64 blob for readability
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
                f"# argv: {log_argv!r}\n"
                f"# run: {run_line!r}\n"
                f"# interpreter: {interpreter}\n"
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
            f"# argv: {log_argv!r}\n"
            f"# run: {run_line!r}\n"
            f"# interpreter: {interpreter}\n"
            f"# returncode: {proc.returncode}\n"
            f"# ---- stdout ----\n"
        )
        stdout_text = stdout or ""
        stderr_text = stderr or ""
        synthetic_note = ""
        if (
            proc.returncode != 0
            and not stdout_text.strip()
            and not stderr_text.strip()
        ):
            synthetic_note = (
                f"[worktree] step exited with code {proc.returncode} via "
                f"{interpreter!r} but produced no stdout or stderr output.\n"
                f"[worktree] effective argv: {log_argv!r}\n"
            )
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(header)
            fh.write(stdout_text)
            fh.write("\n# ---- stderr ----\n")
            fh.write(stderr_text)
            fh.write(synthetic_note)
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
    "_ps_double_evaluated_tokens",
    "_resolve_lower_priority",
    "_resolve_setup_timeout",
    "log_dir_for",
)
