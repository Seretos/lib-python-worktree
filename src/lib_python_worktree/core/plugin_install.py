"""Clone-first install of a worktree's ``enabledPlugins`` (tickets #62, #64).

Since Claude Code v2.1.195, a plugin declared in a project's
``.claude/settings.json`` ``enabledPlugins`` block fails to load unless there
is a registered installation for the *exact* project path in
``~/.claude/plugins/installed_plugins.json``.  A freshly-created git worktree
never has such a registration, so plugins silently fail to load.

For every truthy key in the merged ``enabledPlugins`` map (``settings.json``
+ ``settings.local.json``, local wins per-key), this module now:

1. Looks for an existing, structurally-valid registry entry for that key
   (any scope/projectPath — it only needs a real install on disk) and, if
   found, **clones** it under a lock into a new ``scope: "project"`` entry
   pointed at the worktree path. This is the **primary** mechanism (ticket
   #64): it never shells out, so it cannot hit the Windows
   ``EPERM``/``rm``-style failures that ``claude plugin install`` can trigger
   against files the CLI itself has open.
2. Falls back to shelling out to::

       claude plugin install <name>@<marketplace> --scope project

   with ``cwd`` set to the new worktree, only when no valid clone source
   exists. If the CLI invocation itself fails, a **second-chance clone** is
   attempted before giving up (the CLI may have partially populated the
   registry with a now-valid source).

``core.plugin_seed.seed_plugin_registry`` (ticket #39, workaround for
anthropics/claude-code#61866) is no longer wired from ``manager.py`` as of
#64 — the clone-first mechanism above supersedes it. See that module's
docstring for removal conditions.

The operation is:

- **Best-effort**: the bare ``try/except: pass`` wrapper lives at the call
  site in ``manager.py``, *not* here.  This module never raises for expected
  "nothing to do" conditions (missing settings files, malformed JSON, no CLI
  on PATH) — it reports them via the returned ``PluginInstallResult`` instead.
- **Idempotent**: a plugin key already registered with ``scope: "project"``,
  a matching ``projectPath`` for this worktree, and a *structurally valid*
  ``installPath`` is skipped rather than re-installed. A broken registration
  (missing/corrupt ``installPath``) is treated as not-yet-installed so it
  self-repairs.
- **Batch-resilient**: a failure (nonzero exit, timeout, spawn error) for one
  plugin key never aborts the remaining keys in the batch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import portalocker

from ..setup.runner import _slug, log_dir_for
from .yaml_store import _LOCK_FLAGS, _LOCK_TIMEOUT

_INSTALL_TIMEOUT_ENV = "WORKTREE_PLUGIN_INSTALL_TIMEOUT_SEC"
_INSTALL_TIMEOUT_DEFAULT = 60.0


# ---------------------------------------------------------------------------
# PluginInstallResult
# ---------------------------------------------------------------------------


@dataclass
class PluginInstallResult:
    """Outcome of an :func:`install_enabled_plugins` call."""

    installed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    claude_unavailable: bool = False


# ---------------------------------------------------------------------------
# _resolve_install_timeout — mirrors _git_utils._resolve_git_timeout exactly
# ---------------------------------------------------------------------------


def _resolve_install_timeout(explicit: Optional[float]) -> Optional[float]:
    """Resolve the timeout for a single ``claude plugin install`` invocation.

    Precedence: explicit kwarg > ``WORKTREE_PLUGIN_INSTALL_TIMEOUT_SEC`` env >
    built-in default of 60.0 s.  ``None`` (either as kwarg or env value
    ``""``) disables the timeout entirely.  Env is read on every call so test
    fixtures can change it without re-importing the module.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get(_INSTALL_TIMEOUT_ENV)
    if raw is None:
        return _INSTALL_TIMEOUT_DEFAULT
    raw = raw.strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return _INSTALL_TIMEOUT_DEFAULT
    return value if value > 0 else None


# ---------------------------------------------------------------------------
# enabledPlugins settings merge
# ---------------------------------------------------------------------------


def _read_json_object(path: Path) -> dict:
    """Read *path* as a JSON object; missing/malformed -> ``{}`` (no raise)."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_enabled_plugins(repo_root: str) -> List[str]:
    """Return the truthy ``enabledPlugins`` keys for *repo_root*.

    Reads ``<repo_root>/.claude/settings.json`` then
    ``<repo_root>/.claude/settings.local.json`` (both optional).  Local
    overrides base per-key -- including a local ``false`` disabling a base
    ``true``.  Only truthy values are returned.  MUST read from repo_root,
    never worktree_path: ``.claude/`` is never copied into the worktree.
    """
    claude_dir = Path(repo_root) / ".claude"
    base = _read_json_object(claude_dir / "settings.json")
    local = _read_json_object(claude_dir / "settings.local.json")

    merged: Dict[str, object] = {}
    base_enabled = base.get("enabledPlugins")
    if isinstance(base_enabled, dict):
        merged.update(base_enabled)
    local_enabled = local.get("enabledPlugins")
    if isinstance(local_enabled, dict):
        merged.update(local_enabled)

    return [key for key, value in merged.items() if value]


# ---------------------------------------------------------------------------
# Idempotency check against installed_plugins.json
# ---------------------------------------------------------------------------


def _default_config_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser()


def _load_registry(config_dir: Path) -> dict:
    """Load Schema-v2 ``installed_plugins.json``; anything else -> ``{}``."""
    registry_path = config_dir / "plugins" / "installed_plugins.json"
    if not registry_path.exists():
        return {}
    try:
        raw = registry_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return {}
    if not (
        isinstance(data, dict)
        and data.get("version") == 2
        and isinstance(data.get("plugins"), dict)
    ):
        return {}
    return data


def _already_registered(registry_data: dict, key: str, worktree_path: str) -> bool:
    """True if *key* already has a project-scoped entry for *worktree_path*.

    Uses the same ``Path(...)``/``os.path.normcase`` normalisation approach
    as ``plugin_seed.seed_plugin_registry`` for Windows-safe path comparison.
    A matching entry only counts as "already registered" if its
    ``installPath`` is structurally valid (ticket #64) — a registration
    pointing at a missing/broken install must fall through so the worktree
    self-repairs instead of staying silently broken.
    """
    plugins = registry_data.get("plugins") if isinstance(registry_data, dict) else None
    if not isinstance(plugins, dict):
        return False
    entry_list = plugins.get(key)
    if not isinstance(entry_list, list):
        return False

    norm_wt = os.path.normcase(str(Path(worktree_path)))
    for entry in entry_list:
        if not isinstance(entry, dict):
            continue
        if entry.get("scope") != "project":
            continue
        project_path = entry.get("projectPath")
        if not isinstance(project_path, str):
            continue
        if os.path.normcase(str(Path(project_path))) == norm_wt and _is_structurally_valid(
            entry.get("installPath")
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Clone-first registration (ticket #64)
# ---------------------------------------------------------------------------


def _is_structurally_valid(install_path: Optional[str]) -> bool:
    """True if *install_path* points at a real, parseable plugin install.

    Requires ``<install_path>/.claude-plugin/plugin.json`` to exist and
    parse as JSON. This is the single validity predicate reused by the clone
    source picker and by :func:`_already_registered`.
    """
    if not install_path or not isinstance(install_path, str):
        return False
    manifest = Path(install_path) / ".claude-plugin" / "plugin.json"
    try:
        json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _find_clone_source(registry_data: dict, key: str) -> Optional[dict]:
    """Return the best structurally-valid registry entry to clone for *key*.

    Any scope/projectPath is acceptable — it is only used as a read-only
    template for the ``installPath`` (and version metadata) of an existing,
    on-disk plugin install. Returns ``None`` when no valid candidate exists.

    Picks the entry with the newest ``(installedAt, resolvedVersion)`` pair;
    missing fields sort oldest. Ties are broken in favour of the
    later-listed entry (Python's stable sort + ``max`` keeps the last
    maximum encountered).
    """
    plugins = registry_data.get("plugins") if isinstance(registry_data, dict) else None
    if not isinstance(plugins, dict):
        return None
    entry_list = plugins.get(key)
    if not isinstance(entry_list, list):
        return None

    candidates = [
        entry
        for entry in entry_list
        if isinstance(entry, dict) and _is_structurally_valid(entry.get("installPath"))
    ]
    if not candidates:
        return None

    def _sort_key(entry: dict) -> tuple:
        return (entry.get("installedAt") or "", entry.get("resolvedVersion") or "")

    best = candidates[0]
    best_key = _sort_key(best)
    for entry in candidates[1:]:
        entry_key = _sort_key(entry)
        if entry_key >= best_key:
            best = entry
            best_key = entry_key
    return best


def _clone_entry_to_worktree(
    config_dir: Path, key: str, source_entry: dict, worktree_path: str
) -> bool:
    """Clone *source_entry* into a new project-scoped entry for *worktree_path*.

    Performs the full read-modify-write under an exclusive lock on the
    registry's lock file so concurrent worktree creations never race each
    other or lose an update. Returns ``True`` if a new entry was written,
    ``False`` if nothing was written (already present, or the registry is
    not a valid Schema-v2 document to write into).
    """
    registry_path = config_dir / "plugins" / "installed_plugins.json"
    lock_path = str(registry_path) + ".lock"

    with portalocker.Lock(lock_path, timeout=_LOCK_TIMEOUT, flags=_LOCK_FLAGS):
        data = _load_registry(config_dir)
        if not data:
            return False

        plugins: Dict[str, list] = data["plugins"]
        dest = str(Path(worktree_path))
        norm_dest = os.path.normcase(dest)

        entry_list = plugins.get(key)
        if not isinstance(entry_list, list):
            entry_list = []
            plugins[key] = entry_list

        for entry in entry_list:
            if not isinstance(entry, dict):
                continue
            if (
                entry.get("scope") == "project"
                and isinstance(entry.get("projectPath"), str)
                and os.path.normcase(str(Path(entry["projectPath"]))) == norm_dest
                and entry.get("installPath") == source_entry.get("installPath")
            ):
                return False

        cloned = dict(source_entry)
        cloned["scope"] = "project"
        cloned["projectPath"] = dest
        entry_list.append(cloned)

        tmp_path: Optional[str] = None
        try:
            registry_dir = registry_path.parent
            fd, tmp_path = tempfile.mkstemp(
                dir=str(registry_dir), suffix=".tmp", prefix="installed_plugins_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2)
                    fh.flush()
            except Exception:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise
            os.replace(tmp_path, str(registry_path))
            tmp_path = None
        except Exception:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    return True


# ---------------------------------------------------------------------------
# claude executable resolution
# ---------------------------------------------------------------------------


def _resolve_claude_exe(
    which: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[str]:
    """Resolve the ``claude`` executable, or ``None`` if not on PATH."""
    _which = which if which is not None else shutil.which
    exe = _which("claude")
    if exe:
        return exe
    if sys.platform == "win32":
        for candidate in ("claude.cmd", "claude.exe"):
            exe = _which(candidate)
            if exe:
                return exe
    return None


# ---------------------------------------------------------------------------
# subprocess invocation (mirrors _git_utils._run_git hardening)
# ---------------------------------------------------------------------------


def _write_install_log(
    log_path: Path,
    cmd: List[str],
    returncode: int,
    stdout: str,
    stderr: str,
    *,
    key: str,
) -> None:
    header = (
        f"# plugin install: {key}\n"
        f"# cmd: {' '.join(cmd)}\n"
        f"# returncode: {returncode}\n"
        f"# ---- stdout ----\n"
    )
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(stdout or "")
        fh.write("\n# ---- stderr ----\n")
        fh.write(stderr or "")


def _run_install(
    exe: str,
    key: str,
    cwd: str,
    *,
    timeout: Optional[float],
    log_path: Path,
    runner: Optional[Callable] = None,
) -> "tuple[int, bool]":
    """Run ``<exe> plugin install <key> --scope project`` in *cwd*.

    Returns ``(returncode, timed_out)``. ``returncode`` is ``-1`` for a
    timeout or a spawn error (``OSError``) -- callers treat any nonzero
    ``returncode`` as failure. If *runner* is supplied it replaces the whole
    subprocess invocation (the test seam): it must accept
    ``(cmd, *, cwd, timeout)`` and either return an object with
    ``.returncode``/``.stdout``/``.stderr``, or raise
    ``subprocess.TimeoutExpired`` / ``OSError``.
    """
    cmd = [exe, "plugin", "install", key, "--scope", "project"]

    if runner is not None:
        try:
            proc = runner(cmd, cwd=str(cwd), timeout=timeout)
        except subprocess.TimeoutExpired:
            _write_install_log(log_path, cmd, -1, "", "timed out", key=key)
            return (-1, True)
        except OSError as exc:
            _write_install_log(log_path, cmd, -1, "", str(exc), key=key)
            return (-1, False)
        rc = int(proc.returncode)
        _write_install_log(
            log_path, cmd, rc, getattr(proc, "stdout", "") or "", getattr(proc, "stderr", "") or "", key=key
        )
        return (rc, False)

    popen_kwargs: dict = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except OSError as exc:
        _write_install_log(log_path, cmd, -1, "", str(exc), key=key)
        return (-1, False)

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        _write_install_log(log_path, cmd, -1, "", "timed out", key=key)
        return (-1, True)

    rc = int(proc.returncode)
    _write_install_log(log_path, cmd, rc, stdout, stderr, key=key)
    return (rc, False)


# ---------------------------------------------------------------------------
# install_enabled_plugins
# ---------------------------------------------------------------------------


def install_enabled_plugins(
    repo_root: str,
    worktree_path: str,
    *,
    worktree_id: Optional[str] = None,
    config_dir: Optional[Path] = None,
    timeout: Optional[float] = None,
    runner: Optional[Callable] = None,
    which: Optional[Callable[[str], Optional[str]]] = None,
) -> PluginInstallResult:
    """Install *repo_root*'s ``enabledPlugins`` into *worktree_path*.

    Clone-first primary mechanism (ticket #64): for every enabled plugin key
    not already validly project-registered for this worktree, first tries to
    register it by cloning an existing, structurally-valid registry entry
    (any scope) under a lock — this never shells out, so it cannot hit the
    Windows ``EPERM``-style failures that ``claude plugin install`` can
    trigger. Only when no valid clone source exists does it fall back to::

        claude plugin install <key> --scope project

    with ``cwd`` set to the worktree. If the CLI invocation itself fails, one
    more clone attempt is made (the CLI may have partially populated the
    registry) before the key is recorded as failed. When the ``claude`` CLI
    cannot be resolved on PATH at all, ``claude_unavailable=True`` is still
    set for observability, but clone-first continues to run regardless.

    Never raises for expected conditions; per-key failures are collected
    into ``failed`` and the batch continues.
    """
    result = PluginInstallResult()

    keys = _read_enabled_plugins(repo_root)
    if not keys:
        return result

    exe = _resolve_claude_exe(which)
    if exe is None:
        result.claude_unavailable = True
        result.warnings.append(
            "claude CLI not found on PATH; cannot install enabledPlugins "
            "via 'claude plugin install --scope project'."
        )

    resolved_config_dir = config_dir if config_dir is not None else _default_config_dir()
    registry_data = _load_registry(resolved_config_dir)

    remaining: List[str] = []
    for key in keys:
        if _already_registered(registry_data, key, worktree_path):
            result.skipped.append(key)
        else:
            remaining.append(key)

    if not remaining:
        return result

    resolved_worktree_id = worktree_id or Path(worktree_path).name
    effective_timeout = _resolve_install_timeout(timeout)
    log_dir: Optional[Path] = None

    for key in remaining:
        source = _find_clone_source(registry_data, key)
        if source is not None:
            # Whether newly cloned or already present (idempotent no-op),
            # the key is now validly registered without touching the CLI.
            _clone_entry_to_worktree(resolved_config_dir, key, source, worktree_path)
            result.installed.append(key)
            continue

        if exe is None:
            result.failed.append(key)
            result.warnings.append(
                f"{key!r}: no structurally-valid clone source and claude CLI unavailable"
            )
            continue

        if log_dir is None:
            log_dir = log_dir_for(resolved_worktree_id)
            log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"plugin-install-{_slug(key)}.log"
        rc, timed_out = _run_install(
            exe,
            key,
            worktree_path,
            timeout=effective_timeout,
            log_path=log_path,
            runner=runner,
        )
        if rc == 0:
            result.installed.append(key)
            continue

        # Second-chance clone: the CLI may have partially populated the
        # registry with a now-valid source even though the overall install
        # failed (e.g. the Windows EPERM failure this ticket targets).
        recovery_source = _find_clone_source(_load_registry(resolved_config_dir), key)
        if recovery_source is not None:
            # Whether newly cloned or already present, the key is now
            # validly registered — same "found a source" semantics as the
            # primary clone-first attempt above.
            _clone_entry_to_worktree(resolved_config_dir, key, recovery_source, worktree_path)
            result.installed.append(key)
            result.warnings.append(
                f"{key!r}: CLI install failed (code {rc}); recovered via registry clone"
            )
            continue

        result.failed.append(key)
        if timed_out:
            result.warnings.append(
                f"plugin install for {key!r} timed out after {effective_timeout}s"
            )
        else:
            result.warnings.append(
                f"plugin install for {key!r} exited with code {rc}"
            )

    return result


__all__ = (
    "PluginInstallResult",
    "install_enabled_plugins",
)
