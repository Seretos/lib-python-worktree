"""Workaround for anthropics/claude-code#61866: seed the Claude plugin registry.

Claude's interactive session loads project-scoped Marketplace plugins only when
the session ``cwd`` exactly matches a ``projectPath`` entry in
``~/.claude/plugins/installed_plugins.json``.  A freshly-created git worktree
has no such entry, so plugins appear absent until the user manually issues a
``/reload-plugins`` command.

This module copies every ``scope:"project"`` registry entry whose
``projectPath`` matches the parent repo path, replacing ``projectPath`` with
the new worktree path.  The operation is:

- **Best-effort**: the bare ``except`` swallowing lives at the call site in
  ``manager.py``, *not* here.  Explicit early-return guards handle the common
  no-op cases (missing file, malformed JSON, non-list top-level).
- **Idempotent**: an entry is skipped if one with the same ``projectPath`` and
  ``installPath`` already exists.
- **Atomic**: the registry is written via a temp-file + ``os.replace`` so that
  a crash mid-write leaves the original intact.
- **Removable**: the entire workaround is isolated here.  When Claude fixes the
  upstream bug, delete this file and remove the call in ``manager.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional


def seed_plugin_registry(
    repo_path: str,
    worktree_path: str,
    *,
    config_dir: Optional[Path] = None,
) -> None:
    """Copy project-scoped plugin entries for *repo_path* to *worktree_path*.

    Parameters
    ----------
    repo_path:
        POSIX forward-slash path of the parent repository root as stored in
        ``WorktreeRecord.repo_root`` (``repo_path.as_posix()``).
    worktree_path:
        POSIX forward-slash path of the new worktree as stored in
        ``WorktreeRecord.path`` (``target_path.as_posix()``).
    config_dir:
        Override the Claude config directory (defaults to the value of
        ``$CLAUDE_CONFIG_DIR`` or ``~/.claude``).  Primarily a test seam.
    """
    if config_dir is None:
        config_dir = Path(
            os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")
        ).expanduser()

    registry_path = config_dir / "plugins" / "installed_plugins.json"

    if not registry_path.exists():
        return

    try:
        raw = registry_path.read_text(encoding="utf-8")
        entries = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return

    if not isinstance(entries, list):
        return

    # Native-OS form of the destination path (backslashes on Windows).
    dest_path = str(Path(worktree_path))

    # Build a set of (projectPath, installPath) pairs already present so we
    # can skip duplicates efficiently.
    existing: set[tuple[str, str]] = {
        (e.get("projectPath", ""), e.get("installPath", ""))
        for e in entries
        if isinstance(e, dict)
    }

    new_entries: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("scope") != "project":
            continue
        if entry.get("projectPath") != repo_path:
            continue
        install_path = entry.get("installPath", "")
        if (dest_path, install_path) in existing:
            # Already seeded — skip to stay idempotent.
            continue
        cloned = dict(entry)
        cloned["projectPath"] = dest_path
        new_entries.append(cloned)

    if not new_entries:
        return

    updated = entries + new_entries

    # Atomic write: dump to a sibling temp file, then replace.
    tmp_path: Optional[str] = None
    try:
        registry_dir = registry_path.parent
        fd, tmp_path = tempfile.mkstemp(
            dir=str(registry_dir), suffix=".tmp", prefix="installed_plugins_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(updated, fh, indent=2)
                fh.flush()
        except Exception:
            # fdopen took ownership of fd; if write fails, fd is already
            # closed.  Remove the partial temp file before re-raising.
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        os.replace(tmp_path, str(registry_path))
        tmp_path = None  # replaced successfully; nothing to clean up
    except Exception:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


__all__ = ("seed_plugin_registry",)
