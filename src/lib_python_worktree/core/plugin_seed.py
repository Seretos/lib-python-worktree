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
  no-op cases (missing file, malformed JSON, non-v2-object top-level).
- **Idempotent**: an entry is skipped if one with the same ``projectPath`` and
  ``installPath`` already exists.
- **Atomic**: the registry is written via a temp-file + ``os.replace`` so that
  a crash mid-write leaves the original intact.
- **Removable**: the entire workaround is isolated here.  When Claude fixes the
  upstream bug, delete this file and remove the call in ``manager.py``.

The real Claude plugin registry uses **Schema v2**::

    {
        "version": 2,
        "plugins": {
            "<name>@<marketplace>": [<entry>, ...],
            ...
        }
    }

Any top-level shape other than a dict with ``version == 2`` and a ``plugins``
dict is treated as an unsupported format and the function returns silently.
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
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return

    # Only handle Schema v2: {"version": 2, "plugins": {<name>: [entries...]}}
    if not (
        isinstance(data, dict)
        and data.get("version") == 2
        and isinstance(data.get("plugins"), dict)
    ):
        return

    plugins: dict[str, list] = data["plugins"]

    # Native-OS form of the destination path (backslashes on Windows).
    dest_path = str(Path(worktree_path))

    # Normalised form of the source repo path for case/separator-insensitive
    # comparison (os.path.normcase lowercases on Windows, no-op on Linux/macOS).
    norm_repo = os.path.normcase(str(Path(repo_path)))

    # Build a set of (projectPath, installPath) pairs already present across
    # all per-name lists so we can skip duplicates efficiently.
    existing: set[tuple[str, str]] = set()
    for entry_list in plugins.values():
        if isinstance(entry_list, list):
            for e in entry_list:
                if isinstance(e, dict):
                    existing.add(
                        (e.get("projectPath", ""), e.get("installPath", ""))
                    )

    # For each plugin name, clone matching entries into the same per-name list.
    any_added = False
    for plugin_name, entry_list in plugins.items():
        if not isinstance(entry_list, list):
            continue
        clones: list[dict] = []
        for entry in entry_list:
            if not isinstance(entry, dict):
                continue
            if entry.get("scope") != "project":
                continue
            project_path = entry.get("projectPath")
            if not isinstance(project_path, str):
                # null, missing, or non-string projectPath — skip gracefully.
                continue
            norm_entry = os.path.normcase(str(Path(project_path)))
            if norm_entry != norm_repo:
                continue
            install_path = entry.get("installPath", "")
            if (dest_path, install_path) in existing:
                # Already seeded — skip to stay idempotent.
                continue
            cloned = dict(entry)
            cloned["projectPath"] = dest_path
            clones.append(cloned)
            # Add to existing so a second matching entry in the same list
            # doesn't produce two identical clones.
            existing.add((dest_path, install_path))
        if clones:
            entry_list.extend(clones)
            any_added = True

    if not any_added:
        return

    # Atomic write: dump to a sibling temp file, then replace.
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
