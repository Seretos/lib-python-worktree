"""Checkout classification: distinguishing a primary clone from a linked
worktree, from any path inside a repository (ticket #84).

Two concerns live here, both scoped to *reading* the repo's checkout
structure — never mutating state:

1. ``classify_checkout()`` / ``CheckoutInfo`` / ``primary_id_for()`` — given
   any path inside a repo, resolve which physical checkout it belongs to
   (``"primary"`` vs ``"worktree"``) and the repo's canonical root. This is
   the primitive ``WorktreeManager._validate_repo()`` is built on (fixing the
   pre-existing bug where it resolved a linked worktree's own path instead of
   the main clone).
2. ``EnvironmentEntry`` / ``RepoListing`` / ``list_repo()`` — the repo-scoped
   listing that joins git's live ``worktree list --porcelain`` view against
   already-tracked ``WorktreeRecord``s, synthesising an untracked entry in
   memory (never writing to the store) for the primary checkout and for any
   linked worktree that is on disk but not yet adopted. See
   ``EnvironmentEntry``'s docstring for the untracked-linked-worktree id
   convention (``record.id == ""``).

This module imports only ``_exceptions``, ``_git_utils`` and ``state`` —
never ``manager`` — so it cannot introduce an import cycle; ``manager.py``
imports from here, not the reverse.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

from ._exceptions import InvalidRepoError
from ._git_utils import _run_git
from .state import WorktreeRecord

# Duplicated (deliberately) from manager._SLUG_RE — see manager.py's own
# comment on _slug for why: keeping this module free of a manager import
# avoids a cycle, and the two lines are cheap enough to mirror rather than
# extract into a third shared module.
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str, *, max_len: int = 40) -> str:
    """Lower-case ASCII slug suitable for filesystem use and IDs."""
    s = _SLUG_RE.sub("-", value.lower()).strip("-")
    if not s:
        s = "x"
    return s[:max_len]


# ---------------------------------------------------------------------------
# CheckoutInfo / classify_checkout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckoutInfo:
    """The result of classifying a filesystem path against its git repo.

    ``backing`` is ``"primary"`` when *checkout_path* resolves to the main
    clone, ``"worktree"`` when it resolves to a linked worktree.
    ``repo_root`` is always the main clone's root, regardless of which
    checkout *checkout_path* itself belongs to. ``checkout_path`` is the
    resolved input path as given (may be a subdirectory of either checkout).
    """

    backing: str
    repo_root: Path
    checkout_path: Path


def classify_checkout(path: "Path | str") -> CheckoutInfo:
    """Classify *path* (anywhere inside a repo) as primary or linked worktree.

    Implemented with a single ``git rev-parse --path-format=absolute
    --git-dir --git-common-dir`` call: when the two resolved paths are equal,
    *path* belongs to the primary/main checkout; otherwise it belongs to a
    linked worktree. ``--git-common-dir`` always resolves to
    ``<main-clone>/.git`` in both cases, so ``repo_root`` (its parent) is
    correct for both a primary and a nested worktree subdirectory query
    alike — including when *path* is itself a subdirectory rather than the
    checkout's top-level directory.

    Raises ``InvalidRepoError`` if *path* does not exist, is not a directory
    (e.g. a file inside a repo), or is not inside a git repository. A
    ``GitTimeoutError`` from the underlying ``_run_git`` call propagates
    unchanged.
    """
    raw = str(path)
    checkout_path = Path(path).expanduser()
    try:
        checkout_path = checkout_path.resolve()
    except OSError as exc:  # pragma: no cover - defensive, platform-specific
        raise InvalidRepoError(raw, f"cannot resolve path: {exc}") from exc

    if not checkout_path.exists():
        raise InvalidRepoError(raw, f"repo_root does not exist: {checkout_path}")

    if not checkout_path.is_dir():
        raise InvalidRepoError(raw, f"repo_root is not a directory: {checkout_path}")

    proc = _run_git(
        ["rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir"],
        cwd=checkout_path,
    )
    if proc.returncode != 0:
        raise InvalidRepoError(raw, f"not a git repository: {checkout_path}")

    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise InvalidRepoError(
            raw, f"unexpected 'git rev-parse' output: {proc.stdout!r}"
        )
    git_dir = Path(lines[0]).resolve()
    git_common_dir = Path(lines[1]).resolve()

    backing = "primary" if git_dir == git_common_dir else "worktree"
    repo_root = git_common_dir.parent.resolve()

    return CheckoutInfo(backing=backing, repo_root=repo_root, checkout_path=checkout_path)


def primary_id_for(repo_root: "Path | str") -> str:
    """Deterministic id for the primary/main-checkout environment of a repo.

    Stable across repeated calls for the same root (including a
    non-normalised/relative spelling — the root is resolved first) and
    distinct across different roots (a location-derived hash disambiguates
    same-named repos checked out in different directories).
    """
    resolved = Path(repo_root).expanduser().resolve()
    digest = hashlib.sha256(resolved.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{_slug(resolved.name)}-root-{digest}"


# ---------------------------------------------------------------------------
# EnvironmentEntry / RepoListing / list_repo
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentEntry:
    """One checkout (primary or linked worktree) in a repo-scoped listing.

    ``tracked=False`` marks a *synthesised* entry -- a checkout that exists on
    disk (per ``git worktree list --porcelain``) but has no persisted
    ``WorktreeRecord`` yet. This happens for the primary checkout before its
    first ``WorktreeManager.start()`` call, and for any linked worktree that
    was created outside this tool (e.g. by hand, or by another process) and
    has not been ``adopt()``-ed.

    For a synthesised **primary** entry, ``record.id`` is the deterministic
    ``primary_id_for(repo_root)`` -- stable across calls, so it round-trips
    correctly once the primary is later materialised by ``start()``.

    For a synthesised **linked-worktree** entry there is no such deterministic
    id (unlike the primary, a linked worktree has no canonical identity until
    ``adopt()``/``create()`` mints one), so ``record.id`` is set to the empty
    string ``""``. An empty id can never collide with a real tracked id --
    every id minted by ``create()``/``adopt()``/``primary_id_for()`` is
    non-empty by construction -- but callers must not treat the id as the
    discriminator for "is this tracked": always check ``tracked`` instead.
    Other fields (``path``, ``branch``, ``repo_root``, ``backing``) are
    populated from git's porcelain output where available.
    """

    record: WorktreeRecord
    is_current: bool = False   # this entry's checkout contains the queried path
    tracked: bool = True       # False = synthesised, not (yet) in state.yaml


@dataclass
class RepoListing:
    """All environments (primary + linked worktrees) for a single repo."""

    repo_root: str  # POSIX string, matches WorktreeRecord.repo_root
    entries: List[EnvironmentEntry] = field(default_factory=list)


def _parse_worktree_porcelain(stdout: str) -> List[Dict[str, Optional[str]]]:
    """Parse ``git worktree list --porcelain`` output into per-block dicts.

    Mirrors the parsing already established in ``yaml_store.adopt()``.
    """
    blocks: List[Dict[str, Optional[str]]] = []
    current: Dict[str, Optional[str]] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.rstrip()
        if not line:
            if current:
                blocks.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = line[len("worktree "):].strip()
        elif line.startswith("branch "):
            ref = line[len("branch "):].strip()
            prefix = "refs/heads/"
            current["branch"] = ref[len(prefix):] if ref.startswith(prefix) else ref
        elif line == "detached":
            current["detached"] = "true"
        elif line.startswith("prunable"):
            current["prunable"] = "true"
    if current:
        blocks.append(current)
    return blocks


def _path_contains(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def list_repo(path: "Path | str", records: List[WorktreeRecord]) -> RepoListing:
    """Return the repo-scoped listing (primary + linked worktrees) for *path*.

    Pure function: classifies *path*, enumerates ``git worktree list
    --porcelain`` from the main clone, and joins it against the *records*
    already passed in (filtered by ``repo_root``) — it never touches a state
    store. Both the primary environment and any untracked linked worktree are
    synthesised (``tracked=False``) when no persisted record exists for them
    yet -- the primary materialises via the first ``WorktreeManager.start()``
    call, a linked worktree via ``adopt()``. See ``EnvironmentEntry`` for the
    id convention used for a synthesised linked-worktree entry. Entries are
    ordered primary-first, then linked worktrees sorted by path.

    Raises ``InvalidRepoError`` if the ``git worktree list --porcelain`` call
    itself fails (e.g. a damaged repo or a transient git error) -- consistent
    with ``classify_checkout()``'s and ``_validate_repo()``'s handling of a
    failing ``_run_git`` call; a git failure is a real error, not "zero
    environments".

    **Branch-freshness contract, per entry kind (ticket #85):**

    - **Primary:** ``record.branch`` is always ``None`` (#84) and must be
      resolved live via ``WorktreeManager._effective_branch(record)``.
    - **Tracked linked worktree:** ``record.branch`` is refreshed *in memory*
      from git's live ``worktree list --porcelain`` view on every call --
      the stored value goes stale the moment someone runs a manual
      ``git checkout`` inside the worktree. This is a read view only:
      nothing is written back to the state store, so ``record.branch`` here
      may legitimately differ from what is persisted to ``state.yaml``, and
      callers must not treat it as evidence of a store update.
    - **Detached tracked linked worktree:** porcelain emits no ``branch``
      line for a detached HEAD, so the entry keeps its stored / last-known
      branch instead of being refreshed to ``None``.
    - **Untracked linked worktree:** ``record.branch`` comes from porcelain
      by construction, same as always (unchanged by this ticket).
    """
    info = classify_checkout(path)
    repo_root_str = info.repo_root.as_posix()

    proc = _run_git(["worktree", "list", "--porcelain"], cwd=info.repo_root)
    if proc.returncode != 0:
        raise InvalidRepoError(
            str(path), f"'git worktree list --porcelain' failed: {proc.stderr!r}"
        )
    blocks = _parse_worktree_porcelain(proc.stdout or "")

    records_by_path: Dict[Path, WorktreeRecord] = {
        Path(r.path).resolve(): r
        for r in records
        if r.repo_root == repo_root_str
    }

    primary_path = Path(blocks[0]["path"]).resolve() if blocks else info.repo_root

    entries: List[EnvironmentEntry] = []
    for block in blocks:
        wt_path_raw = block.get("path")
        if not wt_path_raw:
            continue
        if block.get("prunable"):
            # Vanished on-disk; not a real, listable environment (consistent
            # with adopt()'s skipped_prunable handling).
            continue

        wt_path = Path(wt_path_raw).resolve()
        is_primary_block = wt_path == primary_path
        record = records_by_path.get(wt_path)
        tracked = record is not None

        if record is not None and not is_primary_block:
            # Refresh a tracked linked worktree's branch from git's live view: the
            # stored value goes stale after a manual `git checkout` inside the
            # worktree. dataclasses.replace() (never in-place mutation) because
            # InMemoryStateStore.list() hands out its live record objects -- this
            # function is read-only and must never write back to the store.
            live_branch = block.get("branch")
            if live_branch and live_branch != record.branch:
                record = replace(record, branch=live_branch)

        if record is None:
            if is_primary_block:
                record = WorktreeRecord(
                    id=primary_id_for(info.repo_root),
                    repo_root=repo_root_str,
                    branch=None,
                    path=wt_path.as_posix(),
                    status="created",
                    backing="primary",
                )
            else:
                # On disk but not (yet) adopted: still a real, listable
                # environment -- synthesise it as untracked rather than
                # dropping it, so a caller can discover it and call adopt().
                # No deterministic id exists for a linked worktree (unlike
                # the primary), so id="" -- see EnvironmentEntry's docstring
                # for why that can never collide with a real tracked id.
                record = WorktreeRecord(
                    id="",
                    repo_root=repo_root_str,
                    branch=block.get("branch"),
                    path=wt_path.as_posix(),
                    status="created",
                    backing="worktree",
                )

        entries.append(
            EnvironmentEntry(
                record=record,
                is_current=_path_contains(wt_path, info.checkout_path),
                tracked=tracked,
            )
        )

    entries.sort(key=lambda e: (e.record.backing != "primary", e.record.path))

    return RepoListing(repo_root=repo_root_str, entries=entries)


__all__ = [
    "CheckoutInfo",
    "EnvironmentEntry",
    "RepoListing",
    "classify_checkout",
    "list_repo",
    "primary_id_for",
]
