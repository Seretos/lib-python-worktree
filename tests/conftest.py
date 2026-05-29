"""Shared fixtures for the lib-python-worktree test suite.

Provides:
- ``git_repo``: a real git repository in a temp dir with an initial commit.
- ``manager_factory``: builds a ``WorktreeManager`` with a guaranteed cleanup
  of any surviving worktrees in the fixture teardown, so Windows (which holds
  file locks on checked-out dirs) is also properly cleaned.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterator

import pytest

from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
from lib_python_worktree.core.state import InMemoryStateStore


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    """Run ``git <args>`` in ``cwd``, raise on failure, capture output."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Iterator[Path]:
    """A fresh git repository with one initial commit on ``main``.

    Uses ``tmp_path`` so pytest handles filesystem cleanup.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    # Try ``-b main``; fall back for older git that doesn't support -b on init.
    try:
        _git(repo, "init", "-b", "main")
    except subprocess.CalledProcessError:
        _git(repo, "init")
        _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    yield repo


@pytest.fixture
def manager_factory(tmp_path: Path) -> Iterator[Callable[[], WorktreeManager]]:
    """Returns a callable that creates a ``WorktreeManager`` with a private
    store root. On teardown, forcibly removes any surviving worktrees so
    pytest's ``tmp_path`` cleanup does not fail due to files still being
    checked out.
    """
    managers: list[WorktreeManager] = []

    def _make() -> WorktreeManager:
        store_root = tmp_path / f"store-{len(managers)}"
        mgr = WorktreeManager(
            config=ManagerConfig(store_root=store_root),
            state=InMemoryStateStore(),
        )
        managers.append(mgr)
        return mgr

    yield _make

    # Teardown: clean up any tracked worktrees that weren't removed by tests.
    for mgr in managers:
        for record in list(mgr.list()):
            try:
                wt_path = Path(record.path)
                repo_path = Path(record.repo_root)
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(wt_path)],
                    cwd=repo_path,
                    capture_output=True,
                )
                # Fallback: if git refused to release file handles (common on
                # Windows), forcibly remove the directory so pytest's tmp_path
                # cleanup never fails due to a leftover checked-out tree.
                shutil.rmtree(wt_path, ignore_errors=True)
            except Exception:  # pragma: no cover
                pass
