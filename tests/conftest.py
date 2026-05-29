"""Shared pytest fixtures for the lib-python-worktree test suite.

Provides:
- skip_if_no_git: skip fixture for tests that require a real git binary.
- git_repo:        a freshly initialised git repository (function scope).
- manager_factory: factory returning WorktreeManager instances with automatic
                   teardown of any worktrees created during the test.
- manager:         convenience wrapper around manager_factory for single-manager tests.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterator, List

import pytest

from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
from lib_python_worktree.core.state import InMemoryStateStore


# ---------------------------------------------------------------------------
# git availability
# ---------------------------------------------------------------------------

def _git_available() -> bool:
    """Return True if a callable ``git`` binary is on PATH."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def pytest_configure(config) -> None:  # noqa: ANN001
    """Disable the coverage fail-under threshold when git is not available.

    The ``--cov-fail-under=80`` gate in addopts is meaningful only when the
    real-git tests actually run (CI always has git, so the gate is enforced
    there).  On a git-absent machine the requires_git tests are skipped,
    coverage drops below 80 %, and the suite would fail the threshold instead
    of exiting cleanly.  Setting ``cov_fail_under`` to 0 here makes a git-less
    run report skips cleanly without a spurious coverage failure, while leaving
    the gate fully enforced on any runner that has git.
    """
    if not _git_available():
        # pytest-cov stores the threshold on config.option; guard for the
        # attribute so this is a no-op if pytest-cov is not installed.
        if hasattr(config, "option") and hasattr(config.option, "cov_fail_under"):
            config.option.cov_fail_under = 0.0


@pytest.fixture
def skip_if_no_git():
    """Skip the test if git is not available on the current runner."""
    if not _git_available():
        pytest.skip("git not available")


# ---------------------------------------------------------------------------
# git_repo: a fresh temp repo with an initial commit and a feature/alpha branch
# ---------------------------------------------------------------------------

def _run_git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path, skip_if_no_git) -> Iterator[Path]:  # noqa: ARG001
    """Yield a Path to a freshly initialised git repository.

    Layout
    ------
    - branch ``main`` with an initial commit (README.md)
    - branch ``feature/alpha`` branched from ``main``
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git("init", "-q", "-b", "main", cwd=repo)
    _run_git("config", "user.email", "test@example.com", cwd=repo)
    _run_git("config", "user.name", "Test", cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run_git("add", "-A", cwd=repo)
    _run_git("commit", "-q", "-m", "init", cwd=repo)
    _run_git("branch", "feature/alpha", cwd=repo)
    yield repo


# ---------------------------------------------------------------------------
# manager_factory: creates WorktreeManager instances that clean up after themselves
# ---------------------------------------------------------------------------

@pytest.fixture
def manager_factory(
    tmp_path: Path, skip_if_no_git  # noqa: ARG001
) -> Iterator[Callable[..., WorktreeManager]]:
    """Yield a factory callable that returns a ``WorktreeManager``.

    Every manager created via the factory is tracked; on teardown any surviving
    worktrees are removed with ``git worktree remove --force`` (plus a
    ``shutil.rmtree`` fallback) so no stale directories are left on disk.
    """
    managers: List[WorktreeManager] = []
    store_counter = [0]

    def _make(store_root: Path | None = None) -> WorktreeManager:
        store_counter[0] += 1
        root = store_root or (tmp_path / f"store-{store_counter[0]}")
        mgr = WorktreeManager(
            config=ManagerConfig(store_root=root),
            state=InMemoryStateStore(),
        )
        managers.append(mgr)
        return mgr

    yield _make

    # Teardown: remove any surviving worktrees from every manager.
    for mgr in managers:
        for record in mgr.state.list():
            wt_path = Path(record.path)
            repo_root = Path(record.repo_root)
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(wt_path)],
                    cwd=repo_root,
                    capture_output=True,
                )
            except Exception:  # noqa: BLE001
                pass
            # Windows safety net: rmtree even if git already cleaned up.
            shutil.rmtree(wt_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# manager: convenience single-manager fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def manager(manager_factory: Callable[..., WorktreeManager]) -> WorktreeManager:
    """A single ``WorktreeManager`` backed by the shared manager_factory."""
    return manager_factory()
