"""Shared pytest fixtures for the lib-python-worktree test suite.

Provides:
- skip_if_no_git: skip fixture for tests that require a real git binary.
- git_repo:        a freshly initialised git repository (function scope).
- manager_factory: factory returning WorktreeManager instances with automatic
                   teardown of any worktrees created during the test.
- manager:         convenience wrapper around manager_factory for single-manager tests.
- linked_worktree: a real `git worktree add`-ed checkout off git_repo.
- yaml_manager:    factory building a WorktreeManager on a real YamlStateStore
                   (so the real PortAllocator is exercised, not _NoOpPortAllocator).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterator, List
from unittest.mock import patch

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


# ---------------------------------------------------------------------------
# linked_worktree: a real `git worktree add`-ed checkout off git_repo
# ---------------------------------------------------------------------------

@pytest.fixture
def linked_worktree(git_repo: Path, tmp_path: Path) -> Iterator[Path]:
    """Yield a Path to a real linked worktree checked out from ``git_repo``.

    Checked out on ``feature/alpha`` (already branched off ``main`` by the
    ``git_repo`` fixture), so this can stand in wherever a test needs "some
    other, already-existing checkout of the same repo" without going through
    ``WorktreeManager.create()``.
    """
    wt_path = tmp_path / "linked-wt"
    _run_git("worktree", "add", str(wt_path), "feature/alpha", cwd=git_repo)
    yield wt_path
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(wt_path)],
        cwd=git_repo,
        capture_output=True,
    )
    shutil.rmtree(wt_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# yaml_manager: WorktreeManager on a real YamlStateStore (real PortAllocator)
# ---------------------------------------------------------------------------

@pytest.fixture
def yaml_manager(
    tmp_path: Path, skip_if_no_git  # noqa: ARG001
) -> Iterator[Callable[..., WorktreeManager]]:
    """Yield a factory building a ``WorktreeManager`` on a real
    ``YamlStateStore``, so ``self._allocator`` is the real ``PortAllocator``
    (not ``_NoOpPortAllocator``, which the plain ``manager``/``manager_factory``
    fixtures get via ``InMemoryStateStore``).  Needed to exercise real port
    allocation/reservation behaviour (ticket #84, R9) and primary-checkout
    persistence (R7) end-to-end.
    """
    from lib_python_worktree.core.yaml_store import YamlStateStore

    managers: List[WorktreeManager] = []
    store_counter = [0]

    def _make(store_root: Path | None = None) -> WorktreeManager:
        store_counter[0] += 1
        root = store_root or (tmp_path / f"store-{store_counter[0]}")
        state_dir = tmp_path / f"state-{store_counter[0]}"
        mgr = WorktreeManager(
            config=ManagerConfig(store_root=root),
            state=YamlStateStore(state_dir=state_dir),
            reconcile_on_init=False,
        )
        managers.append(mgr)
        return mgr

    yield _make

    for mgr in managers:
        for record in mgr.state.list():
            wt_path = Path(record.path)
            repo_root = Path(record.repo_root)
            if wt_path.resolve() == repo_root.resolve():
                # Never rmtree a primary checkout's directory during test
                # cleanup -- it IS the repo (often the shared git_repo
                # fixture's own tmp_path directory).
                continue
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(wt_path)],
                    cwd=repo_root,
                    capture_output=True,
                )
            except Exception:  # noqa: BLE001
                pass
            shutil.rmtree(wt_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# generous_early_exit_wait: widen process_lifecycle._EARLY_EXIT_WAIT_SEC for
# real-spawn early-exit tests (ticket #137)
# ---------------------------------------------------------------------------

@pytest.fixture
def generous_early_exit_wait() -> Iterator[None]:
    """Patch ``process_lifecycle._EARLY_EXIT_WAIT_SEC`` to a generous value
    for the duration of a test.

    The production default (0.25s, ticket #81) bounds the real early-exit
    poll -- ``proc.wait(timeout=_EARLY_EXIT_WAIT_SEC)`` in
    ``core/process_lifecycle.py`` -- and is deliberately left untouched here;
    this fixture never patches it in production code, only for a test's own
    call. A handful of tests spawn a REAL child process (a bare
    ``sys.executable -c ...``, or -- for the ``start:`` seam -- a
    ``powershell.exe``-wrapped step command) and assert on the early-exit
    outcome. On a loaded CI runner, process startup can structurally exceed
    0.25s (this is especially true of PowerShell startup on
    ``windows-latest``), so ``proc.wait`` times out, the early-exit branch is
    never taken, and the test flakes with ``status="running"`` instead of
    reaching the exited/returncode assertion it actually means to check.
    Widening the window here only affects *this test's* wait: ``proc.wait()``
    still returns the instant the child actually exits, so this costs no
    real wall-clock time in the (overwhelmingly common) fast-exit case -- it
    is a bounded deterministic wait, not a sleep. The constant is read as a
    module global at call time, so patching it here also takes effect
    transparently through ``WorktreeManager.start()``'s delegation into
    ``_lifecycle_start``.
    """
    with patch(
        "lib_python_worktree.core.process_lifecycle._EARLY_EXIT_WAIT_SEC", 30.0
    ):
        yield
