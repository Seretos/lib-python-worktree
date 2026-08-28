"""#154 R14 -- AC1 wall-clock measurement, in-repo half (marker-gated).

This is the "opt-in, not in default CI" duration test the ticket's Q2(c)
answer settles for AC1 -- deterministic structural tests (R1, R5, R6, ...)
carry the CI gate; this file exists purely to produce AC1's numbers
(median < 2s, worst case < 5s over 20 create/remove cycles) on a real,
loaded developer machine, and is excluded from the default `python -m
pytest` selection via the `loaded_host` marker (see pyproject.toml's
`addopts = "... -m \"not loaded_host\""`).

Run explicitly with: `python -m pytest -m loaded_host tests/test_remove_duration.py -v`
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import pytest

from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
from lib_python_worktree.core.state import InMemoryStateStore


pytestmark = [
    pytest.mark.loaded_host,
    pytest.mark.skipif(
        sys.platform != "win32",
        reason="AC1 is a Windows-hang regression; only meaningful on win32",
    ),
]


def _make_manager(store_root: Path) -> WorktreeManager:
    return WorktreeManager(
        config=ManagerConfig(store_root=store_root),
        state=InMemoryStateStore(),
        reconcile_on_init=False,
    )


def test_remove_median_and_worst_case(tmp_path, skip_if_no_git):
    """20 real create/remove cycles of a clean, unheld worktree in one
    process. RED on a loaded Windows host against today's code (2-3 full
    systemwide scans per removal via Gate A + the orphan-scan phase);
    GREEN once removal never scans on the happy path (R1/AC2).

    Population/instrumentation note (plan.md R14): all 20 cycles are
    expected to be first-attempt rename successes -- this test also spies
    `time.sleep` and counts `os.rename` attempts per cycle. If any cycle
    entered the transient-retry loop, that fact is reported explicitly
    rather than silently absorbed into the timings or read as a
    regression: it means a transient holder touched this run and it
    should be re-run on a quiet host.
    """
    import subprocess

    def _run_git(*args: str, cwd: Path) -> None:
        subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )

    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git("init", "-q", "-b", "main", cwd=repo)
    _run_git("config", "user.email", "test@example.com", cwd=repo)
    _run_git("config", "user.name", "Test", cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run_git("add", "-A", cwd=repo)
    _run_git("commit", "-q", "-m", "init", cwd=repo)

    manager = _make_manager(tmp_path / "store")

    durations = []
    retry_loop_entries = []

    from unittest.mock import patch

    for i in range(20):
        branch = f"feature/loaded-host-{i}"
        record = manager.create(str(repo), branch, base="main", fetch=False)

        rename_attempts = {"n": 0}
        sleeps: list = []
        real_rename = __import__("os").rename

        def _counting_rename(src, dst, *a, **kw):
            rename_attempts["n"] += 1
            return real_rename(src, dst, *a, **kw)

        with (
            patch(
                "lib_python_worktree.core.teardown.os.rename",
                side_effect=_counting_rename,
            ),
            patch(
                "lib_python_worktree.core.teardown.time.sleep",
                side_effect=lambda s: sleeps.append(s),
            ),
        ):
            t0 = time.monotonic()
            manager.remove(record.id)
            elapsed = time.monotonic() - t0

        durations.append(elapsed)
        if sleeps or rename_attempts["n"] > 1:
            retry_loop_entries.append((i, rename_attempts["n"], len(sleeps)))

    median = statistics.median(durations)
    worst = max(durations)

    if retry_loop_entries:
        pytest.fail(
            f"a transient holder touched {len(retry_loop_entries)} of 20 "
            f"cycle(s) {retry_loop_entries} -- re-run on a quiet host; "
            f"this is not a code regression. Timings: median={median:.3f}s "
            f"worst={worst:.3f}s"
        )

    assert median < 2.0, f"median remove() duration {median:.3f}s >= 2.0s (AC1)"
    assert worst < 5.0, f"worst-case remove() duration {worst:.3f}s >= 5.0s (AC1)"
