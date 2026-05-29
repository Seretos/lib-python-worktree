"""Unit tests for the internal helpers in core/manager.py.

These are pure-logic or lightly-mocked tests -- no real git subprocess
required (except where noted).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lib_python_worktree.core.manager import (
    WorktreeError,
    _is_path_prunable,
    _parse_already_checked_out,
    _resolve_git_timeout,
    _slug,
)


# ---- _slug ---------------------------------------------------------------


def test_slug_basic():
    assert _slug("feature/alpha") == "feature-alpha"


def test_slug_empty_input_returns_x():
    assert _slug("") == "x"


def test_slug_non_ascii():
    # Non-ASCII chars are treated as separators and stripped.
    result = _slug("café-branch")
    assert result == result.lower()
    assert "caf" in result


def test_slug_max_len_truncation():
    long_value = "a" * 50
    result = _slug(long_value, max_len=10)
    assert len(result) <= 10


def test_slug_default_max_len():
    long_value = "x" * 100
    result = _slug(long_value)
    assert len(result) == 40


def test_slug_only_separators_returns_x():
    # A string of only non-alnum chars should yield "x".
    assert _slug("---!!!---") == "x"


# ---- _resolve_git_timeout -----------------------------------------------


def test_resolve_git_timeout_explicit_value():
    assert _resolve_git_timeout(5.0) == 5.0


def test_resolve_git_timeout_explicit_none_with_no_env(monkeypatch):
    monkeypatch.delenv("WORKTREE_GIT_TIMEOUT_SEC", raising=False)
    # explicit=None + no env variable -> built-in default (30.0)
    result = _resolve_git_timeout(None)
    assert result == 30.0


def test_resolve_git_timeout_env_overrides_default(monkeypatch):
    monkeypatch.setenv("WORKTREE_GIT_TIMEOUT_SEC", "15.5")
    result = _resolve_git_timeout(None)
    assert result == 15.5


def test_resolve_git_timeout_env_empty_string_disables(monkeypatch):
    monkeypatch.setenv("WORKTREE_GIT_TIMEOUT_SEC", "")
    result = _resolve_git_timeout(None)
    assert result is None


def test_resolve_git_timeout_env_non_numeric_falls_back(monkeypatch):
    monkeypatch.setenv("WORKTREE_GIT_TIMEOUT_SEC", "banana")
    result = _resolve_git_timeout(None)
    assert result == 30.0


def test_resolve_git_timeout_env_zero_disables(monkeypatch):
    monkeypatch.setenv("WORKTREE_GIT_TIMEOUT_SEC", "0")
    result = _resolve_git_timeout(None)
    assert result is None


def test_resolve_git_timeout_explicit_beats_env(monkeypatch):
    monkeypatch.setenv("WORKTREE_GIT_TIMEOUT_SEC", "999")
    result = _resolve_git_timeout(7.0)
    assert result == 7.0


# ---- _parse_already_checked_out -----------------------------------------

MODERN_STDERR = (
    "fatal: 'feature/x' is already used by worktree at '/some/path'"
)
OLD_STDERR = (
    "fatal: 'feature/x' is already checked out at '/other/path'"
)


def test_parse_already_checked_out_modern_wording():
    result = _parse_already_checked_out(MODERN_STDERR)
    assert result is not None
    branch, path = result
    assert branch == "feature/x"
    assert path == "/some/path"


def test_parse_already_checked_out_old_wording():
    result = _parse_already_checked_out(OLD_STDERR)
    assert result is not None
    branch, path = result
    assert branch == "feature/x"
    assert path == "/other/path"


def test_parse_already_checked_out_no_match_returns_none():
    assert _parse_already_checked_out("some random error") is None


def test_parse_already_checked_out_empty_string():
    assert _parse_already_checked_out("") is None


def test_parse_already_checked_out_none_input():
    # The function guards against None via ``stderr or ""``.
    assert _parse_already_checked_out(None) is None  # type: ignore[arg-type]


# ---- _is_path_prunable --------------------------------------------------


def test_is_path_prunable_returns_none_when_run_git_raises(tmp_path: Path):
    """If _run_git itself raises (e.g. git not available), the function must
    return None instead of propagating the exception.
    """
    from lib_python_worktree.core import manager as mgr_mod

    with patch.object(mgr_mod, "_run_git", side_effect=WorktreeError("boom")):
        result = _is_path_prunable(tmp_path, "/any/path")
    assert result is None


def test_is_path_prunable_returns_none_for_nonzero_returncode(tmp_path: Path):
    """If git exits non-zero, the function returns None."""
    import subprocess
    from lib_python_worktree.core import manager as mgr_mod

    fake_proc = subprocess.CompletedProcess(
        args=["git", "worktree", "list", "--porcelain"],
        returncode=1,
        stdout="",
        stderr="error",
    )
    with patch.object(mgr_mod, "_run_git", return_value=fake_proc):
        result = _is_path_prunable(tmp_path, "/any/path")
    assert result is None


def test_is_path_prunable_found_not_prunable(tmp_path: Path):
    """Path present in list output but no 'prunable' line -> False."""
    import subprocess
    from lib_python_worktree.core import manager as mgr_mod

    porcelain = "worktree /some/path\nHEAD abc123\nbranch refs/heads/main\n\n"
    fake_proc = subprocess.CompletedProcess(
        args=["git", "worktree", "list", "--porcelain"],
        returncode=0,
        stdout=porcelain,
        stderr="",
    )
    with patch.object(mgr_mod, "_run_git", return_value=fake_proc):
        result = _is_path_prunable(tmp_path, "/some/path")
    assert result is False


def test_is_path_prunable_found_prunable(tmp_path: Path):
    """Path present and has a 'prunable' line -> True."""
    import subprocess
    from lib_python_worktree.core import manager as mgr_mod

    porcelain = (
        "worktree /stale/path\n"
        "HEAD abc123\n"
        "branch refs/heads/gone\n"
        "prunable gitdir file points to non-existent location\n"
        "\n"
    )
    fake_proc = subprocess.CompletedProcess(
        args=["git", "worktree", "list", "--porcelain"],
        returncode=0,
        stdout=porcelain,
        stderr="",
    )
    with patch.object(mgr_mod, "_run_git", return_value=fake_proc):
        result = _is_path_prunable(tmp_path, "/stale/path")
    assert result is True


def test_is_path_prunable_path_not_in_list_returns_none(tmp_path: Path):
    """Path absent from the list -> None (distinct from False)."""
    import subprocess
    from lib_python_worktree.core import manager as mgr_mod

    porcelain = "worktree /other/path\nHEAD abc123\n\n"
    fake_proc = subprocess.CompletedProcess(
        args=["git", "worktree", "list", "--porcelain"],
        returncode=0,
        stdout=porcelain,
        stderr="",
    )
    with patch.object(mgr_mod, "_run_git", return_value=fake_proc):
        result = _is_path_prunable(tmp_path, "/missing/path")
    assert result is None
