"""Tests for core/checkout.py's classify_checkout() / primary_id_for()
(ticket #84, R3/R5, and the primitive B2 fixes _validate_repo() on).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lib_python_worktree.core._exceptions import GitTimeoutError, InvalidRepoError
from lib_python_worktree.core.checkout import classify_checkout, primary_id_for


# ---------------------------------------------------------------------------
# R3 -- classify_checkout() returns the correct (backing, repo_root) pair
# ---------------------------------------------------------------------------

@pytest.mark.requires_git
@pytest.mark.parametrize(
    "path_fn,expected_backing",
    [
        (lambda git_repo, linked_wt: git_repo, "primary"),
        (lambda git_repo, linked_wt: git_repo / "sub", "primary"),
        (lambda git_repo, linked_wt: linked_wt, "worktree"),
        (lambda git_repo, linked_wt: linked_wt / "sub", "worktree"),
    ],
    ids=["main-root", "main-subdir", "linked-root", "linked-subdir"],
)
def test_classify_primary_and_linked(
    git_repo: Path, linked_worktree: Path, path_fn, expected_backing: str
):
    """R3 driving test: classification is correct for both a main clone and
    a linked worktree, whether queried at the top or from a subdirectory."""
    query_path = path_fn(git_repo, linked_worktree)
    query_path.mkdir(exist_ok=True)

    info = classify_checkout(query_path)

    assert info.backing == expected_backing
    assert info.repo_root.resolve() == git_repo.resolve()
    assert info.checkout_path.resolve() == query_path.resolve()


@pytest.mark.requires_git
def test_classify_non_repo_dir_raises_invalid_repo_error(tmp_path: Path):
    plain_dir = tmp_path / "not_a_repo"
    plain_dir.mkdir()
    with pytest.raises(InvalidRepoError):
        classify_checkout(plain_dir)


@pytest.mark.requires_git
def test_classify_file_path_inside_repo_raises_invalid_repo_error(git_repo: Path):
    """A *file* path inside a real repo must raise InvalidRepoError rather
    than letting an unhandled OSError/FileNotFoundError escape from
    ``_run_git(cwd=<file>)`` (review finding, ticket #84 fix cycle)."""
    file_path = git_repo / "README.md"
    assert file_path.is_file()
    with pytest.raises(InvalidRepoError):
        classify_checkout(file_path)


def test_classify_nonexistent_path_raises_invalid_repo_error(tmp_path: Path):
    missing = tmp_path / "does" / "not" / "exist"
    with pytest.raises(InvalidRepoError):
        classify_checkout(missing)


@pytest.mark.requires_git
def test_classify_propagates_git_timeout_error(git_repo: Path, monkeypatch):
    """A GitTimeoutError from the underlying _run_git call propagates
    unchanged (not swallowed / rewrapped)."""
    import lib_python_worktree.core.checkout as checkout_module

    def _raise_timeout(*args, **kwargs):
        raise GitTimeoutError(["git", "rev-parse"], 30.0)

    monkeypatch.setattr(checkout_module, "_run_git", _raise_timeout)

    with pytest.raises(GitTimeoutError):
        classify_checkout(git_repo)


# ---------------------------------------------------------------------------
# R5 -- deterministic primary id
# ---------------------------------------------------------------------------

def test_primary_id_stable_and_location_unique(tmp_path: Path):
    root_a = tmp_path / "a" / "repo"
    root_a.mkdir(parents=True)
    root_b = tmp_path / "b" / "repo"
    root_b.mkdir(parents=True)

    id_a1 = primary_id_for(root_a)
    id_a2 = primary_id_for(root_a)
    id_b = primary_id_for(root_b)

    assert id_a1 == id_a2
    assert id_a1 != id_b

    # Non-normalised / relative spelling still resolves to the same id.
    id_a_relative = primary_id_for(root_a / ".." / "repo")
    assert id_a_relative == id_a1

    import re
    assert re.fullmatch(r"repo-root-[0-9a-f]{8}", id_a1)
