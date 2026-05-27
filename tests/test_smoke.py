"""Smoke test: the empty frame builds and imports.

Proves `pip install -e .` produces an importable package with the declared
version. Real engine tests arrive with the logic migration.
"""
from __future__ import annotations

import lib_python_worktree


def test_package_imports_with_version() -> None:
    assert lib_python_worktree.__version__ == "0.1.0"
