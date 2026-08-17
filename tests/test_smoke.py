"""Smoke test: the empty frame builds and imports.

Proves `pip install -e .` produces an importable package with the declared
version. Real engine tests arrive with the logic migration.
"""
from __future__ import annotations

import lib_python_worktree


def test_package_imports_with_version() -> None:
    assert lib_python_worktree.__version__ == "0.1.0"


def test_checkout_classification_surface_importable_from_package_root() -> None:
    """ticket #84 (R10 edge case): the checkout-classification surface must
    be importable directly from the package root, not just from
    lib_python_worktree.core.checkout/manager."""
    names = [
        "RepoListing",
        "EnvironmentEntry",
        "CheckoutInfo",
        "classify_checkout",
        "list_repo",
        "primary_id_for",
        "untracked_id_for",
        "PrimaryCheckoutError",
        "CheckoutTargetError",
    ]
    for name in names:
        assert hasattr(lib_python_worktree, name), f"missing: {name}"
        assert name in lib_python_worktree.__all__, f"not in __all__: {name}"


def test_stop_detail_surface_importable_from_package_root() -> None:
    """ticket #99 (B5): StopDetail and STOP_REASONS must be importable
    directly from the package root, not just from
    lib_python_worktree.core.state."""
    names = ["StopDetail", "STOP_REASONS"]
    for name in names:
        assert hasattr(lib_python_worktree, name), f"missing: {name}"
        assert name in lib_python_worktree.__all__, f"not in __all__: {name}"
