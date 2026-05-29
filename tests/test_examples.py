"""Schema-validation tests for the canonical example contracts (W7).

These tests assert that the example YAML files in ``examples/`` parse and
validate cleanly through the public ``load()`` API.  They exercise the
file-path round-trip (disk -> YAML -> Pydantic) and confirm the structural
properties documented in each example's README.
"""

from __future__ import annotations

from pathlib import Path

from lib_python_worktree.contract import load

# Resolve once relative to this file so the test suite is path-independent
# whether run from the repo root, a worktree, or a CI checkout.
_EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
_CONTRACT = ".seretos/worktree-setup.yml"


def test_webapp_example_parses():
    contract_path = _EXAMPLES_DIR / "webapp" / _CONTRACT
    assert contract_path.exists(), f"example file missing: {contract_path}"

    c = load(contract_path)

    assert c.isolation == "full"

    # Three setup steps in declaration order.
    assert len(c.setup) == 3
    assert c.setup[0].name == "start services"
    assert c.setup[1].name == "install deps"
    assert c.setup[2].name == "run migrations"

    # Two teardown steps.
    assert len(c.teardown) == 2
    assert c.teardown[0].name == "stop services"
    assert c.teardown[1].name == "remove containers"

    # Port slots are exactly [app, chrome] in that order.
    assert [p.name for p in c.ports] == ["app", "chrome"]


def test_unity_example_parses():
    contract_path = _EXAMPLES_DIR / "unity" / _CONTRACT
    assert contract_path.exists(), f"example file missing: {contract_path}"

    c = load(contract_path)

    assert c.isolation == "none"

    # isolation: none forbids setup/teardown/ports — all must be empty.
    assert c.setup == []
    assert c.teardown == []
    assert c.ports == []
