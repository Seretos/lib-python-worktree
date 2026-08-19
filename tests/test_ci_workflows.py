"""Structural regression guard for `.github/workflows/*.yml` (ticket #116).

Asserts the intended trigger shape (pull_request-only, no push) and the
pytest job's timeout headroom, so a `push:` trigger or a shrunken timeout
can't silently creep back in.

Note the YAML 1.1 gotcha: a bare, unquoted `on:` key parses under
`yaml.safe_load` as the Python boolean ``True``, not the string ``"on"``.
``_triggers_of`` resolves that by falling back to ``doc.get(True)``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _load_workflow(name: str) -> dict:
    if not WORKFLOWS_DIR.is_dir():
        pytest.skip(f"workflows directory not found: {WORKFLOWS_DIR}")
    path = WORKFLOWS_DIR / name
    if not path.is_file():
        pytest.skip(f"workflow file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _triggers_of(doc: dict):
    # PyYAML parses the bare `on:` key as boolean True (YAML 1.1), not the
    # string "on" -- resolve both so this doesn't misreport a valid file as
    # missing its trigger block.
    return doc.get("on", doc.get(True))


def test_test_workflow_triggers_on_pull_request_only():
    doc = _load_workflow("test.yml")
    triggers = _triggers_of(doc)
    assert isinstance(triggers, dict), (
        f"expected the resolved trigger mapping to be a dict, got {type(triggers)!r}"
    )
    assert set(triggers) == {"pull_request"}
    assert triggers["pull_request"] is None


def test_lint_workflow_triggers_on_pull_request_only():
    doc = _load_workflow("lint.yml")
    triggers = _triggers_of(doc)
    assert isinstance(triggers, dict), (
        f"expected the resolved trigger mapping to be a dict, got {type(triggers)!r}"
    )
    assert set(triggers) == {"pull_request"}
    assert triggers["pull_request"] is None


def test_pytest_job_timeout_has_headroom():
    doc = _load_workflow("test.yml")
    # A floor, not an exact pin: the invariant this test guards is "enough
    # headroom for the suite to finish", so a later deliberate increase
    # (30, 45, ...) must keep passing rather than fail on an exact match.
    assert doc["jobs"]["pytest"]["timeout-minutes"] >= 25

    steps = doc["jobs"]["pytest"]["steps"]
    run_step = next(
        (s for s in steps if "--cov-fail-under=80" in s.get("run", "")), None
    )
    assert run_step is not None, (
        "expected a step in the pytest job whose `run` contains "
        "'--cov-fail-under=80'; found steps: "
        f"{[s.get('name') for s in steps]}"
    )

    # Unrelated, deliberately-unchanged value: an exact match here is a
    # "did the edit bleed across files" cross-check, not the headroom
    # invariant this test's name describes -- ticket #116 didn't touch
    # lint.yml's actionlint timeout, so pin it exactly.
    lint_doc = _load_workflow("lint.yml")
    assert lint_doc["jobs"]["actionlint"]["timeout-minutes"] == 5
