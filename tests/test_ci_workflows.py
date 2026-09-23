"""Structural regression guard for `.github/workflows/*.yml` (ticket #116).

Asserts the intended trigger shape (pull_request-only, no push) and the
pytest job's timeout headroom, so a `push:` trigger or a shrunken timeout
can't silently creep back in.

Note the YAML 1.1 gotcha: a bare, unquoted `on:` key parses under
`yaml.safe_load` as the Python boolean ``True``, not the string ``"on"``.
``_triggers_of`` resolves that by falling back to ``doc.get(True)``.
"""

from __future__ import annotations

import re
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


def _steps_of(doc: dict, job: str) -> list:
    return doc["jobs"][job]["steps"]


# --- Behaviour 7 (#156: notes must start at the actual previous tag) --------


def test_release_creation_passes_explicit_notes_start_tag():
    release_doc = _load_workflow("release.yml")
    release_steps = _steps_of(release_doc, "publish")
    create_step = next(
        (s for s in release_steps if s.get("name") == "Create GitHub Release"), None
    )
    assert create_step is not None, (
        "release.yml: expected a step named 'Create GitHub Release'; names present: "
        f"{[s.get('name') for s in release_steps]}"
    )
    run = create_step.get("run", "")

    lines = run.splitlines()
    prev_tag_idx = next(
        (i for i, l in enumerate(lines) if "PREV_TAG=$(" in l), None
    )
    assert prev_tag_idx is not None, (
        "release.yml: expected a 'PREV_TAG=$(' assignment line in the "
        "'Create GitHub Release' run text"
    )
    prev_tag_line = lines[prev_tag_idx]

    # Parse the actual helper-script path out of the run text (rather than
    # asserting a hardcoded expected path exists) so a typo'd path in the
    # real workflow (e.g. '.github/script/' instead of '.github/scripts/')
    # fails this test instead of silently passing (test-critic round 1, gap
    # 2).
    path_match = re.search(r"(\S*prev_release_tag\.py)", prev_tag_line)
    assert path_match is not None, (
        "release.yml: expected the 'PREV_TAG=$(' line to reference a "
        f"*prev_release_tag.py script path, found: {prev_tag_line!r}"
    )
    referenced_path = path_match.group(1)
    repo_root = WORKFLOWS_DIR.parents[1]
    helper_path = repo_root / referenced_path
    assert helper_path.is_file(), (
        f"release.yml references helper script path {referenced_path!r} "
        f"(resolved to {helper_path}), which does not exist on disk"
    )

    # Cross-check the version argument passed to the helper against the
    # real version expression this same step already uses elsewhere (the
    # PRE_FLAG prerelease check), instead of hardcoding "${{ inputs.version
    # }}" here -- this forces the wiring to actually pass the live
    # expression rather than e.g. a hardcoded/wrong literal like "0.0.0"
    # (test-critic round 1, gap 1).
    version_expr_match = re.search(r'\[\[\s*"([^"]+)"\s*==\s*\*-\*\s*\]\]', run)
    assert version_expr_match is not None, (
        "release.yml: expected to find the existing prerelease check "
        '(`if [[ "<version-expr>" == *-* ]]`) in the run text to derive '
        "the canonical version expression from"
    )
    version_expr = version_expr_match.group(1)
    assert f'"{version_expr}"' in prev_tag_line, (
        f"release.yml: expected the 'PREV_TAG=$(' line to pass the same "
        f"version expression ({version_expr!r}) used elsewhere in this "
        f"step to the helper script, found: {prev_tag_line!r}"
    )

    create_call_idx = next(
        (i for i, l in enumerate(lines) if "gh release create" in l), None
    )
    assert create_call_idx is not None, (
        "release.yml: expected a 'gh release create' invocation in the "
        "'Create GitHub Release' run text"
    )
    assert prev_tag_idx < create_call_idx, (
        f"release.yml: expected PREV_TAG assignment ({prev_tag_idx}) to precede "
        f"the gh release create invocation ({create_call_idx})"
    )

    if_empty_idx = next(
        (i for i, l in enumerate(lines) if 'if [ -z "$PREV_TAG" ]' in l), None
    )
    assert if_empty_idx is not None, (
        'release.yml: expected an \'if [ -z "$PREV_TAG" ]\' branch'
    )
    else_idx = next(
        (
            i
            for i, l in enumerate(lines)
            if i > if_empty_idx and l.strip() == "else"
        ),
        None,
    )
    assert else_idx is not None, (
        "release.yml: expected a standalone 'else' line closing the "
        "PREV_TAG-empty if-branch"
    )
    fi_idx = next(
        (i for i, l in enumerate(lines) if i > else_idx and l.strip() == "fi"),
        None,
    )
    assert fi_idx is not None, (
        "release.yml: expected a standalone 'fi' line closing the "
        "PREV_TAG if/else"
    )

    if_branch = "\n".join(lines[if_empty_idx : else_idx + 1])
    else_branch = "\n".join(lines[else_idx : fi_idx + 1])

    assert "::warning::" in if_branch, (
        "release.yml: expected the '::warning::' echo to be inside the "
        'PREV_TAG-empty \'if\' branch specifically'
    )
    assert 'NOTES_START="--notes-start-tag $PREV_TAG"' in else_branch, (
        "release.yml: expected NOTES_START=\"--notes-start-tag $PREV_TAG\" to sit "
        "inside the 'else' branch of the PREV_TAG-empty check"
    )
    assert 'NOTES_START="--notes-start-tag $PREV_TAG"' not in if_branch, (
        "release.yml: expected NOTES_START assignment to NOT be inside the "
        "PREV_TAG-empty 'if' branch"
    )

    # `gh release create` spans multiple continuation lines (each ending in
    # `\`) in the actual step; scan forward from the invocation line to the
    # end of that logical command for the flags, rather than assuming
    # everything sits on one physical line.
    call_block_end = create_call_idx
    while call_block_end < len(lines) and lines[call_block_end].rstrip().endswith(
        "\\"
    ):
        call_block_end += 1
    create_block = "\n".join(lines[create_call_idx : call_block_end + 1])
    assert "$NOTES_START" in create_block, (
        "release.yml: expected the gh release create invocation to reference "
        "$NOTES_START"
    )
    assert "--generate-notes" in create_block, (
        "release.yml: expected the gh release create invocation to keep "
        "--generate-notes"
    )


def test_workflow_topology_unchanged():
    release_doc = _load_workflow("release.yml")
    assert set(release_doc["jobs"].keys()) == {"publish"}
    release_steps = _steps_of(release_doc, "publish")
    release_step_ids = {s.get("id") for s in release_steps if s.get("id")}
    assert "branch" in release_step_ids, (
        f"release.yml: expected step ids to include branch, found: {release_step_ids}"
    )
