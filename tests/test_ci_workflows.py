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


# ---------------------------------------------------------------------------
# WP #149 (#145 changelog-in-body + #144 file-into-Backlog): the bump-ticket
# call sites in release.yml (steps id=file_worktree / id=file_workboard, job
# `publish`) and ticket.yml (step id=file-tickets, job `file-tickets`).
# ---------------------------------------------------------------------------

# The idempotency duplicate-check block, preserved byte-for-byte at all three
# sites per the plan -- normalised only for each line's leading indentation
# (release.yml and ticket.yml indent the same text differently).
_DUPLICATE_CHECK = (
    'ISSUE_URL=$(gh api --paginate "repos/${CONSUMER}/issues?state=open&per_page=100" \\\n'
    "--jq --arg t \"$TITLE\" 'map(select((.pull_request|not) and (.title==$t))) | .[0].html_url // empty' 2>/dev/null || true)"
)


def _steps_of(doc: dict, job: str) -> list:
    return doc["jobs"][job]["steps"]


def _step_by_id(steps: list, step_id: str) -> dict:
    step = next((s for s in steps if s.get("id") == step_id), None)
    assert step is not None, (
        f"expected a step with id={step_id!r}; ids present: {[s.get('id') for s in steps]}"
    )
    return step


def _step_by_name_contains(steps: list, needle: str) -> dict:
    step = next((s for s in steps if needle in s.get("name", "")), None)
    assert step is not None, (
        f"expected a step whose name contains {needle!r}; names present: "
        f"{[s.get('name') for s in steps]}"
    )
    return step


def _fetch_step(steps: list) -> dict:
    step = next((s for s in steps if "gh release view" in s.get("run", "")), None)
    assert step is not None, (
        "expected a changelog-fetch step whose run contains 'gh release view'; "
        f"names present: {[s.get('name') for s in steps]}"
    )
    return step


def _board_block(run_text: str) -> str:
    """Slice the degrading brace group out of a `run:` string.

    From the line containing `ITEM_ID=` to the line containing
    `} || echo "::warning::Could not move`, inclusive -- so assertions
    about the exit-vs-false guard can be scoped to just this block and not
    false-positive on an unrelated `exit 1` elsewhere in the same step
    (ticket.yml's semver validation, in particular).
    """
    lines = run_text.splitlines()
    start = next((i for i, l in enumerate(lines) if "ITEM_ID=" in l), None)
    end = next(
        (i for i, l in enumerate(lines) if '} || echo "::warning::Could not move' in l),
        None,
    )
    assert start is not None, "expected a line containing 'ITEM_ID=' in the run text"
    assert end is not None, (
        'expected a line containing \'} || echo "::warning::Could not move\' in the run text'
    )
    return "\n".join(lines[start : end + 1])


def _sites():
    """The three bump-ticket-filing call sites (id + env + run holder)."""
    release_doc = _load_workflow("release.yml")
    ticket_doc = _load_workflow("ticket.yml")
    release_steps = _steps_of(release_doc, "publish")
    ticket_steps = _steps_of(ticket_doc, "file-tickets")
    return [
        ("release.yml/file_worktree", _step_by_id(release_steps, "file_worktree")),
        ("release.yml/file_workboard", _step_by_id(release_steps, "file_workboard")),
        ("ticket.yml/file-tickets", _step_by_id(ticket_steps, "file-tickets")),
    ]


def _board_sites():
    """The three board-add call sites (release.yml's are separate steps from
    the ticket-filing steps; ticket.yml's shares the file-tickets step)."""
    release_doc = _load_workflow("release.yml")
    ticket_doc = _load_workflow("ticket.yml")
    release_steps = _steps_of(release_doc, "publish")
    ticket_steps = _steps_of(ticket_doc, "file-tickets")
    return [
        (
            "release.yml/Add agent-worktree ticket to project board",
            _step_by_name_contains(
                release_steps, "Add agent-worktree ticket to project board"
            ),
        ),
        (
            "release.yml/Add workboard ticket to project board",
            _step_by_name_contains(
                release_steps, "Add workboard ticket to project board"
            ),
        ),
        ("ticket.yml/file-tickets", _step_by_id(ticket_steps, "file-tickets")),
    ]


def _strip_leading_whitespace(text: str) -> str:
    return "\n".join(line.lstrip() for line in text.splitlines())


# --- Behaviour 1 -------------------------------------------------------------


def test_bump_ticket_body_places_what_changed_before_action_required():
    for label, step in _sites():
        env = step.get("env", {})
        assert "BODY_HEAD" in env, f"{label}: expected env to declare BODY_HEAD"
        assert "BODY_TAIL" in env, f"{label}: expected env to declare BODY_TAIL"
        assert "BODY" not in env, (
            f"{label}: expected no bare BODY env key alongside BODY_HEAD/BODY_TAIL"
        )
        body_head = env.get("BODY_HEAD", "")
        body_tail = env.get("BODY_TAIL", "")
        assert "## Dependency update" in body_head, (
            f"{label}: expected BODY_HEAD to contain '## Dependency update'"
        )
        assert "### Action required" not in body_head, (
            f"{label}: expected BODY_HEAD to NOT contain '### Action required'"
        )
        assert body_tail.lstrip().startswith("### Action required"), (
            f"{label}: expected BODY_TAIL to start with '### Action required'"
        )
        run = step.get("run", "")
        assert '"$BODY_HEAD"' in run, (
            f'{label}: expected the run text to actually reference "$BODY_HEAD" '
            "when composing the body, not just declare it in env"
        )
        assert "### What changed" in run, (
            f"{label}: expected the run text to compose in a '### What changed' heading"
        )
        idx_head = run.index('"$BODY_HEAD"')
        idx_changed = run.index("### What changed")
        idx_tail = run.index('"$BODY_TAIL"')
        assert idx_head < idx_changed < idx_tail, (
            f'{label}: expected the composed order "$BODY_HEAD" ({idx_head}) < '
            f"'### What changed' ({idx_changed}) < \"$BODY_TAIL\" ({idx_tail})"
        )

        # F1: the composition must actually emit the fetched changelog content
        # -- not just the "### What changed" heading with a hardcoded body
        # that never reads $CHANGELOG_FILE at all.
        assert 'cat "$CHANGELOG_FILE"' in run, (
            f'{label}: expected the run text to actually cat "$CHANGELOG_FILE" '
            "when composing the body, not just declare the heading"
        )
        idx_cat = run.index('cat "$CHANGELOG_FILE"')
        assert idx_changed < idx_cat < idx_tail, (
            f'{label}: expected cat "$CHANGELOG_FILE" ({idx_cat}) to sit between '
            f"'### What changed' ({idx_changed}) and \"$BODY_TAIL\" ({idx_tail})"
        )


def test_bump_ticket_body_preserves_pin_instructions():
    for label, step in _sites():
        env = step.get("env", {})
        body_tail = env.get("BODY_TAIL", "")
        assert (
            "git+https://github.com/Seretos/lib-python-worktree@v" in body_tail
        ), f"{label}: expected BODY_TAIL to keep the pin instruction line"
        for item in (
            "1. Update the pin",
            "2. Run `pwsh scripts/test.ps1`",
            "3. Run `pwsh scripts/build.ps1`",
            "4. Commit and open a PR.",
        ):
            assert item in body_tail, (
                f"{label}: expected BODY_TAIL to still contain {item!r}"
            )


def test_bump_ticket_title_unchanged():
    # Preservation guard -- already passes today, TITLE is untouched by the plan.
    for label, step in _sites():
        env = step.get("env", {})
        assert (
            env.get("TITLE")
            == "chore(deps): bump lib-python-worktree to v${{ inputs.version }}"
        ), f"{label}: expected TITLE unchanged, got {env.get('TITLE')!r}"


# --- Behaviour 2 -------------------------------------------------------------


def test_ticket_steps_use_body_file():
    for label, step in _sites():
        run = step.get("run", "")
        assert '--body "$BODY"' not in run, (
            f"{label}: expected no remaining '--body \"$BODY\"' usage"
        )
        assert run.count('--body-file "$BODY_FILE"') == 2, (
            f"{label}: expected exactly two '--body-file \"$BODY_FILE\"' invocations, "
            f"found {run.count('--body-file \"$BODY_FILE\"')}"
        )
        assert "BODY_FILE=" in run, (
            f"{label}: expected BODY_FILE to actually be assigned (e.g. "
            "BODY_FILE=\"$(mktemp)\"), not just referenced"
        )
        assert '> "$BODY_FILE"' in run, (
            f"{label}: expected the body-composition block's output to be "
            'redirected into BODY_FILE via \'> "$BODY_FILE"\''
        )


def test_duplicate_check_invocation_unchanged():
    # Regression guard -- may already pass today; the duplicate-check block
    # is untouched by the plan.
    for label, step in _sites():
        run = step.get("run", "")
        normalized = _strip_leading_whitespace(run)
        assert _DUPLICATE_CHECK in normalized, (
            f"{label}: expected the duplicate-check block preserved verbatim "
            "(modulo indentation)"
        )


def test_label_fallback_pair_preserved():
    for label, step in _sites():
        run = step.get("run", "")
        assert run.count("--label dependencies") == 1, (
            f"{label}: expected exactly one '--label dependencies'"
        )
        assert run.count("gh issue create") == 2, (
            f"{label}: expected exactly two 'gh issue create' invocations"
        )


def test_body_composition_falls_back_when_changelog_missing():
    for label, step in _sites():
        run = step.get("run", "")
        # F2: the two guards must be combined with && (not textually adjacent
        # but independently satisfiable, which would also pass an inverted
        # ||-based arrangement).
        assert '[ -n "${CHANGELOG_FILE:-}" ] && [ -s "${CHANGELOG_FILE:-}" ]' in run, (
            f"{label}: expected the combined changelog-file-present-and-non-empty "
            'guard \'[ -n "${CHANGELOG_FILE:-}" ] && [ -s "${CHANGELOG_FILE:-}" ]\''
        )
        assert "releases/tag/" in run, (
            f"{label}: expected a releases/tag/ fallback link in the composition block"
        )
        assert 'cat "$CHANGELOG_FILE"' in run, (
            f"{label}: expected the composition block to cat \"$CHANGELOG_FILE\""
        )
        idx_cat = run.index('cat "$CHANGELOG_FILE"')
        idx_fallback = run.index("releases/tag/")
        assert idx_cat < idx_fallback, (
            f"{label}: expected cat \"$CHANGELOG_FILE\" ({idx_cat}) -- the 'then' "
            f"branch -- to precede the releases/tag/ fallback ({idx_fallback}) -- "
            "the 'else' branch"
        )


# --- Behaviour 3 -------------------------------------------------------------


def test_changelog_fetch_step_exists_positioned_and_authenticated():
    release_doc = _load_workflow("release.yml")
    release_steps = _steps_of(release_doc, "publish")
    create_idx = next(
        i
        for i, s in enumerate(release_steps)
        if s.get("name") == "Create GitHub Release"
    )
    fetch_idx = next(
        (i for i, s in enumerate(release_steps) if "gh release view" in s.get("run", "")),
        None,
    )
    assert fetch_idx is not None, (
        "release.yml: expected a changelog-fetch step whose run contains 'gh release view'"
    )
    file_worktree_idx = next(
        i for i, s in enumerate(release_steps) if s.get("id") == "file_worktree"
    )
    assert create_idx < fetch_idx < file_worktree_idx, (
        f"release.yml: expected fetch step index ({fetch_idx}) to sit between "
        f"'Create GitHub Release' ({create_idx}) and file_worktree ({file_worktree_idx})"
    )
    release_fetch_run = release_steps[fetch_idx].get("run", "")
    assert (
        'echo "CHANGELOG_FILE=${CHANGELOG_FILE}" >> "$GITHUB_ENV"' in release_fetch_run
    ), (
        "release.yml: expected the fetch step to export CHANGELOG_FILE via the "
        'combined line echo "CHANGELOG_FILE=${CHANGELOG_FILE}" >> "$GITHUB_ENV", '
        "not two independently-satisfiable substrings"
    )
    release_fetch_env = release_steps[fetch_idx].get("env", {})
    assert release_fetch_env.get("GH_TOKEN") == "${{ secrets.GITHUB_TOKEN }}", (
        "release.yml: expected the fetch step's own env GH_TOKEN to be "
        f"secrets.GITHUB_TOKEN, got {release_fetch_env.get('GH_TOKEN')!r}"
    )
    assert 'TAG="v${{ inputs.version }}"' in release_fetch_run, (
        "release.yml: expected the fetch step to derive TAG directly from the "
        'workflow input via the literal TAG="v${{ inputs.version }}", not from '
        "the step-scoped VERSION env var"
    )
    assert 'gh release view "$TAG"' in release_fetch_run, (
        'release.yml: expected the release-view invocation to use gh release '
        'view "$TAG" against the input-derived TAG variable'
    )

    ticket_doc = _load_workflow("ticket.yml")
    ticket_steps = _steps_of(ticket_doc, "file-tickets")
    fetch_idx_t = next(
        (i for i, s in enumerate(ticket_steps) if "gh release view" in s.get("run", "")),
        None,
    )
    assert fetch_idx_t is not None, (
        "ticket.yml: expected a changelog-fetch step whose run contains 'gh release view'"
    )
    file_tickets_idx = next(
        i for i, s in enumerate(ticket_steps) if s.get("id") == "file-tickets"
    )
    assert fetch_idx_t < file_tickets_idx, (
        f"ticket.yml: expected fetch step index ({fetch_idx_t}) to sit before "
        f"the file-tickets step ({file_tickets_idx})"
    )
    ticket_fetch_run = ticket_steps[fetch_idx_t].get("run", "")
    assert (
        'echo "CHANGELOG_FILE=${CHANGELOG_FILE}" >> "$GITHUB_ENV"' in ticket_fetch_run
    ), (
        "ticket.yml: expected the fetch step to export CHANGELOG_FILE via the "
        'combined line echo "CHANGELOG_FILE=${CHANGELOG_FILE}" >> "$GITHUB_ENV", '
        "not two independently-satisfiable substrings"
    )
    ticket_fetch_env = ticket_steps[fetch_idx_t].get("env", {})
    assert ticket_fetch_env.get("GH_TOKEN") == "${{ secrets.GITHUB_TOKEN }}", (
        "ticket.yml: expected the fetch step's own env GH_TOKEN to be "
        f"secrets.GITHUB_TOKEN, got {ticket_fetch_env.get('GH_TOKEN')!r}"
    )
    assert 'TAG="v${{ inputs.version }}"' in ticket_fetch_run, (
        "ticket.yml: expected the fetch step to derive TAG directly from the "
        'workflow input via the literal TAG="v${{ inputs.version }}" -- it '
        "CANNOT reuse the step-scoped VERSION env var, which belongs to the "
        "other, consuming step"
    )
    assert 'gh release view "$TAG"' in ticket_fetch_run, (
        'ticket.yml: expected the release-view invocation to use gh release '
        'view "$TAG" against the input-derived TAG variable'
    )


def test_no_workflow_relies_on_job_level_env_for_gh_token():
    # Preservation guard -- already passes today, neither job has a
    # job-level env block.
    release_doc = _load_workflow("release.yml")
    ticket_doc = _load_workflow("ticket.yml")
    assert "env" not in release_doc["jobs"]["publish"], (
        "release.yml: expected job `publish` to declare no job-level env"
    )
    assert "env" not in ticket_doc["jobs"]["file-tickets"], (
        "ticket.yml: expected job `file-tickets` to declare no job-level env"
    )


def test_fetch_step_targets_this_repo():
    release_doc = _load_workflow("release.yml")
    release_fetch = _fetch_step(_steps_of(release_doc, "publish"))
    ticket_doc = _load_workflow("ticket.yml")
    ticket_fetch = _fetch_step(_steps_of(ticket_doc, "file-tickets"))
    for label, step in (("release.yml", release_fetch), ("ticket.yml", ticket_fetch)):
        assert '--repo "${{ github.repository }}"' in step.get("run", ""), (
            f"{label}: expected the fetch step to target --repo \"${{ github.repository }}\""
        )


def test_ticket_workflow_permissions_allow_release_read():
    # Preservation guard -- already passes today (contents: read).
    doc = _load_workflow("ticket.yml")
    assert doc["permissions"]["contents"] in ("read", "write"), (
        f"expected ticket.yml permissions.contents to allow reads, got "
        f"{doc['permissions']['contents']!r}"
    )


# --- Behaviour 4 -------------------------------------------------------------


def test_board_add_moves_card_to_backlog():
    for label, step in _board_sites():
        run = step.get("run", "")
        assert "gh project item-add 2 --owner Seretos" in run, (
            f"{label}: expected 'gh project item-add 2 --owner Seretos'"
        )
        assert "--format json --jq .id" in run, (
            f"{label}: expected item-add to capture the item id via --format json --jq .id"
        )
        assert "gh project field-list 2 --owner Seretos" in run, (
            f"{label}: expected 'gh project field-list 2 --owner Seretos'"
        )
        assert 'select(.name=="Status")' in run, (
            f"{label}: expected a jq filter selecting the Status field"
        )
        assert 'select(.name=="Backlog")' in run, (
            f"{label}: expected a jq filter selecting the Backlog option"
        )
        assert "gh project item-edit" in run, (
            f"{label}: expected 'gh project item-edit'"
        )
        assert "--single-select-option-id" in run, (
            f"{label}: expected item-edit to use --single-select-option-id"
        )
        assert "--field-id" in run, (
            f"{label}: expected item-edit to be given the resolved --field-id"
        )
        assert "--project-id" in run, (
            f"{label}: expected item-edit to be given the resolved --project-id"
        )
        # F6: pin the exact --field-id/--single-select-option-id pairing (not
        # swapped) and confirm the item-edit call sits on the success path --
        # the `else` branch of `if [ -z "$ITEM_ID" ]`, not the failure branch.
        assert '--field-id "$FIELD_ID" --single-select-option-id "$OPTION_ID"' in run, (
            f"{label}: expected item-edit to pair --field-id \"$FIELD_ID\" with "
            '--single-select-option-id "$OPTION_ID" (not swapped)'
        )
        lines = run.splitlines()
        if_empty_idx = next(
            (i for i, l in enumerate(lines) if 'if [ -z "$ITEM_ID" ]' in l), None
        )
        assert if_empty_idx is not None, (
            f'{label}: expected an \'if [ -z "$ITEM_ID" ]\' branch'
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
            f"{label}: expected a standalone 'else' line closing the "
            "ITEM_ID-missing if-branch"
        )
        item_edit_idx = next(
            i for i, l in enumerate(lines) if "gh project item-edit" in l
        )
        assert item_edit_idx > else_idx, (
            f"{label}: expected 'gh project item-edit' (line {item_edit_idx}) to sit "
            f"inside the success branch, after 'else' (line {else_idx}), not inside "
            'the \'if [ -z "$ITEM_ID" ]\' failure branch'
        )


def test_board_degrade_block_never_uses_exit():
    # Regression guard for the exit-1 bug: `exit` inside a `{ ...; }` group
    # terminates the shell and is NOT caught by the group's `|| echo
    # "::warning::..."` -- only `|| false` degrades cleanly.
    for label, step in _board_sites():
        run = step.get("run", "")
        block = _board_block(run)
        assert "exit 1" not in block, (
            f"{label}: the board-degrade block must not contain 'exit 1'"
        )
        assert "|| false" in block, (
            f"{label}: the board-degrade block must contain '|| false'"
        )


def test_board_add_uses_correct_token_per_site():
    release_doc = _load_workflow("release.yml")
    release_steps = _steps_of(release_doc, "publish")
    worktree_board = _step_by_name_contains(
        release_steps, "Add agent-worktree ticket to project board"
    )
    workboard_board = _step_by_name_contains(
        release_steps, "Add workboard ticket to project board"
    )
    assert (
        worktree_board.get("env", {}).get("GH_TOKEN")
        == "${{ secrets.WORKTREE_TICKET_TOKEN }}"
    ), "release.yml: expected the worktree board-add step to keep WORKTREE_TICKET_TOKEN"
    assert (
        workboard_board.get("env", {}).get("GH_TOKEN")
        == "${{ secrets.WORKBOARD_TICKET_TOKEN }}"
    ), "release.yml: expected the workboard board-add step to keep WORKBOARD_TICKET_TOKEN"

    ticket_doc = _load_workflow("ticket.yml")
    ticket_steps = _steps_of(ticket_doc, "file-tickets")
    file_tickets_step = _step_by_id(ticket_steps, "file-tickets")
    run = file_tickets_step.get("run", "")
    export_idx = run.index('export GH_TOKEN="$TOKEN"')
    item_add_idx = run.index("gh project item-add 2 --owner Seretos")
    assert export_idx < item_add_idx, (
        "ticket.yml: expected the board-add call to come after "
        'export GH_TOKEN="$TOKEN" inside the loop'
    )

    # F3: the export/item-add ordering above is also satisfied if the whole
    # board-add block were hoisted out of the `for CONSUMER in $CONSUMERS`
    # loop and placed after the loop's `done` -- pin the item-add call
    # strictly between the loop's opening `for` line and its closing `done`.
    lines = run.splitlines()
    for_idx = next(
        (i for i, l in enumerate(lines) if "for CONSUMER in $CONSUMERS" in l), None
    )
    assert for_idx is not None, (
        "ticket.yml: expected a 'for CONSUMER in $CONSUMERS' loop opening line"
    )
    done_idx = next(
        (i for i, l in enumerate(lines) if i > for_idx and l.strip() == "done"), None
    )
    assert done_idx is not None, (
        "ticket.yml: expected a standalone 'done' line closing the CONSUMER loop"
    )
    item_add_line_idx = next(
        i for i, l in enumerate(lines) if "gh project item-add 2 --owner Seretos" in l
    )
    assert for_idx < item_add_line_idx < done_idx, (
        f"ticket.yml: expected the board-add call (line {item_add_line_idx}) to sit "
        f"strictly inside the CONSUMER loop (opens at {for_idx}, closes at {done_idx})"
    )


def test_board_add_steps_preserve_guards():
    # Preservation guard -- already passes today.
    release_doc = _load_workflow("release.yml")
    release_steps = _steps_of(release_doc, "publish")
    worktree_board = _step_by_name_contains(
        release_steps, "Add agent-worktree ticket to project board"
    )
    workboard_board = _step_by_name_contains(
        release_steps, "Add workboard ticket to project board"
    )
    assert worktree_board.get("continue-on-error") is True
    assert workboard_board.get("continue-on-error") is True
    assert worktree_board.get("if") == "steps.file_worktree.outputs.issue_url != ''"
    assert workboard_board.get("if") == "steps.file_workboard.outputs.issue_url != ''"


def test_release_board_sites_bind_issue_url_and_token_name():
    release_doc = _load_workflow("release.yml")
    release_steps = _steps_of(release_doc, "publish")
    worktree_board = _step_by_name_contains(
        release_steps, "Add agent-worktree ticket to project board"
    )
    workboard_board = _step_by_name_contains(
        release_steps, "Add workboard ticket to project board"
    )
    worktree_run = worktree_board.get("run", "")
    workboard_run = workboard_board.get("run", "")

    assert (
        'ISSUE_URL="${{ steps.file_worktree.outputs.issue_url }}"' in worktree_run
    ), (
        "release.yml: expected the worktree board-add step to bind ISSUE_URL "
        "exactly to file_worktree's issue_url output"
    )
    assert 'TOKEN_NAME="WORKTREE_TICKET_TOKEN"' in worktree_run, (
        "release.yml: expected the worktree board-add step to bind TOKEN_NAME "
        "exactly to WORKTREE_TICKET_TOKEN"
    )

    assert (
        'ISSUE_URL="${{ steps.file_workboard.outputs.issue_url }}"' in workboard_run
    ), (
        "release.yml: expected the workboard board-add step to bind ISSUE_URL "
        "exactly to file_workboard's issue_url output"
    )
    assert 'TOKEN_NAME="WORKBOARD_TICKET_TOKEN"' in workboard_run, (
        "release.yml: expected the workboard board-add step to bind TOKEN_NAME "
        "exactly to WORKBOARD_TICKET_TOKEN"
    )


# --- Behaviour 5 -------------------------------------------------------------


def test_changelog_fetch_degrades_and_truncates():
    release_doc = _load_workflow("release.yml")
    release_fetch = _fetch_step(_steps_of(release_doc, "publish"))
    ticket_doc = _load_workflow("ticket.yml")
    ticket_fetch = _fetch_step(_steps_of(ticket_doc, "file-tickets"))

    for label, step in (("release.yml", release_fetch), ("ticket.yml", ticket_fetch)):
        run = step.get("run", "")
        assert "gh release view" in run
        assert "|| true" in run, f"{label}: expected the fetch to be guarded with || true"
        assert 'if [ -z "$NOTES" ]' in run, f"{label}: expected an empty-NOTES branch"

        lines = run.splitlines()
        if_line_idx = next(
            i for i, l in enumerate(lines) if 'if [ -z "$NOTES" ]' in l
        )
        else_line_idx = next(
            (i for i, l in enumerate(lines) if i > if_line_idx and l.strip() == "else"),
            None,
        )
        assert else_line_idx is not None, (
            f"{label}: expected a standalone 'else' line closing the empty-NOTES "
            "if-branch so it can be sliced out for scoped assertions"
        )
        empty_branch = "\n".join(lines[if_line_idx : else_line_idx + 1])
        assert "::warning::" in empty_branch, (
            f"{label}: expected the ::warning:: echo to be inside the empty-NOTES "
            "`if` branch specifically (found elsewhere, e.g. the else branch)"
        )
        assert "releases/tag/" in empty_branch, (
            f"{label}: expected the releases/tag/ fallback link to be inside the "
            "empty-NOTES `if` branch specifically (found elsewhere, e.g. the else branch)"
        )

        assert "${#NOTES}" in run, f"{label}: expected a NOTES length check"
        assert "-gt 30000" in run, f"{label}: expected a 30000-char truncation threshold"
        assert "${NOTES:0:30000}" in run, f"{label}: expected NOTES to be sliced to 30000 chars"
        marker = (
            "_(release notes truncated at 30000 characters; "
            "see the full release page.)_"
        )
        assert marker in run, (
            f"{label}: expected the exact truncation marker sentence {marker!r} "
            "to appear in the run text"
        )
        marker_idx = run.index(marker)
        reassign_idx = run.index('NOTES="${NOTES:0:30000}')
        assert reassign_idx < marker_idx, (
            f"{label}: expected the marker sentence to be appended onto the "
            'reassigned NOTES value -- i.e. NOTES="${NOTES:0:30000}\' ('
            f"{reassign_idx}) must precede the marker ({marker_idx}), not be a "
            "disconnected comment elsewhere in the step"
        )

        idx_trunc = run.index("-gt 30000")
        idx_if_empty = run.index('if [ -z "$NOTES" ]')
        idx_write = run.index('printf \'%s\\n\' "$NOTES" > "$CHANGELOG_FILE"')
        assert idx_trunc < idx_if_empty < idx_write, (
            f"{label}: expected ordering truncate ({idx_trunc}) < empty-check "
            f"({idx_if_empty}) < write ({idx_write})"
        )
        assert 'NOTES="${NOTES:0:30000}' in run, (
            f"{label}: expected the truncation to reassign NOTES in place"
        )


def test_truncation_cannot_mask_empty_notes():
    release_doc = _load_workflow("release.yml")
    release_fetch = _fetch_step(_steps_of(release_doc, "publish"))
    ticket_doc = _load_workflow("ticket.yml")
    ticket_fetch = _fetch_step(_steps_of(ticket_doc, "file-tickets"))
    for label, step in (("release.yml", release_fetch), ("ticket.yml", ticket_fetch)):
        run = step.get("run", "")
        assert '"${#NOTES}" -gt 30000' in run, (
            f'{label}: expected the length comparison \'"${{#NOTES}}" -gt 30000\''
        )


def test_fetch_step_never_fails_workflow():
    release_doc = _load_workflow("release.yml")
    release_fetch = _fetch_step(_steps_of(release_doc, "publish"))
    ticket_doc = _load_workflow("ticket.yml")
    ticket_fetch = _fetch_step(_steps_of(ticket_doc, "file-tickets"))
    for label, step in (("release.yml", release_fetch), ("ticket.yml", ticket_fetch)):
        run = step.get("run", "")
        assert "exit 1" not in run, f"{label}: fetch step must not declare exit 1"
        # F4: the guard must be attached directly to the gh call's own tail --
        # two independently-satisfiable substrings ("gh release view" plus an
        # unrelated "2>/dev/null || true" elsewhere) would also pass.
        assert "--jq .body 2>/dev/null || true" in run, (
            f"{label}: expected the fetch call's --json body --jq .body tail to be "
            "directly guarded by 2>/dev/null || true, as one contiguous substring"
        )


# --- Behaviour 6 (regression guard: should pass now AND after implementation) ---


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
    assert {"branch", "file_worktree", "file_workboard"} <= release_step_ids, (
        f"release.yml: expected step ids to include branch/file_worktree/file_workboard, "
        f"found: {release_step_ids}"
    )

    ticket_doc = _load_workflow("ticket.yml")
    assert set(ticket_doc["jobs"].keys()) == {"file-tickets"}
    ticket_steps = _steps_of(ticket_doc, "file-tickets")
    ticket_step_ids = {s.get("id") for s in ticket_steps if s.get("id")}
    assert "file-tickets" in ticket_step_ids
    report_step = _step_by_name_contains(ticket_steps, "Report per-consumer failures")
    report_env = report_step.get("env", {})
    assert (
        report_env.get("FAILED_CONSUMERS")
        == "${{ steps.file-tickets.outputs.failed_consumers }}"
    ), (
        "ticket.yml: expected the final step's env to read "
        "steps.file-tickets.outputs.failed_consumers, got "
        f"{report_env.get('FAILED_CONSUMERS')!r}"
    )
