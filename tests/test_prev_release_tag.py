"""Driving tests for #156: release notes must start at the actual previous
release tag, not the beginning of history.

`.github/scripts/prev_release_tag.py` is a standalone helper invoked from
`release.yml`'s "Create GitHub Release" step (not part of the installed
package), so it is loaded here via `importlib.util.spec_from_file_location`
rather than a normal import.

`previous_tag(version, tags)` returns the greatest `v*` tag whose semver
precedence is strictly below `version` -- real semver precedence (numeric
triple; release > prerelease of the same triple; dot-separated prerelease
identifiers, numeric-before-alphanumeric, fewer fields first), not `sort -V`
and not lexical string comparison.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "prev_release_tag.py"
)


def _load_module():
    if not SCRIPT_PATH.is_file():
        raise FileNotFoundError(
            f"expected helper script at {SCRIPT_PATH}, but it does not exist yet"
        )
    spec = importlib.util.spec_from_file_location("prev_release_tag", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mod():
    return _load_module()


# --- (a) real-shaped list -----------------------------------------------


def test_real_shaped_list_finds_immediate_predecessor(mod):
    tags = [
        "v0.1.0",
        "v0.2.0",
        "v0.3.0",
        "v0.3.1",
        "v0.3.2",
        "v0.3.3",
        "v0.3.4",
        "v0.3.5",
        "v0.3.6",
        "v0.3.7",
        "v0.3.8",
        "v0.3.9",
        "v0.3.10",
        "v0.3.11",
        "v0.3.12",
        "v0.3.13",
    ]
    assert mod.previous_tag("0.3.13", tags) == "v0.3.12"


# --- (b) unsorted input order --------------------------------------------


def test_unsorted_input_order_gives_same_result(mod):
    tags = [
        "v0.3.13",
        "v0.1.0",
        "v0.3.10",
        "v0.3.12",
        "v0.2.0",
        "v0.3.1",
        "v0.3.9",
    ]
    assert mod.previous_tag("0.3.13", tags) == "v0.3.12"


# --- (c) prerelease neighbour ---------------------------------------------


def test_prerelease_tag_is_previous_for_its_own_release(mod):
    tags = ["v0.3.12", "v0.3.13-rc.1"]
    assert mod.previous_tag("0.3.13", tags) == "v0.3.13-rc.1"


def test_prerelease_tag_is_previous_for_a_later_prerelease(mod):
    tags = ["v0.3.12", "v0.3.13-rc.1"]
    assert mod.previous_tag("0.3.13-rc.2", tags) == "v0.3.13-rc.1"


# --- (d) no previous tag ---------------------------------------------------


def test_only_the_new_tag_itself_present_returns_none(mod):
    assert mod.previous_tag("0.3.13", ["v0.3.13"]) is None


def test_empty_tag_list_returns_none(mod):
    assert mod.previous_tag("0.3.13", []) is None


# --- (e) malformed entries are ignored, not raised on ----------------------


def test_malformed_entries_are_ignored(mod):
    tags = ["v0.3.12", "v0.3", "vfoo", "release/0.x", "v0.3.13"]
    assert mod.previous_tag("0.3.13", tags) == "v0.3.12"


# --- (f) main() end-to-end --------------------------------------------------


def test_main_end_to_end_prints_tag_and_exits_zero(mod, monkeypatch, capsys):
    stdin_text = "v0.1.0\nv0.3.12\nv0.3.13\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    monkeypatch.setattr(sys, "argv", ["prev_release_tag.py", "0.3.13"])
    exit_code = None
    try:
        mod.main()
    except SystemExit as e:
        exit_code = e.code
    out = capsys.readouterr().out
    assert out == "v0.3.12\n"
    assert exit_code in (None, 0)


def test_main_end_to_end_prints_nothing_when_no_previous_tag(mod, monkeypatch, capsys):
    stdin_text = "v0.3.13\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    monkeypatch.setattr(sys, "argv", ["prev_release_tag.py", "0.3.13"])
    exit_code = None
    try:
        mod.main()
    except SystemExit as e:
        exit_code = e.code
    out = capsys.readouterr().out
    assert out == ""
    assert exit_code in (None, 0)


def test_main_uses_argv_not_top_of_stdin_list(mod, monkeypatch, capsys):
    # Regression guard (test-critic round 1, gap 3): the two end-to-end
    # tests above both put the new version at the top of the stdin tag
    # list, so a main() that ignores sys.argv entirely and just prints the
    # second-largest stdin tag would still pass both. Here the argv version
    # sits in the middle of the list -- tags exist both above (v0.4.0,
    # v0.5.0, simulating tags landing after the release being computed) and
    # below (v0.1.0, v0.3.12) it -- so only an implementation that actually
    # reads and uses sys.argv[1] computes the right answer.
    stdin_text = "v0.1.0\nv0.3.12\nv0.3.13\nv0.4.0\nv0.5.0\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin_text))
    monkeypatch.setattr(sys, "argv", ["prev_release_tag.py", "0.3.13"])
    exit_code = None
    try:
        mod.main()
    except SystemExit as e:
        exit_code = e.code
    out = capsys.readouterr().out
    assert out == "v0.3.12\n"
    assert exit_code in (None, 0)


# --- additional edge-case coverage: guards against lexical comparison ------


def test_two_digit_component_beats_single_digit_numerically(mod):
    tags = ["v0.3.9", "v0.3.10"]
    assert mod.previous_tag("0.3.11", tags) == "v0.3.10"
