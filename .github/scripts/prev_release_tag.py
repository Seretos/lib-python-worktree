#!/usr/bin/env python3
"""Compute the previous released tag by real semver precedence (ticket #156).

Standalone helper invoked from `release.yml`'s "Create GitHub Release" step
(not part of the installed `lib_python_worktree` package, so it is not
importable as a module -- it is invoked as a script). Each release branch is
force-pushed fresh off `main` and the stamp commit is never merged back, so
no prior `vX.Y.Z` tag is ever an ancestor of the newly created tag; `gh
release create --generate-notes`'s ancestor-walk auto-detect therefore finds
nothing and generates notes from the beginning of history. This script
computes the actual previous released tag explicitly so the caller can pass
it as `--notes-start-tag`.

This is a pragmatic ordering helper, not an RFC-SemVer-spec validator: its
tag parser is deliberately tolerant of malformed input (e.g. leading zeros
in numeric identifiers, empty dot-segments) rather than rejecting it, because
it only ever needs to match this repo's own tag-creation regex
(`^[0-9]+\\.[0-9]+\\.[0-9]+(-[0-9A-Za-z.-]+)?$` in `release.yml`'s "Validate
version is semver" step), which is equally loose. A `+buildmetadata` suffix
is NOT supported by this parser -- like the caller's own validation regex,
it has no `+` handling, so a tag carrying one fails to match `_SEMVER_RE`
and is silently dropped as unparseable, the same as any other malformed or
old-format tag. This is inert in practice: `release.yml`'s validation step
would reject a version containing build metadata before such a tag could
ever be created here.

Usage: `git tag --list 'v*' | python3 prev_release_tag.py <version>`, where
`<version>` is the new release's version (no leading "v"). Prints the
previous tag (with its "v" prefix) to stdout, or nothing if there is no tag
with strictly lower semver precedence than `<version>`.
"""

from __future__ import annotations

import re
import sys
from typing import Iterable, Optional, Tuple

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?$")

# (major, minor, patch, prerelease-or-None)
_ParsedVersion = Tuple[int, int, int, Optional[str]]


def _parse_semver(value: str) -> Optional[_ParsedVersion]:
    """Parse a tag or bare version string into its semver components.

    Accepts an optional leading "v" (tags carry it, the CLI version argument
    does not). Returns None for anything that isn't a well-formed
    MAJOR.MINOR.PATCH[-PRERELEASE] string -- callers treat that as "ignore
    this entry" rather than a fatal error (see `previous_tag`'s docstring for
    why that tolerance is deliberate).
    """
    s = value[1:] if value.startswith("v") else value
    m = _SEMVER_RE.match(s)
    if not m:
        return None
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (major, minor, patch, m.group(4))


def _precedence_key(parsed: _ParsedVersion):
    """Build a tuple that sorts by real semver precedence.

    Real semver precedence, not `sort -V` and not lexical string comparison:
    numeric MAJOR.MINOR.PATCH triple compared numerically; a release
    outranks a prerelease of the same triple; prerelease identifiers are
    compared dot-segment by dot-segment, numeric identifiers always
    outranked by alphanumeric ones, and a longer identifier list outranks a
    shorter one whose shared prefix is equal (Python tuple comparison
    already implements that last rule for free).
    """
    major, minor, patch, prerelease = parsed
    if prerelease is None:
        return (major, minor, patch, 1, ())
    parts = tuple(
        (0, int(p)) if p.isdigit() else (1, p) for p in prerelease.split(".")
    )
    return (major, minor, patch, 0, parts)


def previous_tag(version: str, tags: Iterable[str]) -> Optional[str]:
    """Return the greatest `v*` tag with precedence strictly below `version`.

    `version` is the new release's bare version string (no leading "v");
    `tags` is any iterable of tag strings such as `git tag --list 'v*'`
    output, in any order. Tags that don't parse as semver (an old-format
    tag, a stray non-tag line, etc.) are ignored rather than raised on --
    tolerated deliberately, so one unrelated malformed tag in the
    repository's history can't take down the release step; the caller's own
    "Validate version is semver" step already guards the *new* version
    before this script is ever invoked. Returns None if there is no such
    tag (first-ever release, or every candidate tag is >= `version`).
    """
    target = _parse_semver(version)
    if target is None:
        return None
    target_key = _precedence_key(target)

    best_tag: Optional[str] = None
    best_key = None
    for tag in tags:
        parsed = _parse_semver(tag)
        if parsed is None:
            continue
        key = _precedence_key(parsed)
        if key >= target_key:
            continue
        if best_key is None or key > best_key:
            best_key = key
            best_tag = tag
    return best_tag


def main() -> None:
    version = sys.argv[1]
    tags = [line.strip() for line in sys.stdin if line.strip()]
    tag = previous_tag(version, tags)
    if tag:
        print(tag)


if __name__ == "__main__":
    main()
