"""Ticket #135 -- structural guard that docs/teardown-phase-contract.md
stays in sync with ``teardown._TEARDOWN_PHASES``.

If a phase is added, removed, or reordered in ``core/teardown.py`` without
updating the doc, this test fails loudly rather than letting the doc go
stale.
"""

from __future__ import annotations

import re
from pathlib import Path

from lib_python_worktree.core import teardown

_DOC_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "teardown-phase-contract.md"
)

_PHASE_LINE_RE = re.compile(r"^\d+\.\s+`(_phase_\w+)`", re.MULTILINE)


def _phase_names_from_doc() -> list:
    text = _DOC_PATH.read_text(encoding="utf-8")
    return _PHASE_LINE_RE.findall(text)


def test_doc_exists():
    assert _DOC_PATH.is_file(), f"expected {_DOC_PATH} to exist"


def test_doc_lists_every_phase_in_order():
    doc_names = _phase_names_from_doc()
    actual_names = [f.__name__ for f in teardown._TEARDOWN_PHASES]
    assert doc_names == actual_names, (
        f"docs/teardown-phase-contract.md's numbered phase list has "
        f"drifted from teardown._TEARDOWN_PHASES.\n"
        f"doc:    {doc_names}\n"
        f"actual: {actual_names}"
    )


def test_doc_phase_count_matches():
    assert len(_phase_names_from_doc()) == len(teardown._TEARDOWN_PHASES)
