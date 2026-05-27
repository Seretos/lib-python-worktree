"""File-IO + error formatting for the worktree contract.

The contract lives at ``<repo-root>/.seretos/worktree-setup.yml`` —
aligning with the ``.seretos/`` convention used across the Seretos
plugin family. A missing file is treated as an implicit
``isolation: none`` contract with no setup/teardown/ports. Callers
who want hard errors can check ``Path.exists()`` themselves before
calling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import ValidationError

from .schema import WorktreeContract


# Relative path (NOT just a filename) below the worktree root. Callers
# compose this with ``Path(worktree_root) / CONTRACT_FILENAME`` — the
# ``/`` operator on Path handles the embedded directory segment.
CONTRACT_FILENAME = ".seretos/worktree-setup.yml"


class ContractError(Exception):
    """Base error for contract loading failures."""


class ContractValidationError(ContractError):
    """Raised when the YAML parses but does not satisfy the schema.

    ``path`` is the source file (or ``"<string>"`` for ``load_text``).
    ``errors`` is the list of dicts from pydantic's ``ValidationError.errors()``,
    each carrying a ``loc`` tuple that doubles as a JSON-pointer-ish path.
    """

    def __init__(self, path: str, errors: list[dict[str, Any]]) -> None:
        self.path = path
        self.errors = errors
        super().__init__(self._format(path, errors))

    @staticmethod
    def _format(path: str, errors: list[dict[str, Any]]) -> str:
        lines = [f"Invalid contract at {path}:"]
        for err in errors:
            loc = ".".join(str(x) for x in err.get("loc", ())) or "<root>"
            msg = err.get("msg", "validation error")
            lines.append(f"  - {loc}: {msg}")
        return "\n".join(lines)


def _implicit_none_contract() -> WorktreeContract:
    return WorktreeContract(version=1, isolation="none")


def load(path: Union[str, Path]) -> WorktreeContract:
    """Load and validate a worktree contract from ``path`` (typically
    ``<repo-root>/.seretos/worktree-setup.yml``).

    Missing files become an implicit ``isolation: none`` contract.
    Parse errors become ``ContractError``; schema errors become
    ``ContractValidationError`` with a structured path-prefixed message.
    """

    p = Path(path)
    if not p.exists():
        return _implicit_none_contract()
    try:
        raw_text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"could not read contract {p}: {exc}") from exc
    return load_text(raw_text, source=str(p))


def load_text(text: str, *, source: str = "<string>") -> WorktreeContract:
    """Validate raw YAML text. ``source`` is used in error messages only."""

    try:
        data: Optional[Any] = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ContractError(f"YAML parse error in {source}: {exc}") from exc

    if data is None:
        # Empty file. Treat as implicit isolation: none (D2 extension).
        return _implicit_none_contract()

    if not isinstance(data, dict):
        raise ContractError(
            f"Contract root must be a mapping in {source}, got {type(data).__name__}"
        )

    try:
        return WorktreeContract.model_validate(data)
    except ValidationError as exc:
        raise ContractValidationError(source, exc.errors()) from exc


__all__ = (
    "CONTRACT_FILENAME",
    "ContractError",
    "ContractValidationError",
    "load",
    "load_text",
)
