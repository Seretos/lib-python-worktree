"""Pydantic v2 schema for the worktree contract (lives at
``<repo-root>/.seretos/worktree-setup.yml``).

D1 (Option B): PyYAML for parsing + pydantic v2 for validation.
"""

from __future__ import annotations

import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Isolation = Literal["full", "partial", "none"]

_PORT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class _StrictModel(BaseModel):
    """Base model with `extra="forbid"` so unknown fields raise."""

    model_config = ConfigDict(extra="forbid")


class Step(_StrictModel):
    """A single setup/teardown step.

    ``run`` is the shell command. ``name`` is a human label that the setup
    runner (W5) uses for log filenames. ``shell`` is an optional override
    that W5 honors (D1 of W5, Option B); the schema accepts it here so the
    contract doesn't have to grow later.
    """

    run: str = Field(..., min_length=1)
    name: Optional[str] = None
    shell: Optional[Literal["bash", "pwsh", "sh", "powershell"]] = None


class PortSlot(_StrictModel):
    """A named port slot. Allocation happens in W4."""

    name: str

    @field_validator("name")
    @classmethod
    def _name_must_be_slug(cls, v: str) -> str:
        if not _PORT_NAME_RE.match(v):
            raise ValueError(
                "port slot name must match ^[a-z][a-z0-9_]{0,31}$ "
                f"(got: {v!r})"
            )
        return v


class WorktreeContract(_StrictModel):
    """Top-level shape of `.seretos/worktree-setup.yml`.

    - `version`: load-bearing for future migrations; currently must be 1.
    - `isolation`: full/partial/none; `none` forbids setup/teardown/ports.
    - `setup` / `teardown`: ordered step lists; W5/W8 execute these.
    - `ports`: named slots W4 will allocate against a global range.
    """

    version: Literal[1]
    isolation: Isolation
    setup: List[Step] = Field(default_factory=list)
    teardown: List[Step] = Field(default_factory=list)
    ports: List[PortSlot] = Field(default_factory=list)

    @model_validator(mode="after")
    def _isolation_none_forbids_lists(self) -> "WorktreeContract":
        if self.isolation == "none":
            offenders = []
            if self.setup:
                offenders.append("setup")
            if self.teardown:
                offenders.append("teardown")
            if self.ports:
                offenders.append("ports")
            if offenders:
                raise ValueError(
                    f"isolation: none forbids fields: {', '.join(offenders)}"
                )
        return self

    @model_validator(mode="after")
    def _ports_must_have_unique_names(self) -> "WorktreeContract":
        seen: set[str] = set()
        duplicates: list[str] = []
        for slot in self.ports:
            if slot.name in seen:
                duplicates.append(slot.name)
            seen.add(slot.name)
        if duplicates:
            raise ValueError(
                f"duplicate port slot names: {', '.join(sorted(set(duplicates)))}"
            )
        return self


__all__ = (
    "Isolation",
    "PortSlot",
    "Step",
    "WorktreeContract",
)
