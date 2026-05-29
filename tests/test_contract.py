"""Unit tests for worktree-contract parsing + validation (W3).

Contract lives at ``<repo-root>/.seretos/worktree-setup.yml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lib_python_worktree.contract import (
    ContractError,
    ContractValidationError,
    WorktreeContract,
    load,
    load_text,
)


def test_minimal_isolation_none():
    c = load_text("version: 1\nisolation: none\n")
    assert isinstance(c, WorktreeContract)
    assert c.isolation == "none"
    assert c.setup == []
    assert c.teardown == []
    assert c.ports == []


def test_full_webapp_style_contract():
    text = """
version: 1
isolation: full
setup:
  - name: install deps
    run: pnpm install
  - run: pnpm prisma migrate dev
teardown:
  - run: docker compose down
ports:
  - name: app
  - name: db
  - name: chrome
"""
    c = load_text(text)
    assert c.isolation == "full"
    assert len(c.setup) == 2
    assert c.setup[0].name == "install deps"
    assert c.setup[1].name is None
    assert [p.name for p in c.ports] == ["app", "db", "chrome"]


def test_unity_isolation_none_is_bare():
    c = load_text("version: 1\nisolation: none\n")
    assert c.isolation == "none"


def test_extra_field_at_root_rejected():
    with pytest.raises(ContractValidationError) as exc_info:
        load_text("version: 1\nisolation: full\nbogus: yes\n")
    assert any("bogus" in e["loc"] for e in exc_info.value.errors)


def test_extra_field_in_step_rejected():
    with pytest.raises(ContractValidationError):
        load_text(
            "version: 1\nisolation: full\nsetup:\n  - run: 'x'\n    surprise: 1\n"
        )


def test_wrong_version_rejected():
    with pytest.raises(ContractValidationError):
        load_text("version: 2\nisolation: full\n")


def test_wrong_isolation_rejected():
    with pytest.raises(ContractValidationError):
        load_text("version: 1\nisolation: weird\n")


def test_duplicate_port_name_rejected():
    text = """
version: 1
isolation: full
ports:
  - name: app
  - name: db
  - name: app
"""
    with pytest.raises(ContractValidationError) as exc_info:
        load_text(text)
    assert "duplicate" in str(exc_info.value).lower()


def test_port_name_invalid_format_rejected():
    with pytest.raises(ContractValidationError):
        load_text("version: 1\nisolation: full\nports:\n  - name: App\n")
    with pytest.raises(ContractValidationError):
        load_text("version: 1\nisolation: full\nports:\n  - name: '1bad'\n")


def test_isolation_none_forbids_setup():
    with pytest.raises(ContractValidationError) as exc_info:
        load_text(
            "version: 1\nisolation: none\nsetup:\n  - run: echo hi\n"
        )
    assert "isolation: none" in str(exc_info.value)


def test_isolation_none_forbids_ports():
    with pytest.raises(ContractValidationError):
        load_text(
            "version: 1\nisolation: none\nports:\n  - name: app\n"
        )


def test_load_missing_file_returns_implicit_none(tmp_path: Path):
    c = load(tmp_path / "nope.yml")
    assert c.isolation == "none"
    assert c.setup == []


def test_load_existing_file(tmp_path: Path):
    # Exercise the canonical layout: contract at <repo>/.seretos/worktree-setup.yml.
    p = tmp_path / ".seretos" / "worktree-setup.yml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("version: 1\nisolation: full\n", encoding="utf-8")
    c = load(p)
    assert c.isolation == "full"


def test_contract_filename_is_seretos_relative():
    """CONTRACT_FILENAME must point under `.seretos/` so callers
    composing `<repo>/CONTRACT_FILENAME` land in the right place."""
    from lib_python_worktree.contract import CONTRACT_FILENAME
    assert CONTRACT_FILENAME == ".seretos/worktree-setup.yml"


def test_invalid_yaml_raises_contract_error():
    with pytest.raises(ContractError) as exc_info:
        load_text("version: 1\n  bad: indent: here\n")
    # Must NOT be a ValidationError — wrong parse, not a schema fault.
    assert not isinstance(exc_info.value, ContractValidationError)


def test_root_must_be_mapping():
    with pytest.raises(ContractError):
        load_text("- 1\n- 2\n")


def test_empty_file_is_implicit_none():
    c = load_text("")
    assert c.isolation == "none"


def test_shell_override_accepted_per_step():
    c = load_text(
        "version: 1\nisolation: full\nsetup:\n"
        "  - run: 'pwsh -c echo hi'\n    shell: pwsh\n"
    )
    assert c.setup[0].shell == "pwsh"


def test_shell_override_invalid_value_rejected():
    with pytest.raises(ContractValidationError):
        load_text(
            "version: 1\nisolation: full\nsetup:\n"
            "  - run: 'x'\n    shell: zsh\n"
        )


def test_load_unreadable_file_raises_contract_error(
    tmp_path: Path, monkeypatch
):
    """``load()`` on a file that exists but raises OSError on read must
    raise ``ContractError`` (not the subtype ``ContractValidationError``).
    """
    p = tmp_path / "unreadable.yml"
    p.write_text("", encoding="utf-8")

    # Monkeypatch Path.read_text to simulate a permission / IO error.
    original_read_text = Path.read_text

    def _failing_read_text(self, *args, **kwargs):
        if self == p:
            raise OSError("permission denied (simulated)")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _failing_read_text)

    with pytest.raises(ContractError) as exc_info:
        load(p)
    # Must be a plain ContractError, not a validation subtype.
    assert type(exc_info.value) is ContractError
