"""Unit tests for lib_python_worktree.core.plugin_seed.

All tests run entirely in tmp_path — no git, no state store.
Covers every branch of seed_plugin_registry including atomic-write
litter cleanup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lib_python_worktree.core.plugin_seed import seed_plugin_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(config_dir: Path, entries: object) -> Path:
    """Write *entries* as JSON to the registry file and return its path."""
    plugins_dir = config_dir / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    registry = plugins_dir / "installed_plugins.json"
    registry.write_text(json.dumps(entries), encoding="utf-8")
    return registry


def _read_registry(config_dir: Path) -> object:
    registry = config_dir / "plugins" / "installed_plugins.json"
    return json.loads(registry.read_text(encoding="utf-8"))


def _make_entry(
    project_path: str,
    install_path: str = "/path/to/plugin",
    scope: str = "project",
    version: str = "1.0.0",
) -> dict:
    return {
        "scope": scope,
        "projectPath": project_path,
        "installPath": install_path,
        "version": version,
    }


# ---------------------------------------------------------------------------
# Guard cases — silent return without writing
# ---------------------------------------------------------------------------

def test_registry_absent_returns_silently(tmp_path: Path):
    """When the registry file does not exist, the function returns without error."""
    config_dir = tmp_path / "claude"
    # Do NOT create the plugins dir or file.
    seed_plugin_registry("/repo", "/wt", config_dir=config_dir)
    # No file should have been created.
    assert not (config_dir / "plugins" / "installed_plugins.json").exists()


def test_malformed_json_returns_silently(tmp_path: Path):
    """Malformed JSON in the registry file is silently ignored."""
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    registry = plugins_dir / "installed_plugins.json"
    registry.write_text("{ this is not valid json", encoding="utf-8")

    config_dir = tmp_path
    seed_plugin_registry("/repo", "/wt", config_dir=config_dir)
    # File must be unchanged.
    assert registry.read_text(encoding="utf-8") == "{ this is not valid json"


def test_non_list_top_level_returns_silently(tmp_path: Path):
    """If the top-level JSON value is not a list, return silently."""
    _make_registry(tmp_path, {"scope": "project", "projectPath": "/repo"})
    seed_plugin_registry("/repo", "/wt", config_dir=tmp_path)
    # File stays as a dict — no modification.
    assert isinstance(_read_registry(tmp_path), dict)


def test_no_matching_entries_returns_without_writing(tmp_path: Path):
    """When no entries match the source repo_path, the file is not rewritten."""
    entry = _make_entry("/other/repo")
    _make_registry(tmp_path, [entry])
    registry_path = tmp_path / "plugins" / "installed_plugins.json"
    mtime_before = registry_path.stat().st_mtime

    seed_plugin_registry("/repo", "/wt", config_dir=tmp_path)

    # File not modified (same mtime and content).
    assert registry_path.stat().st_mtime == mtime_before


def test_scope_not_project_is_ignored(tmp_path: Path):
    """Entries with scope != 'project' must not be cloned."""
    entry = _make_entry("/repo", scope="global")
    _make_registry(tmp_path, [entry])
    registry_path = tmp_path / "plugins" / "installed_plugins.json"
    mtime_before = registry_path.stat().st_mtime

    seed_plugin_registry("/repo", "/wt", config_dir=tmp_path)

    assert registry_path.stat().st_mtime == mtime_before


# ---------------------------------------------------------------------------
# Happy path — entries are cloned
# ---------------------------------------------------------------------------

def test_matching_entry_cloned_with_native_path(tmp_path: Path):
    """A matching project-scoped entry is cloned with the native-OS worktree path."""
    repo_path = "/home/user/repo"
    wt_path_posix = "/home/user/store/wt1"
    entry = _make_entry(repo_path, install_path="/path/to/plugin", version="2.3.4")
    _make_registry(tmp_path, [entry])

    seed_plugin_registry(repo_path, wt_path_posix, config_dir=tmp_path)

    result = _read_registry(tmp_path)
    assert isinstance(result, list)
    assert len(result) == 2  # original + clone

    cloned = result[1]
    assert cloned["projectPath"] == str(Path(wt_path_posix))
    assert cloned["installPath"] == "/path/to/plugin"
    assert cloned["version"] == "2.3.4"
    assert cloned["scope"] == "project"


def test_install_path_and_version_preserved(tmp_path: Path):
    """installPath and version from the original entry are preserved in the clone."""
    repo_path = "/repo"
    entry = _make_entry(repo_path, install_path="/custom/install", version="9.9.9")
    _make_registry(tmp_path, [entry])

    seed_plugin_registry(repo_path, "/wt", config_dir=tmp_path)

    result = _read_registry(tmp_path)
    cloned = result[-1]
    assert cloned["installPath"] == "/custom/install"
    assert cloned["version"] == "9.9.9"


def test_multiple_matching_entries_all_cloned(tmp_path: Path):
    """All project-scoped entries for the repo are cloned, not just the first."""
    repo_path = "/repo"
    entries = [
        _make_entry(repo_path, install_path="/plugin/a", version="1.0.0"),
        _make_entry(repo_path, install_path="/plugin/b", version="2.0.0"),
        _make_entry("/other/repo", install_path="/plugin/c"),
    ]
    _make_registry(tmp_path, entries)

    seed_plugin_registry(repo_path, "/wt", config_dir=tmp_path)

    result = _read_registry(tmp_path)
    assert len(result) == 5  # 3 original + 2 clones
    cloned = [e for e in result if e["projectPath"] == str(Path("/wt"))]
    assert len(cloned) == 2
    install_paths = {e["installPath"] for e in cloned}
    assert install_paths == {"/plugin/a", "/plugin/b"}


def test_non_matching_project_path_not_cloned(tmp_path: Path):
    """Entries whose projectPath does not match repo_path are left untouched."""
    entries = [
        _make_entry("/other/repo", install_path="/plugin/x"),
    ]
    _make_registry(tmp_path, entries)

    seed_plugin_registry("/repo", "/wt", config_dir=tmp_path)

    result = _read_registry(tmp_path)
    assert len(result) == 1
    assert result[0]["projectPath"] == "/other/repo"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_already_seeded_entry_skipped(tmp_path: Path):
    """Re-running with the same worktree path does not add a duplicate entry."""
    repo_path = "/repo"
    wt_path = str(Path("/wt"))
    entries = [
        _make_entry(repo_path, install_path="/plugin/a"),
        _make_entry(wt_path, install_path="/plugin/a"),  # already present
    ]
    _make_registry(tmp_path, entries)

    seed_plugin_registry(repo_path, wt_path, config_dir=tmp_path)

    result = _read_registry(tmp_path)
    # Count entries with dest projectPath — should still be 1.
    dest_entries = [e for e in result if e.get("projectPath") == wt_path]
    assert len(dest_entries) == 1


def test_idempotent_second_call_no_extra_entries(tmp_path: Path):
    """Calling seed_plugin_registry twice produces the same result as calling once."""
    repo_path = "/repo"
    wt_path = "/wt"
    entry = _make_entry(repo_path, install_path="/plugin/a")
    _make_registry(tmp_path, [entry])

    seed_plugin_registry(repo_path, wt_path, config_dir=tmp_path)
    seed_plugin_registry(repo_path, wt_path, config_dir=tmp_path)

    result = _read_registry(tmp_path)
    dest_entries = [e for e in result if e.get("projectPath") == str(Path(wt_path))]
    assert len(dest_entries) == 1


# ---------------------------------------------------------------------------
# CLAUDE_CONFIG_DIR env-var override
# ---------------------------------------------------------------------------

def test_claude_config_dir_env_override(tmp_path: Path, monkeypatch):
    """CLAUDE_CONFIG_DIR env var is used when config_dir is not supplied explicitly."""
    config_dir = tmp_path / "my_claude"
    repo_path = "/repo"
    entry = _make_entry(repo_path, install_path="/plugin/env")
    _make_registry(config_dir, [entry])

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    # Call WITHOUT passing config_dir — must pick it up from the env var.
    seed_plugin_registry(repo_path, "/wt")

    result = _read_registry(config_dir)
    dest = [e for e in result if e.get("projectPath") == str(Path("/wt"))]
    assert len(dest) == 1
    assert dest[0]["installPath"] == "/plugin/env"


# ---------------------------------------------------------------------------
# Atomic write — no temp-file litter after a successful call
# ---------------------------------------------------------------------------

def test_atomic_write_leaves_no_temp_files(tmp_path: Path):
    """After a successful seed, no *.tmp files are left in the plugins directory."""
    repo_path = "/repo"
    entry = _make_entry(repo_path, install_path="/plugin/a")
    _make_registry(tmp_path, [entry])

    seed_plugin_registry(repo_path, "/wt", config_dir=tmp_path)

    plugins_dir = tmp_path / "plugins"
    tmp_files = list(plugins_dir.glob("*.tmp"))
    assert tmp_files == [], f"Unexpected temp files left behind: {tmp_files}"


def test_atomic_write_leaves_no_litter_on_write_error(tmp_path: Path, monkeypatch):
    """If the write step raises, the temp file is removed and the error propagates."""
    import tempfile as _tempfile

    repo_path = "/repo"
    entry = _make_entry(repo_path, install_path="/plugin/a")
    _make_registry(tmp_path, [entry])

    original_mkstemp = _tempfile.mkstemp
    created_tmp: list[str] = []

    def _patched_mkstemp(**kwargs):
        fd, path = original_mkstemp(**kwargs)
        created_tmp.append(path)
        return fd, path

    monkeypatch.setattr(_tempfile, "mkstemp", _patched_mkstemp)

    # Patch os.replace to raise so the write "fails".
    monkeypatch.setattr(os, "replace", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("simulated replace failure")))

    with pytest.raises(OSError, match="simulated replace failure"):
        seed_plugin_registry(repo_path, "/wt", config_dir=tmp_path)

    # The temp file must have been cleaned up.
    for p in created_tmp:
        assert not os.path.exists(p), f"Temp file not cleaned up: {p}"
