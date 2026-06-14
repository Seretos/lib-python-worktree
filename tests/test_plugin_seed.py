"""Unit tests for lib_python_worktree.core.plugin_seed.

All tests run entirely in tmp_path — no git, no state store.
Covers every branch of seed_plugin_registry including atomic-write
litter cleanup.

The real Claude plugin registry uses Schema v2::

    {"version": 2, "plugins": {"<name>@<marketplace>": [<entry>, ...]}}

All fixtures use this schema.  Guard-case tests for unknown-version and
bare-list top-levels are also included.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from lib_python_worktree.core.plugin_seed import seed_plugin_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_v2_registry(config_dir: Path, plugins: dict) -> Path:
    """Write a Schema v2 registry object and return its path."""
    plugins_dir = config_dir / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    registry = plugins_dir / "installed_plugins.json"
    registry.write_text(
        json.dumps({"version": 2, "plugins": plugins}), encoding="utf-8"
    )
    return registry


def _read_registry(config_dir: Path) -> object:
    """Return the full parsed JSON object from the registry file."""
    registry = config_dir / "plugins" / "installed_plugins.json"
    return json.loads(registry.read_text(encoding="utf-8"))


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

    seed_plugin_registry("/repo", "/wt", config_dir=tmp_path)
    # File must be unchanged.
    assert registry.read_text(encoding="utf-8") == "{ this is not valid json"


def test_unknown_version_returns_silently(tmp_path: Path):
    """A registry with version != 2 returns silently without writing."""
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    registry = plugins_dir / "installed_plugins.json"
    registry.write_text(
        json.dumps({"version": 3, "plugins": {}}), encoding="utf-8"
    )
    mtime_before = registry.stat().st_mtime

    seed_plugin_registry("/repo", "/wt", config_dir=tmp_path)

    assert registry.stat().st_mtime == mtime_before
    result = _read_registry(tmp_path)
    assert result == {"version": 3, "plugins": {}}


def test_bare_list_returns_silently(tmp_path: Path):
    """If the top-level JSON value is a bare list, return silently."""
    entry = _make_entry("/repo")
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    registry = plugins_dir / "installed_plugins.json"
    registry.write_text(json.dumps([entry]), encoding="utf-8")
    mtime_before = registry.stat().st_mtime

    seed_plugin_registry("/repo", "/wt", config_dir=tmp_path)

    assert registry.stat().st_mtime == mtime_before
    assert isinstance(_read_registry(tmp_path), list)


# ---------------------------------------------------------------------------
# Primary regression — v2 schema seeded correctly
# ---------------------------------------------------------------------------

def test_v2_schema_seeded_correctly(tmp_path: Path):
    """Primary regression: a matching project-scoped entry in v2 registry is cloned.

    The per-name list grows by one entry with projectPath equal to the
    native-OS form of worktree_path.
    """
    repo_path = "/home/user/repo"
    wt_path = "/home/user/store/wt1"
    entry = _make_entry(repo_path, install_path="/path/to/plugin", version="2.3.4")
    _make_v2_registry(tmp_path, {"plugin-a@marketplace": [entry]})

    seed_plugin_registry(repo_path, wt_path, config_dir=tmp_path)

    result = _read_registry(tmp_path)
    assert result["version"] == 2
    plugin_list = result["plugins"]["plugin-a@marketplace"]
    assert len(plugin_list) == 2  # original + clone

    cloned = plugin_list[1]
    assert cloned["projectPath"] == str(Path(wt_path))
    assert cloned["installPath"] == "/path/to/plugin"
    assert cloned["version"] == "2.3.4"
    assert cloned["scope"] == "project"


# ---------------------------------------------------------------------------
# Path normalisation — Windows backslash vs POSIX slash
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows path normalisation (backslash vs forward-slash via os.path.normcase) — POSIX platforms cannot reproduce",
)
def test_path_normalisation_windows_backslash(tmp_path: Path):
    """Registry stores backslash paths; caller passes POSIX slash — entry is cloned.

    The backslash string is constructed directly so this test runs on all
    platforms without relying on the OS path separator.
    """
    # Backslash path as it would be stored in the registry on Windows.
    backslash_repo = "C:\\Users\\foo\\repo"
    posix_repo = "C:/Users/foo/repo"
    wt_path = "C:/Users/foo/store/wt1"

    entry = _make_entry(backslash_repo, install_path="/plugin/x")
    _make_v2_registry(tmp_path, {"plugin-b@marketplace": [entry]})

    # Caller passes the POSIX form; the function must normalise both sides.
    seed_plugin_registry(posix_repo, wt_path, config_dir=tmp_path)

    result = _read_registry(tmp_path)
    plugin_list = result["plugins"]["plugin-b@marketplace"]
    assert len(plugin_list) == 2, (
        "Entry should have been cloned despite backslash vs forward-slash difference"
    )
    assert plugin_list[1]["projectPath"] == str(Path(wt_path))


# ---------------------------------------------------------------------------
# Guard cases — no write when nothing matches
# ---------------------------------------------------------------------------

def test_v2_no_matching_project_path(tmp_path: Path):
    """Only entry has a different projectPath; file is not rewritten."""
    entry = _make_entry("/other/repo")
    _make_v2_registry(tmp_path, {"plugin-a@marketplace": [entry]})
    registry_path = tmp_path / "plugins" / "installed_plugins.json"
    mtime_before = registry_path.stat().st_mtime

    seed_plugin_registry("/repo", "/wt", config_dir=tmp_path)

    assert registry_path.stat().st_mtime == mtime_before


def test_v2_scope_not_project_ignored(tmp_path: Path):
    """Entries with scope != 'project' must not be cloned."""
    entry = _make_entry("/repo", scope="global")
    _make_v2_registry(tmp_path, {"plugin-a@marketplace": [entry]})
    registry_path = tmp_path / "plugins" / "installed_plugins.json"
    mtime_before = registry_path.stat().st_mtime

    seed_plugin_registry("/repo", "/wt", config_dir=tmp_path)

    assert registry_path.stat().st_mtime == mtime_before


# ---------------------------------------------------------------------------
# Multiple plugin names
# ---------------------------------------------------------------------------

def test_v2_multiple_plugin_names_all_seeded(tmp_path: Path):
    """Two distinct plugin names each with a matching entry both get a clone."""
    repo_path = "/repo"
    entry_a = _make_entry(repo_path, install_path="/plugin/a", version="1.0.0")
    entry_b = _make_entry(repo_path, install_path="/plugin/b", version="2.0.0")
    entry_other = _make_entry("/other/repo", install_path="/plugin/c")

    _make_v2_registry(tmp_path, {
        "plugin-a@marketplace": [entry_a],
        "plugin-b@marketplace": [entry_b],
        "plugin-c@marketplace": [entry_other],
    })

    seed_plugin_registry(repo_path, "/wt", config_dir=tmp_path)

    result = _read_registry(tmp_path)
    dest_path = str(Path("/wt"))

    list_a = result["plugins"]["plugin-a@marketplace"]
    list_b = result["plugins"]["plugin-b@marketplace"]
    list_c = result["plugins"]["plugin-c@marketplace"]

    assert len(list_a) == 2
    assert list_a[1]["projectPath"] == dest_path
    assert list_a[1]["installPath"] == "/plugin/a"

    assert len(list_b) == 2
    assert list_b[1]["projectPath"] == dest_path
    assert list_b[1]["installPath"] == "/plugin/b"

    # Non-matching plugin unchanged.
    assert len(list_c) == 1
    assert list_c[0]["projectPath"] == "/other/repo"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_v2_idempotency(tmp_path: Path):
    """Seeding twice does not duplicate the cloned entry."""
    repo_path = "/repo"
    wt_path = "/wt"
    entry = _make_entry(repo_path, install_path="/plugin/a")
    _make_v2_registry(tmp_path, {"plugin-a@marketplace": [entry]})

    seed_plugin_registry(repo_path, wt_path, config_dir=tmp_path)
    seed_plugin_registry(repo_path, wt_path, config_dir=tmp_path)

    result = _read_registry(tmp_path)
    plugin_list = result["plugins"]["plugin-a@marketplace"]
    dest_path = str(Path(wt_path))
    dest_entries = [e for e in plugin_list if e.get("projectPath") == dest_path]
    assert len(dest_entries) == 1


def test_v2_already_seeded_entry_skipped(tmp_path: Path):
    """Registry already contains a clone for the worktree path — no duplicate added."""
    repo_path = "/repo"
    wt_path = str(Path("/wt"))
    original = _make_entry(repo_path, install_path="/plugin/a")
    already_cloned = _make_entry(wt_path, install_path="/plugin/a")
    _make_v2_registry(tmp_path, {"plugin-a@marketplace": [original, already_cloned]})

    seed_plugin_registry(repo_path, wt_path, config_dir=tmp_path)

    result = _read_registry(tmp_path)
    plugin_list = result["plugins"]["plugin-a@marketplace"]
    dest_entries = [e for e in plugin_list if e.get("projectPath") == wt_path]
    assert len(dest_entries) == 1


# ---------------------------------------------------------------------------
# installPath and version preservation
# ---------------------------------------------------------------------------

def test_install_path_and_version_preserved(tmp_path: Path):
    """installPath and version from the original entry are preserved in the clone."""
    repo_path = "/repo"
    entry = _make_entry(repo_path, install_path="/custom/install", version="9.9.9")
    _make_v2_registry(tmp_path, {"plugin-a@marketplace": [entry]})

    seed_plugin_registry(repo_path, "/wt", config_dir=tmp_path)

    result = _read_registry(tmp_path)
    plugin_list = result["plugins"]["plugin-a@marketplace"]
    cloned = plugin_list[-1]
    assert cloned["installPath"] == "/custom/install"
    assert cloned["version"] == "9.9.9"


# ---------------------------------------------------------------------------
# CLAUDE_CONFIG_DIR env-var override
# ---------------------------------------------------------------------------

def test_claude_config_dir_env_override(tmp_path: Path, monkeypatch):
    """CLAUDE_CONFIG_DIR env var is used when config_dir is not supplied explicitly."""
    config_dir = tmp_path / "my_claude"
    repo_path = "/repo"
    entry = _make_entry(repo_path, install_path="/plugin/env")
    _make_v2_registry(config_dir, {"plugin-a@marketplace": [entry]})

    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))

    # Call WITHOUT passing config_dir — must pick it up from the env var.
    seed_plugin_registry(repo_path, "/wt")

    result = _read_registry(config_dir)
    plugin_list = result["plugins"]["plugin-a@marketplace"]
    dest = [e for e in plugin_list if e.get("projectPath") == str(Path("/wt"))]
    assert len(dest) == 1
    assert dest[0]["installPath"] == "/plugin/env"


# ---------------------------------------------------------------------------
# Atomic write — no temp-file litter after a successful call
# ---------------------------------------------------------------------------

def test_atomic_write_leaves_no_temp_files(tmp_path: Path):
    """After a successful seed, no *.tmp files are left in the plugins directory."""
    repo_path = "/repo"
    entry = _make_entry(repo_path, install_path="/plugin/a")
    _make_v2_registry(tmp_path, {"plugin-a@marketplace": [entry]})

    seed_plugin_registry(repo_path, "/wt", config_dir=tmp_path)

    plugins_dir = tmp_path / "plugins"
    tmp_files = list(plugins_dir.glob("*.tmp"))
    assert tmp_files == [], f"Unexpected temp files left behind: {tmp_files}"


def test_atomic_write_leaves_no_litter_on_write_error(tmp_path: Path, monkeypatch):
    """If the write step raises, the temp file is removed and the error propagates."""
    import tempfile as _tempfile

    repo_path = "/repo"
    entry = _make_entry(repo_path, install_path="/plugin/a")
    _make_v2_registry(tmp_path, {"plugin-a@marketplace": [entry]})

    original_mkstemp = _tempfile.mkstemp
    created_tmp: list[str] = []

    def _patched_mkstemp(**kwargs):
        fd, path = original_mkstemp(**kwargs)
        created_tmp.append(path)
        return fd, path

    monkeypatch.setattr(_tempfile, "mkstemp", _patched_mkstemp)

    # Patch os.replace to raise so the write "fails".
    monkeypatch.setattr(
        os,
        "replace",
        lambda *_a, **_kw: (_ for _ in ()).throw(OSError("simulated replace failure")),
    )

    with pytest.raises(OSError, match="simulated replace failure"):
        seed_plugin_registry(repo_path, "/wt", config_dir=tmp_path)

    # The temp file must have been cleaned up.
    for p in created_tmp:
        assert not os.path.exists(p), f"Temp file not cleaned up: {p}"


# ---------------------------------------------------------------------------
# Malformed entries — null / missing projectPath must not abort seeding
# ---------------------------------------------------------------------------

def test_null_project_path_skipped_gracefully(tmp_path: Path):
    """An entry with a JSON null projectPath is skipped without raising TypeError.

    Regression test for the bug where ``entry.get("projectPath", "")`` returns
    ``None`` when the key is present with a null value, and ``Path(None)``
    raises ``TypeError`` that propagates and aborts all seeding.

    A valid entry in the same list must still be cloned.
    """
    repo_path = "/home/user/repo"
    wt_path = "/home/user/store/wt1"

    null_entry = {
        "scope": "project",
        "projectPath": None,   # JSON null
        "installPath": "/plugin/null",
        "version": "1.0.0",
    }
    valid_entry = _make_entry(repo_path, install_path="/plugin/valid")

    _make_v2_registry(tmp_path, {
        "plugin-a@marketplace": [null_entry, valid_entry],
    })

    # Must not raise — the null entry is skipped, the valid one is cloned.
    seed_plugin_registry(repo_path, wt_path, config_dir=tmp_path)

    result = _read_registry(tmp_path)
    plugin_list = result["plugins"]["plugin-a@marketplace"]
    dest_path = str(Path(wt_path))
    cloned = [e for e in plugin_list if e.get("projectPath") == dest_path]
    assert len(cloned) == 1, "Valid entry should have been cloned"
    assert cloned[0]["installPath"] == "/plugin/valid"


def test_missing_project_path_skipped_gracefully(tmp_path: Path):
    """An entry whose projectPath key is absent is skipped without error.

    A valid entry in the same plugin list must still be cloned.
    """
    repo_path = "/home/user/repo"
    wt_path = "/home/user/store/wt2"

    no_path_entry = {
        "scope": "project",
        # No "projectPath" key at all.
        "installPath": "/plugin/missing",
        "version": "1.0.0",
    }
    valid_entry = _make_entry(repo_path, install_path="/plugin/ok")

    _make_v2_registry(tmp_path, {
        "plugin-b@marketplace": [no_path_entry, valid_entry],
    })

    seed_plugin_registry(repo_path, wt_path, config_dir=tmp_path)

    result = _read_registry(tmp_path)
    plugin_list = result["plugins"]["plugin-b@marketplace"]
    dest_path = str(Path(wt_path))
    cloned = [e for e in plugin_list if e.get("projectPath") == dest_path]
    assert len(cloned) == 1, "Valid entry should have been cloned"
    assert cloned[0]["installPath"] == "/plugin/ok"


def test_all_null_project_paths_no_write(tmp_path: Path):
    """When every entry has a null projectPath, nothing matches and the file is not rewritten."""
    null_entry = {
        "scope": "project",
        "projectPath": None,
        "installPath": "/plugin/null",
        "version": "1.0.0",
    }
    _make_v2_registry(tmp_path, {"plugin-a@marketplace": [null_entry]})
    registry_path = tmp_path / "plugins" / "installed_plugins.json"
    mtime_before = registry_path.stat().st_mtime

    seed_plugin_registry("/repo", "/wt", config_dir=tmp_path)

    assert registry_path.stat().st_mtime == mtime_before
