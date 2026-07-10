"""Unit tests for lib_python_worktree.core.plugin_install (ticket #62).

All tests run entirely in tmp_path — no git, no real `claude` subprocess.
`runner` / `which` seams are injected throughout so nothing here ever spawns
a real process. `WORKTREE_LOG_ROOT` is redirected via monkeypatch so log
files never touch the real home directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from lib_python_worktree.core.plugin_install import (
    PluginInstallResult,
    _already_registered,
    _clone_entry_to_worktree,
    _find_clone_source,
    _is_structurally_valid,
    _load_registry,
    _read_enabled_plugins,
    _resolve_claude_exe,
    _resolve_install_timeout,
    install_enabled_plugins,
)
from lib_python_worktree.setup.runner import LOG_ROOT_ENV, _slug, log_dir_for


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_settings(repo_root: Path, enabled: dict, *, local: bool = False) -> None:
    claude_dir = repo_root / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    name = "settings.local.json" if local else "settings.json"
    (claude_dir / name).write_text(
        json.dumps({"enabledPlugins": enabled}), encoding="utf-8"
    )


def _make_v2_registry(config_dir: Path, plugins: dict) -> Path:
    plugins_dir = config_dir / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    registry = plugins_dir / "installed_plugins.json"
    registry.write_text(
        json.dumps({"version": 2, "plugins": plugins}), encoding="utf-8"
    )
    return registry


def _make_entry(project_path: str, *, scope: str = "project", install_path: str = "/x") -> dict:
    return {
        "scope": scope,
        "projectPath": project_path,
        "installPath": install_path,
        "version": "1.0.0",
    }


def _make_valid_install(base: Path, name: str) -> str:
    """Create a real on-disk plugin install and return its installPath.

    Builds ``base/name/.claude-plugin/plugin.json`` so
    ``_is_structurally_valid`` (and thus ``_find_clone_source`` /
    ``_already_registered``) treats the resulting path as a genuine,
    parseable plugin install.
    """
    install_dir = base / name
    manifest_dir = install_dir / ".claude-plugin"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "plugin.json").write_text("{}", encoding="utf-8")
    return str(install_dir)


class _FakeRunner:
    """Records every invocation and returns a scripted outcome per key."""

    def __init__(self, outcomes: dict | None = None, default_rc: int = 0) -> None:
        self.calls: list[dict] = []
        self.outcomes = outcomes or {}
        self.default_rc = default_rc

    def __call__(self, cmd, *, cwd, timeout):  # noqa: ANN001
        self.calls.append({"cmd": cmd, "cwd": cwd, "timeout": timeout})
        key = cmd[3]  # ["claude", "plugin", "install", "<key>", "--scope", "project"]
        outcome = self.outcomes.get(key)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(cmd, timeout)
        if outcome == "oserror":
            raise OSError("spawn failed")
        rc = outcome if isinstance(outcome, int) else self.default_rc
        return types.SimpleNamespace(returncode=rc, stdout="ok", stderr="")


def _fake_which(resolvable: set) -> "callable":
    def _which(name):  # noqa: ANN001
        return f"/usr/bin/{name}" if name in resolvable else None
    return _which


# ---------------------------------------------------------------------------
# 1. enabledPlugins parsed from settings.json alone
# ---------------------------------------------------------------------------


def test_read_enabled_plugins_settings_json_only(tmp_path: Path):
    _write_settings(tmp_path, {"a@mkt": True, "b@mkt": False, "c@mkt": 1})
    keys = _read_enabled_plugins(str(tmp_path))
    assert sorted(keys) == ["a@mkt", "c@mkt"]


# ---------------------------------------------------------------------------
# 2. settings.local.json merge — local overrides base per-key
# ---------------------------------------------------------------------------


def test_read_enabled_plugins_local_overrides_base(tmp_path: Path):
    _write_settings(tmp_path, {"a@mkt": True, "b@mkt": True})
    _write_settings(tmp_path, {"b@mkt": False, "c@mkt": True}, local=True)

    keys = set(_read_enabled_plugins(str(tmp_path)))
    assert keys == {"a@mkt", "c@mkt"}, "local false must disable base true; local adds new key"


# ---------------------------------------------------------------------------
# 3. Missing settings.json / missing .claude dir -> empty result, no side effects
# ---------------------------------------------------------------------------


def test_install_missing_claude_dir_is_noop(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()

    runner = _FakeRunner()
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert result == PluginInstallResult()
    assert runner.calls == []
    assert not (tmp_path / "logs").exists(), "no log-dir side effects when there is nothing enabled"


# ---------------------------------------------------------------------------
# 4. Idempotency
# ---------------------------------------------------------------------------


def test_install_skips_already_registered_key(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    config_dir = tmp_path / "claude_config"
    install_path = _make_valid_install(tmp_path / "cache", "a-plugin")
    _make_v2_registry(
        config_dir, {"a@mkt": [_make_entry(str(worktree_path), install_path=install_path)]}
    )

    runner = _FakeRunner()
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=config_dir,
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert result.skipped == ["a@mkt"]
    assert result.installed == []
    assert result.failed == []
    assert runner.calls == [], "runner must never be called for an already-registered key"


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows path normalisation (backslash vs forward-slash) — POSIX cannot reproduce",
)
def test_install_skips_already_registered_windows_path_normalisation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    worktree_path = tmp_path / "wt"
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    # Registry stores the worktree path with forward slashes; the caller
    # passes the native (backslash) Path — normalisation must still match.
    posix_form = str(worktree_path).replace("\\", "/")
    config_dir = tmp_path / "claude_config"
    install_path = _make_valid_install(tmp_path / "cache", "a-plugin")
    _make_v2_registry(
        config_dir, {"a@mkt": [_make_entry(posix_form, install_path=install_path)]}
    )

    runner = _FakeRunner()
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=config_dir,
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert result.skipped == ["a@mkt"]
    assert runner.calls == []


# ---------------------------------------------------------------------------
# 5. CLI missing -> claude_unavailable, but clone-first still runs (ticket #64)
# ---------------------------------------------------------------------------


def test_install_cli_unavailable_no_clone_source_fails(tmp_path: Path, monkeypatch):
    """No CLI and no valid clone source -> key fails, runner never called.

    ticket #64 changed the CLI-unavailable behaviour from an early return to
    "continue with clone-first". With no valid source anywhere in the
    (isolated) registry, the key must end up in ``failed`` rather than
    silently doing nothing.
    """
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    runner = _FakeRunner()
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=tmp_path / "claude_config",  # isolated: no ambient ~/.claude registry
        which=_fake_which(set()),  # nothing resolves
        runner=runner,
    )

    assert result.claude_unavailable is True
    assert result.installed == []
    assert result.skipped == []
    assert result.failed == ["a@mkt"]
    assert result.warnings, "a warning must explain why nothing was installed"
    assert runner.calls == []
    assert not (tmp_path / "logs").exists(), "no CLI attempt means no log side effects"


def test_install_cli_unavailable_with_clone_source_still_installs(tmp_path: Path, monkeypatch):
    """No CLI, but a valid clone source exists -> key is installed via clone.

    This is the core ticket #64 fix: registration must not depend on the
    ``claude`` CLI being resolvable at all.
    """
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    config_dir = tmp_path / "claude_config"
    install_path = _make_valid_install(tmp_path / "cache", "a-plugin")
    _make_v2_registry(
        config_dir,
        {"a@mkt": [_make_entry("/some/other/project", install_path=install_path)]},
    )

    runner = _FakeRunner()
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=config_dir,
        which=_fake_which(set()),  # nothing resolves
        runner=runner,
    )

    assert result.claude_unavailable is True
    assert result.installed == ["a@mkt"]
    assert result.failed == []
    assert runner.calls == [], "clone-first must never shell out"

    data = json.loads((config_dir / "plugins" / "installed_plugins.json").read_text(encoding="utf-8"))
    cloned = [e for e in data["plugins"]["a@mkt"] if e.get("projectPath") == str(worktree_path)]
    assert len(cloned) == 1
    assert cloned[0]["installPath"] == install_path
    assert cloned[0]["scope"] == "project"


# ---------------------------------------------------------------------------
# 6. Happy path
# ---------------------------------------------------------------------------


def test_install_happy_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    runner = _FakeRunner(outcomes={"a@mkt": 0})
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        worktree_id="my-worktree-id",
        config_dir=tmp_path / "claude_config",  # isolated: no ambient ~/.claude registry
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert result.installed == ["a@mkt"]
    assert result.failed == []
    assert result.skipped == []

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["cmd"] == ["/usr/bin/claude", "plugin", "install", "a@mkt", "--scope", "project"]
    assert call["cwd"] == str(worktree_path)

    log_dir = log_dir_for("my-worktree-id")
    expected_log = log_dir / f"plugin-install-{_slug('a@mkt')}.log"
    assert expected_log.exists(), f"expected log file at {expected_log}"


# ---------------------------------------------------------------------------
# 7. Per-plugin failure — batch continues
# ---------------------------------------------------------------------------


def test_install_batch_continues_after_one_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True, "b@mkt": True, "c@mkt": True})

    runner = _FakeRunner(outcomes={"a@mkt": 0, "b@mkt": 1, "c@mkt": 0})
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=tmp_path / "claude_config",  # isolated: no ambient ~/.claude registry
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert sorted(result.installed) == ["a@mkt", "c@mkt"]
    assert result.failed == ["b@mkt"]
    assert any("b@mkt" in w for w in result.warnings)
    # All three keys must have been attempted despite the mid-batch failure.
    assert len(runner.calls) == 3


# ---------------------------------------------------------------------------
# 8. Timeout path — batch continues
# ---------------------------------------------------------------------------


def test_install_timeout_continues_batch(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True, "b@mkt": True})

    runner = _FakeRunner(outcomes={"a@mkt": "timeout", "b@mkt": 0})
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=tmp_path / "claude_config",  # isolated: no ambient ~/.claude registry
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert result.failed == ["a@mkt"]
    assert result.installed == ["b@mkt"]
    assert any("timed out" in w for w in result.warnings)
    assert len(runner.calls) == 2


def test_install_spawn_error_continues_batch(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True, "b@mkt": True})

    runner = _FakeRunner(outcomes={"a@mkt": "oserror", "b@mkt": 0})
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=tmp_path / "claude_config",  # isolated: no ambient ~/.claude registry
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert result.failed == ["a@mkt"]
    assert result.installed == ["b@mkt"]


# ---------------------------------------------------------------------------
# 9. _resolve_install_timeout precedence (mirrors _resolve_git_timeout)
# ---------------------------------------------------------------------------


def test_resolve_install_timeout_explicit_wins(monkeypatch):
    monkeypatch.setenv("WORKTREE_PLUGIN_INSTALL_TIMEOUT_SEC", "5")
    assert _resolve_install_timeout(12.5) == 12.5


def test_resolve_install_timeout_env_used_when_no_explicit(monkeypatch):
    monkeypatch.setenv("WORKTREE_PLUGIN_INSTALL_TIMEOUT_SEC", "5")
    assert _resolve_install_timeout(None) == 5.0


def test_resolve_install_timeout_empty_env_disables(monkeypatch):
    monkeypatch.setenv("WORKTREE_PLUGIN_INSTALL_TIMEOUT_SEC", "")
    assert _resolve_install_timeout(None) is None


def test_resolve_install_timeout_garbage_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("WORKTREE_PLUGIN_INSTALL_TIMEOUT_SEC", "not-a-number")
    assert _resolve_install_timeout(None) == 60.0


def test_resolve_install_timeout_no_env_uses_default(monkeypatch):
    monkeypatch.delenv("WORKTREE_PLUGIN_INSTALL_TIMEOUT_SEC", raising=False)
    assert _resolve_install_timeout(None) == 60.0


def test_resolve_install_timeout_zero_or_negative_env_disables(monkeypatch):
    monkeypatch.setenv("WORKTREE_PLUGIN_INSTALL_TIMEOUT_SEC", "0")
    assert _resolve_install_timeout(None) is None
    monkeypatch.setenv("WORKTREE_PLUGIN_INSTALL_TIMEOUT_SEC", "-1")
    assert _resolve_install_timeout(None) is None


# ---------------------------------------------------------------------------
# 10. Log filename slugging
# ---------------------------------------------------------------------------


def test_log_filename_slugging_is_filesystem_safe():
    slug = _slug("foo@bar-marketplace")
    assert "@" not in slug
    # No characters outside [a-z0-9-] should survive.
    assert all(c.isalnum() or c == "-" for c in slug)


# ---------------------------------------------------------------------------
# _resolve_claude_exe
# ---------------------------------------------------------------------------


def test_resolve_claude_exe_finds_plain_claude():
    assert _resolve_claude_exe(_fake_which({"claude"})) == "/usr/bin/claude"


def test_resolve_claude_exe_returns_none_when_unresolvable():
    assert _resolve_claude_exe(_fake_which(set())) is None


@pytest.mark.skipif(sys.platform != "win32", reason="win32-only fallback candidates")
def test_resolve_claude_exe_windows_fallback_candidates():
    assert _resolve_claude_exe(_fake_which({"claude.cmd"})) == "/usr/bin/claude.cmd"


# ---------------------------------------------------------------------------
# _already_registered — direct unit coverage
# ---------------------------------------------------------------------------


def test_already_registered_false_for_missing_registry():
    assert _already_registered({}, "a@mkt", "/wt") is False


def test_already_registered_false_for_non_project_scope():
    registry = {"plugins": {"a@mkt": [_make_entry("/wt", scope="global")]}}
    assert _already_registered(registry, "a@mkt", "/wt") is False


def test_already_registered_false_for_structurally_broken_install_path(tmp_path: Path):
    """Regression (#64): a project-scoped entry with a broken installPath is
    not treated as already-registered — it must self-repair instead."""
    registry = {
        "plugins": {"a@mkt": [_make_entry("/wt", install_path=str(tmp_path / "missing"))]}
    }
    assert _already_registered(registry, "a@mkt", "/wt") is False


# ---------------------------------------------------------------------------
# _is_structurally_valid — direct unit coverage (ticket #64)
# ---------------------------------------------------------------------------


def test_is_structurally_valid_true_for_real_install(tmp_path: Path):
    install_path = _make_valid_install(tmp_path, "real-plugin")
    assert _is_structurally_valid(install_path) is True


def test_is_structurally_valid_false_for_missing_manifest(tmp_path: Path):
    broken = tmp_path / "broken-plugin"
    broken.mkdir()
    assert _is_structurally_valid(str(broken)) is False


def test_is_structurally_valid_false_for_falsy_or_non_str():
    assert _is_structurally_valid(None) is False
    assert _is_structurally_valid("") is False


def test_is_structurally_valid_false_for_corrupt_manifest(tmp_path: Path):
    manifest_dir = tmp_path / "corrupt-plugin" / ".claude-plugin"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "plugin.json").write_text("{not json", encoding="utf-8")
    assert _is_structurally_valid(str(tmp_path / "corrupt-plugin")) is False


# ---------------------------------------------------------------------------
# _find_clone_source — picker coverage (ticket #64)
# ---------------------------------------------------------------------------


def test_find_clone_source_ignores_newer_but_broken(tmp_path: Path):
    """A newer entry with a broken installPath must not beat an older, valid one."""
    valid_path = _make_valid_install(tmp_path / "cache", "old-valid")
    registry = {
        "plugins": {
            "a@mkt": [
                {
                    "scope": "project",
                    "projectPath": "/old",
                    "installPath": valid_path,
                    "installedAt": "2024-01-01T00:00:00Z",
                    "resolvedVersion": "1.0.0",
                },
                {
                    "scope": "project",
                    "projectPath": "/new",
                    "installPath": str(tmp_path / "does-not-exist"),
                    "installedAt": "2025-01-01T00:00:00Z",
                    "resolvedVersion": "2.0.0",
                },
            ]
        }
    }
    source = _find_clone_source(registry, "a@mkt")
    assert source is not None
    assert source["installPath"] == valid_path


def test_find_clone_source_returns_none_when_no_valid_entries(tmp_path: Path):
    registry = {
        "plugins": {"a@mkt": [_make_entry("/wt", install_path=str(tmp_path / "missing"))]}
    }
    assert _find_clone_source(registry, "a@mkt") is None


def test_find_clone_source_picks_newest_among_valid(tmp_path: Path):
    older = _make_valid_install(tmp_path / "cache", "older")
    newer = _make_valid_install(tmp_path / "cache", "newer")
    registry = {
        "plugins": {
            "a@mkt": [
                {
                    "scope": "project",
                    "projectPath": "/old",
                    "installPath": older,
                    "installedAt": "2024-01-01T00:00:00Z",
                    "resolvedVersion": "1.0.0",
                },
                {
                    "scope": "project",
                    "projectPath": "/new",
                    "installPath": newer,
                    "installedAt": "2025-01-01T00:00:00Z",
                    "resolvedVersion": "2.0.0",
                },
            ]
        }
    }
    source = _find_clone_source(registry, "a@mkt")
    assert source["installPath"] == newer


# ---------------------------------------------------------------------------
# CLI failure + valid source -> second-chance clone recovery (ticket #64)
# ---------------------------------------------------------------------------


def test_install_cli_eperm_failure_recovers_via_clone(tmp_path: Path, monkeypatch):
    """Windows-EPERM-style CLI failure, where the CLI itself partially
    populated the registry with a now-valid source before failing -> the
    key still ends up installed via the second-chance clone.

    No valid source may exist *before* the CLI call (otherwise clone-first
    would pre-empt the CLI attempt entirely and this wouldn't exercise the
    recovery path) -- the fake runner writes the valid entry as a side
    effect of being invoked, simulating the real `claude plugin install`
    downloading/registering the plugin cache before failing to finish
    (the Windows EPERM failure ticket #64 targets).
    """
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    config_dir = tmp_path / "claude_config"
    install_path = _make_valid_install(tmp_path / "cache", "a-plugin")
    # No pre-existing valid source: registry starts empty.
    _make_v2_registry(config_dir, {})

    class _EpermRunner:
        def __init__(self):
            self.calls = []

        def __call__(self, cmd, *, cwd, timeout):
            self.calls.append({"cmd": cmd, "cwd": cwd, "timeout": timeout})
            # Simulate the CLI partially populating the registry (download +
            # register succeeded) before failing with EPERM on a later step.
            _make_v2_registry(
                config_dir,
                {"a@mkt": [_make_entry("/some/other/project", install_path=install_path)]},
            )
            raise OSError("[WinError 5] Access is denied: 'installed_plugins.json'")

    runner = _EpermRunner()
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=config_dir,
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert result.installed == ["a@mkt"]
    assert result.failed == []
    assert any("recovered via registry clone" in w for w in result.warnings)
    assert len(runner.calls) == 1, "the CLI must still be tried before falling back"

    data = json.loads((config_dir / "plugins" / "installed_plugins.json").read_text(encoding="utf-8"))
    cloned = [e for e in data["plugins"]["a@mkt"] if e.get("projectPath") == str(worktree_path)]
    assert len(cloned) == 1


def test_install_cli_generic_nonzero_failure_recovers_via_clone(tmp_path: Path, monkeypatch):
    """A plain nonzero exit code, where the CLI attempt itself left behind a
    now-valid source, also recovers via the second-chance clone."""
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    config_dir = tmp_path / "claude_config"
    install_path = _make_valid_install(tmp_path / "cache", "a-plugin")
    _make_v2_registry(config_dir, {})

    class _PartialFailRunner:
        def __init__(self, rc: int):
            self.rc = rc
            self.calls = []

        def __call__(self, cmd, *, cwd, timeout):
            self.calls.append({"cmd": cmd, "cwd": cwd, "timeout": timeout})
            _make_v2_registry(
                config_dir,
                {"a@mkt": [_make_entry("/some/other/project", install_path=install_path)]},
            )
            return types.SimpleNamespace(returncode=self.rc, stdout="", stderr="boom")

    runner = _PartialFailRunner(rc=1)
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=config_dir,
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert result.installed == ["a@mkt"]
    assert result.failed == []
    assert any("recovered via registry clone" in w for w in result.warnings)
    assert len(runner.calls) == 1


def test_install_cli_failure_no_source_still_fails(tmp_path: Path, monkeypatch):
    """CLI failure with no valid clone source anywhere -> key stays failed."""
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    runner = _FakeRunner(outcomes={"a@mkt": 1})
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=tmp_path / "claude_config",  # isolated: no ambient registry
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert result.installed == []
    assert result.failed == ["a@mkt"]
    assert any("exited with code 1" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Idempotency across two runs (ticket #64)
# ---------------------------------------------------------------------------


def test_install_clone_idempotent_across_two_runs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    config_dir = tmp_path / "claude_config"
    install_path = _make_valid_install(tmp_path / "cache", "a-plugin")
    _make_v2_registry(
        config_dir,
        {"a@mkt": [_make_entry("/some/other/project", install_path=install_path)]},
    )

    runner = _FakeRunner()

    first = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=config_dir,
        which=_fake_which(set()),
        runner=runner,
    )
    assert first.installed == ["a@mkt"]

    second = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=config_dir,
        which=_fake_which(set()),
        runner=runner,
    )
    assert second.skipped == ["a@mkt"]
    assert second.installed == []
    assert runner.calls == [], "clone-first must never shell out in this scenario"

    data = json.loads((config_dir / "plugins" / "installed_plugins.json").read_text(encoding="utf-8"))
    cloned = [e for e in data["plugins"]["a@mkt"] if e.get("projectPath") == str(worktree_path)]
    assert len(cloned) == 1, "exactly one project-scoped clone must exist after two runs"


# ---------------------------------------------------------------------------
# Concurrent racing writers (ticket #64, Q4)
# ---------------------------------------------------------------------------


def test_clone_entry_to_worktree_concurrent_writers_no_lost_update(tmp_path: Path):
    """Two threads cloning different keys into the same registry file
    concurrently must not lose either update (portalocker serialises the
    read-modify-write cycle)."""
    import threading

    config_dir = tmp_path / "claude_config"
    install_a = _make_valid_install(tmp_path / "cache", "plugin-a")
    install_b = _make_valid_install(tmp_path / "cache", "plugin-b")
    _make_v2_registry(
        config_dir,
        {
            "a@mkt": [_make_entry("/other-a", install_path=install_a)],
            "b@mkt": [_make_entry("/other-b", install_path=install_b)],
        },
    )

    worktree_path = str(tmp_path / "wt")
    source_a = {"scope": "project", "projectPath": "/other-a", "installPath": install_a}
    source_b = {"scope": "project", "projectPath": "/other-b", "installPath": install_b}

    results = {}

    def _clone_a():
        results["a"] = _clone_entry_to_worktree(config_dir, "a@mkt", source_a, worktree_path)

    def _clone_b():
        results["b"] = _clone_entry_to_worktree(config_dir, "b@mkt", source_b, worktree_path)

    t1 = threading.Thread(target=_clone_a)
    t2 = threading.Thread(target=_clone_b)
    t1.start()
    t2.start()
    t1.join(timeout=15)
    t2.join(timeout=15)

    assert results == {"a": True, "b": True}

    data = json.loads((config_dir / "plugins" / "installed_plugins.json").read_text(encoding="utf-8"))
    assert data["version"] == 2

    a_clones = [e for e in data["plugins"]["a@mkt"] if e.get("projectPath") == worktree_path]
    b_clones = [e for e in data["plugins"]["b@mkt"] if e.get("projectPath") == worktree_path]
    assert len(a_clones) == 1, "key 'a@mkt' clone must survive the race"
    assert len(b_clones) == 1, "key 'b@mkt' clone must survive the race"
    # Originals must still be intact too.
    assert len(data["plugins"]["a@mkt"]) == 2
    assert len(data["plugins"]["b@mkt"]) == 2


# ---------------------------------------------------------------------------
# Integration-style: fake claude exe + real on-disk cache tree (ticket #64)
# ---------------------------------------------------------------------------


def test_install_integration_fake_cli_plus_real_cache_tree(tmp_path: Path, monkeypatch):
    """End-to-end: a fake `claude` exe (via runner) plus a real cache tree
    with an actual `.claude-plugin/plugin.json` -- drive the full
    install_enabled_plugins() and assert the registry end-state."""
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    config_dir = tmp_path / "claude_config"
    install_path = _make_valid_install(tmp_path / "cache", "a-plugin")
    _make_v2_registry(
        config_dir,
        {"a@mkt": [_make_entry("/some/other/project", install_path=install_path)]},
    )

    runner = _FakeRunner()  # would return rc=0 if ever invoked
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=config_dir,
        which=_fake_which({"claude"}),
        runner=runner,
    )

    assert result.installed == ["a@mkt"]
    assert runner.calls == [], "a valid clone source must pre-empt any CLI call"

    data = json.loads((config_dir / "plugins" / "installed_plugins.json").read_text(encoding="utf-8"))
    plugin_list = data["plugins"]["a@mkt"]
    cloned = [e for e in plugin_list if e.get("projectPath") == str(worktree_path)]
    assert len(cloned) == 1
    assert cloned[0]["installPath"] == install_path
    assert cloned[0]["scope"] == "project"


# ---------------------------------------------------------------------------
# Regression: broken installPath is not "already installed" (ticket #64)
# ---------------------------------------------------------------------------


def test_install_repairs_broken_registration(tmp_path: Path, monkeypatch):
    """A worktree with a project-scoped registry entry whose installPath is
    broken must be repaired (not skipped as already-installed).

    This reproduces the reported problem: a prior failed/partial install (or
    manual corruption) left a project-scoped entry pointing at a
    missing/broken installPath, so the worktree silently never got the
    plugin loaded. Confirms the entry is treated as not-yet-installed and
    gets repaired via clone once a valid source appears.
    """
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "logs"))
    repo_root = tmp_path / "repo"
    worktree_path = tmp_path / "wt"
    repo_root.mkdir()
    worktree_path.mkdir()

    _write_settings(repo_root, {"a@mkt": True})

    config_dir = tmp_path / "claude_config"
    broken_install_path = str(tmp_path / "gone")
    valid_install_path = _make_valid_install(tmp_path / "cache", "a-plugin")
    _make_v2_registry(
        config_dir,
        {
            "a@mkt": [
                # Broken registration for *this* worktree -- must not count
                # as already-installed.
                _make_entry(str(worktree_path), install_path=broken_install_path),
                # A separate, valid source elsewhere to repair from.
                _make_entry("/some/other/project", install_path=valid_install_path),
            ]
        },
    )

    runner = _FakeRunner()
    result = install_enabled_plugins(
        str(repo_root),
        str(worktree_path),
        config_dir=config_dir,
        which=_fake_which(set()),
        runner=runner,
    )

    assert result.skipped == [], "the broken registration must not be treated as already-installed"
    assert result.installed == ["a@mkt"]
    assert runner.calls == []

    data = json.loads((config_dir / "plugins" / "installed_plugins.json").read_text(encoding="utf-8"))
    plugin_list = data["plugins"]["a@mkt"]
    repaired = [
        e
        for e in plugin_list
        if e.get("projectPath") == str(worktree_path) and e.get("installPath") == valid_install_path
    ]
    assert len(repaired) == 1, "a repaired clone with the valid installPath must be added"
