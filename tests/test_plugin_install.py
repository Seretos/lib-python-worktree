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


def _make_entry(project_path: str, *, scope: str = "project") -> dict:
    return {"scope": scope, "projectPath": project_path, "installPath": "/x", "version": "1.0.0"}


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
    _make_v2_registry(
        config_dir, {"a@mkt": [_make_entry(str(worktree_path))]}
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
    _make_v2_registry(config_dir, {"a@mkt": [_make_entry(posix_form)]})

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
# 5. CLI missing -> claude_unavailable, runner never called, no logs written
# ---------------------------------------------------------------------------


def test_install_cli_unavailable(tmp_path: Path, monkeypatch):
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
        which=_fake_which(set()),  # nothing resolves
        runner=runner,
    )

    assert result.claude_unavailable is True
    assert result.installed == []
    assert result.skipped == []
    assert result.failed == []
    assert result.warnings, "a warning must explain why nothing was installed"
    assert runner.calls == []
    assert not (tmp_path / "logs").exists()


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
