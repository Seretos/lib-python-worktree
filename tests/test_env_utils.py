"""Tests for ``lib_python_worktree.core._env_utils._get_user_profile_env``.

All tests are platform-portable: the Windows registry branch is exercised via
mocks so that the test suite passes on Linux/macOS CI runners too, without
requiring a real Windows registry.
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# The module under test.
import lib_python_worktree.core._env_utils as _env_utils_module
from lib_python_worktree.core._env_utils import _get_user_profile_env


# ---------------------------------------------------------------------------
# Non-Windows path
# ---------------------------------------------------------------------------


def test_non_windows_returns_copy_of_os_environ(monkeypatch):
    """On non-Windows, _get_user_profile_env() returns a plain copy of os.environ."""
    monkeypatch.setattr(_env_utils_module.sys, "platform", "linux")

    result = _get_user_profile_env()

    assert result == dict(os.environ), "result must equal dict(os.environ) on non-Windows"
    assert result is not os.environ, "result must be a copy, not the same object"


# ---------------------------------------------------------------------------
# Windows registry mocking helpers
# ---------------------------------------------------------------------------

def _make_fake_winreg(
    *,
    hklm_vars: dict | None = None,
    hkcu_vars: dict | None = None,
    open_key_raises: bool = False,
):
    """Build a fake ``winreg`` module for mocking.

    Each variable is stored as a (name, value, kind) tuple where kind is
    ``REG_SZ`` (1) by default.
    """
    hklm_vars = hklm_vars or {}
    hkcu_vars = hkcu_vars or {}

    REG_SZ = 1
    REG_EXPAND_SZ = 2
    HKEY_LOCAL_MACHINE = "HKLM"
    HKEY_CURRENT_USER = "HKCU"

    # Build value lists for each hive key.
    hklm_values = [(k, v, REG_SZ) for k, v in hklm_vars.items()]
    hkcu_values = [(k, v, REG_SZ) for k, v in hkcu_vars.items()]

    class _FakeKey:
        def __init__(self, values):
            self._values = list(values)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def _open_key(hive, subkey):
        if open_key_raises:
            raise OSError("simulated registry open failure")
        if hive == HKEY_LOCAL_MACHINE:
            return _FakeKey(hklm_values)
        if hive == HKEY_CURRENT_USER:
            return _FakeKey(hkcu_values)
        raise OSError("unknown hive")

    def _enum_value(key, index):
        if index >= len(key._values):
            raise OSError("no more values")
        return key._values[index]

    def _expand_env_strings(s):
        # Simple pass-through; callers that need expansion mock this further.
        return s

    fake_winreg = types.ModuleType("winreg")
    fake_winreg.HKEY_LOCAL_MACHINE = HKEY_LOCAL_MACHINE
    fake_winreg.HKEY_CURRENT_USER = HKEY_CURRENT_USER
    fake_winreg.REG_SZ = REG_SZ
    fake_winreg.REG_EXPAND_SZ = REG_EXPAND_SZ
    fake_winreg.OpenKey = _open_key
    fake_winreg.EnumValue = _enum_value
    fake_winreg.ExpandEnvironmentStrings = _expand_env_strings
    return fake_winreg


# ---------------------------------------------------------------------------
# Windows tests (all mocked)
# ---------------------------------------------------------------------------


def test_windows_registry_machine_vars_included(monkeypatch):
    """HKLM machine env vars appear in the result on Windows."""
    monkeypatch.setattr(_env_utils_module.sys, "platform", "win32")

    fake_winreg = _make_fake_winreg(hklm_vars={"MACHINE_VAR": "from_machine"})

    with patch.dict("sys.modules", {"winreg": fake_winreg}):
        result = _get_user_profile_env()

    assert "MACHINE_VAR" in result, "HKLM variable must appear in the result"
    assert result["MACHINE_VAR"] == "from_machine"


def test_windows_user_vars_override_machine(monkeypatch):
    """HKCU vars win over HKLM when both define the same key."""
    monkeypatch.setattr(_env_utils_module.sys, "platform", "win32")

    fake_winreg = _make_fake_winreg(
        hklm_vars={"SHARED_KEY": "machine_value"},
        hkcu_vars={"SHARED_KEY": "user_value"},
    )

    with patch.dict("sys.modules", {"winreg": fake_winreg}):
        result = _get_user_profile_env()

    assert result["SHARED_KEY"] == "user_value", (
        "HKCU value must win over HKLM value for the same key"
    )


def test_windows_os_environ_overlaid_last(monkeypatch):
    """os.environ is overlaid last so live process vars (e.g. PATH additions) survive."""
    monkeypatch.setattr(_env_utils_module.sys, "platform", "win32")
    monkeypatch.setenv("FOO", "process_value")

    fake_winreg = _make_fake_winreg(hkcu_vars={"FOO": "registry_value"})

    with patch.dict("sys.modules", {"winreg": fake_winreg}):
        result = _get_user_profile_env()

    assert result["FOO"] == "process_value", (
        "os.environ must win over registry values (overlaid last)"
    )


def test_windows_registry_open_failure_falls_back_gracefully(monkeypatch):
    """When winreg.OpenKey raises OSError for both keys, result still contains os.environ vars."""
    monkeypatch.setattr(_env_utils_module.sys, "platform", "win32")

    fake_winreg = _make_fake_winreg(open_key_raises=True)

    with patch.dict("sys.modules", {"winreg": fake_winreg}):
        result = _get_user_profile_env()

    # Must not raise and must still contain os.environ content.
    assert isinstance(result, dict)
    for key in os.environ:
        assert key in result, f"os.environ key {key!r} must survive a registry open failure"


def test_windows_reg_expand_sz_expanded(monkeypatch):
    """REG_EXPAND_SZ values are expanded via ExpandEnvironmentStrings."""
    monkeypatch.setattr(_env_utils_module.sys, "platform", "win32")

    REG_EXPAND_SZ = 2
    REG_SZ = 1
    HKEY_LOCAL_MACHINE = "HKLM"
    HKEY_CURRENT_USER = "HKCU"

    # A single HKLM value of type REG_EXPAND_SZ.
    raw_value = r"%SystemRoot%\system32"
    expanded_value = r"C:\Windows\system32"

    class _FakeKey:
        def __init__(self, values):
            self._values = list(values)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    hklm_values = [("Path", raw_value, REG_EXPAND_SZ)]

    def _open_key(hive, subkey):
        if hive == HKEY_LOCAL_MACHINE:
            return _FakeKey(hklm_values)
        return _FakeKey([])

    def _enum_value(key, index):
        if index >= len(key._values):
            raise OSError("no more values")
        return key._values[index]

    def _expand_env_strings(s):
        # Simulate expansion for the known pattern.
        return s.replace("%SystemRoot%", "C:\\Windows")

    fake_winreg = types.ModuleType("winreg")
    fake_winreg.HKEY_LOCAL_MACHINE = HKEY_LOCAL_MACHINE
    fake_winreg.HKEY_CURRENT_USER = HKEY_CURRENT_USER
    fake_winreg.REG_SZ = REG_SZ
    fake_winreg.REG_EXPAND_SZ = REG_EXPAND_SZ
    fake_winreg.OpenKey = _open_key
    fake_winreg.EnumValue = _enum_value
    fake_winreg.ExpandEnvironmentStrings = _expand_env_strings

    with patch.dict("sys.modules", {"winreg": fake_winreg}):
        result = _get_user_profile_env()

    # If os.environ doesn't override "Path", the expanded value must be present.
    # On Windows "Path" might be in os.environ too; just verify ExpandEnvironmentStrings
    # was called — we do this by checking that if the key survived, it is expanded.
    # On non-Windows CI runners "Path" won't be in os.environ (case-sensitive),
    # so we should see the expanded value directly.
    if "Path" not in os.environ:
        assert result.get("Path") == expanded_value, (
            f"REG_EXPAND_SZ value must be expanded; got {result.get('Path')!r}"
        )
    else:
        # os.environ wins — that's correct behaviour; just verify no exception was raised.
        assert isinstance(result, dict)
