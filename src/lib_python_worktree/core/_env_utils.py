"""Environment-building utilities.

Provides ``_get_user_profile_env``, which reconstructs a full user-profile
environment from the Windows registry on Windows, or returns a copy of
``os.environ`` on all other platforms.

This exists because MCP servers run headless (no interactive logon shell),
so their ``os.environ`` may be missing Windows user-profile variables such as
``APPDATA``, ``LOCALAPPDATA``, ``USERPROFILE``, ``TEMP``, ``TMP``,
``USERNAME``, and ``COMPUTERNAME``.  Child processes (e.g. Unity Accelerator)
that are spawned directly (not via a login shell) inherit that incomplete
environment and may crash or behave incorrectly without those variables.

On Windows the function reads:
  1. HKLM ``SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment``
     (machine-wide env vars, including ``SystemRoot``, ``ProgramFiles``, etc.)
  2. HKCU ``Environment`` (per-user vars, including ``APPDATA``, ``TEMP``, etc.)

HKCU wins over HKLM on collision.  ``REG_EXPAND_SZ`` values are expanded via
``winreg.ExpandEnvironmentStrings`` so that values like ``%SystemRoot%\\system32``
are resolved correctly.  Finally, the live ``os.environ`` is overlaid last so
that any additions made by the MCP server or its launch wrapper (e.g. extra
``PATH`` entries, ``VIRTUAL_ENV``, etc.) survive into the child environment.

Per-key ``OSError`` exceptions from ``winreg`` are silently swallowed so that
a single missing or unreadable registry key never aborts the merge.
"""

from __future__ import annotations

import os
import sys
from typing import Dict

__all__ = ("_get_user_profile_env",)


def _get_user_profile_env() -> Dict[str, str]:
    """Return a complete user-profile environment dict.

    On non-Windows platforms this is equivalent to ``dict(os.environ)`` — a
    plain copy so callers can mutate it freely.

    On Windows the dict is built by merging:
      HKLM machine vars  <--  HKCU user vars  <--  os.environ (rightmost wins)

    Any ``OSError`` raised while accessing the registry is silently swallowed
    on a per-key basis.
    """
    if sys.platform != "win32":
        return dict(os.environ)

    # Import winreg inside the Windows-only branch so non-Windows interpreters
    # (which don't have this stdlib module) never attempt to import it.
    import winreg  # type: ignore[import]  # noqa: PLC0415

    env: Dict[str, str] = {}

    def _read_key(hive: int, subkey: str) -> None:
        """Read all values from a registry key and merge them into ``env``."""
        try:
            key = winreg.OpenKey(hive, subkey)
        except OSError:
            return
        with key:
            index = 0
            while True:
                try:
                    name, data, kind = winreg.EnumValue(key, index)
                except OSError:
                    # No more values (ERROR_NO_MORE_ITEMS) or other error.
                    break
                index += 1
                try:
                    if kind == winreg.REG_EXPAND_SZ:
                        data = winreg.ExpandEnvironmentStrings(data)
                    if isinstance(data, str):
                        env[name] = data
                except OSError:
                    # ExpandEnvironmentStrings or name/data access failed.
                    pass

    # Step 1: machine-wide vars (HKLM wins over nothing; HKCU will override).
    _read_key(
        winreg.HKEY_LOCAL_MACHINE,
        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
    )

    # Step 2: per-user vars (HKCU overrides HKLM).
    _read_key(winreg.HKEY_CURRENT_USER, "Environment")

    # Step 3: live os.environ overlaid last (MCP-server additions survive).
    env.update(os.environ)

    return env
