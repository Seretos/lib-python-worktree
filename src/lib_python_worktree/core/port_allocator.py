"""Port allocator for the worktree engine (W4).

Picks free TCP ports from a configurable range, verifies availability both
against ``ports.yaml`` (via ``_PortsFile``) and against the OS (via
``_port_in_use``), and releases them when a worktree is removed.

Concurrency note
----------------
``_PortsFile.get_all()`` and ``_PortsFile.set_all()`` each acquire their own
exclusive portalocker lock on ``ports.yaml.lock``.  A separate read + write
pair would therefore NOT be atomic against concurrent callers.

To achieve a single atomic read-modify-write we acquire the lock *once* via
``portalocker.Lock`` (the same lock file that ``_PortsFile`` uses) and call
the private ``_load()`` / ``_save()`` helpers directly while the lock is held.
This mirrors the pattern used inside ``reconcile()`` in ``yaml_store.py``, so
it is consistent with the existing established pattern in this codebase.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import portalocker

from .yaml_store import _LOCK_FLAGS, _LOCK_TIMEOUT, _PortsFile, _port_in_use

_KEY_SEP = ":"


class PortAllocationError(RuntimeError):
    """Raised when no free port can be found for a requested slot."""


class PortAllocator:
    """Allocate and release named port slots for worktrees.

    Parameters
    ----------
    ports_file:
        A ``_PortsFile`` instance wrapping the on-disk ``ports.yaml``.
    port_range:
        Inclusive ``(low, high)`` range from which ports are drawn.
    """

    def __init__(
        self,
        ports_file: _PortsFile,
        port_range: tuple[int, int] = (30000, 40000),
    ) -> None:
        self._ports_file = ports_file
        self._port_range = port_range

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _lock_path(self) -> str:
        """Canonical path for the ports-file exclusive lock."""
        return str(self._ports_file._path) + ".lock"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate(self, slots: List[str], worktree_id: str) -> Dict[str, int]:
        """Allocate one port per slot name for the given worktree.

        The entire read-modify-write is done under a single exclusive lock so
        concurrent callers cannot race and claim the same port.

        Parameters
        ----------
        slots:
            Ordered list of slot names (must be unique; validation is the
            caller's responsibility).
        worktree_id:
            The worktree id; used as the key prefix in ``ports.yaml``.

        Returns
        -------
        dict[slot_name, port_number]
            Empty dict when ``slots`` is empty (no lock acquired).

        Raises
        ------
        PortAllocationError
            If no free port exists for any requested slot.
        """
        if not slots:
            return {}

        low, high = self._port_range
        all_ports = list(range(low, high + 1))

        with portalocker.Lock(self._lock_path, timeout=_LOCK_TIMEOUT, flags=_LOCK_FLAGS):
            allocated: Dict[str, int] = self._ports_file._load()
            # Ports already claimed by any worktree.
            taken: set[int] = set(allocated.values())
            result: Dict[str, int] = {}

            for slot in slots:
                # Shuffle a fresh copy so iteration order is random.
                candidates = all_ports[:]
                random.shuffle(candidates)
                chosen: Optional[int] = None
                for port in candidates:
                    if port in taken:
                        continue
                    if _port_in_use(port):
                        continue
                    chosen = port
                    break

                if chosen is None:
                    raise PortAllocationError(
                        f"No free port found in range {low}-{high} for slot "
                        f"'{slot}' of worktree '{worktree_id}'"
                    )

                key = f"{worktree_id}{_KEY_SEP}{slot}"
                allocated[key] = chosen
                taken.add(chosen)
                result[slot] = chosen

            self._ports_file._save(allocated)

        return result

    def release(self, worktree_id: str) -> None:
        """Remove all port entries belonging to ``worktree_id``.

        Idempotent: a second call with the same id is a no-op.

        Parameters
        ----------
        worktree_id:
            The worktree id whose entries should be removed.
        """
        prefix = f"{worktree_id}{_KEY_SEP}"
        with portalocker.Lock(self._lock_path, timeout=_LOCK_TIMEOUT, flags=_LOCK_FLAGS):
            allocated: Dict[str, int] = self._ports_file._load()
            keys_to_remove = [k for k in allocated if k.startswith(prefix)]
            if not keys_to_remove:
                return
            for k in keys_to_remove:
                del allocated[k]
            self._ports_file._save(allocated)


class _NoOpPortAllocator:
    """Stub allocator used when the state store is not file-backed.

    Returns empty mappings; release is a no-op.  This avoids any file-system
    side effects in unit tests that use ``InMemoryStateStore``.
    """

    def allocate(self, slots: List[str], worktree_id: str) -> Dict[str, int]:  # noqa: ARG002
        return {}

    def release(self, worktree_id: str) -> None:  # noqa: ARG002
        return


__all__ = [
    "PortAllocationError",
    "PortAllocator",
]
