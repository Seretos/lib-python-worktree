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
from typing import Dict, List, Mapping, Optional

import portalocker

from .yaml_store import _LOCK_FLAGS, _LOCK_TIMEOUT, _PORT_KEY_SEP, _PortsFile, _port_in_use

# Alias of yaml_store._PORT_KEY_SEP -- yaml_store owns the canonical
# separator; this module only imports from yaml_store, never the reverse.
_KEY_SEP = _PORT_KEY_SEP


class PortAllocationError(RuntimeError):
    """Raised when no free port can be found for a requested slot."""


class PinnedPortUnavailableError(PortAllocationError):
    """Raised when a contract-pinned port cannot be claimed (ticket #120).

    Subclasses ``PortAllocationError`` so ``create()``'s rollback and every
    existing ``except PortAllocationError`` consumer keep working unchanged.
    Never falls back to a random port -- a pin is a hard requirement.

    Attributes
    ----------
    slot:
        The port-slot name that requested the pin.
    port:
        The pinned port number that could not be claimed.
    worktree_id:
        The worktree id the allocation call was made for.
    reason:
        ``"taken"`` if the port is already recorded in ``ports.yaml`` for a
        different worktree/slot key, or ``"in_use"`` if it is free in
        ``ports.yaml`` but the OS reports it busy (``_port_in_use()``).
    owner:
        The conflicting ``ports.yaml`` key (``"<worktree_id>:<slot>"``) when
        ``reason == "taken"``; ``None`` for an OS-level conflict.
    """

    def __init__(
        self,
        slot: str,
        port: int,
        worktree_id: str,
        reason: str,
        owner: Optional[str] = None,
    ) -> None:
        self.slot = slot
        self.port = port
        self.worktree_id = worktree_id
        self.reason = reason
        self.owner = owner
        super().__init__(
            f"Pinned port {port} for slot '{slot}' of worktree '{worktree_id}' "
            f"is unavailable ({reason})"
            + (f", owned by '{owner}'" if owner is not None else "")
        )


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

    def allocate(
        self,
        slots: List[str],
        worktree_id: str,
        *,
        pinned: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, int]:
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
        pinned:
            Optional mapping of slot name -> explicit port number (ticket
            #120). Any slot present here is claimed exactly at that number
            instead of being drawn at random from ``self._port_range`` --
            claiming happens in a first pass, before the remaining
            (unpinned) slots are auto-allocated in a second pass, so an auto
            slot can never randomly grab a number a pinned slot needs. A
            pinned port is exempt from the range check but still joins the
            ``taken`` set so pass two cannot collide with it. Re-claiming a
            pin this exact worktree/slot already holds is a no-op (skips
            both the ``taken`` check and the ``_port_in_use()`` probe) so a
            service of ours already listening on it is never read as a
            collision. Any other conflict -- the port already owned by a
            different worktree/slot, or reported busy by the OS -- raises
            ``PinnedPortUnavailableError`` and never falls back to a random
            port.

        Returns
        -------
        dict[slot_name, port_number]
            Empty dict when ``slots`` is empty (no lock acquired).

        Raises
        ------
        PortAllocationError
            If no free port exists for any requested unpinned slot.
        PinnedPortUnavailableError
            If a pinned slot's requested port cannot be claimed.
        """
        if not slots:
            return {}

        pinned = pinned or {}
        low, high = self._port_range
        all_ports = list(range(low, high + 1))

        with portalocker.Lock(self._lock_path, timeout=_LOCK_TIMEOUT, flags=_LOCK_FLAGS):
            allocated: Dict[str, int] = self._ports_file._load()

            # Own-key exclusion: a slot's own stale entry (about to be
            # (re)written by this very call) must not be treated as taken by
            # itself -- this is what makes both the idempotent self re-claim
            # and a start()-time re-pin work.
            own_keys = {f"{worktree_id}{_KEY_SEP}{s}" for s in slots}
            taken: set[int] = {p for k, p in allocated.items() if k not in own_keys}
            result: Dict[str, int] = {}

            # Pass 1: claim every slot present in `pinned`. Ordering is
            # load-bearing -- this must run before pass 2's auto-allocation,
            # or an auto slot could randomly grab a number a later slot pins.
            for slot in slots:
                if slot not in pinned:
                    continue
                port = pinned[slot]
                key = f"{worktree_id}{_KEY_SEP}{slot}"

                # Idempotent self re-claim: already holds exactly this pin.
                # Still must join `taken` -- own_keys excluded this slot's
                # stale entry from the initial `taken` set above, so without
                # this the port would never be recorded as taken during this
                # call and pass two could randomly hand it to a different
                # unpinned slot.
                if allocated.get(key) == port:
                    result[slot] = port
                    taken.add(port)
                    continue

                owner = next(
                    (k for k, p in allocated.items() if p == port and k != key),
                    None,
                )
                if port in taken:
                    raise PinnedPortUnavailableError(
                        slot, port, worktree_id, reason="taken", owner=owner
                    )
                if _port_in_use(port):
                    raise PinnedPortUnavailableError(
                        slot, port, worktree_id, reason="in_use", owner=None
                    )

                allocated[key] = port
                taken.add(port)
                result[slot] = port

            # Pass 2: auto-allocate the remaining (unpinned) slots exactly
            # as before.
            for slot in slots:
                if slot in pinned:
                    continue
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

        # Preserve input slot order in the returned mapping.
        return {slot: result[slot] for slot in slots}

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

    def allocate(
        self,
        slots: List[str],
        worktree_id: str,
        *,
        pinned: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, int]:  # noqa: ARG002
        return {}

    def release(self, worktree_id: str) -> None:  # noqa: ARG002
        return


__all__ = [
    "PinnedPortUnavailableError",
    "PortAllocationError",
    "PortAllocator",
]
