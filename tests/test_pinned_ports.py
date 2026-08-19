"""Tests for ticket #120: explicit port pinning in the contract ``ports:``
block, focused on ``WorktreeManager.start()``'s lazy allocation call site
(R6, R7) and the end-to-end two-worktree collision scenario (R8).

Mirrors ``tests/test_start_setup_failed.py``'s pattern: helpers build a
``WorktreeRecord`` directly and add it to the state store, no real git
required. Unlike that module, R6/R7/R8 need a **real** ``PortAllocator``
(not the ``_NoOpPortAllocator`` stub used for ``InMemoryStateStore``) to
exercise pin-honouring behaviour, so these tests use a real
``YamlStateStore`` rooted at ``tmp_path`` (see
``WorktreeManager.__init__``'s allocator-selection comment).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lib_python_worktree.core.manager import ManagerConfig, WorktreeManager
from lib_python_worktree.core.port_allocator import PinnedPortUnavailableError
from lib_python_worktree.core.state import WorktreeRecord
from lib_python_worktree.core.yaml_store import YamlStateStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_mgr_yaml(tmp_path: Path) -> WorktreeManager:
    state = YamlStateStore(state_dir=tmp_path / "state")
    return WorktreeManager(
        config=ManagerConfig(store_root=tmp_path / "store"),
        state=state,
        reconcile_on_init=False,
    )


def _make_wt_record(wt_id: str, repo_root: Path, **kwargs) -> WorktreeRecord:
    defaults = dict(
        id=wt_id,
        repo_root=str(repo_root),
        branch="feature/x",
        path=str(repo_root / "worktrees" / wt_id),
    )
    defaults.update(kwargs)
    return WorktreeRecord(**defaults)


def _write_contract(repo_root: Path, text: str) -> None:
    p = repo_root / ".seretos" / "worktree-setup.yml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# R6 -- start()'s lazy allocation forwards pins for a missing slot
# ---------------------------------------------------------------------------


def test_start_lazy_allocation_honours_pin(tmp_path: Path):
    """Driving test: a record with no ports yet and a contract pinning one
    slot gets exactly that pinned port on start(), persisted to state."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nports:\n  - name: app\n    port: 31000\n",
    )

    mgr = _make_mgr_yaml(tmp_path)
    record = _make_wt_record("wt-r6", repo_root, ports={})
    mgr.state.add(record)

    with patch(
        "lib_python_worktree.core.port_allocator._port_in_use", return_value=False
    ):
        result = mgr.start(worktree_id="wt-r6")

    assert result.ports == {"app": 31000}
    reloaded = mgr.state.get("wt-r6")
    assert reloaded.ports == {"app": 31000}


def test_start_slot_already_matching_pin_causes_no_allocator_call(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nports:\n  - name: app\n    port: 31000\n",
    )

    mgr = _make_mgr_yaml(tmp_path)
    record = _make_wt_record("wt-r6b", repo_root, ports={"app": 31000})
    mgr.state.add(record)

    with patch.object(
        mgr._allocator, "allocate", wraps=mgr._allocator.allocate
    ) as spy_allocate:
        result = mgr.start(worktree_id="wt-r6b")

    assert result.ports == {"app": 31000}
    spy_allocate.assert_not_called()


def test_start_name_only_slot_alongside_pinned_slot_still_auto_allocates(
    tmp_path: Path,
):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nports:\n"
        "  - name: app\n    port: 31000\n"
        "  - name: api\n",
    )

    mgr = _make_mgr_yaml(tmp_path)
    record = _make_wt_record("wt-r6c", repo_root, ports={})
    mgr.state.add(record)

    with patch(
        "lib_python_worktree.core.port_allocator._port_in_use", return_value=False
    ):
        result = mgr.start(worktree_id="wt-r6c")

    assert result.ports["app"] == 31000
    assert isinstance(result.ports["api"], int)
    assert result.ports["api"] != 31000


# ---------------------------------------------------------------------------
# R7 -- start() treats the pin as authoritative on mismatch
# ---------------------------------------------------------------------------


def test_start_repins_mismatched_port(tmp_path: Path):
    """Driving test: a persisted port that disagrees with the contract's
    pin is overwritten in place, and the stale port is freed."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nports:\n  - name: app\n    port: 31000\n",
    )

    mgr = _make_mgr_yaml(tmp_path)
    record = _make_wt_record("wt-r7", repo_root, ports={"app": 39999})
    mgr.state.add(record)
    # Seed ports.yaml with the stale entry, mirroring what create() would
    # have written for the original (unpinned) auto allocation.
    mgr.state._ports.set_all({"wt-r7:app": 39999})

    with patch(
        "lib_python_worktree.core.port_allocator._port_in_use", return_value=False
    ):
        result = mgr.start(worktree_id="wt-r7")

    assert result.ports == {"app": 31000}
    reloaded = mgr.state.get("wt-r7")
    assert reloaded.ports == {"app": 31000}
    all_ports = mgr.state._ports.get_all()
    assert all_ports["wt-r7:app"] == 31000
    assert 39999 not in all_ports.values()


def test_start_repin_collision_with_other_worktree_raises_and_mutates_nothing(
    tmp_path: Path,
):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nports:\n  - name: app\n    port: 31000\n",
    )

    mgr = _make_mgr_yaml(tmp_path)
    record = _make_wt_record("wt-r7d", repo_root, ports={"app": 39999})
    mgr.state.add(record)
    mgr.state._ports.set_all({"wt-r7d:app": 39999, "wt-other:web": 31000})

    with patch(
        "lib_python_worktree.core.port_allocator._port_in_use", return_value=False
    ):
        with pytest.raises(PinnedPortUnavailableError):
            mgr.start(worktree_id="wt-r7d")

    reloaded = mgr.state.get("wt-r7d")
    assert reloaded.ports == {"app": 39999}
    all_ports = mgr.state._ports.get_all()
    assert all_ports == {"wt-r7d:app": 39999, "wt-other:web": 31000}


def test_start_repin_os_busy_raises_and_mutates_nothing(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nports:\n  - name: app\n    port: 31000\n",
    )

    mgr = _make_mgr_yaml(tmp_path)
    record = _make_wt_record("wt-r7e", repo_root, ports={"app": 39999})
    mgr.state.add(record)
    mgr.state._ports.set_all({"wt-r7e:app": 39999})

    with patch(
        "lib_python_worktree.core.port_allocator._port_in_use",
        side_effect=lambda p: p == 31000,
    ):
        with pytest.raises(PinnedPortUnavailableError) as excinfo:
            mgr.start(worktree_id="wt-r7e")
    assert excinfo.value.reason == "in_use"

    reloaded = mgr.state.get("wt-r7e")
    assert reloaded.ports == {"app": 39999}


def test_start_mismatched_pin_plus_matching_auto_slot_only_repins_pinned(
    tmp_path: Path,
):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nports:\n"
        "  - name: app\n    port: 31000\n"
        "  - name: api\n",
    )

    mgr = _make_mgr_yaml(tmp_path)
    record = _make_wt_record(
        "wt-r7f", repo_root, ports={"app": 39999, "api": 32500}
    )
    mgr.state.add(record)
    mgr.state._ports.set_all({"wt-r7f:app": 39999, "wt-r7f:api": 32500})

    with patch(
        "lib_python_worktree.core.port_allocator._port_in_use", return_value=False
    ):
        result = mgr.start(worktree_id="wt-r7f")

    assert result.ports["app"] == 31000
    assert result.ports["api"] == 32500


def test_start_removing_pin_leaves_existing_port_alone(tmp_path: Path):
    """A slot that used to be pinned but is now name:-only in the contract
    is not touched -- its persisted port already satisfies `missing`'s
    condition (present, and no pin to disagree with)."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(repo_root, "version: 1\nisolation: full\nports:\n  - name: app\n")

    mgr = _make_mgr_yaml(tmp_path)
    record = _make_wt_record("wt-r7g", repo_root, ports={"app": 39999})
    mgr.state.add(record)
    mgr.state._ports.set_all({"wt-r7g:app": 39999})

    with patch.object(
        mgr._allocator, "allocate", wraps=mgr._allocator.allocate
    ) as spy_allocate:
        result = mgr.start(worktree_id="wt-r7g")

    assert result.ports == {"app": 39999}
    spy_allocate.assert_not_called()


# ---------------------------------------------------------------------------
# R8 -- deterministic two-worktree collision (the ticket's headline unblocker)
# ---------------------------------------------------------------------------


def test_two_worktrees_pinning_same_port_collide(tmp_path: Path):
    """Driving test: create() for a second worktree pinning the same port
    as an already-created first worktree fails loudly with full rollback,
    naming the first worktree's key as the owner."""
    from unittest.mock import MagicMock

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_contract(
        repo_root,
        "version: 1\nisolation: full\nports:\n  - name: app\n    port: 31000\n",
    )

    mgr = _make_mgr_yaml(tmp_path)

    with (
        patch(
            "lib_python_worktree.core.manager._run_git",
            return_value=MagicMock(returncode=0, stdout=f"{repo_root}\n", stderr=""),
        ),
        patch(
            "lib_python_worktree.core.manager._load_contract",
        ) as mock_load,
        patch(
            "lib_python_worktree.core.port_allocator._port_in_use", return_value=False
        ),
    ):
        from lib_python_worktree.contract.schema import PortSlot, WorktreeContract

        mock_contract = MagicMock(spec=WorktreeContract)
        mock_contract.ports = [PortSlot(name="app", port=31000)]
        mock_contract.setup = []
        mock_load.return_value = mock_contract

        with patch.object(mgr, "_validate_repo", return_value=repo_root):
            with patch.object(mgr, "_branch_exists", return_value=True):
                with patch("lib_python_worktree.core.manager.Path.mkdir"):
                    (tmp_path / "store" / "repo").mkdir(parents=True, exist_ok=True)
                    record1 = mgr.create(str(repo_root), "feature/one")

                    assert record1.ports == {"app": 31000}

                    with pytest.raises(PinnedPortUnavailableError) as excinfo:
                        mgr.create(str(repo_root), "feature/two")

    assert excinfo.value.owner == f"{record1.id}:app"
    all_ports = mgr.state._ports.get_all()
    assert all_ports == {f"{record1.id}:app": 31000}

    ids = {r.id for r in mgr.state.list()}
    assert ids == {record1.id}


def test_two_worktrees_pinning_same_port_second_succeeds_after_first_removed(
    tmp_path: Path,
):
    """After release()-ing the first worktree's ports, a fresh create() with
    the same pin succeeds."""
    from unittest.mock import MagicMock

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    mgr = _make_mgr_yaml(tmp_path)

    with (
        patch(
            "lib_python_worktree.core.manager._run_git",
            return_value=MagicMock(returncode=0, stdout=f"{repo_root}\n", stderr=""),
        ),
        patch(
            "lib_python_worktree.core.manager._load_contract",
        ) as mock_load,
        patch(
            "lib_python_worktree.core.port_allocator._port_in_use", return_value=False
        ),
    ):
        from lib_python_worktree.contract.schema import PortSlot, WorktreeContract

        mock_contract = MagicMock(spec=WorktreeContract)
        mock_contract.ports = [PortSlot(name="app", port=31000)]
        mock_contract.setup = []
        mock_load.return_value = mock_contract

        with patch.object(mgr, "_validate_repo", return_value=repo_root):
            with patch.object(mgr, "_branch_exists", return_value=True):
                with patch("lib_python_worktree.core.manager.Path.mkdir"):
                    (tmp_path / "store" / "repo").mkdir(parents=True, exist_ok=True)
                    record1 = mgr.create(str(repo_root), "feature/one")
                    mgr._allocator.release(record1.id)
                    mgr.state.remove(record1.id)

                    record2 = mgr.create(str(repo_root), "feature/two")

    assert record2.ports == {"app": 31000}
