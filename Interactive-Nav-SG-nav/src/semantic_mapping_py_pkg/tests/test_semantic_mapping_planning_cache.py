"""Regression tests for semantic OCC planning-product caching.

These tests deliberately exercise the node method with a small ROS message
double.  A mapper frequently publishes a new header for an unchanged map;
that case must not rerun the portal rasterization or copy every cell.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPTS = PACKAGE_ROOT / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))
MLLM_SCRIPTS = PACKAGE_ROOT.parent / "semantic_mllm_py_pkg" / "scripts"
if str(MLLM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MLLM_SCRIPTS))

pytest.importorskip("rospy")

import semantic_mapping_node as semantic_mapping_module
from semantic_mapping_node import OccupancyGrid, SemanticMappingNode
from semantic_mapping_py_pkg.semantic_occ_overlay import (
    OverlayUpdateRegionTracker,
    SemanticOccupancyOverlay,
)
from semantic_mapping_py_pkg.room_segmentation import RoomSegmenter


class _CountingOverlay(SemanticOccupancyOverlay):
    def __init__(self):
        super().__init__(enabled=True, clear_padding_m=-0.05)
        self.apply_calls = 0

    def apply(self, *args, **kwargs):
        self.apply_calls += 1
        return super().apply(*args, **kwargs)


def _raw(stamp: int, *, changed_index: int | None = None) -> OccupancyGrid:
    grid = OccupancyGrid()
    grid.header.seq = stamp
    grid.header.stamp = stamp
    grid.header.frame_id = "map"
    grid.info.width = 20
    grid.info.height = 20
    grid.info.resolution = 0.1
    grid.info.origin.orientation.w = 1.0
    grid.data = [100] * 400
    if changed_index is not None:
        grid.data[changed_index] = 0
    return grid


def _node() -> tuple[SemanticMappingNode, _CountingOverlay]:
    node = object.__new__(SemanticMappingNode)
    overlay = _CountingOverlay()
    node._planning_overlay_lock = threading.RLock()
    node.semantic_occ_overlay = overlay
    node.semantic_occ_update_tracker = OverlayUpdateRegionTracker()
    node._planning_clear_mask_initialized = False
    node._planning_clear_mask_geometry_key = None
    node._planning_overlay_was_active = False
    node._planning_products_cache = None
    return node, overlay


_OPEN_PORTAL_GRAPH = {
    "nodes": [
        {
            "id": "door_1",
            "type": "portal",
            "interaction": {"state": "open"},
            "attributes": {
                "interaction_reference_aabb_center": [0.5, 0.5, 1.0],
                "interaction_reference_aabb_size": [0.2, 1.0, 2.0],
            },
        }
    ]
}


def test_header_only_map_update_reuses_materialized_overlay_payload():
    node, overlay = _node()
    first = node._build_planning_products_from_snapshot(
        _raw(1), _OPEN_PORTAL_GRAPH
    )
    second = node._build_planning_products_from_snapshot(
        _raw(2), _OPEN_PORTAL_GRAPH
    )

    assert overlay.apply_calls == 1
    assert first[0].data is second[0].data
    assert first[2].data is second[2].data
    assert first[1] is not None and second[1] is not None
    assert first[1].data is second[1].data
    assert second[0].header.seq == 2
    assert second[0].header.stamp == 2


def test_fresh_unchanged_map_avoids_rehashing_cached_content(monkeypatch):
    """A fresh ROS list should hit the C-level equality fast path.

    ``OccupancyGrid`` headers advance even when GMapping's cells do not.  The
    planning cache keeps the previous source list, so an equivalent fresh list
    must not allocate another NumPy buffer or run BLAKE2 on the full map.
    """

    node, _ = _node()
    digest_calls = 0
    original_blake2b = semantic_mapping_module.hashlib.blake2b

    def counting_blake2b(*args, **kwargs):
        nonlocal digest_calls
        digest_calls += 1
        return original_blake2b(*args, **kwargs)

    monkeypatch.setattr(
        semantic_mapping_module.hashlib, "blake2b", counting_blake2b
    )
    node._build_planning_products_from_snapshot(_raw(1), _OPEN_PORTAL_GRAPH)
    node._build_planning_products_from_snapshot(_raw(2), _OPEN_PORTAL_GRAPH)

    assert digest_calls == 1


def test_changed_cells_or_overlay_geometry_invalidate_cache():
    node, overlay = _node()
    first = node._build_planning_products_from_snapshot(
        _raw(1), _OPEN_PORTAL_GRAPH
    )
    changed = node._build_planning_products_from_snapshot(
        _raw(2, changed_index=0), _OPEN_PORTAL_GRAPH
    )
    assert overlay.apply_calls == 2
    assert first[0].data is not changed[0].data

    moved_graph = {
        "nodes": [
            {
                **_OPEN_PORTAL_GRAPH["nodes"][0],
                "id": "door_2",
                "attributes": {
                    "interaction_reference_aabb_center": [0.8, 0.5, 1.0],
                    "interaction_reference_aabb_size": [0.2, 1.0, 2.0],
                },
            }
        ]
    }
    node._build_planning_products_from_snapshot(_raw(3, changed_index=0), moved_graph)
    assert overlay.apply_calls == 3


def test_publish_signature_ignores_detector_only_graph_revision():
    """YOLO visibility refreshes must not force a full OCC publication."""

    node = object.__new__(SemanticMappingNode)
    node.latest_occupancy_grid = _raw(1)
    node._room_last_commit = {}
    node._scene_grid_revision = 0
    node.latest_room_segment_grid = None
    node.semantic_occ_overlay = SimpleNamespace(active_portal_ids=[])
    node._planning_overlay_was_active = False
    node.graph_store = SimpleNamespace(graph_revision=10)
    first = node._publish_input_signature_locked()
    node.graph_store.graph_revision = 11
    second = node._publish_input_signature_locked()
    assert second == first


def test_publish_snapshot_reuses_graph_payload_between_heartbeats():
    """The timer must not rebuild/deep-copy graph JSON on every tick."""

    class TrackingLock:
        def __init__(self):
            self.depth = 0

        def __enter__(self):
            self.depth += 1
            return self

        def __exit__(self, *_args):
            self.depth -= 1

        def locked(self):
            return self.depth > 0

    node = object.__new__(SemanticMappingNode)
    node.lock = TrackingLock()
    node.enable_scene_mapping = False
    node.enable_object_mapping = False
    node.ablation = SimpleNamespace(module1="dynamic_rule")
    node.graph_publish_period_sec = 60.0
    node._graph_payload_cache = None
    node._graph_payload_cache_token = None
    node._last_graph_publish_mono = time.monotonic()
    node.latest_occupancy_grid = None
    node.latest_room_segment_grid = None
    calls = []

    def graph_snapshot():
        calls.append(True)
        return {"graph_revision": 4, "nodes": [], "edges": []}

    node.graph_store = SimpleNamespace(graph_revision=4, as_graph_dict=graph_snapshot)

    first = SemanticMappingNode._snapshot_publish_inputs_locked(node)
    # Materialization is where the expensive deepcopy occurs.  The fake
    # ablation below asserts that it runs after the mapper lock is released.
    original_ablation = semantic_mapping_module.apply_module1_ablation
    ablation_lock_states = []

    def checking_ablation(graph, mode):
        ablation_lock_states.append(node.lock.locked())
        return original_ablation(graph, mode)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        semantic_mapping_module, "apply_module1_ablation", checking_ablation
    )
    try:
        SemanticMappingNode._materialize_graph_payload_from_snapshot(node, first)
    finally:
        monkeypatch.undo()
    second = SemanticMappingNode._snapshot_publish_inputs_locked(node)

    assert len(calls) == 1
    assert ablation_lock_states == [False]
    assert first["graph_payload_raw"] is not None
    assert second["graph_payload"] is node._graph_payload_cache
    assert first["graph_cache_token"] == second["graph_cache_token"]
    assert first["publish_graph_products"] is True
    assert second["publish_graph_products"] is False


def test_room_topology_key_reuses_ternary_categories_for_header_only_update():
    node = object.__new__(SemanticMappingNode)
    node.room_free_threshold = 20
    node.room_segmenter = RoomSegmenter(room_free_threshold=20)
    node._room_topology_input_cache = None

    first = _raw(1)
    second = _raw(2)
    key_first, categories_first = node._room_topology_cache_key(
        first, topology_revision=0, epoch=0
    )
    key_second, categories_second = node._room_topology_cache_key(
        second, topology_revision=0, epoch=0
    )

    assert key_first == key_second
    # The second call must reuse the cached native array instead of allocating
    # another int16/uint8 conversion for an unchanged OCC payload.
    assert categories_second is categories_first

    second.data[0] = 0
    key_changed, categories_changed = node._room_topology_cache_key(
        second, topology_revision=0, epoch=0
    )
    assert key_changed != key_second
    assert categories_changed is not categories_second


def test_room_topology_cache_invalidates_when_free_threshold_changes():
    node = object.__new__(SemanticMappingNode)
    node.room_free_threshold = 20
    node.room_segmenter = RoomSegmenter(room_free_threshold=20)
    node._room_topology_input_cache = None

    grid = _raw(1)
    first_key, first_categories = node._room_topology_cache_key(
        grid, topology_revision=0, epoch=0
    )
    node.room_free_threshold = 10
    node.room_segmenter.room_free_threshold = 10
    second_key, second_categories = node._room_topology_cache_key(
        grid, topology_revision=0, epoch=0
    )

    assert second_key != first_key
    assert second_categories is not first_categories


def test_room_grid_cache_reuses_encoded_payload_and_retimes_header_info(monkeypatch):
    """Header-only OCC updates must not re-encode the room raster."""

    node = object.__new__(SemanticMappingNode)
    node.world_frame = "map"
    room_ids = [-1] * 400
    room_ids[0] = 7
    first = node._build_room_segment_grid(room_ids, raw=_raw(1))

    second_raw = _raw(2)
    second_raw.info.map_load_time = 123.0

    def fail_asarray(*_args, **_kwargs):
        raise AssertionError("cache hit must not materialize room IDs")

    monkeypatch.setattr(
        semantic_mapping_module.np,
        "asarray",
        fail_asarray,
    )
    second = node._build_room_segment_grid(
        tuple(room_ids), raw=second_raw, encoded_data=first.data
    )

    assert second.data is first.data
    assert second.data == first.data
    assert second.header.seq == 2
    assert second.header.stamp == 2
    assert second.header.frame_id == first.header.frame_id == "map"
    assert second.info.width == first.info.width == 20
    assert second.info.height == first.info.height == 20
    assert second.info.resolution == first.info.resolution == 0.1
    assert second.info.map_load_time == 123.0
