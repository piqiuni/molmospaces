from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(PACKAGE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SCRIPTS))
MLLM_SCRIPTS = PACKAGE_SCRIPTS.parents[1] / "semantic_mllm_py_pkg" / "scripts"
if str(MLLM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MLLM_SCRIPTS))

rospy = pytest.importorskip("rospy")

import semantic_mapping_node as semantic_mapping_module
from semantic_mapping_node import OccupancyGrid, SemanticMappingNode
from semantic_mapping_py_pkg.semantic_occ_overlay import SemanticOccupancyOverlay


class _Overlay:
    def __init__(self) -> None:
        self.pending = []

    def set_interaction_pending(self, node_id, pending):
        self.pending.append((node_id, pending))
        return True


class _RoomSegmenter:
    def __init__(self) -> None:
        self.calls = []

    def update_portal_hints(self, observations, source_mode, *, refresh_active):
        self.calls.append((observations, source_mode, refresh_active))
        return True


class _GraphStore:
    def update_interaction_result(self, result, stamp=None):
        self.result = dict(result)
        self.stamp = stamp
        return True

    @staticmethod
    def as_graph_dict():
        return {
            "nodes": [
                {
                    "id": "portal_door_1",
                    "type": "portal",
                    # This is deliberately the open door-leaf geometry.  The
                    # room cut must use the immutable reference instead.
                    "aabb_center": [9.0, 9.0, 1.0],
                    "aabb_size": [1.0, 1.0, 2.0],
                    "interaction": {"state": "open"},
                    "attributes": {
                        "instance_id": "door_instance",
                        "source_object_name": "door_source",
                        "interaction_reference_aabb_center": [2.0, 3.0, 1.0],
                        "interaction_reference_aabb_size": [0.2, 1.2, 2.0],
                    },
                }
            ]
        }


class _AlreadyOpenGraphStore(_GraphStore):
    """A duplicate evaluator result leaves the graph unchanged."""

    def update_interaction_result(self, result, stamp=None):
        self.result = dict(result)
        self.stamp = stamp
        return False


def test_successful_open_defers_room_refresh_until_after_direct_raw_publish(
    monkeypatch,
):
    monkeypatch.setattr(semantic_mapping_module.rospy, "loginfo", lambda *_args: None)
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node.pending_interaction_commands = {}
    node.semantic_occ_overlay = _Overlay()
    node.graph_store = _GraphStore()
    node.room_segmenter = _RoomSegmenter()
    node.room_post_open_force_refresh = True
    node.scene_store = _SceneStore()
    node.world_frame = "world"
    events = []
    node.planning_occupancy_grid_pub = SimpleNamespace(
        publish=lambda msg: events.append(("publish", msg))
    )
    node._refresh_room_grid_locked = lambda *, force_stable=False: events.append(
        ("room_refresh", force_stable)
    )
    bundle_calls = []
    node._collect_publish_bundle_locked = lambda: bundle_calls.append(True) or {}
    node._safe_publish_bundle = lambda _bundle: None

    node.interaction_result_callback(
        SimpleNamespace(
            data=json.dumps(
                {
                    "node_id": "portal_door_1",
                    "object_id": "door_instance",
                    "action": "open",
                    "success": True,
                    "stamp_sec": 12.0,
                }
            )
        )
    )

    assert node.semantic_occ_overlay.pending == [("portal_door_1", False)]
    # The interaction callback must leave the raw OCC queue unblocked.
    assert events == []
    assert bundle_calls == []
    assert node.room_segmenter.calls == []
    assert node._post_open_planning_refresh_after_stamp_sec == 12.0

    fresh_raw = _raw_occupancy(12.1)
    SemanticMappingNode.occupancy_callback(node, fresh_raw)

    # The planner receives the exact raw message before portal hints/room work.
    assert events == [("publish", fresh_raw), ("room_refresh", True)]
    assert node._post_open_room_refresh_result is None
    observations, source_mode, refresh_active = node.room_segmenter.calls[0]
    assert source_mode == "realtime_gt_observation"
    assert refresh_active is True
    assert observations == [
        {
            "id": "door_source",
            "name": "door",
            "is_door": True,
            "box_3d": {
                "center": [2.0, 3.0, 1.0],
                "size": [0.2, 1.2, 2.0],
            },
        }
    ]


def test_static_portal_result_does_not_arm_post_open_transition() -> None:
    static_result = {
        "action": "open",
        "success": True,
        "post_state": "static_open",
        # Deliberately omit interaction_capability: this is the compact bridge
        # result shape consumed by the semantic mapper.
        "source": "executor_static_portal",
    }
    physical_result = {
        "action": "open",
        "success": True,
        "post_state": "open",
        "interaction_capability": "articulated",
    }

    assert not SemanticMappingNode._is_successful_open_result(static_result)
    assert SemanticMappingNode._is_successful_open_result(physical_result)


class _SceneStore:
    def __init__(self) -> None:
        self.grids = []

    def initialize_from_occupancy_grid(self, grid) -> None:
        self.grids.append(grid)


def test_occupancy_callback_coalesces_room_work_without_waiting_for_worker():
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node._room_work_condition = threading.Condition()
    node._room_work_pending = None
    node._room_worker_stopping = False
    node._room_input_revision = 0
    node._room_epoch = 0
    node._post_open_planning_refresh_after_stamp_sec = None
    node._post_open_room_refresh_result = None
    node.scene_store = _SceneStore()
    node._timing_counts = {}
    node._timing_windows = {}
    node._timing_log_every = 1000

    started = threading.Event()
    release_first = threading.Event()
    processed = []

    def process(request):
        processed.append(request)
        if len(processed) == 1:
            started.set()
            assert release_first.wait(timeout=1.0)

    node._process_room_refresh_request = process
    worker = threading.Thread(target=node._room_worker_loop, daemon=True)
    node._room_worker_thread = worker
    worker.start()
    try:
        SemanticMappingNode.occupancy_callback(node, _raw_occupancy(1.0))
        assert started.wait(timeout=1.0)

        started_at = time.monotonic()
        SemanticMappingNode.occupancy_callback(node, _raw_occupancy(1.2))
        assert time.monotonic() - started_at < 0.25
        with node._room_work_condition:
            assert node._room_work_pending is not None
            assert node._room_work_pending["input_revision"] == 2
            assert node._room_work_pending["requested_source"]["step_index"] == 12
            assert node._room_work_pending["requested_source"]["stamp_sec"] == pytest.approx(1.2)

        release_first.set()
        deadline = time.monotonic() + 1.0
        while len(processed) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(processed) == 2
        assert processed[-1]["input_revision"] == 2
    finally:
        release_first.set()
        SemanticMappingNode._stop_room_worker(node)
        worker.join(timeout=1.0)


class _EpisodeGraphStore:
    episode_id = "old_episode"

    def reset(self, *, episode_id, source_mode) -> None:
        self.episode_id = episode_id
        self.reset_args = (episode_id, source_mode)

    @staticmethod
    def update_observations(*_args, **_kwargs) -> None:
        return None


class _Resettable:
    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        self.calls += 1


class _EpisodeRoomSegmenter:
    state = None

    @staticmethod
    def update_portal_hints(*_args, **_kwargs) -> bool:
        return False


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, msg) -> None:
        self.messages.append(msg)


def _raw_occupancy(stamp_sec: float):
    grid = OccupancyGrid()
    grid.header.seq = int(stamp_sec * 10)
    grid.header.stamp = rospy.Time.from_sec(stamp_sec)
    grid.header.frame_id = "raw_occ_frame"
    grid.info.resolution = 0.1
    grid.info.width = 2
    grid.info.height = 2
    grid.data = [0, 0, 100, -1]
    return grid


def test_room_segment_grid_compacts_stable_ids_for_int8_ros_message() -> None:
    node = object.__new__(SemanticMappingNode)
    raw = _raw_occupancy(10.0)

    # Stable IDs are unbounded across topology changes, while OccupancyGrid is
    # an int8 transport.  The graph retains these IDs; the display grid must
    # stay serializable instead of crashing the mapper once an ID reaches 128.
    grid = SemanticMappingNode._build_cropped_room_segment_grid(
        node,
        [1, 128, 130, -1],
        raw=raw,
    )

    assert list(grid.data) == [1, 2, 3, -1]
    assert all(-128 <= int(value) <= 127 for value in grid.data)


def test_post_open_occupancy_applies_portal_overlay_before_room_segmentation(
    monkeypatch,
):
    monkeypatch.setattr(semantic_mapping_module.rospy, "loginfo", lambda *_args: None)
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node.scene_store = _SceneStore()
    node._post_open_planning_refresh_after_stamp_sec = 12.0
    node.world_frame = "world"
    node.graph_store = SimpleNamespace(as_graph_dict=lambda: {"nodes": []})
    node.semantic_occ_overlay = object()
    node.ablation = SimpleNamespace(module1="full")
    monkeypatch.setattr(
        semantic_mapping_module,
        "apply_module1_ablation",
        lambda graph, _mode: graph,
    )
    publisher = _Publisher()
    node.planning_occupancy_grid_pub = publisher
    refresh_publish_counts = []
    node._refresh_room_grid_locked = lambda: refresh_publish_counts.append(
        len(publisher.messages)
    )

    # A queued map with an older source stamp cannot consume the refresh.
    SemanticMappingNode.occupancy_callback(node, _raw_occupancy(11.9))
    assert publisher.messages == []
    assert node._post_open_planning_refresh_after_stamp_sec == 12.0

    # The first source-new raw map publishes directly, before segmentation.
    fresh_raw = _raw_occupancy(12.1)
    effective = _raw_occupancy(12.1)
    effective.data = [0 for _ in fresh_raw.data]
    node._build_planning_products_from_snapshot = lambda raw, graph: (
        effective,
        None,
        None,
        {"active_portal_ids": ["door_1"], "cleared_cells": len(raw.data)},
    )
    SemanticMappingNode.occupancy_callback(node, fresh_raw)
    assert len(publisher.messages) == 1
    planning_grid = publisher.messages[0]
    assert planning_grid is effective
    assert planning_grid.header.seq == fresh_raw.header.seq
    assert planning_grid.header.stamp.to_sec() == fresh_raw.header.stamp.to_sec()
    assert planning_grid.header.frame_id == fresh_raw.header.frame_id
    assert list(planning_grid.data) == [0 for _ in fresh_raw.data]
    assert refresh_publish_counts[-1] == 1
    assert node._post_open_planning_refresh_after_stamp_sec is None

    # Later map frames return to timer-driven publishing and add no direct map.
    SemanticMappingNode.occupancy_callback(node, _raw_occupancy(12.2))
    assert len(publisher.messages) == 1


def test_successful_portal_open_arms_raw_bridge_when_graph_is_already_open(
    monkeypatch,
):
    monkeypatch.setattr(semantic_mapping_module.rospy, "loginfo", lambda *_args: None)
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node.pending_interaction_commands = {}
    node.semantic_occ_overlay = _Overlay()
    node.graph_store = _AlreadyOpenGraphStore()
    node.room_post_open_force_refresh = False
    node._collect_publish_bundle_locked = lambda: {}
    node._safe_publish_bundle = lambda _bundle: None

    # The opaque result lacks node_type.  Its matching, already-open portal
    # is the safe fallback, and graph-store ``changed=False`` must not lose
    # the one-shot post-open raw OCC bridge.
    node.interaction_result_callback(
        SimpleNamespace(
            data=json.dumps(
                {
                    "node_id": "portal_door_1",
                    "object_id": "door_instance",
                    "action": "open",
                    "success": True,
                    "stamp_sec": 12.0,
                }
            )
        )
    )

    assert node.graph_store.stamp == 12.0
    assert node._post_open_planning_refresh_after_stamp_sec == 12.0


def test_episode_reset_clears_pending_post_open_planning_refresh():
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node.graph_store = _EpisodeGraphStore()
    node.semantic_occ_overlay = _Resettable()
    node.semantic_occ_update_tracker = _Resettable()
    node.pending_interaction_commands = {"old": {}}
    node.object_store = SimpleNamespace(objects=[object()], next_id=7)
    node.room_segmenter = _EpisodeRoomSegmenter()
    node._post_open_planning_refresh_after_stamp_sec = 12.0
    node._save_episode_graph_locked = lambda **_kwargs: None
    node._refresh_room_grid_locked = lambda: None
    node._collect_publish_bundle_locked = lambda: {}
    node._safe_publish_bundle = lambda _bundle: None

    node.gt_observation_callback(
        SimpleNamespace(
            data=json.dumps(
                {
                    "episode_id": "new_episode",
                    "episode_reset": True,
                    "stamp_sec": 15.0,
                    "observations": [],
                }
            )
        )
    )

    assert node._post_open_planning_refresh_after_stamp_sec is None
    assert node.graph_store.reset_args == ("new_episode", "realtime_gt_observation")


def test_room_mllm_request_contains_only_room_and_member_object_metadata():
    node = object.__new__(SemanticMappingNode)
    node.room_mllm_enabled = True
    node.room_mllm_min_evidence_objects = 1
    node.ablation = SimpleNamespace(module1="dynamic_mllm")
    graph_payload = {
        "episode_id": "episode_1",
        "timestamp": 10.0,
        "graph_revision": 4,
        "capture_step": 21,
        "nodes": [
            {
                "id": "room_2",
                "type": "room",
                "room_id": 2,
                "attributes": {"active": True, "cell_count": 42},
            },
            {
                "id": "object_stove_1",
                "type": "object",
                "room_id": 2,
                "name": "Stove",
                "confidence": 0.9,
                "is_currently_visible": True,
                "aabb_center": [99.0, 99.0, 99.0],
                "attributes": {
                    "instance_id": "stove_1",
                    "category": "appliance",
                    "image": "must_not_leave_mapper",
                    "crop": "must_not_leave_mapper",
                },
            },
            {
                "id": "room_1000000",
                "type": "room",
                "room_id": 1000000,
                "attributes": {"active": True, "is_potential_room": True},
            },
        ],
    }

    request = node._build_room_attribute_request_locked(graph_payload)

    assert request == {
        "episode_id": "episode_1",
        "stamp_sec": 10.0,
        "graph_revision": 4,
        "capture_step": 21,
        "rooms": [
            {
                "room_id": 2,
                "room_node_id": "room_2",
                "objects": [
                    {
                        "object_id": "stove_1",
                        "node_id": "object_stove_1",
                        "name": "Stove",
                        "category": "appliance",
                        "type": "object",
                        "confidence": 0.9,
                        "currently_visible": True,
                    }
                ],
            }
        ],
    }
    serialized = json.dumps(request)
    for forbidden in ("image", "crop", "aabb", "geometry", "pose"):
        assert forbidden not in serialized


def test_strict_step_ready_requires_same_occ_room_and_graph_source():
    node = object.__new__(SemanticMappingNode)
    node.step_ready_require_room_graph = True
    node._room_epoch = 0
    raw = _raw_occupancy(20.0)
    stale_room = _raw_occupancy(19.8)
    base_bundle = {
        "raw_occupancy_grid": raw,
        "room_segment_grid": stale_room,
        "room_commit": {
            "source": {"step_index": raw.header.seq, "stamp_sec": 20.0},
            "epoch": 0,
            "graph_revision": 7,
            "episode_id": "episode_1",
        },
        "graph_payload": {
            "graph_revision": 7,
            "episode_id": "episode_1",
            "capture_step": 20,
            "timestamp": 20.0,
        },
    }

    waiting = node._semantic_mapping_ready_payload_for_bundle(base_bundle)
    assert not waiting["ready"]
    assert waiting["raw_occ_ready"]
    assert not waiting["room_segmentation_ready"]
    assert not waiting["unified_graph_ready"]
    assert waiting["missing_stages"] == ["room_segmentation", "unified_graph"]

    base_bundle["room_segment_grid"] = _raw_occupancy(20.0)
    base_bundle["graph_payload"]["graph_revision"] = 6
    graph_waiting = node._semantic_mapping_ready_payload_for_bundle(base_bundle)
    assert not graph_waiting["ready"]
    assert graph_waiting["room_segmentation_ready"]
    assert not graph_waiting["unified_graph_ready"]

    base_bundle["graph_payload"]["graph_revision"] = 7
    ready = node._semantic_mapping_ready_payload_for_bundle(base_bundle)
    assert ready["ready"]
    assert ready["room_segmentation_ready"]
    assert ready["unified_graph_ready"]
    assert ready["occupancy_source"] == ready["room_segmentation_source"]


def test_fast_step_ready_contract_remains_occ_only():
    node = object.__new__(SemanticMappingNode)
    node.step_ready_require_room_graph = False
    node._room_epoch = 0
    raw = _raw_occupancy(21.0)
    payload = node._semantic_mapping_ready_payload_for_bundle(
        {
            "raw_occupancy_grid": raw,
            "room_segment_grid": None,
            "room_commit": {},
            "graph_payload": {},
        }
    )
    assert payload["ready"]
    assert payload["causal_contract"] == "occ_only"


def test_publish_bundle_retains_raw_occ_source_for_strict_ready():
    node = object.__new__(SemanticMappingNode)
    node._build_planning_products_from_snapshot = lambda raw, _graph: (
        raw,
        None,
        None,
        {},
    )
    node._build_room_attribute_request_locked = lambda _graph: None
    raw = _raw_occupancy(22.0)
    bundle = node._build_publish_bundle_from_snapshot(
        {
            "scene_info": None,
            "scene_data": None,
            "scene_confidence_data": None,
            "scene_revision": 0,
            "room_segment_grid": None,
            "room_commit": {},
            "graph_payload": {},
            "raw_occupancy_grid": raw,
        }
    )
    assert bundle["raw_occupancy_grid"] is raw


def test_causal_ready_deduplicates_by_stamp_when_pipeline_resequences(
    monkeypatch,
):
    node = object.__new__(SemanticMappingNode)
    node.step_ready_require_room_graph = True
    node.lock = threading.RLock()
    node._last_causal_ready_source = None
    recorded = []
    logged = []
    node._record_component_timing = lambda kind, value: recorded.append((kind, value))
    monkeypatch.setattr(semantic_mapping_module.rospy, "loginfo", lambda *_args: logged.append(True))

    payload = {
        "ready": True,
        "occupancy_source": {"step_index": 75, "stamp_sec": 20.0},
        "room_segmentation_source": {"step_index": 2, "stamp_sec": 20.0},
        "published_graph_revision": 9,
        "published_graph_capture_step": 75,
        "room_commit_latency_ms": 3.0,
        "room_worker_total_ms": 2.5,
    }
    SemanticMappingNode._record_causal_ready_once(node, payload)
    payload["occupancy_source"] = {"step_index": 0, "stamp_sec": 20.0}
    SemanticMappingNode._record_causal_ready_once(node, payload)
    assert len(recorded) == 1
    assert len(logged) == 1

    payload["occupancy_source"] = {"step_index": 0, "stamp_sec": 20.2}
    SemanticMappingNode._record_causal_ready_once(node, payload)
    assert len(recorded) == 2
    assert len(logged) == 2


def test_room_topology_overlay_uses_raw_grid_without_confirmed_portal():
    node = object.__new__(SemanticMappingNode)
    node.room_segment_use_semantic_overlay = True
    node._room_segmentation_overlay = SemanticOccupancyOverlay()
    raw = _raw_occupancy(30.0)

    effective = SemanticMappingNode._room_segmentation_occupancy_from_snapshot(
        node, raw, {"nodes": []}
    )

    assert effective is raw


def test_exact_room_topology_cache_reuses_content_but_not_source():
    node = object.__new__(SemanticMappingNode)
    node.room_topology_cache_enabled = True
    node.room_grid_stability_frames = 1
    node.room_free_threshold = 20
    node._room_topology_cache = None
    node.room_segmenter = semantic_mapping_module.RoomSegmenter(
        room_min_component_cells=1,
        room_core_min_component_cells=1,
        room_core_clearance_cells=1,
        room_remove_enclosed_occupied=False,
        room_grid_stability_frames=1,
    )
    first = _raw_occupancy(31.0)
    room_ids, room_conf = node.room_segmenter.segment(first)
    room_merges = node.room_segmenter.consume_confirmed_merges()
    SemanticMappingNode._store_room_topology_cache(
        node,
        first,
        topology_revision=2,
        epoch=4,
        force_stable=False,
        room_ids=room_ids,
        room_conf=room_conf,
        room_merges=room_merges,
    )

    same_content_new_source = _raw_occupancy(31.2)
    cached, _key, _categories = SemanticMappingNode._load_room_topology_cache(
        node,
        same_content_new_source,
        topology_revision=2,
        epoch=4,
        force_stable=False,
    )
    assert cached is not None
    assert cached["room_ids"] == tuple(room_ids)

    changed = _raw_occupancy(31.4)
    changed.data[0] = 100
    cached, _key, _categories = SemanticMappingNode._load_room_topology_cache(
        node,
        changed,
        topology_revision=2,
        epoch=4,
        force_stable=False,
    )
    assert cached is None


def test_strict_room_commit_publishes_pinned_core_bundle_immediately(monkeypatch):
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node._publish_lock = threading.RLock()
    node.step_ready_require_room_graph = True
    raw = _raw_occupancy(32.0)
    room_grid = _raw_occupancy(32.0)
    commit = {
        "source": {"step_index": raw.header.seq, "stamp_sec": 32.0},
        "epoch": 0,
        "graph_revision": 7,
        "episode_id": "episode_1",
    }
    node._room_last_commit = dict(commit)
    node.graph_store = SimpleNamespace(
        as_graph_dict=lambda: {"graph_revision": 7, "episode_id": "episode_1"}
    )
    node.ablation = SimpleNamespace(module1="dynamic_rule")
    snapshots = []
    published = []
    timings = []
    node._build_publish_bundle_from_snapshot = lambda snapshot: snapshots.append(snapshot) or snapshot
    node._publish_bundle = lambda bundle: published.append(bundle)
    node._semantic_mapping_ready_payload_for_bundle = lambda _bundle: {"ready": True}
    node.step_ready_pub = _Publisher()
    node._record_causal_ready_once = lambda _payload: None
    node._record_component_timing = lambda key, value: timings.append((key, value))
    monkeypatch.setattr(
        semantic_mapping_module,
        "apply_module1_ablation",
        lambda payload, _module: payload,
    )

    assert SemanticMappingNode._publish_causal_room_commit(
        node, raw, room_grid, commit
    )
    assert published == snapshots
    assert snapshots[0]["raw_occupancy_grid"] is raw
    assert snapshots[0]["room_segment_grid"] is room_grid
    assert snapshots[0]["causal_core_only"] is True
    assert len(node.step_ready_pub.messages) == 1
    assert {key for key, _value in timings} == {
        "room_commit_publish_build",
        "room_commit_publish_ros",
        "room_commit_publish_total",
    }
