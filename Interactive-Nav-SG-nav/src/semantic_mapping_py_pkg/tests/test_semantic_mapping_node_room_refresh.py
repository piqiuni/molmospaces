from __future__ import annotations

import json
import sys
import threading
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


class _SceneStore:
    def __init__(self) -> None:
        self.grids = []

    def initialize_from_occupancy_grid(self, grid) -> None:
        self.grids.append(grid)


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


def test_post_open_raw_occupancy_directly_publishes_before_room_segmentation(
    monkeypatch,
):
    monkeypatch.setattr(semantic_mapping_module.rospy, "loginfo", lambda *_args: None)
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node.scene_store = _SceneStore()
    node._post_open_planning_refresh_after_stamp_sec = 12.0
    node.world_frame = "world"
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
    SemanticMappingNode.occupancy_callback(node, fresh_raw)
    assert len(publisher.messages) == 1
    planning_grid = publisher.messages[0]
    assert planning_grid is fresh_raw
    assert planning_grid.header.seq == fresh_raw.header.seq
    assert planning_grid.header.stamp.to_sec() == fresh_raw.header.stamp.to_sec()
    assert planning_grid.header.frame_id == fresh_raw.header.frame_id
    assert list(planning_grid.data) == list(fresh_raw.data)
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
