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
from semantic_mapping_py_pkg.semantic_map_store import ObjectMapStore
from semantic_mapping_py_pkg.semantic_occ_overlay import SemanticOccupancyOverlay


class _Overlay:
    def __init__(self) -> None:
        self.pending = []

    def set_interaction_pending(self, node_id, pending):
        self.pending.append((node_id, pending))
        return True


@pytest.mark.parametrize("time_fields", [
    {"secs": 10, "nsecs": 0}, {"stamp": 10.0},
    {"stamp_sec": 10.0}, {"capture_stamp_sec": 10.0}, {"stamp": "invalid"},
])
def test_replayed_detector_capture_does_not_reach_graph_or_publish(time_fields):
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node.enable_object_mapping = True
    node.object_store = ObjectMapStore()
    det = {"semantic_class": "door", "world_position": [1, 2, 1],
           "world_box3d_size": [1, 0.1, 2], "confidence": 0.9}
    node.object_store.update([det], stamp=10.0)
    node.object_store.as_tracked_detections = lambda **kwargs: pytest.fail("duplicate reached graph preparation")
    node._process_object_message(SimpleNamespace(data=json.dumps({
        **time_fields, "detections": [det],
    })))
    assert node.object_store.objects[0]["observation_count"] == 1


@pytest.mark.parametrize("conflict", ["node_id", "object_id", "action", "episode_id", "missing_id", "unknown_id"])
def test_physical_result_guard_protects_graph_and_pending_overlay(monkeypatch, conflict):
    monkeypatch.setattr(semantic_mapping_module.rospy, "logwarn_throttle", lambda *args: None)
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node.require_interaction_command_id = True
    command = {"command_id": "cmd1", "episode_id": "e1", "node_id": "fridge1",
               "object_id": "track1", "action": "close"}
    node.pending_interaction_commands = {"cmd1": dict(command)}
    updates, overlays, bundles = [], [], []
    node.graph_store = SimpleNamespace(episode_id="e1", update_interaction_result=lambda result, stamp: updates.append(result) or True)
    node._set_planning_interaction_pending = lambda *args: overlays.append(args) or True
    node._collect_publish_bundle = lambda: {}
    node._safe_publish_bundle = bundles.append
    result = {**command, "success": True, "post_state": "closed", "stamp_sec": 12.0}
    invalid = dict(result)
    if conflict == "missing_id":
        invalid.pop("command_id")
    elif conflict == "unknown_id":
        invalid["command_id"] = "other-command"
    else:
        invalid[conflict] = "wrong"
    node.interaction_result_callback(SimpleNamespace(data=json.dumps(invalid)))
    assert node.pending_interaction_commands == {"cmd1": command}
    assert not updates and not overlays and not bundles
    node.interaction_result_callback(SimpleNamespace(data=json.dumps(result)))
    node.interaction_result_callback(SimpleNamespace(data=json.dumps(result)))
    assert len(updates) == len(overlays) == len(bundles) == 1
    assert updates[0]["node_id"] == "fridge1"
    assert not node.pending_interaction_commands


def test_foreign_episode_command_cannot_arm_planning_overlay():
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node.graph_store = SimpleNamespace(episode_id="new")
    node.pending_interaction_commands = {}
    node.interaction_command_callback(SimpleNamespace(data=json.dumps({
        "command_id": "old-command", "episode_id": "old", "action": "open", "node_id": "door1",
    })))
    assert not node.pending_interaction_commands


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


class _GraphObservationRecorder:
    def __init__(self) -> None:
        self.observations = []

    def has_confirmed_open_refrigerator(self):
        return True

    def update_observations(self, observations, *, stamp=None, source_mode=None):
        self.observations.append(
            {
                "observations": list(observations),
                "stamp": stamp,
                "source_mode": source_mode,
            }
        )

    def prune_stale_nodes(self, _stale_after_sec, *, now=None):
        return None


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


@pytest.mark.parametrize("reset_after_snapshot", [False, True])
def test_open_refrigerator_admits_one_frame_content_to_graph_only(monkeypatch, reset_after_snapshot):
    """The strict M1 topic stays empty while the graph receives the item."""

    monkeypatch.setattr(
        semantic_mapping_module.rospy,
        "loginfo_throttle",
        lambda *_args: None,
        raising=False,
    )
    node = object.__new__(SemanticMappingNode)
    node.enable_object_mapping = True
    node.tracked_stream_epoch = "before-reset"
    node.lock = threading.RLock()
    node.object_store = ObjectMapStore(
        match_distance=0.5,
        min_confirmations=3,
        class_min_confirmations={"bottle": 3},
    )
    node.graph_min_observations = 3
    node.open_refrigerator_content_mapping_enabled = True
    node.open_refrigerator_content_min_observations = 1
    node.open_refrigerator_content_ignore_class_confirmations = True
    node.object_stale_after_sec = 0.0
    node.graph_store = _GraphObservationRecorder()
    published = []
    node.tracked_detections_pub = SimpleNamespace(
        publish=lambda message: published.append(message)
    )
    node._update_room_portal_hints = lambda *_args, **_kwargs: False
    node._collect_publish_bundle = lambda: {}
    node._safe_publish_bundle = lambda _bundle: None
    timings = []
    def record_timing(key, value):
        timings.append((key, value))
        if reset_after_snapshot:
            with node.lock:
                node.tracked_stream_epoch = "after-reset"

    node._record_component_timing = record_timing

    SemanticMappingNode.object_callback(
        node,
        SimpleNamespace(
            data=json.dumps(
                {
                    "secs": 1,
                    "nsecs": 0,
                    "detections": [
                        {
                            "semantic_class": "bottle",
                            "confidence": 0.8,
                            "world_position": {"x": 1.0, "y": 2.0, "z": 0.8},
                            "world_box3d_center": {
                                "x": 1.0,
                                "y": 2.0,
                                "z": 0.8,
                            },
                            "world_box3d_size": {
                                "x": 0.08,
                                "y": 0.08,
                                "z": 0.24,
                            },
                        }
                    ],
                }
            )
        ),
    )

    assert len(published) == 1
    assert json.loads(published[0].data)["episode_id"] == "before-reset"
    assert json.loads(published[0].data)["detections"] == []
    assert len(node.graph_store.observations) == 1
    admitted = node.graph_store.observations[0]["observations"]
    assert len(admitted) == 1
    assert admitted[0]["semantic_name"] == "bottle"
    assert admitted[0]["tracking_confirmed"] is False
    assert admitted[0]["graph_admission_source"] == "open_refrigerator_exposure"
    timing_keys = {key for key, _value in timings}
    assert {
        "object_store_update",
        "object_store_as_tracked_detections",
        "object_store_total",
        "graph_update_observations",
        "graph_prune_unqualified_rooms",
        "graph_ensure_provisional_rooms",
        "graph_prune_stale_nodes",
        "graph_prune_ensure_total",
    } <= timing_keys
    assert all(value >= 0.0 for _key, value in timings)


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


def test_source_portal_topology_survives_delayed_m1_type_demotion() -> None:
    """A mutable M1 type must not suppress a source-qualified door result."""

    assert SemanticMappingNode._node_is_topology_portal(
        {
            "type": "object",
            "attributes": {"topology_type": "portal"},
        }
    )
    assert not SemanticMappingNode._node_is_topology_portal(
        {
            "type": "portal",
            "attributes": {"topology_type": "container"},
        }
    )
    # Legacy/replay payloads without provenance retain their public-type
    # behavior for backwards compatibility.
    assert SemanticMappingNode._node_is_topology_portal({"type": "portal"})


class _SceneStore:
    def __init__(self) -> None:
        self.grids = []

    def initialize_from_occupancy_grid(self, grid) -> None:
        self.grids.append(grid)


def test_occ_receipt_time_includes_lock_wait_and_source_age_is_measured_at_entry(monkeypatch):
    node = object.__new__(SemanticMappingNode)
    clock = [10.0]

    class DelayedLock:
        def __enter__(self):
            clock[0] += 0.4

        def __exit__(self, *args):
            return False

    node.lock = DelayedLock()
    node._post_open_planning_refresh_after_stamp_sec = None
    node.scene_store = _SceneStore()
    node._enqueue_room_refresh = lambda **kwargs: None
    samples = {}
    node._record_component_timing = lambda key, value: samples.update({key: value})
    monkeypatch.setattr(semantic_mapping_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(semantic_mapping_module.rospy, "get_time", lambda: 990. + clock[0])
    node.occupancy_callback(_raw_occupancy(999.8))
    assert node._latest_occupancy_received_mono_s == 10.0
    assert samples["occupancy_source_age_at_callback"] == pytest.approx(200.)


@pytest.mark.parametrize("stamp,now,expected", [
    (1., 1.2, 200.), (1., 1., 0.), (0., 1., None),
    (2., 1., None), (1., float("nan"), None), (1., float("inf"), None),
])
def test_occ_source_age_does_not_report_invalid_clock_as_zero(stamp, now, expected):
    actual = SemanticMappingNode._occupancy_source_age_ms(_raw_occupancy(stamp), now)
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


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


def test_room_refresh_retimes_same_ternary_map_but_rejects_changed_content():
    """A slow room job may follow headers, never a changed OCC topology."""

    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node._room_epoch = 0
    node._room_topology_revision = 0
    node._room_input_revision = 2
    node.room_free_threshold = 20
    node.world_frame = "world"

    old_raw = _raw_occupancy(40.0)
    # Free-cell confidence changes do not change room segmentation's ternary
    # input.  The newer source must nevertheless own the emitted header.
    latest_same = _raw_occupancy(40.2)
    latest_same.data = [10, 20, 100, -1]
    node.latest_occupancy_grid = latest_same
    room_grid = _raw_occupancy(40.0)
    key = SemanticMappingNode._room_occupancy_content_key(node, old_raw)

    resolved_raw, resolved_grid, revision, status = (
        SemanticMappingNode._resolve_room_refresh_commit_source(
            node,
            old_raw,
            room_grid,
            1,
            key,
            epoch=0,
            topology_revision=0,
        )
    )
    assert status == "retimed"
    assert resolved_raw is latest_same
    assert revision == 2
    assert resolved_grid is not room_grid
    assert resolved_grid.data is room_grid.data
    assert resolved_grid.header.stamp.to_sec() == pytest.approx(40.2)

    # A changed unknown/free/occupied category invalidates the result.  The
    # caller must enqueue the latest source instead of publishing old labels.
    latest_changed = _raw_occupancy(40.3)
    latest_changed.data[0] = 100
    node.latest_occupancy_grid = latest_changed
    node._room_input_revision = 3
    resolved = SemanticMappingNode._resolve_room_refresh_commit_source(
        node,
        old_raw,
        room_grid,
        1,
        key,
        epoch=0,
        topology_revision=0,
    )
    assert resolved[3] == "content_changed"
    assert resolved[0] is None


def test_room_refresh_content_supersession_is_urgent_and_latest_only():
    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node._room_work_condition = threading.Condition()
    node._room_worker_thread = object()
    node._room_worker_stopping = False
    node._room_work_pending = None
    node._room_epoch = 0
    node._room_input_revision = 7
    node._room_topology_revision = 0
    node.latest_occupancy_grid = _raw_occupancy(41.0)

    assert SemanticMappingNode._enqueue_room_refresh(
        node,
        reason="room_content_superseded",
        epoch=0,
        urgent=True,
    )
    assert node._room_work_pending["urgent"] is True

    # A later ordinary OCC enqueue coalesces into the same request and must
    # not clear the immediate retry flag.
    node._room_input_revision = 8
    node.latest_occupancy_grid = _raw_occupancy(41.1)
    SemanticMappingNode._enqueue_room_refresh(node, reason="occupancy", epoch=0)
    assert node._room_work_pending["urgent"] is True
    assert node._room_work_pending["requested_source"]["stamp_sec"] == pytest.approx(41.1)


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
    grid = SemanticMappingNode._build_room_segment_grid(
        node,
        [1, 128, 130, -1],
        raw=raw,
    )

    assert list(grid.data) == [1, 2, 3, -1]
    assert grid.info.width == raw.info.width
    assert grid.info.height == raw.info.height
    assert grid.info.origin.position.x == raw.info.origin.position.x
    assert grid.info.origin.position.y == raw.info.origin.position.y
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
    node.object_store = ObjectMapStore()
    node.object_store.m1_canonical_labels["track_0001"] = "refrigerator"
    node.tracked_stream_epoch = "old-stream"
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
    assert not node.object_store.m1_canonical_labels
    assert node.tracked_stream_epoch != "old-stream"


def test_m1_tracker_name_changes_only_after_graph_accepts_confirmation():
    from semantic_mapping_py_pkg.interaction_graph_store import InteractionGraphStore

    node = object.__new__(SemanticMappingNode)
    node.lock = threading.RLock()
    node.ablation = SimpleNamespace(module1="dynamic_mllm")
    node.tracked_stream_epoch = "stream-current"
    node.object_store = ObjectMapStore()
    node.graph_store = InteractionGraphStore(scene_id="offline")
    node.graph_store.episode_id = "run"
    node._safe_publish_bundle = lambda bundle: None
    node._collect_publish_bundle = lambda: {}
    observation = {"instance_id": "track_0001", "semantic_name": "locker", "category": "locker",
                   "confidence": 0.9, "position": [1.0, 2.0, 1.0],
                   "aabb_center": [1.0, 2.0, 1.0], "aabb_size": [0.7, 0.7, 1.8]}
    for stamp in (1.0, 2.0):
        node.graph_store.update_observations([observation], stamp=stamp, source_mode="detector_online")
    patch = {"object_id": "object_track_0001", "attribute_status": "ready",
             "source": "mllm_attribute_inference", "confidence": 0.9,
             "interaction_class": "container", "interactable": True, "coarse_state": "closed",
             "observed_object_name": "refrigerator"}

    def send(sequence, epoch="stream-current", **changes):
        node.attribute_updates_callback(SimpleNamespace(data=json.dumps({
            "episode_id": epoch, "stamp_sec": float(sequence + 2),
            "updates": [{**patch, "request_sequence": sequence,
                         "observation_signature": f"view-{sequence}",
                         "observation_frame_index": sequence + 2, **changes}],
        })))

    send(1)
    assert not node.object_store.m1_canonical_labels  # First fridge proposal is still pending.
    send(2)
    assert node.object_store.m1_canonical_labels == {"track_0001": "refrigerator"}
    send(1, observed_object_name="water_dispenser")  # Stale sequence cannot rename the tracker.
    send(3, epoch="stream-old", observed_object_name="water_dispenser")
    assert node.object_store.m1_canonical_labels == {"track_0001": "refrigerator"}
    send(3, observed_object_name="water_dispenser", interaction_class="none", interactable=False)
    assert node.object_store.m1_canonical_labels == {"track_0001": "water_dispenser"}


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


def test_room_topology_overlay_reuses_unchanged_materialized_cells():
    node = object.__new__(SemanticMappingNode)
    node.room_segment_use_semantic_overlay = True
    node._room_segmentation_overlay = SemanticOccupancyOverlay()
    node._room_overlay_cache = None
    raw = _raw_occupancy(30.0)
    graph = {
        "nodes": [
            {
                "id": "portal_door_1",
                "type": "portal",
                "interaction": {"state": "open"},
                "attributes": {
                    "interaction_reference_aabb_center": [0.05, 0.05, 0.0],
                    "interaction_reference_aabb_size": [0.1, 0.1, 1.0],
                },
            }
        ]
    }
    overlay = node._room_segmentation_overlay
    original_apply = overlay.apply
    apply_calls = []

    def counted_apply(*args, **kwargs):
        apply_calls.append(True)
        return original_apply(*args, **kwargs)

    overlay.apply = counted_apply
    first = SemanticMappingNode._room_segmentation_occupancy_from_snapshot(
        node, raw, graph
    )
    same_content_new_source = _raw_occupancy(30.2)
    second = SemanticMappingNode._room_segmentation_occupancy_from_snapshot(
        node, same_content_new_source, graph
    )

    assert len(apply_calls) == 1
    assert first.data is second.data
    assert second.header.stamp == same_content_new_source.header.stamp

    # A few in-process bridges reuse and mutate the message buffer.  The
    # cache owns its source list, so this must invalidate rather than reuse a
    # stale effective raster.
    same_content_new_source.data[0] = 100
    SemanticMappingNode._room_segmentation_occupancy_from_snapshot(
        node, same_content_new_source, graph
    )
    assert len(apply_calls) == 2

    changed = _raw_occupancy(30.4)
    changed.data[0] = 100
    SemanticMappingNode._room_segmentation_occupancy_from_snapshot(
        node, changed, graph
    )
    assert len(apply_calls) == 2


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


def test_collect_and_publish_keeps_concurrent_snapshots_in_causal_order():
    node = object.__new__(SemanticMappingNode)
    node._publish_lock = threading.RLock()
    first_collect_started = threading.Event()
    release_first_collect = threading.Event()
    second_collect_started = threading.Event()
    revision = {"value": 1}
    published = []

    def collect():
        value = revision["value"]
        if value == 1:
            first_collect_started.set()
            assert release_first_collect.wait(2.0)
        else:
            second_collect_started.set()
        return {"revision": value}

    node._collect_publish_bundle = collect
    node._publish_bundle = published.append

    first = threading.Thread(target=node._collect_and_publish_bundle)
    first.start()
    assert first_collect_started.wait(1.0)

    revision["value"] = 2
    second = threading.Thread(target=node._collect_and_publish_bundle)
    second.start()
    # Snapshot construction itself is behind the publication lock.  The
    # second callback cannot collect revision 2 and publish it before the
    # already-started revision-1 callback completes.
    assert not second_collect_started.wait(0.05)

    release_first_collect.set()
    first.join(2.0)
    second.join(2.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert published == [{"revision": 1}, {"revision": 2}]
