import inspect
import math
import queue
import threading

import numpy as np

from molmo_spaces.policy.learned_policy.ros_bridge_policy import RosBridgePolicy


def test_odom_twist_is_disabled_by_default() -> None:
    parameter = inspect.signature(RosBridgePolicy.__init__).parameters["publish_odom_twist"]

    assert parameter.default is False


def test_pointcloud_defaults_to_full_resolution() -> None:
    parameter = inspect.signature(RosBridgePolicy.__init__).parameters["pointcloud_stride"]

    assert parameter.default == 1


def test_estimate_planar_twist_uses_body_frame() -> None:
    previous = np.array([1.0, 2.0, math.pi / 2.0], dtype=np.float32)
    current = np.array([1.0, 2.2, math.pi / 2.0], dtype=np.float32)

    vx, vy, wz = RosBridgePolicy._estimate_planar_twist(previous, current, 0.2)

    assert math.isclose(vx, 1.0, abs_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-6)
    assert math.isclose(wz, 0.0, abs_tol=1e-6)


def test_estimate_planar_twist_wraps_yaw_delta() -> None:
    previous = np.array([0.0, 0.0, math.pi - 0.05], dtype=np.float32)
    current = np.array([0.0, 0.0, -math.pi + 0.05], dtype=np.float32)

    vx, vy, wz = RosBridgePolicy._estimate_planar_twist(previous, current, 0.2)

    assert math.isclose(vx, 0.0, abs_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-6)
    assert math.isclose(wz, 0.5, abs_tol=1e-5)


def test_extract_planar_twist_uses_instantaneous_base_qvel() -> None:
    base_group = type("BaseGroup", (), {"joint_vel": np.array([0.0, 1.0, 0.4])})()
    robot_view = type(
        "RobotView",
        (),
        {"get_move_group": lambda self, name: base_group},
    )()
    robot = type("Robot", (), {"robot_view": robot_view})()
    env = type("Env", (), {"current_robot": robot})()
    task = type("Task", (), {"env": env})()
    policy = RosBridgePolicy.__new__(RosBridgePolicy)
    policy.task = task

    twist = policy._extract_planar_twist_from_task(math.pi / 2.0)

    assert twist is not None
    vx, vy, wz = twist
    assert math.isclose(vx, 1.0, abs_tol=1e-6)
    assert math.isclose(vy, 0.0, abs_tol=1e-6)
    assert math.isclose(wz, 0.4, abs_tol=1e-6)


def test_publish_realtime_gt_now_forces_current_snapshot() -> None:
    calls = []

    class Publisher:
        def publish(self, task, *, stamp, step_index, force):
            calls.append((task, stamp, step_index, force))
            return {"frame_index": step_index}

    policy = RosBridgePolicy.__new__(RosBridgePolicy)
    policy.task = object()
    policy._step_idx = 7
    policy._realtime_gt_publisher = Publisher()
    policy._latest_gt_payload = None
    policy._next_common_stamp = lambda: "stamp"

    payload = policy.publish_realtime_gt_now(step_index=11)

    assert payload == {"frame_index": 11}
    assert policy._latest_gt_payload == payload
    assert calls == [(policy.task, "stamp", 11, True)]


def test_external_public_payload_is_consumed_by_next_rgb_frame_only() -> None:
    class Stamp:
        def to_sec(self):
            return 12.5

    class Image:
        width = 2
        height = 1
        data = bytes([0, 1, 2, 3, 4, 5])

    policy = RosBridgePolicy.__new__(RosBridgePolicy)
    policy._lock = threading.Lock()
    policy._step_frame_thread = object()
    policy._step_frame_queue = queue.Queue()
    policy._latest_gt_payload = {"schema_version": "legacy"}
    policy._pending_step_frame_public_payload = None
    published = {
        "schema_version": "semantic_minimal_gt_v1",
        "episode_id": "episode-1",
        "capture_step": 0,
        "stamp_sec": 12.4,
        "observations": [{"id": "obj_000001", "name": "door"}],
    }

    assert policy.queue_step_frame_public_payload(published)
    # The bridge owns a snapshot, rather than a mutable reference to evaluator
    # state that can be changed before the writer drains its queue.
    published["observations"].append({"id": "obj_000002", "name": "drawer"})
    policy._enqueue_step_frame(None, Stamp(), 0)
    policy._enqueue_step_frame(Image(), Stamp(), 0)
    policy._enqueue_step_frame(Image(), Stamp(), 1)
    assert policy.queue_step_frame_public_payload(
        {
            "schema_version": "semantic_minimal_gt_v1",
            "episode_id": "episode-1",
            "capture_step": 1,
            "stamp_sec": 12.5,
            "observations": [],
        }
    )
    policy._enqueue_step_frame(Image(), Stamp(), 2)

    first = policy._step_frame_queue.get_nowait()
    second = policy._step_frame_queue.get_nowait()
    third = policy._step_frame_queue.get_nowait()
    assert first[-1] == {
        "schema_version": "semantic_minimal_gt_v1",
        "episode_id": "episode-1",
        "capture_step": 0,
        "stamp_sec": 12.4,
        "observations": [{"id": "obj_000001", "name": "door"}],
    }
    assert second[-1] == {"schema_version": "legacy"}
    assert third[-1]["observations"] == []


def test_missing_navigation_arm_actions_hold_the_current_reset_pose() -> None:
    left = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    right = np.array([-0.1, -0.2, -0.3], dtype=np.float32)
    base = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    class RobotView:
        def move_group_ids(self):
            return ["base", "left_arm", "right_arm"]

        def get_noop_ctrl_dict(self, names):
            values = {"base": base, "left_arm": left, "right_arm": right}
            return {name: values[name] for name in names}

    action = {"done": False}
    RosBridgePolicy._fill_missing_navigation_holds(action, RobotView())

    np.testing.assert_array_equal(action["base"], base)
    np.testing.assert_array_equal(action["left_arm"], left)
    np.testing.assert_array_equal(action["right_arm"], right)
    assert action["left_arm"] is not left
    assert action["right_arm"] is not right


def test_tf_keepalive_republishes_only_cached_pose_transforms() -> None:
    calls = []
    policy = RosBridgePolicy.__new__(RosBridgePolicy)
    policy.publish_odom = True
    policy.base_frame_id = "base"
    policy.pointcloud_frame_id = "lidar"
    policy._tf_cache_lock = threading.Lock()
    policy._latest_odom_tf_state = tuple(float(value) for value in range(10))
    policy._latest_base_to_lidar_tf = tuple(float(value) for value in range(7))
    policy._next_common_stamp = lambda: "fresh-stamp"
    policy._publish_odom_and_base_tf_from_state = lambda state, stamp: calls.append(("odom", state, stamp))
    policy._publish_base_to_lidar_tf_from_state = lambda state, stamp: calls.append(("lidar", state, stamp))

    policy._tf_keepalive_callback(None)

    assert calls == [
        ("odom", tuple(float(value) for value in range(10)), "fresh-stamp"),
        ("lidar", tuple(float(value) for value in range(7)), "fresh-stamp"),
    ]


def test_cmd_vel_action_uses_fixed_policy_dt_not_wall_clock() -> None:
    policy = RosBridgePolicy.__new__(RosBridgePolicy)
    policy.cmd_vel_linear_gain = 1.0
    policy.cmd_vel_control_dt_s = 0.2
    policy._extract_base_pose_from_observation = lambda _observation: np.array(
        [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )

    action = policy._cmd_vel_to_base_action(
        np.array([0.5, 0.0, 1.25], dtype=np.float32), object()
    )

    assert action is not None
    np.testing.assert_allclose(action["base"], [1.1, 2.0, 0.25], atol=1e-6)


def test_step_sync_reports_action_source_and_fixed_control_dt() -> None:
    import json

    published = []

    class Publisher:
        def publish(self, message):
            published.append(message)

    class StringMessage:
        def __init__(self, *, data):
            self.data = data

    class Stamp:
        def to_sec(self):
            return 12.5

    policy = RosBridgePolicy.__new__(RosBridgePolicy)
    policy._step_sync_pub = Publisher()
    policy._String = StringMessage
    policy._step_idx = 7
    policy.last_action_source = "cmd_vel"
    policy.cmd_vel_control_dt_s = 0.2

    policy._publish_step_sync(Stamp())

    payload = json.loads(published[0].data)
    assert payload == {
        "step_index": 7,
        "stamp_sec": 12.5,
        "action_source": "cmd_vel",
        "cmd_vel_control_dt_s": 0.2,
    }


def test_step_ready_trace_keeps_same_source_room_graph_evidence() -> None:
    evidence = RosBridgePolicy._step_ready_stage_evidence(
        {
            "ready": True,
            "step_index": 9,
            "stamp_sec": 4.5,
            "source_alignment_required": True,
            "source_aligned": True,
            "source_tuples": {
                "semantic_mapping": {"step_index": 9, "stamp_sec": 4.5},
                "explore_py": {"step_index": 9, "stamp_sec": 4.5},
            },
            "modules": {
                "semantic_mapping": {
                    "causal_contract": "occ_room_graph_same_source",
                    "raw_occ_ready": True,
                    "room_segmentation_ready": True,
                    "unified_graph_ready": True,
                    "occupancy_source": {"step_index": 9, "stamp_sec": 4.5},
                    "room_segmentation_source": {"step_index": 9, "stamp_sec": 4.5},
                    "unified_graph_room_source": {"step_index": 9, "stamp_sec": 4.5},
                    "published_graph_revision": 17,
                }
            },
        }
    )

    assert evidence["aggregate"]["source_aligned"]
    assert evidence["semantic_mapping"]["causal_contract"] == "occ_room_graph_same_source"
    assert evidence["semantic_mapping"]["unified_graph_room_source"] == {
        "step_index": 9,
        "stamp_sec": 4.5,
    }
