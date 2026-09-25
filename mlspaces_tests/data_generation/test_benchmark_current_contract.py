"""Cross-boundary regressions for the current ROS benchmark integration."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts'))
from semantic_mapping_py_pkg.image_frame_pairing import select_image_record, stamp_key_from_ros
from scripts.InteractiveNav.evaluation.benchmark_interaction_adapter import (
    sanitize_public_interaction_command, sanitize_public_interaction_outcome,
    validate_public_interaction_pose,
)
from scripts.InteractiveNav.evaluation.benchmark_interaction_executor import validate_runtime_interaction_pose
from scripts.InteractiveNav.evaluation import benchmark_runner
from molmo_spaces.policy.learned_policy.ros_bridge_policy import RosBridgePolicy


def test_macro_views_publish_matching_rgb_without_consuming_action_steps():
    class Time:
        @staticmethod
        def from_sec(value):
            return SimpleNamespace(secs=int(value), nsecs=int((value - int(value)) * 1e9))
    bridge = RosBridgePolicy.__new__(RosBridgePolicy)
    bridge._rospy = SimpleNamespace(Time=Time)
    bridge._step_idx = 9
    bridge._extract_image_from_observation = lambda obs: obs['rgb']
    bridge._to_image_msg = lambda frame, stamp, seq: SimpleNamespace(header=SimpleNamespace(stamp=stamp, seq=seq), frame=frame.copy())
    bridge._publish_odom_and_tf = Mock()
    messages = []
    bridge._obs_pub = SimpleNamespace(publish=messages.append)
    published = []
    states = [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
    current = {'rgb': states[0]}

    def publish(payload, *, capture_step, observation_pose_xyyaw):
        assert observation_pose_xyyaw == [1.0, 2.0, 0.0]
        wire = dict(payload, capture_step=capture_step, stamp_sec=1789459200.0 + len(published) * .01)
        published.append(wire)
        return wire

    runtime = benchmark_runner.RestrictedRosObjectGoalRuntime(
        perception=SimpleNamespace(build=lambda *args, **kw: {'observations': []}, camera_name='head_camera'),
        adapter=SimpleNamespace(publish_restricted_gt_frame=publish), skill=None,
        opaque_to_source_name={}, opaque_to_joints={}, target_source_by_opaque_id={},
        goal_evidence=SimpleNamespace(record_frame=Mock()),
        public_rgb_sink=bridge.publish_public_rgb_frame,
    )
    task = SimpleNamespace(get_observations=lambda: current, env=SimpleNamespace(
        camera_manager=SimpleNamespace(registry={'head_camera': SimpleNamespace(
            pos=[1., 2., 3.], forward=[1., 0., 0.])})))
    for state in states:
        current = {'rgb': state}
        assert benchmark_runner._publish_restricted_ros_frame(runtime, task, decision_index=9)
    assert bridge._step_idx == 9
    assert len(messages) == 3
    for wire, message, state in zip(published, messages, states):
        np.testing.assert_array_equal(message.frame, state)
        _, reason = select_image_record(wire, [{'stamp_key': stamp_key_from_ros(message.header.stamp)}])
        assert reason == 'matched'


def test_explicit_navigation_tolerance_survives_public_boundary():
    command = sanitize_public_interaction_command({
        'interaction_approach_pose_xyyaw': [0., 0., 0.],
        'interaction_ready_distance_m': .25,
        'navigation_goal_position_tolerance_m': .15,
        'navigation_goal_yaw_tolerance_rad': .15,
        'joint_name': 'must_not_pass',
    })
    pose = np.eye(4)
    pose[0, 3] = .20
    env = SimpleNamespace(current_robot=SimpleNamespace(robot_view=SimpleNamespace(base=SimpleNamespace(pose=pose))))
    assert 'joint_name' not in command
    assert not validate_public_interaction_pose(command, actual_pose_xyyaw=[.20, 0., 0.])['valid']
    result = validate_runtime_interaction_pose(env, command, object_name='private_object')
    assert not result['valid']
    assert result['reason'] == 'interaction_position_misaligned'
    assert result['recovery_action'] == 'reposition_same_face'


@pytest.mark.parametrize('reason,recovery', [
    ('interaction_wrong_face', 'select_other_face'),
    ('interaction_front_unverified', 'reobserve_front'),
    ('interaction_position_misaligned', 'reposition_same_face'),
    ('interaction_orientation_misaligned', 'realign_yaw'),
])
def test_recovery_contract_preserves_semantics_but_not_private_geometry(reason, recovery):
    result = sanitize_public_interaction_outcome({
        'failure_reason': reason, 'recovery_action': recovery,
        'reject_selected_face': reason == 'interaction_wrong_face',
        'physical_front_axis': [1., 0.], 'joint_name': 'private',
    }, success=False)
    assert result['failure_reason'] == reason
    assert result['recovery_action'] == recovery
    assert result['reject_selected_face'] == (reason == 'interaction_wrong_face')
    assert 'physical_front_axis' not in result and 'joint_name' not in result


def test_execution_progress_is_command_scoped_throttled_and_stops_after_result():
    import json
    from scripts.InteractiveNav.evaluation.ros_object_goal_adapter import (
        RosObjectGoalEvaluatorAdapter, EvaluatorInteractionRequest,
    )
    now = [0.0]
    messages = {}

    def publisher(topic, *args, **kwargs):
        messages[topic] = []
        return SimpleNamespace(publish=messages[topic].append)

    adapter = RosObjectGoalEvaluatorAdapter(
        rospy_module=SimpleNamespace(Publisher=publisher, Subscriber=lambda *a, **kw: None),
        string_message_type=lambda data: SimpleNamespace(data=data), clock=lambda: now[0],
    )
    adapter.start()
    adapter._episode_id = 'episode_000001'
    request = EvaluatorInteractionRequest(
        command_id='cmd', episode_id=adapter.episode_id, instance_id='obj_000001',
        action='open', private_handle=object(), decision_id='decision',
    )
    adapter._pending_by_command_id['cmd'] = request
    adapter._pending_order.append('cmd')
    topic = '/semantic_decision/interaction_action_feedback'
    adapter.report_interaction_progress()
    assert topic not in messages  # Queued is not executing.
    assert adapter.pop_next_interaction_request() is request
    adapter.report_interaction_progress()
    adapter.report_interaction_progress()
    assert len(messages[topic]) == 1
    now[0] = 2.0
    adapter.report_interaction_progress()
    assert len(messages[topic]) == 2
    assert json.loads(messages[topic][0].data) == {
        'status': 'RUNNING', 'execution_active': True, 'command_id': 'cmd',
        'decision_id': 'decision', 'object_id': 'obj_000001', 'progress_age_s': 0.0,
    }
    adapter.complete_interaction('cmd', success=True)
    now[0] = 5.0
    adapter.report_interaction_progress()
    assert len(messages[topic]) == 2
