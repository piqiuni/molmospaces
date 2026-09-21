import json

import mujoco
import numpy as np
import pytest
from types import SimpleNamespace

from scripts.InteractiveNav.collection.gt_trajectory import (
    Sampling, navigation_samples, uniform_values, rule_instruction, wrap,
)
from scripts.InteractiveNav.collection.gt_operation import bind_operation
from scripts.InteractiveNav.collect_full_gt import select_samples, restore_head_camera
from scripts.InteractiveNav.collection.gt_parking import sweep_clearance, parking_candidates, sweep_boxes, sweep_sequence
from scripts.InteractiveNav.collection.gt_topdown import project_world


def test_frozen_head_mount_overrides_mjcf_default():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <camera name="robot_0/head_camera"/><body name="head" pos="1 2 1.5"/>
      </worldbody></mujoco>''')
    episode = {"cameras": [{"name": "head_camera", "reference_body_names": ["head"],
                "camera_offset": [.05, 0, .05], "camera_quaternion": [.5, .5, -.5, -.5], "fov": 139}]}
    restore_head_camera(m, episode)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.cam_xpos[0] == pytest.approx([1.05, 2, 1.55])
    assert -d.cam_xmat[0].reshape(3, 3)[:, 2] == pytest.approx([1, 0, 0])
    assert m.cam_fovy[0] == 139


def test_fixed_rate_full_path_and_final_yaw():
    s = Sampling(hz=7, base_speed=0.43, yaw_speed=0.7)
    path = [[0, 0], [1, 0], [1, 1.3]]
    samples = list(navigation_samples(path, 0, -np.pi + 0.1, s))
    old = np.array([0, 0, 0])
    for pose, phase, last in samples:
        if phase == "navigate":
            speed = np.linalg.norm(pose[:2] - old[:2]) * s.hz
            expected = s.base_speed
        else:
            assert pose[:2] == pytest.approx(old[:2])
            speed = abs(wrap(pose[2] - old[2])) * s.hz
            expected = s.yaw_speed
        assert speed <= expected + 1e-8
        if not last:
            assert speed == pytest.approx(expected)
        old = pose
    assert old[:2] == pytest.approx([1, 1.3])
    assert wrap(old[2] - (-np.pi + 0.1)) == pytest.approx(0)


@pytest.mark.parametrize("start,end", [(0, -1.57), (-0.35, 0), (0, 0.35)])
def test_signed_joint_motion_reaches_endpoint(start, end):
    values = list(uniform_values(start, end, 0.2, 5))
    assert values[-1][0] == pytest.approx(end)
    assert values[-1][1]


def test_shortest_yaw_wrap_does_not_spin():
    samples = list(navigation_samples([[0, 0]], np.pi - .01, -np.pi + .01, Sampling()))
    assert len(samples) == 1
    assert abs(samples[0][0][2] - (np.pi - .01)) == pytest.approx(.02)


def test_named_handle_tracks_nested_hinge_and_unlock_joint():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="door"><joint name="hinge" axis="0 0 1" range="0 90"/>
      <geom type="box" size=".5 .02 1" pos=".5 0 1"/>
      <body name="handle" pos=".9 -.04 1"><joint name="handle_turn" axis="0 1 0" range="0 30"/>
        <geom name="handle_visual" type="capsule" size=".01 .06" contype="0" conaffinity="0"/>
      </body></body></worldbody></mujoco>''')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    env = SimpleNamespace(current_model=m, current_data=d)
    op = bind_operation(env, {"joint_name": "hinge"}, np.array([.9, -1., .8]))
    assert op.handle_joint_id == m.joint('handle_turn').id
    p0 = op.world_pose(d)
    d.qpos[0] = np.pi / 2
    d.qpos[1] = .3
    mujoco.mj_forward(m, d)
    p1 = op.world_pose(d)
    assert p1[:3, 3] == pytest.approx([.04, .9, 1])
    assert not np.allclose(p1[:3, :3], p0[:3, :3])
    assert op.metadata['latch_simulated'] is False


def test_integrated_mesh_handle_is_not_drawer_center():
    import trimesh

    panel = trimesh.creation.box(extents=[1., .3, .03])
    bar = trimesh.creation.box(extents=[.3, .02, .02])
    bar.apply_translation([0, .08, .06])
    mesh = trimesh.util.concatenate([panel, bar])
    vertices = ' '.join(map(str, mesh.vertices.ravel()))
    faces = ' '.join(map(str, mesh.faces.ravel()))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
      <mesh name="combined" vertex="{vertices}" face="{faces}"/></asset>
      <worldbody><body name="drawer"><joint name="slide" type="slide" axis="0 1 0" range="0 .3"/>
      <geom type="mesh" mesh="combined" contype="0" conaffinity="0"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(current_model=model, current_data=data)
    op = bind_operation(env, {'joint_name': 'slide'}, np.array([0, 1., .5]))
    assert op.metadata['source'] == 'mesh_handle_component'
    start = op.world_pose(data)
    assert start[:3, 3] == pytest.approx([0, .08, .06], abs=1e-6)
    data.qpos[0] = .3
    mujoco.mj_forward(model, data)
    end = op.world_pose(data)
    assert end[:3, 3] - start[:3, 3] == pytest.approx([0, .3, 0], abs=1e-6)
    assert end[:3, :3] == pytest.approx(start[:3, :3])


def test_instruction_preserves_segment_alignment_and_no_invented_unlock():
    segments = [dict(phase='navigate', distance_m=1.2, start_frame=1, end_frame=6),
                dict(phase='rotate_handle', start_frame=7, end_frame=9),
                dict(phase='open', object_category='drawer', pull=True, start_frame=10, end_frame=15)]
    result = rule_instruction(segments, 'apple')
    assert '（约 1.2 米）' in result['instruction']
    assert '拉开抽屉' in result['instruction']
    assert '解锁' not in result['instruction']
    assert result['clauses'][-1]['end_frame'] == 15


def test_sparse_instruction_merges_jitter_but_keeps_large_corner():
    segments = []
    points = [[0, 0], [1, .05], [2, 0], [3, .05], [3, 2.05]]
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        segments.extend([dict(phase='turn', start_frame=2*i, end_frame=2*i, yaw_delta=.05, start_yaw=0),
                         dict(phase='navigate', start_frame=2*i+1, end_frame=2*i+1,
                              start_xy=a, end_xy=b, distance_m=np.linalg.norm(np.array(b)-a))])
    result = rule_instruction(segments, 'apple')
    assert len(result['clauses']) == 1
    assert result['instruction'].count('左转') == 1
    assert result['instruction'].count('直行') == 2
    assert '度' not in result['instruction']
    assert '3.0 米' in result['instruction'] and '2.0 米' in result['instruction']
    assert result['clauses'][0]['start_frame'] == 0 and result['clauses'][0]['end_frame'] == 7


def test_retreat_faces_target_and_clearance_rejects_sweep():
    boxes = np.array([[[0, -.5], [1, .5]]])
    assert sweep_clearance(np.array([.5, 0]), boxes) == 0
    assert sweep_clearance(np.array([-1, 0]), boxes) == 1
    pose, retreat, side = next(parking_candidates(np.array([-1., 0.]), np.array([0., 0.]), .45))
    assert pose == pytest.approx([-1.45, 0, 0])
    assert retreat == .45 and side == 0


def test_sweep_preview_restores_state_and_covers_door_arc():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
    <joint name="hinge" axis="0 0 1" range="0 90"/>
    <geom type="box" pos=".5 0 1" size=".5 .02 1"/>
    </body></worldbody></mujoco>''')
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    before = d.qpos.copy()
    boxes, count = sweep_boxes(m, d, 0, np.pi/2)
    assert count >= 46
    assert sweep_clearance(np.array([.6, .6]), boxes) < .05
    assert d.qpos == pytest.approx(before)


def test_topdown_projection_has_x_right_y_up():
    pixels = project_world([[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                           np.array([0, 0, 10]), np.array([0, 0, -1]), np.array([0, 1, 0]), .1, .05, 640)
    assert pixels[0] == pytest.approx([320, 320])
    assert pixels[1, 0] > 320 and pixels[1, 1] == 320
    assert pixels[2, 0] == 320 and pixels[2, 1] < 320


def test_one_stop_sweep_includes_later_drawer_and_restores_all_joints():
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body pos="0 0 1"><joint name="door" type="hinge" axis="0 0 1" range="0 90"/>
        <geom type="box" pos=".5 0 0" size=".5 .02 .5"/></body>
      <body pos="2 0 .5"><joint name="drawer" type="slide" axis="0 1 0" range="0 1"/>
        <geom type="box" size=".2 .2 .2"/></body>
      </worldbody></mujoco>''')
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    original = d.qpos.copy()
    boxes, counts = sweep_sequence(m, d, [(0, np.pi/2), (1, 1.)])
    assert len(counts) == 2
    assert sweep_clearance(np.array([2., 1.]), boxes) == 0
    assert d.qpos == pytest.approx(original)


def test_selection_reproducible_and_has_distinct_houses():
    episodes = [dict(house_index=i, interactive_nav=dict(interaction_domains=['channel', 'container'],
        interactions=[dict(type='container_' + ('hinge' if i % 2 else 'slide'))])) for i in range(20)]
    a = select_samples(episodes, 10, 123)
    assert a == select_samples(episodes, 10, 123)
    assert len(set(a)) == 10
    assert {i % 2 for i in a} == {0, 1}


@pytest.mark.parametrize('field', ['hz', 'base_speed', 'yaw_speed', 'hinge_speed', 'slide_speed'])
def test_invalid_sampling_rejected(field):
    with pytest.raises(ValueError):
        Sampling(**{field: 0})
