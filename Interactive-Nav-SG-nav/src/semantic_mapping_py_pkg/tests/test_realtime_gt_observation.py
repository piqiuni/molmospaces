from __future__ import annotations

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmo_spaces.policy.learned_policy import realtime_gt_observation as realtime_gt


class FakeString:
    def __init__(self, data=""):
        self.data = data


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeRospy:
    def __init__(self):
        self.publisher = FakePublisher()

    def Publisher(self, *_args, **_kwargs):
        return self.publisher

    @staticmethod
    def logwarn_throttle(*_args, **_kwargs):
        return None


class FakeNamedElement:
    def __init__(self, element_id, name=""):
        self.id = element_id
        self.name = name


class FakeModel:
    njnt = 0
    ngeom = 3
    body_rootid = np.asarray([0, 1, 2])
    body_parentid = np.asarray([0, 0, 0])
    geom_bodyid = np.asarray([1, 1, 2])
    jnt_bodyid = np.asarray([], dtype=np.int32)

    def body(self, name_or_id):
        if isinstance(name_or_id, int):
            return FakeNamedElement(name_or_id)
        return FakeNamedElement({"chair_body": 1, "cup_body": 2}[name_or_id])

    @staticmethod
    def joint(joint_id):
        return FakeNamedElement(joint_id)


class FakeData:
    xpos = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.5], [12.0, 0.0, 0.8]])
    xmat = np.asarray([np.eye(3).reshape(-1), np.eye(3).reshape(-1), np.eye(3).reshape(-1)])


class FakeObjectManager:
    @staticmethod
    def find_door_names():
        return []

    @staticmethod
    def has_receptacle_site(_name):
        return False

    @staticmethod
    def has_free_joint(_name):
        return False

    @staticmethod
    def is_object_articulable(_name):
        return False


class FakeCamera:
    pos = np.asarray([0.0, 0.0, 0.0])
    forward = np.asarray([1.0, 0.0, 0.0])
    up = np.asarray([0.0, 0.0, 1.0])
    fov = 70.0


class FakeCameraManager:
    registry = {"head_camera": FakeCamera()}


class FakeEnv:
    current_batch_index = 0
    current_model = FakeModel()
    current_data = FakeData()
    object_managers = [FakeObjectManager()]
    camera_manager = FakeCameraManager()
    current_scene_metadata = {
        "objects": {
            "chair_body": {"category": "Chair", "object_id": "chair|1"},
            "cup_body": {"category": "Cup", "object_id": "cup|1"},
        }
    }
    segmentation = np.zeros((5, 5, 2), dtype=np.int32)

    @classmethod
    def render_segmentation_frame(cls, _camera_name):
        return cls.segmentation.copy()


class FakeTask:
    env = FakeEnv()


def _set_geom_pixels(geom_ids):
    segmentation = np.zeros((5, 5, 2), dtype=np.int32)
    segmentation[..., 1] = -1
    flat = segmentation.reshape(-1, 2)
    for index, geom_id in enumerate(geom_ids):
        flat[index, 0] = geom_id
        flat[index, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
    FakeEnv.segmentation = segmentation


def test_numeric_mujoco_joint_types_are_normalized():
    assert realtime_gt._joint_type_name(int(mujoco.mjtJoint.mjJNT_HINGE)) == "hinge"
    assert realtime_gt._joint_type_name(int(mujoco.mjtJoint.mjJNT_SLIDE)) == "slide"
    assert realtime_gt._joint_type_name(np.asarray([int(mujoco.mjtJoint.mjJNT_HINGE)])) == "hinge"


def test_slide_joint_open_direction_defines_container_front_axis():
    class JointModel:
        jnt_bodyid = np.asarray([1])

        @staticmethod
        def joint(_name):
            return FakeNamedElement(0, "drawer_slide")

    class JointData:
        xaxis = np.asarray([[0.0, -1.0, 0.0]])

    axis = realtime_gt._interaction_approach_axis_xy(
        JointModel(),
        JointData(),
        [
            {
                "joint_name": "drawer_slide",
                "joint_type": "slide",
                "joint_range": [0.0, 0.5],
                "joint_value": 0.0,
            }
        ],
    )

    assert axis == [0.0, -1.0]


def test_visible_fraction_rejects_small_observed_extent():
    camera_position = np.asarray([0.0, 0.0, 0.0])
    camera_forward = np.asarray([1.0, 0.0, 0.0])
    camera_up = np.asarray([0.0, 0.0, 1.0])
    center = np.asarray([3.0, 0.0, 0.0])
    size = np.asarray([1.0, 1.0, 1.0])
    projected = realtime_gt._project_aabb_bbox(
        camera_position,
        camera_forward,
        camera_up,
        70.0,
        [100, 100],
        center,
        size,
    )
    assert projected is not None
    full_bbox = [int(value) for value in projected]
    full_fraction, _ = realtime_gt._visible_fraction(
        full_bbox,
        camera_position,
        camera_forward,
        camera_up,
        70.0,
        [100, 100],
        center,
        size,
    )
    small_fraction, _ = realtime_gt._visible_fraction(
        [48, 48, 51, 51],
        camera_position,
        camera_forward,
        camera_up,
        70.0,
        [100, 100],
        center,
        size,
    )
    assert full_fraction > 0.8
    assert small_fraction < 0.2


def test_articulated_doorway_root_is_the_canonical_gt_spec():
    model = type("DoorModel", (), {"body_rootid": np.asarray([0, 1, 1, 3])})()
    specs = [
        realtime_gt._ObjectSpec("door_root", {}, 1, ("hinge",), True, False, True, False),
        realtime_gt._ObjectSpec("door_leaf", {}, 2, ("hinge",), True, False, False, False),
        realtime_gt._ObjectSpec("fixed_door", {}, 3, (), True, False, False, False),
    ]
    assert realtime_gt.RealtimeGTObservationPublisher._canonical_door_root_specs(model, specs) == {1: 0}


def test_door_geom_mapping_excludes_unrelated_sibling_under_same_root():
    model = type(
        "DoorModel",
        (),
        {
            "body_parentid": np.asarray([0, 0, 1, 1]),
        },
    )()
    mapping = {2: 7}

    assert (
        realtime_gt.RealtimeGTObservationPublisher._door_spec_for_body(
            model, 2, mapping
        )
        == 7
    )
    assert (
        realtime_gt.RealtimeGTObservationPublisher._door_spec_for_body(
            model, 3, mapping
        )
        is None
    )


def test_one_pass_visibility_step_interval_stable_ids_and_episode_reset():
    fake_rospy = FakeRospy()
    publisher = realtime_gt.RealtimeGTObservationPublisher(
        fake_rospy,
        FakeString,
        min_visible_pixels=4,
        min_visible_fraction=0.0,
        required_consecutive_observations=1,
        max_distance_m=8.0,
        step_interval=3,
        async_processing=False,
    )
    original_aabb = realtime_gt.body_aabb

    def fake_aabb(_model, data, body_id, visual_only=True):
        assert visual_only is True
        return data.xpos[body_id].copy(), np.asarray([0.5, 0.5, 1.0])

    realtime_gt.body_aabb = fake_aabb
    try:
        publisher.reset()
        _set_geom_pixels([0] * 3 + [1] * 3 + [2] * 5)
        first = publisher.publish(FakeTask(), step_index=0)
        assert first["episode_reset"] is True
        assert first["capture_step"] == 0
        assert first["image_size"] == [5, 5]
        observation = first["observations"][0]
        assert set(observation) == {
            "id",
            "name",
            "bbox_2d",
            "visible_pixels",
            "visible_fraction",
            "box_3d",
        }
        assert observation["id"] == "chair_body"
        assert observation["name"] == "Chair"
        assert observation["bbox_2d"] == [0, 0, 4, 1]
        assert observation["visible_pixels"] == 6
        assert observation["visible_fraction"] == 0.6
        assert observation["box_3d"] == {
            "center": [2.0, 0.0, 0.5],
            "size": [0.5, 0.5, 1.0],
            "frame_id": "world",
        }
        forbidden = {
            "joint_infos",
            "joint_type",
            "joint_range",
            "joint_value",
            "parent",
            "is_door",
            "is_receptacle",
            "is_articulable",
            "orientation",
            "interaction_approach_axis_xy",
        }
        assert forbidden.isdisjoint(observation)
        assert publisher.publish(FakeTask(), step_index=1) is None
        assert publisher.publish(FakeTask(), step_index=2) is None

        _set_geom_pixels([2] * 5 + [0] * 4)
        second = publisher.publish(FakeTask(), step_index=3)
        assert [item["id"] for item in second["observations"]] == ["chair_body"]
        assert len(fake_rospy.publisher.messages) == 2
        assert json.loads(fake_rospy.publisher.messages[-1].data)["frame_index"] == 1

        publisher.reset()
        third = publisher.publish(FakeTask(), step_index=0)
        assert third["episode_id"] == "episode_000002"
        assert third["episode_reset"] is True
        assert third["observations"][0]["id"] == "chair_body"
    finally:
        realtime_gt.body_aabb = original_aabb
        publisher.close()


def test_raw_gt_publisher_does_not_add_temporal_reliability_fields():
    fake_rospy = FakeRospy()
    publisher = realtime_gt.RealtimeGTObservationPublisher(
        fake_rospy,
        FakeString,
        min_visible_pixels=4,
        min_visible_fraction=0.0,
        required_consecutive_observations=2,
        max_distance_m=8.0,
        step_interval=3,
        async_processing=False,
    )
    original_aabb = realtime_gt.body_aabb

    def fake_aabb(_model, data, body_id, visual_only=True):
        assert visual_only is True
        return data.xpos[body_id].copy(), np.asarray([0.5, 0.5, 1.0])

    realtime_gt.body_aabb = fake_aabb
    try:
        publisher.reset()
        _set_geom_pixels([0] * 6)
        first = publisher.publish(FakeTask(), step_index=0)
        second = publisher.publish(FakeTask(), step_index=3)
        assert len(first["observations"]) == 1
        assert len(second["observations"]) == 1
        assert "consecutive_observations" not in first["observations"][0]
        assert first["observations"][0]["visible_fraction"] == 0.6

        _set_geom_pixels([])
        publisher.publish(FakeTask(), step_index=6)
        _set_geom_pixels([0] * 6)
        after_gap = publisher.publish(FakeTask(), step_index=9)
        assert len(after_gap["observations"]) == 1
    finally:
        realtime_gt.body_aabb = original_aabb
        publisher.close()
