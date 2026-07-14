from __future__ import annotations

import json

import mujoco
import numpy as np

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
    xpos = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 0.5], [12.0, 0.0, 0.8]])
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


def test_one_pass_visibility_step_interval_stable_ids_and_episode_reset():
    fake_rospy = FakeRospy()
    publisher = realtime_gt.RealtimeGTObservationPublisher(
        fake_rospy,
        FakeString,
        min_visible_pixels=4,
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
        assert [item["source_object_name"] for item in first["observations"]] == ["chair_body"]
        assert first["observations"][0]["visible_pixels"] == 6
        assert first["observations"][0]["instance_id"] == "gt_000001"
        assert first["observations"][0]["bbox_2d"] == [0, 0, 4, 1]
        assert publisher.publish(FakeTask(), step_index=1) is None
        assert publisher.publish(FakeTask(), step_index=2) is None

        _set_geom_pixels([2] * 5 + [0] * 4)
        second = publisher.publish(FakeTask(), step_index=3)
        assert [item["source_object_name"] for item in second["observations"]] == ["chair_body"]
        assert second["observations"][0]["instance_id"] == "gt_000001"
        assert len(fake_rospy.publisher.messages) == 2
        assert json.loads(fake_rospy.publisher.messages[-1].data)["frame_index"] == 1

        publisher.reset()
        third = publisher.publish(FakeTask(), step_index=0)
        assert third["episode_id"] == "episode_000002"
        assert third["episode_reset"] is True
        assert third["observations"][0]["instance_id"] == "gt_000001"
    finally:
        realtime_gt.body_aabb = original_aabb
        publisher.close()
