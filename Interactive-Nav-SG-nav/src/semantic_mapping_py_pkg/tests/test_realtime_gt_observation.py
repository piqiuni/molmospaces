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


class SnapshotEnv(FakeEnv):
    render_calls = 0

    @classmethod
    def render_segmentation_frame(cls, _camera_name):
        cls.render_calls += 1
        return cls.segmentation.copy()


class SnapshotTask:
    def __init__(self, snapshot):
        self.env = SnapshotEnv()
        self._snapshot = snapshot
        self.snapshot_requests = []
        self.snapshot_invalidations = 0

    def get_private_realtime_gt_segmentation_snapshot(self, camera_name):
        self.snapshot_requests.append(camera_name)
        return self._snapshot

    def invalidate_private_realtime_gt_segmentation_snapshot(self):
        self.snapshot_invalidations += 1
        self._snapshot = None


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


def test_minimal_gt_observation_uses_generic_door_name_and_redacts_source_id():
    spec = realtime_gt._ObjectSpec(
        "doorframe_static_1",
        {"category": "Doorframe"},
        1,
        (),
        True,
        False,
        False,
        False,
    )

    observation = realtime_gt.RealtimeGTObservationPublisher._build_observation(
        None,
        spec,
        [0, 0, 3, 3],
        1,
        np.asarray([1.0, 2.0, 1.0]),
        np.asarray([1.0, 0.2, 2.0]),
    )

    assert observation["name"] == observation["id"] == "door_0001"
    assert "doorframe" not in str(observation).casefold()
    assert "gt_" not in str(observation).casefold()


def test_realtime_gt_door_id_is_stable_and_resolves_only_inside_publisher():
    publisher = realtime_gt.RealtimeGTObservationPublisher(
        FakeRospy(), FakeString, async_processing=False
    )
    spec = realtime_gt._ObjectSpec(
        "private_doorframe_root",
        {"category": "Doorframe"},
        1,
        (),
        True,
        False,
        False,
        False,
    )

    first = publisher._build_observation(
        spec,
        [0, 0, 3, 3],
        1,
        np.asarray([1.0, 2.0, 1.0]),
        np.asarray([1.0, 0.2, 2.0]),
    )
    second = publisher._build_observation(
        spec,
        [0, 0, 3, 3],
        1,
        np.asarray([1.0, 2.0, 1.0]),
        np.asarray([1.0, 0.2, 2.0]),
    )

    assert first["id"] == second["id"] == "door_0001"
    assert first["name"] == "door_0001"
    assert "doorframe" not in str(first).casefold()
    assert "gt_" not in str(first).casefold()
    assert publisher.resolve_public_object_id(first["id"]) == "private_doorframe_root"


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


def _visible_instance_publisher(
    min_visible_pixels: int,
    *,
    is_door: bool = False,
    min_visible_bbox_short_side_px: int = 2,
    min_portal_bbox_short_side_px: int = 8,
):
    publisher = object.__new__(realtime_gt.RealtimeGTObservationPublisher)
    publisher._specs = [type("VisibleSpec", (), {"is_door": bool(is_door)})()]
    publisher._geom_to_spec = np.asarray([0], dtype=np.int32)
    publisher.min_visible_pixels = min_visible_pixels
    publisher.min_visible_bbox_short_side_px = min_visible_bbox_short_side_px
    publisher.min_portal_bbox_short_side_px = min_portal_bbox_short_side_px
    return publisher


def _segmentation_for_geom_pixels(shape, pixels):
    segmentation = np.zeros((*shape, 2), dtype=np.int32)
    segmentation[..., 1] = -1
    for y, x in pixels:
        segmentation[y, x, 0] = 0
        segmentation[y, x, 1] = int(mujoco.mjtObj.mjOBJ_GEOM)
    return segmentation


def test_visible_instances_uses_dominant_connected_component_bbox():
    publisher = _visible_instance_publisher(min_visible_pixels=4)
    dominant = [(y, x) for y in range(1, 5) for x in range(1, 5)]
    detached = [(10, 10), (10, 11), (11, 10), (11, 11)]
    segmentation = _segmentation_for_geom_pixels((16, 16), dominant + detached)

    visible = publisher._visible_instances(segmentation)

    assert len(visible) == 1
    assert visible[0][:3] == (0, 16, [1, 1, 4, 4])
    assert visible[0][3]["size"] == [16, 16]
    assert sum(visible[0][3]["counts"][1::2]) == 16


def test_visible_instances_does_not_sum_disconnected_fragments_to_pass_threshold():
    publisher = _visible_instance_publisher(min_visible_pixels=12)
    first = [(y, x) for y in range(1, 4) for x in range(1, 4)]
    second = [(y, x) for y in range(10, 13) for x in range(10, 13)]
    segmentation = _segmentation_for_geom_pixels((16, 16), first + second)

    assert publisher._visible_instances(segmentation) == []


def test_realtime_gt_rejects_one_pixel_portal_sliver_before_publication():
    publisher = _visible_instance_publisher(
        min_visible_pixels=16,
        is_door=True,
        min_portal_bbox_short_side_px=8,
    )
    # This mirrors the House 4 leakage: a 1x52 wall-edge fragment has enough
    # segmentation pixels to pass an area-only gate but is not identifiable as
    # a doorway in the RGB image.
    sliver = [(y, 4) for y in range(2, 54)]
    segmentation = _segmentation_for_geom_pixels((64, 16), sliver)

    assert publisher._visible_instances(segmentation) == []


def test_realtime_gt_keeps_visually_resolved_portal_component():
    publisher = _visible_instance_publisher(
        min_visible_pixels=16,
        is_door=True,
        min_portal_bbox_short_side_px=8,
    )
    pixels = [(y, x) for y in range(4, 16) for x in range(3, 11)]
    segmentation = _segmentation_for_geom_pixels((24, 16), pixels)

    visible = publisher._visible_instances(segmentation)
    assert len(visible) == 1
    assert visible[0][:3] == (0, 96, [3, 4, 10, 15])
    assert sum(visible[0][3]["counts"][1::2]) == 96


def test_publisher_applies_min_visible_fraction_to_projected_object_extent():
    publisher = realtime_gt.RealtimeGTObservationPublisher(
        FakeRospy(),
        FakeString,
        min_visible_pixels=1,
        min_visible_fraction=0.2,
        required_consecutive_observations=1,
        max_distance_m=8.0,
        step_interval=1,
        async_processing=False,
    )
    original_aabb = realtime_gt.body_aabb

    def fake_aabb(_model, data, body_id, visual_only=True):
        assert visual_only is True
        return data.xpos[body_id].copy(), np.asarray([1.0, 1.0, 1.0])

    realtime_gt.body_aabb = fake_aabb
    try:
        # One segmentation pixel is materially smaller than the visible AABB
        # projection.  It must not become a public interaction observation.
        _set_geom_pixels([0])
        publisher.reset()

        payload = publisher.publish(FakeTask(), step_index=0)

        assert payload is not None
        assert payload["observations"] == []
    finally:
        realtime_gt.body_aabb = original_aabb
        publisher.close()


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
            "mask_rle",
            "box_3d",
        }
        assert observation["id"] == "chair_body"
        assert observation["name"] == "Chair"
        assert observation["bbox_2d"] == [0, 0, 4, 1]
        assert observation["visible_pixels"] == 6
        assert "segmentation" not in observation
        assert sum(observation["mask_rle"]["counts"][1::2]) == 6
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


def test_rule_oracle_gt_axis_is_opt_in_on_a_public_observation():
    spec = realtime_gt._ObjectSpec(
        "fridge_private",
        {"category": "Fridge"},
        1,
        ("hinge",),
        False,
        False,
        True,
        False,
    )
    publisher = realtime_gt.RealtimeGTObservationPublisher(
        FakeRospy(), FakeString, emit_interaction_approach_axis=True, async_processing=False
    )
    try:
        observation = publisher._build_observation(
            spec,
            [0, 0, 3, 3],
            16,
            np.asarray([1.0, 2.0, 1.0]),
            np.asarray([1.0, 1.0, 2.0]),
            interaction_approach_axis_xy=[1.0, 0.0],
        )
        assert observation["interaction_approach_axis_xy"] == [1.0, 0.0]
        assert observation["oracle_rule_gt_interaction_axis"] is True
        assert (
            observation["interaction_approach_axis_source"]
            == "rule_oracle_gt_joint_geometry"
        )
    finally:
        publisher.close()


def test_realtime_gt_reuses_private_snapshot_but_force_always_renders_fresh():
    fake_rospy = FakeRospy()
    publisher = realtime_gt.RealtimeGTObservationPublisher(
        fake_rospy,
        FakeString,
        min_visible_pixels=1,
        min_visible_fraction=0.0,
        required_consecutive_observations=1,
        max_distance_m=8.0,
        step_interval=1,
        async_processing=False,
    )
    original_aabb = realtime_gt.body_aabb

    def fake_aabb(_model, data, body_id, visual_only=True):
        assert visual_only is True
        return data.xpos[body_id].copy(), np.asarray([0.5, 0.5, 1.0])

    realtime_gt.body_aabb = fake_aabb
    try:
        _set_geom_pixels([0] * 6)
        SnapshotEnv.render_calls = 0
        task = SnapshotTask(FakeEnv.segmentation.copy())

        assert publisher.should_publish_step(0)
        reused = publisher.publish(task, step_index=0)

        assert reused is not None
        assert publisher.last_snapshot_used is True
        assert task.snapshot_requests == ["head_camera"]
        assert SnapshotEnv.render_calls == 0
        assert reused["source_mode"] == "geometry_observation"
        assert "segmentation" not in reused
        assert "segmentation" not in reused["observations"][0]

        fresh = publisher.publish(task, step_index=1, force=True)

        assert fresh is not None
        assert publisher.last_snapshot_used is False
        assert task.snapshot_invalidations == 1
        assert SnapshotEnv.render_calls == 1
    finally:
        realtime_gt.body_aabb = original_aabb
        publisher.close()
