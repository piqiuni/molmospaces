"""Fast regression coverage for 640x480 RGB-D intrinsics.

These tests intentionally use small fake camera/task objects.  They do not
start ROS or MuJoCo, but exercise the same sensor and bridge helpers used by
the native navigation evaluation.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from molmo_spaces.env.sensors import get_nav_task_sensors
from molmo_spaces.env.sensors_cameras import CameraParameterSensor
from molmo_spaces.policy.learned_policy.ros_bridge_policy import RosBridgePolicy


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_FOV_DEG = 90.0


class _FakeCamera:
    fov = CAMERA_FOV_DEG

    def get_pose(self) -> np.ndarray:
        return np.eye(4, dtype=np.float32)


def _expected_focal_length() -> float:
    return (IMAGE_HEIGHT * 0.5) / math.tan(math.radians(CAMERA_FOV_DEG * 0.5))


def _task_config_with_head_camera() -> SimpleNamespace:
    return SimpleNamespace(
        camera_config=SimpleNamespace(
            cameras=[SimpleNamespace(name="head_camera", fov=CAMERA_FOV_DEG)]
        )
    )


def test_nav_sensor_factory_forwards_actual_camera_resolution() -> None:
    """The nav path must not fall back to CameraParameterSensor's 480x480 default."""
    exp_config = SimpleNamespace(
        camera_config=SimpleNamespace(
            cameras=[SimpleNamespace(name="head_camera", record_depth=True)],
            img_resolution=(IMAGE_WIDTH, IMAGE_HEIGHT),
        ),
        task_config=SimpleNamespace(
            pickup_obj_candidates=[],
            pickup_obj_name=None,
            action_dtype="float32",
        ),
    )

    sensors = get_nav_task_sensors(exp_config)
    params_sensor = next(
        sensor
        for sensor in sensors
        if isinstance(sensor, CameraParameterSensor) and sensor.camera_name == "head_camera"
    )

    assert params_sensor.img_resolution == (IMAGE_WIDTH, IMAGE_HEIGHT)


def test_camera_parameter_sensor_builds_isotropic_640x480_intrinsics() -> None:
    sensor = CameraParameterSensor(
        camera_name="head_camera",
        img_resolution=(IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    env = SimpleNamespace(
        camera_manager=SimpleNamespace(registry={"head_camera": _FakeCamera()})
    )

    intrinsics = np.asarray(sensor.get_observation(env, task=None)["intrinsic_cv"])
    expected_focal = _expected_focal_length()

    np.testing.assert_allclose(
        intrinsics,
        np.array(
            [
                [expected_focal, 0.0, IMAGE_WIDTH * 0.5],
                [0.0, expected_focal, IMAGE_HEIGHT * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_bridge_normalization_keeps_matching_640x480_intrinsics_isotropic() -> None:
    focal = _expected_focal_length()
    matching_intrinsics = (focal, focal, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5)

    normalized = RosBridgePolicy._normalize_intrinsics_to_image_shape(
        matching_intrinsics,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )

    assert normalized is not None
    np.testing.assert_allclose(normalized, matching_intrinsics, rtol=1e-6, atol=1e-6)
    assert math.isclose(normalized[0] / normalized[1], 1.0, abs_tol=1e-6)


def test_task_config_fov_fallback_recovers_legacy_aspect_ratio_skew() -> None:
    """A 480x480 legacy K may become skewed, but bridge FoV fallback repairs it."""
    policy = object.__new__(RosBridgePolicy)
    policy.task = SimpleNamespace(config=_task_config_with_head_camera())
    # The policy config is intentionally missing camera parameters: native
    # evaluation constructs the policy before its episode config is installed.
    policy.config = SimpleNamespace(camera_config=None)

    focal = _expected_focal_length()
    legacy_intrinsics = (focal, focal, 240.0, 240.0)
    normalized_legacy = RosBridgePolicy._normalize_intrinsics_to_image_shape(
        legacy_intrinsics,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )
    assert normalized_legacy is not None
    assert math.isclose(
        normalized_legacy[0] / normalized_legacy[1],
        4.0 / 3.0,
        abs_tol=1e-6,
    )

    recovered = policy._intrinsics_from_fov(
        "head_camera",
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )

    assert recovered is not None
    np.testing.assert_allclose(
        recovered,
        (focal, focal, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5),
        rtol=1e-6,
        atol=1e-6,
    )
    assert math.isclose(recovered[0] / recovered[1], 1.0, abs_tol=1e-6)


def test_projection_lut_has_no_25_percent_horizontal_compression() -> None:
    policy = object.__new__(RosBridgePolicy)
    policy._pointcloud_projection_cache = {}
    focal = _expected_focal_length()
    fx, fy, cx, cy = focal, focal, IMAGE_WIDTH * 0.5, IMAGE_HEIGHT * 0.5

    proj_x, _ = policy._get_pointcloud_projection_lut(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        fx,
        fy,
        cx,
        cy,
    )
    depth_m = 3.0
    pixel_x = 480  # 160 px right of cx
    projected_x = float(proj_x[int(cy), pixel_x] * depth_m)
    expected_x = ((pixel_x - cx) / fx) * depth_m
    incorrectly_compressed_x = ((pixel_x - cx) / (fx * (4.0 / 3.0))) * depth_m

    assert math.isclose(projected_x, expected_x, abs_tol=1e-6)
    assert not math.isclose(projected_x, incorrectly_compressed_x, abs_tol=1e-6)
    assert math.isclose(incorrectly_compressed_x / projected_x, 0.75, abs_tol=1e-6)
