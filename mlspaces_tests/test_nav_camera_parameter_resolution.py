import math
from types import SimpleNamespace

import numpy as np

from molmo_spaces.env.sensors import get_nav_task_sensors
from molmo_spaces.env.sensors_cameras import CameraParameterSensor


def test_nav_camera_parameters_use_the_render_resolution() -> None:
    config = SimpleNamespace(
        camera_config=SimpleNamespace(
            img_resolution=(640, 480),
            cameras=[SimpleNamespace(name="head_camera", record_depth=True)],
        ),
        task_config=SimpleNamespace(
            pickup_obj_candidates=[],
            pickup_obj_name="",
            action_dtype="float32",
        ),
    )

    sensors = get_nav_task_sensors(config)
    camera_params = next(
        sensor for sensor in sensors if isinstance(sensor, CameraParameterSensor)
    )

    assert camera_params.camera_name == "head_camera"
    assert camera_params.img_resolution == (640, 480)

    environment = SimpleNamespace(
        camera_manager=SimpleNamespace(
            registry={
                "head_camera": SimpleNamespace(
                    fov=139.0,
                    get_pose=lambda: np.eye(4, dtype=np.float32),
                )
            }
        )
    )
    intrinsics = camera_params.get_observation(environment, task=None)["intrinsic_cv"]
    expected_focal = (480 / 2.0) / math.tan(math.radians(139.0 / 2.0))

    np.testing.assert_allclose(
        intrinsics,
        [[expected_focal, 0.0, 320.0], [0.0, expected_focal, 240.0], [0.0, 0.0, 1.0]],
    )
