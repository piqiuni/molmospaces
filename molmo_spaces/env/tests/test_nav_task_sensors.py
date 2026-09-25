from types import SimpleNamespace

from molmo_spaces.env.sensors import DepthSensor, get_nav_task_sensors


def _nav_config(*, record_depth: bool):
    return SimpleNamespace(
        camera_config=SimpleNamespace(
            img_resolution=(32, 24),
            cameras=[SimpleNamespace(name="head_camera", record_depth=record_depth)],
        ),
        task_config=SimpleNamespace(
            pickup_obj_candidates=[],
            pickup_obj_name=None,
            action_dtype="float32",
        ),
    )


def test_nav_sensors_include_configured_depth_camera():
    sensors = get_nav_task_sensors(_nav_config(record_depth=True))

    depth_sensors = [sensor for sensor in sensors if isinstance(sensor, DepthSensor)]
    assert [sensor.uuid for sensor in depth_sensors] == ["head_camera_depth"]


def test_nav_sensors_skip_disabled_depth_camera():
    sensors = get_nav_task_sensors(_nav_config(record_depth=False))

    assert not any(isinstance(sensor, DepthSensor) for sensor in sensors)
