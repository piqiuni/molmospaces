from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from molmo_spaces.configs.task_configs import NavToObjTaskConfig
from molmo_spaces.evaluation.benchmark_schema import NavToObjTaskSpec
from molmo_spaces.tasks.nav_task import NavToObjTask


def _make_task(visibility: float) -> NavToObjTask:
    """Build the smallest NavToObjTask surface needed for metric evaluation."""
    task = NavToObjTask.__new__(NavToObjTask)
    task._env = SimpleNamespace(n_batch=1, check_visibility=Mock(return_value=visibility))
    task.config = SimpleNamespace(task_config=SimpleNamespace(succ_pos_threshold=1.0))
    task.episode_step_count = 1
    task._visibility_reward_cache_active = False
    task._visibility_cache = {}
    task._reward_cache_for_current_state = None
    task._realtime_gt_snapshot_requested = False
    task.observation_cache = []
    task.reward_cache = []
    task.terminal_cache = []
    task.truncated_cache = []
    task.success_cache = []
    task.retain_history = True
    task.get_observations = lambda: [{}]
    task.is_terminal = lambda: np.array([False])
    task.is_timed_out = lambda: np.array([False])
    task.get_nearest_nav_object = lambda _index: SimpleNamespace(name="target")
    task.calculate_distance = lambda _index: 0.5
    return task


def test_specific_instance_survives_schema_and_runtime_config_validation() -> None:
    task_spec = NavToObjTaskSpec.model_validate(
        {
            "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
            "task_type": "nav_to_obj",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_name": "target_instance",
            "pickup_obj_candidates": ["target_instance", "other_instance"],
            "selection_mode": "specific_instance",
        }
    )
    runtime_payload = task_spec.model_dump()
    runtime_payload["task_cls"] = None
    runtime_config = NavToObjTaskConfig(**runtime_payload)

    assert task_spec.selection_mode == "specific_instance"
    assert runtime_config.selection_mode == "specific_instance"


def test_nav_visibility_and_reward_cache_is_scoped_to_one_metric_snapshot() -> None:
    task = _make_task(visibility=0.0)

    _, reward, _, _, info = task.get_and_cache_all_step_information()

    assert reward.tolist() == [0.0]
    assert not bool(info[0]["success"])
    # get_reward(), get_info(), and the failed judge_success() all inspect the
    # same state, yet cause only one segmentation visibility query.
    assert task._env.check_visibility.call_count == 1
    assert task._visibility_reward_cache_active is False
    assert task._visibility_cache == {}

    # A later snapshot must observe the changed live state rather than reuse the
    # preceding step's visibility/reward cache.
    task._env.check_visibility.return_value = 1.0
    _, next_reward, _, _, next_info = task.get_and_cache_all_step_information()

    assert next_reward.tolist() == [0.5]
    assert bool(next_info[0]["success"])
    assert task._env.check_visibility.call_count == 2


def test_nav_reward_is_not_cached_outside_metric_snapshot() -> None:
    task = _make_task(visibility=1.0)

    first_reward = task.get_reward()
    second_reward = task.get_reward()

    assert first_reward.tolist() == [0.5]
    assert second_reward.tolist() == [0.5]
    assert task._env.check_visibility.call_count == 2


def test_private_gt_segmentation_snapshot_is_same_state_only() -> None:
    task = _make_task(visibility=0.0)
    segmentation = np.zeros((3, 4, 2), dtype=np.int32)
    render = Mock(return_value=segmentation)
    data = SimpleNamespace(qpos=np.asarray([0.0, 1.0]), time=1.0)
    camera = SimpleNamespace(
        pos=np.asarray([0.0, 0.0, 1.0]),
        forward=np.asarray([1.0, 0.0, 0.0]),
        up=np.asarray([0.0, 0.0, 1.0]),
        fov=70.0,
    )
    task._env = SimpleNamespace(
        n_batch=1,
        current_batch_index=0,
        current_model=object(),
        current_data=data,
        camera_manager=SimpleNamespace(registry={"head_camera": camera}),
        render_segmentation_frame=render,
        segmentation_fraction=Mock(return_value=1.0),
        check_visibility=Mock(),
    )

    # The bridge enables this only when the next policy call will publish GT.
    task.request_private_realtime_gt_segmentation_snapshot(True)
    _, reward, _, _, _ = task.get_and_cache_all_step_information()

    snapshot = task.get_private_realtime_gt_segmentation_snapshot("head_camera")
    assert reward.tolist() == [0.5]
    assert render.call_count == 1
    assert task._env.check_visibility.call_count == 0
    assert snapshot is not None
    assert np.array_equal(snapshot, segmentation)
    assert not snapshot.flags.writeable

    # Direct qpos edits outside task.step are detected defensively, so a
    # subsequent GT publish falls back to a fresh render instead of reusing it.
    data.qpos[0] = 0.25
    assert task.get_private_realtime_gt_segmentation_snapshot("head_camera") is None
    assert task._realtime_gt_segmentation_snapshot is None


def test_private_gt_snapshot_is_not_captured_when_next_gt_frame_is_not_due() -> None:
    task = _make_task(visibility=0.0)
    render = Mock(return_value=np.zeros((3, 4, 2), dtype=np.int32))
    task._env = SimpleNamespace(
        n_batch=1,
        current_batch_index=0,
        current_model=object(),
        current_data=SimpleNamespace(qpos=np.asarray([0.0]), time=1.0),
        camera_manager=SimpleNamespace(
            registry={
                "head_camera": SimpleNamespace(
                    pos=np.asarray([0.0, 0.0, 1.0]),
                    forward=np.asarray([1.0, 0.0, 0.0]),
                    up=np.asarray([0.0, 0.0, 1.0]),
                    fov=70.0,
                )
            }
        ),
        render_segmentation_frame=render,
        segmentation_fraction=Mock(return_value=1.0),
        check_visibility=Mock(return_value=1.0),
    )

    task.get_and_cache_all_step_information()

    assert render.call_count == 0
    assert task._env.check_visibility.call_count == 1
    assert task.get_private_realtime_gt_segmentation_snapshot("head_camera") is None


def test_specific_instance_replaces_category_candidates() -> None:
    task = NavToObjTask.__new__(NavToObjTask)
    task.config = SimpleNamespace(
        task_config=SimpleNamespace(
            pickup_obj_name="target_instance",
            pickup_obj_candidates=["target_instance", "same_category_instance"],
            selection_mode="specific_instance",
        )
    )
    env = SimpleNamespace(current_batch_index=0)

    task._reconstruct_candidate_list_if_needed(env)

    assert task.config.task_config.pickup_obj_candidates == ["target_instance"]


def test_specific_instance_builds_only_the_selected_nav_object() -> None:
    task = NavToObjTask.__new__(NavToObjTask)
    task.config = SimpleNamespace(
        task_config=SimpleNamespace(
            pickup_obj_name="target_instance",
            pickup_obj_candidates=["target_instance", "same_category_instance"],
            selection_mode="specific_instance",
        )
    )
    task._env = SimpleNamespace(n_batch=1, mj_datas=[object()])

    with patch(
        "molmo_spaces.tasks.nav_task.MlSpacesObject",
        side_effect=lambda *, data, object_name: SimpleNamespace(
            data=data, name=object_name
        ),
    ):
        nav_objects = task._get_nav_objects()

    assert [[obj.name for obj in batch] for batch in nav_objects] == [
        ["target_instance"]
    ]
