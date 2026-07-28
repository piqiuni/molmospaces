"""Fast regressions for native/V3 evaluation wiring and process ownership."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from molmo_spaces.evaluation.eval_main import _resolve_parent_policy
from molmo_spaces.evaluation import json_eval_runner
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from scripts.InteractiveNav.run_native_nav_to_obj_eval import (
    NativeNavToObjEvalConfig,
    _target_metadata,
)


def test_schema_import_does_not_load_heavy_eval_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import molmo_spaces.evaluation.benchmark_schema; "
                "assert 'molmo_spaces.evaluation.eval_main' not in sys.modules"
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_v3_runner_import_does_not_load_optional_heavy_stacks() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import scripts.InteractiveNav.evaluation.benchmark_runner; "
                "unexpected = {"
                "name for name in ("
                "'torch', "
                "'open_clip', "
                "'molmo_spaces.configs.base_nav_to_obj_config', "
                "'molmo_spaces.configs.robot_configs', "
                "'molmo_spaces.planner.curobo_planner'"
                ") if name in sys.modules}; "
                "assert not unexpected, unexpected"
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_v3_minimal_replay_config_preserves_runtime_contract(tmp_path: Path) -> None:
    from scripts.InteractiveNav.evaluation.benchmark_runner import (
        BenchmarkEvaluationConfig,
        _build_replay_config,
    )

    config = BenchmarkEvaluationConfig(
        benchmark=Path("benchmark.json"),
        output_dir=tmp_path,
        max_steps=73,
        policy_dt_ms=200.0,
        ctrl_dt_ms=10.0,
        sim_dt_ms=10.0,
    )
    replay = _build_replay_config(config, tmp_path)

    assert replay.task_type == "nav_to_obj"
    assert replay.task_horizon == 73
    assert replay.policy_dt_ms == 200.0
    assert replay.ctrl_dt_ms == 10.0
    assert replay.sim_dt_ms == 10.0
    assert replay.seed_torch is False
    assert replay.robot_config.name == "rby1"
    assert replay.robot_config.robot_factory.__name__ == "RBY1"
    assert replay.robot_config.action_noise_config.enabled is False
    assert set(replay.robot_config.init_qpos) == {
        "base",
        "head",
        "left_arm",
        "left_gripper",
        "right_arm",
        "right_gripper",
        "torso",
    }
    assert all(
        not values.any()
        for values in replay.robot_config.init_qpos_noise_range.values()
    )


def test_multiworker_eval_does_not_construct_or_share_parent_policy() -> None:
    constructed: list[tuple[object, object]] = []

    class _Policy:
        def __init__(self, config, task_type) -> None:
            constructed.append((config, task_type))

    config = SimpleNamespace(
        task_type="nav_to_obj",
        policy_config=SimpleNamespace(policy_cls=_Policy),
    )

    assert _resolve_parent_policy(config, None, 2) is None
    assert constructed == []
    with pytest.raises(ValueError, match="num_workers=1"):
        _resolve_parent_policy(config, object(), 2)

    policy = _resolve_parent_policy(config, None, 1)
    assert isinstance(policy, _Policy)
    assert constructed == [(config, "nav_to_obj")]


def test_native_target_context_uses_fields_consumed_by_semantic_decision() -> None:
    class _ObjectManager:
        @staticmethod
        def category_from_name(_name: str) -> str:
            return "Apple"

        @staticmethod
        def fallback_expression(_name: str) -> str:
            return "apple"

    task = SimpleNamespace(
        config=SimpleNamespace(
            task_config=SimpleNamespace(
                pickup_obj_name="apple_0",
                pickup_obj_candidates=["apple_0", "apple_1"],
                selection_mode="any_candidate",
            )
        ),
        env=SimpleNamespace(current_batch_index=0, object_managers=[_ObjectManager()]),
    )

    context = _target_metadata(task)["target_context"]
    assert context["require_same_room"] is False
    assert context["allow_connected_room"] is True
    assert context["min_visible_pixels"] == 16
    assert context["min_visible_fraction"] == 0.0
    assert context["min_consecutive_observations"] == 1
    assert "target_require_same_room" not in context
    assert "target_allow_connected_room" not in context
    assert "target_min_visible_pixels" not in context


def test_native_eval_stops_on_first_official_success() -> None:
    assert NativeNavToObjEvalConfig.model_fields["end_on_success"].default is True


def test_json_eval_runner_applies_max_episodes_inside_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    episodes = [SimpleNamespace(house_index=index // 2) for index in range(6)]
    monkeypatch.setattr(json_eval_runner, "load_all_episodes", lambda _path: list(episodes))
    monkeypatch.setattr(ParallelRolloutRunner, "__init__", lambda _self, _config: None)
    config = SimpleNamespace(
        eval_runtime_params=SimpleNamespace(
            episode_idx=None,
            max_episodes=3,
            add_custom_object=False,
            custom_object_path=None,
            custom_object_name=None,
        ),
        task_sampler_config=SimpleNamespace(house_inds=None, samples_per_house=0),
        benchmark_path=None,
    )

    runner = json_eval_runner.JsonEvalRunner(config, tmp_path)
    assert sorted(runner._episodes_by_house) == [0, 1]
    assert sum(len(rows) for rows in runner._episodes_by_house.values()) == 3
    assert config.task_sampler_config.house_inds == [0, 1]

    logger = SimpleNamespace(error=lambda *_args, **_kwargs: None, info=lambda *_args, **_kwargs: None)
    selected, _ = json_eval_runner.JsonEvalRunner.load_episodes_for_house(
        config,
        house_id=1,
        batch_suffix="",
        worker_task_sampler=None,
        worker_logger=logger,
    )
    assert selected == [episodes[2]]
