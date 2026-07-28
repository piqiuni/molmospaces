"""Fast regression tests for the standalone InteractiveNav V3 evaluator.

These tests deliberately exercise only protocol, metrics, and zero-episode
resume behavior.  They never create a MuJoCo task or load a scene.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.InteractiveNav.evaluation import benchmark_runner
from scripts.InteractiveNav.evaluation.benchmark_metrics import (
    oracle_terminal_goal_consistency,
    path_length_bin,
    reference_path_length_m,
    spl,
    summarise_results,
)
from scripts.InteractiveNav.evaluation.benchmark_policies import (
    ScriptedOraclePolicy,
    normalize_policy_action,
)
from scripts.InteractiveNav.evaluation.benchmark_types import PolicyObservation, PublicEpisode


def _public_episode() -> PublicEpisode:
    return PublicEpisode(
        house_index=7,
        scene_dataset="procthor-10k",
        data_split="val",
        instruction="Find the target.",
        task_type="nav_to_obj",
        camera_names=["head_camera"],
        image_resolution=(640, 480),
    )


def _episode(requirement: str, validation: dict[str, float | None]) -> dict[str, object]:
    return {
        "interactive_nav": {
            "interaction_requirement": requirement,
            "generation_validation": {"navigation_validation": validation},
        }
    }


def _result_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "domains": ["channel"],
        "interaction_requirement": "required",
        "recipe": "channel_door_closed",
        "interaction_types": ["channel_hinged_door"],
        "path_length_bin": "[3,5)",
        "success": True,
        "nav_success": True,
        "required_interaction_success": True,
        "sequence_success": True,
        "non_interaction_success": None,
        "interaction_action_count": 3,
        "correct_interaction_action_count": 2,
        "step_count": 5,
        "navigation_path_length_m": 4.0,
        "reference_path_length_m": 3.0,
        "spl": 0.75,
        "total_simulated_seconds": 3.0,
        "extra_interaction_action_count": 1,
        "invalid_interaction_action_count": 0,
        "terminal_reason": "interactive_nav_success",
    }
    row.update(overrides)
    return row


def test_nav_sampler_skips_pick_only_joint_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler = object.__new__(benchmark_runner.V3BenchmarkTaskSampler)
    sampler.episode_spec = SimpleNamespace(task={"task_type": "nav_to_obj"})
    calls: list[object] = []
    monkeypatch.setattr(
        benchmark_runner.JsonEvalTaskSampler,
        "set_joint_values",
        lambda _self, env: calls.append(env),
    )

    sampler.set_joint_values(object())
    assert calls == []

    sampler.episode_spec.task["task_type"] = "pick"
    env = object()
    sampler.set_joint_values(env)
    assert calls == [env]


def test_normalize_policy_action_accepts_wrapped_visual_interaction() -> None:
    action = normalize_policy_action(
        {
            "action": {
                "kind": "interact",
                "target_pixel": [12.2, 9.7],
                "normalized_pixel": [0.25, 0.75],
                "joint_index": "3",
                "operation": "open",
            }
        }
    )

    assert action.kind == "interact"
    assert action.pixel_xy == (12, 10)
    assert action.normalized_pixel_xy == (0.25, 0.75)
    assert action.joint_index == 3
    assert action.object_name is None


def test_normalize_policy_action_strips_protocol_keys_from_base_action() -> None:
    action = normalize_policy_action(
        {
            "kind": "base",
            "base": [0.1, 0.0, 0.2],
            "pixel_xy": [11, 12],
            "camera_name": "head_camera",
            "done": False,
        }
    )

    assert action.kind == "base"
    assert action.base_action == {"base": [0.1, 0.0, 0.2]}
    assert normalize_policy_action(None).kind == "stop"
    assert normalize_policy_action({"head_qpos": [0.0, -0.2]}).kind == "view"


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"action": "not-a-mapping"}, TypeError),
        ({"kind": "interact", "pixel_xy": [1]}, ValueError),
        ({"kind": "unsupported"}, ValueError),
    ],
)
def test_normalize_policy_action_rejects_malformed_protocol(payload: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        normalize_policy_action(payload)


def test_scripted_oracle_preserves_navigation_gate_and_view_steps() -> None:
    policy = ScriptedOraclePolicy()
    policy.reset(_public_episode())
    policy.reset_oracle(
        [
            {"type": "navigate", "goal_point": [1.0, 2.0, 0.0], "goal_yaw": 0.5},
            {"type": "set_view", "head_qpos": [0.1, -0.2], "reason": "look"},
            {"type": "observe_target", "reason": "refresh"},
            {
                "type": "open_joint",
                "interaction_id": "door_1",
                "object_name": "private_door_name",
                "joint_name": "door_joint",
                "joint_index": 4,
            },
        ]
    )
    observation = PolicyObservation(None, "Find the target.", 0, 0.0, None)

    navigation = policy.act(observation)
    assert navigation.kind == "base"
    policy.notify_action_result(navigation, reached=False)
    assert policy.act(observation).kind == "base"
    policy.notify_action_result(navigation, reached=True)

    view = policy.act(observation)
    assert view.kind == "view"
    assert view.head_qpos == [0.1, -0.2]
    assert policy.act(observation).kind == "observe"
    interaction = policy.act(observation)
    assert interaction.kind == "interact"
    assert interaction.metadata["oracle_interaction_id"] == "door_1"
    assert policy.act(observation).kind == "stop"


def test_reference_path_length_and_spl_follow_v3_semantics() -> None:
    required = _episode(
        "required",
        {
            "oracle_restored_path_length_m": 6.0,
            "path_length_m": 5.0,
            "all_open_path_length_m": 4.0,
        },
    )
    unnecessary = _episode(
        "unnecessary",
        {
            "initial_state_path_length_m": 2.0,
            "path_length_m": 5.0,
        },
    )

    assert reference_path_length_m(required) == 6.0
    assert reference_path_length_m(unnecessary) == 2.0
    assert path_length_bin(None) is None
    assert path_length_bin(2.999) == "[0,3)"
    assert path_length_bin(3.0) == "[3,5)"
    assert path_length_bin(20.0) == "[20,inf)"
    assert spl(False, 3.0, 1.0) == 0.0
    assert spl(True, None, 1.0) is None
    assert spl(True, 3.0, 2.0) == 1.0
    assert spl(True, 3.0, 6.0) == 0.5


def _budget_episode(
    *,
    path_length_m: float | None,
    interaction_types: list[str],
    container_joint_count: int = 0,
    initial_target_visible: bool = False,
) -> dict[str, object]:
    interactions = [
        {"interaction_id": f"interaction_{index}", "type": interaction_type}
        for index, interaction_type in enumerate(interaction_types)
    ]
    navigation_validation = {}
    if path_length_m is not None:
        key = "oracle_restored_path_length_m" if interactions else "initial_state_path_length_m"
        navigation_validation[key] = path_length_m
    articulation_states = [
        {
            "object_name": "container_1",
            "joint_name": f"joint_{index}",
            "joint_index": index,
        }
        for index in range(container_joint_count)
    ]
    return {
        "task": {"robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
        "scene_modifications": {"articulation_states": articulation_states},
        "interactive_nav": {
            "interaction_requirement": "required" if interactions else "unnecessary",
            "initial_state": {"target_visible": initial_target_visible},
            "target": {"container_name": "container_1" if container_joint_count else None},
            "interactions": interactions,
            "oracle_plan": {
                "plan_id": "oracle_0",
                "required_interaction_ids": [row["interaction_id"] for row in interactions],
                "steps": [],
            },
            "generation_validation": {"navigation_validation": navigation_validation},
        },
    }


def test_dynamic_step_budget_matches_short_visible_example() -> None:
    config = benchmark_runner.BenchmarkEvaluationConfig(
        benchmark=Path("benchmark.json"),
        output_dir=Path("out"),
        max_steps=2000,
        step_budget_mode="dynamic",
    )

    hidden_budget, _hidden_basis = benchmark_runner.episode_step_budget(
        config,
        _budget_episode(
            path_length_m=3.0,
            interaction_types=["channel_hinged_door", "container_hinged_door"],
            container_joint_count=4,
            initial_target_visible=False,
        ),
    )
    budget, basis = benchmark_runner.episode_step_budget(
        config,
        _budget_episode(
            path_length_m=3.0,
            interaction_types=["channel_hinged_door", "container_hinged_door"],
            container_joint_count=4,
            initial_target_visible=True,
        ),
    )

    assert budget == 300
    assert hidden_budget == 800
    assert basis["initial_target_visible"] is True


def test_dynamic_step_budget_accounts_for_mixed_drawer_search() -> None:
    config = benchmark_runner.BenchmarkEvaluationConfig(
        benchmark=Path("benchmark.json"),
        output_dir=Path("out"),
        max_steps=2000,
        step_budget_mode="dynamic",
    )

    budget, basis = benchmark_runner.episode_step_budget(
        config,
        _budget_episode(
            path_length_m=11.5,
            interaction_types=["channel_hinged_door", "container_sliding_drawer"],
            container_joint_count=4,
        ),
    )

    assert budget == 1000
    assert basis["required_interaction_type_counts"] == {"channel": 1, "container": 1, "unknown": 0}
    assert basis["container_joint_count"] == 4


def test_dynamic_step_budget_clamps_and_fixed_mode_stays_compatible() -> None:
    dynamic = benchmark_runner.BenchmarkEvaluationConfig(
        benchmark=Path("benchmark.json"),
        output_dir=Path("out"),
        max_steps=2000,
        step_budget_mode="dynamic",
    )
    episode = _budget_episode(
        path_length_m=100.0,
        interaction_types=["channel_hinged_door", "container_hinged_door"],
        container_joint_count=12,
    )
    fixed = replace(dynamic, step_budget_mode="fixed", max_steps=777)

    assert benchmark_runner.episode_step_budget(dynamic, episode)[0] == 2000
    assert benchmark_runner.episode_step_budget(fixed, episode)[0] == 777


def test_dynamic_step_budget_uses_cap_when_path_is_missing() -> None:
    config = benchmark_runner.BenchmarkEvaluationConfig(
        benchmark=Path("benchmark.json"),
        output_dir=Path("out"),
        max_steps=2000,
        step_budget_mode="dynamic",
    )

    budget, basis = benchmark_runner.episode_step_budget(
        config,
        _budget_episode(
            path_length_m=None,
            interaction_types=["channel_hinged_door", "container_hinged_door"],
        ),
    )

    assert budget == 2000
    assert basis["conservative_fallback"] is True


def test_public_step_budget_basis_redacts_private_gt_inputs() -> None:
    public = benchmark_runner._public_step_budget_basis(
        {
            "formula_version": "v1",
            "mode": "dynamic",
            "effective_max_steps": 900,
            "reference_path_length_m": 12.5,
            "required_plan_id": "oracle_0",
            "required_interaction_count": 2,
            "container_joint_count": 7,
        }
    )

    assert public == {
        "evaluator_private": True,
        "formula_version": "v1",
        "mode": "dynamic",
        "effective_max_steps": 900,
        "conservative_fallback": False,
    }


def test_dynamic_config_enforces_the_2000_step_hard_cap() -> None:
    with pytest.raises(ValueError, match="<= 2000"):
        benchmark_runner.BenchmarkEvaluationConfig(
            benchmark=Path("benchmark.json"),
            output_dir=Path("out"),
            max_steps=2001,
            step_budget_mode="dynamic",
        ).validate()


def test_summary_groups_and_interaction_precision() -> None:
    rows = [
        _result_row(),
        _result_row(
            domains=["container"],
            interaction_requirement="unnecessary",
            recipe="container_visible",
            interaction_types=["container_hinged_door"],
            path_length_bin="[0,3)",
            success=False,
            non_interaction_success=False,
            interaction_action_count=0,
            correct_interaction_action_count=0,
            spl=0.25,
            terminal_reason="policy_stop",
        ),
    ]

    summary = summarise_results(rows)["groups"]
    assert summary["overall"]["episode_count"] == 2
    assert summary["overall"]["interaction_precision"] == pytest.approx(2 / 3)
    assert summary["overall"]["mean_spl"] == pytest.approx(0.5)
    assert summary["domain/channel"]["success_rate"] == 1.0
    assert summary["requirement/unnecessary"]["non_interaction_success_rate"] == 0.0
    assert summary["interaction_type/container_hinged_door"]["episode_count"] == 1

    no_interaction_summary = summarise_results(
        [_result_row(interaction_action_count=0, correct_interaction_action_count=0)]
    )
    assert no_interaction_summary["groups"]["overall"]["interaction_precision"] is None


def test_summary_excludes_runtime_ineligible_rows_from_formal_metrics() -> None:
    summary = summarise_results(
        [
            _result_row(success=True),
            _result_row(
                domains=["container"],
                success=False,
                scoring_eligible=False,
                scoring_exclusion_reasons=["terminal_goal_target_mismatch"],
                terminal_reason="runtime_goal_inconsistent",
            ),
        ]
    )

    assert summary["total_episode_count"] == 2
    assert summary["scoring_eligible_episode_count"] == 1
    assert summary["runtime_ineligible_episode_count"] == 1
    assert summary["runtime_ineligible_reason_counts"] == {
        "terminal_goal_target_mismatch": 1
    }
    assert summary["groups"]["overall"]["episode_count"] == 1
    assert "domain/container" not in summary["groups"]


def test_task_robot_base_pose_is_runtime_authoritative() -> None:
    episode = {
        "task": {
            "robot_base_pose": [5.0, 2.0, 0.0, 0.9950041653, 0.0, 0.0, 0.0998334166]
        },
        "robot": {"init_qpos": {"base": [1.0, 1.0, -0.5]}},
    }

    actual = benchmark_runner._expected_runtime_base_xyyaw(episode)

    assert actual.tolist() == pytest.approx([5.0, 2.0, 0.2])


def test_oracle_terminal_goal_consistency_detects_selected_instance_mismatch() -> None:
    class _Objects:
        def get_object_by_name(self, name: str) -> SimpleNamespace:
            assert name == "selected_target"
            return SimpleNamespace(position=[0.0, 0.0, 0.5])

    task = SimpleNamespace(env=SimpleNamespace(current_batch_index=0, object_managers=[_Objects()]))
    episode = {
        "interactive_nav": {
            "target": {"selected_instance": "selected_target"},
            "success_criteria": {"distance": {"threshold_m": 1.5}},
            "oracle_plan": {
                "plan_id": "oracle_0",
                "steps": [
                    {
                        "type": "navigate",
                        "reason": "satisfy_nav_to_obj_success",
                        "goal_point": [4.0, 0.0, 0.0],
                        "position_tolerance_m": 0.25,
                    }
                ],
            },
        }
    }

    consistency = oracle_terminal_goal_consistency(task, episode)
    assert consistency["checked"] is True
    assert consistency["consistent"] is False
    assert consistency["terminal_goal_candidates"][0]["distance_to_live_target_m"] == 4.0
    assert consistency["terminal_goal_candidates"][0]["allowed_distance_m"] == 1.75


def test_config_validation_and_index_selection_are_local() -> None:
    config = benchmark_runner.BenchmarkEvaluationConfig(
        benchmark=Path("benchmark.json"),
        output_dir=Path("out"),
        episode_indices=[2, 0, 1],
        max_episodes=2,
    )
    config.validate()
    assert benchmark_runner._selected_indices(config, [{}, {}, {}]) == [2, 0]
    with pytest.raises(IndexError, match="outside"):
        benchmark_runner._selected_indices(replace(config, episode_indices=[3]), [{}, {}, {}])
    with pytest.raises(ValueError, match="workers 1"):
        benchmark_runner.BenchmarkEvaluationConfig(
            benchmark=Path("benchmark.json"), output_dir=Path("out"), policy="ros_bridge", workers=2
        ).validate()
    with pytest.raises(ValueError, match="integer multiple"):
        benchmark_runner.BenchmarkEvaluationConfig(
            benchmark=Path("benchmark.json"), output_dir=Path("out"), policy_dt_ms=205.0
        ).validate()


def test_empty_benchmark_resume_and_signature_guard(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"episodes": []}))
    output_dir = tmp_path / "results"
    config = benchmark_runner.BenchmarkEvaluationConfig(benchmark=benchmark, output_dir=output_dir)

    first = benchmark_runner.run_evaluation(config)
    assert first["summary"]["result_count"] == 0
    assert (output_dir / "run_manifest.json").is_file()
    resumed = benchmark_runner.run_evaluation(replace(config, resume=True, workers=8))
    assert resumed["summary"]["run_signature"] == first["summary"]["run_signature"]
    with pytest.raises(ValueError, match="different benchmark/evaluation signature"):
        benchmark_runner.run_evaluation(replace(config, resume=True, max_steps=501))
