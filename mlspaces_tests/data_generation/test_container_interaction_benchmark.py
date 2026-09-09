import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from molmo_spaces.evaluation.benchmark_schema import (
    InteractiveNavSpec,
    InteractiveNavV3Spec,
    NavToObjTaskSpec,
)
from molmo_spaces.tasks.nav_task import NavToObjTask
from scripts.InteractiveNav import interactive_nav_v3 as v3
from scripts.InteractiveNav.container_scene_probe import joint_closed_open_values
from scripts.InteractiveNav.build_container_interaction_benchmark import (
    articulation_initial_states,
    build_oracle_plan,
    generated_episode,
    semantic_open_fraction,
    slide_trace_has_consistent_partial_motion,
    visibility_trace_reveals_on_final_step,
)
from scripts.InteractiveNav.collect_container_fine_parallel import merge_results
from scripts.InteractiveNav.select_container_interaction_candidates import (
    build_dynamic_collection_plan,
)


def test_negative_hinge_range_maps_zero_to_closed() -> None:
    assert joint_closed_open_values([-1.5708, 0.0]) == (0.0, -1.5708)


def test_semantic_open_fraction_uses_measured_joint_value() -> None:
    assert semantic_open_fraction(0.0, 0.0, 1.5) == 0.0
    assert semantic_open_fraction(1.5, 0.0, 1.5) == 1.0
    assert semantic_open_fraction(-0.75, 0.0, -1.5) == 0.5


def visibility_row(value: float, pixels: int = 0) -> dict:
    return {"visibility_fraction": value, "visible_pixels": pixels}


def test_visibility_unlock_requires_final_step() -> None:
    valid, reason = visibility_trace_reveals_on_final_step(
        [visibility_row(0.0), visibility_row(0.0), visibility_row(0.01)],
        1e-4,
    )
    assert valid
    assert reason is None


def test_visibility_unlock_rejects_visible_prerequisite() -> None:
    valid, reason = visibility_trace_reveals_on_final_step(
        [visibility_row(0.0), visibility_row(0.01), visibility_row(0.02)],
        1e-4,
    )
    assert not valid
    assert reason == "target_visible_before_controlling_joint"


def test_articulation_initial_states_skip_free_joints(monkeypatch) -> None:
    class FakeProbe:
        @staticmethod
        def collect_door_records(_ctx):
            return []

        @staticmethod
        def joint_mujoco_type_name(_env, joint):
            return joint["joint_type"]

        @staticmethod
        def joint_value_by_name(_env, _joint_name):
            return 0.0

    monkeypatch.setattr(
        "scripts.InteractiveNav.build_container_interaction_benchmark.probe.collect_door_records",
        FakeProbe.collect_door_records,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.build_container_interaction_benchmark.probe.joint_mujoco_type_name",
        FakeProbe.joint_mujoco_type_name,
    )
    monkeypatch.setattr(
        "scripts.InteractiveNav.build_container_interaction_benchmark.probe.joint_value_by_name",
        FakeProbe.joint_value_by_name,
    )
    context = type("Context", (), {"env": object()})()
    containers = [
        {
            "name": "cabinet",
            "joints": [
                {
                    "joint_name": "cabinet_free",
                    "joint_index": 0,
                    "joint_type": "free",
                    "closed_value": 0.0,
                },
                {
                    "joint_name": "drawer_slide",
                    "joint_index": 1,
                    "joint_type": "slide",
                    "closed_value": 0.0,
                    "open_value": 0.5,
                },
            ],
        }
    ]

    states = articulation_initial_states(context, containers)

    assert [state["joint_name"] for state in states] == ["drawer_slide"]


def test_visibility_unlock_rejects_initial_visibility() -> None:
    valid, reason = visibility_trace_reveals_on_final_step(
        [visibility_row(0.01), visibility_row(0.02)],
        1e-4,
    )
    assert not valid
    assert reason == "target_visible_before_interaction"


def test_small_object_accepts_first_positive_pixel() -> None:
    valid, reason = visibility_trace_reveals_on_final_step(
        [visibility_row(0.0), visibility_row(1e-6, pixels=1)],
        1e-4,
        min_visible_pixels=1,
    )
    assert valid
    assert reason is None


def test_fridge_visibility_accepts_first_positive_pixel() -> None:
    valid, reason = visibility_trace_reveals_on_final_step(
        [visibility_row(0.0, pixels=0), visibility_row(1e-8, pixels=1)],
        1e-4,
        min_visible_pixels=1,
    )
    assert valid
    assert reason is None


def test_partial_slide_motion_accepts_following_object() -> None:
    trace = [
        {
            "target_joint_aabb": {"center": [0.0, 0.0, 0.0]},
            "object_position": [0.0, 0.0, 0.0],
        },
        {
            "target_joint_aabb": {"center": [0.2, 0.0, 0.0]},
            "object_position": [0.12, 0.0, 0.0],
        },
    ]

    valid, metrics = slide_trace_has_consistent_partial_motion(trace)

    assert valid
    assert metrics["joint_motion_m"] == 0.2
    assert metrics["motion_ratio"] == 0.6


def test_low_view_precedes_open_joint_in_oracle_plan() -> None:
    selected = {
        "robot_pose": np.eye(4),
        "joint_sequence": [2],
        "view_profile": "drawer_low_view",
        "view_state": {"head_qpos": [0.0, 0.3], "torso_qpos": [0.35] * 6},
    }
    container = {
        "name": "cabinet",
        "joints": [{"joint_index": 2, "joint_name": "drawer_joint"}],
    }

    plan = build_oracle_plan(container, selected, "target", 1e-4)

    assert [step["type"] for step in plan["steps"]] == [
        "navigate",
        "set_view",
        "open_joint",
        "observe_target",
    ]
    assert plan["plan_id"] == "oracle_0"
    assert plan["steps"][0]["reason"] == "approach_container_interaction"
    assert plan["steps"][-1]["visibility_threshold"] == 0.0


def test_slide_controlling_joint_uses_force_control() -> None:
    selected = {
        "robot_pose": np.eye(4),
        "joint_sequence": [0, 2],
        "joint": {"joint_index": 2, "joint_name": "drawer_joint"},
        "joint_type": "slide",
        "view_profile": "default",
        "view_state": {"head_qpos": None, "torso_qpos": None},
    }
    container = {
        "name": "fridge",
        "joints": [
            {"joint_index": 0, "joint_name": "outer_door"},
            {"joint_index": 2, "joint_name": "drawer_joint"},
        ],
    }

    plan = build_oracle_plan(container, selected, "target", 1e-4)
    open_steps = [step for step in plan["steps"] if step["type"] == "open_joint"]

    assert [step["control_mode"] for step in open_steps] == ["force", "force"]


def test_interactive_nav_schema_supports_multiple_oracles() -> None:
    plan = {
        "steps": [
            {
                "type": "observe_target",
                "object_name": "egg",
                "visibility_threshold": 1e-4,
                "reason": "verify_target_visible",
            }
        ]
    }
    spec = InteractiveNavSpec.model_validate(
        {
            "interaction_domain": "container",
            "case_id": "multi-oracle",
            "target": {},
            "initial_state": {},
            "oracle_plan": plan,
            "oracle_plans": [plan, plan],
        }
    )

    assert len(spec.oracle_plans) == 2


def test_v3_container_interactions_encode_mechanical_dependency() -> None:
    container = {
        "name": "fridge",
        "category": "Fridge",
        "joints": [
            {"joint_index": 0, "joint_name": "outer", "joint_type": "hinge"},
            {"joint_index": 2, "joint_name": "drawer", "joint_type": "slide"},
        ],
    }
    interactions, ids = v3.build_container_interactions(
        container=container,
        oracle_candidates=[{"joint_sequence": [0, 2]}],
        articulation_states=[
            {"joint_name": "outer", "open_fraction": 0.0},
            {"joint_name": "drawer", "open_fraction": 0.0},
        ],
    )

    assert [row["type"] for row in interactions] == [
        "container_hinged_door",
        "container_sliding_drawer",
    ]
    assert interactions[0]["effect_types"] == ["enable_interaction"]
    assert interactions[1]["prerequisites"] == [
        {"interaction_id": ids[0], "type": "mechanical"}
    ]


def test_v3_container_interactions_accept_mujoco_numeric_joint_types() -> None:
    interactions, _ = v3.build_container_interactions(
        container={
            "name": "cabinet",
            "category": "Dresser",
            "joints": [
                {"joint_index": 0, "joint_name": "drawer", "joint_type": [2]},
                {"joint_index": 1, "joint_name": "door", "joint_type": [3]},
            ],
        },
        oracle_candidates=[{"joint_sequence": [1, 0]}],
        articulation_states=[
            {"joint_name": "drawer", "open_fraction": 0.0},
            {"joint_name": "door", "open_fraction": 0.0},
        ],
    )

    assert [interaction["type"] for interaction in interactions] == [
        "container_hinged_door",
        "container_sliding_drawer",
    ]


def test_v3_task_selection_mode_is_explicit_and_round_trips() -> None:
    task = NavToObjTaskSpec.model_validate(
        {
            "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
            "task_type": "nav_to_obj",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "selection_mode": "specific_instance",
            "pickup_obj_name": "apple_0",
            "pickup_obj_candidates": ["apple_0"],
            "succ_pos_threshold": 1.5,
        }
    )

    assert task.model_dump()["selection_mode"] == "specific_instance"


def test_specific_instance_runtime_discards_other_candidates() -> None:
    task = NavToObjTask.__new__(NavToObjTask)
    task.config = SimpleNamespace(
        task_config=SimpleNamespace(
            selection_mode="specific_instance",
            pickup_obj_name="apple_0",
            pickup_obj_candidates=["apple_0", "apple_1"],
        )
    )

    task._reconstruct_candidate_list_if_needed(object())

    assert task.config.task_config.pickup_obj_candidates == ["apple_0"]


def test_v3_schema_round_trip_preserves_interactions() -> None:
    payload = json.loads(
        Path(
            "scripts/InteractiveNav/dataset_definition/v3/examples/container_episode.json"
        ).read_text()
    )["interactive_nav"]

    spec = InteractiveNavV3Spec.model_validate(payload)

    assert spec.schema_version == "interactive_nav_v3"
    assert spec.interactions[0].effect_types == ["reveal_target_object"]


def test_generated_episode_emits_valid_container_v3() -> None:
    robot_pose = np.eye(4)
    object_position = [0.5, 0.0, 0.5]
    template = {
        "house_index": 1,
        "scene_dataset": "procthor-10k",
        "data_split": "val",
        "seed": 1,
        "robot": {"robot_name": "rby1", "init_qpos": {}},
        "img_resolution": [640, 480],
        "cameras": [
            {
                "name": "head_camera",
                "type": "robot_mounted",
                "reference_body_names": ["robot_0/head"],
                "camera_offset": [0.0, 0.0, 0.0],
                "lookat_offset": [0.0, 0.0, 1.0],
                "camera_quaternion": [1.0, 0.0, 0.0, 0.0],
                "fov": 60.0,
            }
        ],
        "scene_modifications": {
            "added_objects": {},
            "object_poses": {},
            "removed_objects": [],
            "articulation_states": [],
        },
        "task": {
            "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
            "task_type": "nav_to_obj",
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "pickup_obj_name": "placeholder",
            "pickup_obj_candidates": ["placeholder"],
            "succ_pos_threshold": 1.5,
        },
        "language": {"task_description": "find the placeholder."},
    }
    source_episode = {
        "task": {
            "robot_base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "succ_pos_threshold": 1.5,
        }
    }
    container = {
        "name": "fridge_0",
        "category": "Fridge",
        "aabb_center": [0.5, 0.0, 1.0],
        "aabb_size": [1.0, 1.0, 2.0],
        "joints": [
            {"joint_index": 0, "joint_name": "fridge_joint", "joint_type": "hinge"}
        ],
    }
    object_record = {
        "name": "apple_0",
        "category": "apple",
        "aabb_center": object_position,
        "aabb_size": [0.1, 0.1, 0.1],
    }
    selected = {
        "joint": container["joints"][0],
        "joint_sequence": [0],
        "robot_pose": robot_pose,
        "pose_meta": {"candidate_label": "front"},
        "view_profile": "default",
        "view_state": {"head_qpos": None, "torso_qpos": None},
        "visibility_trace": [
            {"visibility_fraction": 0.0, "visible_pixels": 0, "object_position": object_position},
            {"visibility_fraction": 0.01, "visible_pixels": 10, "object_position": object_position},
        ],
        "joint_type": "hinge",
        "binding": {"applicable": False, "consistent": True},
        "start_validation": {
            "valid": True,
            "path_found": True,
            "path_length_m": 2.0,
            "start_visibility_fraction": 0.0,
            "start_visible_pixels": 0,
        },
    }

    episode = generated_episode(
        template,
        source_episode,
        3,
        "case_0",
        container,
        object_record,
        selected,
        [selected],
        {},
        [
            {
                "object_name": "fridge_0",
                "joint_name": "fridge_joint",
                "joint_index": 0,
                "position": 0.0,
                "open_fraction": 0.0,
            }
        ],
        selected["start_validation"],
        selected["binding"],
        1e-4,
        2,
        {"threshold": 0.99, "door_count": 0, "all_open": True, "doors": []},
        {"threshold": 0.01, "joint_count": 1, "all_closed": True, "joints": [
            {
                "container_name": "fridge_0",
                "joint_name": "fridge_joint",
                "joint_value": 0.0,
                "closed_value": 0.0,
                "open_value": 1.0,
                "open_fraction": 0.0,
                "passed": True,
            }
        ]},
    )

    assert episode["interactive_nav"]["schema_version"] == "interactive_nav_v3"
    assert episode["task"]["selection_mode"] == "specific_instance"
    assert episode["interactive_nav"]["target"]["grounding"] == {
        "unique": False,
        "matching_instance_count": 2,
        "description": "apple",
        "attributes": {},
    }
    assert episode["interactive_nav"]["generation_validation"]["success_evidence"][
        "expected_task_success"
    ]


def _rough_house(house_index: int, object_count: int) -> dict:
    return {
        "house_index": house_index,
        "containers": [
            {
                "name": f"fridge_{house_index}",
                "category": "Fridge",
                "asset_id": "Fridge_Test",
                "aabb_size": [1.0, 1.0, 2.0],
                "strict_contained_objects": [
                    {
                        "name": f"apple_{house_index}_{index}",
                        "category": "apple",
                        "aabb_size": [0.1, 0.1, 0.1],
                        "source_starts": [
                            {
                                "episode_index": house_index * 10 + index,
                                "distance_to_object_m": 4.0 + index,
                                "planar_distance_to_object_m": 4.0 + index,
                            }
                        ],
                    }
                    for index in range(object_count)
                ],
            }
        ],
    }


def test_dynamic_plan_keeps_all_house_fallback_candidates(tmp_path: Path) -> None:
    catalog = tmp_path / "rough_catalog.json"
    catalog.write_text(
        json.dumps({"houses": [_rough_house(1, 4), _rough_house(2, 3)]})
    )

    plan = build_dynamic_collection_plan(
        catalog,
        max_samples=4,
        samples_per_house=2,
        target_house_count=2,
        seed=0,
    )

    assert plan["selection"]["requested_sample_count"] == 4
    assert [row["target_sample_count"] for row in plan["houses"]] == [2, 2]
    assert sorted(len(row["candidates"]) for row in plan["houses"]) == [3, 4]


def test_parallel_merge_retains_partial_fixed_house(tmp_path: Path) -> None:
    output_dir = tmp_path / "merged"
    shard_dir = tmp_path / "shard" / "benchmark"
    shard_dir.mkdir(parents=True)
    valid_pairs = [
        {"case_id": "h1a", "house_index": 1},
        {"case_id": "h1b", "house_index": 1},
        {"case_id": "h2a", "house_index": 2},
    ]
    benchmark = [
        {
            "interactive_nav": {
                "schema_version": "interactive_nav_v3",
                "case_id": row["case_id"],
            }
        }
        for row in valid_pairs
    ]
    for name, payload in {
        "benchmark.json": benchmark,
        "valid_pairs.json": valid_pairs,
        "rejected_pairs.json": [],
        "house_catalog.json": [],
        "fridge_slide_compartment_candidates.json": [],
        "failures.json": [],
    }.items():
        (shard_dir / name).write_text(json.dumps(payload))
    plan = {
        "selection": {},
        "houses": [
            {"house_index": 1, "target_sample_count": 2, "candidates": []},
            {"house_index": 2, "target_sample_count": 2, "candidates": []},
        ],
    }
    args = Namespace(output_dir=output_dir, workers=1)

    summary = merge_results(
        args,
        plan,
        [{"shard_index": 0, "output_dir": str(shard_dir.parent)}],
        elapsed_sec=1.0,
    )

    assert summary["generated_episode_count"] == 3
    assert summary["complete_collection_house_count"] == 1
    assert summary["partial_collection_house_count"] == 1
    assert len(json.loads((output_dir / "benchmark.json").read_text())) == 3
