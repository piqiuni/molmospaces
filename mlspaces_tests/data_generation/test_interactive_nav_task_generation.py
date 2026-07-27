from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.InteractiveNav.generate_instruction_goal_v3 import (
    apply_instruction,
    extract_full_rollout_keyframes,
    rule_instruction,
    validate_model_instruction,
)
from scripts.InteractiveNav.generate_point_goal_v3 import (
    build_point_goal_episode,
    select_point_goal_candidate,
)
from scripts.InteractiveNav.interactive_nav_grounded_plan import (
    build_grounded_plan,
    build_path_corridor_graph,
    select_segment_keyframes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "scripts/InteractiveNav/dataset_definition/v3/examples"


def load_example(name: str) -> dict:
    episode = json.loads((EXAMPLES / name).read_text())
    episode["source"] = None
    episode["cameras"] = [
        {
            "name": "head_camera",
            "type": "robot_mounted",
            "reference_body_names": ["base"],
            "camera_offset": [0.0, 0.0, 0.0],
            "lookat_offset": [1.0, 0.0, 0.0],
            "camera_quaternion": [1.0, 0.0, 0.0, 0.0],
            "fov": 60.0,
        }
    ]
    return episode


def point_sample(interaction_aware: bool = True) -> dict:
    return {
        "goal_xy": [4.2, 1.0],
        "straight_line_distance_m": 4.0,
        "open_path": [[0.0, 0.0], [2.0, 0.0], [4.2, 1.0]],
        "open_path_length_m": 4.5,
        "closed_path": None if interaction_aware else [[0.0, 0.0], [4.2, 1.0]],
        "closed_path_length_m": None if interaction_aware else 4.5,
        "interaction_aware": interaction_aware,
        "candidate_index": 0,
        "attempted_candidate_count": 1,
        "failure_counts": {},
    }


def test_point_goal_candidate_requires_interaction_when_requested() -> None:
    candidates = [[1.0, 0.0], [3.0, 0.0]]

    def open_path(goal):
        return np.asarray([[0.0, 0.0], goal], dtype=float)

    def closed_path(goal):
        return None if float(goal[0]) >= 3.0 else open_path(goal)

    selected = select_point_goal_candidate(
        candidates,
        start_xy=[0.0, 0.0],
        open_path_fn=open_path,
        closed_path_fn=closed_path,
        rng=np.random.default_rng(3),
        min_distance_m=2.0,
        max_distance_m=None,
        interaction_aware=True,
        max_attempts=10,
    )

    assert selected["goal_xy"] == [3.0, 0.0]
    assert selected["closed_path"] is None
    assert selected["open_path"] is not None


def test_build_interaction_aware_point_goal_v3() -> None:
    episode = build_point_goal_episode(
        load_example("channel_episode.json"),
        sample=point_sample(True),
        source_episode_index=7,
        sampling_source="test_inflated_occupancy",
        clearance_m=0.3,
    )

    assert episode["task"]["task_type"] == "nav_to_point"
    assert episode["interactive_nav"]["target"]["target_type"] == "point"
    assert episode["interactive_nav"]["success_criteria"]["type"] == "nav_to_point"
    assert episode["interactive_nav"]["interactions"]
    assert episode["interactive_nav"]["oracle_plan"]["steps"][-1]["reason"] == (
        "satisfy_nav_to_point_success"
    )
    assert all(
        step["type"] != "observe_target"
        for step in episode["interactive_nav"]["oracle_plan"]["steps"]
    )


def test_build_reachable_point_goal_removes_interactions() -> None:
    episode = build_point_goal_episode(
        load_example("channel_episode.json"),
        sample=point_sample(False),
        source_episode_index=7,
        sampling_source="test_inflated_occupancy",
        clearance_m=0.3,
    )

    assert episode["interactive_nav"]["interaction_requirement"] == "unnecessary"
    assert episode["interactive_nav"]["interactions"] == []
    assert episode["interactive_nav"]["oracle_plan"]["required_interaction_ids"] == []


def test_raw_point_goal_preserves_null_parent_index() -> None:
    episode = build_point_goal_episode(
        load_example("channel_episode.json"),
        sample=point_sample(False),
        source_episode_index=None,
        sampling_source="raw_scene_inflated_occupancy_free_cell",
        clearance_m=0.3,
    )

    assert episode["interactive_nav"]["parent_benchmark_episode_index"] is None


def test_build_beneficial_point_goal_records_path_delta() -> None:
    sample = point_sample(True)
    sample["closed_path"] = [[0.0, 0.0], [3.0, 3.0], [4.2, 1.0]]
    sample["closed_path_length_m"] = 7.0
    episode = build_point_goal_episode(
        load_example("channel_episode.json"),
        sample=sample,
        source_episode_index=7,
        sampling_source="test_inflated_occupancy",
        clearance_m=0.3,
    )

    interactive = episode["interactive_nav"]
    assert interactive["interaction_requirement"] == "beneficial"
    assert interactive["generation_validation"]["navigation_validation"][
        "path_length_delta_m"
    ] == pytest.approx(2.5)
    assert all(
        "reduce_navigation_cost" in interaction["effect_types"]
        for interaction in interactive["interactions"]
    )


def test_rule_instruction_variants_are_grounded() -> None:
    source = load_example("mixed_episode.json")
    plan = build_grounded_plan(source)

    hidden = rule_instruction(plan, "hidden")
    explicit = rule_instruction(plan, "explicit")

    assert "open" not in hidden["instruction"].casefold()
    assert "door" not in hidden["instruction"].casefold()
    assert "door_29_0" not in hidden["grounded_entity_ids"]
    assert "open" in explicit["instruction"].casefold()
    assert explicit["grounded_entity_ids"]
    output = apply_instruction(source, explicit, generation_mode="rule")
    assert output["language"]["task_input_mode"] == "instruction"
    assert output["language"]["grounded_entity_ids"] == explicit["grounded_entity_ids"]


def test_instruction_overlay_accepts_grounded_graph_landmark() -> None:
    source = load_example("channel_episode.json")
    graph_context = {
        "nodes": [
            {"id": "room_1", "type": "room", "name": "kitchen"},
        ],
        "edges": [],
    }
    generated = {
        "instruction": "Go through the kitchen and find the mug.",
        "instruction_type": "route_instruction",
        "interaction_disclosure": "hidden",
        "grounded_entity_ids": ["room_1", "mug_target_19_0"],
        "grounded_plan_step_indices": [0, 2],
    }

    output = apply_instruction(
        source,
        generated,
        generation_mode="llm",
        graph_context=graph_context,
    )

    assert output["language"]["grounded_entity_ids"] == [
        "room_1",
        "mug_target_19_0",
    ]


def test_path_corridor_graph_filters_distant_landmarks() -> None:
    graph = {
        "nodes": [
            {"id": "scene", "type": "scene", "centroid": [0.0, 0.0, 0.0]},
            {"id": "near", "type": "object", "centroid": [1.0, 0.4, 0.0]},
            {"id": "far", "type": "object", "centroid": [1.0, 5.0, 0.0]},
            {"id": "required", "type": "portal", "centroid": [8.0, 8.0, 0.0]},
        ],
        "edges": [
            {"src_id": "scene", "relation": "has_child", "dst_id": "near"},
            {"src_id": "scene", "relation": "has_child", "dst_id": "far"},
        ],
    }
    corridor = build_path_corridor_graph(
        graph,
        [[0.0, 0.0], [2.0, 0.0]],
        radius_m=1.0,
        required_entity_ids=["required"],
    )

    ids = {node["id"] for node in corridor["nodes"]}
    assert ids == {"scene", "near", "required"}
    assert corridor["edges"] == [
        {"src_id": "scene", "relation": "has_child", "dst_id": "near"}
    ]


def test_model_instruction_rejects_unknown_grounding() -> None:
    plan = build_grounded_plan(load_example("channel_episode.json"))
    with pytest.raises(ValueError, match="unknown entities"):
        validate_model_instruction(
            {
                "instruction": "Open the imaginary gate.",
                "interaction_disclosure": "explicit",
                "grounded_entity_ids": ["imaginary_gate"],
                "grounded_plan_step_indices": [0],
            },
            plan=plan,
            disclosure="explicit",
            graph_context=None,
        )


def test_segment_keyframes_and_h5_extraction(tmp_path: Path) -> None:
    segments = ["initial", "nav", "nav", "nav", "open", "open"]
    assert select_segment_keyframes(segments) == [0, 1, 2, 3, 4, 5]
    trajectory = tmp_path / "trajectory.h5"
    with h5py.File(trajectory, "w") as handle:
        steps = handle.create_group("steps")
        text_dtype = h5py.string_dtype(encoding="utf-8")
        steps.create_dataset("segment", data=np.asarray(segments, dtype=object), dtype=text_dtype)
        images = steps.create_group("images")
        images.create_dataset(
            "head_camera",
            data=np.zeros((len(segments), 8, 12, 3), dtype=np.uint8),
        )

    paths, indices = extract_full_rollout_keyframes(
        trajectory, tmp_path / "frames"
    )

    assert indices == [0, 1, 2, 3, 4, 5]
    assert all(path.exists() for path in paths)
