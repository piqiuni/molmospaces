import json
from pathlib import Path

import numpy as np
import pytest

from scripts.InteractiveNav import interactive_nav_v3 as v3
from scripts.InteractiveNav.build_mixed_interaction_benchmark import (
    attach_channel_prerequisite,
    build_channel_interaction,
    build_minimal_plan_validation,
    candidate_sources,
    load_container_validation_cache,
    rejection_reason,
)
from scripts.InteractiveNav.collect_mixed_rough_catalog import (
    CANDIDATE_SELECTION,
    DOOR_APPROACH_SEMANTICS,
    MIXED_REQUIRED_ROLE,
    SELECTION_SCOPE,
    classify_non_crossing_pair,
    house_ids,
    path_door_approach,
    strict_pair_count,
    summarize_candidates,
)


def test_path_door_approach_uses_pre_entry_standoff() -> None:
    path = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    door = {"aabb_center": [1.0, 0.0, 1.0], "aabb_size": [0.2, 1.0, 2.0]}

    result = path_door_approach(
        path,
        door,
        padding_m=0.0,
        sample_step_m=0.1,
        standoff_m=0.5,
    )

    assert result is not None
    assert result["door_entry_distance_m"] >= 0.8
    assert result["approach_distance_from_start_m"] < result["door_entry_distance_m"]
    assert result["standoff_m"] >= 0.4


def test_mixed_rough_scope_includes_every_house_with_a_strict_pair() -> None:
    catalog = {
        "houses": [
            {
                "house_index": 1,
                "strict_pair_count": 2,
                "door_required": False,
                "containers": [
                    {"strict_contained_objects": [{"name": "a"}, {"name": "b"}]}
                ],
            },
            {
                "house_index": 2,
                "strict_pair_count": 1,
                "door_required": True,
                "containers": [{"strict_contained_objects": [{"name": "c"}]}],
            },
            {
                "house_index": 3,
                "strict_pair_count": 0,
                "containers": [{"strict_contained_objects": []}],
            },
        ]
    }

    assert house_ids(catalog, explicit=None, max_houses=None) == [1, 2]
    assert strict_pair_count(catalog["houses"][0]) == 2
    assert SELECTION_SCOPE == "all_container_rough_houses_with_strict_pairs"
    assert CANDIDATE_SELECTION == "all_open_gt_path_crosses_interactive_door_portal"
    assert MIXED_REQUIRED_ROLE.endswith("not_rough_input_gate")
    assert DOOR_APPROACH_SEMANTICS.endswith("not_manipulation_validated")


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({}, "no_available_source_episode"),
        (
            {"source_episode_available": 2},
            "no_open_path_from_any_source",
        ),
        (
            {"source_episode_available": 2, "open_path_found": 2},
            "open_paths_without_interactive_door",
        ),
    ],
)
def test_non_crossing_pair_reasons_are_stage_specific(
    counts: dict[str, int], expected: str
) -> None:
    from collections import Counter

    assert classify_non_crossing_pair(Counter(counts)) == expected


def test_channel_interaction_uses_measured_closed_and_open_readback() -> None:
    closed = {
        "object_name": "door_leaf",
        "joint_name": "door_joint",
        "joint_index": 0,
        "joint_position": 0.0,
        "open_fraction": 0.0,
    }
    opened = {**closed, "joint_position": 1.57, "open_fraction": 1.0}

    interaction = build_channel_interaction(
        case_id="case",
        root_name="door_root",
        closed_leaf=closed,
        opened_leaf=opened,
    )

    assert interaction["initial_joint_position"] == 0.0
    assert interaction["target_joint_position"] == 1.57
    assert interaction["initial_state"]["semantic_state"] == "closed"
    assert interaction["target_state"]["semantic_state"] == "open"

    beneficial = build_channel_interaction(
        case_id="case",
        root_name="door_root",
        closed_leaf=closed,
        opened_leaf=opened,
        interaction_requirement="beneficial",
    )
    assert beneficial["effect_types"] == ["reduce_navigation_cost"]


def test_channel_prerequisite_is_attached_to_first_container_interaction() -> None:
    interactions = [
        {
            "interaction_id": "container_0",
            "prerequisites": [],
        },
        {
            "interaction_id": "container_1",
            "prerequisites": [
                {"interaction_id": "container_0", "type": "mechanical"}
            ],
        },
    ]

    attach_channel_prerequisite(interactions, "channel_0")

    assert interactions[0]["prerequisites"] == [
        {"interaction_id": "channel_0", "type": "reachability"}
    ]
    assert interactions[1]["prerequisites"] == [
        {"interaction_id": "container_0", "type": "mechanical"}
    ]


def test_beneficial_mixed_minimal_validation_keeps_container_required() -> None:
    validation = build_minimal_plan_validation(
        channel_interaction={"interaction_id": "channel_0"},
        container_interactions=[
            {"interaction_id": "container_0", "joint_index": 0}
        ],
        selected={
            "joint_sequence": [0],
            "visibility_trace": [
                {"visibility_fraction": 0.0, "visible_pixels": 0}
            ],
        },
        initial_path_found=True,
        interaction_requirement="beneficial",
        path_evidence={
            "mixed_shortcut_verified": True,
            "path_length_delta_m": 2.0,
        },
    )

    assert validation["status"] == "passed"
    assert validation["omission_results"][0]["required"] is False
    assert validation["omission_results"][0]["beneficial"] is True
    assert validation["omission_results"][1]["required"] is True


def test_candidate_sources_can_pin_a_measured_source_variant() -> None:
    episodes = {3: {"episode": 3}, 4: {"episode": 4}}
    candidate = {
        "_preferred_source_episode_index": 4,
        "path_options": [
            {"source_episode_index": 3},
            {"source_episode_index": 4},
        ],
    }

    assert candidate_sources(candidate, episodes) == [(4, {"episode": 4})]


def test_container_validation_cache_indexes_reusable_trace(tmp_path: Path) -> None:
    benchmark = tmp_path / "container_benchmark.json"
    benchmark.write_text(
        json.dumps(
            [
                {
                    "house_index": 7,
                    "interactive_nav": {
                        "case_id": "container-case",
                        "parent_benchmark_episode_index": 13,
                        "target": {
                            "container_name": "drawer_a",
                            "selected_instance": "apple_a",
                        },
                        "generation_validation": {
                            "interaction_validations": [
                                {
                                    "controlling_joint_index": 2,
                                    "joint_sequence": [1, 2],
                                    "interaction_pose": [
                                        1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0
                                    ],
                                    "view_profile": "drawer_low_view",
                                    "visibility_trace": [
                                        {"visible_pixels": 0},
                                        {"visible_pixels": 4},
                                    ],
                                    "start_validation": {"valid": True},
                                }
                            ]
                        },
                    },
                }
            ]
        )
    )

    cache = load_container_validation_cache(benchmark)

    rows = cache[(7, "drawer_a", "apple_a")]
    assert len(rows) == 1
    assert rows[0]["joint_sequence"] == [1, 2]
    assert rows[0]["source_episode_index"] == 13
    assert rows[0]["source_case_id"] == "container-case"


def test_rejection_reason_keeps_actionable_failure_category() -> None:
    assert rejection_reason(
        ValueError("No measured mixed container visibility unlock: no_controlling_joint")
    ) == "no_measured_mixed_container_visibility_unlock"


def production_mixed_example() -> dict:
    path = Path("scripts/InteractiveNav/dataset_definition/v3/examples/mixed_episode.json")
    episode = json.loads(path.read_text())
    make_episode_schema_ready(episode)
    payload = episode["interactive_nav"]
    payload["target"]["grounding"]["unique"] = True
    payload["target"]["grounding"]["matching_instance_count"] = 1
    generation = payload["generation_validation"]
    generation["navigation_validation"] = {
        "all_open_path_found": True,
        "all_open_path_length_m": 7.4,
        "all_open_path_crossed_door_roots": ["door_root_0"],
        "initial_state_path_found": False,
        "approach_path_found": True,
        "oracle_restored_path_found": True,
        "start_visibility_fraction": 0.0,
        "start_visible_pixels": 0,
    }
    generation["door_state_validation"] = {
        "door_count": 1,
        "required_closed_root_names": ["door_root_0"],
        "all_required_closed": True,
        "doors": [
            {
                "interaction_id": "channel_door_0",
                "door_root_name": "door_root_0",
                "joint_name": "door_29_0_joint0",
                "joint_value": 0.0,
                "open_fraction": 0.0,
                "passed_closed": True,
            }
        ],
    }
    generation["container_state_validation"] = {
        "joint_count": 2,
        "all_closed": True,
        "joints": [
            {
                "joint_name": "refrigerator_29_0_joint3",
                "joint_value": 0.0,
                "open_fraction": 0.0,
                "passed": True,
            },
            {
                "joint_name": "refrigerator_29_0_joint1",
                "joint_value": 0.0,
                "open_fraction": 0.0,
                "passed": True,
            },
        ],
    }
    generation["minimal_plan_verified"] = True
    generation["minimal_plan_validation"] = {
        "status": "passed",
        "omission_results": [
            {"omitted_interaction_id": interaction_id, "required": True}
            for interaction_id in payload["oracle_plan"]["required_interaction_ids"]
        ],
    }
    payload["initial_state"].update(
        {
            "container_joints_closed": True,
            "target_visible": False,
        }
    )
    return episode


def make_episode_schema_ready(episode: dict) -> None:
    episode.pop("source", None)
    episode["cameras"] = [
        {
            "name": "head_camera",
            "type": "robot_mounted",
            "reference_body_names": ["robot_0/head"],
            "camera_offset": [0.0, 0.0, 0.0],
            "lookat_offset": [0.0, 0.0, 1.0],
            "camera_quaternion": [1.0, 0.0, 0.0, 0.0],
            "fov": 60.0,
        }
    ]


@pytest.mark.parametrize("example_name", ["channel_episode.json", "no_interaction_episode.json"])
def test_unified_validator_supports_channel_and_no_interaction(example_name: str) -> None:
    path = Path("scripts/InteractiveNav/dataset_definition/v3/examples") / example_name
    episode = json.loads(path.read_text())
    make_episode_schema_ready(episode)

    validated = v3.validate_interactive_nav_v3_episode(
        episode, expected_domains=["channel"]
    )

    assert validated["interactive_nav"]["interaction_domains"] == ["channel"]


def test_unified_validator_accepts_production_mixed_and_preserves_minimal_true() -> None:
    episode = production_mixed_example()

    validated = v3.validate_interactive_nav_v3_episode(episode)

    assert validated["interactive_nav"]["interaction_domains"] == [
        "channel",
        "container",
    ]
    assert validated["interactive_nav"]["generation_validation"][
        "minimal_plan_verified"
    ] is True


def test_mixed_validator_rejects_path_that_does_not_cross_required_door() -> None:
    episode = production_mixed_example()
    episode["interactive_nav"]["generation_validation"]["navigation_validation"][
        "all_open_path_crossed_door_roots"
    ] = []

    with pytest.raises(ValueError, match="all-open GT path"):
        v3.validate_mixed_v3_episode(episode)


def test_mixed_validator_rejects_unverified_minimal_plan() -> None:
    episode = production_mixed_example()
    episode["interactive_nav"]["generation_validation"]["minimal_plan_verified"] = None

    with pytest.raises(ValueError, match="verify plan necessity/benefit"):
        v3.validate_mixed_v3_episode(episode)


def test_unified_validator_accepts_measured_beneficial_mixed_episode() -> None:
    episode = production_mixed_example()
    payload = episode["interactive_nav"]
    payload["interaction_requirement"] = "beneficial"
    channel_interactions = [
        row for row in payload["interactions"] if row["type"].startswith("channel_")
    ]
    container_interactions = [
        row for row in payload["interactions"] if row["type"].startswith("container_")
    ]
    channel_ids = {row["interaction_id"] for row in channel_interactions}
    for row in channel_interactions:
        row["effect_types"] = ["reduce_navigation_cost"]
    for row in container_interactions:
        row["prerequisites"] = [
            prerequisite
            for prerequisite in row.get("prerequisites", [])
            if prerequisite["interaction_id"] not in channel_ids
        ]
    for step in payload["oracle_plan"]["steps"]:
        if step.get("type") == "open_joint" and step.get("interaction_id") in channel_ids:
            step["reason"] = "reduce_navigation_cost"
    payload["oracle_plans"] = [payload["oracle_plan"]]

    navigation = payload["generation_validation"]["navigation_validation"]
    navigation.update(
        {
            "initial_state_path_found": True,
            "initial_state_path_length_m": 10.0,
            "oracle_restored_path_length_m": 7.4,
            "path_length_delta_m": 2.6,
            "path_length_ratio_delta": 2.6 / 7.4,
            "shortcut_thresholds": {"min_delta_m": 0.25, "min_ratio": 0.02},
            "shortcut_verified": True,
        }
    )
    door_validation = payload["generation_validation"]["door_state_validation"]
    door_validation.update(
        {
            "closed_root_names": ["door_root_0"],
            "required_closed_root_names": [],
            "beneficial_closed_root_names": ["door_root_0"],
            "all_closed": True,
        }
    )
    payload["generation_validation"]["minimal_plan_validation"] = {
        "status": "passed",
        "omission_results": [
            {
                "omitted_interaction_id": row["interaction_id"],
                "required": False,
                "beneficial": True,
            }
            for row in channel_interactions
        ]
        + [
            {
                "omitted_interaction_id": row["interaction_id"],
                "required": True,
                "beneficial": False,
            }
            for row in container_interactions
        ],
    }

    validated = v3.validate_mixed_v3_episode(episode)

    assert validated["interactive_nav"]["interaction_requirement"] == "beneficial"


def test_rough_summary_reports_path_and_interaction_distributions() -> None:
    candidates = [
        {
            "house_index": 0,
            "all_open_path_length_m": 4.0,
            "rough_candidate_type": "mixed_required_verified",
            "mixed_required_verified": True,
            "selected_required_evidence": {
                "all_open_path_length_m": 4.0,
                "required_door_roots": ["door_0"],
            },
            "estimated_total_interaction_count": 2,
            "container_category": "Fridge",
            "target_category": "apple",
            "controlling_joint_type": "hinge",
            "estimated_container_interaction_count": 1,
            "estimated_channel_interaction_count": 1,
            "crossed_door_roots": ["door_0"],
        },
        {
            "house_index": 0,
            "all_open_path_length_m": 9.0,
            "rough_candidate_type": "door_crossing_only",
            "mixed_required_verified": False,
            "selected_required_evidence": None,
            "estimated_total_interaction_count": 3,
            "container_category": "Dresser",
            "target_category": "pen",
            "controlling_joint_type": "slide",
            "estimated_container_interaction_count": 2,
            "estimated_channel_interaction_count": 1,
            "crossed_door_roots": ["door_0", "door_1"],
        },
    ]
    houses = [
        {
            "source_container_pair_count": 4,
            "pair_result_counts": {
                "mixed_required_verified": 1,
                "door_crossing_only": 1,
                "open_paths_without_interactive_door": 2,
            },
            "source_path_status_counts": {
                "open_path_found": 4,
                "door_crossing_path_found": 2,
            },
            "door_requirement_status_counts": {
                "mixed_required_verified": 1,
                "goal_still_reachable_when_closed": 1,
            },
            "rejection_reason_counts": {
                "open_paths_without_interactive_door": 2
            },
        }
    ]

    summary = summarize_candidates(
        candidates,
        houses,
        [],
        elapsed_sec=1.0,
        expected_house_count=1,
        expected_pair_count=4,
    )

    assert summary["door_crossing_pair_count"] == 2
    assert summary["door_crossing_house_count"] == 1
    assert summary["mixed_required_pair_count"] == 1
    assert summary["mixed_required_house_count"] == 1
    assert summary["mixed_required_house_rate"] == 1.0
    assert summary["no_mixed_required_house_count"] == 0
    assert summary["door_crossing_only_pair_count"] == 1
    assert summary["mixed_required_within_crossing_rate"] == 0.5
    assert summary["path_length_m"]["median"] == 6.5
    assert summary["total_interaction_count"]["distribution"] == {"2": 1, "3": 1}
    assert summary["rejection_reason_counts"] == {
        "open_paths_without_interactive_door": 2
    }
    assert summary["pair_result_counts"] == {
        "mixed_required_verified": 1,
        "door_crossing_only": 1,
        "open_paths_without_interactive_door": 2,
    }
    assert summary["door_required_house_prefilter_used"] is False
    assert summary["resolved_strict_pair_count"] == 4
    assert summary["pair_coverage_complete"] is True
