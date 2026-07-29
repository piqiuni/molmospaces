from __future__ import annotations

import math

from semantic_decision_py_pkg.behavior_candidates import (
    CandidateGenerator,
    CandidateGeneratorConfig,
)


def test_generator_combines_frontiers_and_closed_portals() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(interaction_types=("portal",), portal_standoff_m=1.15)
    )
    explorer_status = {
        "frontier_clusters": [
            {
                "cluster_id": "frontier_1",
                "subgoal_world": [5.0, 0.0],
                "subgoal_yaw": 0.0,
                "centroid_world": [5.5, 0.0],
                "information_gain": 20.0,
                "distance_to_robot": 5.0,
                "score": 1.0,
                "score_terms": {},
                "cell_count": 20,
            }
        ]
    }
    graph = {
        "nodes": [
            {
                "id": "portal_1",
                "type": "portal",
                "name": "double_door",
                "centroid": [2.0, 0.0, 1.0],
                "state_age_sec": 1.0,
                "is_currently_visible": True,
                "attributes": {
                    "source_object_name": "double_door_root",
                    "connected_room_ids": [1],
                    "connectivity_status": "partial",
                },
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                    "interaction_cost": 1.0,
                    "interaction_mode": "open_close",
                },
            },
            {
                "id": "portal_open",
                "type": "portal",
                "centroid": [1.0, 2.0, 1.0],
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": False,
                    "state": "open",
                    "state_confidence": 1.0,
                },
            },
        ]
    }
    candidates = generator.generate(explorer_status, graph, robot_xy=(0.0, 0.0))
    assert [candidate.behavior_type for candidate in candidates] == ["EXPLORE", "INTERACT"]
    interaction = next(candidate for candidate in candidates if candidate.behavior_type == "INTERACT")
    assert interaction.target_name == "double_door_root"
    assert math.isclose(interaction.goal_xyyaw[0], 0.85, abs_tol=1e-6)
    assert interaction.metadata["requires_approach"] is True


def test_successfully_opened_portal_generates_one_way_traversal_goal() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("portal",),
            portal_traversal_distance_m=0.9,
            portal_traversal_max_start_distance_m=2.0,
            portal_traversal_completion_margin_m=0.35,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "portal_door_1",
                "type": "portal",
                "name": "door_1",
                "aabb_center": [0.0, 0.0, 1.0],
                "aabb_size": [0.1, 0.9, 2.0],
                "attributes": {
                    "source_object_name": "door_1",
                    "interaction_reference_aabb_center": [0.0, 0.0, 1.0],
                    "interaction_reference_aabb_size": [0.1, 0.9, 2.0],
                    "connected_room_ids": [1, 1000000],
                    "portal_child_room_id": 1000000,
                    "portal_child_source_room_id": 1,
                    "potential_room_ids": [1000000],
                },
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": False,
                    "state": "open",
                    "state_confidence": 1.0,
                    "traversable": True,
                    "operation_history": [
                        {
                            "event_id": "open_event_1",
                            "action": "open",
                            "success": True,
                            "post_state": "open",
                            "approach_goal_xyyaw": [-1.0, 0.0, 0.0],
                        }
                    ],
                },
            }
        ]
    }

    candidates = generator.generate({}, graph, robot_xy=(-1.0, 0.0))

    assert len(candidates) == 1
    traversal = candidates[0]
    assert traversal.candidate_id == "traverse:portal_door_1:open_event_1"
    assert traversal.behavior_type == "NAVIGATE"
    assert traversal.source == "post_interaction_portal"
    assert traversal.goal_xyyaw == [0.9, 0.0, 0.0]
    assert traversal.metadata["post_interaction_traversal"] is True
    assert traversal.metadata["target_room_id"] == 1000000
    assert traversal.metadata["source_room_id"] == 1
    assert traversal.metadata["potential_room"] is True

    # Once the base is clearly beyond the original door plane, the traversal
    # candidate cannot reverse and send it back through the same portal.
    assert generator.generate({}, graph, robot_xy=(0.4, 0.0)) == []

    graph["nodes"][0]["interaction"]["operation_history"][0][
        "approach_goal_xyyaw"
    ] = [0.0, 1.0, -math.pi / 2.0]
    opposite_axis = generator.generate({}, graph, robot_xy=(0.0, 1.0))[0]
    assert math.isclose(opposite_axis.goal_xyyaw[0], 0.0, abs_tol=1e-6)
    assert math.isclose(opposite_axis.goal_xyyaw[1], -0.9, abs_tol=1e-6)
    assert math.isclose(opposite_axis.goal_xyyaw[2], -math.pi / 2.0, abs_tol=1e-6)


def test_frontier_just_beyond_opened_portal_keeps_potential_room_identity() -> None:
    generator = CandidateGenerator()
    explorer_status = {
        "proposals": [
            {
                "proposal_id": "child_frontier",
                "goal_xyyaw": [0.9, 0.0, 0.0],
                "frontier_point": [1.0, 0.0],
                "raw_features": {
                    "information_gain": 4.0,
                    "unknown_component_area_m2": 2.0,
                    "expected_visible_unknown_area_m2": 1.0,
                    "distance_m": 0.9,
                },
                "geometry": {"hard_constraints_passed": True},
            }
        ]
    }
    graph = {
        "nodes": [
            {
                "id": "room_1000000",
                "type": "room",
                "room_id": 1000000,
                "centroid": [0.9, 0.0, 0.1],
                "aabb_center": [0.9, 0.0, 0.1],
                "aabb_size": [1.2, 1.2, 0.2],
                "attributes": {
                    "active": True,
                    "is_potential_room": True,
                    "source_portal_id": "portal_door_1",
                    "observed_free_space": False,
                },
            }
        ]
    }

    candidates = generator.generate(explorer_status, graph, robot_xy=(0.0, 0.0))
    frontier = next(candidate for candidate in candidates if candidate.behavior_type == "EXPLORE")

    assert frontier.metadata["target_room_id"] == 1000000
    assert frontier.metadata["room_assignment_source"] == "portal_open_potential_child"
    assert frontier.metadata["potential_room"] is True


def test_frontiers_include_physical_current_and_target_room_metadata() -> None:
    generator = CandidateGenerator(CandidateGeneratorConfig(max_frontier_candidates=2))
    candidates = generator.generate(
        {
            "proposals": [
                {
                    "proposal_id": "same_room",
                    "goal_xyyaw": [0.5, 0.0, 0.0],
                    "frontier_point": [0.6, 0.0],
                    "raw_features": {"distance_m": 0.5},
                },
                {
                    "proposal_id": "door_child",
                    "goal_xyyaw": [3.0, 0.0, 0.0],
                    "frontier_point": [3.1, 0.0],
                    "raw_features": {"distance_m": 3.0},
                },
            ]
        },
        {
            "nodes": [
                {
                    "id": "room_1",
                    "type": "room",
                    "room_id": 1,
                    "aabb_center": [0.0, 0.0, 0.0],
                    "aabb_size": [2.0, 2.0, 1.0],
                    "attributes": {"active": True},
                },
                {
                    "id": "room_2",
                    "type": "room",
                    "room_id": 2,
                    "aabb_center": [3.0, 0.0, 0.0],
                    "aabb_size": [1.5, 1.5, 1.0],
                    "attributes": {
                        "active": True,
                        "is_potential_room": True,
                        "source_portal_id": "portal_1",
                        "room_attribute": "kitchen",
                        "room_attribute_confidence": 0.9,
                    },
                },
            ]
        },
        robot_xy=(0.0, 0.0),
    )
    by_id = {candidate.candidate_id: candidate for candidate in candidates}

    same_room = by_id["frontier:same_room"].metadata
    assert same_room["robot_room_id"] == 1
    assert same_room["current_room_id"] == 1
    assert same_room["target_room_id"] == 1
    assert same_room["room_relation"] == "current_room"

    child = by_id["frontier:door_child"].metadata
    assert child["robot_room_id"] == 1
    assert child["target_room_id"] == 2
    assert child["potential_room"] is True
    assert child["source_portal_id"] == "portal_1"
    assert child["room_attribute"] == "kitchen"


def test_frontier_cap_reserves_other_room_proposal_before_global_area_fill() -> None:
    generator = CandidateGenerator(CandidateGeneratorConfig(max_frontier_candidates=1))
    candidates = generator.generate(
        {
            "proposals": [
                {
                    "proposal_id": "large_current",
                    "goal_xyyaw": [0.5, 0.0, 0.0],
                    "frontier_point": [0.5, 0.0],
                    "raw_features": {
                        "distance_m": 0.5,
                        "unknown_component_area_m2": 50.0,
                        "expected_visible_unknown_area_m2": 40.0,
                    },
                },
                {
                    "proposal_id": "small_child",
                    "goal_xyyaw": [3.0, 0.0, 0.0],
                    "frontier_point": [3.0, 0.0],
                    "raw_features": {
                        "distance_m": 3.0,
                        "unknown_component_area_m2": 2.0,
                        "expected_visible_unknown_area_m2": 1.0,
                    },
                },
            ]
        },
        {
            "nodes": [
                {
                    "id": "room_1",
                    "type": "room",
                    "room_id": 1,
                    "aabb_center": [0.0, 0.0, 0.0],
                    "aabb_size": [2.0, 2.0, 1.0],
                    "attributes": {"active": True},
                },
                {
                    "id": "room_2",
                    "type": "room",
                    "room_id": 2,
                    "aabb_center": [3.0, 0.0, 0.0],
                    "aabb_size": [1.5, 1.5, 1.0],
                    "attributes": {"active": True, "is_potential_room": True},
                },
            ]
        },
        robot_xy=(0.0, 0.0),
    )

    assert [candidate.candidate_id for candidate in candidates] == ["frontier:small_child"]


def test_generator_does_not_use_object_name_to_reject_portal_fixture() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(interaction_types=("portal",))
    )
    graph = {
        "nodes": [
            {
                "id": "portal_doorframe_1",
                "type": "portal",
                "name": "Door",
                "centroid": [2.0, 0.0, 1.0],
                "state_age_sec": 1.0,
                "is_currently_visible": True,
                "attributes": {
                    "source_object_name": "doorframe_static_1",
                    "connected_room_ids": [1, 2],
                },
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "unknown",
                    "state_confidence": 1.0,
                },
            }
        ]
    }

    candidates = generator.generate({}, graph, robot_xy=(0.0, 0.0))
    assert len(candidates) == 1
    assert candidates[0].target_id == "portal_doorframe_1"


def test_portal_open_waits_for_ready_module1_state_in_full_mllm_mode() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("portal",),
            portal_require_attribute_ready=True,
            portal_allow_unknown_state=False,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "portal_1",
                "type": "portal",
                "centroid": [2.0, 0.0, 1.0],
                "state_age_sec": 0.0,
                "attributes": {"attribute_status": "pending"},
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            }
        ]
    }

    assert generator.generate({}, graph, robot_xy=(0.0, 0.0)) == []

    graph["nodes"][0]["attributes"]["attribute_status"] = "ready"
    candidates = generator.generate({}, graph, robot_xy=(0.0, 0.0))
    assert [candidate.target_id for candidate in candidates] == ["portal_1"]

    # A refresh publishes pending before Module 1 returns.  Keep the last
    # successful closed-door evidence eligible rather than making the portal
    # flicker out of the candidate stream.
    graph["nodes"][0]["attributes"].update(
        {
            "attribute_status": "pending",
            "attribute_source": "mllm_attribute_inference",
            "attribute_confidence": 0.9,
            "interaction_state_override": {
                "state": "closed",
                "state_source": "mllm_attribute_inference",
                "state_confidence": 0.9,
            },
        }
    )
    candidates = generator.generate({}, graph, robot_xy=(0.0, 0.0))
    assert [candidate.target_id for candidate in candidates] == ["portal_1"]

    graph["nodes"][0]["interaction"]["state"] = "unknown"
    assert generator.generate({}, graph, robot_xy=(0.0, 0.0)) == []


def test_pending_portal_refresh_requires_valid_previous_module1_evidence() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("portal",),
            portal_require_attribute_ready=True,
            portal_allow_unknown_state=False,
            max_state_age_sec=5.0,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "portal_1",
                "type": "portal",
                "centroid": [2.0, 0.0, 1.0],
                "state_age_sec": 1.0,
                "attributes": {
                    "attribute_status": "pending",
                    "attribute_source": "mllm_attribute_inference",
                    "attribute_confidence": 0.4,
                    "interaction_state_override": {
                        "state": "closed",
                        "state_source": "mllm_attribute_inference",
                        "state_confidence": 0.4,
                    },
                },
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 0.9,
                },
            }
        ]
    }

    # A pending request alone, or a weak previous result, is not sufficient.
    assert generator.generate({}, graph, robot_xy=(0.0, 0.0)) == []

    graph["nodes"][0]["attributes"]["attribute_confidence"] = 0.9
    graph["nodes"][0]["attributes"]["interaction_state_override"][
        "state_confidence"
    ] = 0.9
    graph["nodes"][0]["state_age_sec"] = 6.0
    assert generator.generate({}, graph, robot_xy=(0.0, 0.0)) == []


def test_portal_approach_uses_stable_closed_reference_geometry() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("portal",),
            portal_standoff_m=1.0,
            interaction_safety_margin_m=0.0,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "portal_1",
                "type": "portal",
                "centroid": [2.0, 0.0, 1.0],
                "aabb_center": [2.0, 0.5, 1.0],
                "aabb_size": [1.0, 1.0, 2.0],
                "state_age_sec": 0.0,
                "attributes": {
                    "interaction_reference_aabb_center": [2.0, 0.0, 1.0],
                    "interaction_reference_aabb_size": [0.1, 1.0, 2.0],
                },
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "ajar",
                    "state_confidence": 1.0,
                },
            }
        ]
    }

    candidate = generator.generate({}, graph, robot_xy=(0.0, 0.0))[0]

    assert math.isclose(candidate.goal_xyyaw[0], 0.95, abs_tol=1e-6)
    assert math.isclose(candidate.goal_xyyaw[1], 0.0, abs_tol=1e-6)
    assert math.isclose(candidate.goal_xyyaw[2], 0.0, abs_tol=1e-6)


def test_interaction_always_requires_navigation_to_pose() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("portal",),
            portal_standoff_m=1.0,
            interaction_safety_margin_m=0.0,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "portal_1",
                "type": "portal",
                "centroid": [1.0, 0.0, 1.0],
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            }
        ]
    }

    candidate = generator.generate({}, graph, robot_xy=(0.0, 0.0))[0]

    assert candidate.metadata["requires_approach"] is True


def test_raw_frontier_proposals_do_not_reuse_explorer_score_as_semantics() -> None:
    generator = CandidateGenerator()
    explorer_proposals = {
        "proposals": [
            {
                "proposal_id": "frontier_1",
                "source": "explore_py",
                "goal_xyyaw": [2.0, 1.0, 0.5],
                "frontier_point": [2.5, 1.0],
                "raw_features": {
                    "frontier_cell_count": 20,
                    "information_gain": 20.0,
                    "distance_m": 2.2,
                    "unknown_component_area_m2": 18.5,
                },
                "geometry": {
                    "proposal_score": 99.0,
                    "proposal_score_terms": {"information": 1.0},
                    "hard_constraints_passed": True,
                },
            }
        ],
        "frontier_clusters": [
            {
                "cluster_id": "stale_legacy_cluster",
                "subgoal_world": [9.0, 9.0],
                "information_gain": 100.0,
                "score": 100.0,
            }
        ],
    }
    candidates = generator.generate(explorer_proposals, {}, robot_xy=(0.0, 0.0))
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.candidate_id == "frontier:frontier_1"
    assert candidate.goal_xyyaw == [2.0, 1.0, 0.5]
    assert candidate.features["semantic_gain"] == 0.0
    assert candidate.features["priority"] == 0.0
    assert candidate.metadata["explorer_score"] == 99.0
    assert candidate.metadata["unknown_component_area_m2"] == 18.5


def test_frontier_top_k_prefers_larger_unknown_component_over_nearer_cluster() -> None:
    generator = CandidateGenerator(CandidateGeneratorConfig(max_frontier_candidates=1))
    candidates = generator.generate(
        {
            "proposals": [
                {
                    "proposal_id": "near_small",
                    "goal_xyyaw": [1.0, 0.0, 0.0],
                    "raw_features": {
                        "information_gain": 20.0,
                        "distance_m": 1.0,
                        "unknown_component_area_m2": 2.0,
                    },
                },
                {
                    "proposal_id": "far_large",
                    "goal_xyyaw": [5.0, 0.0, 0.0],
                    "raw_features": {
                        "information_gain": 10.0,
                        "distance_m": 5.0,
                        "unknown_component_area_m2": 20.0,
                    },
                },
            ]
        },
        {},
        robot_xy=(0.0, 0.0),
    )

    assert [candidate.candidate_id for candidate in candidates] == [
        "frontier:far_large"
    ]


def test_frontier_top_k_prefers_visible_aperture_over_large_hidden_component() -> None:
    generator = CandidateGenerator(CandidateGeneratorConfig(max_frontier_candidates=1))
    candidates = generator.generate(
        {
            "proposals": [
                {
                    "proposal_id": "narrow_large_component",
                    "goal_xyyaw": [1.0, 0.0, 0.0],
                    "raw_features": {
                        "information_gain": 4.0,
                        "distance_m": 1.0,
                        "unknown_component_area_m2": 60.0,
                        "expected_visible_unknown_area_m2": 2.0,
                    },
                },
                {
                    "proposal_id": "wide_opening",
                    "goal_xyyaw": [4.0, 0.0, 0.0],
                    "raw_features": {
                        "information_gain": 30.0,
                        "distance_m": 4.0,
                        "unknown_component_area_m2": 25.0,
                        "expected_visible_unknown_area_m2": 18.0,
                    },
                },
            ]
        },
        {},
        robot_xy=(0.0, 0.0),
    )

    assert [candidate.candidate_id for candidate in candidates] == [
        "frontier:wide_opening"
    ]


def test_legacy_frontier_top_k_also_prefers_larger_unknown_component() -> None:
    generator = CandidateGenerator(CandidateGeneratorConfig(max_frontier_candidates=1))
    candidates = generator.generate(
        {
            "frontier_clusters": [
                {
                    "cluster_id": "near_small",
                    "subgoal_world": [1.0, 0.0],
                    "distance_to_robot": 1.0,
                    "information_gain": 20.0,
                    "unknown_component_area_m2": 2.0,
                    "score": 10.0,
                },
                {
                    "cluster_id": "far_large",
                    "subgoal_world": [5.0, 0.0],
                    "distance_to_robot": 5.0,
                    "information_gain": 10.0,
                    "unknown_component_area_m2": 20.0,
                    "score": 1.0,
                },
            ]
        },
        {},
        robot_xy=(0.0, 0.0),
    )

    assert [candidate.candidate_id for candidate in candidates] == [
        "frontier:far_large"
    ]


def test_frontier_records_nearby_container_semantics() -> None:
    generator = CandidateGenerator()
    candidates = generator.generate(
        {
            "proposals": [
                {
                    "proposal_id": "near_fridge",
                    "goal_xyyaw": [2.0, 2.0, 0.0],
                    "raw_features": {
                        "information_gain": 8.0,
                        "distance_m": 3.0,
                        "unknown_component_area_m2": 12.0,
                    },
                }
            ]
        },
        {
            "nodes": [
                {
                    "id": "container_fridge",
                    "type": "container",
                    "label": "refrigerator",
                    "aabb_center": [2.5, 2.0, 1.0],
                    "is_currently_visible": True,
                }
            ]
        },
        robot_xy=(0.0, 0.0),
    )

    assert candidates[0].metadata["nearby_semantic_nodes"] == [
        {
            "id": "container_fridge",
            "type": "container",
            "label": "refrigerator",
            "distance_m": 0.5,
            "visible": True,
        }
    ]


def test_empty_raw_proposal_stream_does_not_fall_back_to_stale_clusters() -> None:
    generator = CandidateGenerator()
    candidates = generator.generate(
        {
            "proposals": [],
            "frontier_clusters": [
                {
                    "cluster_id": "stale_cluster",
                    "subgoal_world": [1.0, 1.0],
                    "information_gain": 10.0,
                    "score": 1.0,
                }
            ],
        },
        {},
        robot_xy=(0.0, 0.0),
    )
    assert candidates == []


def test_initial_scan_suppresses_all_behavior_candidates() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(interaction_types=("container",))
    )
    graph = {
        "nodes": [
            {
                "id": "container_1",
                "type": "container",
                "centroid": [1.0, 0.0, 0.5],
                "state_age_sec": 0.0,
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            }
        ]
    }

    assert generator.generate(
        {"initial_scan_complete": False}, graph, robot_xy=(0.0, 0.0)
    ) == []


def test_container_candidates_can_be_enabled_without_changing_portal_logic() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(interaction_types=("portal", "container"))
    )
    graph = {
        "nodes": [
            {
                "id": "container_1",
                "type": "container",
                "centroid": [1.0, 0.0, 0.5],
                "state_age_sec": 0.0,
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            }
        ]
    }
    candidates = generator.generate({}, graph, robot_xy=(0.0, 0.0))
    assert len(candidates) == 1
    assert candidates[0].metadata["node_type"] == "container"


def test_native_interaction_standoffs_keep_drawers_with_containers() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("portal", "container"),
            portal_standoff_m=0.85,
            container_standoff_m=0.50,
            drawer_standoff_m=0.50,
            interaction_safety_margin_m=0.25,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "portal_1",
                "type": "portal",
                "centroid": [3.0, 0.0, 1.0],
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            },
            {
                "id": "fridge_1",
                "type": "container",
                "name": "refrigerator",
                "centroid": [0.0, 3.0, 1.0],
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            },
            {
                "id": "drawer_1",
                "type": "container",
                "name": "chest_of_drawers",
                "centroid": [0.0, -3.0, 0.5],
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            },
        ]
    }

    candidates = {
        candidate.target_id: candidate
        for candidate in generator.generate({}, graph, robot_xy=(0.0, 0.0))
    }

    assert math.isclose(
        candidates["portal_1"].metadata["interaction_standoff_m"], 1.10
    )
    assert candidates["portal_1"].metadata["interaction_standoff_source"] == "portal"
    assert math.isclose(
        candidates["fridge_1"].metadata["interaction_standoff_m"], 0.75
    )
    assert candidates["fridge_1"].metadata["interaction_standoff_source"] == "container"
    assert math.isclose(
        candidates["drawer_1"].metadata["interaction_standoff_m"], 0.75
    )
    assert candidates["drawer_1"].metadata["interaction_standoff_source"] == "drawer"


def test_multi_drawer_metadata_emits_id_only_open_candidate() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("container",),
            require_current_visibility=True,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "container_drawers",
                "type": "container",
                "name": "dresser",
                "centroid": [1.0, 0.0, 0.5],
                "state_age_sec": 0.0,
                "is_currently_visible": True,
                "attributes": {
                    "source_object_name": "dresser_1",
                    "joint_infos": [
                        {
                            "joint_name": "drawer_top",
                            "joint_type": "slide",
                            "joint_range": [0.0, 0.4],
                            "joint_value": 0.0,
                        },
                        {
                            "joint_name": "drawer_bottom",
                            "joint_type": "slide",
                            "joint_range": [0.0, 0.4],
                            "joint_value": 0.0,
                        },
                    ],
                    "interaction_groups": [
                        {
                            "group_id": "drawer_1",
                            "target_joint_names": ["drawer_top"],
                            "close_other_joint_names": ["drawer_bottom"],
                            "close_other_joints": True,
                            "mode": "open_drawer",
                            "view_profile": "drawer_low_view",
                        },
                        {
                            "group_id": "drawer_2",
                            "target_joint_names": ["drawer_bottom"],
                            "close_other_joint_names": ["drawer_top"],
                            "close_other_joints": True,
                            "mode": "open_drawer",
                            "view_profile": "drawer_low_view",
                        },
                    ],
                },
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            }
        ]
    }

    candidates = generator.generate({}, graph, robot_xy=(0.0, 0.0))

    assert [candidate.candidate_id for candidate in candidates] == [
        "interaction:container_drawers:open"
    ]
    command = candidates[0].interaction_command
    assert command["object_id"] == "dresser_1"
    assert command["action"] == "open"
    assert "joint_names" not in command
    assert "interaction_groups" not in command
    assert "sequence_type" not in command
    assert math.isclose(candidates[0].goal_xyyaw[0], 0.0, abs_tol=1e-6)
    assert candidates[0].metadata["interaction_standoff_m"] == 1.0


def test_legacy_completed_drawer_groups_do_not_shape_planner_command() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(interaction_types=("container",))
    )
    node = {
        "id": "container_drawers",
        "type": "container",
        "name": "dresser",
        "centroid": [1.0, 0.0, 0.5],
        "state_age_sec": 0.0,
        "attributes": {
            "joint_infos": [
                {"joint_name": "top", "open_fraction": 0.0},
                {"joint_name": "bottom", "open_fraction": 0.0},
            ],
            "interaction_groups": [
                {
                    "group_id": "drawer_1",
                    "target_joint_names": ["top"],
                    "view_profile": "drawer_low_view",
                },
                {
                    "group_id": "drawer_2",
                    "target_joint_names": ["bottom"],
                    "view_profile": "drawer_low_view",
                },
            ],
        },
        "interaction": {
            "is_interactable": True,
            "requires_interaction": True,
            "state": "closed",
            "state_confidence": 1.0,
            "completed_interaction_groups": ["drawer_1"],
        },
    }

    candidates = generator.generate({}, {"nodes": [node]}, robot_xy=(0.0, 0.0))
    assert [candidate.candidate_id for candidate in candidates] == [
        "interaction:container_drawers:open"
    ]
    assert candidates[0].interaction_command["object_id"] == "dresser"
    assert "interaction_groups" not in candidates[0].interaction_command

    target_candidates = generator.generate(
        {},
        {"nodes": [node]},
        robot_xy=(0.0, 0.0),
        target_context={
            "enabled": True,
            "require_interaction": True,
            "object_labels": ["dresser"],
        },
    )
    interaction_ids = [
        candidate.candidate_id
        for candidate in target_candidates
        if candidate.behavior_type == "INTERACT"
    ]
    assert interaction_ids == ["interaction:container_drawers:open"]


def test_legacy_failed_drawer_groups_do_not_split_candidates() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(interaction_types=("container",))
    )
    node = {
        "id": "container_drawers",
        "type": "container",
        "centroid": [1.0, 0.0, 0.5],
        "state_age_sec": 0.0,
        "attributes": {
            "interaction_groups": [
                {"group_id": "drawer_1", "target_joint_names": ["top"]},
                {"group_id": "drawer_2", "target_joint_names": ["bottom"]},
            ],
        },
        "interaction": {
            "is_interactable": True,
            "requires_interaction": True,
            "state": "ajar",
            "state_confidence": 1.0,
            "failed_interaction_groups": ["drawer_1"],
        },
    }

    candidates = generator.generate({}, {"nodes": [node]}, robot_xy=(0.0, 0.0))
    assert [candidate.candidate_id for candidate in candidates] == [
        "interaction:container_drawers:open"
    ]


def test_portal_approach_ignores_unstable_body_orientation() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(interaction_types=("portal",), portal_standoff_m=1.0)
    )
    graph = {
        "nodes": [
            {
                "id": "portal_1",
                "type": "portal",
                "centroid": [0.0, 0.0, 1.0],
                "aabb_size": [0.1, 1.0, 2.0],
                "state_age_sec": 0.0,
                "attributes": {
                    "interaction_reference_aabb_center": [0.0, 0.0, 1.0],
                    "interaction_reference_aabb_size": [0.1, 1.0, 2.0],
                    "interaction_reference_orientation": [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
                },
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            }
        ]
    }

    candidate = generator.generate({}, graph, robot_xy=(0.0, -2.0))[0]
    assert math.isclose(candidate.goal_xyyaw[0], 1.05, abs_tol=1e-6)
    assert math.isclose(candidate.goal_xyyaw[1], 0.0, abs_tol=1e-6)
    assert math.isclose(candidate.goal_xyyaw[2], math.pi, abs_tol=1e-6)


def test_portal_approach_uses_door_aabb_normal() -> None:
    generator = CandidateGenerator(CandidateGeneratorConfig(portal_standoff_m=1.15))
    graph = {
        "nodes": [
            {
                "id": "portal_double",
                "type": "portal",
                "centroid": [5.0, 5.0, 1.0],
                "aabb_center": [5.4, 5.0, 1.0],
                "aabb_size": [1.0, 2.0, 2.1],
                "attributes": {
                    "interaction_reference_aabb_center": [5.0, 5.0, 1.0],
                    "interaction_reference_aabb_size": [0.2, 2.0, 2.1],
                },
                "state_age_sec": 0.0,
                "is_currently_visible": True,
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            }
        ]
    }
    candidate = generator.generate({}, graph, robot_xy=(2.0, 3.0))[0]
    assert math.isclose(candidate.goal_xyyaw[0], 3.75, abs_tol=1e-6)
    assert math.isclose(candidate.goal_xyyaw[1], 5.0, abs_tol=1e-6)
    assert math.isclose(candidate.goal_xyyaw[2], 0.0, abs_tol=1e-6)
    assert candidate.metadata["approach_strategy"] == "portal_aabb_normal"
    assert candidate.metadata["portal_aabb_center_xy"] == [5.0, 5.0]
    assert candidate.metadata["portal_clearance_aabb_center_xy"] == [5.4, 5.0]
    assert candidate.metadata["portal_clearance_aabb_size_xy"] == [1.0, 2.0]
    goals = candidate.metadata["goal_xyyaw_candidates"]
    assert len(goals) == 18
    assert math.isclose(
        goals[1][0], 3.50, abs_tol=1e-6
    )
    # Tangential fallbacks and the opposite doorway side are both present.
    assert any(
        math.isclose(goal[0], 3.75, abs_tol=1e-6)
        and math.isclose(abs(goal[1] - 5.0), 0.20, abs_tol=1e-6)
        for goal in goals
    )
    assert any(math.isclose(goal[0], 6.25, abs_tol=1e-6) for goal in goals)
    # All approach poses, including mirrored/tangential ones, face the portal.
    for goal_x, goal_y, goal_yaw in goals:
        expected_yaw = math.atan2(5.0 - goal_y, 5.0 - goal_x)
        assert math.isclose(
            math.atan2(math.sin(goal_yaw - expected_yaw), math.cos(goal_yaw - expected_yaw)),
            0.0,
            abs_tol=1e-6,
        )


def test_observed_target_generates_navigation_candidate() -> None:
    generator = CandidateGenerator()
    graph = {
        "nodes": [
            {
                "id": "container_fridge",
                "type": "container",
                "label": "fridge",
                "name": "refrigerator_asset",
                "centroid": [4.0, 2.0, 1.0],
                "confidence": 1.0,
                "state_age_sec": 2.0,
                "is_currently_visible": True,
                "attributes": {"visible_pixels": 32},
            }
        ]
    }
    candidates = generator.generate(
        {},
        graph,
        robot_xy=(1.0, 2.0),
        target_context={"enabled": True, "object_labels": ["fridge"]},
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.behavior_type == "NAVIGATE"
    assert candidate.candidate_id == "target:container_fridge"
    assert candidate.features["target_relevance"] == 1.0
    assert candidate.metadata["target_goal"] is True
    assert candidate.metadata["target_visible_now"] is True
    assert candidate.metadata["verify_target_visibility"] is True
    assert candidate.metadata["target_min_visible_pixels"] == 16
    assert candidate.metadata["target_navigation_required"] is True
    assert candidate.metadata["target_goal_distance_m"] > 0.35


def test_contained_target_navigation_anchors_outside_parent_container() -> None:
    generator = CandidateGenerator()
    graph = {
        "nodes": [
            {
                "id": "container_fridge",
                "type": "container",
                "label": "fridge",
                "aabb_center": [4.0, 2.0, 1.0],
                "aabb_size": [1.0, 1.0, 2.0],
            },
            {
                "id": "object_apple",
                "type": "object",
                "label": "apple",
                "parent_id": "container_fridge",
                "centroid": [4.0, 2.0, 1.0],
                "confidence": 1.0,
                "state_age_sec": 0.0,
                "is_currently_visible": True,
                "attributes": {"visible_pixels": 64},
            },
        ]
    }
    candidate = generator.generate(
        {},
        graph,
        robot_xy=(1.0, 2.0),
        target_context={"enabled": True, "object_labels": ["apple"]},
    )[0]

    assert math.isclose(candidate.goal_xyyaw[0], 2.5, abs_tol=1e-6)
    assert math.isclose(candidate.goal_xyyaw[1], 2.0, abs_tol=1e-6)
    assert candidate.metadata["navigation_anchor_id"] == "container_fridge"
    assert candidate.metadata["approach_strategy"] == "target_parent_container_standoff"


def test_target_near_container_uses_geometric_anchor_without_contains_edge() -> None:
    generator = CandidateGenerator()
    graph = {
        "nodes": [
            {
                "id": "container_dresser",
                "type": "container",
                "label": "dresser",
                "room_id": 1,
                "aabb_center": [4.0, 2.0, 0.5],
                "aabb_size": [1.0, 1.0, 1.0],
            },
            {
                "id": "object_remote",
                "type": "object",
                "label": "remotecontrol",
                "room_id": 1,
                "centroid": [4.1, 2.0, 1.05],
                "state_age_sec": 0.0,
                "is_currently_visible": True,
                "attributes": {"visible_pixels": 64},
            },
        ]
    }
    candidate = generator.generate(
        {},
        graph,
        robot_xy=(1.0, 2.0),
        target_context={"enabled": True, "object_labels": ["remotecontrol"]},
    )[0]

    assert candidate.metadata["navigation_anchor_id"] == "container_dresser"
    assert candidate.metadata["approach_strategy"] == "target_parent_container_standoff"


def test_target_reuses_successful_parent_container_approach_pose() -> None:
    generator = CandidateGenerator()
    graph = {
        "nodes": [
            {
                "id": "container_fridge",
                "type": "container",
                "label": "fridge",
                "aabb_center": [4.0, 2.0, 1.0],
                "aabb_size": [2.0, 2.0, 2.0],
                "interaction": {
                    "operation_history": [
                        {
                            "success": True,
                            "approach_goal_xyyaw": [3.25, 2.5, 0.75],
                        }
                    ]
                },
            },
            {
                "id": "object_apple",
                "type": "object",
                "label": "apple",
                "parent_id": "container_fridge",
                "centroid": [4.0, 2.0, 1.0],
                "state_age_sec": 0.0,
                "is_currently_visible": True,
                "attributes": {"visible_pixels": 64},
            },
        ]
    }
    candidate = generator.generate(
        {},
        graph,
        robot_xy=(1.0, 2.0),
        target_context={"enabled": True, "object_labels": ["apple"]},
    )[0]

    assert candidate.goal_xyyaw == [3.25, 2.5, 0.75]
    assert candidate.metadata["approach_strategy"] == (
        "target_last_successful_interaction_pose"
    )


def test_contained_target_reuses_container_interaction_pose() -> None:
    generator = CandidateGenerator()
    graph = {
        "nodes": [
            {
                "id": "container_fridge",
                "type": "container",
                "centroid": [4.0, 2.0, 1.0],
                "attributes": {
                    "interaction_approach_pose_xyyaw": [3.0, 2.0, 0.0],
                    "interaction_approach_axis_xy": [-1.0, 0.0],
                },
            },
            {
                "id": "object_apple",
                "type": "object",
                "label": "apple",
                "centroid": [4.1, 2.0, 1.0],
                "state_age_sec": 0.0,
                "is_currently_visible": True,
                "attributes": {
                    "visible_pixels": 64,
                    "visible_fraction": 0.5,
                    "consecutive_observations": 2,
                },
            },
        ],
        "edges": [
            {
                "src_id": "container_fridge",
                "relation": "contains",
                "dst_id": "object_apple",
            }
        ],
    }

    candidate = generator.generate(
        {},
        graph,
        robot_xy=(2.8, 2.0),
        target_context={"enabled": True, "object_labels": ["apple"]},
    )[0]

    assert candidate.goal_xyyaw == [3.0, 2.0, 0.0]
    assert candidate.metadata["approach_strategy"] == "target_containing_container_pose"
    assert candidate.metadata["containing_container_id"] == "container_fridge"
    assert candidate.metadata["direct_goal_tolerance_m"] == 0.45


def test_explicit_target_container_is_relevant_without_target_object_interaction() -> None:
    generator = CandidateGenerator()
    graph = {
        "nodes": [
            {
                "id": "container_fridge",
                "type": "container",
                "name": "refrigerator_asset",
                "centroid": [4.0, 2.0, 1.0],
                "is_currently_visible": True,
                "state_age_sec": 0.0,
                "attributes": {
                    "instance_id": "refrigerator_asset",
                    "source_object_name": "refrigerator_asset",
                },
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            }
        ]
    }

    candidate = generator.generate(
        {},
        graph,
        robot_xy=(2.0, 2.0),
        target_context={
            "enabled": True,
            "object_labels": ["apple"],
            "require_interaction": False,
            "target_container_source_object_name": "refrigerator_asset",
        },
    )[0]

    assert candidate.metadata["target_match"] is True
    assert candidate.features["target_relevance"] == 1.0


def test_weak_target_observation_is_not_verified_as_visible() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            target_min_visible_pixels=128,
            target_min_visible_fraction=0.2,
            target_min_consecutive_observations=2,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "object_pencil",
                "type": "object",
                "label": "pencil",
                "centroid": [2.0, 0.0, 0.5],
                "is_currently_visible": True,
                "attributes": {
                    "visible_pixels": 24,
                    "visible_fraction": 0.08,
                    "consecutive_observations": 1,
                },
            }
        ]
    }
    candidates = generator.generate(
        {}, graph, robot_xy=(0.0, 0.0), target_context={"enabled": True, "object_labels": ["pencil"]}
    )
    assert candidates == []


def test_reliably_observed_target_remains_candidate_after_leaving_view() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            target_min_visible_pixels=128,
            target_min_visible_fraction=0.2,
            target_min_consecutive_observations=2,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "object_pencil",
                "type": "object",
                "label": "pencil",
                "centroid": [2.0, 0.0, 0.5],
                "is_currently_visible": False,
                "attributes": {
                    "visible_pixels": 0,
                    "visible_fraction": 0.0,
                    "consecutive_observations": 0,
                    "max_visible_pixels": 256,
                    "max_visible_fraction": 0.4,
                    "max_consecutive_observations": 3,
                },
            }
        ]
    }
    candidates = generator.generate(
        {}, graph, robot_xy=(0.0, 0.0), target_context={"enabled": True, "object_labels": ["pencil"]}
    )
    assert [candidate.candidate_id for candidate in candidates] == ["target:object_pencil"]
    assert candidates[0].metadata["target_visible_now"] is False


def test_aabb_approach_standoff_is_outside_container_box() -> None:
    generator = CandidateGenerator()
    graph = {
        "nodes": [
            {
                "id": "container_fridge",
                "type": "container",
                "label": "fridge",
                "aabb_center": [4.0, 2.0, 1.0],
                "aabb_size": [1.0, 1.0, 2.0],
                "state_age_sec": 0.0,
                "is_currently_visible": True,
                "attributes": {"visible_pixels": 32},
            }
        ]
    }
    candidate = generator.generate(
        {},
        graph,
        robot_xy=(1.0, 2.0),
        target_context={"enabled": True, "object_labels": ["fridge"]},
    )[0]
    assert math.isclose(candidate.goal_xyyaw[0], 2.5, abs_tol=1e-6)
    assert math.isclose(candidate.goal_xyyaw[1], 2.0, abs_tol=1e-6)


def test_container_front_axis_overrides_nearest_radial_side() -> None:
    node = {
        "id": "container_fridge",
        "type": "container",
        "label": "fridge",
        "aabb_center": [4.0, 2.0, 1.0],
        "aabb_size": [1.0, 2.0, 2.0],
        "state_age_sec": 0.0,
        "is_currently_visible": True,
        "attributes": {
            "visible_pixels": 32,
            "interaction_approach_axis_xy": [0.0, -1.0],
        },
        "interaction": {
            "is_interactable": True,
            "requires_interaction": True,
            "state": "closed",
            "state_confidence": 1.0,
        },
    }
    target = CandidateGenerator().generate(
        {},
        {"nodes": [node]},
        robot_xy=(4.0, 4.0),
        target_context={"enabled": True, "object_labels": ["fridge"]},
    )[0]
    interaction = CandidateGenerator(
        CandidateGeneratorConfig(interaction_types=("container",))
    ).generate({}, {"nodes": [node]}, robot_xy=(4.0, 4.0))[0]

    for candidate in (target, interaction):
        expected_y = 0.0
        assert math.isclose(candidate.goal_xyyaw[0], 4.0, abs_tol=1e-6)
        assert math.isclose(candidate.goal_xyyaw[1], expected_y, abs_tol=1e-6)
        assert math.isclose(candidate.goal_xyyaw[2], math.pi / 2.0, abs_tol=1e-6)
        assert candidate.metadata["interaction_approach_axis_xy"] == (0.0, -1.0)


def test_container_explicit_interaction_pose_overrides_live_aabb() -> None:
    node = {
        "id": "container_fridge",
        "type": "container",
        "label": "fridge",
        "aabb_center": [9.0, 9.0, 1.0],
        "aabb_size": [2.0, 2.0, 2.0],
        "state_age_sec": 0.0,
        "is_currently_visible": True,
        "attributes": {
            "source_object_name": "fridge_house7",
            "interaction_approach_axis_xy": [1.0, 0.0],
            "interaction_approach_pose_xyyaw": [8.25, 1.05, math.pi],
        },
        "interaction": {
            "is_interactable": True,
            "requires_interaction": True,
            "state": "closed",
            "state_confidence": 1.0,
        },
    }

    candidate = CandidateGenerator(
        CandidateGeneratorConfig(interaction_types=("container",))
    ).generate({}, {"nodes": [node]}, robot_xy=(4.0, 4.0))[0]

    assert candidate.goal_xyyaw == [8.25, 1.05, math.pi]
    assert candidate.metadata["approach_strategy"] == "container_explicit_pose"
    assert candidate.metadata["goal_xyyaw_candidates"][1] == [8.5, 1.05, math.pi]
    assert candidate.interaction_command["interaction_approach_pose_xyyaw"] == [
        8.25,
        1.05,
        math.pi,
    ]


def test_target_current_visibility_can_be_required() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(target_require_current_visibility=True)
    )
    graph = {
        "nodes": [
            {
                "id": "container_fridge",
                "type": "container",
                "label": "fridge",
                "centroid": [4.0, 2.0, 1.0],
                "state_age_sec": 2.0,
                "is_currently_visible": False,
                "attributes": {"visible_pixels": 32},
            }
        ]
    }
    assert generator.generate(
        {},
        graph,
        robot_xy=(1.0, 2.0),
        target_context={"enabled": True, "object_labels": ["fridge"]},
    ) == []


def test_observed_target_candidate_persists_after_leaving_view() -> None:
    generator = CandidateGenerator()
    graph = {
        "nodes": [
            {
                "id": "container_fridge",
                "type": "container",
                "label": "fridge",
                "centroid": [4.0, 2.0, 1.0],
                "confidence": 1.0,
                "state_age_sec": 2.0,
                "is_currently_visible": False,
                "attributes": {"visible_pixels": 256},
            }
        ]
    }
    candidates = generator.generate(
        {},
        graph,
        robot_xy=(1.0, 2.0),
        target_context={"enabled": True, "object_labels": ["fridge"]},
    )
    assert [candidate.candidate_id for candidate in candidates] == ["target:container_fridge"]
    assert candidates[0].metadata["target_visible_now"] is False


def test_generic_storage_box_is_not_an_interaction_candidate() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("container",),
        )
    )
    graph = {
        "nodes": [
            {
                "id": "container_1",
                "type": "container",
                "name": "box",
                "centroid": [3.5, 0.0, 0.5],
                "state_age_sec": 0.0,
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            }
        ]
    }
    candidates = generator.generate({}, graph, robot_xy=(0.0, 0.0))
    assert candidates == []


def test_container_interaction_can_require_same_room() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("container",),
            container_require_same_room=True,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "room_1",
                "type": "room",
                "room_id": 1,
                "aabb_center": [1.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "room_2",
                "type": "room",
                "room_id": 2,
                "aabb_center": [5.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "container_1",
                "type": "container",
                "room_id": 2,
                "centroid": [5.0, 0.0, 0.5],
                "state_age_sec": 0.0,
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            },
        ]
    }

    assert generator.generate({}, graph, robot_xy=(1.0, 0.0)) == []
    candidates = generator.generate({}, graph, robot_xy=(5.0, 0.0))
    assert len(candidates) == 1
    assert candidates[0].metadata["robot_room_id"] == 2
    assert candidates[0].metadata["target_room_id"] == 2


def test_target_can_require_robot_to_be_in_target_room() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(target_require_same_room=True)
    )
    graph = {
        "nodes": [
            {
                "id": "room_1",
                "type": "room",
                "room_id": 1,
                "aabb_center": [1.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "room_2",
                "type": "room",
                "room_id": 2,
                "aabb_center": [5.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "container_fridge",
                "type": "container",
                "room_id": 2,
                "label": "fridge",
                "aabb_center": [5.0, 0.0, 1.0],
                "aabb_size": [1.0, 1.0, 2.0],
                "is_currently_visible": True,
                "attributes": {"visible_pixels": 256},
                "interaction": {"state": "closed", "is_interactable": True},
            },
        ]
    }
    assert generator.generate(
        {},
        graph,
        robot_xy=(1.0, 0.0),
        target_context={"enabled": True, "object_labels": ["fridge"]},
    ) == []
    candidates = generator.generate(
        {},
        graph,
        robot_xy=(5.0, 0.0),
        target_context={"enabled": True, "object_labels": ["fridge"]},
    )
    assert len(candidates) == 1
    assert candidates[0].metadata["target_room_id"] == 2


def test_container_same_room_filter_allows_traversable_room_transition() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("container",),
            container_require_same_room=True,
            container_allow_connected_room=True,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "room_1",
                "type": "room",
                "room_id": 1,
                "aabb_center": [1.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "room_2",
                "type": "room",
                "room_id": 2,
                "aabb_center": [5.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "container_1",
                "type": "container",
                "room_id": 2,
                "centroid": [5.0, 0.0, 0.5],
                "state_age_sec": 0.0,
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            },
        ],
        "edges": [
            {
                "relation": "connects",
                "src_id": "portal_1",
                "dst_id": "room_1",
                "attributes": {
                    "candidate_connected_room_ids": [1, 2],
                    "traversable": True,
                },
            }
        ],
    }

    candidates = generator.generate({}, graph, robot_xy=(1.0, 0.0))

    assert len(candidates) == 1
    assert candidates[0].metadata["room_transition_required"] is True
    assert candidates[0].metadata["room_hops"] == 1
    assert candidates[0].metadata["room_reachable"] is True


def test_container_same_room_filter_rejects_closed_room_transition() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            interaction_types=("container",),
            container_require_same_room=True,
            container_allow_connected_room=True,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "room_1",
                "type": "room",
                "room_id": 1,
                "aabb_center": [1.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "room_2",
                "type": "room",
                "room_id": 2,
                "aabb_center": [5.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "container_1",
                "type": "container",
                "room_id": 2,
                "centroid": [5.0, 0.0, 0.5],
                "state_age_sec": 0.0,
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            },
        ],
        "edges": [
            {
                "relation": "connects",
                "src_id": "portal_1",
                "dst_id": "room_1",
                "attributes": {
                    "candidate_connected_room_ids": [1, 2],
                    "traversable": False,
                },
            }
        ],
    }

    assert generator.generate({}, graph, robot_xy=(1.0, 0.0)) == []


def test_invisible_unreachable_container_is_not_an_interaction_candidate() -> None:
    """A stale container in a disconnected room cannot create INTERACT."""

    generator = CandidateGenerator(
        CandidateGeneratorConfig(interaction_types=("container",))
    )
    graph = {
        "nodes": [
            {
                "id": "room_1",
                "type": "room",
                "room_id": 1,
                "aabb_center": [1.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "room_2",
                "type": "room",
                "room_id": 2,
                "aabb_center": [5.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "container_dresser",
                "type": "container",
                "room_id": 2,
                "label": "dresser",
                "centroid": [5.0, 0.0, 0.5],
                "state_age_sec": 0.0,
                "is_currently_visible": False,
                "interaction": {
                    "is_interactable": True,
                    "requires_interaction": True,
                    "state": "closed",
                    "state_confidence": 1.0,
                },
            },
        ],
        "edges": [
            {
                "relation": "connects",
                "src_id": "portal_1",
                "dst_id": "room_1",
                "attributes": {
                    "candidate_connected_room_ids": [1, 2],
                    "traversable": False,
                },
            }
        ],
    }

    assert generator.generate({}, graph, robot_xy=(1.0, 0.0)) == []


def test_target_same_room_filter_allows_traversable_room_transition() -> None:
    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            target_require_same_room=True,
            target_allow_connected_room=True,
        )
    )
    graph = {
        "nodes": [
            {
                "id": "room_1",
                "type": "room",
                "room_id": 1,
                "aabb_center": [1.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "room_2",
                "type": "room",
                "room_id": 2,
                "aabb_center": [5.0, 0.0, 0.1],
                "aabb_size": [2.0, 2.0, 0.2],
            },
            {
                "id": "container_fridge",
                "type": "container",
                "room_id": 2,
                "label": "fridge",
                "centroid": [5.0, 0.0, 1.0],
                "is_currently_visible": True,
                "attributes": {"visible_pixels": 256},
            },
        ],
        "edges": [
            {
                "relation": "connects",
                "src_id": "portal_1",
                "dst_id": "room_1",
                "attributes": {
                    "candidate_connected_room_ids": [1, 2],
                    "traversable": True,
                },
            }
        ],
    }

    candidates = generator.generate(
        {},
        graph,
        robot_xy=(1.0, 0.0),
        target_context={"enabled": True, "object_labels": ["fridge"]},
    )

    assert len(candidates) == 1
    assert candidates[0].metadata["room_hops"] == 1
