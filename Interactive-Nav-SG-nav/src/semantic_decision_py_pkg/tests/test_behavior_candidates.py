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
                "aabb_size": [0.2, 2.0, 2.1],
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
    assert len(candidate.metadata["goal_xyyaw_candidates"]) == 3
    assert math.isclose(
        candidate.metadata["goal_xyyaw_candidates"][1][0], 3.50, abs_tol=1e-6
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
