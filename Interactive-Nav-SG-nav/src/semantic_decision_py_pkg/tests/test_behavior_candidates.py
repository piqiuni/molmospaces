from __future__ import annotations

import math

from semantic_decision_py_pkg.behavior_candidates import CandidateGenerator, CandidateGeneratorConfig


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
