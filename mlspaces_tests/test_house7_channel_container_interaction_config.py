from __future__ import annotations

import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DECISION_SCRIPTS = (
    REPO_ROOT
    / "Interactive-Nav-SG-nav"
    / "src"
    / "semantic_decision_py_pkg"
    / "scripts"
)
if str(DECISION_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DECISION_SCRIPTS))


from semantic_decision_py_pkg.behavior_candidates import (
    CandidateGenerator,
    CandidateGeneratorConfig,
)


def test_house7_channel_container_override_generates_both_interaction_classes() -> None:
    config_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "house7_channel_container_interaction_rule.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidate_config = config["candidate"]
    assert candidate_config["interaction_types"] == ["portal", "container"]
    assert candidate_config["max_frontier_candidates"] == 0

    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            max_frontier_candidates=int(candidate_config["max_frontier_candidates"]),
            interaction_types=tuple(candidate_config["interaction_types"]),
            min_state_confidence=float(candidate_config["min_state_confidence"]),
            portal_require_attribute_ready=bool(
                candidate_config["portal_require_attribute_ready"]
            ),
            portal_allow_unknown_state=bool(
                candidate_config["portal_allow_unknown_state"]
            ),
            portal_unknown_default_interact=bool(
                candidate_config["portal_unknown_default_interact"]
            ),
            portal_standoff_m=float(candidate_config["portal_standoff_m"]),
            container_standoff_m=float(candidate_config["container_standoff_m"]),
            interaction_safety_margin_m=float(
                candidate_config["interaction_safety_margin_m"]
            ),
            interaction_ready_distance_m=float(
                candidate_config["interaction_ready_distance_m"]
            ),
        )
    )
    graph = {
        "nodes": [
            {
                "id": "door_0001",
                "type": "portal",
                "name": "door_0001",
                "label": "door",
                "aabb_center": [2.0, 0.0, 1.0],
                "aabb_size": [0.2, 1.0, 2.0],
                "is_currently_visible": True,
                "interaction": {
                    "state": "unknown",
                    "state_confidence": 0.0,
                    "is_interactable": True,
                    "requires_interaction": True,
                    "interaction_mode": "open_close",
                },
                "attributes": {"instance_id": "door_0001", "category": "door"},
            },
            {
                "id": "container_fridge_0001",
                "type": "container",
                "name": "Fridge",
                "label": "fridge",
                "aabb_center": [1.0, 2.0, 1.0],
                "aabb_size": [0.8, 0.8, 1.8],
                "is_currently_visible": True,
                "interaction": {
                    "state": "unknown",
                    "state_confidence": 1.0,
                    "is_interactable": True,
                    "requires_interaction": True,
                    "interaction_mode": "open_close",
                },
                "attributes": {
                    "instance_id": "refrigerator_0001",
                    "category": "fridge",
                },
            },
        ]
    }
    candidates = generator.generate(
        {"initial_scan_complete": True}, graph, robot_xy=(0.0, 0.0)
    )
    assert {
        candidate.metadata["node_type"] for candidate in candidates
    } == {"portal", "container"}
