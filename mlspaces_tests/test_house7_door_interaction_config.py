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
    BEHAVIOR_INTERACT,
    CandidateGenerator,
    CandidateGeneratorConfig,
)
from semantic_decision_py_pkg.rule_policy import RulePolicy, RulePolicyConfig


def test_house7_door_interaction_override_selects_unknown_generic_door() -> None:
    """The reproducible validation lane must exercise the public door path."""

    config_path = (
        REPO_ROOT
        / "scripts"
        / "InteractiveNav"
        / "configs"
        / "semantic_decision"
        / "house7_door_interaction_rule.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    candidate_config = config["candidate"]
    policy_config = config["policy"]

    assert candidate_config["interaction_types"] == ["portal"]
    assert candidate_config["max_frontier_candidates"] == 0
    assert candidate_config["portal_unknown_default_interact"] is True
    assert candidate_config["portal_allow_unknown_state"] is True
    assert float(policy_config["portal_bonus"]) > 0.0
    assert float(policy_config["interaction_priority_bonus"]) > 0.0

    generator = CandidateGenerator(
        CandidateGeneratorConfig(
            max_frontier_candidates=int(candidate_config["max_frontier_candidates"]),
            interaction_types=tuple(candidate_config["interaction_types"]),
            min_state_confidence=float(candidate_config["min_state_confidence"]),
            portal_require_attribute_ready=bool(
                candidate_config["portal_require_attribute_ready"]
            ),
            portal_allow_unknown_state=bool(candidate_config["portal_allow_unknown_state"]),
            portal_unknown_default_interact=bool(
                candidate_config["portal_unknown_default_interact"]
            ),
            portal_standoff_m=float(candidate_config["portal_standoff_m"]),
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
                "id": "portal_door_0001",
                # ``portal`` is the graph topology type.  The generic public
                # label/ID intentionally says only that this is a door.
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
                "attributes": {
                    "instance_id": "door_0001",
                    "category": "door",
                },
            }
        ]
    }
    candidates = generator.generate(
        {"initial_scan_complete": True}, graph, robot_xy=(0.0, 0.0)
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.behavior_type == BEHAVIOR_INTERACT
    assert candidate.target_id == "portal_door_0001"
    assert candidate.interaction_command["object_id"] == "door_0001"
    assert candidate.interaction_command["action"] == "open"

    selected = RulePolicy(
        RulePolicyConfig(
            portal_bonus=float(policy_config["portal_bonus"]),
            interaction_priority_bonus=float(
                policy_config["interaction_priority_bonus"]
            ),
            nearby_interaction_radius_m=float(
                policy_config["nearby_interaction_radius_m"]
            ),
            nearby_interaction_bonus=float(policy_config["nearby_interaction_bonus"]),
        )
    ).select(candidates)
    assert selected is candidate
