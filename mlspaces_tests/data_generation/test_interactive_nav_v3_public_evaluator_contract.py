"""Small contract tests for the public V3 ROS evaluator boundary.

These tests remain simulator- and ROS-free.  They assert that the evaluator
derives a dynamic goal only from language, adapts the canonical restricted-GT
frame without restoring private information, and reports task completion
separately from interaction-conditioned completion.
"""

from __future__ import annotations

import json

import numpy as np

from scripts.InteractiveNav.evaluation.benchmark_metrics import summarise_results
from scripts.InteractiveNav.evaluation.benchmark_runner import (
    _redact_runtime_consistency_for_restricted_policy,
)
from scripts.InteractiveNav.evaluation.public_goal import build_public_target_context
from scripts.InteractiveNav.evaluation.restricted_gt_perception import (
    RESTRICTED_GT_PROTOCOL_VERSION,
    encode_binary_mask_rle,
)
from scripts.InteractiveNav.evaluation.ros_object_goal_adapter import (
    adapt_restricted_gt_frame_for_legacy_mapping,
)


def test_public_goal_uses_language_aliases_without_selected_instance_leakage() -> None:
    language = {
        "referral_expressions": {"object_name": "the fridge"},
        # Deliberately present but not part of the public-language schema.  The
        # target builder must not inspect or serialize it.
        "interactive_nav": {"target": {"selected_instance": "private_fridge_body_928"}},
    }

    context = build_public_target_context(language, instruction="Find the selected refrigerator.")

    assert context["target_name"] == "the fridge"
    assert context["object_labels"] == ["the fridge", "fridge", "refrigerator", "the"]
    assert "require_interaction" not in context
    assert "completion_requires_visibility" not in context
    serialized = json.dumps(context, sort_keys=True)
    assert "private_fridge_body_928" not in serialized
    assert "selected_instance" not in serialized


def test_canonical_restricted_gt_adapter_derives_only_legacy_geometry_aliases() -> None:
    mask = np.asarray([[False, True, False], [True, True, False]], dtype=bool)
    canonical_payload = {
        "protocol_version": RESTRICTED_GT_PROTOCOL_VERSION,
        "episode_id": "episode_000001",
        "episode_reset": True,
        "frame_index": 7,
        "observations": [
            {
                "instance_id": "obj_000001",
                "name": "refrigerator",
                "bbox_2d_xyxy": [1, 0, 2, 1],
                "mask_rle": encode_binary_mask_rle(mask),
                "bbox_3d": {
                    "center": [1.0, 2.0, 3.0],
                    "size": [0.8, 0.6, 1.7],
                    "frame_id": "world",
                },
            }
        ],
    }

    adapted = adapt_restricted_gt_frame_for_legacy_mapping(
        canonical_payload,
        stamp_sec=12.5,
        consecutive_observations={"stale_obj": 3},
    )

    assert adapted["schema_version"] == "interactive_nav_v3_restricted_gt_legacy_adapter_v1"
    assert adapted["episode_id"] == "episode_000001"
    assert adapted["capture_step"] == 7
    observation = adapted["observations"][0]
    assert observation["instance_id"] == "obj_000001"
    assert observation["source_object_name"] == "obj_000001"
    assert observation["semantic_name"] == "refrigerator"
    assert observation["bbox_2d"] == [1, 0, 2, 1]
    assert observation["position"] == [1.0, 2.0, 3.0]
    assert observation["aabb_center"] == [1.0, 2.0, 3.0]
    assert observation["aabb_size"] == [0.8, 0.6, 1.7]
    assert observation["visible_pixels"] == 3
    assert observation["consecutive_observations"] == 1
    serialized = json.dumps(adapted, sort_keys=True)
    assert "private_fridge_body_928" not in serialized
    assert "joint" not in serialized
    assert "open_fraction" not in serialized


def test_summary_separates_task_success_from_interaction_conditioned_success() -> None:
    common = {
        "domains": ["container"],
        "interaction_requirement": "required",
        "recipe": "container_hidden",
        "interaction_types": ["container_hinged_door"],
        "path_length_bin": "[3,5)",
        "nav_success": True,
        "required_interaction_success": True,
        "sequence_success": True,
        "interaction_action_count": 1,
        "correct_interaction_action_count": 1,
        "step_count": 3,
        "navigation_path_length_m": 4.0,
        "reference_path_length_m": 3.0,
        "spl": 0.75,
        "total_simulated_seconds": 1.0,
        "extra_interaction_action_count": 0,
        "invalid_interaction_action_count": 0,
        "terminal_reason": "target_found",
    }
    rows = [
        {
            **common,
            "success": False,
            "task_success": True,
            "interaction_conditioned_success": False,
        },
        {
            **common,
            "success": True,
            "task_success": False,
            "interaction_conditioned_success": True,
        },
    ]

    group = summarise_results(rows)["groups"]["overall"]

    assert group["success_rate"] == 0.5
    assert group["task_success_rate"] == 0.5
    assert group["interaction_conditioned_success_rate"] == 0.5


def test_restricted_result_redacts_private_runtime_consistency_details() -> None:
    raw = {
        "eligible": False,
        "exclusion_reasons": ["articulation_state_mismatch"],
        "checks": {
            "oracle_terminal_goal": {
                "checked": True,
                "consistent": False,
                "target_name": "private_target_body_93",
                "target_position_xy": [4.0, 5.0],
            },
            "articulation_state_readback": {
                "passed": False,
                "joint_count": 1,
                "failed_joints": [{"joint_name": "private_door_hinge"}],
            },
            "interaction_resolution": {
                "passed": False,
                "interaction_count": 1,
                "failed_interactions": [{"object_name": "private_door_body"}],
            },
        },
    }

    goal, public = _redact_runtime_consistency_for_restricted_policy(raw)

    assert goal == {"checked": True, "consistent": False}
    assert public is not None
    assert public["checks"]["articulation_state_readback"] == {
        "passed": False,
        "joint_count": 1,
    }
    assert public["checks"]["interaction_resolution"] == {
        "passed": False,
        "interaction_count": 1,
    }
    serialized = json.dumps(public, sort_keys=True)
    assert "private_target_body_93" not in serialized
    assert "private_door_hinge" not in serialized
    assert "private_door_body" not in serialized
