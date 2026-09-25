"""Container policies and backend contracts without a ROS executor or robot."""

import ast
from copy import deepcopy
import inspect
import math

import pytest

from semantic_decision_py_pkg import behavior_execution, container_approach
from semantic_decision_py_pkg import container_evidence, interaction_commands
from semantic_decision_py_pkg.approach_geometry import normalize_angle
from semantic_decision_py_pkg.container_evidence import (
    container_m1_front_axis_from_capture_pose,
    container_m1_pre_action_ready,
    finite_public_bbox,
    public_step_or_none,
)
from semantic_decision_py_pkg.interaction_commands import (
    build_interaction_command,
    has_valid_drawer_visual_contract,
)


@pytest.mark.parametrize("module,allowed", [
    (container_approach, {"__future__", "math", "re", "typing", "approach_geometry"}),
    (container_evidence, {"__future__", "math"}),
    (interaction_commands, {
        "__future__", "copy", "math", "container_evidence", "visual_interaction_planning",
    }),
])
def test_contract_modules_have_no_transport_or_state_machine_dependency(module, allowed):
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.ImportFrom):
            assert node.module in allowed
        elif isinstance(node, ast.Import):
            assert all(alias.name in allowed for alias in node.names)


def test_existing_container_and_angle_imports_remain_identical():
    exported = [name for name in vars(container_approach) if name.startswith(
        ("container_two_stage_", "is_container_two_stage_")
    )]
    assert len(exported) == 13
    for name in exported:
        assert getattr(behavior_execution, name) is getattr(container_approach, name)
    assert behavior_execution.normalize_angle is normalize_angle
    assert behavior_execution._goal_xyyaw_option is container_approach._goal_xyyaw_option


def _staged_candidate():
    return {"metadata": {
        "container_two_stage_approach": True,
        "container_two_stage_phase": "m1_capture",
        "container_staging_goal_xyyaw_candidates": [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0], [4.0, 0.0, 0.0],
        ],
        "container_m1_viewpoint_order": [3, 1, 0, 2, 4],
        "container_staging_pose_labels": [
            "aabb_face_pos_x", "aabb_fan_pos_x_angle_+15_clearance_0.50",
            "aabb_face_pos_y", "aabb_face_neg_x", "aabb_face_neg_y",
        ],
        "container_m1_unavailable_staging_indices": [1],
        "container_m1_sampled_capture_poses_xyyaw": [[3.05, 0.0, 0.0]],
        "container_m1_capture_goal_xyyaw_by_staging_index": [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0], [3.05, 0.0, 2 * math.pi],
        ],
        "container_action_goal_xyyaw_by_staging_index": [[0.4, 0.0, 0.0]],
        "container_action_goal_xyyaw_options_by_staging_index": [
            [[0.4, 0.2, 0.0], [0.4, 0.0, 2 * math.pi], [0.4, 0.2, 0.0]],
        ],
    }}


def test_viewpoint_policies_keep_canonical_indices_and_do_not_mutate_candidates():
    candidate = _staged_candidate()
    before = deepcopy(candidate)
    # Requested 3 and its capture alias 4 have already been observed; 1 failed planning.
    assert container_approach.container_two_stage_m1_preflight_batch_indices(candidate, 3) == [0, 2]
    assert container_approach.container_two_stage_next_m1_viewpoint_index(
        candidate, excluded_indices=[0, 1],
    ) == 2
    assert container_approach.container_two_stage_face_indices(candidate, 0) == [0, 1]
    assert container_approach.container_two_stage_capture_goal_for_staging(candidate, 4) == (
        3.05, 0.0, 2 * math.pi,
    )
    assert container_approach.container_two_stage_action_goal_options_for_staging(candidate, 0) == [
        (0.4, 0.0, 0.0), (0.4, 0.2, 0.0),
    ]
    assert candidate == before


def _ready_update():
    return {
        "attribute_status": "ready", "is_currently_visible": True,
        "observed_bbox_2d": [10, 20, 110, 200], "observation_capture_step": 42,
        "view_state": "front", "front_surface_visible": True, "approach_ready": True,
    }


@pytest.mark.parametrize("changes,expected", [
    ({}, "ready"),
    ({"attribute_status": "waiting"}, "m1_attribute_status_waiting"),
    ({"is_currently_visible": False}, "m1_target_not_currently_visible"),
    ({"observed_bbox_2d": [0, 0, 0, 5]}, "m1_public_bbox_unavailable"),
    ({"observed_bbox_2d": [0, 0, math.nan, 5]}, "m1_public_bbox_unavailable"),
    ({"observation_capture_step": None}, "m1_capture_step_unavailable"),
    ({"observation_capture_step": 41}, "m1_capture_not_fresh"),
    ({"observation_capture_step": 40}, "m1_capture_not_fresh"),
    ({"view_state": "oblique"}, "m1_view_state_oblique"),
    ({"front_surface_visible": False}, "m1_front_surface_not_visible"),
    ({"approach_ready": False}, "m1_approach_not_ready"),
    ({"needs_reobserve": True}, "m1_needs_reobserve"),
    ({"visual_evidence_truncated": True}, "m1_visual_evidence_truncated"),
    ({"visual_evidence_truncated": True, "visual_evidence_truncated_edges": ["left"]},
     "m1_visual_evidence_laterally_truncated"),
    ({"visual_evidence_truncated": True, "visual_evidence_truncated_edges": ["bottom"]},
     "ready"),
])
def test_m1_ready_requires_fresh_visible_front_evidence(changes, expected):
    update = {**_ready_update(), **changes}
    request = {"minimum_capture_step": 41}
    before = deepcopy((update, request))
    assert container_m1_pre_action_ready(update, request) == expected
    assert (update, request) == before


def test_m1_oblique_acceptance_is_explicit_and_does_not_relax_freshness():
    update = {**_ready_update(), "view_state": "oblique"}
    assert container_m1_pre_action_ready(update, {}, require_direct_front=False) == "ready"
    assert container_m1_pre_action_ready(
        update, {"minimum_capture_step": 42}, require_direct_front=False,
    ) == "m1_capture_not_fresh"


@pytest.mark.parametrize("value,expected", [(0, 0), (42, 42), ("42", 42), (-1, None), (True, None), (None, None)])
def test_public_step_contract(value, expected):
    assert public_step_or_none(value) == expected


def test_public_bbox_normalizes_without_mutating_detector_coordinates():
    box = [110, 200, 10, 20]
    assert finite_public_bbox(box) == [10, 20, 110, 200]
    assert box == [110, 200, 10, 20]


def test_front_evidence_uses_capture_ray_not_unconfirmed_graph_axis():
    candidate = {"metadata": {
        "container_m1_front_axis_from_capture": True,
        "container_geometry_anchor_xy": [1.0, 2.0],
        "interaction_approach_axis_xy": [-1.0, 0.0],
    }}
    before = deepcopy(candidate)
    front = container_m1_front_axis_from_capture_pose(candidate, [1.0, 4.0, -math.pi / 2])
    assert front["m1_front_axis_xy"] == [0.0, 1.0]
    assert front["m1_front_yaw"] == pytest.approx(-math.pi / 2)
    assert front["m1_front_axis_source"] == "m1_confirmed_capture_pose"
    assert container_m1_front_axis_from_capture_pose(candidate, [1.0, 2.0, 0.0]) is None
    assert candidate == before


def test_selected_montage_face_survives_action_geometry_projection():
    candidate = {"metadata": {
        "container_two_stage_approach": True,
        "container_m1_front_axis_from_capture": True,
        "container_m1_face_selection_enabled": True,
        "container_geometry_anchor_xy": [1.0, 2.0],
        "container_geometry_aabb_size_xy": [2.0, 1.0],
        "container_physical_action_standoff_m": 0.6,
        "container_action_lateral_offset_m": 0.2,
    }}
    before = deepcopy(candidate)
    evidence = container_m1_front_axis_from_capture_pose(
        candidate, [3.0, 4.0, 0.0], selected_face_axis_xy=[0.0, 2.0],
        selected_face_index=3, selected_face_id="front-y",
    )
    evidence.update(capture_step=42, capture_pose_xyyaw=[3.0, 4.0, 0.0])
    options, labels, front = container_approach.container_two_stage_action_goal_options_from_m1_front_evidence(
        candidate, evidence,
    )
    assert options[0] == pytest.approx((1.0, 3.1, -math.pi / 2))
    assert options[1][:2] == pytest.approx((0.8, 3.1))
    assert options[2][:2] == pytest.approx((1.2, 3.1))
    assert len(labels) == 3
    assert front["m1_front_capture_step"] == 42
    assert evidence["m1_front_staging_index"] == 3
    assert candidate == before


def _command_candidate():
    return {
        "decision_id": "decision-1", "candidate_id": "candidate-1",
        "episode_id": "episode-1", "target_id": "node-1", "target_name": "fridge",
        "goal_xyyaw": [1.0, 2.0, 0.0],
        "metadata": {"semantic_type": "refrigerator", "private_graph": {"joint": "secret"}},
        "interaction_command": {
            "node_type": "container", "action": "open",
            "open_regions": [{"center": [0.2, 0.7]}],
            "visual_operation_plan": {"open_regions": [{"center": [0.2, 0.7]}]},
            "navigation_goal_position_tolerance_m": 0.1,
            "navigation_goal_yaw_tolerance_rad": 0.2,
            "navigation_goal_tolerance_contract_explicit": True,
            "private_joint_name": "do-not-send",
        },
    }


def _build(candidate, **kwargs):
    return build_interaction_command(
        candidate, command_id_base="decision:candidate", interaction_sequence=3, **kwargs,
    )


def test_public_command_ids_geometry_and_explicit_tolerances():
    candidate = _command_candidate()
    payload = _build(candidate, fallback_episode_id="wrong-episode")
    assert payload["command_id"] == "decision:candidate:interaction:003"
    assert payload["event_id"] == "decision-1_interaction_003"
    assert payload["episode_id"] == "episode-1"
    assert payload["node_id"] == "node-1"
    assert payload["object_id"] == "fridge"
    assert payload["target_kind"] == "fridge"
    assert payload["approach_goal_xyyaw"] == [1.0, 2.0, 0.0]
    assert payload["navigation_goal_position_tolerance_m"] == 0.1
    assert payload["navigation_goal_yaw_tolerance_rad"] == 0.2
    assert payload["navigation_goal_tolerance_contract_explicit"] is True
    assert "private_graph" not in payload
    assert "private_joint_name" not in payload
    candidate["episode_id"] = ""
    assert _build(candidate, fallback_episode_id="fallback")["episode_id"] == "fallback"


def test_command_payload_is_detached_from_nested_candidate_data():
    candidate = _command_candidate()
    before = deepcopy(candidate)
    payload = _build(candidate)
    assert candidate == before
    payload["open_regions"][0]["center"][0] = 0.99
    payload["visual_operation_plan"]["open_regions"][0]["center"][0] = 0.99
    payload["approach_goal_xyyaw"][0] = 100.0
    assert candidate == before


def test_portal_command_contains_only_public_aperture_fields():
    candidate = _command_candidate()
    candidate["interaction_command"].update({
        "node_type": "portal", "portal_aperture_observation": {
            "door_leaf": "absent", "connectivity": "open", "confidence": 0.9,
            "private_joint_name": "secret",
        },
    })
    payload = _build(candidate)
    assert payload["target_kind"] == "door"
    assert payload["portal_aperture_observation"] == {
        "door_leaf": "absent", "connectivity": "open", "confidence": 0.9,
    }


@pytest.mark.parametrize("opaque,action,expected", [(False, "close", "closed"), (True, "open", "open")])
def test_opaque_open_mode_preserves_existing_action_contract(opaque, action, expected):
    candidate = _command_candidate()
    candidate["interaction_command"]["action"] = "close"
    payload = _build(candidate, opaque_open_only=opaque)
    assert payload["action"] == action
    assert payload["expected_state"] == expected


@pytest.mark.parametrize("sequence,regions,fallback,valid", [
    ("drawer_scan", [], True, True),
    ("drawer_scan", [], False, False),
    ("drawer_open", [], True, False),
    ("drawer_open", [{"center": [0.2, 0.7]}], False, True),
    ("drawer_scan", [{"center": [math.nan, 0.7]}], True, False),
    ("drawer_scan", [{"center": [1.2, 0.7]}], True, False),
    ("drawer_scan", None, True, False),
])
def test_drawer_macro_requires_valid_visual_contract(sequence, regions, fallback, valid):
    candidate = _command_candidate()
    interaction = candidate["interaction_command"]
    interaction.update({
        "sequence_type": sequence, "open_regions": regions,
        "drawer_scan_fallback_to_all": fallback,
        "drawer_container_bbox_2d": [10, 20, 110, 200],
        "drawer_container_capture_step": 42,
    })
    assert has_valid_drawer_visual_contract(interaction) is valid
    if not valid:
        with pytest.raises(ValueError, match="drawer_visual_contract_missing_before_bridge"):
            _build(candidate)
    else:
        payload = _build(candidate)
        assert payload["drawer_container_bbox_2d"] == [10, 20, 110, 200]
        assert payload["drawer_container_capture_step"] == 42
        assert payload["sequence_type"] == sequence
