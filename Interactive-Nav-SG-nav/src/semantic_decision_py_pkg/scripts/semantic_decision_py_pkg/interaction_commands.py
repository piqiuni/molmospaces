"""Public backend command serialization; no IDs, state or ROS transports owned here."""

from __future__ import annotations

from copy import deepcopy
import math

from .container_evidence import finite_public_bbox, public_step_or_none
from .visual_interaction_planning import action_for_opaque_open_contract


def has_valid_drawer_visual_contract(interaction: dict) -> bool:
    sequence_type = str(interaction.get("sequence_type") or "").strip().casefold()
    regions = interaction.get("open_regions")
    if not isinstance(regions, list):
        return False
    if regions:
        valid_region = False
        for region in regions:
            if not isinstance(region, dict):
                continue
            center = region.get("center")
            if not isinstance(center, (list, tuple)) or len(center) < 2:
                continue
            try:
                x, y = float(center[0]), float(center[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                valid_region = True
                break
        if not valid_region:
            return False
    elif sequence_type != "drawer_scan" or not bool(
        interaction.get("drawer_scan_fallback_to_all", False)
    ):
        # ``drawer_open`` and legacy hand-authored commands must remain
        # grounded to at least one visible action region.  Only the sealed
        # drawer_scan macro may use the trusted bridge's explicit
        # all-slide-joints fallback when a low view has no usable region.
        return False
    if finite_public_bbox(
        interaction.get("drawer_container_bbox_2d")
    ) is None:
        return False
    return public_step_or_none(
        interaction.get("drawer_container_capture_step")
    ) is not None


def build_interaction_command(
    candidate: dict,
    *,
    command_id_base: str,
    interaction_sequence: int,
    fallback_episode_id: str = "",
    opaque_open_only: bool = False,
) -> dict:
    """Build a detached public payload from an already selected candidate.

    The executor allocates the sequence and binds result ownership. This helper
    must not advance a state machine, publish, or infer private graph geometry.
    """

    metadata = candidate.get("metadata") or {}
    interaction = candidate.get("interaction_command") or {}
    drawer_sequence_type = str(interaction.get("sequence_type") or "").casefold()
    if drawer_sequence_type in {"drawer_scan", "drawer_open"}:
        if not has_valid_drawer_visual_contract(interaction):
            raise ValueError("drawer_visual_contract_missing_before_bridge")
    action = action_for_opaque_open_contract(
        interaction.get("action", "open"),
        enabled=opaque_open_only,
    )
    target_kind = str(
        interaction.get("target_kind")
        or interaction.get("container_kind")
        or metadata.get("semantic_type")
        or metadata.get("semantic_class")
        or ""
    ).casefold()
    node_type = str(
        interaction.get("node_type") or metadata.get("node_type") or ""
    ).casefold()
    if node_type == "portal":
        target_kind = "door"
    elif target_kind in {"refrigerator", "refrigerator_door", "fridge_door"}:
        target_kind = "fridge"
    payload = {
        # The bridge deduplicates physical requests by command_id.  One
        # semantic decision may legitimately issue a second physical
        # attempt from a different bounded recovery anchor, so the
        # per-decision base id is not a sufficient physical request id.
        "command_id": (
            f"{command_id_base}:interaction:{interaction_sequence:03d}"
        ),
        "decision_id": candidate.get("decision_id", ""),
        "candidate_id": candidate.get("candidate_id", ""),
        "event_id": f"{candidate.get('decision_id', 'decision')}_interaction_{interaction_sequence:03d}",
        "episode_id": str(
            candidate.get("episode_id")
            or fallback_episode_id
            or ""
        ),
        "node_id": interaction.get("node_id", candidate.get("target_id", "")),
        "object_id": interaction.get("object_id", candidate.get("target_name", "")),
        "node_type": node_type,
        "target_kind": target_kind,
        "action": action,
        "interaction_mode": interaction.get("interaction_mode", "open_close"),
        "container_kind": str(interaction.get("container_kind") or ""),
        "expected_state": str(
            "closed"
            if action == "close"
            else "open"
            if action == "open"
            else interaction.get("expected_state") or ""
        ),
        "sequence_type": interaction.get("sequence_type", ""),
        "operation_method": interaction.get("operation_method", "unknown"),
        "open_regions": list(interaction.get("open_regions") or []),
        "approach_goal_xyyaw": list(
            interaction.get("interaction_approach_pose_xyyaw")
            or candidate.get("goal_xyyaw")
            or []
        ),
        "visual_operation_plan": dict(
            interaction.get("visual_operation_plan") or {}
        ),
        "interaction_approach_pose_xyyaw": list(
            interaction.get("interaction_approach_pose_xyyaw") or []
        ),
        "interaction_approach_axis_xy": list(
            interaction.get("interaction_approach_axis_xy") or []
        ),
        "interaction_target_center_xy": list(
            interaction.get("interaction_target_center_xy") or []
        ),
        "interaction_front_axis_source": str(
            interaction.get("interaction_front_axis_source") or ""
        ),
        "interaction_front_axis_validation_required": bool(
            interaction.get("interaction_front_axis_validation_required", False)
        ),
        "m1_observation_fallback": bool(
            metadata.get("interaction_observation_fallback_used", False)
        ),
        "m1_observation_fallback_reason": str(
            metadata.get("interaction_observation_fallback_reason") or ""
        ),
        "interaction_ready_distance_m": float(
            interaction.get("interaction_ready_distance_m", 0.45) or 0.45
        ),
        "interaction_ready_yaw_tolerance_rad": float(
            interaction.get("interaction_ready_yaw_tolerance_rad", 0.55) or 0.55
        ),
        # Preserve the explicit per-goal navigation contract through the
        # ROS JSON bridge.  Without forwarding these two fields the bridge
        # falls back to its broad legacy ready envelope and can accept a
        # pose that move_base was never required to converge to.
        "navigation_goal_position_tolerance_m": float(
            interaction.get(
                "navigation_goal_position_tolerance_m",
                interaction.get("interaction_ready_distance_m", 0.45),
            )
            or 0.45
        ),
        "navigation_goal_yaw_tolerance_rad": float(
            interaction.get(
                "navigation_goal_yaw_tolerance_rad",
                interaction.get("interaction_ready_yaw_tolerance_rad", 0.55),
            )
            or 0.55
        ),
        "navigation_goal_tolerance_contract_explicit": bool(
            interaction.get("navigation_goal_tolerance_contract_explicit", False)
        ),
        "interaction_front_position_tolerance_rad": float(
            interaction.get(
                "interaction_front_position_tolerance_rad",
                interaction.get("interaction_ready_yaw_tolerance_rad", 0.55),
            )
            or 0.55
        ),
        "interaction_front_yaw_tolerance_rad": float(
            interaction.get(
                "interaction_front_yaw_tolerance_rad",
                interaction.get("interaction_ready_yaw_tolerance_rad", 0.55),
            )
            or 0.55
        ),
    }
    # The bridge accepts only this compact public geometry contract for a
    # fixed portal.  Do not forward graph internals, source object names,
    # joints, or asset identifiers with the interaction command.
    aperture_observation = interaction.get("portal_aperture_observation")
    if payload["node_type"] == "portal" and isinstance(aperture_observation, dict):
        payload["portal_aperture_observation"] = {
            key: aperture_observation[key]
            for key in ("door_leaf", "connectivity", "confidence")
            if key in aperture_observation
        }
    if drawer_sequence_type in {"drawer_scan", "drawer_open"}:
        drawer_box = interaction.get("drawer_container_bbox_2d")
        if isinstance(drawer_box, (list, tuple)):
            payload["drawer_container_bbox_2d"] = list(drawer_box)
        drawer_capture_step = interaction.get("drawer_container_capture_step")
        if isinstance(drawer_capture_step, int) and not isinstance(
            drawer_capture_step, bool
        ):
            payload["drawer_container_capture_step"] = drawer_capture_step
    return deepcopy(payload)
