from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any


BEHAVIOR_EXPLORE = "EXPLORE"
BEHAVIOR_INTERACT = "INTERACT"
BEHAVIOR_NAVIGATE = "NAVIGATE"
BEHAVIOR_SCAN = "SCAN"

SPATIAL_CONTEXT_LABELS = {
    "bed",
    "bathtub",
    "cabinet",
    "counter",
    "countertop",
    "couch",
    "desk",
    "dishwasher",
    "dresser",
    "fridge",
    "microwave",
    "nightstand",
    "oven",
    "rack",
    "refrigerator",
    "shelf",
    "shower",
    "sink",
    "sofa",
    "stove",
    "table",
    "toilet",
    "tv",
    "wardrobe",
}


def _is_confirmed_portal_open_history(
    interaction: dict[str, Any], event: dict[str, Any]
) -> bool:
    """Whether an operation history entry may drive a post-open traversal.

    ``static_open`` acknowledges an existing fixed passage.  It must never be
    interpreted as a successful articulated open merely because its event uses
    the public ``action=open`` request verb.
    """

    capability = str(interaction.get("capability") or "").strip().casefold()
    if capability not in {"confirmed", "articulated"}:
        return False
    event_capability = str(
        event.get("interaction_capability") or event.get("capability") or ""
    ).strip().casefold()
    event_state = str(event.get("post_state") or "").strip().casefold()
    if event_capability in {
        "static",
        "blocked",
        "unsupported",
        "unavailable",
        "locked",
    } or event_state in {"static", "static_open", "static_closed"}:
        return False
    return bool(
        event.get("success")
        and str(event.get("action") or "").casefold() == "open"
        and event_state in {"open", "opened"}
        and len(list(event.get("approach_goal_xyyaw") or [])) >= 2
    )


@dataclass
class BehaviorCandidate:
    candidate_id: str
    behavior_type: str
    source: str
    target_id: str
    target_name: str
    goal_xyyaw: list[float] | None = None
    interaction_command: dict[str, Any] | None = None
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    score_terms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateGeneratorConfig:
    max_frontier_candidates: int = 12
    interaction_types: tuple[str, ...] = ("portal", "container")
    container_require_same_room: bool = False
    container_allow_connected_room: bool = False
    max_state_age_sec: float = 300.0
    min_state_confidence: float = 0.5
    # A portal's default graph state is deliberately not sufficient to open it
    # in the full-MLLM policy.  Wait for Module 1 to classify the observed
    # object so an already-open static door is not treated as a closed door.
    portal_require_attribute_ready: bool = False
    portal_allow_unknown_state: bool = True
    # Rule-only fallback for restricted-GT portals.  Their state confidence is
    # correctly zero while unobserved, but an unknown portal should still be
    # tried once so executor feedback can establish open/static_open/blocked.
    # The ROS candidate node enables this only for the rule policy lane; model
    # policies retain the ordinary visual-evidence gate.
    portal_unknown_default_interact: bool = False
    # A visual Module-1 ``unknown`` is neither an open command nor an absent
    # candidate.  It navigates to the normal portal standoff and asks for one
    # or more fresh observations before an action can be dispatched.
    portal_unknown_observation_max_attempts: int = 2
    portal_standoff_m: float = 1.0
    portal_traversal_distance_m: float = 0.9
    portal_traversal_max_start_distance_m: float = 2.0
    portal_traversal_completion_margin_m: float = 0.35
    container_standoff_m: float = 1.0
    # Refrigerator leaves need more clearance than a drawer front.  This is a
    # public type-level safety policy, not an oracle hinge direction: the
    # simulator still decides whether the actual leaf sweep is safe.
    fridge_standoff_m: float | None = None
    # A remembered container AABB has no reliable semantic "front" when the
    # perception stream did not publish one.  Keep several physically distinct
    # standoff viewpoints so fresh Module 1 can reject an oblique/side view and
    # ask the executor to re-observe from the next one instead of opening from
    # a radial guess.
    container_multiview_enabled: bool = True
    container_multiview_face_count: int = 4
    # When enabled, AABB geometry contributes only the four cardinal face
    # candidates.  A current robot-side ray or remembered graph axis must not
    # select the face; the targeted M1 result is the face authorization token.
    container_m1_face_selection_enabled: bool = False
    # Refrigerator leaves can hide the interior from widely separated side
    # poses. Scale only the fridge multiview angular offsets; 0.5 maps the
    # former 0/+90/-90/180 degree set to 0/+45/-45/90 degrees.
    fridge_multiview_angular_scale: float = 1.0
    # Full-MLLM drawer scans require one targeted, causally later M1 view at
    # the arrived approach pose.  That view supplies frontality plus visible
    # drawer action regions; it is never replaced with a scene/joint fallback.
    drawer_pre_action_mllm: bool = False
    drawer_pre_action_observation_max_attempts: int = 2
    # Containers use the same fresh-view contract before a physical open.  A
    # remembered AABB/ring pose may be useful to obtain a view, but is never by
    # itself authorization to pull/open a fridge, cabinet, or drawer.
    container_pre_action_mllm: bool = False
    container_pre_action_observation_max_attempts: int = 4
    # ``None`` means drawers inherit ``container_standoff_m``.  These legacy
    # fields describe the requested physical standoff *before* the legacy
    # safety margin is applied.  New model-lane configs should instead use the
    # explicit action/capture clearances below, so a visual observation never
    # silently inherits a physical-distance calculation.
    drawer_standoff_m: float | None = None
    # Explicit AABB-surface clearances for the three distinct container phases.
    # ``*_action_standoff_m`` includes any desired safety allowance and is the
    # final physical bridge target.  ``*_m1_capture_standoff_m`` is the actual
    # visual evidence pose.  ``None`` preserves legacy configs: action falls
    # back to the old standoff + margin and capture falls back to the old
    # observation field.  A capture nearer than its action pose is rejected by
    # the resolver and explicitly falls back to the action clearance; this is
    # a compatibility validation path, not the former runtime ``max(a, b)``
    # policy.
    container_action_standoff_m: float | None = None
    drawer_action_standoff_m: float | None = None
    fridge_action_standoff_m: float | None = None
    container_m1_capture_standoff_m: float | None = None
    drawer_m1_capture_standoff_m: float | None = None
    fridge_m1_capture_standoff_m: float | None = None
    # Deprecated names retained as a legacy capture-clearance fallback.  New
    # configs should use the explicit ``*_m1_capture_standoff_m`` fields.
    container_observation_standoff_m: float = 0.7
    drawer_observation_standoff_m: float = 0.65
    fridge_observation_standoff_m: float = 0.8
    # Navigation anchors are recovery / costmap escape points only.  M1 must
    # not be queried from these offsets: the executor first maps an anchor to
    # the direct capture point above.  The old safe-staging names remain as
    # fallbacks so external policy YAMLs do not break during this migration.
    container_navigation_anchor_outer_offset_m: float | None = None
    container_navigation_anchor_ring_count: int | None = None
    container_navigation_anchor_tangent_offset_m: float | None = None
    # In the simplified model lane, one selected anchor is the complete
    # interaction contract: move_base arrival, targeted M1 capture, and the
    # physical bridge all bind to exactly the same x/y/yaw.  Keep this opt-in so
    # legacy traces with a separate outer/capture/action mapping remain readable.
    container_anchor_shared_pose_enabled: bool = False
    # Type-specific navigation anchors can be sampled from the actual AABB
    # surface in a bounded fan around each cardinal face normal.  M1 still
    # authorizes which geometric face is the semantic front.
    # Clearances are measured from the ray/box intersection, never the centre.
    drawer_navigation_anchor_aabb_fan_enabled: bool = False
    drawer_navigation_anchor_fan_clearances_m: tuple[float, ...] = (
        0.50,
        0.85,
        1.20,
    )
    drawer_navigation_anchor_fan_angles_deg: tuple[float, ...] = (
        -30.0,
        -15.0,
        0.0,
        15.0,
        30.0,
    )
    fridge_navigation_anchor_aabb_fan_enabled: bool = False
    fridge_navigation_anchor_fan_clearances_m: tuple[float, ...] = (
        1.15,
        1.35,
        1.55,
    )
    fridge_navigation_anchor_fan_angles_deg: tuple[float, ...] = (
        -15.0,
        0.0,
        15.0,
    )
    # Deprecated navigation-anchor aliases.
    container_safe_staging_outer_offset_m: float = 0.35
    # Include one additional robot-side/four-view ring beyond ``safe_far`` by
    # default.  It gives a wall-adjacent container a costmap-reachable visual
    # staging option without falling back to an inner/close M1 pose.  Keep the
    # enumeration bounded because each option still participates in normal
    # preflight and visual re-observation accounting.
    container_safe_staging_ring_count: int = 3
    # Keep the normal safe-outer standoff while adding at most one bounded
    # left/right tangential view for each face.  A zero value preserves the
    # original ring exactly; non-zero values are still ordinary move_base/M1
    # staging candidates and never authorize a physical action by themselves.
    container_safe_staging_tangent_offset_m: float = 0.0
    # The two-stage M1 gate may need to inspect more than the legacy four
    # observations because a finite outer ring can contain several
    # costmap-reachable sides.  Keep this explicit and bounded; it is not a
    # retry-until-success policy.  The candidate also clamps it to the number
    # of generated staging goals.
    container_two_stage_observation_max_attempts: int = 12
    # M1 is a stochastic visual authorizer.  A negative result first receives
    # a fresh same-pose confirmation, then consumes a distinct capture
    # viewpoint.  These caps are deliberately separate from the navigation
    # anchor count so one image flip cannot exhaust a container candidate.
    container_m1_same_pose_samples_per_view: int = 2
    container_m1_max_viewpoints: int = 4
    container_m1_max_total_requests: int = 8
    # M1-only staging accepts the ordinary move_base terminal error separately
    # from the stricter physical bridge gate.  The default offset grows by the
    # same 5 cm as this envelope, preserving the previous minimum clearance to
    # the container surface.
    container_safe_staging_arrival_tolerance_m: float = 0.30
    container_interaction_ready_distance_m: float = 0.18
    # A global plan can reach the nominal physical stance while the local
    # controller cannot settle there beside an appliance corner.  Keep two
    # bounded tangential alternatives at the same normal clearance; they are
    # tried only after a fresh M1 acceptance for that exact outer staging face.
    # This is not a smaller standoff or a costmap exception.
    container_action_lateral_offset_m: float = 0.22
    # M1 does not estimate a map-space normal from one image.  Once it confirms
    # that a direct capture is frontal, the executor freezes that calibrated
    # capture ray as the physical-action face for this decision.
    container_m1_front_axis_from_capture: bool = True
    interaction_safety_margin_m: float = 0.0
    interaction_ready_distance_m: float = 0.45
    # This is the public yaw gate shared by final approach validation and the
    # physical bridge.  Keep the conservative legacy default while allowing a
    # policy YAML to require a more front-facing interaction pose.
    interaction_ready_yaw_tolerance_rad: float = 0.55
    require_current_visibility: bool = False
    # A remembered portal may remain in the graph after leaving the camera
    # view.  Physical portal interaction is stricter than ordinary graph
    # retention: the current RGB observation must contain enough of the leaf
    # to make the approach decision meaningful.  A confirmed successful open
    # is stored as an interaction postcondition and is not regenerated by this
    # candidate path, so this gate does not erase confirmed state.
    portal_require_current_visibility: bool = False
    portal_min_visible_pixels: int = 128
    portal_min_visible_fraction: float = 0.2
    remembered_portal_reobservation_enabled: bool = False
    target_standoff_m: float = 1.0
    target_max_state_age_sec: float = 300.0
    target_require_current_visibility: bool = False
    target_require_same_room: bool = False
    target_allow_connected_room: bool = False
    target_require_visibility_verification: bool = True
    target_min_visible_pixels: int = 16
    target_min_visible_fraction: float = 0.2
    target_min_consecutive_observations: int = 2
    target_arrival_tolerance_m: float = 0.35


class CandidateGenerator:
    def __init__(self, config: CandidateGeneratorConfig | None = None) -> None:
        self.config = config or CandidateGeneratorConfig()

    def generate(
        self,
        explorer_status: dict[str, Any] | None,
        graph: dict[str, Any] | None,
        robot_xy: tuple[float, float] | None,
        target_context: dict[str, Any] | None = None,
    ) -> list[BehaviorCandidate]:
        if (explorer_status or {}).get("initial_scan_complete") is False:
            return []
        candidates = self._frontier_candidates(explorer_status or {})
        if robot_xy is not None:
            candidates.extend(
                self._interaction_candidates(
                    graph or {}, robot_xy, target_context or {}
                )
            )
            candidates.extend(
                self._portal_traversal_candidates(graph or {}, robot_xy)
            )
            candidates.extend(
                self._target_candidates(graph or {}, robot_xy, target_context or {})
            )
            if not candidates and self.config.remembered_portal_reobservation_enabled:
                candidates.extend(
                    self._remembered_portal_reobservation_candidates(
                        graph or {}, robot_xy
                    )
                )
        self._attach_portal_child_room_context(candidates, graph or {})
        self._attach_frontier_room_context(candidates, graph or {}, robot_xy)
        self._attach_spatial_context(candidates, graph or {})
        candidates = self._limit_frontier_candidates(candidates)
        return sorted(candidates, key=lambda candidate: candidate.candidate_id)

    def _remembered_portal_reobservation_candidates(
        self,
        graph: dict[str, Any],
        robot_xy: tuple[float, float],
    ) -> list[BehaviorCandidate]:
        """Keep a concrete subgoal while unresolved doors are out of view.

        Full-MLLM physical door candidates deliberately require current pixels.
        That visibility gate must not turn remembered unresolved doors into a
        permanently empty candidate stream.  Navigate to the ordinary safe
        portal approach first; the next graph revision can then emit the real
        visually-authorized INTERACT candidate.
        """

        candidates: list[BehaviorCandidate] = []
        for node in graph.get("nodes") or []:
            if str(node.get("type") or "").strip().casefold() != "portal":
                continue
            interaction = node.get("interaction") or {}
            if not bool(
                interaction.get(
                    "requires_interaction", node.get("requires_interaction", False)
                )
            ):
                continue
            state = str(
                interaction.get("state", node.get("interaction_state")) or "unknown"
            ).strip().casefold()
            capability = str(
                interaction.get("capability", node.get("interaction_capability")) or ""
            ).strip().casefold()
            if state in {"open", "opened", "succeeded", "satisfied"} or capability in {
                "blocked",
                "unavailable",
                "unsupported",
                "locked",
            }:
                continue
            attributes = node.get("attributes") or {}
            visible_now = bool(node.get("is_currently_visible")) and int(
                attributes.get("visible_pixels", 0) or 0
            ) >= int(self.config.portal_min_visible_pixels) and float(
                attributes.get("visible_fraction", 0.0) or 0.0
            ) >= float(self.config.portal_min_visible_fraction)
            if visible_now:
                continue
            position = list(
                node.get("aabb_center")
                or node.get("centroid")
                or node.get("position")
                or []
            )
            if len(position) < 2:
                continue
            target_xy = (float(position[0]), float(position[1]))
            goals, labels = self._approach_candidates(
                robot_xy,
                target_xy,
                node,
                self.config.portal_standoff_m,
                "portal",
            )
            if not goals:
                continue
            node_id = str(node.get("id") or "")
            candidates.append(
                BehaviorCandidate(
                    candidate_id=f"reobserve_portal:{node_id}",
                    behavior_type=BEHAVIOR_NAVIGATE,
                    source="remembered_interaction_reobserve",
                    target_id=node_id,
                    target_name=str(node.get("label") or node.get("name") or "door"),
                    goal_xyyaw=list(goals[0]),
                    features={
                        "exploration_gain": 0.8,
                        "visibility_gain": 1.0,
                        "semantic_gain": 0.8,
                        "distance_m": math.hypot(
                            float(goals[0][0]) - robot_xy[0],
                            float(goals[0][1]) - robot_xy[1],
                        ),
                        "priority": 0.9,
                    },
                    metadata={
                        "reobserve_interaction_target": True,
                        "requires_approach": True,
                        "goal_xyyaw_candidates": [list(goal) for goal in goals],
                        "interaction_approach_pose_labels": list(labels),
                        "is_currently_visible": False,
                    },
                )
            )
        return candidates

    def _target_candidates(
        self,
        graph: dict[str, Any],
        robot_xy: tuple[float, float],
        target_context: dict[str, Any],
    ) -> list[BehaviorCandidate]:
        if not bool(target_context.get("enabled")):
            return []
        candidates = []
        nodes_by_id = {
            str(node.get("id") or ""): node
            for node in graph.get("nodes") or []
            if str(node.get("id") or "")
        }
        robot_room_id = self._room_id_for_xy(graph, robot_xy)
        for node in graph.get("nodes") or []:
            if str(node.get("type") or "") in {"scene", "room", "portal"}:
                continue
            if not self._matches_target(node, target_context):
                continue
            target_room_id = node.get("room_id")
            require_same_room = bool(
                target_context.get(
                    "require_same_room",
                    self.config.target_require_same_room,
                )
            )
            allow_connected_room = bool(
                target_context.get(
                    "allow_connected_room",
                    self.config.target_allow_connected_room,
                )
            )
            room_hops = self._room_hops(graph, robot_room_id, target_room_id)
            if (
                require_same_room
                and target_room_id is not None
                and robot_room_id is not None
                and int(target_room_id) != int(robot_room_id)
                and not (allow_connected_room and room_hops is not None)
            ):
                continue
            state_age_sec = max(0.0, float(node.get("state_age_sec", 0.0) or 0.0))
            if state_age_sec > self.config.target_max_state_age_sec:
                continue
            navigation_anchor = self._target_navigation_anchor(
                node,
                graph,
                nodes_by_id,
            )
            position = self._node_xy(navigation_anchor, prefer_aabb=True)
            if position is None:
                continue
            previous_interaction_goal = self._successful_interaction_approach(
                navigation_anchor
            )
            configured_interaction_goal = (
                self._container_approach_pose(navigation_anchor)
                if navigation_anchor is not node
                else None
            )
            goal = previous_interaction_goal or configured_interaction_goal or self._approach_pose(
                robot_xy,
                position,
                float(target_context.get("standoff_m", self.config.target_standoff_m)),
                node=navigation_anchor,
                fixed_axis=self._container_approach_axis(navigation_anchor),
            )
            distance_m = math.hypot(goal[0] - robot_xy[0], goal[1] - robot_xy[1])
            target_arrival_tolerance_m = max(
                0.0,
                float(
                    target_context.get(
                        "arrival_tolerance_m",
                        self.config.target_arrival_tolerance_m,
                    )
                ),
            )
            node_id = str(node.get("id") or "")
            containing_container = (
                navigation_anchor
                if navigation_anchor is not node
                and str(navigation_anchor.get("type") or "") == "container"
                else None
            )
            attributes = node.get("attributes") or {}
            visible_pixels = int(attributes.get("visible_pixels", 0) or 0)
            target_min_visible_pixels = int(
                target_context.get(
                    "min_visible_pixels",
                    self.config.target_min_visible_pixels,
                )
            )
            visible_fraction = float(
                attributes.get("visible_fraction", 1.0) or 0.0
            )
            consecutive_observations = int(
                attributes.get("consecutive_observations", 2) or 0
            )
            target_min_visible_fraction = float(
                target_context.get(
                    "min_visible_fraction",
                    self.config.target_min_visible_fraction,
                )
            )
            target_min_consecutive_observations = int(
                target_context.get(
                    "min_consecutive_observations",
                    self.config.target_min_consecutive_observations,
                )
            )
            target_visible_now = (
                bool(node.get("is_currently_visible"))
                and visible_pixels >= target_min_visible_pixels
                and visible_fraction >= target_min_visible_fraction
                and consecutive_observations >= target_min_consecutive_observations
            )
            max_visible_pixels = int(
                attributes.get("max_visible_pixels", visible_pixels) or 0
            )
            max_visible_fraction = float(
                attributes.get("max_visible_fraction", visible_fraction) or 0.0
            )
            max_consecutive_observations = int(
                attributes.get("max_consecutive_observations", consecutive_observations)
                or 0
            )
            reliably_observed = (
                max_visible_pixels >= target_min_visible_pixels
                and max_visible_fraction >= target_min_visible_fraction
                and max_consecutive_observations >= target_min_consecutive_observations
            )
            if not reliably_observed:
                continue
            require_current_visibility = bool(
                target_context.get(
                    "require_current_visibility",
                    self.config.target_require_current_visibility,
                )
            )
            if require_current_visibility and not target_visible_now:
                continue
            verify_visibility = bool(
                target_context.get(
                    "completion_requires_visibility",
                    self.config.target_require_visibility_verification,
                )
            )
            candidates.append(
                BehaviorCandidate(
                    candidate_id=f"target:{node_id}",
                    behavior_type=BEHAVIOR_NAVIGATE,
                    source="unified_graph_target",
                    target_id=node_id,
                    target_name=str(node.get("name") or node.get("label") or node_id),
                    goal_xyyaw=goal,
                    features={
                        "exploration_gain": 0.0,
                        "visibility_gain": 0.2 if target_visible_now else 1.0,
                        "semantic_gain": 1.0,
                        "target_relevance": 1.0,
                        "distance_m": distance_m,
                        "interaction_cost": 0.0,
                        "state_age_ratio": min(
                            1.0,
                            state_age_sec / max(self.config.target_max_state_age_sec, 1e-6),
                        ),
                        "confidence": float(node.get("confidence", 1.0) or 1.0),
                        "priority": 1.0,
                    },
                    metadata={
                        "target_goal": True,
                        "target_context": dict(target_context),
                        "node_type": str(node.get("type") or "object"),
                        "navigation_anchor_id": str(
                            navigation_anchor.get("id") or node_id
                        ),
                        "navigation_anchor_type": str(
                            navigation_anchor.get("type") or node.get("type") or "object"
                        ),
                        "is_currently_visible": bool(node.get("is_currently_visible")),
                        "target_visible_now": target_visible_now,
                        "target_require_current_visibility": require_current_visibility,
                        "target_require_same_room": require_same_room,
                        "target_allow_connected_room": allow_connected_room,
                        "target_room_id": target_room_id,
                        "robot_room_id": robot_room_id,
                        "room_transition_required": room_hops not in {None, 0},
                        "room_hops": room_hops,
                        "room_reachable": room_hops is not None,
                        "target_visible_pixels": visible_pixels,
                        "target_max_visible_pixels": max_visible_pixels,
                        "target_visible_fraction": visible_fraction,
                        "target_max_visible_fraction": max_visible_fraction,
                        "target_consecutive_observations": consecutive_observations,
                        "target_max_consecutive_observations": max_consecutive_observations,
                        "target_reliably_observed": reliably_observed,
                        "target_goal_distance_m": distance_m,
                        "target_arrival_tolerance_m": target_arrival_tolerance_m,
                        "target_navigation_required": (
                            distance_m > target_arrival_tolerance_m
                        ),
                        "target_min_visible_fraction": target_min_visible_fraction,
                        "target_min_consecutive_observations": target_min_consecutive_observations,
                        "verify_target_visibility": verify_visibility,
                        "target_min_visible_pixels": target_min_visible_pixels,
                        "state_age_sec": state_age_sec,
                        "approach_strategy": (
                            "target_last_successful_interaction_pose"
                            if previous_interaction_goal is not None
                            else "target_containing_container_pose"
                            if configured_interaction_goal is not None
                            else "target_container_front_axis"
                            if navigation_anchor is not node
                            and self._container_approach_axis(navigation_anchor) is not None
                            else "target_parent_container_standoff"
                            if navigation_anchor is not node
                            else "target_container_front_axis"
                            if self._container_approach_axis(node) is not None
                            else "target_radial_standoff"
                        ),
                        "interaction_approach_axis_xy": self._container_approach_axis(
                            navigation_anchor
                        ),
                        "containing_container_id": (
                            str(containing_container.get("id") or "")
                            if containing_container is not None
                            else ""
                        ),
                        "direct_goal_tolerance_m": (
                            0.45 if containing_container is not None else 0.0
                        ),
                        "direct_goal_yaw_tolerance_rad": (
                            0.65 if containing_container is not None else 0.0
                        ),
                    },
                )
            )
        return candidates

    @staticmethod
    def _containing_container_node(
        graph: dict[str, Any], target_node_id: str
    ) -> dict[str, Any] | None:
        container_ids = {
            str(edge.get("src_id") or "")
            for edge in graph.get("edges") or []
            if str(edge.get("relation") or "") == "contains"
            and str(edge.get("dst_id") or "") == str(target_node_id)
        }
        if not container_ids:
            return None
        return next(
            (
                node
                for node in graph.get("nodes") or []
                if str(node.get("id") or "") in container_ids
                and str(node.get("type") or "") == "container"
            ),
            None,
        )

    @staticmethod
    def _matches_target(node: dict[str, Any], target_context: dict[str, Any]) -> bool:
        attributes = node.get("attributes") or {}
        requested_instance_id = str(target_context.get("target_instance_id") or "").strip().casefold()
        observed_instance_id = str(
            attributes.get("instance_id") or node.get("id") or ""
        ).strip().casefold()
        if requested_instance_id:
            return requested_instance_id == observed_instance_id
        requested_source_name = str(
            target_context.get("target_source_object_name") or ""
        ).strip().casefold()
        observed_source_name = str(
            attributes.get("source_object_name") or node.get("name") or ""
        ).strip().casefold()
        if requested_source_name:
            return requested_source_name == observed_source_name
        requested = list(target_context.get("object_labels") or [])
        for key in ("object_label", "target_object", "target_name"):
            value = target_context.get(key)
            if value:
                requested.append(value)
        requested_tokens = {
            str(value).strip().casefold() for value in requested if str(value).strip()
        }
        if not requested_tokens:
            return False
        observed = {
            str(value).strip().casefold()
            for value in (
                node.get("label"),
                node.get("name"),
                attributes.get("category"),
                attributes.get("semantic_name"),
                attributes.get("source_object_name"),
            )
            if str(value or "").strip()
        }
        return any(
            requested == observed_value
            or requested in observed_value
            or observed_value in requested
            for requested in requested_tokens
            for observed_value in observed
        )

    @classmethod
    def _matches_interaction_target(
        cls, node: dict[str, Any], target_context: dict[str, Any]
    ) -> bool:
        if cls._matches_target(node, target_context):
            return True
        attributes = node.get("attributes") or {}
        requested_instance_id = str(
            target_context.get("target_container_instance_id") or ""
        ).strip().casefold()
        observed_instance_id = str(
            attributes.get("instance_id") or node.get("id") or ""
        ).strip().casefold()
        if requested_instance_id:
            return requested_instance_id == observed_instance_id
        requested_source_name = str(
            target_context.get("target_container_source_object_name") or ""
        ).strip().casefold()
        observed_source_name = str(
            attributes.get("source_object_name") or node.get("name") or ""
        ).strip().casefold()
        if requested_source_name:
            return requested_source_name == observed_source_name
        if not bool(target_context.get("require_interaction")):
            return False
        requested_labels = list(target_context.get("target_container_labels") or [])
        if target_context.get("target_container_name"):
            requested_labels.append(target_context["target_container_name"])
        requested_tokens = {
            str(value).strip().casefold() for value in requested_labels if str(value).strip()
        }
        observed = {
            str(value).strip().casefold()
            for value in (
                node.get("label"),
                node.get("name"),
                attributes.get("category"),
                attributes.get("semantic_name"),
                attributes.get("source_object_name"),
            )
            if str(value or "").strip()
        }
        return any(
            requested == observed_value
            or requested in observed_value
            or observed_value in requested
            for requested in requested_tokens
            for observed_value in observed
        )

    def _frontier_candidates(self, status: dict[str, Any]) -> list[BehaviorCandidate]:
        is_raw_proposal_stream = (
            "proposals" in status or "exploration_proposals" in status
        )
        raw_proposals = status.get("proposals")
        if raw_proposals is None:
            raw_proposals = status.get("exploration_proposals")
        if isinstance(raw_proposals, dict):
            raw_proposals = raw_proposals.get("proposals") or []
        proposals = list(
            (raw_proposals or [])
            if is_raw_proposal_stream
            else (status.get("frontier_clusters") or [])
        )
        if not proposals:
            return []
        if is_raw_proposal_stream:
            proposals.sort(
                key=lambda proposal: (
                    -float(
                        (proposal.get("raw_features") or {}).get(
                            "expected_visible_unknown_area_m2", 0.0
                        )
                    ),
                    -float(
                        (proposal.get("raw_features") or {}).get(
                            "unknown_component_area_m2", 0.0
                        )
                    ),
                    -float(
                        (proposal.get("raw_features") or {}).get(
                            "information_gain", 0.0
                        )
                    ),
                    float(
                        (proposal.get("raw_features") or {}).get("distance_m", 0.0)
                    ),
                    str(proposal.get("proposal_id") or proposal.get("cluster_id") or ""),
                )
            )
        else:
            proposals.sort(
                key=lambda proposal: (
                    -float(
                        proposal.get("expected_visible_unknown_area_m2", 0.0)
                        or 0.0
                    ),
                    -float(proposal.get("unknown_component_area_m2", 0.0) or 0.0),
                    -float(proposal.get("information_gain", 0.0) or 0.0),
                    float(proposal.get("distance_to_robot", 0.0) or 0.0),
                    str(proposal.get("cluster_id") or ""),
                )
            )
        raw_information = []
        raw_unknown_areas = []
        raw_expected_visible_areas = []
        for proposal in proposals:
            value = (
                (proposal.get("raw_features") or {}).get("information_gain", 0.0)
                if is_raw_proposal_stream
                else proposal.get("information_gain", 0.0)
            )
            raw_information.append(max(0.0, float(value or 0.0)))
            area_value = (
                (proposal.get("raw_features") or {}).get(
                    "unknown_component_area_m2", 0.0
                )
                if is_raw_proposal_stream
                else proposal.get("unknown_component_area_m2", 0.0)
            )
            raw_unknown_areas.append(max(0.0, float(area_value or 0.0)))
            visible_area_value = (
                (proposal.get("raw_features") or {}).get(
                    "expected_visible_unknown_area_m2", 0.0
                )
                if is_raw_proposal_stream
                else proposal.get("expected_visible_unknown_area_m2", 0.0)
            )
            if not visible_area_value:
                frontier_length_m = (
                    (proposal.get("raw_features") or {}).get(
                        "frontier_length_m", 0.0
                    )
                    if is_raw_proposal_stream
                    else proposal.get("frontier_length_m", 0.0)
                )
                visible_area_value = min(
                    max(0.0, float(area_value or 0.0)),
                    max(0.0, float(frontier_length_m or 0.0)) * 5.0,
                )
            raw_expected_visible_areas.append(
                max(0.0, float(visible_area_value or 0.0))
            )
        max_information = max(raw_information) if raw_information else 1.0
        max_unknown_area = max(raw_unknown_areas) if raw_unknown_areas else 0.0
        max_expected_visible_area = (
            max(raw_expected_visible_areas) if raw_expected_visible_areas else 0.0
        )
        map_resolution = max(0.0, float(status.get("map_resolution", 0.0) or 0.0))
        candidates = []
        for proposal, information_gain, unknown_area_m2, expected_visible_area_m2 in zip(
            proposals,
            raw_information,
            raw_unknown_areas,
            raw_expected_visible_areas,
        ):
            cluster_id = str(
                proposal.get("proposal_id") or proposal.get("cluster_id") or ""
            )
            if is_raw_proposal_stream:
                subgoal = list(proposal.get("goal_xyyaw") or [])
                frontier_point = list(proposal.get("frontier_point") or [])
                raw_features = dict(proposal.get("raw_features") or {})
                geometry = dict(proposal.get("geometry") or {})
                explorer_score = float(geometry.get("proposal_score", 0.0) or 0.0)
                explorer_score_terms = dict(geometry.get("proposal_score_terms") or {})
                distance_m = max(
                    0.0, float(raw_features.get("distance_m", 0.0) or 0.0)
                )
            else:
                subgoal = list(proposal.get("subgoal_world") or [])
                frontier_point = list(proposal.get("centroid_world") or [])
                raw_features = {}
                geometry = {}
                explorer_score = float(proposal.get("score", 0.0) or 0.0)
                explorer_score_terms = dict(proposal.get("score_terms") or {})
                distance_m = max(
                    0.0,
                    float(proposal.get("distance_to_robot", 0.0) or 0.0),
                )
            if not cluster_id or len(subgoal) < 2:
                continue
            yaw = (
                float(subgoal[2])
                if len(subgoal) > 2
                else float(proposal.get("subgoal_yaw", 0.0))
            )
            information_normalized = (
                math.log1p(information_gain) / max(math.log1p(max_information), 1e-6)
            )
            unknown_area_normalized = (
                math.log1p(unknown_area_m2)
                / max(math.log1p(max_unknown_area), 1e-6)
                if max_unknown_area > 0.0
                else information_normalized
            )
            expected_visible_area_normalized = (
                math.log1p(expected_visible_area_m2)
                / max(math.log1p(max_expected_visible_area), 1e-6)
                if max_expected_visible_area > 0.0
                else unknown_area_normalized
            )
            candidates.append(
                BehaviorCandidate(
                    candidate_id=f"frontier:{cluster_id}",
                    behavior_type=BEHAVIOR_EXPLORE,
                    source="explore_py",
                    target_id=cluster_id,
                    target_name=cluster_id,
                    goal_xyyaw=[float(subgoal[0]), float(subgoal[1]), yaw],
                    features={
                        "exploration_gain": expected_visible_area_normalized,
                        "visibility_gain": expected_visible_area_normalized,
                        "semantic_gain": 0.0,
                        "distance_m": distance_m,
                        "interaction_cost": 0.0,
                        "state_age_ratio": 0.0,
                        "confidence": 1.0,
                        "priority": 0.0,
                    },
                    metadata={
                        "cluster_id": cluster_id,
                        "frontier_point": frontier_point,
                        "cell_count": int(
                            raw_features.get(
                                "frontier_cell_count", proposal.get("cell_count", 0)
                            )
                        ),
                        "unknown_component_area_m2": unknown_area_m2,
                        "expected_visible_unknown_area_m2": expected_visible_area_m2,
                        "frontier_length_m": float(
                            raw_features.get(
                                "frontier_length_m",
                                proposal.get("frontier_length_m", 0.0),
                            )
                            or 0.0
                        ),
                        "map_resolution": map_resolution,
                        "explorer_score": explorer_score,
                        "explorer_score_terms": explorer_score_terms,
                        "proposal_source": str(proposal.get("source") or "explore_py"),
                        "hard_constraints_passed": bool(
                            (geometry.get("hard_constraints_passed", True))
                        ),
                    },
                )
            )
        return candidates

    @classmethod
    def _attach_spatial_context(
        cls,
        candidates: list[BehaviorCandidate],
        graph: dict[str, Any],
        max_distance_m: float = 3.0,
    ) -> None:
        semantic_nodes = []
        for node in graph.get("nodes") or []:
            node_type = str(node.get("type") or "").casefold()
            if node_type in {"scene", "room", "portal"}:
                continue
            label = str(node.get("label") or node.get("name") or "").strip()
            normalized_label = label.casefold().replace(" ", "_")
            if (
                node_type not in {"container", "support"}
                and normalized_label not in SPATIAL_CONTEXT_LABELS
            ):
                continue
            position = cls._node_xy(node, prefer_aabb=True)
            if position is None:
                continue
            semantic_nodes.append(
                {
                    "id": str(node.get("id") or ""),
                    "type": node_type,
                    "label": label or node_type,
                    "position": position,
                    "is_currently_visible": bool(node.get("is_currently_visible")),
                }
            )
        for candidate in candidates:
            goal = list(candidate.goal_xyyaw or [])
            if len(goal) < 2:
                continue
            nearby = []
            for node in semantic_nodes:
                distance_m = math.hypot(
                    float(goal[0]) - node["position"][0],
                    float(goal[1]) - node["position"][1],
                )
                if node["id"] == str(candidate.target_id):
                    candidate.metadata["goal_to_subject_distance_m"] = distance_m
                    continue
                if distance_m <= max_distance_m:
                    nearby.append(
                        {
                            "id": node["id"],
                            "type": node["type"],
                            "label": node["label"],
                            "distance_m": round(distance_m, 2),
                            "visible": node["is_currently_visible"],
                        }
                    )
            nearby.sort(key=lambda item: (item["distance_m"], item["id"]))
            if nearby:
                candidate.metadata["nearby_semantic_nodes"] = nearby[:3]

    @staticmethod
    def _attach_portal_child_room_context(
        candidates: list[BehaviorCandidate], graph: dict[str, Any]
    ) -> None:
        """Associate near-door frontiers with graph-only portal child rooms.

        The mapping side never assigns the child ID to unknown occupancy cells.
        This bounded geometric association is only a candidate label, so a
        fresh frontier just beyond an opened door can retain a stable room
        identity until ordinary free-space segmentation observes that room.
        """

        potential_rooms = []
        for node in graph.get("nodes") or []:
            if str(node.get("type") or "").casefold() != "room":
                continue
            attributes = node.get("attributes") or {}
            if not bool(attributes.get("is_potential_room")) or not bool(
                attributes.get("active", True)
            ):
                continue
            room_id = node.get("room_id")
            center = list(node.get("aabb_center") or node.get("centroid") or [])
            size = list(node.get("aabb_size") or [])
            if room_id is None or len(center) < 2 or len(size) < 2:
                continue
            potential_rooms.append(
                (
                    max(abs(float(size[0])) * abs(float(size[1])), 1e-6),
                    int(room_id),
                    float(center[0]),
                    float(center[1]),
                    0.5 * abs(float(size[0])),
                    0.5 * abs(float(size[1])),
                    str(attributes.get("source_portal_id") or ""),
                )
            )
        if not potential_rooms:
            return
        for candidate in candidates:
            if candidate.behavior_type != BEHAVIOR_EXPLORE:
                continue
            goal = list(candidate.goal_xyyaw or [])
            if len(goal) < 2:
                continue
            matches = [
                item
                for item in potential_rooms
                if abs(float(goal[0]) - item[2]) <= item[4] + 0.15
                and abs(float(goal[1]) - item[3]) <= item[5] + 0.15
            ]
            if not matches:
                continue
            _, room_id, _x, _y, _half_x, _half_y, portal_id = min(matches)
            candidate.metadata.update(
                {
                    "target_room_id": room_id,
                    "room_id": room_id,
                    "room_assignment_source": "portal_open_potential_child",
                    "potential_room": True,
                    "source_portal_id": portal_id,
                }
            )

    @staticmethod
    def _room_context_for_xy(
        graph: dict[str, Any], xy: tuple[float, float] | list[float] | None
    ) -> dict[str, Any]:
        """Return an occupancy/graph room label for a physical XY position.

        Potential rooms are useful door-side labels, but they must not steal a
        normal segmented room simply because their synthetic AABB overlaps the
        doorway.  Prefer an observed room when both contain the point; use a
        potential room only when it is the only available physical label.
        """

        values = list(xy or [])
        if len(values) < 2:
            return {}
        matches: list[tuple[int, float, str, Any, dict[str, Any]]] = []
        for node in graph.get("nodes") or []:
            if str(node.get("type") or "").casefold() != "room":
                continue
            attributes = node.get("attributes") or {}
            if not bool(attributes.get("active", True)):
                continue
            center = list(node.get("aabb_center") or node.get("centroid") or [])
            size = list(node.get("aabb_size") or [])
            room_id = node.get("room_id")
            if room_id is None:
                room_id = node.get("id")
            if len(center) < 2 or len(size) < 2 or room_id in (None, ""):
                continue
            half_x = 0.5 * abs(float(size[0]))
            half_y = 0.5 * abs(float(size[1]))
            if (
                abs(float(values[0]) - float(center[0])) > half_x + 1e-6
                or abs(float(values[1]) - float(center[1])) > half_y + 1e-6
            ):
                continue
            is_potential = bool(attributes.get("is_potential_room"))
            matches.append(
                (
                    1 if is_potential else 0,
                    max(half_x * half_y, 1e-6),
                    str(room_id),
                    room_id,
                    node,
                )
            )
        if not matches:
            return {}
        _potential_rank, _area, _sortable_room_id, room_id, node = min(matches)
        attributes = node.get("attributes") or {}
        result: dict[str, Any] = {
            "room_id": room_id,
            "potential_room": bool(attributes.get("is_potential_room")),
        }
        source_portal_id = str(attributes.get("source_portal_id") or "")
        if source_portal_id:
            result["source_portal_id"] = source_portal_id
        room_attribute = str(attributes.get("room_attribute") or "").strip()
        if room_attribute and room_attribute.casefold() != "unknown":
            result["room_attribute"] = room_attribute
            result["room_attribute_confidence"] = max(
                0.0,
                min(1.0, float(attributes.get("room_attribute_confidence", 0.0) or 0.0)),
            )
            scores = dict(attributes.get("room_attribute_scores") or {})
            if scores:
                result["room_attribute_scores"] = scores
        return result

    @staticmethod
    def _room_context_by_id(graph: dict[str, Any], room_id: Any) -> dict[str, Any]:
        if room_id in (None, ""):
            return {}
        normalized = str(room_id).removeprefix("room_")
        for node in graph.get("nodes") or []:
            if str(node.get("type") or "").casefold() != "room":
                continue
            node_room_id = node.get("room_id")
            if node_room_id is None:
                node_room_id = node.get("id")
            if node_room_id in (None, ""):
                continue
            if str(node_room_id).removeprefix("room_") != normalized:
                continue
            attributes = node.get("attributes") or {}
            result: dict[str, Any] = {
                "room_id": node_room_id,
                "potential_room": bool(attributes.get("is_potential_room")),
            }
            source_portal_id = str(attributes.get("source_portal_id") or "")
            if source_portal_id:
                result["source_portal_id"] = source_portal_id
            room_attribute = str(attributes.get("room_attribute") or "").strip()
            if room_attribute and room_attribute.casefold() != "unknown":
                result["room_attribute"] = room_attribute
                result["room_attribute_confidence"] = max(
                    0.0,
                    min(
                        1.0,
                        float(attributes.get("room_attribute_confidence", 0.0) or 0.0),
                    ),
                )
                scores = dict(attributes.get("room_attribute_scores") or {})
                if scores:
                    result["room_attribute_scores"] = scores
            return result
        return {}

    @classmethod
    def _attach_frontier_room_context(
        cls,
        candidates: list[BehaviorCandidate],
        graph: dict[str, Any],
        robot_xy: tuple[float, float] | None,
    ) -> None:
        """Attach explicit current/target room metadata to every frontier.

        ExplorePy proposals are geometric and normally have no room identity.
        Assigning it here (after the portal-child association) prevents all
        frontiers from being treated as a generic ``NEW_ROOM_FRONTIER`` and
        keeps a door-created child room distinct from the room containing the
        robot.
        """

        robot_context = cls._room_context_for_xy(graph, robot_xy)
        robot_room_id = robot_context.get("room_id")
        for candidate in candidates:
            if candidate.behavior_type != BEHAVIOR_EXPLORE:
                continue
            metadata = candidate.metadata
            target_room_id = metadata.get("target_room_id") or metadata.get("room_id")
            target_context = cls._room_context_by_id(graph, target_room_id)
            assignment_source = str(metadata.get("room_assignment_source") or "")
            if not target_context:
                point = list(metadata.get("frontier_point") or candidate.goal_xyyaw or [])
                target_context = cls._room_context_for_xy(graph, point)
                if target_context:
                    assignment_source = "occupancy_room_aabb"
            if robot_room_id not in (None, ""):
                metadata["robot_room_id"] = robot_room_id
                metadata["current_room_id"] = robot_room_id
            if target_context:
                target_room_id = target_context["room_id"]
                metadata["target_room_id"] = target_room_id
                metadata["room_id"] = target_room_id
                metadata["potential_room"] = bool(target_context.get("potential_room"))
                for key in (
                    "source_portal_id",
                    "room_attribute",
                    "room_attribute_confidence",
                    "room_attribute_scores",
                ):
                    if key in target_context:
                        metadata[key] = target_context[key]
                if assignment_source:
                    metadata["room_assignment_source"] = assignment_source
                if robot_room_id not in (None, ""):
                    metadata["room_relation"] = (
                        "current_room"
                        if str(target_room_id).removeprefix("room_")
                        == str(robot_room_id).removeprefix("room_")
                        else "other_room"
                    )
            # A frontier emitted by ExplorePy already passed a local geometry
            # feasibility test.  Do not turn an absent topology edge into a
            # false hard constraint; explicit false remains respected.
            metadata.setdefault("room_reachable", True)

    @staticmethod
    def _frontier_priority_key(candidate: BehaviorCandidate) -> tuple[float, float, float, float, str]:
        metadata = candidate.metadata or {}
        return (
            -float(metadata.get("expected_visible_unknown_area_m2", 0.0) or 0.0),
            -float(metadata.get("unknown_component_area_m2", 0.0) or 0.0),
            -float(candidate.features.get("exploration_gain", 0.0) or 0.0),
            max(0.0, float(candidate.features.get("distance_m", 0.0) or 0.0)),
            candidate.candidate_id,
        )

    def _limit_frontier_candidates(
        self, candidates: list[BehaviorCandidate]
    ) -> list[BehaviorCandidate]:
        """Apply the frontier cap without dropping every door-side room.

        The raw proposal stream is globally ranked by visible area.  Reserve
        one feasible proposal per distinct other-room label before filling the
        remaining slots, so a small frontier just beyond a door can reach the
        curator and the MLLM instead of being erased by same-room area alone.
        """

        limit = max(0, int(self.config.max_frontier_candidates))
        frontiers = [
            candidate
            for candidate in candidates
            if candidate.behavior_type == BEHAVIOR_EXPLORE
        ]
        if limit <= 0:
            return [candidate for candidate in candidates if candidate not in frontiers]
        if len(frontiers) <= limit:
            return candidates
        ordered = sorted(frontiers, key=self._frontier_priority_key)
        reserved_by_room: dict[str, BehaviorCandidate] = {}
        for candidate in ordered:
            metadata = candidate.metadata or {}
            room_id = str(
                metadata.get("target_room_id") or metadata.get("room_id") or ""
            )
            robot_room_id = str(metadata.get("robot_room_id") or "")
            if not room_id or metadata.get("room_reachable") is False:
                continue
            if robot_room_id and room_id.removeprefix("room_") == robot_room_id.removeprefix("room_"):
                continue
            reserved_by_room.setdefault(room_id, candidate)
        reserved = sorted(reserved_by_room.values(), key=self._frontier_priority_key)[:limit]
        selected_ids = {candidate.candidate_id for candidate in reserved}
        for candidate in ordered:
            if len(selected_ids) >= limit:
                break
            selected_ids.add(candidate.candidate_id)
        return [
            candidate
            for candidate in candidates
            if candidate.behavior_type != BEHAVIOR_EXPLORE
            or candidate.candidate_id in selected_ids
        ]

    def _interaction_candidates(
        self,
        graph: dict[str, Any],
        robot_xy: tuple[float, float],
        target_context: dict[str, Any],
    ) -> list[BehaviorCandidate]:
        candidates = []
        allowed_types = set(self.config.interaction_types)
        robot_room_id = self._room_id_for_xy(graph, robot_xy)
        for node in graph.get("nodes") or []:
            node_type = str(node.get("type") or "")
            if node_type not in allowed_types:
                continue
            interaction = node.get("interaction") or {}
            node_state = str(interaction.get("state") or "unknown")
            attributes = node.get("attributes") or {}
            visual_unknown_portal_reobserve = bool(
                node_type == "portal"
                and not self.config.portal_unknown_default_interact
                and self._is_ready_visual_unknown_portal(
                    node,
                    interaction,
                    attributes,
                    node_state,
                )
            )
            if not bool(interaction.get("is_interactable", False)) and not (
                visual_unknown_portal_reobserve
            ):
                continue
            if node_type == "container" and not self._is_openable_container(node):
                continue
            confidence = float(
                interaction.get("state_confidence", interaction.get("confidence", node.get("confidence", 0.0)))
                or 0.0
            )
            unknown_portal_rule_fallback = bool(
                node_type == "portal"
                and node_state.casefold() == "unknown"
                and self.config.portal_unknown_default_interact
                and bool(interaction.get("is_interactable", False))
                and bool(interaction.get("requires_interaction", False))
            )
            if (
                confidence < self.config.min_state_confidence
                and not unknown_portal_rule_fallback
                and not visual_unknown_portal_reobserve
            ):
                continue
            state_age_sec = max(0.0, float(node.get("state_age_sec", 0.0) or 0.0))
            if state_age_sec > self.config.max_state_age_sec:
                continue
            if node_type == "portal":
                if (
                    (
                        not bool(interaction.get("requires_interaction"))
                        and not visual_unknown_portal_reobserve
                    )
                    or node_state not in {"closed", "ajar", "unknown"}
                ):
                    continue
                if self.config.portal_require_attribute_ready:
                    attribute_status = str(
                        attributes.get("attribute_status")
                        or interaction.get("attribute_status")
                        or ""
                    ).strip().lower()
                    if attribute_status != "ready" and not self._has_verified_pending_portal_state(
                        interaction,
                        attributes,
                        node_state,
                        confidence,
                        state_age_sec,
                    ):
                        continue
                if (
                    node_state == "unknown"
                    and not self.config.portal_allow_unknown_state
                    and not visual_unknown_portal_reobserve
                ):
                    continue
                if self.config.portal_require_current_visibility:
                    portal_visible_pixels = int(
                        attributes.get("visible_pixels", 0) or 0
                    )
                    portal_visible_fraction = float(
                        attributes.get("visible_fraction", 0.0) or 0.0
                    )
                    if (
                        not bool(node.get("is_currently_visible"))
                        or portal_visible_pixels
                        < int(self.config.portal_min_visible_pixels)
                        or portal_visible_fraction
                        < float(self.config.portal_min_visible_fraction)
                    ):
                        continue
                if (
                    node_state == "unknown"
                    and not unknown_portal_rule_fallback
                    and not visual_unknown_portal_reobserve
                ):
                    # A graph-default/restricted-GT unknown must not become a
                    # model-lane physical action.  The only model-lane unknown
                    # admitted here is an explicit, current Module-1 visual
                    # observation, and it is marked for re-observation below.
                    continue
            if self.config.require_current_visibility and not bool(node.get("is_currently_visible")):
                continue
            node_room_id = node.get("room_id")
            allow_connected_room = bool(self.config.container_allow_connected_room)
            room_hops = self._room_hops(graph, robot_room_id, node_room_id)
            # A remembered container can remain in the graph after it leaves
            # the current camera view.  Do not turn that stale observation
            # into a physical INTERACT request when the observed room graph
            # also has no reachable route to it.  This is deliberately a
            # conjunction: a visible container or a reachable remembered one
            # may still be useful for planning.
            if (
                node_type == "container"
                and node.get("is_currently_visible") is False
                and room_hops is None
            ):
                continue
            if (
                node_type == "container"
                and self.config.container_require_same_room
                and node_room_id is not None
                and robot_room_id is not None
                and int(node_room_id) != int(robot_room_id)
                and not (allow_connected_room and room_hops is not None)
            ):
                continue
            # Container staging/physical rings are defined against the current
            # visible AABB.  ``centroid`` can be a retained semantic/object
            # point and may drift from that box; mixing the two made a
            # ``current_view_safe_outer`` pose face the centroid while its
            # clearance was computed from the AABB.  Use one geometry anchor
            # for both the AABB-normal approach axis and the AABB surface
            # offset.  This remains only a visual re-observation pose when M1
            # lacks a front axis; it does not consume an oracle/front-direction
            # field.
            position = self._node_xy(
                node,
                prefer_aabb=(node_type == "container"),
            )
            if position is None:
                continue
            object_distance = math.hypot(position[0] - robot_xy[0], position[1] - robot_xy[1])
            is_drawer_container = bool(
                node_type == "container" and self._is_drawer_container(node)
            )
            is_refrigerator_container = bool(
                node_type == "container" and self._is_refrigerator_container(node)
            )
            container_pre_action = bool(
                node_type == "container" and self.config.container_pre_action_mllm
            )
            drawer_pre_action = bool(
                is_drawer_container and self.config.drawer_pre_action_mllm
            )
            if node_type == "portal":
                standoff = self.config.portal_standoff_m
                standoff_source = "portal"
            elif is_drawer_container:
                standoff = (
                    self.config.drawer_standoff_m
                    if self.config.drawer_standoff_m is not None
                    else self.config.container_standoff_m
                )
                standoff_source = "drawer"
            elif is_refrigerator_container:
                standoff = (
                    self.config.fridge_standoff_m
                    if self.config.fridge_standoff_m is not None
                    else self.config.container_standoff_m
                )
                standoff_source = "refrigerator"
            else:
                standoff = self.config.container_standoff_m
                standoff_source = "container"
            visual_container_axis = (
                self._mllm_current_view_approach_axis(node, robot_xy, position)
                if node_type == "container"
                else None
            )
            node_id = str(node.get("id") or "")
            source_object_name = str(
                attributes.get("source_object_name") or node.get("name") or node_id
            )
            connected_room_ids = list(attributes.get("connected_room_ids") or [])
            expected_effect = str(
                interaction.get("expected_effect")
                or ("unlock_connectivity" if node_type == "portal" else "reveal_contents")
            )
            exploration_gain = 1.0 if node_type == "portal" else 0.65
            if node_type == "portal" and len(connected_room_ids) < 2:
                exploration_gain = 1.10

            explicit_target_reinteraction = bool(
                target_context.get("enabled")
                and self._matches_interaction_target(node, target_context)
            )
            if node_type == "container" and node_state in {"open", "static_open"}:
                continue
            object_id = str(
                attributes.get("instance_id") or source_object_name or node_id
            )
            legacy_interaction_standoff = self._nonnegative_clearance(
                standoff + self.config.interaction_safety_margin_m
            )
            container_kind = (
                "drawer"
                if is_drawer_container
                else "refrigerator"
                if is_refrigerator_container
                else "container"
            )
            physical_action_standoff = legacy_interaction_standoff
            physical_action_standoff_source = "legacy_type_standoff_plus_safety_margin"
            m1_capture_standoff = physical_action_standoff
            m1_capture_standoff_source = "not_applicable"
            m1_capture_standoff_validation = "not_applicable"
            container_two_stage_requested = bool(
                node_type == "container" and (container_pre_action or drawer_pre_action)
            )
            container_staging_ready_distance_m = min(
                self.config.interaction_ready_distance_m,
                max(
                    0.05,
                    self.config.container_safe_staging_arrival_tolerance_m,
                ),
            )
            container_physical_action_ready_distance_m = min(
                self.config.interaction_ready_distance_m,
                max(0.05, self.config.container_interaction_ready_distance_m),
            )
            navigation_anchor_outer_offset_m = 0.0
            navigation_anchor_ring_count = 1
            navigation_anchor_tangent_offset_m = 0.0
            navigation_anchor_source = "not_applicable"
            if node_type == "container":
                (
                    physical_action_standoff,
                    physical_action_standoff_source,
                    m1_capture_standoff,
                    m1_capture_standoff_source,
                    m1_capture_standoff_validation,
                ) = self._container_phase_standoffs(
                    container_kind=container_kind,
                    legacy_action_standoff_m=legacy_interaction_standoff,
                )
                if container_two_stage_requested:
                    (
                        navigation_anchor_outer_offset_m,
                        navigation_anchor_ring_count,
                        navigation_anchor_tangent_offset_m,
                        navigation_anchor_source,
                    ) = self._navigation_anchor_settings()
            shared_container_anchor_pose = bool(
                container_two_stage_requested
                and self.config.container_anchor_shared_pose_enabled
            )
            container_m1_face_selection = bool(
                container_two_stage_requested
                and self.config.container_m1_face_selection_enabled
            )
            if container_m1_face_selection:
                # A graph-level attribute refresh may already have classified
                # the current image.  It must not preselect the active
                # decision's face: targeted M1 at one of the four explicit
                # AABB faces owns that decision.
                visual_container_axis = None
            if shared_container_anchor_pose:
                # There is one arrival contract for navigation, observation and
                # action.  The physical-ready envelope is the conservative one;
                # no wider outer-staging tolerance may authorize this pose.
                container_staging_ready_distance_m = (
                    container_physical_action_ready_distance_m
                )
                navigation_anchor_outer_offset_m = 0.0
                navigation_anchor_ring_count = 1
                navigation_anchor_tangent_offset_m = 0.0
                navigation_anchor_source = "shared_navigation_m1_action_anchor"
            # Store only the object reference geometry that was already used to
            # generate the candidate ring.  A later M1-positive capture can use
            # this to project its confirmed camera-side ray back to a safe
            # AABB-surface action pose; it must not read a graph/oracle front
            # axis for that purpose.
            container_geometry_anchor_xy: list[float] = []
            container_geometry_aabb_size_xy: list[float] = []
            if node_type == "container":
                try:
                    container_geometry_anchor_xy = [
                        float(position[0]),
                        float(position[1]),
                    ]
                except (TypeError, ValueError, IndexError):
                    container_geometry_anchor_xy = []
                raw_container_size = list(node.get("aabb_size") or [])
                if len(raw_container_size) >= 2:
                    try:
                        container_geometry_aabb_size_xy = [
                            abs(float(raw_container_size[0])),
                            abs(float(raw_container_size[1])),
                        ]
                    except (TypeError, ValueError):
                        container_geometry_aabb_size_xy = []
            goal_candidates, approach_pose_labels = self._approach_candidates(
                robot_xy,
                position,
                node,
                (
                    physical_action_standoff
                    if shared_container_anchor_pose
                    else m1_capture_standoff
                    if container_two_stage_requested
                    else physical_action_standoff
                ),
                node_type,
                visual_container_axis=visual_container_axis,
                navigation_anchor_outer_offset_m=(
                    0.0
                    if shared_container_anchor_pose
                    else navigation_anchor_outer_offset_m
                    if container_two_stage_requested
                    else 0.0
                ),
                navigation_anchor_ring_count=navigation_anchor_ring_count,
                navigation_anchor_tangent_offset_m=navigation_anchor_tangent_offset_m,
                navigation_anchor_aabb_fan_clearances_m=(
                    self.config.drawer_navigation_anchor_fan_clearances_m
                    if is_drawer_container
                    and self.config.drawer_navigation_anchor_aabb_fan_enabled
                    else self.config.fridge_navigation_anchor_fan_clearances_m
                    if is_refrigerator_container
                    and self.config.fridge_navigation_anchor_aabb_fan_enabled
                    else ()
                ),
                navigation_anchor_aabb_fan_angles_deg=(
                    self.config.drawer_navigation_anchor_fan_angles_deg
                    if is_drawer_container
                    and self.config.drawer_navigation_anchor_aabb_fan_enabled
                    else self.config.fridge_navigation_anchor_fan_angles_deg
                    if is_refrigerator_container
                    and self.config.fridge_navigation_anchor_aabb_fan_enabled
                    else ()
                ),
                container_multiview_angular_scale=(
                    self.config.fridge_multiview_angular_scale
                    if is_refrigerator_container
                    else 1.0
                ),
                container_m1_face_selection_enabled=(
                    container_m1_face_selection
                ),
            )
            container_face_axes_by_staging = [
                list(axis) if axis is not None else []
                for axis in (
                    self._container_face_axis_from_label(label)
                    for label in approach_pose_labels
                )
            ]
            container_anchor_robot_distances_m = [
                math.hypot(float(goal[0]) - robot_xy[0], float(goal[1]) - robot_xy[1])
                for goal in goal_candidates
            ]
            # Keep the visual staging geometry and the physical action
            # geometry as two explicit, index-aligned lists.  The outer pose is
            # generated without a semantic front claim; its radial axis is
            # merely carried inward to the type-specific action standoff.  M1
            # later authorizes the action from the image captured at that exact
            # direct capture pose, but it never supplies (or receives) this
            # mapping.  The outer goal is a navigation anchor only: M1 is
            # deliberately never queried until the paired capture pose arrives.
            container_two_stage_mapping_ready = False
            container_m1_capture_goals_by_staging: list[list[float]] = []
            container_m1_capture_labels_by_staging: list[str] = []
            container_action_goals_by_staging: list[list[float]] = []
            container_action_labels_by_staging: list[str] = []
            container_staging_source_index_by_index: list[int] = list(
                range(len(goal_candidates))
            )
            if container_two_stage_requested:
                container_staging_source_index_by_index = (
                    self._container_staging_source_indices(approach_pose_labels)
                )
                if shared_container_anchor_pose:
                    container_m1_capture_goals_by_staging = [
                        list(goal) for goal in goal_candidates
                    ]
                    container_m1_capture_labels_by_staging = [
                        f"{label}_shared_m1_capture"
                        for label in approach_pose_labels
                    ]
                    container_action_goals_by_staging = [
                        list(goal) for goal in goal_candidates
                    ]
                    container_action_labels_by_staging = [
                        f"{label}_shared_physical_action"
                        for label in approach_pose_labels
                    ]
                    container_action_goal_options_by_staging = [
                        [list(goal)] for goal in goal_candidates
                    ]
                    container_action_option_labels_by_staging = [
                        [container_action_labels_by_staging[index]]
                        for index in range(len(goal_candidates))
                    ]
                else:
                    (
                        container_m1_capture_goals_by_staging,
                        container_m1_capture_labels_by_staging,
                    ) = self._container_m1_capture_goals_for_staging(
                        target_xy=position,
                        staging_goals=goal_candidates,
                        staging_labels=approach_pose_labels,
                        capture_standoff_m=m1_capture_standoff,
                        node=node,
                        staging_source_indices=container_staging_source_index_by_index,
                    )
                    (
                        container_action_goals_by_staging,
                        container_action_labels_by_staging,
                        container_action_goal_options_by_staging,
                        container_action_option_labels_by_staging,
                    ) = self._container_action_goals_for_staging(
                        target_xy=position,
                        staging_goals=goal_candidates,
                        staging_labels=approach_pose_labels,
                        physical_standoff_m=physical_action_standoff,
                        lateral_offset_m=self.config.container_action_lateral_offset_m,
                        node=node,
                        staging_source_indices=container_staging_source_index_by_index,
                    )
                container_two_stage_mapping_ready = bool(
                    container_m1_capture_goals_by_staging
                    and len(container_m1_capture_goals_by_staging)
                    == len(goal_candidates)
                    and
                    container_action_goals_by_staging
                    and len(container_action_goals_by_staging) == len(goal_candidates)
                    and len(container_action_goal_options_by_staging)
                    == len(goal_candidates)
                    and all(container_action_goal_options_by_staging)
                )
            else:
                container_action_goal_options_by_staging = []
                container_action_option_labels_by_staging = []
            approach = goal_candidates[0]
            portal_aperture_observation = (
                self._portal_aperture_observation(node)
                if node_type == "portal"
                else None
            )
            portal_aabb_center_xy: list[float] = []
            portal_aabb_size_xy: list[float] = []
            portal_clearance_aabb_center_xy: list[float] = []
            portal_clearance_aabb_size_xy: list[float] = []
            if node_type == "portal":
                reference_center = list(
                    attributes.get("interaction_reference_aabb_center")
                    or node.get("aabb_center")
                    or node.get("centroid")
                    or []
                )
                reference_size = list(
                    attributes.get("interaction_reference_aabb_size")
                    or node.get("aabb_size")
                    or []
                )
                clearance_center = list(
                    node.get("aabb_center") or reference_center or []
                )
                clearance_size = list(node.get("aabb_size") or reference_size or [])
                if len(reference_center) >= 2:
                    try:
                        portal_aabb_center_xy = [
                            float(reference_center[0]),
                            float(reference_center[1]),
                        ]
                    except (TypeError, ValueError):
                        portal_aabb_center_xy = []
                if len(reference_size) >= 2:
                    try:
                        portal_aabb_size_xy = [
                            abs(float(reference_size[0])),
                            abs(float(reference_size[1])),
                        ]
                    except (TypeError, ValueError):
                        portal_aabb_size_xy = []
                if len(clearance_center) >= 2:
                    try:
                        portal_clearance_aabb_center_xy = [
                            float(clearance_center[0]),
                            float(clearance_center[1]),
                        ]
                    except (TypeError, ValueError):
                        portal_clearance_aabb_center_xy = []
                if len(clearance_size) >= 2:
                    try:
                        portal_clearance_aabb_size_xy = [
                            abs(float(clearance_size[0])),
                            abs(float(clearance_size[1])),
                        ]
                    except (TypeError, ValueError):
                        portal_clearance_aabb_size_xy = []
            approach_distance = math.hypot(
                approach[0] - robot_xy[0], approach[1] - robot_xy[1]
            )
            interaction_command = {
                "node_id": node_id,
                "object_id": object_id,
                "action": "open",
                "interaction_mode": str(
                    interaction.get("interaction_mode") or "open_close"
                ),
                "expected_state": "open",
                "interaction_approach_pose_xyyaw": list(approach),
                # An interaction candidate must not turn a scene-specific
                # geometry/oracle axis into a physical precondition.  The only
                # declared container front accepted here comes from a current,
                # ready M1 view; otherwise the executor receives a bounded
                # observation ring below.
                "interaction_approach_axis_xy": list(visual_container_axis or []),
                "interaction_approach_pose_labels": list(approach_pose_labels),
                "interaction_ready_distance_m": (
                    container_staging_ready_distance_m
                    if container_two_stage_requested
                    else self.config.interaction_ready_distance_m
                ),
                "interaction_ready_yaw_tolerance_rad": max(
                    0.05, float(self.config.interaction_ready_yaw_tolerance_rad)
                ),
                # Move-base has no per-goal tolerance fields in MoveBaseGoal.
                # The executor consumes this explicit pair, applies it to DWA
                # before dispatch, and uses the same pair for arrival checks.
                "navigation_goal_position_tolerance_m": (
                    container_staging_ready_distance_m
                    if container_two_stage_requested
                    else self.config.interaction_ready_distance_m
                ),
                "navigation_goal_yaw_tolerance_rad": max(
                    0.05, float(self.config.interaction_ready_yaw_tolerance_rad)
                ),
                "navigation_goal_tolerance_contract_explicit": True,
            }
            if node_type == "container":
                interaction_command.update(
                    {
                        "container_staging_ready_distance_m": (
                            container_staging_ready_distance_m
                            if container_two_stage_requested
                            else 0.0
                        ),
                        "container_physical_action_ready_distance_m": (
                            container_physical_action_ready_distance_m
                            if container_two_stage_requested
                            else 0.0
                        ),
                        # M1 capture is a direct, action-clearance observation
                        # pose; only the preceding navigation anchor uses the
                        # wider outer staging tolerance.
                        "container_m1_capture_ready_distance_m": (
                            container_physical_action_ready_distance_m
                            if container_two_stage_requested
                            else 0.0
                        ),
                        "container_kind": container_kind,
                    }
                )
            if portal_aperture_observation is not None:
                interaction_command["portal_aperture_observation"] = dict(
                    portal_aperture_observation
                )
            candidates.append(
                BehaviorCandidate(
                    candidate_id=f"interaction:{node_id}:open",
                    behavior_type=BEHAVIOR_INTERACT,
                    source="unified_graph",
                    target_id=node_id,
                    target_name=source_object_name,
                    goal_xyyaw=approach,
                    interaction_command=interaction_command,
                    features={
                        "exploration_gain": exploration_gain,
                        "visibility_gain": 1.0 if node_type == "portal" else 0.80,
                        "semantic_gain": 1.0 if node_type == "portal" else 0.75,
                        "target_relevance": 1.0 if explicit_target_reinteraction else 0.0,
                        "distance_m": approach_distance,
                        "interaction_cost": float(
                            interaction.get(
                                "interaction_cost", interaction.get("cost", 1.0)
                            )
                            or 1.0
                        ),
                        "state_age_ratio": min(
                            1.0,
                            state_age_sec / max(self.config.max_state_age_sec, 1e-6),
                        ),
                        "confidence": confidence,
                        "priority": 1.0 if node_type == "portal" else 0.75,
                    },
                    metadata={
                        "node_type": node_type,
                        "semantic_name": str(
                            attributes.get("semantic_name")
                            or attributes.get("category")
                            or node.get("label")
                            or node.get("name")
                            or node_type
                        ),
                        "state": node_state,
                        "expected_effect": expected_effect,
                        "connected_room_ids": connected_room_ids,
                        "connectivity_status": attributes.get(
                            "connectivity_status", "unknown"
                        ),
                        "is_currently_visible": bool(node.get("is_currently_visible")),
                        "state_age_sec": state_age_sec,
                        "object_distance_m": object_distance,
                        "robot_room_id": robot_room_id,
                        "target_room_id": node_room_id,
                        "room_transition_required": room_hops not in {None, 0},
                        "room_hops": room_hops,
                        "room_reachable": room_hops is not None,
                        # Keep the legacy names pointed at the physical action
                        # clearance.  Previously ``interaction_standoff_m``
                        # meant the implicit M1 observation base for two-stage
                        # containers, which made traces report a distance that
                        # neither described the outer navigation anchor nor the
                        # bridge target.
                        "interaction_standoff_m": physical_action_standoff,
                        "configured_interaction_standoff_m": physical_action_standoff,
                        "interaction_standoff_source": standoff_source,
                        "interaction_safety_margin_m": self.config.interaction_safety_margin_m,
                        "container_physical_action_standoff_m": (
                            physical_action_standoff
                            if node_type == "container"
                            else 0.0
                        ),
                        "container_physical_action_standoff_source": (
                            physical_action_standoff_source
                            if node_type == "container"
                            else ""
                        ),
                        "container_action_lateral_offset_m": (
                            0.0
                            if shared_container_anchor_pose
                            else max(0.0, float(self.config.container_action_lateral_offset_m))
                            if container_two_stage_requested
                            else 0.0
                        ),
                        # Immutable object-reference geometry for the active
                        # decision.  It carries no semantic/front-axis claim;
                        # only an accepted M1 image at a calibrated capture
                        # pose may turn its object-to-camera ray into an action
                        # face.
                        "container_geometry_anchor_xy": (
                            list(container_geometry_anchor_xy)
                            if container_two_stage_requested
                            else []
                        ),
                        "container_geometry_aabb_size_xy": (
                            list(container_geometry_aabb_size_xy)
                            if container_two_stage_requested
                            else []
                        ),
                        "container_m1_front_axis_from_capture": bool(
                            container_two_stage_requested
                            and (
                                not shared_container_anchor_pose
                                or container_m1_face_selection
                            )
                            and self.config.container_m1_front_axis_from_capture
                        ),
                        "container_anchor_shared_pose": shared_container_anchor_pose,
                        "container_m1_face_selection_enabled": bool(
                            container_m1_face_selection
                        ),
                        "container_m1_face_selection_contract": (
                            "aabb_four_faces_m1_authorized"
                            if container_m1_face_selection
                            else "legacy_robot_side_or_mllm_view"
                        ),
                        "container_face_axis_xy_by_staging_index": (
                            container_face_axes_by_staging
                            if container_two_stage_requested
                            else []
                        ),
                        "container_anchor_robot_distance_m_by_staging_index": (
                            container_anchor_robot_distances_m
                            if container_two_stage_requested
                            else []
                        ),
                        "container_kind": (
                            "drawer"
                            if is_drawer_container
                            else "refrigerator"
                            if is_refrigerator_container
                            else "container"
                            if node_type == "container"
                            else ""
                        ),
                        "m1_observation_staging_required": bool(
                            node_type == "container"
                            and (container_pre_action or drawer_pre_action)
                        ),
                        # ``m1_observation_standoff_m`` remains a readable
                        # compatibility alias.  The canonical fields below
                        # make it explicit that it is a direct capture point,
                        # not an outer navigation ring.
                        "m1_observation_standoff_m": (
                            m1_capture_standoff
                            if node_type == "container"
                            and (container_pre_action or drawer_pre_action)
                            else 0.0
                        ),
                        "container_m1_capture_standoff_m": (
                            m1_capture_standoff
                            if container_two_stage_requested
                            else 0.0
                        ),
                        "container_m1_capture_standoff_source": (
                            m1_capture_standoff_source
                            if container_two_stage_requested
                            else ""
                        ),
                        "container_m1_capture_standoff_validation": (
                            m1_capture_standoff_validation
                            if container_two_stage_requested
                            else "not_applicable"
                        ),
                        "container_navigation_anchor_outer_offset_m": (
                            navigation_anchor_outer_offset_m
                            if container_two_stage_requested
                            else 0.0
                        ),
                        "container_navigation_anchor_ring_count": (
                            navigation_anchor_ring_count
                            if container_two_stage_requested
                            else 0
                        ),
                        "container_navigation_anchor_tangent_offset_m": (
                            navigation_anchor_tangent_offset_m
                            if container_two_stage_requested
                            else 0.0
                        ),
                        "container_navigation_anchor_source": (
                            navigation_anchor_source
                            if container_two_stage_requested
                            else ""
                        ),
                        # Deprecated alias used by existing executor versions
                        # while they migrate from staging observations to
                        # navigation anchors.
                        "m1_safe_staging_outer_offset_m": (
                            navigation_anchor_outer_offset_m
                            if container_two_stage_requested
                            else 0.0
                        ),
                        "m1_safe_staging_arrival_tolerance_m": (
                            container_physical_action_ready_distance_m
                            if shared_container_anchor_pose
                            else min(
                                self.config.interaction_ready_distance_m,
                                max(
                                    self.config.container_interaction_ready_distance_m,
                                    self.config.container_safe_staging_arrival_tolerance_m,
                                ),
                            )
                            if node_type == "container"
                            and (container_pre_action or drawer_pre_action)
                            else 0.0
                        ),
                        "target_enabled": bool(target_context.get("enabled")),
                        "target_match": explicit_target_reinteraction,
                        "requires_approach": True,
                        "approach_strategy": (
                            "portal_aabb_normal"
                            if node_type == "portal"
                            else (
                                "container_mllm_current_view"
                                if visual_container_axis is not None
                                else "container_multiview_reobserve"
                                if self.config.container_multiview_enabled
                                else "radial_standoff"
                            )
                        ),
                        "interaction_approach_axis_xy": list(
                            visual_container_axis or []
                        ),
                        "goal_xyyaw_candidates": goal_candidates,
                        "interaction_approach_pose_labels": approach_pose_labels,
                        "interaction_multiview_reobserve": bool(
                            node_type == "container"
                            and len(goal_candidates) > 1
                            and visual_container_axis is None
                        ),
                        # Keep the candidate ID/action stable so M2 history
                        # and cooldowns refer to the same portal.  The state
                        # machine treats this as a navigation-to-observe phase,
                        # not an authorization to publish a physical ``open``.
                        "observation_required": bool(
                            visual_unknown_portal_reobserve
                            or container_pre_action
                            or drawer_pre_action
                        ),
                        "reobserve": bool(
                            visual_unknown_portal_reobserve
                            or container_pre_action
                            or drawer_pre_action
                        ),
                        "observation_reason": (
                            "mllm_portal_state_unknown"
                            if visual_unknown_portal_reobserve
                            else "mllm_drawer_pre_action_visual"
                            if drawer_pre_action
                            else "mllm_container_pre_action_visual"
                            if container_pre_action
                            else ""
                        ),
                        "interaction_observation_max_attempts": (
                            min(
                                len(goal_candidates),
                                max(
                                    1,
                                    int(
                                        self.config.container_two_stage_observation_max_attempts
                                    ),
                                ),
                            )
                            if container_two_stage_requested
                            else max(
                                1,
                                int(
                                    self.config.drawer_pre_action_observation_max_attempts
                                    if drawer_pre_action
                                    else self.config.container_pre_action_observation_max_attempts
                                    if container_pre_action
                                    else self.config.portal_unknown_observation_max_attempts
                                ),
                            )
                        ),
                        "container_two_stage_observation_max_attempts": (
                            min(
                                len(goal_candidates),
                                max(
                                    1,
                                    int(
                                        self.config.container_two_stage_observation_max_attempts
                                    ),
                                ),
                            )
                            if container_two_stage_requested
                            else 0
                        ),
                        # Evidence budgets are intentionally not the outer
                        # anchor count.  Every viewpoint earns up to the given
                        # number of independently fresh M1 frames before a
                        # negative image advances to another capture pose.
                        "interaction_observation_same_pose_samples_per_view": (
                            max(
                                1,
                                int(
                                    self.config.container_m1_same_pose_samples_per_view
                                ),
                            )
                            if container_two_stage_requested
                            else 1
                        ),
                        "interaction_observation_max_viewpoints": (
                            min(
                                len(goal_candidates),
                                max(
                                    1,
                                    int(self.config.container_m1_max_viewpoints),
                                ),
                            )
                            if container_two_stage_requested
                            else 0
                        ),
                        "interaction_observation_max_total_requests": (
                            min(
                                max(
                                    1,
                                    int(
                                        self.config.container_m1_max_total_requests
                                    ),
                                ),
                                len(goal_candidates)
                                * max(
                                    1,
                                    int(
                                        self.config.container_m1_same_pose_samples_per_view
                                    ),
                                ),
                            )
                            if container_two_stage_requested
                            else 0
                        ),
                        "interaction_observation_source": (
                            "mllm_attribute_inference"
                            if (
                                visual_unknown_portal_reobserve
                                or container_pre_action
                                or drawer_pre_action
                            )
                            else ""
                        ),
                        "drawer_pre_action_observation": drawer_pre_action,
                        "container_pre_action_observation": container_pre_action,
                        # A failed mapping is deliberately still marked as a
                        # two-stage request.  The state machine will exhaust
                        # outer observations fail-closed instead of reverting to
                        # a legacy direct open from the staging pose.
                        "container_two_stage_approach": container_two_stage_requested,
                        "container_two_stage_mapping_ready": (
                            container_two_stage_mapping_ready
                        ),
                        "container_two_stage_phase": (
                            "staging" if container_two_stage_requested else ""
                        ),
                        "container_staging_goal_xyyaw_candidates": [
                            list(goal) for goal in goal_candidates
                        ]
                        if container_two_stage_requested
                        else [],
                        "container_staging_pose_labels": list(approach_pose_labels)
                        if container_two_stage_requested
                        else [],
                        # Canonical names for the same legacy staging fields:
                        # these positions are navigation recovery anchors, not
                        # M1 observation locations.
                        "container_navigation_anchor_goal_xyyaw_candidates": [
                            list(goal) for goal in goal_candidates
                        ]
                        if container_two_stage_requested
                        else [],
                        "container_navigation_anchor_pose_labels": list(
                            approach_pose_labels
                        )
                        if container_two_stage_requested
                        else [],
                        "container_staging_source_index_by_index": list(
                            container_staging_source_index_by_index
                        )
                        if container_two_stage_requested
                        else [],
                        # M1 visual retries use a separate, face-diverse order
                        # from navigation's face-major recovery list.  This
                        # keeps the planner-friendly anchor ordering while a
                        # single oblique M1 image can promptly sample another
                        # face rather than spend its entire evidence budget at
                        # farther radii of the same face.
                        "container_m1_viewpoint_order": (
                            self._container_m1_viewpoint_order(approach_pose_labels)
                            if container_two_stage_requested
                            else []
                        ),
                        # Index i is a complete three-phase contract:
                        # navigation_anchor[i] -> m1_capture[i] ->
                        # physical_action[i].  Capture positions are direct
                        # configured clearances; the navigation-only outer
                        # offset is not part of their geometry.
                        "container_m1_capture_goal_xyyaw_by_staging_index": [
                            list(goal)
                            for goal in container_m1_capture_goals_by_staging
                        ],
                        "container_m1_capture_pose_labels_by_staging_index": list(
                            container_m1_capture_labels_by_staging
                        ),
                        "container_action_goal_xyyaw_by_staging_index": [
                            list(goal) for goal in container_action_goals_by_staging
                        ],
                        "container_action_pose_labels_by_staging_index": list(
                            container_action_labels_by_staging
                        ),
                        # The primary inner pose is retained above for backward
                        # compatible traces.  These are the only bounded
                        # within-face alternatives: same normal standoff, then
                        # a small left/right tangent.  They never authorize a
                        # new M1 request or a different container face.
                        "container_action_goal_xyyaw_options_by_staging_index": [
                            [list(option) for option in options]
                            for options in container_action_goal_options_by_staging
                        ],
                        "container_action_pose_option_labels_by_staging_index": [
                            list(labels)
                            for labels in container_action_option_labels_by_staging
                        ],
                        # These values restore exactly the pre-action flags when
                        # an inner physical approach fails and the next outer
                        # staging pose must earn a new M1 view.
                        "container_two_stage_staging_observation_required": bool(
                            container_pre_action or drawer_pre_action
                        ),
                        "container_two_stage_staging_container_pre_action_observation": (
                            container_pre_action
                        ),
                        "container_two_stage_staging_drawer_pre_action_observation": (
                            drawer_pre_action
                        ),
                        # Preserve the geometry used to construct the two-sided
                        # approach options.  The post-open continuation uses it
                        # to select an AABB-clear goal on the far side instead
                        # of recreating one radial point at the door center.
                        "portal_aabb_center_xy": portal_aabb_center_xy,
                        "portal_aabb_size_xy": portal_aabb_size_xy,
                        "portal_clearance_aabb_center_xy": (
                            portal_clearance_aabb_center_xy
                        ),
                        "portal_clearance_aabb_size_xy": (
                            portal_clearance_aabb_size_xy
                        ),
                        "portal_aperture_observation": (
                            dict(portal_aperture_observation)
                            if portal_aperture_observation is not None
                            else {}
                        ),
                    },

                )
            )
        return candidates

    @staticmethod
    def _is_ready_visual_unknown_portal(
        node: dict[str, Any],
        interaction: dict[str, Any],
        attributes: dict[str, Any],
        node_state: str,
    ) -> bool:
        """Whether Module 1, rather than GT/default graph state, saw unknown.

        This is intentionally stricter than a generic portal ``unknown``:
        model runs must never approach an unobserved restricted-GT doorway and
        then blindly issue an open command.  A ready visual M1 result on a
        currently visible object is useful, however—it grants a stable
        *re-observation* candidate with the normal interaction geometry.
        """

        if str(node_state or "").strip().casefold() != "unknown":
            return False
        if not bool(node.get("is_currently_visible")):
            return False
        if str(attributes.get("attribute_status") or "").strip().casefold() != "ready":
            return False
        sources = (
            attributes.get("attribute_source"),
            interaction.get("state_source"),
            (attributes.get("interaction_state_override") or {}).get("state_source"),
        )
        return any("mllm" in str(source or "").casefold() for source in sources)

    @staticmethod
    def _portal_aperture_observation(node: dict[str, Any]) -> dict[str, Any] | None:
        """Combine public visual morphology with independent room/OCC evidence.

        Module 1 supplies ``door_leaf`` from the current RGB observation.  The
        mapping side supplies connectivity only when two observed room
        endpoints are connected (and no synthetic/potential room is involved).
        Missing evidence is intentionally represented as ``unknown``; the
        force bridge then returns a terminal ``unavailable`` result.
        """

        attributes = node.get("attributes") or {}
        morphology = attributes.get("portal_morphology")
        if not isinstance(morphology, dict):
            return None
        leaf = str(
            morphology.get("door_leaf") or morphology.get("leaf") or "unknown"
        ).strip().casefold()
        if leaf in {"absent", "none", "missing", "no_leaf", "no_door"}:
            leaf = "absent"
        elif leaf in {"present", "leaf", "door", "door_leaf", "visible"}:
            leaf = "present"
        else:
            leaf = "unknown"
        try:
            morphology_confidence = max(
                0.0, min(1.0, float(morphology.get("confidence", 0.0)))
            )
        except (TypeError, ValueError):
            morphology_confidence = 0.0
        observed_room_ids = {
            str(room_id)
            for room_id in (attributes.get("observed_connected_room_ids") or [])
            if room_id is not None and str(room_id)
        }
        topology_connected = bool(
            node.get("is_currently_visible") is True
            and str(attributes.get("connectivity_status") or "").casefold()
            == "connected"
            and len(observed_room_ids) >= 2
            and not bool(attributes.get("potential_room_ids"))
        )
        return {
            "door_leaf": leaf,
            "connectivity": "open" if topology_connected else "unknown",
            "confidence": (
                min(morphology_confidence, 0.8)
                if topology_connected
                else morphology_confidence
            ),
        }

    def _has_verified_pending_portal_state(
        self,
        interaction: dict[str, Any],
        attributes: dict[str, Any],
        node_state: str,
        confidence: float,
        state_age_sec: float,
    ) -> bool:
        """Keep a verified closed portal actionable while its refresh is pending.

        Module 1 publishes ``pending`` before issuing a refresh request.  A
        portal that was already classified successfully would otherwise blink
        out of the candidate stream for the duration of that request.  The
        exception is deliberately narrow: default graph state and never-ready
        portals remain blocked by ``portal_require_attribute_ready``.
        """

        if str(attributes.get("attribute_status") or "").strip().lower() != "pending":
            return False
        if node_state not in {"closed", "ajar"}:
            return False
        if not bool(interaction.get("is_interactable")):
            return False
        if not bool(interaction.get("requires_interaction")):
            return False
        if confidence < self.config.min_state_confidence:
            return False
        if state_age_sec > self.config.max_state_age_sec:
            return False

        verified = attributes.get("interaction_state_override") or {}
        if not isinstance(verified, dict):
            return False
        verified_state = str(verified.get("state") or "").strip().lower()
        verified_source = str(
            verified.get("state_source") or attributes.get("attribute_source") or ""
        ).strip().lower()
        try:
            verified_confidence = float(
                verified.get(
                    "state_confidence",
                    attributes.get("attribute_confidence", 0.0),
                )
                or 0.0
            )
        except (TypeError, ValueError):
            return False
        return (
            verified_state in {"closed", "ajar"}
            and verified_source == "mllm_attribute_inference"
            and verified_confidence >= self.config.min_state_confidence
        )

    def _portal_traversal_candidates(
        self,
        graph: dict[str, Any],
        robot_xy: tuple[float, float],
    ) -> list[BehaviorCandidate]:
        """Create a one-shot goal beyond a portal opened from the current side."""

        if "portal" not in set(self.config.interaction_types):
            return []
        candidates: list[BehaviorCandidate] = []
        for node in graph.get("nodes") or []:
            if str(node.get("type") or "").casefold() != "portal":
                continue
            interaction = node.get("interaction") or {}
            state = str(interaction.get("state") or "unknown").casefold()
            if state != "open":
                continue
            # Do not treat a missing/unknown traversability bit as a valid
            # post-open route.  Only a confirmed backend result or an OCC/room
            # connectivity observation may create a traversal candidate.
            if interaction.get("traversable") is not True:
                continue
            history = list(interaction.get("operation_history") or [])
            open_event = next(
                (
                    event
                    for event in reversed(history)
                    if _is_confirmed_portal_open_history(interaction, event)
                ),
                None,
            )
            if open_event is None:
                continue
            approach = list(open_event.get("approach_goal_xyyaw") or [])
            attributes = node.get("attributes") or {}
            center = list(
                attributes.get("interaction_reference_aabb_center")
                or node.get("aabb_center")
                or node.get("centroid")
                or []
            )
            if len(center) < 2:
                continue
            center_x, center_y = float(center[0]), float(center[1])
            through_x = center_x - float(approach[0])
            through_y = center_y - float(approach[1])
            through_norm = math.hypot(through_x, through_y)
            if through_norm <= 1e-6:
                continue
            unit_x = through_x / through_norm
            unit_y = through_y / through_norm
            robot_to_center = math.hypot(
                float(robot_xy[0]) - center_x,
                float(robot_xy[1]) - center_y,
            )
            if robot_to_center > max(
                0.0, float(self.config.portal_traversal_max_start_distance_m)
            ):
                continue
            signed_progress = (
                (float(robot_xy[0]) - center_x) * unit_x
                + (float(robot_xy[1]) - center_y) * unit_y
            )
            if signed_progress >= max(
                0.0, float(self.config.portal_traversal_completion_margin_m)
            ):
                continue
            traversal_distance = max(
                0.0, float(self.config.portal_traversal_distance_m)
            )
            goal = [
                center_x + unit_x * traversal_distance,
                center_y + unit_y * traversal_distance,
                math.atan2(unit_y, unit_x),
            ]
            distance_m = math.hypot(
                goal[0] - float(robot_xy[0]), goal[1] - float(robot_xy[1])
            )
            node_id = str(node.get("id") or "")
            event_id = str(open_event.get("event_id") or "latest_open")
            source_object_name = str(
                attributes.get("source_object_name") or node.get("name") or node_id
            )
            potential_room_ids = list(attributes.get("potential_room_ids") or [])
            target_room_id = attributes.get("portal_child_room_id")
            if target_room_id is None and potential_room_ids:
                target_room_id = potential_room_ids[0]
            source_room_id = attributes.get("portal_child_source_room_id")
            candidates.append(
                BehaviorCandidate(
                    candidate_id=f"traverse:{node_id}:{event_id}",
                    behavior_type=BEHAVIOR_NAVIGATE,
                    source="post_interaction_portal",
                    target_id=node_id,
                    target_name=source_object_name,
                    goal_xyyaw=goal,
                    features={
                        "exploration_gain": 1.25,
                        "visibility_gain": 1.15,
                        "semantic_gain": 1.0,
                        "target_relevance": 0.0,
                        "distance_m": distance_m,
                        "interaction_cost": 0.0,
                        "state_age_ratio": 0.0,
                        "confidence": float(
                            interaction.get("state_confidence", 1.0) or 1.0
                        ),
                        "priority": 1.0,
                    },
                    metadata={
                        "node_type": "portal",
                        "semantic_name": str(
                            attributes.get("semantic_name")
                            or attributes.get("category")
                            or node.get("label")
                            or "door"
                        ),
                        "state": state,
                        "post_interaction_traversal": True,
                        "opened_portal_id": node_id,
                        "source_interaction_event_id": event_id,
                        "connected_room_ids": list(
                            attributes.get("connected_room_ids") or []
                        ),
                        "target_room_id": target_room_id,
                        "room_id": target_room_id,
                        "source_room_id": source_room_id,
                        "potential_room": target_room_id is not None,
                        "room_transition_required": True,
                        "requires_approach": False,
                        "verify_target_visibility": False,
                        "goal_xyyaw_candidates": [goal],
                    },
                )
            )
        return candidates

    @staticmethod
    def _mllm_current_view_approach_axis(
        node: dict[str, Any],
        robot_xy: tuple[float, float],
        target_xy: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Return a container approach axis only from a fresh M1 view contract.

        ``interaction_approach_axis_xy`` and
        ``interaction_approach_pose_xyyaw`` are intentionally excluded here:
        they may be supplied by scene geometry or an oracle.  M1 instead says
        whether the *current* camera view is front/oblique enough to approach;
        the actual axis is then snapped to the visible AABB face direction
        nearest that view, not the arbitrary robot-to-centre angle.
        """

        attributes = node.get("attributes") or {}
        if not bool(node.get("is_currently_visible")):
            return None
        if str(attributes.get("attribute_status") or "").strip().casefold() != "ready":
            return None
        sources = (
            attributes.get("attribute_source"),
            attributes.get("view_state_source"),
            attributes.get("approach_source"),
        )
        if not any("mllm" in str(source or "").casefold() for source in sources):
            return None
        if attributes.get("attribute_is_current") is False:
            return None
        if attributes.get("visual_evidence_truncated") is True:
            return None
        view_state = str(
            attributes.get("view_state")
            or attributes.get("interaction_view_state")
            or ""
        ).strip().casefold()
        # A side/oblique image can identify a container but cannot establish a
        # contact face.  Keep it as a re-observation cue and use the ring
        # viewpoints instead; only a direct frontal M1 judgment can seed a
        # contact-axis pose.
        if view_state != "front":
            return None
        if not CandidateGenerator._truthy_mllm_attribute(
            attributes.get("front_surface_visible")
        ):
            return None
        if not CandidateGenerator._truthy_mllm_attribute(
            attributes.get("approach_ready")
        ):
            return None
        aabb_axis = CandidateGenerator._aabb_surface_axis(node, robot_xy, target_xy)
        if aabb_axis is not None:
            return aabb_axis
        axis_x = float(robot_xy[0]) - float(target_xy[0])
        axis_y = float(robot_xy[1]) - float(target_xy[1])
        norm = math.hypot(axis_x, axis_y)
        if norm <= 1e-6:
            return None
        return axis_x / norm, axis_y / norm

    @staticmethod
    def _aabb_surface_axis(
        node: dict[str, Any],
        robot_xy: tuple[float, float],
        target_xy: tuple[float, float],
    ) -> tuple[float, float] | None:
        """Return the AABB face normal on the side currently seen by the robot."""

        size = list(node.get("aabb_size") or [])
        if len(size) < 2:
            return None
        try:
            half_x = 0.5 * abs(float(size[0]))
            half_y = 0.5 * abs(float(size[1]))
        except (TypeError, ValueError):
            return None
        if half_x <= 1e-6 or half_y <= 1e-6:
            return None
        dx = float(robot_xy[0]) - float(target_xy[0])
        dy = float(robot_xy[1]) - float(target_xy[1])
        scaled_x = abs(dx) / half_x
        scaled_y = abs(dy) / half_y
        if scaled_x >= scaled_y:
            return (1.0, 0.0) if dx >= 0.0 else (-1.0, 0.0)
        return (0.0, 1.0) if dy >= 0.0 else (0.0, -1.0)

    @staticmethod
    def _truthy_mllm_attribute(value: Any) -> bool:
        return value is True or str(value or "").strip().casefold() in {
            "1",
            "true",
            "yes",
        }

    @staticmethod
    def _nonnegative_clearance(value: Any, fallback: float = 0.0) -> float:
        """Parse a public surface-clearance setting without propagating NaNs."""

        try:
            clearance = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        if not math.isfinite(clearance) or clearance < 0.0:
            return float(fallback)
        return clearance

    def _explicit_container_clearance(
        self,
        *,
        container_kind: str,
        phase: str,
    ) -> tuple[float | None, str]:
        """Return the direct type/generic phase setting, if one was supplied.

        The generic direct setting is intentionally lower precedence than a
        type-specific direct setting, but higher precedence than every legacy
        standoff field.  This lets a policy declare one contract for all
        containers while overriding, for example, refrigerator clearance.
        """

        if phase not in {"action", "m1_capture"}:
            return None, ""
        type_prefix = {
            "drawer": "drawer",
            "refrigerator": "fridge",
            "container": "container",
        }.get(container_kind, "container")
        field_suffix = f"_{phase}_standoff_m"
        field_names = [f"{type_prefix}{field_suffix}"]
        if type_prefix != "container":
            field_names.append(f"container{field_suffix}")
        for field_name in field_names:
            value = getattr(self.config, field_name, None)
            if value is None:
                continue
            clearance = self._nonnegative_clearance(value, fallback=-1.0)
            if clearance < 0.0:
                continue
            return clearance, f"explicit_{field_name}"
        return None, ""

    def _container_phase_standoffs(
        self,
        *,
        container_kind: str,
        legacy_action_standoff_m: float,
    ) -> tuple[float, str, float, str, str]:
        """Resolve direct action/capture clearances for a container candidate.

        Legacy policy files still provide a physical standoff plus a safety
        margin and a separate observation setting.  They are migrated here
        once into explicit values.  In new configs no action/capture ``max`` is
        taken at candidate generation: each phase has its declared clearance.
        If a legacy or explicit capture setting violates the safety invariant,
        the resolver records that validation and falls back to the action
        clearance rather than producing an unsafe capture pose.
        """

        legacy_action = self._nonnegative_clearance(legacy_action_standoff_m)
        explicit_action, action_source = self._explicit_container_clearance(
            container_kind=container_kind,
            phase="action",
        )
        if explicit_action is None:
            action_standoff = legacy_action
            action_source = "legacy_type_standoff_plus_safety_margin"
        else:
            action_standoff = explicit_action

        explicit_capture, capture_source = self._explicit_container_clearance(
            container_kind=container_kind,
            phase="m1_capture",
        )
        if explicit_capture is None:
            legacy_capture_field = {
                "drawer": "drawer_observation_standoff_m",
                "refrigerator": "fridge_observation_standoff_m",
                "container": "container_observation_standoff_m",
            }.get(container_kind, "container_observation_standoff_m")
            capture_standoff = self._nonnegative_clearance(
                getattr(self.config, legacy_capture_field, 0.0)
            )
            capture_source = f"legacy_{legacy_capture_field}"
        else:
            capture_standoff = explicit_capture

        validation = "valid"
        if capture_standoff + 1e-6 < action_standoff:
            # Do not repeat the old max(action, observation) rule here.  This
            # is an explicit invalid-config fallback, exposed in candidate
            # metadata so the applied capture clearance is never mysterious.
            capture_standoff = action_standoff
            validation = "capture_below_action_fallback_to_action"
        return (
            action_standoff,
            action_source,
            capture_standoff,
            capture_source,
            validation,
        )

    def _navigation_anchor_settings(self) -> tuple[float, int, float, str]:
        """Resolve outer recovery anchors without turning them into M1 poses."""

        explicit_offset = self.config.container_navigation_anchor_outer_offset_m
        explicit_count = self.config.container_navigation_anchor_ring_count
        explicit_tangent = self.config.container_navigation_anchor_tangent_offset_m
        offset_source = "explicit_navigation_anchor"
        if explicit_offset is None:
            explicit_offset = self.config.container_safe_staging_outer_offset_m
            offset_source = "legacy_safe_staging"
        if explicit_count is None:
            explicit_count = self.config.container_safe_staging_ring_count
        if explicit_tangent is None:
            explicit_tangent = self.config.container_safe_staging_tangent_offset_m
        offset = self._nonnegative_clearance(explicit_offset)
        tangent = self._nonnegative_clearance(explicit_tangent)
        try:
            ring_count = int(explicit_count)
        except (TypeError, ValueError):
            ring_count = 1
        return offset, max(1, min(4, ring_count)), tangent, offset_source

    def _approach_candidates(
        self,
        robot_xy: tuple[float, float],
        target_xy: tuple[float, float],
        node: dict[str, Any],
        standoff_m: float,
        node_type: str,
        *,
        visual_container_axis: tuple[float, float] | None = None,
        navigation_anchor_outer_offset_m: float = 0.0,
        navigation_anchor_ring_count: int = 1,
        navigation_anchor_tangent_offset_m: float = 0.0,
        navigation_anchor_aabb_fan_clearances_m: tuple[float, ...] = (),
        navigation_anchor_aabb_fan_angles_deg: tuple[float, ...] = (),
        container_multiview_angular_scale: float = 1.0,
        container_m1_face_selection_enabled: bool = False,
    ) -> tuple[list[list[float]], list[str]]:
        candidates: list[list[float]] = []
        labels: list[str] = []

        def append_unique(pose: list[float], label: str) -> None:
            if any(
                math.hypot(pose[0] - previous[0], pose[1] - previous[1]) <= 1e-6
                and abs(
                    math.atan2(
                        math.sin(pose[2] - previous[2]),
                        math.cos(pose[2] - previous[2]),
                    )
                )
                <= 1e-6
                for previous in candidates
            ):
                return
            candidates.append(pose)
            labels.append(str(label))

        if node_type == "portal":
            # A single radial approach can put all fallbacks on the blocked
            # side of a doorway.  Keep the original (robot-side, zero-offset)
            # candidates first, then try small tangential offsets and the
            # opposite doorway side.  The executor preflights these in order
            # and stops at the first reachable pose.
            for side_multiplier in (1.0, -1.0):
                for tangent_offset_m in (0.0, 0.20, -0.20):
                    for extra_standoff in (0.0, 0.25, 0.50):
                        candidate_standoff = max(0.0, float(standoff_m)) + extra_standoff
                        pose = self._portal_approach_pose(
                            robot_xy,
                            target_xy,
                            node,
                            candidate_standoff,
                            side_multiplier=side_multiplier,
                            tangent_offset_m=tangent_offset_m,
                        )
                        append_unique(
                            pose,
                            "portal_source_side"
                            if side_multiplier > 0.0
                            else "portal_opposite_side",
                        )
            return candidates, labels

        outer_offset = self._nonnegative_clearance(
            navigation_anchor_outer_offset_m
        )

        fan_clearances = sorted(
            {
                self._nonnegative_clearance(value)
                for value in navigation_anchor_aabb_fan_clearances_m
                if self._nonnegative_clearance(value) > 1e-6
            }
        )
        fan_angles_deg = []
        for raw_angle in navigation_anchor_aabb_fan_angles_deg:
            try:
                angle_deg = max(-89.0, min(89.0, float(raw_angle)))
            except (TypeError, ValueError):
                continue
            if angle_deg not in fan_angles_deg:
                fan_angles_deg.append(angle_deg)
        # A fan is a geometric set, not an execution queue.  Publish its stable
        # canonical order in the same near/straight-first preference used by
        # the batch planner: 0, -small, +small, -large, +large.
        fan_angles_deg.sort(
            key=lambda value: (abs(value), 0 if value <= 0.0 else 1)
        )
        if fan_clearances and fan_angles_deg:
            # Treat the public AABB as the real object bounds. Select the face
            # intersected by the centre-to-robot ray, then sample a surface fan
            # around that exact cardinal face normal. This removes the former
            # arbitrary robot-ray tilt while keeping the anchors on the visible
            # half of the box.
            size = list(node.get("aabb_size") or [])
            if len(size) >= 2:
                half_x = 0.5 * abs(float(size[0]))
                half_y = 0.5 * abs(float(size[1]))
            else:
                half_x = half_y = 0.0
            if half_x > 1e-6 and half_y > 1e-6:
                dx = float(robot_xy[0]) - float(target_xy[0])
                dy = float(robot_xy[1]) - float(target_xy[1])
                scaled_x = abs(dx) / half_x
                scaled_y = abs(dy) / half_y
                if container_m1_face_selection_enabled:
                    face_axes = (
                        (0.0, "pos_x"),
                        (math.pi / 2.0, "pos_y"),
                        (math.pi, "neg_x"),
                        (-math.pi / 2.0, "neg_y"),
                    )
                elif scaled_x >= scaled_y:
                    face_axes = ((0.0 if dx >= 0.0 else math.pi,
                                  "pos_x" if dx >= 0.0 else "neg_x"),)
                else:
                    face_axes = ((math.pi / 2.0 if dy >= 0.0 else -math.pi / 2.0,
                                  "pos_y" if dy >= 0.0 else "neg_y"),)
                for normal_angle, face_name in face_axes:
                    for clearance in fan_clearances:
                        for angle_deg in fan_angles_deg:
                            angle = normal_angle + math.radians(angle_deg)
                            axis = (math.cos(angle), math.sin(angle))
                            ray_scale = max(
                                abs(axis[0]) / half_x,
                                abs(axis[1]) / half_y,
                            )
                            boundary_distance = 1.0 / ray_scale
                            x = target_xy[0] + axis[0] * (
                                boundary_distance + clearance
                            )
                            y = target_xy[1] + axis[1] * (
                                boundary_distance + clearance
                            )
                            append_unique(
                                [
                                    x,
                                    y,
                                    math.atan2(target_xy[1] - y, target_xy[0] - x),
                                ],
                                (
                                    f"aabb_fan_{face_name}_angle_{angle_deg:+g}_"
                                    f"clearance_{clearance:.2f}"
                                ),
                            )
                return candidates, labels

        # M1 may explicitly say that the current RGB view contains the front
        # (or an adequate oblique face) of a container.  Snap that outward
        # approach axis to the current AABB face normal, not the arbitrary
        # robot-to-centre angle.  Do not read interaction_approach_axis_xy or an
        # interaction pose here: both can originate from GT/oracle geometry.
        if visual_container_axis is not None and not container_m1_face_selection_enabled and outer_offset <= 1e-6:
            # Preserve the rule/default behavior for callers that did not ask
            # for a model-lane safe staging ring.
            for extra_standoff in (0.0, 0.25, 0.50):
                candidate_standoff = max(0.0, float(standoff_m)) + extra_standoff
                append_unique(
                    self._approach_pose(
                        robot_xy,
                        target_xy,
                        candidate_standoff,
                        node=node,
                        fixed_axis=visual_container_axis,
                    ),
                    "mllm_current_view",
                )
            return candidates, labels

        try:
            angular_scale = max(0.0, min(1.0, float(container_multiview_angular_scale)))
        except (TypeError, ValueError):
            angular_scale = 1.0

        def rotate_axis(axis: tuple[float, float], angle: float) -> tuple[float, float]:
            cosine = math.cos(angle)
            sine = math.sin(angle)
            return (
                axis[0] * cosine - axis[1] * sine,
                axis[0] * sine + axis[1] * cosine,
            )

        if visual_container_axis is not None and not container_m1_face_selection_enabled:
            radial_axis = visual_container_axis
            axes = (
                radial_axis,
                rotate_axis(radial_axis, 0.5 * math.pi * angular_scale),
                rotate_axis(radial_axis, -0.5 * math.pi * angular_scale),
                rotate_axis(radial_axis, math.pi * angular_scale),
            )
            face_labels = (
                "mllm_current_view",
                "mllm_quarter_turn_left",
                "mllm_quarter_turn_right",
                "mllm_opposite_view",
            )
        elif not self.config.container_multiview_enabled:
            append_unique(
                self._approach_pose(
                    robot_xy,
                    target_xy,
                    max(0.0, float(standoff_m)),
                    node=node,
                ),
                "current_view",
            )
            return candidates, labels

        else:
            # No fresh M1 front observation was available.  The first pose
            # retains the current AABB face, then the executor may re-observe
            # from three orthogonal faces.  These labels describe only the
            # candidate-ring order; they do not assert a semantic front or
            # reuse oracle geometry.
            dx = float(robot_xy[0]) - float(target_xy[0])
            dy = float(robot_xy[1]) - float(target_xy[1])
            distance = math.hypot(dx, dy)
            aabb_axis = self._aabb_surface_axis(node, robot_xy, target_xy)
            radial_axis = aabb_axis or (
                (-1.0, 0.0)
                if distance <= 1e-6
                else (dx / distance, dy / distance)
            )
            if container_m1_face_selection_enabled:
                # The AABB supplies only geometry.  Enumerate all four
                # cardinal faces in a stable order; M1, not the robot's
                # current side, decides which one is the usable front.
                axes = (
                    (1.0, 0.0),
                    (0.0, 1.0),
                    (-1.0, 0.0),
                    (0.0, -1.0),
                )
                face_labels = (
                    "aabb_face_pos_x",
                    "aabb_face_pos_y",
                    "aabb_face_neg_x",
                    "aabb_face_neg_y",
                )
            else:
                axes = (
                    radial_axis,
                    rotate_axis(radial_axis, 0.5 * math.pi * angular_scale),
                    rotate_axis(radial_axis, -0.5 * math.pi * angular_scale),
                    rotate_axis(radial_axis, math.pi * angular_scale),
                )
                face_labels = (
                    "current_view",
                    "quarter_turn_left",
                    "quarter_turn_right",
                    "opposite_view",
                )
        face_count = max(1, min(len(axes), int(self.config.container_multiview_face_count)))
        if outer_offset > 1e-6:
            # Make every navigation anchor farther from the direct capture
            # point.  A wall-adjacent container can leave the closest route in
            # a local-inflation shoulder even though an anchor another 30 cm
            # away is reachable.  The executor later moves from this anchor to
            # the index-aligned direct M1 capture pose; M1 is not called here.
            # The bounded anchors remain ordinary move_base goals, so no
            # costmap cells are cleared or treated as hard-free.
            try:
                ring_count = int(navigation_anchor_ring_count)
            except (TypeError, ValueError):
                ring_count = 1
            ring_count = max(1, min(4, ring_count))
            ring_labels = ("safe_outer", "safe_far", "safe_farthest", "safe_max")
            # Keep each visual face contiguous across its bounded safe radii.
            # A two-stage container retry advances linearly through this list.
            # Ring-major ordering used to interleave faces (outer/current,
            # outer/left, ... far/current), so a navigation failure on one face
            # could repeatedly spend M1 attempts on an adjacent, non-front face
            # at every radius.  Face-major ordering instead tests the same
            # geometry from progressively safer/farther stances before moving
            # to a different side.  It changes only retry scheduling: every
            # pose still goes through ordinary make-plan and bridge checks.
            for axis, label in zip(axes[:face_count], face_labels[:face_count]):
                for ring_index in range(1, ring_count + 1):
                    ring_standoff = (
                        max(0.0, float(standoff_m)) + ring_index * outer_offset
                    )
                    ring_label = ring_labels[ring_index - 1]
                    base_label = f"{label}_{ring_label}"
                    base_pose = self._approach_pose(
                        robot_xy,
                        target_xy,
                        ring_standoff,
                        node=node,
                        fixed_axis=axis,
                    )
                    append_unique(base_pose, base_label)
                    # Preserve optional tangential recovery anchors on the
                    # closest ring.  Their capture mapping retains the same
                    # tangent shift at the direct capture clearance, while the
                    # physical action mapping continues to reuse the base face.
                    tangent_offset_m = self._nonnegative_clearance(
                        navigation_anchor_tangent_offset_m
                    )
                    if ring_index != 1 or tangent_offset_m <= 1e-6:
                        continue
                    tangent_x, tangent_y = -axis[1], axis[0]
                    for direction, tangent_label in (
                        (1.0, "tangent_left"),
                        (-1.0, "tangent_right"),
                    ):
                        x = float(base_pose[0]) + direction * tangent_offset_m * tangent_x
                        y = float(base_pose[1]) + direction * tangent_offset_m * tangent_y
                        append_unique(
                            [
                                x,
                                y,
                                math.atan2(target_xy[1] - y, target_xy[0] - x),
                            ],
                            f"{base_label}_{tangent_label}",
                        )
        else:
            for axis, label in zip(axes[:face_count], face_labels[:face_count]):
                append_unique(
                    self._approach_pose(
                        robot_xy,
                        target_xy,
                        max(0.0, float(standoff_m)),
                        node=node,
                        fixed_axis=axis,
                    ),
                    label,
                )
        return candidates, labels

    @staticmethod
    def _container_face_axis_from_label(
        label: str,
    ) -> tuple[float, float] | None:
        """Return the cardinal AABB face carried by an internal anchor label."""

        text = str(label)
        for face_name, axis in (
            ("pos_x", (1.0, 0.0)),
            ("pos_y", (0.0, 1.0)),
            ("neg_x", (-1.0, 0.0)),
            ("neg_y", (0.0, -1.0)),
        ):
            if text.startswith(f"aabb_face_{face_name}") or text.startswith(
                f"aabb_fan_{face_name}_"
            ):
                return axis
        return None

    @staticmethod
    def _container_staging_source_indices(staging_labels: list[str]) -> list[int]:
        """Map an outer tangent anchor back to its base-face action pose.

        The labels are internal candidate geometry labels, not semantic front
        claims.  Tangential anchors retain their camera shift at M1 capture,
        but must never rotate the subsequent physical contact axis.
        """

        labels = [str(label) for label in staging_labels]
        base_indices = {label: index for index, label in enumerate(labels)}
        source_indices: list[int] = []
        for index, label in enumerate(labels):
            source_index = index
            for suffix in ("_tangent_left", "_tangent_right"):
                if not label.endswith(suffix):
                    continue
                base_label = label[: -len(suffix)]
                candidate_index = base_indices.get(base_label)
                if candidate_index is not None and candidate_index < index:
                    source_index = candidate_index
                break
            source_indices.append(source_index)
        return source_indices

    @staticmethod
    def _container_m1_viewpoint_order(staging_labels: list[str]) -> list[int]:
        """Return a bounded, face-diverse M1 order over immutable anchors.

        Navigation remains face-major because it has to keep nearby recovery
        routes coherent.  Visual evidence has a different job: after a
        same-pose confirmation, it should expose one direct outer view from
        each available face before spending budget on tangents and farther
        rings.  Labels are generator-private bookkeeping only; no semantic
        front axis or oracle geometry is introduced by this scheduling.
        """

        labels = [str(label) for label in staging_labels]
        face_names = ("pos_x", "pos_y", "neg_x", "neg_y")
        face_fans: dict[str, list[tuple[float, float, int, int]]] = {
            face: [] for face in face_names
        }
        for index, label in enumerate(labels):
            match = re.fullmatch(
                r"aabb_fan_(pos_x|pos_y|neg_x|neg_y)_angle_"
                r"(?P<angle>[+-]?[0-9.]+)_clearance_(?P<clearance>[0-9.]+)",
                label,
            )
            if match is None:
                continue
            angle = float(match.group("angle"))
            clearance = float(match.group("clearance"))
            face_fans[match.group(1)].append(
                (clearance, abs(angle), 0 if angle <= 0.0 else 1, index)
            )
        if any(face_fans.values()):
            for values in face_fans.values():
                values.sort()
            ordered: list[int] = []
            max_per_face = max(len(values) for values in face_fans.values())
            # Round-robin across AABB faces.  One negative M1 view therefore
            # advances to a different physical face before spending evidence
            # budget on a wider angle/radius of the same face.
            for rank in range(max_per_face):
                for face in face_names:
                    values = face_fans[face]
                    if rank < len(values):
                        ordered.append(values[rank][3])
            ordered.extend(
                index for index in range(len(labels)) if index not in ordered
            )
            return ordered
        buckets: list[list[int]] = [[], [], [], [], []]
        for index, label in enumerate(labels):
            if "_safe_outer_tangent_" in label:
                buckets[1].append(index)
            elif label.endswith("_safe_outer"):
                buckets[0].append(index)
            elif label.endswith("_safe_far"):
                buckets[2].append(index)
            elif label.endswith("_safe_farthest"):
                buckets[3].append(index)
            elif label.endswith("_safe_max"):
                buckets[4].append(index)
        ordered = [index for bucket in buckets for index in bucket]
        # Hand-authored/legacy labels may not follow the generator naming
        # scheme.  Retain every valid anchor deterministically instead of
        # silently losing a safe fallback.
        ordered.extend(index for index in range(len(labels)) if index not in ordered)
        return ordered

    @classmethod
    def _container_m1_capture_goals_for_staging(
        cls,
        *,
        target_xy: tuple[float, float],
        staging_goals: list[list[float]],
        staging_labels: list[str],
        capture_standoff_m: float,
        node: dict[str, Any],
        staging_source_indices: list[int] | None = None,
    ) -> tuple[list[list[float]], list[str]]:
        """Map navigation anchors to direct, index-aligned M1 capture poses.

        An anchor may sit farther from the object solely to escape an inflated
        costmap.  It must not decide the visual evidence range.  Each mapped
        pose restores the configured direct capture clearance on the same
        radial face.  Tangent anchors retain their tangent displacement at the
        capture clearance so they remain genuinely different camera views;
        the separate physical-action mapping still reuses the base face.
        """

        capture_goals: list[list[float]] = []
        capture_labels: list[str] = []
        direct_capture_standoff = cls._nonnegative_clearance(capture_standoff_m)
        for index, staging_goal in enumerate(staging_goals):
            source_index = (
                int(staging_source_indices[index])
                if staging_source_indices is not None
                and index < len(staging_source_indices)
                else index
            )
            if source_index < 0 or source_index >= len(staging_goals):
                return [], []
            source_goal = list(staging_goals[source_index] or [])
            values = list(staging_goal or [])
            if len(source_goal) < 2 or len(values) < 2:
                return [], []
            try:
                axis_x = float(source_goal[0]) - float(target_xy[0])
                axis_y = float(source_goal[1]) - float(target_xy[1])
            except (TypeError, ValueError):
                return [], []
            axis_norm = math.hypot(axis_x, axis_y)
            if axis_norm <= 1e-6:
                return [], []
            axis = axis_x / axis_norm, axis_y / axis_norm
            capture = cls._approach_pose(
                target_xy,
                target_xy,
                direct_capture_standoff,
                node=node,
                fixed_axis=axis,
            )
            # Preserve a recovery anchor's tangent offset at the direct capture
            # radius.  Project rather than copying XY deltas so a different
            # outer radial ring cannot leak into the visual clearance.
            try:
                staging_delta_x = float(values[0]) - float(source_goal[0])
                staging_delta_y = float(values[1]) - float(source_goal[1])
            except (TypeError, ValueError):
                return [], []
            tangent_x, tangent_y = -axis[1], axis[0]
            tangent_offset = (
                staging_delta_x * tangent_x + staging_delta_y * tangent_y
            )
            if abs(tangent_offset) > 1e-9:
                capture[0] += tangent_offset * tangent_x
                capture[1] += tangent_offset * tangent_y
                capture[2] = math.atan2(
                    float(target_xy[1]) - capture[1],
                    float(target_xy[0]) - capture[0],
                )
            capture_goals.append(capture)
            label = (
                str(staging_labels[index])
                if index < len(staging_labels)
                else f"navigation_anchor_{index}"
            )
            capture_labels.append(f"{label}_m1_capture")
        return capture_goals, capture_labels

    @classmethod
    def _container_action_goals_for_staging(
        cls,
        *,
        target_xy: tuple[float, float],
        staging_goals: list[list[float]],
        staging_labels: list[str],
        physical_standoff_m: float,
        lateral_offset_m: float,
        node: dict[str, Any],
        staging_source_indices: list[int] | None = None,
    ) -> tuple[
        list[list[float]],
        list[str],
        list[list[list[float]]],
        list[list[str]],
    ]:
        """Map every outer M1 pose to bounded within-face physical poses.

        The mapping deliberately derives its axis from the already selected
        outer goal, not from an object joint, a graph-provided front axis, or a
        later M1 response.  This preserves the safe viewpoint's side while the
        inner point restores the original type-specific interaction standoff.
        """

        action_goals: list[list[float]] = []
        action_labels: list[str] = []
        action_goal_options: list[list[list[float]]] = []
        action_option_labels: list[list[str]] = []
        lateral_offset = max(0.0, float(lateral_offset_m))
        for index, staging_goal in enumerate(staging_goals):
            source_index = (
                int(staging_source_indices[index])
                if staging_source_indices is not None
                and index < len(staging_source_indices)
                else index
            )
            if 0 <= source_index < index:
                # The outer tangent is a new M1 capture pose, not a new
                # physical face.  Copy the already-built base face options so
                # the eventual bridge target cannot drift with camera offset.
                action_goals.append(list(action_goals[source_index]))
                action_labels.append(str(action_labels[source_index]))
                action_goal_options.append(
                    [list(option) for option in action_goal_options[source_index]]
                )
                action_option_labels.append(
                    list(action_option_labels[source_index])
                )
                continue
            values = list(staging_goal or [])
            if len(values) < 2:
                return [], [], [], []
            try:
                dx = float(values[0]) - float(target_xy[0])
                dy = float(values[1]) - float(target_xy[1])
            except (TypeError, ValueError):
                return [], [], [], []
            distance = math.hypot(dx, dy)
            if distance <= 1e-6:
                return [], [], [], []
            axis = dx / distance, dy / distance
            primary = cls._approach_pose(
                target_xy,
                target_xy,
                physical_standoff_m,
                node=node,
                fixed_axis=axis,
            )
            options = [primary]
            if lateral_offset > 1e-6:
                tangent_x, tangent_y = -axis[1], axis[0]
                for direction in (1.0, -1.0):
                    x = primary[0] + direction * lateral_offset * tangent_x
                    y = primary[1] + direction * lateral_offset * tangent_y
                    options.append(
                        [
                            x,
                            y,
                            math.atan2(target_xy[1] - y, target_xy[0] - x),
                        ]
                    )
            action_goals.append(primary)
            action_goal_options.append(options)
            label = (
                str(staging_labels[index])
                if index < len(staging_labels)
                else f"staging_{index}"
            )
            action_labels.append(f"{label}_physical_action")
            action_option_labels.append(
                [
                    f"{label}_physical_action",
                    f"{label}_physical_action_tangent_left",
                    f"{label}_physical_action_tangent_right",
                ][: len(options)]
            )
        return (
            action_goals,
            action_labels,
            action_goal_options,
            action_option_labels,
        )

    @staticmethod
    def _is_drawer_container(node: dict[str, Any]) -> bool:
        """Identify drawer semantics without treating every container as one.

        The graph keeps drawers as ``container`` nodes.  Prefer the structured
        interaction recipe, then fall back to stable object labels for legacy
        graphs that did not publish drawer groups.
        """

        attributes = node.get("attributes") or {}
        interaction = node.get("interaction") or {}
        groups = list(attributes.get("interaction_groups") or [])
        for group in groups:
            if not isinstance(group, dict):
                continue
            values = (
                group.get("mode"),
                group.get("view_profile"),
                group.get("group_id"),
                group.get("target_joint_names"),
            )
            if any("drawer" in str(value or "").casefold() for value in values):
                return True
        joint_infos = list(attributes.get("joint_infos") or [])
        if any(
            "drawer" in str(info.get("joint_name") or "").casefold()
            for info in joint_infos
            if isinstance(info, dict)
        ):
            return True
        labels = (
            node.get("label"),
            node.get("name"),
            attributes.get("semantic_name"),
            attributes.get("category"),
            attributes.get("source_object_name"),
            interaction.get("interaction_mode"),
        )
        return any(
            marker in str(label or "").casefold()
            for label in labels
            for marker in ("drawer", "dresser", "chest_of_drawers", "chestofdrawers")
        )

    @staticmethod
    def _is_refrigerator_container(node: dict[str, Any]) -> bool:
        """Identify refrigerator-like hinged containers for clearance policy."""

        attributes = node.get("attributes") or {}
        labels = (
            node.get("label"),
            node.get("name"),
            attributes.get("semantic_name"),
            attributes.get("category"),
            attributes.get("source_object_name"),
        )
        return any(
            marker in str(label or "").casefold()
            for label in labels
            for marker in ("refrigerator", "fridge")
        )

    @staticmethod
    def _is_openable_container(node: dict[str, Any]) -> bool:
        attributes = node.get("attributes") or {}
        labels = (
            node.get("label"),
            node.get("name"),
            attributes.get("semantic_name"),
            attributes.get("category"),
            attributes.get("source_object_name"),
        )
        if any(
            str(label or "").strip().casefold() in {"box", "storage_bin"}
            for label in labels
        ):
            return False
        interaction = node.get("interaction") or {}
        mode = str(interaction.get("interaction_mode") or "")
        if mode and mode not in {"open_close", "slide"}:
            return False
        return True

    @staticmethod
    def _node_xy(
        node: dict[str, Any], prefer_aabb: bool = False
    ) -> tuple[float, float] | None:
        keys = ("aabb_center", "centroid", "position") if prefer_aabb else (
            "centroid",
            "aabb_center",
            "position",
        )
        for key in keys:
            values = list(node.get(key) or [])
            if len(values) >= 2:
                return float(values[0]), float(values[1])
        return None

    @classmethod
    def _target_navigation_anchor(
        cls,
        node: dict[str, Any],
        graph: dict[str, Any],
        nodes_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        target_id = str(node.get("id") or "")
        parent_ids = [str(node.get("parent_id") or "")]
        parent_ids.extend(
            str(edge.get("src_id") or "")
            for edge in graph.get("edges") or []
            if str(edge.get("dst_id") or "") == target_id
            and str(edge.get("relation") or "").casefold() == "contains"
        )
        for parent_id in parent_ids:
            parent = nodes_by_id.get(parent_id, {})
            if str(parent.get("type") or "").casefold() == "container":
                return parent

        target_center = list(node.get("aabb_center") or node.get("centroid") or [])
        if len(target_center) < 3:
            return node
        target_room_id = node.get("room_id")
        nearby = []
        margin_m = 0.25
        for candidate in graph.get("nodes") or []:
            if str(candidate.get("type") or "").casefold() != "container":
                continue
            if (
                target_room_id is not None
                and candidate.get("room_id") is not None
                and int(candidate.get("room_id")) != int(target_room_id)
            ):
                continue
            center = list(candidate.get("aabb_center") or candidate.get("centroid") or [])
            size = list(candidate.get("aabb_size") or [])
            if len(center) < 3 or len(size) < 3:
                continue
            normalized_distance = 0.0
            inside_expanded_box = True
            for axis in range(3):
                half_extent = 0.5 * abs(float(size[axis]))
                delta = abs(float(target_center[axis]) - float(center[axis]))
                if delta > half_extent + margin_m:
                    inside_expanded_box = False
                    break
                normalized_distance += (delta / max(half_extent + margin_m, 1e-6)) ** 2
            if inside_expanded_box:
                nearby.append(
                    (normalized_distance, str(candidate.get("id") or ""), candidate)
                )
        return min(nearby, default=(0.0, "", node))[2]

    @staticmethod
    def _successful_interaction_approach(
        node: dict[str, Any],
    ) -> list[float] | None:
        history = list((node.get("interaction") or {}).get("operation_history") or [])
        for event in reversed(history):
            if not bool(event.get("success")):
                continue
            values = list(event.get("approach_goal_xyyaw") or [])
            if len(values) >= 2:
                return [
                    float(values[0]),
                    float(values[1]),
                    float(values[2]) if len(values) > 2 else 0.0,
                ]
        return None

    @staticmethod
    def _container_approach_axis(node: dict[str, Any]) -> tuple[float, float] | None:
        attributes = node.get("attributes") or {}
        values = attributes.get("interaction_approach_axis_xy")
        if values is None:
            values = node.get("interaction_approach_axis_xy")
        values = list(values or [])
        if len(values) < 2:
            return None
        norm = math.hypot(float(values[0]), float(values[1]))
        if norm <= 1e-6:
            return None
        return float(values[0]) / norm, float(values[1]) / norm

    @staticmethod
    def _container_approach_pose(node: dict[str, Any]) -> list[float] | None:
        attributes = node.get("attributes") or {}
        values = list(
            attributes.get("interaction_approach_pose_xyyaw")
            or node.get("interaction_approach_pose_xyyaw")
            or []
        )
        if len(values) < 3:
            return None
        return [float(values[0]), float(values[1]), float(values[2])]

    @staticmethod
    def _room_id_for_xy(
        graph: dict[str, Any], robot_xy: tuple[float, float]
    ) -> int | None:
        matches: list[tuple[float, int]] = []
        for node in graph.get("nodes") or []:
            if str(node.get("type") or "") != "room":
                continue
            if not bool((node.get("attributes") or {}).get("active", True)):
                continue
            center = list(node.get("aabb_center") or node.get("centroid") or [])
            size = list(node.get("aabb_size") or [])
            room_id = node.get("room_id")
            if len(center) < 2 or len(size) < 2 or room_id is None:
                continue
            half_x = 0.5 * abs(float(size[0]))
            half_y = 0.5 * abs(float(size[1]))
            if (
                abs(float(robot_xy[0]) - float(center[0])) <= half_x + 1e-6
                and abs(float(robot_xy[1]) - float(center[1])) <= half_y + 1e-6
            ):
                matches.append((max(half_x * half_y, 1e-6), int(room_id)))
        if not matches:
            return None
        return min(matches)[1]

    @classmethod
    def _room_hops(
        cls,
        graph: dict[str, Any],
        source_room_id: int | None,
        target_room_id: Any,
    ) -> int | None:
        if source_room_id is None or target_room_id is None:
            return None
        source = int(source_room_id)
        target = int(target_room_id)
        if source == target:
            return 0
        node_room_ids = {
            str(node.get("id")): int(node.get("room_id"))
            for node in graph.get("nodes") or []
            if str(node.get("type") or "") == "room"
            and node.get("id") is not None
            and node.get("room_id") is not None
        }
        adjacency: dict[int, set[int]] = {}
        portal_rooms: dict[str, set[int]] = {}
        for edge in graph.get("edges") or []:
            attributes = edge.get("attributes") or {}
            traversable = attributes.get("traversable")
            if traversable is not True and str(traversable).casefold() != "true":
                continue
            connected_room_ids = {
                int(value)
                for value in (
                    attributes.get("candidate_connected_room_ids")
                    or attributes.get("connected_room_ids")
                    or []
                )
                if value is not None
            }
            endpoint_rooms = {
                node_room_ids[endpoint]
                for endpoint in (edge.get("src_id"), edge.get("dst_id"))
                if str(endpoint) in node_room_ids
            }
            connected_room_ids.update(endpoint_rooms)
            portal_key = str(
                attributes.get("portal_node_id")
                or edge.get("src_id")
                or edge.get("dst_id")
                or ""
            )
            if len(connected_room_ids) >= 2:
                room_list = sorted(connected_room_ids)
                for index, left in enumerate(room_list):
                    for right in room_list[index + 1 :]:
                        adjacency.setdefault(left, set()).add(right)
                        adjacency.setdefault(right, set()).add(left)
            elif portal_key and connected_room_ids:
                portal_rooms.setdefault(portal_key, set()).update(connected_room_ids)
        for connected_room_ids in portal_rooms.values():
            room_list = sorted(connected_room_ids)
            for index, left in enumerate(room_list):
                for right in room_list[index + 1 :]:
                    adjacency.setdefault(left, set()).add(right)
                    adjacency.setdefault(right, set()).add(left)
        queue = [(source, 0)]
        visited = {source}
        while queue:
            current, hops = queue.pop(0)
            for neighbor in adjacency.get(current, set()):
                if neighbor == target:
                    return hops + 1
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, hops + 1))
        return None

    @staticmethod
    def _approach_pose(
        robot_xy: tuple[float, float],
        target_xy: tuple[float, float],
        standoff_m: float,
        node: dict[str, Any] | None = None,
        fixed_axis: tuple[float, float] | None = None,
    ) -> list[float]:
        if fixed_axis is not None:
            unit_x, unit_y = fixed_axis
        else:
            dx = robot_xy[0] - target_xy[0]
            dy = robot_xy[1] - target_xy[1]
            distance = math.hypot(dx, dy)
            if distance <= 1e-6:
                unit_x, unit_y = -1.0, 0.0
            else:
                unit_x, unit_y = dx / distance, dy / distance
        boundary_distance = 0.0
        if node is not None:
            size = list(node.get("aabb_size") or [])
            if len(size) >= 2:
                half_x = 0.5 * abs(float(size[0]))
                half_y = 0.5 * abs(float(size[1]))
                if half_x > 1e-6 and half_y > 1e-6:
                    ray_denominator = abs(unit_x) / half_x + abs(unit_y) / half_y
                    if ray_denominator > 1e-6:
                        boundary_distance = 1.0 / ray_denominator
        offset = max(0.0, standoff_m) + boundary_distance
        x = target_xy[0] + unit_x * offset
        y = target_xy[1] + unit_y * offset
        yaw = math.atan2(target_xy[1] - y, target_xy[0] - x)
        return [x, y, yaw]

    @classmethod
    def _portal_approach_pose(
        cls,
        robot_xy: tuple[float, float],
        target_xy: tuple[float, float],
        node: dict[str, Any],
        standoff_m: float,
        side_multiplier: float = 1.0,
        tangent_offset_m: float = 0.0,
    ) -> list[float]:
        attributes = node.get("attributes") or {}
        reference_center = list(
            attributes.get("interaction_reference_aabb_center")
            or node.get("aabb_center")
            or target_xy
        )
        if len(reference_center) >= 2:
            target_xy = float(reference_center[0]), float(reference_center[1])
        size = list(
            attributes.get("interaction_reference_aabb_size")
            or node.get("aabb_size")
            or []
        )
        boundary_distance = 0.0
        size_x = max(0.0, float(size[0])) if len(size) >= 1 else 0.0
        size_y = max(0.0, float(size[1])) if len(size) >= 2 else 0.0
        major = max(size_x, size_y)
        minor = min(size_x, size_y)
        elongated = major > 1e-6 and major / max(minor, 1e-6) >= 1.35
        if elongated:
            # The short AABB axis is the doorway normal.  Use the robot side
            # for the primary pose and allow the caller to request the other
            # side with side_multiplier=-1.
            if size_x <= size_y:
                normal_x, normal_y = 1.0, 0.0
            else:
                normal_x, normal_y = 0.0, 1.0
            side = 1.0 if (
                (robot_xy[0] - target_xy[0]) * normal_x
                + (robot_xy[1] - target_xy[1]) * normal_y
            ) >= 0.0 else -1.0
            normal_x *= side * float(side_multiplier)
            normal_y *= side * float(side_multiplier)
            boundary_distance = 0.5 * minor
        else:
            # Rotated/nearly-square AABBs do not expose a reliable normal.
            # Retain the old radial direction as the primary side, and mirror
            # that direction for the opposite-side fallback.
            dx = float(robot_xy[0]) - float(target_xy[0])
            dy = float(robot_xy[1]) - float(target_xy[1])
            distance = math.hypot(dx, dy)
            if distance <= 1e-6:
                normal_x, normal_y = -1.0, 0.0
            else:
                normal_x, normal_y = dx / distance, dy / distance
            if size_x > 1e-6 and size_y > 1e-6:
                ray_denominator = abs(normal_x) / (0.5 * size_x) + abs(normal_y) / (0.5 * size_y)
                if ray_denominator > 1e-6:
                    boundary_distance = 1.0 / ray_denominator
            normal_x *= float(side_multiplier)
            normal_y *= float(side_multiplier)
        offset = max(0.0, standoff_m) + boundary_distance
        tangent_x, tangent_y = -normal_y, normal_x
        x = target_xy[0] + normal_x * offset + tangent_x * float(tangent_offset_m)
        y = target_xy[1] + normal_y * offset + tangent_y * float(tangent_offset_m)
        # Every candidate faces the interaction reference, including the
        # opposite-side and tangential fallbacks.
        yaw = math.atan2(target_xy[1] - y, target_xy[0] - x)
        return [x, y, yaw]
