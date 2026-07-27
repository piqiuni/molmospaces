from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


BEHAVIOR_EXPLORE = "EXPLORE"
BEHAVIOR_INTERACT = "INTERACT"
BEHAVIOR_NAVIGATE = "NAVIGATE"

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
    portal_standoff_m: float = 1.0
    container_standoff_m: float = 1.0
    drawer_standoff_m: float = 1.0
    interaction_safety_margin_m: float = 0.0
    interaction_ready_distance_m: float = 0.45
    require_current_visibility: bool = False
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
                self._target_candidates(graph or {}, robot_xy, target_context or {})
            )
        self._attach_spatial_context(candidates, graph or {})
        return sorted(candidates, key=lambda candidate: candidate.candidate_id)

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
        proposals = proposals[: max(0, int(self.config.max_frontier_candidates))]
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
            if not bool(interaction.get("is_interactable", False)):
                continue
            if node_type == "container" and not self._is_openable_container(node):
                continue
            node_state = str(interaction.get("state") or "unknown")
            if node_type == "portal" and (
                not bool(interaction.get("requires_interaction"))
                or node_state not in {"closed", "ajar", "unknown"}
            ):
                continue
            confidence = float(
                interaction.get("state_confidence", interaction.get("confidence", node.get("confidence", 0.0)))
                or 0.0
            )
            if confidence < self.config.min_state_confidence:
                continue
            state_age_sec = max(0.0, float(node.get("state_age_sec", 0.0) or 0.0))
            if state_age_sec > self.config.max_state_age_sec:
                continue
            if self.config.require_current_visibility and not bool(node.get("is_currently_visible")):
                continue
            node_room_id = node.get("room_id")
            allow_connected_room = bool(self.config.container_allow_connected_room)
            room_hops = self._room_hops(graph, robot_room_id, node_room_id)
            if (
                node_type == "container"
                and self.config.container_require_same_room
                and node_room_id is not None
                and robot_room_id is not None
                and int(node_room_id) != int(robot_room_id)
                and not (allow_connected_room and room_hops is not None)
            ):
                continue
            position = self._node_xy(node)
            if position is None:
                continue
            object_distance = math.hypot(position[0] - robot_xy[0], position[1] - robot_xy[1])
            standoff = (
                self.config.portal_standoff_m
                if node_type == "portal"
                else self.config.container_standoff_m
            )
            node_id = str(node.get("id") or "")
            attributes = node.get("attributes") or {}
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
            interaction_standoff = standoff + self.config.interaction_safety_margin_m
            goal_candidates = self._approach_candidates(
                robot_xy,
                position,
                node,
                interaction_standoff,
                node_type,
            )
            approach = goal_candidates[0]
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
                "interaction_approach_axis_xy": list(
                    self._container_approach_axis(node) or []
                ),
                "interaction_ready_distance_m": self.config.interaction_ready_distance_m,
                "interaction_ready_yaw_tolerance_rad": 0.55,
            }
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
                        "interaction_standoff_m": interaction_standoff,
                        "interaction_safety_margin_m": self.config.interaction_safety_margin_m,
                        "target_enabled": bool(target_context.get("enabled")),
                        "target_match": explicit_target_reinteraction,
                        "requires_approach": True,
                        "approach_strategy": (
                            "portal_aabb_normal"
                            if node_type == "portal"
                            else (
                                "container_explicit_pose"
                                if self._container_approach_pose(node) is not None
                                else (
                                    "container_front_axis"
                                    if self._container_approach_axis(node) is not None
                                    else "radial_standoff"
                                )
                            )
                        ),
                        "interaction_approach_axis_xy": self._container_approach_axis(
                            node
                        ),
                        "goal_xyyaw_candidates": goal_candidates,
                    },

                )
            )
        return candidates

    @classmethod
    def _approach_candidates(
        cls,
        robot_xy: tuple[float, float],
        target_xy: tuple[float, float],
        node: dict[str, Any],
        standoff_m: float,
        node_type: str,
    ) -> list[list[float]]:
        candidates: list[list[float]] = []
        for extra_standoff in (0.0, 0.25, 0.50):
            candidate_standoff = max(0.0, float(standoff_m)) + extra_standoff
            if node_type == "portal":
                pose = cls._portal_approach_pose(
                    robot_xy, target_xy, node, candidate_standoff
                )
            else:
                explicit_pose = cls._container_approach_pose(node)
                if explicit_pose is not None:
                    axis = cls._container_approach_axis(node)
                    if axis is None:
                        axis = (
                            math.cos(float(explicit_pose[2]) + math.pi),
                            math.sin(float(explicit_pose[2]) + math.pi),
                        )
                    pose = [
                        float(explicit_pose[0]) + axis[0] * extra_standoff,
                        float(explicit_pose[1]) + axis[1] * extra_standoff,
                        float(explicit_pose[2]),
                    ]
                else:
                    pose = cls._approach_pose(
                        robot_xy,
                        target_xy,
                        candidate_standoff,
                        node=node,
                        fixed_axis=cls._container_approach_axis(node),
                    )
            if not any(
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
                candidates.append(pose)
        return candidates

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
        if len(size) < 2:
            return cls._approach_pose(robot_xy, target_xy, standoff_m)
        size_x = max(0.0, float(size[0]))
        size_y = max(0.0, float(size[1]))
        major = max(size_x, size_y)
        minor = min(size_x, size_y)
        if major <= 1e-6 or major / max(minor, 1e-6) < 1.35:
            return cls._approach_pose(robot_xy, target_xy, standoff_m)
        if size_x <= size_y:
            normal_x, normal_y = 1.0, 0.0
        else:
            normal_x, normal_y = 0.0, 1.0
        side = 1.0 if (
            (robot_xy[0] - target_xy[0]) * normal_x
            + (robot_xy[1] - target_xy[1]) * normal_y
        ) >= 0.0 else -1.0
        offset = max(0.0, standoff_m) + 0.5 * minor
        x = target_xy[0] + side * normal_x * offset
        y = target_xy[1] + side * normal_y * offset
        yaw = math.atan2(target_xy[1] - y, target_xy[0] - x)
        return [x, y, yaw]
