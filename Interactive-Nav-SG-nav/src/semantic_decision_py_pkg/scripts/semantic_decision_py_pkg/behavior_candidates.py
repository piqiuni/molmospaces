from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any


BEHAVIOR_EXPLORE = "EXPLORE"
BEHAVIOR_INTERACT = "INTERACT"
BEHAVIOR_NAVIGATE = "NAVIGATE"


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
    interaction_types: tuple[str, ...] = ("portal",)
    max_interaction_distance_m: float = 6.0
    max_state_age_sec: float = 60.0
    min_state_confidence: float = 0.5
    portal_standoff_m: float = 1.15
    container_standoff_m: float = 0.90
    interaction_ready_distance_m: float = 0.45
    require_current_visibility: bool = False
    target_standoff_m: float = 0.90
    target_max_state_age_sec: float = 300.0


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
        candidates = self._frontier_candidates(explorer_status or {})
        if robot_xy is not None:
            candidates.extend(self._interaction_candidates(graph or {}, robot_xy))
            candidates.extend(
                self._target_candidates(graph or {}, robot_xy, target_context or {})
            )
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
        for node in graph.get("nodes") or []:
            if str(node.get("type") or "") in {"scene", "room", "portal"}:
                continue
            if not self._matches_target(node, target_context):
                continue
            state_age_sec = max(0.0, float(node.get("state_age_sec", 0.0) or 0.0))
            if state_age_sec > self.config.target_max_state_age_sec:
                continue
            position = self._node_xy(node)
            if position is None:
                continue
            goal = self._approach_pose(
                robot_xy,
                position,
                float(target_context.get("standoff_m", self.config.target_standoff_m)),
            )
            distance_m = math.hypot(goal[0] - robot_xy[0], goal[1] - robot_xy[1])
            node_id = str(node.get("id") or "")
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
                        "visibility_gain": 0.2,
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
                        "is_currently_visible": bool(node.get("is_currently_visible")),
                        "state_age_sec": state_age_sec,
                        "approach_strategy": "target_radial_standoff",
                    },
                )
            )
        return candidates

    @staticmethod
    def _matches_target(node: dict[str, Any], target_context: dict[str, Any]) -> bool:
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
        attributes = node.get("attributes") or {}
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
        clusters = list(status.get("frontier_clusters") or [])
        clusters.sort(key=lambda cluster: float(cluster.get("score", 0.0)), reverse=True)
        clusters = clusters[: max(0, int(self.config.max_frontier_candidates))]
        if not clusters:
            return []
        raw_information = [max(0.0, float(cluster.get("information_gain", 0.0))) for cluster in clusters]
        max_information = max(raw_information) if raw_information else 1.0
        raw_scores = [float(cluster.get("score", 0.0)) for cluster in clusters]
        min_score = min(raw_scores)
        score_span = max(max(raw_scores) - min_score, 1e-6)
        candidates = []
        for cluster, information_gain, explorer_score in zip(
            clusters, raw_information, raw_scores
        ):
            cluster_id = str(cluster.get("cluster_id") or "")
            subgoal = list(cluster.get("subgoal_world") or [])
            if not cluster_id or len(subgoal) < 2:
                continue
            yaw = float(cluster.get("subgoal_yaw", 0.0))
            information_normalized = (
                math.log1p(information_gain) / max(math.log1p(max_information), 1e-6)
            )
            score_normalized = (explorer_score - min_score) / score_span
            distance_m = max(0.0, float(cluster.get("distance_to_robot", 0.0)))
            candidates.append(
                BehaviorCandidate(
                    candidate_id=f"frontier:{cluster_id}",
                    behavior_type=BEHAVIOR_EXPLORE,
                    source="explore_py",
                    target_id=cluster_id,
                    target_name=cluster_id,
                    goal_xyyaw=[float(subgoal[0]), float(subgoal[1]), yaw],
                    features={
                        "exploration_gain": information_normalized,
                        "visibility_gain": information_normalized,
                        "semantic_gain": score_normalized,
                        "distance_m": distance_m,
                        "interaction_cost": 0.0,
                        "state_age_ratio": 0.0,
                        "confidence": 1.0,
                        "priority": score_normalized,
                    },
                    metadata={
                        "cluster_id": cluster_id,
                        "frontier_point": list(cluster.get("centroid_world") or []),
                        "cell_count": int(cluster.get("cell_count", 0)),
                        "explorer_score": explorer_score,
                        "explorer_score_terms": dict(cluster.get("score_terms") or {}),
                    },
                )
            )
        return candidates

    def _interaction_candidates(
        self, graph: dict[str, Any], robot_xy: tuple[float, float]
    ) -> list[BehaviorCandidate]:
        candidates = []
        allowed_types = set(self.config.interaction_types)
        for node in graph.get("nodes") or []:
            node_type = str(node.get("type") or "")
            if node_type not in allowed_types:
                continue
            interaction = node.get("interaction") or {}
            if not bool(interaction.get("is_interactable", True)):
                continue
            if not bool(interaction.get("requires_interaction")):
                continue
            if str(interaction.get("state") or "unknown") not in {"closed", "ajar", "unknown"}:
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
            position = self._node_xy(node)
            if position is None:
                continue
            object_distance = math.hypot(position[0] - robot_xy[0], position[1] - robot_xy[1])
            if object_distance > self.config.max_interaction_distance_m:
                continue
            standoff = (
                self.config.portal_standoff_m
                if node_type == "portal"
                else self.config.container_standoff_m
            )
            approach = (
                self._portal_approach_pose(robot_xy, position, node, standoff)
                if node_type == "portal"
                else self._approach_pose(robot_xy, position, standoff)
            )
            approach_distance = math.hypot(
                approach[0] - robot_xy[0], approach[1] - robot_xy[1]
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
            candidates.append(
                BehaviorCandidate(
                    candidate_id=f"interaction:{node_id}:open",
                    behavior_type=BEHAVIOR_INTERACT,
                    source="unified_graph",
                    target_id=node_id,
                    target_name=source_object_name,
                    goal_xyyaw=approach,
                    interaction_command={
                        "node_id": node_id,
                        "source_object_name": source_object_name,
                        "action": "open",
                        "interaction_mode": str(interaction.get("interaction_mode") or "open_close"),
                        "expected_state": "open",
                    },
                    features={
                        "exploration_gain": exploration_gain,
                        "visibility_gain": 1.0 if node_type == "portal" else 0.80,
                        "semantic_gain": 1.0 if node_type == "portal" else 0.75,
                        "distance_m": approach_distance,
                        "interaction_cost": float(
                            interaction.get("interaction_cost", interaction.get("cost", 1.0)) or 1.0
                        ),
                        "state_age_ratio": min(
                            1.0, state_age_sec / max(self.config.max_state_age_sec, 1e-6)
                        ),
                        "confidence": confidence,
                        "priority": 1.0 if node_type == "portal" else 0.75,
                    },
                    metadata={
                        "node_type": node_type,
                        "state": str(interaction.get("state") or "unknown"),
                        "expected_effect": expected_effect,
                        "connected_room_ids": connected_room_ids,
                        "connectivity_status": attributes.get("connectivity_status", "unknown"),
                        "is_currently_visible": bool(node.get("is_currently_visible")),
                        "state_age_sec": state_age_sec,
                        "object_distance_m": object_distance,
                        "requires_approach": approach_distance > self.config.interaction_ready_distance_m,
                        "approach_strategy": (
                            "portal_aabb_normal" if node_type == "portal" else "radial_standoff"
                        ),
                    },
                )
            )
        return candidates

    @staticmethod
    def _node_xy(node: dict[str, Any]) -> tuple[float, float] | None:
        for key in ("centroid", "aabb_center", "position"):
            values = list(node.get(key) or [])
            if len(values) >= 2:
                return float(values[0]), float(values[1])
        return None

    @staticmethod
    def _approach_pose(
        robot_xy: tuple[float, float], target_xy: tuple[float, float], standoff_m: float
    ) -> list[float]:
        dx = robot_xy[0] - target_xy[0]
        dy = robot_xy[1] - target_xy[1]
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            unit_x, unit_y = -1.0, 0.0
        else:
            unit_x, unit_y = dx / distance, dy / distance
        x = target_xy[0] + unit_x * max(0.0, standoff_m)
        y = target_xy[1] + unit_y * max(0.0, standoff_m)
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
        size = list(node.get("aabb_size") or [])
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
