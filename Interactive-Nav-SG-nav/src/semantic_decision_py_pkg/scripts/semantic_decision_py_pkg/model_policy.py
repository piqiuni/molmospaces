from __future__ import annotations

from dataclasses import dataclass
import json
import shlex
import subprocess
import time
from typing import Any, Iterable
from .behavior_candidates import BehaviorCandidate
from semantic_mllm_py_pkg.client import MLLMClient, MLLMClientConfig
from semantic_mllm_py_pkg.schemas import validate_subgoal_selection


@dataclass
class ModelPolicyConfig:
    mode: str = "disabled"
    command: str = ""
    endpoint: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "qwen3.6-35b-a3b"
    protocol: str = "openai_chat"
    timeout_s: float = 3.0
    temperature: float = 0.0
    max_tokens: int = 256
    reasoning_effort: str = "off"
    image_detail: str = "low"
    max_graph_nodes: int = 80
    max_graph_edges: int = 160
    metrics_path: str = ""


@dataclass
class ModelCircuitBreaker:
    consecutive_timeout_limit: int = 2
    cooldown_s: float = 60.0
    consecutive_timeouts: int = 0
    open_until: float = 0.0
    last_error: str = ""

    def allow_request(self, now: float | None = None) -> bool:
        return float(now if now is not None else time.monotonic()) >= self.open_until

    def record_success(self) -> None:
        self.consecutive_timeouts = 0
        self.last_error = ""

    def record_failure(self, error: str, now: float | None = None) -> bool:
        self.last_error = str(error or "model_request_failed")
        if "timed out" not in self.last_error.casefold():
            self.consecutive_timeouts = 0
            return False
        self.consecutive_timeouts += 1
        if self.consecutive_timeouts < max(1, int(self.consecutive_timeout_limit)):
            return False
        current = float(now if now is not None else time.monotonic())
        self.open_until = current + max(0.0, float(self.cooldown_s))
        self.consecutive_timeouts = 0
        return True


def compact_graph(graph: dict[str, Any], max_nodes: int = 80, max_edges: int = 160) -> dict[str, Any]:
    nodes = []
    for node in list(graph.get("nodes") or [])[: max(0, int(max_nodes))]:
        attributes = node.get("attributes") or {}
        interaction = node.get("interaction") or {}
        nodes.append(
            {
                "id": node.get("id"),
                "type": node.get("type"),
                "label": node.get("label"),
                "name": node.get("name"),
                "centroid": list(node.get("centroid") or [])[:3],
                "room_id": node.get("room_id"),
                "is_currently_visible": bool(node.get("is_currently_visible")),
                "state_age_sec": node.get("state_age_sec", 0.0),
                "source_object_name": attributes.get("source_object_name"),
                "connected_room_ids": list(attributes.get("connected_room_ids") or []),
                "interaction_state": interaction.get("state"),
                "requires_interaction": interaction.get("requires_interaction"),
                "traversable": interaction.get("traversable"),
            }
        )
    edges = [
        {
            "src_id": edge.get("src_id"),
            "relation": edge.get("relation"),
            "dst_id": edge.get("dst_id"),
            "attributes": dict(edge.get("attributes") or {}),
        }
        for edge in list(graph.get("edges") or [])[: max(0, int(max_edges))]
    ]
    return {
        "scene_id": graph.get("scene_id", ""),
        "episode_id": graph.get("episode_id", ""),
        "graph_revision": graph.get("graph_revision", 0),
        "nodes": nodes,
        "edges": edges,
    }


def _room_node_id(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    return text if text.startswith("room_") else f"room_{text}"


def _node_interaction(node: dict[str, Any]) -> dict[str, Any]:
    interaction = dict(node.get("interaction") or {})
    for key in ("state", "requires_interaction", "traversable"):
        compact_key = f"interaction_{key}" if key == "state" else key
        if key not in interaction and compact_key in node:
            interaction[key] = node.get(compact_key)
    return interaction


def _current_room_id(
    rooms: list[dict[str, Any]], robot_context: dict[str, Any] | None
) -> str:
    robot_xy = list((robot_context or {}).get("robot_xy") or [])
    if len(robot_xy) < 2:
        return ""
    closest_id = ""
    closest_distance_sq = float("inf")
    for room in rooms:
        centroid = list(room.get("_centroid") or [])
        if len(centroid) < 2:
            continue
        distance_sq = (float(centroid[0]) - float(robot_xy[0])) ** 2 + (
            float(centroid[1]) - float(robot_xy[1])
        ) ** 2
        if distance_sq < closest_distance_sq:
            closest_distance_sq = distance_sq
            closest_id = str(room.get("id") or "")
    return closest_id


def compact_semantic_graph(
    graph: dict[str, Any],
    robot_context: dict[str, Any] | None = None,
    max_nodes: int = 80,
) -> dict[str, Any]:
    rooms = []
    portals = []
    containers = []
    for node in list(graph.get("nodes") or [])[: max(0, int(max_nodes))]:
        node_type = str(node.get("type") or "").casefold()
        if node_type not in {"room", "portal", "container"}:
            continue
        attributes = node.get("attributes") or {}
        interaction = _node_interaction(node)
        node_id = str(node.get("id") or "")
        semantic_name = str(node.get("label") or node.get("name") or node_id)
        if node_type == "room":
            rooms.append(
                {
                    "id": node_id,
                    "type": semantic_name or "unknown",
                    "_centroid": list(node.get("centroid") or [])[:2],
                }
            )
            continue
        item = {
            "id": node_id,
            "type": semantic_name or node_type,
            "state": str(interaction.get("state") or "unknown"),
            "interaction_available": bool(
                interaction.get("requires_interaction")
                or interaction.get("is_interactable")
            ),
        }
        if node_type == "portal":
            connected_rooms = node.get("connected_room_ids")
            if connected_rooms is None:
                connected_rooms = attributes.get("connected_room_ids")
            item["connects"] = [
                room_id
                for room_id in (_room_node_id(value) for value in connected_rooms or [])
                if room_id
            ]
            portals.append(item)
        else:
            room_id = _room_node_id(node.get("room_id"))
            if room_id:
                item["room_id"] = room_id
            containers.append(item)
    current_room = _current_room_id(rooms, robot_context)
    for room in rooms:
        room.pop("_centroid", None)
    result = {"rooms": rooms, "portals": portals, "containers": containers}
    if current_room:
        result["current_room"] = current_room
    return result


def compact_target_context(target_context: dict[str, Any]) -> dict[str, Any]:
    if not bool(target_context.get("enabled")):
        return {"mode": "explore_all"}
    labels = [str(value) for value in target_context.get("object_labels") or [] if value]
    target_name = str(
        target_context.get("target_name")
        or target_context.get("target_object")
        or target_context.get("object_label")
        or (labels[0] if labels else "unknown")
    )
    target = {"name": target_name, "visible": bool(target_context.get("visible", False))}
    if labels:
        target["labels"] = labels
    container_name = str(target_context.get("target_container_name") or "")
    if container_name:
        target["likely_container"] = container_name
    return {"mode": "object_goal", "target": target}


def compact_robot_context(
    graph: dict[str, Any], robot_context: dict[str, Any] | None
) -> dict[str, Any]:
    semantic_graph = compact_semantic_graph(graph, robot_context=robot_context)
    result = {}
    if semantic_graph.get("current_room"):
        result["current_room"] = semantic_graph["current_room"]
    return result


def compact_candidate(candidate: BehaviorCandidate) -> dict[str, Any]:
    behavior_type = str(candidate.behavior_type or "").upper()
    metadata = candidate.metadata or {}
    interaction = candidate.interaction_command or {}
    node_type = str(metadata.get("node_type") or "").casefold()
    if behavior_type == "INTERACT":
        action = str(interaction.get("action") or "open").casefold()
        effect = "access_room" if node_type == "portal" else "reveal_contents"
        subject_type = node_type or "interaction_object"
    elif behavior_type == "NAVIGATE":
        action = "navigate"
        effect = "approach_target" if bool(metadata.get("target_goal")) else "reach_location"
        subject_type = node_type or "target"
    else:
        action = "explore"
        effect = "reveal_space"
        subject_type = "frontier"
    result = {
        "id": candidate.candidate_id,
        "action": action,
        "subject_id": candidate.target_id,
        "subject_type": subject_type,
        "effect": effect,
        "distance_m": round(max(0.0, float(candidate.features.get("distance_m", 0.0))), 2),
    }
    if behavior_type != "EXPLORE" and candidate.target_name:
        result["subject_name"] = candidate.target_name
    room_id = _room_node_id(metadata.get("target_room_id") or metadata.get("room_id"))
    if room_id:
        result["room_id"] = room_id
    return result


def _candidate_room_id(candidate: BehaviorCandidate, graph: dict[str, Any]) -> str:
    metadata = candidate.metadata or {}
    room_id = _room_node_id(metadata.get("target_room_id") or metadata.get("room_id"))
    if room_id:
        return room_id
    goal = list(candidate.goal_xyyaw or [])
    if len(goal) < 2:
        return ""
    closest_id = ""
    closest_distance_sq = float("inf")
    for node in graph.get("nodes") or []:
        if str(node.get("type") or "").casefold() != "room":
            continue
        centroid = list(node.get("centroid") or [])
        if len(centroid) < 2:
            continue
        distance_sq = (float(centroid[0]) - float(goal[0])) ** 2 + (
            float(centroid[1]) - float(goal[1])
        ) ** 2
        if distance_sq < closest_distance_sq:
            closest_distance_sq = distance_sq
            closest_id = str(node.get("id") or "")
    return closest_id


def _candidate_execution_key(candidate: BehaviorCandidate) -> tuple[float, float, str]:
    return (
        -float(candidate.score),
        max(0.0, float(candidate.features.get("distance_m", 0.0))),
        candidate.candidate_id,
    )


def candidate_group_id(candidate: BehaviorCandidate, graph: dict[str, Any]) -> str:
    if str(candidate.behavior_type or "").upper() == "EXPLORE":
        room_id = _candidate_room_id(candidate, graph) or "unknown"
        return f"explore:{room_id}"
    return candidate.candidate_id


def _history_by_group(decision_history: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in decision_history or []:
        group_id = str(entry.get("group_id") or "")
        if group_id:
            result[group_id] = entry
    return result


def compact_candidate_groups(
    candidates: list[BehaviorCandidate],
    graph: dict[str, Any],
    decision_history: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[BehaviorCandidate]]]:
    grouped: dict[str, list[BehaviorCandidate]] = {}
    for candidate in candidates:
        group_id = candidate_group_id(candidate, graph)
        grouped.setdefault(group_id, []).append(candidate)
    history_lookup = _history_by_group(decision_history)
    projected = []
    for group_id, members in sorted(grouped.items()):
        members.sort(key=_candidate_execution_key)
        item = compact_candidate(members[0])
        item["id"] = group_id
        if str(members[0].behavior_type or "").upper() == "EXPLORE":
            room_id = group_id.removeprefix("explore:")
            item["subject_id"] = room_id
            item["subject_type"] = "room" if room_id != "unknown" else "unknown_space"
            item["member_count"] = len(members)
            frontier_cell_count = sum(
                max(0, int((member.metadata or {}).get("cell_count", 0) or 0))
                for member in members
            )
            map_resolution = max(
                [
                    max(0.0, float((member.metadata or {}).get("map_resolution", 0.0) or 0.0))
                    for member in members
                ]
                or [0.0]
            )
            item["frontier_cell_count"] = frontier_cell_count
            if map_resolution > 0.0:
                item["frontier_length_m"] = round(
                    frontier_cell_count * map_resolution, 2
                )
            history = history_lookup.get(group_id) or {}
            if history:
                item["history"] = {
                    "selection_count": int(history.get("selection_count", 0) or 0),
                    "consecutive_selection_count": int(
                        history.get("consecutive_selection_count", 0) or 0
                    ),
                    "last_selected_steps_ago": int(
                        history.get("last_selected_steps_ago", 0) or 0
                    ),
                    "last_result": str(history.get("last_result") or "UNKNOWN"),
                    "last_frontier_length_delta_m": round(
                        float(history.get("last_frontier_length_delta_m", 0.0) or 0.0),
                        2,
                    ),
                    "low_gain_repeat_count": int(
                        history.get("low_gain_repeat_count", 0) or 0
                    ),
                }
        projected.append(item)
    return projected, grouped


class ModelPolicyClient:
    def __init__(self, config: ModelPolicyConfig | None = None) -> None:
        self.config = config or ModelPolicyConfig()
        self.last_error = ""
        self.last_metrics: dict[str, Any] = {}
        self.last_result_source = "not_called"
        self.last_ranking_ids: list[str] = []
        self.last_selected_group_id = ""
        self.last_reason = ""
        self.last_confidence = ""
        self.last_candidate_groups: list[dict[str, Any]] = []
        self._mllm_client = MLLMClient(
            MLLMClientConfig(
                mode=self.config.mode,
                command=self.config.command,
                endpoint=self.config.endpoint,
                api_key_env=self.config.api_key_env,
                model=self.config.model,
                protocol=self.config.protocol,
                timeout_s=self.config.timeout_s,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                reasoning_effort=self.config.reasoning_effort,
                image_detail=self.config.image_detail,
                metrics_path=self.config.metrics_path,
            )
        )

    def select(
        self,
        candidates: Iterable[BehaviorCandidate],
        target_context: dict[str, Any] | None = None,
        graph: dict[str, Any] | None = None,
        robot_context: dict[str, Any] | None = None,
        metrics_context: dict[str, Any] | None = None,
    ) -> BehaviorCandidate | None:
        candidates = list(candidates)
        self.last_ranking_ids = []
        self.last_selected_group_id = ""
        self.last_reason = ""
        self.last_confidence = ""
        if not candidates:
            return None
        payload = self.build_request(
            candidates,
            target_context or {},
            graph or {},
            robot_context or {},
        )
        response = self._request(payload, metrics_context=metrics_context)
        if response is None:
            return None
        decision_history = list((robot_context or {}).get("group_history") or [])
        _, candidate_groups = compact_candidate_groups(
            candidates, graph or {}, decision_history=decision_history
        )
        try:
            response = validate_subgoal_selection(response, set(candidate_groups))
        except (TypeError, ValueError) as exc:
            self.last_error = f"invalid_model_selection: {exc}"
            self.last_result_source = "rule_fallback_invalid_response"
            return None
        selected_id = response["candidate_id"]
        self.last_ranking_ids = list(response.get("ranked_ids") or [selected_id])
        self.last_selected_group_id = selected_id
        self.last_reason = str(response.get("reason") or "")
        self.last_confidence = str(response.get("confidence") or "")
        scores = response["scores"]
        for candidate in candidates:
            if candidate.candidate_id in scores:
                candidate.score = float(scores[candidate.candidate_id])
                candidate.score_terms = {"model_score": candidate.score}
        self.last_result_source = "model"
        return candidate_groups[selected_id][0]

    def build_request(
        self,
        candidates: list[BehaviorCandidate],
        target_context: dict[str, Any],
        graph: dict[str, Any],
        robot_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decision_history = list((robot_context or {}).get("group_history") or [])
        recent_decisions = list((robot_context or {}).get("decision_history") or [])
        compact_candidates, _ = compact_candidate_groups(
            candidates, graph, decision_history=decision_history
        )
        self.last_candidate_groups = list(compact_candidates)
        return {
            "schema_version": 2,
            "instruction": (
                "Rank candidate actions by semantic relevance to the mission. All candidates are "
                "already feasible. Use graph semantics and distance, but do not infer geometry, "
                "navigation cost, information gain, or manipulation difficulty. For object goals, "
                "prefer actions likely to reveal or reach the target. For exploration, prefer actions "
                "that reveal new rooms or container contents. Avoid selecting the same room again when "
                "its recent attempts had low frontier shrink; prefer a different room unless the repeated "
                "room has the largest remaining frontier, is the only reachable option, or is required by "
                "the target. Use frontier_length_m as the room-level unknown-boundary signal and history "
                "only as a repetition/observed-gain signal. Return exactly one compact JSON object "
                "with ranked_ids containing at most three IDs, reason set to NEW_ROOM, "
                "REVEAL_CONTENTS, TARGET_RELEVANCE, DISTANCE_TIEBREAK, or "
                "NO_SEMANTIC_PREFERENCE, and confidence set to low, medium, or high. "
                "Do not return prose, scores, markdown, or additional keys."
            ),
            "mission": compact_target_context(target_context),
            "robot": compact_robot_context(graph, robot_context),
            "recent_decisions": recent_decisions[-8:],
            "graph": compact_semantic_graph(
                graph,
                robot_context=robot_context,
                max_nodes=self.config.max_graph_nodes,
            ),
            "candidates": compact_candidates,
        }

    def _request(
        self, payload: dict[str, Any], metrics_context: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        self.last_error = ""
        self.last_metrics = {}
        self.last_result_source = "rule_fallback"
        mode = str(self.config.mode or "disabled").casefold()
        if mode == "disabled":
            self.last_error = "model_disabled"
            return None
        if mode == "mock":
            candidates = payload.get("candidates") or []
            ranked = sorted(
                candidates,
                key=lambda item: (
                    0 if item.get("effect") == "approach_target" else 1,
                    float(item.get("distance_m", 0.0)),
                    str(item.get("id", "")),
                ),
            )
            self.last_result_source = "model_mock"
            return {"ranked_ids": [ranked[0].get("id")]} if ranked else None
        try:
            if mode == "command":
                return self._request_command(payload)
            if mode == "http":
                return self._request_http(payload, metrics_context=metrics_context)
            raise ValueError(f"unsupported model policy mode: {self.config.mode}")
        except (OSError, ValueError, subprocess.SubprocessError, TimeoutError) as exc:
            self.last_error = str(exc)
            return None

    def _request_command(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = shlex.split(self.config.command)
        if not command:
            raise ValueError("model command is empty")
        completed = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=self.config.timeout_s,
            check=True,
        )
        response = json.loads(completed.stdout)
        if not isinstance(response, dict):
            raise ValueError("model command response must be a JSON object")
        return response

    def _request_http(
        self, payload: dict[str, Any], metrics_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        response = self._mllm_client.request_json(
            role="subgoal_selection",
            instruction=str(payload.get("instruction") or ""),
            context={
                "mission": payload.get("mission") or {},
                "robot": payload.get("robot") or {},
                "graph": payload.get("graph") or {},
                "candidates": payload.get("candidates") or [],
                "recent_decisions": payload.get("recent_decisions") or [],
            },
            timeout_s=self.config.timeout_s,
            max_tokens=self.config.max_tokens,
            metrics_context=metrics_context,
        )
        self.last_metrics = response.metrics()
        if response.error:
            raise ValueError(response.error)
        if not isinstance(response.payload, dict):
            raise ValueError("model HTTP response must be a JSON object")
        return response.payload
