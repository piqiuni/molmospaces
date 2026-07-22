from __future__ import annotations

from dataclasses import dataclass
import json
import shlex
import subprocess
from typing import Any, Iterable
from .behavior_candidates import BehaviorCandidate
from semantic_mllm_py_pkg.client import MLLMClient, MLLMClientConfig


@dataclass
class ModelPolicyConfig:
    mode: str = "disabled"
    command: str = ""
    endpoint: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = ""
    protocol: str = "openai_responses"
    timeout_s: float = 20.0
    max_graph_nodes: int = 80
    max_graph_edges: int = 160
    metrics_path: str = ""


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


class ModelPolicyClient:
    def __init__(self, config: ModelPolicyConfig | None = None) -> None:
        self.config = config or ModelPolicyConfig()
        self.last_error = ""
        self._mllm_client = MLLMClient(
            MLLMClientConfig(
                mode=self.config.mode,
                command=self.config.command,
                endpoint=self.config.endpoint,
                api_key_env=self.config.api_key_env,
                model=self.config.model,
                protocol=self.config.protocol,
                timeout_s=self.config.timeout_s,
                metrics_path=self.config.metrics_path,
            )
        )

    def select(
        self,
        candidates: Iterable[BehaviorCandidate],
        target_context: dict[str, Any] | None = None,
        graph: dict[str, Any] | None = None,
        robot_context: dict[str, Any] | None = None,
    ) -> BehaviorCandidate | None:
        candidates = list(candidates)
        if not candidates:
            return None
        payload = self.build_request(
            candidates,
            target_context or {},
            graph or {},
            robot_context or {},
        )
        response = self._request(payload)
        if response is None:
            return None
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        selected_id = str(response.get("candidate_id") or "")
        scores = response.get("scores") or {}
        for candidate in candidates:
            if candidate.candidate_id in scores:
                candidate.score = float(scores[candidate.candidate_id])
                candidate.score_terms = {"model_score": candidate.score}
        if selected_id in candidate_by_id:
            return candidate_by_id[selected_id]
        ranked = sorted(candidates, key=lambda item: (-item.score, item.candidate_id))
        return ranked[0] if ranked else None

    def build_request(
        self,
        candidates: list[BehaviorCandidate],
        target_context: dict[str, Any],
        graph: dict[str, Any],
        robot_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "instruction": (
                "Select one candidate behavior for interactive navigation. "
                "Prefer actions that make the target reachable while accounting for distance, "
                "visibility gain, interaction cost, and current state. Return candidate_id or scores."
            ),
            "target_context": dict(target_context),
            "robot_context": dict(robot_context or {}),
            "graph": compact_graph(
                graph,
                max_nodes=self.config.max_graph_nodes,
                max_edges=self.config.max_graph_edges,
            ),
            "candidates": [candidate.to_dict() for candidate in candidates],
        }

    def _request(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        self.last_error = ""
        mode = str(self.config.mode or "disabled").casefold()
        if mode == "disabled":
            return None
        if mode == "mock":
            candidates = payload.get("candidates") or []
            ranked = sorted(
                candidates,
                key=lambda item: (
                    -float((item.get("features") or {}).get("target_relevance", 0.0)),
                    -float(item.get("score", 0.0)),
                    float((item.get("features") or {}).get("distance_m", 0.0)),
                    str(item.get("candidate_id", "")),
                ),
            )
            return {"candidate_id": ranked[0].get("candidate_id")} if ranked else None
        try:
            if mode == "command":
                return self._request_command(payload)
            if mode == "http":
                return self._request_http(payload)
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

    def _request_http(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._mllm_client.request_json(
            role="subgoal_selection",
            instruction=str(payload.get("instruction") or ""),
            context={
                "target_context": payload.get("target_context") or {},
                "robot_context": payload.get("robot_context") or {},
                "graph": payload.get("graph") or {},
                "candidates": payload.get("candidates") or [],
            },
        )
        if response.error:
            raise ValueError(response.error)
        if not isinstance(response.payload, dict):
            raise ValueError("model HTTP response must be a JSON object")
        return response.payload
