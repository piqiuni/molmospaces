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
    max_tokens: int = 128
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


class ModelPolicyClient:
    def __init__(self, config: ModelPolicyConfig | None = None) -> None:
        self.config = config or ModelPolicyConfig()
        self.last_error = ""
        self.last_metrics: dict[str, Any] = {}
        self.last_result_source = "not_called"
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
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        try:
            response = validate_subgoal_selection(response, set(candidate_by_id))
        except (TypeError, ValueError) as exc:
            self.last_error = f"invalid_model_selection: {exc}"
            self.last_result_source = "rule_fallback_invalid_response"
            return None
        selected_id = response["candidate_id"]
        scores = response["scores"]
        for candidate in candidates:
            if candidate.candidate_id in scores:
                candidate.score = float(scores[candidate.candidate_id])
                candidate.score_terms = {"model_score": candidate.score}
        self.last_result_source = "model"
        return candidate_by_id[selected_id]

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
                    -float((item.get("features") or {}).get("target_relevance", 0.0)),
                    -float(item.get("score", 0.0)),
                    float((item.get("features") or {}).get("distance_m", 0.0)),
                    str(item.get("candidate_id", "")),
                ),
            )
            self.last_result_source = "model_mock"
            return {"candidate_id": ranked[0].get("candidate_id"), "scores": {}} if ranked else None
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
                "target_context": payload.get("target_context") or {},
                "robot_context": payload.get("robot_context") or {},
                "graph": payload.get("graph") or {},
                "candidates": payload.get("candidates") or [],
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
