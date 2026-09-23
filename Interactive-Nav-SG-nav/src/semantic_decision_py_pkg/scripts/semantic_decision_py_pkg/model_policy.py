from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any, Iterable
from .behavior_candidates import BehaviorCandidate
from .candidate_curator import candidate_history_key
from .room_context import room_context_for_xy
from semantic_mllm_py_pkg.client import MLLMClient, MLLMClientConfig
from semantic_mllm_py_pkg.schemas import validate_subgoal_selection


ROOM_ANCHOR_LABELS = {
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


# This context is deliberately compact: it is sent alongside the regular
# candidate payload in the existing request, rather than triggering a second
# model call for room classification.
ROOM_OBJECT_REASONING_MAX_ROOMS = 12
ROOM_OBJECT_REASONING_MAX_PORTALS = 12
ROOM_OBJECT_REASONING_MAX_CONTAINERS = 12
ROOM_OBJECT_REASONING_MAX_ANCHORS_PER_ROOM = 4
ROOM_OBJECT_REASONING_MAX_LABELS = 6

# The M2 protocol is intentionally a small categorical response.  Keeping the
# vocabulary in one place lets the request schema and the prompt stay aligned.
SUBGOAL_REASON_CODES = (
    "TARGET_VISIBLE",
    "REVEAL_TARGET_CONTAINER",
    "UNLOCK_ROUTE",
    "EXPLORE_TARGET_ROOM",
    "INFORMATION_GAIN",
    "INTERACTION_COVERAGE",
    "RECOVERY_DIVERSIFICATION",
    "DISTANCE_TIEBREAK",
    "NO_SEMANTIC_PREFERENCE",
)
SUBGOAL_CONFIDENCE_CODES = ("low", "medium", "high")


def build_subgoal_selection_response_schema(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the strict OpenAI JSON-schema descriptor for an M2 response.

    Candidate IDs are copied from the compact, current request only.  This
    makes an echoed historical graph or mission unable to satisfy structured
    decoding.  An empty candidate pool cannot produce an executable M2 result,
    so fail before constructing an unconstrained schema.
    """

    candidate_ids: list[str] = []
    for candidate in payload.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("id") or "").strip()
        if candidate_id and candidate_id not in candidate_ids:
            candidate_ids.append(candidate_id)
    if not candidate_ids:
        raise ValueError("subgoal response schema requires at least one candidate ID")
    ranked_items: dict[str, Any] = {
        "type": "string",
        "enum": candidate_ids,
    }
    return {
        "name": "subgoal_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ranked_ids": {
                    "type": "array",
                    "items": ranked_items,
                    "minItems": 1,
                    "maxItems": 3,
                },
                "reason": {
                    "type": "string",
                    "enum": list(SUBGOAL_REASON_CODES),
                },
                "confidence": {
                    "type": "string",
                    "enum": list(SUBGOAL_CONFIDENCE_CODES),
                },
            },
            "required": ["ranked_ids", "reason", "confidence"],
        },
    }

# Semantic priors guide the model toward observed rooms and containers, but
# must never be treated as proof that an unobserved object is present there.
TARGET_ROOM_OBJECT_PRIORS = (
    {
        "semantic_class": "food",
        "terms": frozenset(
            {
                "apple",
                "banana",
                "bread",
                "cake",
                "carrot",
                "cheese",
                "drink",
                "egg",
                "food",
                "fruit",
                "irishpotato",
                "juice",
                "milk",
                "orange",
                "potato",
                "snack",
                "tomato",
                "vegetable",
                "water",
            }
        ),
        "room_types": ("kitchen", "dining_room"),
        "container_types": (
            "refrigerator",
            "fridge",
            "cabinet",
            "pantry",
            "cupboard",
        ),
    },
    {
        "semantic_class": "bedside_personal_item",
        "terms": frozenset(
            {
                "alarm_clock",
                "book",
                "charger",
                "glasses",
                "key",
                "keys",
                "phone",
                "remote",
                "wallet",
            }
        ),
        "room_types": ("bedroom", "office", "living_room"),
        "container_types": ("dresser", "drawer", "desk", "nightstand"),
    },
)


@dataclass
class ModelPolicyConfig:
    mode: str = "disabled"
    command: str = ""
    endpoint: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "qwen3.6-35b-a3b"
    protocol: str = "openai_chat"
    timeout_s: float = 3.0
    # M2 retries only requests that ended in an explicit timeout.  Schema,
    # authentication, HTTP and other model errors return immediately so a bad
    # request cannot be amplified into repeated traffic.
    timeout_retry_count: int = 0
    timeout_retry_backoff_s: float = 1.0
    temperature: float = 0.0
    max_tokens: int = 256
    context_window_tokens: int = 16384
    reasoning_effort: str = "off"
    image_detail: str = "low"
    max_graph_nodes: int = 80
    max_graph_edges: int = 160
    metrics_path: str = ""
    selection_granularity: str = "candidate"
    history_region_size_m: float = 1.0
    include_pre_scores: bool = False
    context_profile: str = "public_facts_v2"
    prompt_file: str = ""
    recent_decision_limit: int = 30
    # A model may still overlook an observed route-critical portal.  The guard
    # only applies to deterministic high-priority hints and leaves ordinary
    # semantic/exploration ranking to the MLLM.
    pre_score_guard_margin: float = 0.75


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
        if not is_model_timeout_error(self.last_error):
            self.consecutive_timeouts = 0
            return False
        self.consecutive_timeouts += 1
        if self.consecutive_timeouts < max(1, int(self.consecutive_timeout_limit)):
            return False
        current = float(now if now is not None else time.monotonic())
        self.open_until = current + max(0.0, float(self.cooldown_s))
        self.consecutive_timeouts = 0
        return True


_MODEL_TIMEOUT_MARKERS = (
    "timed out",
    "timeout",
    "deadline exceeded",
    "http 408",
    "http 504",
)


def is_model_timeout_error(error: BaseException | str) -> bool:
    """Return whether an M2 failure is safe for the timeout-only retry path."""

    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        return True
    message = str(error or "").casefold()
    return any(marker in message for marker in _MODEL_TIMEOUT_MARKERS)


_PUBLIC_DOOR_CONTEXT_ID_RE = re.compile(r"^door_(?:[0-9]{1,8}|x)$")
_PUBLIC_DOOR_NODE_ID_RE = re.compile(r"^portal_(door_(?:[0-9]{1,8}|x))$")
_PRIVATE_PORTAL_ID_TOKEN_RE = re.compile(
    r"(?:^|_)(?:door|doorframe|doorway|gt|mujoco|private)(?:_|$)"
)


def _portal_context_id(node: dict[str, Any], ordinal: int) -> str:
    """Return a generic door ID for MLLM-facing portal context.

    The live graph supplies ``door_<ordinal>`` identities.  This small
    defense-in-depth normalizer also prevents an older producer's
    ``portal_doorframe_*`` or ``portal_gt_*`` node ID from reaching a prompt.
    """

    attributes = node.get("attributes") or {}
    public_instance_id = str(attributes.get("instance_id") or "")
    if _PUBLIC_DOOR_CONTEXT_ID_RE.fullmatch(public_instance_id.casefold()):
        return public_instance_id
    node_id = str(node.get("id") or "")
    if _PUBLIC_DOOR_CONTEXT_ID_RE.fullmatch(node_id.casefold()):
        return node_id
    match = _PUBLIC_DOOR_NODE_ID_RE.fullmatch(node_id.casefold())
    if match:
        return match.group(1)
    # Existing public graph producers may already use a stable non-descriptive
    # portal ID (for example ``portal_1``).  Preserve it for reference
    # continuity; only rewrite IDs carrying a clear private/GT token.
    if node_id and not _PRIVATE_PORTAL_ID_TOKEN_RE.search(node_id.casefold()):
        return node_id
    return f"door_{int(ordinal) + 1:04d}"


def _public_portal_context_ids(nodes: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(node.get("id") or ""): _portal_context_id(node, ordinal)
        for ordinal, node in enumerate(nodes)
        if str(node.get("type") or "").casefold() == "portal"
    }


def compact_graph(graph: dict[str, Any], max_nodes: int = 80, max_edges: int = 160) -> dict[str, Any]:
    graph_nodes = list(graph.get("nodes") or [])[: max(0, int(max_nodes))]
    public_portal_ids = _public_portal_context_ids(graph_nodes)
    nodes = []
    for node in graph_nodes:
        attributes = node.get("attributes") or {}
        interaction = node.get("interaction") or {}
        node_type = str(node.get("type") or "")
        public_door_id = public_portal_ids.get(str(node.get("id") or ""), "")
        # Keep a portal's generic door reference, never its simulator/asset
        # label.  This remains a defense-in-depth boundary if an older graph
        # producer accidentally sends ``doorframe`` or a source-derived name.
        public_label = (
            public_door_id
            if node_type == "portal"
            and _PUBLIC_DOOR_CONTEXT_ID_RE.fullmatch(str(public_door_id).casefold())
            else "portal"
            if node_type == "portal"
            else node.get("label")
        )
        public_name = public_label if node_type == "portal" else node.get("name")
        public_attributes = {
            key: attributes[key]
            for key in (
                "active", "is_potential_room", "observed_free_space",
                "room_attribute", "room_attribute_confidence", "room_attribute_source",
                "last_observation_frame_index", "max_visible_pixels",
            ) if key in attributes
        }
        source_portal_id = str(attributes.get("source_portal_id") or "")
        if source_portal_id in public_portal_ids:
            public_attributes["source_portal_id"] = public_portal_ids[source_portal_id]
        elif source_portal_id and not _PRIVATE_PORTAL_ID_TOKEN_RE.search(source_portal_id.casefold()):
            public_attributes["source_portal_id"] = source_portal_id
        nodes.append(
            {
                "id": public_door_id or node.get("id"),
                "type": node_type,
                "label": public_label,
                "name": public_name,
                "centroid": list(node.get("centroid") or [])[:3],
                "aabb_center": list(node.get("aabb_center") or [])[:3],
                "aabb_size": list(node.get("aabb_size") or [])[:3],
                "room_id": node.get("room_id"),
                "confidence": node.get("confidence"),
                "observation_count": node.get("observation_count"),
                "last_seen": node.get("last_seen"),
                "attributes": public_attributes,
                "is_currently_visible": bool(node.get("is_currently_visible")),
                "state_age_sec": node.get("state_age_sec", 0.0),
                "connected_room_ids": list(attributes.get("connected_room_ids") or []),
                "interaction_state": interaction.get("state"),
                "requires_interaction": interaction.get("requires_interaction"),
                "traversable": interaction.get("traversable"),
                "is_interactable": interaction.get("is_interactable"),
                "interaction_capability": interaction.get("capability"),
                "interaction_capability_source": interaction.get("capability_source"),
                "interaction_failure_reason": interaction.get("failure_reason"),
            }
        )
    edges = []
    for edge in list(graph.get("edges") or [])[: max(0, int(max_edges))]:
        attributes = dict(edge.get("attributes") or {})
        for key in ("portal_node_id", "source_portal_id"):
            raw_value = str(attributes.get(key) or "")
            if raw_value in public_portal_ids:
                attributes[key] = public_portal_ids[raw_value]
        edges.append(
            {
                "src_id": public_portal_ids.get(
                    str(edge.get("src_id") or ""), edge.get("src_id")
                ),
                "relation": edge.get("relation"),
                "dst_id": public_portal_ids.get(
                    str(edge.get("dst_id") or ""), edge.get("dst_id")
                ),
                "attributes": attributes,
            }
        )
    result = {
        "scene_id": graph.get("scene_id", ""),
        "episode_id": graph.get("episode_id", ""),
        "graph_revision": graph.get("graph_revision", 0),
        "nodes": nodes,
        "edges": edges,
    }
    for key in ("frame_id", "units", "capture_step", "timestamp"):
        if key in graph:
            result[key] = graph[key]
    return result


def public_decision_graph_context(graph: dict[str, Any]) -> dict[str, Any]:
    """Internal public graph snapshot; M2 applies its structural projection later."""
    return compact_graph(
        graph, max_nodes=len(graph.get("nodes") or []),
        max_edges=len(graph.get("edges") or []),
    )


def _room_node_id(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    return text if text.startswith("room_") else f"room_{text}"


def _node_interaction(node: dict[str, Any]) -> dict[str, Any]:
    interaction = dict(node.get("interaction") or {})
    compact_keys = {
        "state": "interaction_state",
        "requires_interaction": "requires_interaction",
        "traversable": "traversable",
        "is_interactable": "is_interactable",
        "capability": "interaction_capability",
        "capability_source": "interaction_capability_source",
        "failure_reason": "interaction_failure_reason",
    }
    for key, compact_key in compact_keys.items():
        if key not in interaction and compact_key in node:
            interaction[key] = node.get(compact_key)
    return interaction


def _current_room_id(
    graph: dict[str, Any], robot_context: dict[str, Any] | None
) -> str:
    # Resolve on the full graph with the candidate generator's containment
    # rule; prompt truncation and nearby centroids must not change the room.
    robot_context = robot_context or {}
    uses_graph_pose = "robot_graph_xy" in robot_context
    xy = robot_context.get("robot_graph_xy") if uses_graph_pose else robot_context.get("robot_xy")
    position_frame = robot_context.get("robot_graph_frame_id") if uses_graph_pose else robot_context.get("position_frame_id")
    graph_frame = graph.get("frame_id")
    if position_frame and graph_frame and str(position_frame).lstrip("/") != str(graph_frame).lstrip("/"):
        return ""
    context = room_context_for_xy(graph, xy)
    return _room_node_id(context.get("room_id"))


def _public_xy(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        xy = [float(value[0]), float(value[1])]
    except (ValueError, TypeError):
        return None
    return [round(item, 4) for item in xy] if all(map(math.isfinite, xy)) else None


def compact_semantic_graph(
    graph: dict[str, Any],
    robot_context: dict[str, Any] | None = None,
    max_nodes: int = 80,
) -> dict[str, Any]:
    rooms = []
    portals = []
    containers = []
    anchors_by_room: dict[str, list[dict[str, Any]]] = {}
    unassigned_anchors = []
    # All topology/interaction nodes survive; ordinary objects contribute only
    # bounded room names, never consume the old generic node budget.
    graph_nodes = list(graph.get("nodes") or [])
    robot_context = robot_context or {}
    public_portal_ids = _public_portal_context_ids(graph_nodes)
    for node in graph_nodes:
        node_type = str(node.get("type") or "").casefold()
        if node_type == "room" and not bool(
            (node.get("attributes") or {}).get("active", True)
        ):
            continue
        if node_type in {"scene", "room", "portal"}:
            continue
        label = str(node.get("label") or node.get("name") or node_type).strip()
        normalized_label = label.casefold().replace(" ", "_")
        if normalized_label not in ROOM_ANCHOR_LABELS:
            continue
        size = list(node.get("aabb_size") or [])
        footprint_m2 = (
            abs(float(size[0]) * float(size[1])) if len(size) >= 2 else 0.0
        )
        anchor = {
            "type": label or node_type,
            "_rank": (
                footprint_m2,
                float(node.get("confidence", 0.0) or 0.0),
            ),
        }
        room_id = _room_node_id(node.get("room_id"))
        if room_id:
            anchors_by_room.setdefault(room_id, []).append(anchor)
        else:
            anchor["id"] = str(node.get("id") or "")
            anchor["observed_xy"] = _public_xy(node.get("centroid"))
            evidence = node.get("attributes") or {}
            for key in (
                "observation_count", "last_seen", "last_seen_step", "last_seen_frame",
                "last_observation_frame_index", "max_visible_pixels", "confidence",
            ):
                if key in node or key in evidence:
                    anchor[key] = node.get(key, evidence.get(key))
            unassigned_anchors.append(anchor)
    for node in graph_nodes:
        node_type = str(node.get("type") or "").casefold()
        if node_type == "room" and not bool(
            (node.get("attributes") or {}).get("active", True)
        ):
            continue
        if node_type not in {"room", "portal", "container"}:
            continue
        attributes = node.get("attributes") or {}
        interaction = _node_interaction(node)
        raw_node_id = str(node.get("id") or "")
        node_id = public_portal_ids.get(raw_node_id, raw_node_id)
        semantic_name = (
            "door"
            if node_type == "portal"
            and _PUBLIC_DOOR_CONTEXT_ID_RE.fullmatch(node_id.casefold())
            else "portal"
            if node_type == "portal"
            else str(node.get("label") or node.get("name") or node_id)
        )
        if node_type == "room":
            inferred_attribute = str(attributes.get("room_attribute") or "").strip()
            inferred_known = bool(
                inferred_attribute and inferred_attribute.casefold() != "unknown"
            )
            room = {
                "id": node_id,
                "type": inferred_attribute if inferred_known else semantic_name or "unknown",
                "centroid_xy": _public_xy(node.get("centroid")),
                "aabb_center_xy": _public_xy(node.get("aabb_center")),
                "aabb_size_xy": _public_xy(node.get("aabb_size")),
            }
            for source_key, output_key in (
                ("room_frontier_lengths", "eligible_frontier_length_m"),
                ("observed_room_frontier_lengths", "observed_frontier_length_m"),
                ("room_frontier_counts", "eligible_frontier_count"),
                ("observed_room_frontier_counts", "observed_frontier_count"),
            ):
                values = robot_context.get(source_key) or {}
                value = values.get(node_id, values.get(node.get("room_id")))
                room[output_key] = (
                    int(value) if output_key.endswith("_count")
                    else round(float(value), 3)
                ) if value is not None else None
            source = attributes.get("room_attribute_source")
            if source:
                room["room_attribute_source"] = str(source)
            if bool(attributes.get("is_potential_room")):
                # This is a graph-only room created after a successful portal
                # open.  It carries topology for exploration but deliberately
                # has no occupancy-derived free-space evidence yet.
                room["potential_room"] = True
                room["observed_free_space"] = bool(
                    attributes.get("observed_free_space", False)
                )
                source_portal_id = public_portal_ids.get(
                    str(attributes.get("source_portal_id") or ""),
                    str(attributes.get("source_portal_id") or ""),
                )
                if source_portal_id:
                    room["source_portal_id"] = source_portal_id
            if inferred_known:
                if semantic_name and semantic_name.casefold() != inferred_attribute.casefold():
                    room["observed_type"] = semantic_name
                room["room_attribute_confidence"] = round(
                    float(attributes.get("room_attribute_confidence", 0.0) or 0.0),
                    2,
                )
            room_anchors = sorted(
                anchors_by_room.get(node_id, []),
                key=lambda item: (-item["_rank"][0], -item["_rank"][1], item["type"]),
            )
            if room_anchors:
                room["anchor_objects"] = [
                    {"type": name} for name in list(dict.fromkeys(
                        anchor["type"] for anchor in room_anchors
                    ))[:6]
                ]
            rooms.append(room)
            continue
        item = {
            "id": node_id,
            "type": semantic_name or node_type,
            "state": str(interaction.get("state") or "unknown"),
            "interaction_available": bool(
                interaction.get("requires_interaction")
                or interaction.get("is_interactable")
            ),
            "center_xy": _public_xy(node.get("centroid")),
        }
        capability = str(interaction.get("capability") or "").strip()
        if capability and capability.casefold() != "unknown":
            item["capability"] = capability
        failure_reason = str(interaction.get("failure_reason") or "").strip()
        if failure_reason:
            item["failure_reason"] = failure_reason
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
    current_room = _current_room_id(graph, robot_context)
    result = {"rooms": rooms, "portals": portals, "containers": containers}
    for key in ("frame_id", "units", "capture_step", "graph_revision", "timestamp"):
        if key in graph:
            result[key] = graph[key]
    frontier_statistics = (
        robot_context.get("frontier_statistics_source")
        or robot_context.get("room_frontier_statistics")
    )
    if frontier_statistics:
        result["frontier_statistics"] = dict(frontier_statistics)
    if unassigned_anchors:
        result["remembered_objects_without_room"] = [
            {key: value for key, value in anchor.items() if key != "_rank"}
            for anchor in sorted(
                unassigned_anchors,
                key=lambda item: (-item["_rank"][0], -item["_rank"][1], item["type"]),
            )[:6]
        ]
    if current_room:
        result["current_room"] = current_room
    return result


def compact_target_context(target_context: dict[str, Any]) -> dict[str, Any]:
    if not bool(target_context.get("enabled")):
        # A target-disabled episode is still an interaction-exploration
        # mission, not a generic frontier-only walk.  This explicit mode keeps
        # M2 from suppressing untried viable containers behind an arbitrary
        # object-goal prompt.
        return {"mode": "interaction_coverage_exploration"}
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
    current_room = _current_room_id(graph, robot_context)
    robot_context = robot_context or {}
    result = {
        "current_xy": _public_xy(robot_context.get("robot_xy")),
        "initial_xy": _public_xy(robot_context.get("initial_xy")),
        "position_frame_id": robot_context.get("position_frame_id") or None,
    }
    if robot_context.get("initial_position_source"):
        result["initial_position_source"] = robot_context["initial_position_source"]
    if "robot_graph_xy" in robot_context:
        result["current_xy"] = _public_xy(robot_context.get("robot_graph_xy"))
        result["position_frame_id"] = robot_context.get("robot_graph_frame_id") or None
        result["pose_source"] = dict(robot_context.get("robot_graph_pose_source") or {})
    if current_room:
        result["current_room"] = current_room
    visits = []
    for raw in robot_context.get("room_visit_history") or []:
        entry = raw if isinstance(raw, dict) else {"room_id": raw}
        room_id = _room_node_id(entry.get("room_id"))
        if not room_id:
            continue
        visit = {"room_id": room_id}
        for key in ("entry_step", "source", "timestamp"):
            if key in entry:
                visit[key] = entry[key]
        if "entry_xy" in entry:
            visit["entry_xy"] = _public_xy(entry["entry_xy"])
        visits.append(visit)
    result["room_visit_history"] = visits
    return result


def historical_compat_context(
    graph: dict[str, Any], robot_context: dict[str, Any] | None,
    *, max_nodes: int = 80,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Historical public representation, retaining corrected pose/room facts.

    The old producer truncated before M2 and omitted geometry/room attributes.
    Reproduce that projection locally, never downgrade the shared live graph.
    This is compatibility, not a claim to recreate an old execution or TF bug.
    """
    robot_context = robot_context or {}
    nodes = []
    anchors_by_room: dict[str, list[dict[str, Any]]] = {}
    unassigned = []
    for node in list(graph.get("nodes") or [])[: max(0, int(max_nodes))]:
        attributes = node.get("attributes") or {}
        if str(node.get("type") or "").casefold() == "room" and not attributes.get("active", True):
            continue
        projected = {
            key: node[key] for key in (
                "id", "type", "label", "name", "centroid", "room_id",
                "is_currently_visible", "state_age_sec", "connected_room_ids",
                "interaction_state", "requires_interaction", "traversable",
                "is_interactable", "interaction_capability",
                "interaction_capability_source", "interaction_failure_reason",
            ) if key in node
        }
        # Accept both the live public producer's flat form and raw public test
        # fixtures without dropping a known interaction state on re-projection.
        projected["interaction"] = _node_interaction(node)
        if "connected_room_ids" not in projected and "connected_room_ids" in attributes:
            projected["connected_room_ids"] = list(attributes["connected_room_ids"])
        safety_attributes = {
            key: attributes[key] for key in (
                "is_potential_room", "observed_free_space", "source_portal_id",
            ) if key in attributes
        }
        if safety_attributes:
            projected["attributes"] = safety_attributes
        nodes.append(projected)
        node_type = str(node.get("type") or "").casefold()
        label = str(node.get("label") or node.get("name") or node_type).strip()
        if node_type in {"scene", "room", "portal"} or label.casefold().replace(" ", "_") not in ROOM_ANCHOR_LABELS:
            continue
        anchor = {"type": label, "visible": bool(node.get("is_currently_visible"))}
        room_id = _room_node_id(node.get("room_id"))
        if room_id:
            anchors_by_room.setdefault(room_id, []).append(anchor)
        else:
            unassigned.append(anchor)
    semantic = compact_semantic_graph({"nodes": nodes}, robot_context={})
    for room in semantic["rooms"]:
        for key in (
            "centroid_xy", "aabb_center_xy", "aabb_size_xy",
            "eligible_frontier_length_m", "observed_frontier_length_m",
            "eligible_frontier_count", "observed_frontier_count", "anchor_objects",
        ):
            room.pop(key, None)
        anchors = sorted(anchors_by_room.get(room["id"], []), key=lambda item: item["type"])[:6]
        if anchors:
            room["anchor_objects"] = anchors
    for item in semantic["portals"] + semantic["containers"]:
        item.pop("center_xy", None)
    semantic.pop("remembered_objects_without_room", None)
    if unassigned:
        semantic["unassigned_anchor_objects"] = sorted(unassigned, key=lambda item: item["type"])[:6]
    robot = {}
    current_room = _current_room_id(graph, robot_context)
    if current_room:
        semantic["current_room"] = current_room
        robot["current_room"] = current_room
    entered = sorted({
        _room_node_id(value) for value in robot_context.get("entered_room_ids") or []
        if _room_node_id(value)
    })[:12]
    if entered:
        robot["entered_rooms"] = entered
    return semantic, robot


def _restore_historical_candidate_context(
    options: list[dict[str, Any]], candidates: list[BehaviorCandidate],
    robot_context: dict[str, Any],
) -> None:
    """Restore model-facing legacy priors only; never mutate execution inputs."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    hints = robot_context.get("candidate_decision_hints") or {}
    for option in options:
        candidate = by_id.get(option["id"])
        if candidate is None:
            continue
        metadata = candidate.metadata or {}
        if str(candidate.behavior_type).upper() == "EXPLORE":
            if metadata.get("room_status"):
                option["room_status"] = str(metadata["room_status"])
            if "room_target_affinity" in metadata:
                option["room_target_affinity"] = round(float(metadata["room_target_affinity"] or 0.0), 2)
            if metadata.get("room_target_affinity_reason"):
                option["room_target_affinity_reason"] = str(metadata["room_target_affinity_reason"])
        for public, raw in zip(option.get("nearby_semantic_nodes") or [], metadata.get("nearby_semantic_nodes") or []):
            if "visible" in raw:
                public["visible"] = bool(raw["visible"])
        if hints.get(candidate.candidate_id):
            option["decision_hint"] = str(hints[candidate.candidate_id])


def _normalized_semantic_terms(values: Iterable[Any]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        text = str(value or "").casefold().replace("_", " ").replace("-", " ")
        normalized = " ".join(text.split())
        if not normalized:
            continue
        terms.add(normalized)
        terms.add(normalized.replace(" ", ""))
        terms.update(normalized.split())
    return terms


def _normalized_semantic_labels(values: Iterable[Any]) -> set[str]:
    labels: set[str] = set()
    for value in values:
        text = str(value or "").casefold().replace("_", " ").replace("-", " ")
        normalized = " ".join(text.split())
        if normalized:
            labels.add(normalized)
            labels.add(normalized.replace(" ", ""))
    return labels


def _compact_semantic_label(value: Any, limit: int = 64) -> str:
    return str(value or "").strip()[:limit]


def _target_room_object_prior(mission: dict[str, Any]) -> dict[str, Any]:
    target = dict(mission.get("target") or {})
    labels = [
        _compact_semantic_label(value)
        for value in list(target.get("labels") or [])[:ROOM_OBJECT_REASONING_MAX_LABELS]
        if _compact_semantic_label(value)
    ]
    name = _compact_semantic_label(target.get("name") or "unknown")
    terms = _normalized_semantic_terms([name, *labels])
    profile = next(
        (
            item
            for item in TARGET_ROOM_OBJECT_PRIORS
            if terms.intersection(_normalized_semantic_terms(item["terms"]))
        ),
        None,
    )
    result: dict[str, Any] = {
        "name": name or "unknown",
        "visible": bool(target.get("visible", False)),
        "semantic_class": str((profile or {}).get("semantic_class") or "unknown"),
        "plausible_room_types": list((profile or {}).get("room_types") or []),
        "plausible_container_types": list(
            (profile or {}).get("container_types") or []
        ),
    }
    if labels:
        result["labels"] = labels
    declared_container = _compact_semantic_label(target.get("likely_container"))
    if declared_container:
        result["declared_likely_container"] = declared_container
        if declared_container not in result["plausible_container_types"]:
            result["plausible_container_types"].append(declared_container)
    return result


def _compact_room_object_anchor(anchor: dict[str, Any]) -> dict[str, Any] | None:
    anchor_type = _compact_semantic_label(anchor.get("type"))
    if not anchor_type:
        return None
    return {"type": anchor_type, "visible": bool(anchor.get("visible", False))}


def build_room_object_reasoning_context(
    mission: dict[str, Any], semantic_graph: dict[str, Any]
) -> dict[str, Any]:
    """Build bounded stage-one evidence for a single two-stage MLLM request.

    The fields labelled ``observed_*`` originate from the compact semantic
    graph. ``target`` carries only general semantic priors, so a matching room
    remains a candidate for exploration rather than asserted containment.
    """

    target = _target_room_object_prior(mission)
    plausible_rooms = _normalized_semantic_labels(target["plausible_room_types"])
    plausible_containers = _normalized_semantic_labels(
        target["plausible_container_types"]
    )
    raw_containers = sorted(
        list(semantic_graph.get("containers") or []),
        key=lambda item: str(item.get("id") or ""),
    )[:ROOM_OBJECT_REASONING_MAX_CONTAINERS]
    observed_containers = []
    containers_by_room: dict[str, list[str]] = {}
    for container in raw_containers:
        container_type = _compact_semantic_label(container.get("type"))
        container_id = _compact_semantic_label(container.get("id"))
        if not container_type or not container_id:
            continue
        entry = {
            "id": container_id,
            "type": container_type,
            "state": _compact_semantic_label(container.get("state") or "unknown"),
            "interaction_available": bool(container.get("interaction_available")),
        }
        room_id = _compact_semantic_label(container.get("room_id"))
        if room_id:
            entry["room_id"] = room_id
            containers_by_room.setdefault(room_id, []).append(container_type)
        observed_containers.append(entry)

    observed_rooms = []
    for room in sorted(
        list(semantic_graph.get("rooms") or []),
        key=lambda item: str(item.get("id") or ""),
    )[:ROOM_OBJECT_REASONING_MAX_ROOMS]:
        room_id = _compact_semantic_label(room.get("id"))
        room_type = _compact_semantic_label(room.get("type") or "unknown")
        if not room_id:
            continue
        entry: dict[str, Any] = {"id": room_id, "type": room_type or "unknown"}
        if bool(room.get("potential_room")):
            entry["potential_room"] = True
            entry["observed_free_space"] = bool(
                room.get("observed_free_space", False)
            )
            source_portal_id = _compact_semantic_label(room.get("source_portal_id"))
            if source_portal_id:
                entry["source_portal_id"] = source_portal_id
        if "room_attribute_confidence" in room:
            entry["room_attribute_confidence"] = round(
                float(room.get("room_attribute_confidence") or 0.0), 2
            )
        anchors = []
        for anchor in list(room.get("anchor_objects") or [])[
            :ROOM_OBJECT_REASONING_MAX_ANCHORS_PER_ROOM
        ]:
            compact_anchor = _compact_room_object_anchor(dict(anchor or {}))
            if compact_anchor is not None:
                anchors.append(compact_anchor)
        if anchors:
            entry["anchor_objects"] = anchors

        evidence = []
        if _normalized_semantic_labels([room_type]).intersection(plausible_rooms):
            evidence.append(f"room_type:{room_type}")
        for anchor in anchors:
            anchor_type = str(anchor["type"])
            if _normalized_semantic_labels([anchor_type]).intersection(
                plausible_containers
            ):
                evidence.append(f"anchor_object:{anchor_type}")
        for container_type in containers_by_room.get(room_id, []):
            if _normalized_semantic_labels([container_type]).intersection(
                plausible_containers
            ):
                evidence.append(f"container:{container_type}")
        entry["target_plausibility"] = {
            "matches_semantic_prior": bool(evidence),
            "evidence": evidence[:ROOM_OBJECT_REASONING_MAX_ANCHORS_PER_ROOM],
        }
        observed_rooms.append(entry)

    observed_portals = []
    for portal in sorted(
        list(semantic_graph.get("portals") or []),
        key=lambda item: str(item.get("id") or ""),
    )[:ROOM_OBJECT_REASONING_MAX_PORTALS]:
        portal_id = _compact_semantic_label(portal.get("id"))
        if not portal_id:
            continue
        observed_portals.append(
            {
                "id": portal_id,
                "type": _compact_semantic_label(portal.get("type") or "portal"),
                "state": _compact_semantic_label(portal.get("state") or "unknown"),
                "interaction_available": bool(portal.get("interaction_available")),
                "connects": [
                    _compact_semantic_label(room_id)
                    for room_id in list(portal.get("connects") or [])[:4]
                    if _compact_semantic_label(room_id)
                ],
            }
        )

    result: dict[str, Any] = {
        "stage": "observed_room_target_plausibility",
        "target": target,
        "observed_rooms": observed_rooms,
        "observed_portals": observed_portals,
        "observed_containers": observed_containers,
    }
    current_room = _compact_semantic_label(semantic_graph.get("current_room"))
    if current_room:
        result["current_room"] = current_room
    unassigned_anchors = []
    for anchor in list(semantic_graph.get("unassigned_anchor_objects") or [])[
        :ROOM_OBJECT_REASONING_MAX_ANCHORS_PER_ROOM
    ]:
        compact_anchor = _compact_room_object_anchor(dict(anchor or {}))
        if compact_anchor is not None:
            unassigned_anchors.append(compact_anchor)
    if unassigned_anchors:
        result["unassigned_anchor_objects"] = unassigned_anchors
    return result


def _compact_history(history: dict[str, Any] | None) -> dict[str, Any]:
    history = history or {}
    if not history:
        return {}
    result = {"last_result": str(history.get("last_result") or "UNKNOWN")}
    for key in ("selection_count", "last_selected_steps_ago", "low_gain_repeat_count"):
        if history.get(key) is not None:
            result[key] = int(history[key])
    if history.get("last_frontier_shrink_m") is not None:
        result["last_frontier_shrink_m"] = round(float(history["last_frontier_shrink_m"]), 2)
    return result


def compact_candidate(
    candidate: BehaviorCandidate,
    *,
    history: dict[str, Any] | None = None,
    room_frontier_length_m: float = 0.0,
    pre_score: float | None = None,
    pre_score_terms: dict[str, Any] | None = None,
    decision_hint: str = "",
) -> dict[str, Any]:
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
    semantic_name = str(metadata.get("semantic_name") or "").strip()
    if behavior_type == "INTERACT" and semantic_name:
        result["subject_semantic_type"] = semantic_name
    room_id = _room_node_id(metadata.get("target_room_id") or metadata.get("room_id"))
    if room_id:
        result["room_id"] = room_id
    if behavior_type == "EXPLORE":
        frontier_cell_count = max(0, int(metadata.get("cell_count", 0) or 0))
        map_resolution = max(0.0, float(metadata.get("map_resolution", 0.0) or 0.0))
        if frontier_cell_count:
            result["frontier_cell_count"] = frontier_cell_count
        if frontier_cell_count and map_resolution:
            result["frontier_length_m"] = round(
                frontier_cell_count * map_resolution, 2
            )
        if room_frontier_length_m > 0.0:
            result["room_frontier_length_m"] = round(room_frontier_length_m, 2)
        unknown_area_m2 = max(
            0.0, float(metadata.get("unknown_component_area_m2", 0.0) or 0.0)
        )
        if unknown_area_m2 > 0.0:
            result["unknown_component_area_m2"] = round(unknown_area_m2, 2)
        expected_visible_area_m2 = max(
            0.0,
            float(
                metadata.get("expected_visible_unknown_area_m2", 0.0) or 0.0
            ),
        )
        if expected_visible_area_m2 > 0.0:
            result["expected_visible_unknown_area_m2"] = round(
                expected_visible_area_m2, 2
            )
        room_status = str(metadata.get("room_status") or "")
        if "mismatch" in room_status.casefold():
            room_status = (
                "unentered_new_room" if "unentered" in room_status.casefold()
                else "entered_room" if "entered" in room_status.casefold()
                else "unknown_room"
            )
        if room_status:
            result["room_status"] = room_status
        robot_room_id = _room_node_id(
            metadata.get("robot_room_id") or metadata.get("current_room_id")
        )
        if robot_room_id:
            result["robot_room_id"] = robot_room_id
        if bool(metadata.get("potential_room")):
            result["potential_room"] = True
        source_portal_id = str(metadata.get("source_portal_id") or "")
        if source_portal_id:
            result["source_portal_id"] = source_portal_id
        room_attribute = str(metadata.get("room_attribute") or "").strip()
        if room_attribute:
            result["room_attribute"] = room_attribute
        if "room_attribute_confidence" in metadata:
            result["room_attribute_confidence"] = round(
                max(0.0, min(1.0, float(metadata.get("room_attribute_confidence") or 0.0))),
                2,
            )
    elif behavior_type == "INTERACT":
        state = str(metadata.get("state") or "")
        if state:
            result["state"] = state
        connected_rooms = list(metadata.get("connected_room_ids") or [])
        if connected_rooms:
            result["connected_room_count"] = len(connected_rooms)
        if bool(metadata.get("target_match")):
            result["explicit_target_match"] = True
        goal_to_subject_distance_m = metadata.get("goal_to_subject_distance_m")
        if goal_to_subject_distance_m is not None:
            result["goal_to_subject_distance_m"] = round(
                max(0.0, float(goal_to_subject_distance_m)), 2
            )
        approach_goals = list(metadata.get("goal_xyyaw_candidates") or [])
        if approach_goals:
            result["approach_goal_count"] = len(approach_goals)
        approach_strategy = str(metadata.get("approach_strategy") or "")
        if approach_strategy:
            result["approach_strategy"] = approach_strategy
    elif bool(metadata.get("target_goal")):
        result["target_visible_now"] = bool(metadata.get("target_visible_now"))
    nearby_semantic_nodes = list(metadata.get("nearby_semantic_nodes") or [])[:3]
    if nearby_semantic_nodes:
        result["nearby_semantic_nodes"] = [
            {
                "type": str(item.get("label") or item.get("type") or "object"),
                "distance_m": round(
                    max(0.0, float(item.get("distance_m", 0.0) or 0.0)), 2
                ),
            }
            for item in nearby_semantic_nodes
        ]
    compact_history = _compact_history(history)
    if compact_history:
        result["history"] = compact_history
    if pre_score is not None:
        result["pre_score"] = round(float(pre_score), 3)
    if pre_score_terms:
        result["pre_score_terms"] = {
            str(key): round(float(value), 3)
            for key, value in pre_score_terms.items()
        }
    if decision_hint == "NEW_ROOM_FRONTIER_HIGH_CONFIDENCE_MISMATCH":
        decision_hint = "NEW_ROOM_FRONTIER"
    if decision_hint and decision_hint not in {"PLAUSIBLE_TARGET_CONTAINER", "UNKNOWN_CONTAINER"}:
        result["decision_hint"] = str(decision_hint)
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
            item["unknown_component_area_m2"] = round(
                sum(
                    max(
                        0.0,
                        float(
                            (member.metadata or {}).get(
                                "unknown_component_area_m2", 0.0
                            )
                            or 0.0
                        ),
                    )
                    for member in members
                ),
                2,
            )
            item["expected_visible_unknown_area_m2"] = round(
                sum(
                    max(
                        0.0,
                        float(
                            (member.metadata or {}).get(
                                "expected_visible_unknown_area_m2", 0.0
                            )
                            or 0.0
                        ),
                    )
                    for member in members
                ),
                2,
            )
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


def compact_candidate_options(
    candidates: list[BehaviorCandidate],
    graph: dict[str, Any],
    *,
    selection_granularity: str = "candidate",
    decision_history: list[dict[str, Any]] | None = None,
    candidate_history: dict[str, dict[str, Any]] | None = None,
    history_region_size_m: float = 1.0,
    room_frontier_lengths: dict[str, float] | None = None,
    pre_scores: dict[str, float] | None = None,
    pre_score_terms: dict[str, dict[str, float]] | None = None,
    decision_hints: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[BehaviorCandidate]]]:
    if str(selection_granularity or "candidate").casefold() == "room":
        return compact_candidate_groups(
            candidates,
            graph,
            decision_history=decision_history,
        )
    candidate_history = candidate_history or {}
    room_frontier_lengths = dict(
        room_frontier_lengths
        if room_frontier_lengths is not None
        else aggregate_room_frontier_lengths(candidates, graph)
    )
    pre_scores = dict(pre_scores or {})
    pre_score_terms = dict(pre_score_terms or {})
    decision_hints = dict(decision_hints or {})
    lookup: dict[str, list[BehaviorCandidate]] = {}
    projected = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        candidate_id = candidate.candidate_id
        lookup[candidate_id] = [candidate]
        history_key = candidate_history_key(
            candidate,
            graph,
            history_region_size_m,
        )
        history = candidate_history.get(candidate_id) or candidate_history.get(history_key)
        room_id = _candidate_room_id(candidate, graph) or "unknown"
        item = compact_candidate(
            candidate,
            history=history,
            room_frontier_length_m=room_frontier_lengths.get(room_id, 0.0),
            pre_score=(
                pre_scores[candidate_id] if candidate_id in pre_scores else None
            ),
            pre_score_terms=pre_score_terms.get(candidate_id),
            decision_hint=decision_hints.get(candidate_id, ""),
        )
        if (
            str(candidate.behavior_type or "").upper() == "EXPLORE"
            and room_id != "unknown"
        ):
            item["room_id"] = room_id
        projected.append(item)
    return projected, lookup


def aggregate_room_frontier_lengths(
    candidates: Iterable[BehaviorCandidate], graph: dict[str, Any]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for candidate in candidates:
        if str(candidate.behavior_type or "").upper() != "EXPLORE":
            continue
        room_id = _candidate_room_id(candidate, graph) or "unknown"
        metadata = candidate.metadata or {}
        result[room_id] = result.get(room_id, 0.0) + (
            max(0, int(metadata.get("cell_count", 0) or 0))
            * max(0.0, float(metadata.get("map_resolution", 0.0) or 0.0))
        )
    return result


class ModelPolicyClient:
    def __init__(self, config: ModelPolicyConfig | None = None) -> None:
        self.config = config or ModelPolicyConfig()
        if self.config.context_profile not in {"public_facts_v2", "historical_compat_v1"}:
            raise ValueError(f"Unsupported M2 context profile: {self.config.context_profile}")
        if not 0 <= int(self.config.recent_decision_limit) <= 30:
            raise ValueError("M2 recent_decision_limit must be between 0 and 30")
        if self.config.context_profile == "historical_compat_v1" and self.config.selection_granularity.casefold() != "candidate":
            raise ValueError("Historical M2 context compatibility requires candidate granularity")
        self._prompt_override = ""
        self._prompt_file_sha256 = ""
        if self.config.prompt_file:
            prompt_bytes = Path(self.config.prompt_file).expanduser().read_bytes()
            self._prompt_override = prompt_bytes.decode("utf-8").strip()
            self._prompt_file_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
            if not self._prompt_override:
                raise ValueError("M2 prompt file must not be empty")
        self.last_request_context: dict[str, Any] = {}
        self.last_error = ""
        self.last_metrics: dict[str, Any] = {}
        self.last_result_source = "not_called"
        self.last_ranking_ids: list[str] = []
        self.last_selected_group_id = ""
        self.last_selected_candidate_id = ""
        self.last_reason = ""
        self.last_confidence = ""
        self.last_pre_score_guard = ""
        self.last_rejected_model_ids: list[str] = []
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
        self.last_selected_candidate_id = ""
        self.last_reason = ""
        self.last_confidence = ""
        self.last_pre_score_guard = ""
        self.last_rejected_model_ids = []
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
        candidate_history = dict((robot_context or {}).get("candidate_history") or {})
        _, candidate_groups = compact_candidate_options(
            candidates,
            graph or {},
            selection_granularity=self.config.selection_granularity,
            decision_history=decision_history,
            candidate_history=candidate_history,
            history_region_size_m=self.config.history_region_size_m,
            room_frontier_lengths=dict(
                (robot_context or {}).get("room_frontier_lengths") or {}
            )
            or None,
        )
        response, rejected_model_ids = self._sanitize_model_selection(
            response,
            set(candidate_groups),
        )
        self.last_rejected_model_ids = rejected_model_ids
        if response is None:
            self.last_error = (
                "invalid_model_selection:no_valid_curated_candidate"
                + (
                    f":rejected={','.join(rejected_model_ids)}"
                    if rejected_model_ids
                    else ""
                )
            )
            self.last_result_source = "curated_fallback_invalid_response"
            fallback_id = self._curated_fallback_id(
                candidate_groups,
                robot_context or {},
            )
            if not fallback_id:
                return None
            self.last_ranking_ids = [fallback_id]
            self.last_selected_group_id = fallback_id
            self.last_selected_candidate_id = candidate_groups[fallback_id][0].candidate_id
            self.last_reason = "CURATED_FALLBACK_INVALID_MODEL_ID"
            self.last_confidence = "low"
            return candidate_groups[fallback_id][0]
        try:
            response = validate_subgoal_selection(response, set(candidate_groups))
        except (TypeError, ValueError) as exc:
            self.last_error = f"invalid_model_selection: {exc}"
            self.last_result_source = "curated_fallback_invalid_response"
            return None
        selected_id = response["candidate_id"]
        self.last_ranking_ids = list(response.get("ranked_ids") or [selected_id])
        self.last_reason = str(response.get("reason") or "")
        self.last_confidence = str(response.get("confidence") or "")
        selected_id = self._apply_pre_score_guard(
            selected_id,
            candidate_groups,
            robot_context or {},
        )
        if selected_id not in self.last_ranking_ids:
            self.last_ranking_ids.insert(0, selected_id)
        elif self.last_ranking_ids[0] != selected_id:
            self.last_ranking_ids.remove(selected_id)
            self.last_ranking_ids.insert(0, selected_id)
        self.last_selected_group_id = selected_id
        self.last_selected_candidate_id = candidate_groups[selected_id][0].candidate_id
        scores = response["scores"]
        for candidate in candidates:
            if candidate.candidate_id in scores:
                candidate.score = float(scores[candidate.candidate_id])
                candidate.score_terms = {"model_score": candidate.score}
        self.last_result_source = (
            "model_pre_score_guard"
            if self.last_pre_score_guard
            else "model_filtered_unknown_ids"
            if self.last_rejected_model_ids
            else "model"
        )
        return candidate_groups[selected_id][0]

    @staticmethod
    def _sanitize_model_selection(
        response: Any,
        candidate_ids: set[str],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Drop hallucinated option IDs before schema validation/execution.

        The model may return one valid option after an invalid one.  Keeping
        the valid suffix is preferable to throwing away the whole decision,
        while an all-invalid response remains a hard failure for the caller's
        curated-pool fallback.
        """
        if not isinstance(response, dict):
            return None, []
        raw_ranked = response.get("ranked_ids") or []
        if not isinstance(raw_ranked, list):
            return None, []
        if not raw_ranked and response.get("candidate_id"):
            raw_ranked = [response.get("candidate_id")]
        valid: list[str] = []
        rejected: list[str] = []
        for value in raw_ranked:
            candidate_id = str(value or "")
            if candidate_id not in candidate_ids:
                if candidate_id and candidate_id not in rejected:
                    rejected.append(candidate_id)
                continue
            if candidate_id not in valid:
                valid.append(candidate_id)
        if not valid:
            return None, rejected
        sanitized = dict(response)
        sanitized["candidate_id"] = valid[0]
        sanitized["ranked_ids"] = valid[:3]
        raw_scores = response.get("scores") or {}
        if isinstance(raw_scores, dict):
            sanitized["scores"] = {
                str(key): value
                for key, value in raw_scores.items()
                if str(key) in candidate_ids
            }
        return sanitized, rejected

    @staticmethod
    def _curated_fallback_id(
        candidate_groups: dict[str, list[BehaviorCandidate]],
        robot_context: dict[str, Any],
    ) -> str:
        if not candidate_groups:
            return ""
        pre_scores = dict(robot_context.get("candidate_pre_scores") or {})

        def fallback_key(group_id: str) -> tuple[float, float, float, str]:
            candidate = candidate_groups[group_id][0]
            return (
                -float(
                    pre_scores.get(
                        group_id,
                        pre_scores.get(candidate.candidate_id, 0.0),
                    )
                    or 0.0
                ),
                -float(candidate.score),
                max(0.0, float(candidate.features.get("distance_m", 0.0) or 0.0)),
                group_id,
            )

        return min(candidate_groups, key=fallback_key)

    def _apply_pre_score_guard(
        self,
        selected_id: str,
        candidate_groups: dict[str, list[BehaviorCandidate]],
        robot_context: dict[str, Any],
    ) -> str:
        if self.config.selection_granularity.casefold() == "room":
            return selected_id
        margin = max(0.0, float(self.config.pre_score_guard_margin))
        pre_scores = dict(robot_context.get("candidate_pre_scores") or {})
        hints = dict(robot_context.get("candidate_decision_hints") or {})
        protected_hints = {
            "TARGET_GOAL",
            "POST_INTERACTION_TRAVERSE",
            "TARGET_CONTAINER",
            "NEXT_ROUTE_PORTAL",
            "INTERACTION_COVERAGE_CONTAINER",
            "INTERACTION_COVERAGE_PORTAL",
        }
        protected = [
            candidate_id
            for candidate_id in candidate_groups
            if str(hints.get(candidate_id) or "").upper() in protected_hints
            and candidate_id in pre_scores
        ]
        if not protected:
            protected_selected = selected_id
        else:
            best_id = max(
                protected,
                key=lambda candidate_id: (
                    float(pre_scores.get(candidate_id, 0.0) or 0.0),
                    candidate_id,
                ),
            )
            selected_score = float(pre_scores.get(selected_id, 0.0) or 0.0)
            best_score = float(pre_scores.get(best_id, 0.0) or 0.0)
            if best_id == selected_id or best_score < selected_score + margin:
                protected_selected = selected_id
            else:
                hint = str(hints.get(best_id) or "").upper()
                self.last_pre_score_guard = (
                    f"{hint}:{selected_id}->{best_id}:margin={best_score - selected_score:.3f}"
                )
                self.last_reason = f"PRE_SCORE_GUARD_{hint}"
                self.last_confidence = "high"
                protected_selected = best_id

        # Target, post-door, target-container, and route-portal guards always
        # win over room novelty.  Only when none is selected do we protect an
        # actually unentered room from an arbitrary MLLM detour back into a
        # previously entered room.  High-confidence semantic mismatches are
        # deliberately excluded from this guard and remain an exploration
        # fallback rather than a forced choice.
        selected_hint = str(hints.get(protected_selected) or "").upper()
        if selected_hint in protected_hints:
            return protected_selected
        new_room_ids = [
            candidate_id
            for candidate_id, members in candidate_groups.items()
            if str(hints.get(candidate_id) or "").upper() == "NEW_ROOM_FRONTIER"
            and bool((members[0].metadata or {}).get("room_status") == "unentered_new_room")
        ]
        if not new_room_ids:
            return protected_selected
        best_new_room_id = max(
            new_room_ids,
            key=lambda candidate_id: (
                float(pre_scores.get(candidate_id, 0.0) or 0.0),
                candidate_id,
            ),
        )
        selected_score = float(pre_scores.get(protected_selected, 0.0) or 0.0)
        best_new_room_score = float(pre_scores.get(best_new_room_id, 0.0) or 0.0)
        if (
            best_new_room_id == protected_selected
            or best_new_room_score < selected_score + margin
        ):
            return protected_selected
        self.last_pre_score_guard = (
            "NEW_ROOM_FRONTIER:"
            f"{protected_selected}->{best_new_room_id}:"
            f"margin={best_new_room_score - selected_score:.3f}"
        )
        self.last_reason = "PRE_SCORE_GUARD_NEW_ROOM_FRONTIER"
        self.last_confidence = "high"
        return best_new_room_id

    def build_request(
        self,
        candidates: list[BehaviorCandidate],
        target_context: dict[str, Any],
        graph: dict[str, Any],
        robot_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        decision_history = list((robot_context or {}).get("group_history") or [])
        recent_decisions = list((robot_context or {}).get("decision_history") or [])
        candidate_history = dict((robot_context or {}).get("candidate_history") or {})
        historical = self.config.context_profile == "historical_compat_v1"
        expose_pre_scores = historical or self.config.include_pre_scores
        compact_candidates, _ = compact_candidate_options(
            candidates,
            graph,
            selection_granularity=self.config.selection_granularity,
            decision_history=decision_history,
            candidate_history=candidate_history,
            history_region_size_m=self.config.history_region_size_m,
            room_frontier_lengths=dict(
                (robot_context or {}).get("room_frontier_lengths") or {}
            )
            or None,
            pre_scores=dict(
                (robot_context or {}).get("candidate_pre_scores") or {}
            ) if expose_pre_scores else None,
            pre_score_terms=dict(
                (robot_context or {}).get("candidate_pre_score_terms") or {}
            ) if expose_pre_scores else None,
            decision_hints=dict(
                (robot_context or {}).get("candidate_decision_hints") or {}
            ),
        )
        if historical:
            _restore_historical_candidate_context(compact_candidates, candidates, robot_context or {})
        # Preserve corrected room facts in compatibility mode; the default
        # public profile leaves previously frozen replay inputs unchanged.
        if historical or "robot_graph_xy" in (robot_context or {}):
            current_room = _current_room_id(graph, robot_context)
            for option in compact_candidates:
                if "robot_room_id" in option:
                    if current_room:
                        option["robot_room_id"] = current_room
                    else:
                        option.pop("robot_room_id", None)
        self.last_candidate_groups = list(compact_candidates)
        mission = compact_target_context(target_context)
        if historical:
            semantic_graph, public_robot = historical_compat_context(
                graph, robot_context, max_nodes=self.config.max_graph_nodes,
            )
            room_object_reasoning = build_room_object_reasoning_context(mission, semantic_graph)
        else:
            semantic_graph = compact_semantic_graph(
                graph, robot_context=robot_context, max_nodes=self.config.max_graph_nodes,
            )
            public_robot = compact_robot_context(graph, robot_context)
            room_object_reasoning = {}
        interaction_coverage_mission = (
            str(mission.get("mode") or "").casefold()
            == "interaction_coverage_exploration"
        )
        if self.config.selection_granularity.casefold() == "room":
            instruction = (
                "Rank the room-level exploration groups and concrete interaction or navigation actions "
                "by semantic relevance to the mission. Return option IDs unchanged. Use graph semantics, "
                "distance, room frontier length, and recent room history. Return exactly one compact JSON "
                "object with ranked_ids containing at most three IDs, a short reason code, and confidence "
                "set to low, medium, or high. Do not return prose, scores, markdown, or additional keys."
            )
        elif interaction_coverage_mission:
            instruction = (
                "Rank the concrete subgoals for an interaction-coverage exploration mission. "
                "Each ID is an executable navigation, interaction, or frontier action and must be "
                "returned unchanged. This is a text-only decision: do not infer object geometry, "
                "view direction, handles, or actions not explicitly represented by the candidates. "
                "Only IDs present in the current candidates array may appear in ranked_ids; IDs in "
                "history or graph are historical context and are forbidden. Apply this order: (1) "
                "POST_INTERACTION_TRAVERSE immediately after a successful portal interaction; "
                "(2) an untried viable interaction candidate that changes connectivity or reveals "
                "contents, favoring a graph-reachable container or portal with an "
                "INTERACTION_COVERAGE_* decision_hint; (3) a useful frontier; (4) a repeated or "
                "previously failed action only after alternatives. A failed approach pose or visual "
                "frontality check does not prove the object non-interactable: another listed "
                "approach/viewpoint can remain viable, so do not globally reject that object. Do not "
                "repeat a successful interaction. Use candidate history, decision_hint, pre_score, "
                "and observed graph state only. Return exactly one compact JSON object with ranked_ids "
                "containing at most three IDs, reason set to INTERACTION_COVERAGE, UNLOCK_ROUTE, "
                "INFORMATION_GAIN, RECOVERY_DIVERSIFICATION, DISTANCE_TIEBREAK, or "
                "NO_SEMANTIC_PREFERENCE, and confidence set to low, medium, or high. Do not return "
                "prose, scores, markdown, or additional keys. Output must begin exactly as a compact "
                "object whose first key is ranked_ids; never echo the supplied context."
            )
        else:
            instruction = (
                'Find the requested target. Rank up to three CURRENT executable candidate IDs. Work through the following checks internally; return only the final JSON, not the reasoning.\n'
                '1. EVIDENCE: Read mission, room_object_reasoning and the observed graph. Separate observed visibility/connectivity from uncertain room labels, semantic priors and pre_score. Priors never prove containment. A potential room is unexplored topology, not observed free space. Never invent facts or select historical IDs.\n'
                '2. DEPENDENCIES: Prefer a reliably observed TARGET_GOAL. Otherwise choose POST_INTERACTION_TRAVERSE after opening a portal, unless that exact candidate failed. Next prefer NEXT_ROUTE_PORTAL when observed topology establishes a prerequisite route. Do not infer route necessity from proximity.\n'
                '3. COMPATIBILITY: Before ranking a container, check whether it can physically and semantically contain the requested target, or its observed effect enables the route. Strong size, function or storage-context incompatibility makes interaction extremely low priority: rank it below all useful frontiers, newly accessible rooms and plausible containers, regardless of proximity or pre_score. Do not treat every openable object as useful. Unknown plausible containers remain eligible; low likelihood is not a hard ban. Use only the supplied mission and observed semantics, never hidden object names or episode-specific assumptions. This is target search, not interaction coverage.\n'
                'NEGATIVE PRIORITY: When the requested object is unrelated to a container storage function, explore another area or a promising closed doorway instead of approaching, inspecting or retrying that container. Being nearby, openable, or previously approached is not evidence of relevance. A failed face does not increase semantic relevance. Such containers are last-resort choices only when no useful alternative is listed.\n'
                'POSITIVE CONTAINER PRIORITY: An unsearched, closed container with a strong ordinary storage relationship to the target is a direct target-discovery opportunity. Prefer opening and inspecting that compatible container over generic frontiers or revisiting explored rooms, unless a reliable target goal or a necessary route prerequisite takes priority. Do not wait until every frontier is exhausted. Mere capacity to fit the object is not a strong relationship; use public category, function and observed room context. Unknown relevance remains uncertain, not automatically incompatible.\n'
                'CLOSED DOOR PRIORITY: Prefer an executable closed-door candidate that can reveal an unentered or unknown room over generic exploration in already entered rooms. Increase its interest when the associated room is compatible with the target; an unknown room is still valuable unexplored space. Use observed connectivity and room-entry history, not proximity alone. This does not prove the room contains the target. Never prefer known static, unavailable, already open or repeatedly failed portals without new evidence.\n'
                'AFTER OPENING: Prioritize exploring the newly accessible, unentered room before unrelated container interactions in the old room. If a far-side traversal point failed, prefer a CURRENT frontier toward that doorway or new room; do not repeat the blocked point or infer that the room is exhausted. Unknown space is not proof of a clear path; the executor still checks known obstacles.\n'
                '4. PROGRESS: Compare eligible candidates by expected target discovery. Prefer compatible or unknown unentered rooms over exhausted regions; treat high-confidence room mismatch as a fallback. Among comparable frontiers use expected_visible_unknown_area_m2, then unknown_component_area_m2; distance breaks ties. Do not repeat completed interactions. After failure or no information gain, diversify unless evidence changed; a failed approach alone does not rule out another listed viewpoint.\n'
                'FAILURE MEMORY: A new frontier ID or room label near a failed position is not a new opportunity. Do not revisit the same failed region merely because its ID changed or an unrelated action occurred. Prefer a different reachable region, a compatible unsearched container, or a closed door that changes connectivity; retry only with new route or observation evidence.\n'
                '5. VALIDATE: Check current candidate membership, unmet prerequisites and recent outcomes. Return only {"ranked_ids":[...],"reason":"...","confidence":"..."}, with ranked_ids first and no additional keys. Allowed reason: TARGET_VISIBLE, REVEAL_TARGET_CONTAINER, UNLOCK_ROUTE, EXPLORE_TARGET_ROOM, INFORMATION_GAIN, RECOVERY_DIVERSIFICATION, DISTANCE_TIEBREAK, NO_SEMANTIC_PREFERENCE. Allowed confidence: low, medium, high.\n'
            )
        if self._prompt_override:
            instruction = self._prompt_override
        self.last_request_context = {
            "m2_context_profile": self.config.context_profile,
            "m2_prompt_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
            "m2_prompt_file": str(self.config.prompt_file),
            "m2_prompt_file_sha256": self._prompt_file_sha256,
            "m2_recent_decision_limit": int(self.config.recent_decision_limit),
        }
        limit = int(self.config.recent_decision_limit)
        return {
            "schema_version": 5,
            "instruction": instruction,
            "mission": mission,
            "robot": public_robot,
            "recent_decisions": recent_decisions[-limit:] if limit else [],
            "graph": semantic_graph,
            "room_object_reasoning": room_object_reasoning,
            "candidates": compact_candidates,
        }

    def _request(
        self, payload: dict[str, Any], metrics_context: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        self.last_error = ""
        self.last_metrics = {}
        self.last_result_source = "rule_fallback"
        mode = str(self.config.mode or "disabled").casefold()
        # Freeze the public input before issuing/retrying any network request.
        # This stores no transport headers, credentials or private simulator data.
        metrics_context = {
            **(metrics_context or {}),
            **self.last_request_context,
            "public_request": json.loads(json.dumps(payload, ensure_ascii=False)),
            "request_started_ts": time.time(),
        }
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
        retry_limit = min(3, max(0, int(self.config.timeout_retry_count)))
        backoff_base_s = max(0.0, float(self.config.timeout_retry_backoff_s))
        request_started = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        for attempt_index in range(retry_limit + 1):
            attempt_started = time.perf_counter()
            self.last_metrics = {}
            attempt_context = {
                **(metrics_context or {}),
                "m2_request_attempt": attempt_index + 1,
                "m2_timeout_retry_limit": retry_limit,
                "m2_is_timeout_retry": attempt_index > 0,
            }
            try:
                if mode == "command":
                    response = self._request_command(payload)
                elif mode == "http":
                    response = self._request_http(
                        payload,
                        metrics_context=attempt_context,
                    )
                else:
                    raise ValueError(
                        f"unsupported model policy mode: {self.config.mode}"
                    )
            except (OSError, ValueError, subprocess.SubprocessError, TimeoutError) as exc:
                timeout_failure = is_model_timeout_error(exc)
                attempts.append(
                    self._request_attempt_metrics(
                        attempt_index=attempt_index,
                        elapsed_s=time.perf_counter() - attempt_started,
                        error=str(exc),
                        timeout=timeout_failure,
                    )
                )
                if not timeout_failure or attempt_index >= retry_limit:
                    self.last_error = str(exc)
                    self.last_metrics = self._request_summary_metrics(
                        attempts,
                        request_started=request_started,
                        retry_limit=retry_limit,
                    )
                    return None
                backoff_s = backoff_base_s * (2**attempt_index)
                attempts[-1]["backoff_s"] = backoff_s
                if backoff_s > 0.0:
                    time.sleep(backoff_s)
                continue

            attempts.append(
                self._request_attempt_metrics(
                    attempt_index=attempt_index,
                    elapsed_s=time.perf_counter() - attempt_started,
                    error="",
                    timeout=False,
                )
            )
            self.last_metrics = self._request_summary_metrics(
                attempts,
                request_started=request_started,
                retry_limit=retry_limit,
            )
            return response
        return None

    def _request_attempt_metrics(
        self,
        *,
        attempt_index: int,
        elapsed_s: float,
        error: str,
        timeout: bool,
    ) -> dict[str, Any]:
        """Keep bounded per-attempt M2 telemetry in the decision event."""

        response_metrics = dict(self.last_metrics)
        return {
            "attempt": attempt_index + 1,
            "latency_s": float(
                response_metrics.get("latency_s", elapsed_s) or elapsed_s
            ),
            "prompt_tokens": int(response_metrics.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(response_metrics.get("completion_tokens", 0) or 0),
            "total_tokens": int(response_metrics.get("total_tokens", 0) or 0),
            "timeout": bool(timeout),
            "error": str(error or response_metrics.get("error") or ""),
        }

    def _request_summary_metrics(
        self,
        attempts: list[dict[str, Any]],
        *,
        request_started: float,
        retry_limit: int,
    ) -> dict[str, Any]:
        final_metrics = dict(self.last_metrics)
        final_metrics.update(
            {
                **self.last_request_context,
                "model": self.config.model,
                "timeout_s": float(self.config.timeout_s),
                "request_total_latency_s": time.perf_counter() - request_started,
                "request_attempts": len(attempts),
                "timeout_retry_count": max(0, len(attempts) - 1),
                "timeout_retry_limit": int(retry_limit),
                "attempts": attempts,
            }
        )
        return final_metrics

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
        context, output_tokens, budget_metrics = self._bounded_http_context(payload)
        response = self._mllm_client.request_json(
            role="subgoal_selection",
            instruction=str(payload.get("instruction") or ""),
            context=context,
            response_schema=build_subgoal_selection_response_schema(payload),
            timeout_s=self.config.timeout_s,
            max_tokens=output_tokens,
            metrics_context={**(metrics_context or {}), **budget_metrics},
        )
        self.last_metrics = response.metrics()
        if response.error:
            raise ValueError(response.error)
        if not isinstance(response.payload, dict):
            raise ValueError("model HTTP response must be a JSON object")
        return response.payload

    def _bounded_http_context(self, payload: dict[str, Any]) -> tuple[dict, int, dict]:
        context = json.loads(json.dumps({key: payload.get(key) or default for key, default in (
            ("mission", {}), ("robot", {}), ("graph", {}), ("room_object_reasoning", {}),
            ("candidates", []), ("recent_decisions", []))}, ensure_ascii=False))
        output_tokens = min(512, max(1, int(self.config.max_tokens)))
        limit = int(self.config.context_window_tokens) - output_tokens - 1024
        instruction_bytes = len(str(payload.get("instruction") or "").encode("utf-8"))
        def size():
            return instruction_bytes + len(json.dumps(context, ensure_ascii=False).encode("utf-8"))
        original_size = size()
        while size() > limit and context["recent_decisions"]:
            context["recent_decisions"].pop(0)
        if size() > limit:
            context["room_object_reasoning"] = {}
        while size() > limit:
            graph_lists = [value for value in context["graph"].values() if isinstance(value, list) and value]
            if not graph_lists:
                break
            max(graph_lists, key=lambda value: len(json.dumps(value, ensure_ascii=False))).pop()
        if size() > limit:
            context["graph"] = {}
            context["robot"] = {key: value for key, value in context["robot"].items()
                                if key in {"robot_xy", "room_id", "current_room_id"}}
        if size() > limit:
            raise ValueError("M2 context budget exceeded by mandatory mission/candidates; refusing oversized request")
        return context, output_tokens, {
            "m2_context_window_tokens": self.config.context_window_tokens,
            "m2_input_utf8_bytes_before": original_size,
            "m2_input_utf8_bytes_after": size(),
            "m2_context_compacted": size() < original_size,
            "m2_retained_candidate_count": len(context["candidates"]),
        }
