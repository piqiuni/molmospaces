from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Iterable

from .behavior_candidates import BehaviorCandidate


def _as_int(value: object, default: int | None = None) -> int | None:
    """Return an integer metric without treating booleans as measurements."""

    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def candidate_observation_step(candidate_snapshot: dict[str, Any]) -> int | None:
    """Return the public semantic-map capture step, if the publisher has one."""

    return _as_int(
        (candidate_snapshot.get("exploration_context") or {}).get(
            "observation_step"
        )
    )


@dataclass(frozen=True)
class PostInteractionRefreshConfig:
    """Bounded post-open barrier before a one-shot portal traversal.

    Candidate publication is independent of mapping, so a new sequence,
    revision, or optional RGB-D capture alone is not evidence that *the opened
    portal* has propagated through the map.  The normal release condition
    combines configured freshness with the matching portal's confirmed
    open/traversable state.  A timeout is only a bounded hand-off to the
    executor's fail-closed ``make_plan`` wait; it never itself proves that
    traversal is safe.
    """

    enabled: bool = True
    min_candidate_updates: int = 2
    timeout_s: float = 30.0
    require_graph_revision: bool = True
    require_observation_step: bool = True
    require_portal_open_confirmation: bool = True


@dataclass(frozen=True)
class PortalOpenConfirmation:
    """Confirmed public state for the portal that just completed ``open``."""

    portal_id: str
    observed: bool
    state: str
    state_open: bool
    traversable: bool
    requires_interaction_cleared: bool

    @property
    def ready(self) -> bool:
        return bool(
            self.observed
            and self.state_open
            and self.traversable
            and self.requires_interaction_cleared
        )


def _as_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _portal_state_sources(
    candidate_snapshot: dict[str, Any], portal_id: str
) -> list[dict[str, Any]]:
    """Return compact-graph records belonging to exactly one portal.

    Mapping publishes the state both on the portal node and on connectivity
    edges.  Accept either representation so that a compact-graph transition
    cannot be missed merely because a node/edge publication races the other.
    """

    graph = candidate_snapshot.get("graph_context") or {}
    portal_id = str(portal_id or "")
    if not portal_id:
        return []
    sources: list[dict[str, Any]] = []
    for node in graph.get("nodes") or []:
        if str(node.get("id") or "") != portal_id:
            continue
        attributes = node.get("attributes") or {}
        override = attributes.get("interaction_state_override") or {}
        interaction = node.get("interaction") or {}
        sources.extend((node, interaction, override, attributes))
    for edge in graph.get("edges") or []:
        attributes = edge.get("attributes") or {}
        if (
            str(attributes.get("portal_node_id") or "") == portal_id
            or str(edge.get("src_id") or "") == portal_id
            or str(edge.get("dst_id") or "") == portal_id
        ):
            sources.append(attributes)
    return sources


def portal_open_confirmation(
    candidate_snapshot: dict[str, Any], portal_id: str
) -> PortalOpenConfirmation:
    """Read only same-portal open/traversable evidence from a candidate graph."""

    sources = _portal_state_sources(candidate_snapshot, portal_id)
    states = [
        str(source.get("interaction_state") or source.get("state") or "").casefold()
        for source in sources
    ]
    open_states = {"open", "static_open", "opened", "completed"}
    state_open = any(state in open_states for state in states)
    state = next((state for state in states if state in open_states), "")
    if not state:
        state = next((state for state in states if state), "unknown")
    traversable = any(
        _as_optional_bool(source.get("traversable")) is True
        for source in sources
    )
    requires_interaction_cleared = any(
        _as_optional_bool(source.get("requires_interaction")) is False
        for source in sources
    )
    return PortalOpenConfirmation(
        portal_id=str(portal_id or ""),
        observed=bool(sources),
        state=state,
        state_open=state_open,
        traversable=traversable,
        requires_interaction_cleared=requires_interaction_cleared,
    )


@dataclass(frozen=True)
class PostInteractionRefreshStatus:
    """The gate decision plus diagnostic counters for the decision trace."""

    active: bool
    ready: bool
    timed_out: bool
    reason: str
    candidate_updates: int
    graph_updated: bool
    observation_updated: bool
    baseline_sequence: int
    baseline_graph_revision: int
    baseline_observation_step: int | None
    latest_sequence: int
    latest_graph_revision: int
    latest_observation_step: int | None
    portal_id: str
    portal_observed: bool
    portal_state: str
    portal_open: bool
    portal_traversable: bool
    portal_requires_interaction_cleared: bool
    portal_confirmed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": bool(self.active),
            "ready": bool(self.ready),
            "timed_out": bool(self.timed_out),
            "reason": self.reason,
            "candidate_updates": int(self.candidate_updates),
            "graph_updated": bool(self.graph_updated),
            "observation_updated": bool(self.observation_updated),
            "baseline_sequence": int(self.baseline_sequence),
            "baseline_graph_revision": int(self.baseline_graph_revision),
            "baseline_observation_step": self.baseline_observation_step,
            "latest_sequence": int(self.latest_sequence),
            "latest_graph_revision": int(self.latest_graph_revision),
            "latest_observation_step": self.latest_observation_step,
            "portal_id": self.portal_id,
            "portal_observed": bool(self.portal_observed),
            "portal_state": self.portal_state,
            "portal_open": bool(self.portal_open),
            "portal_traversable": bool(self.portal_traversable),
            "portal_requires_interaction_cleared": bool(
                self.portal_requires_interaction_cleared
            ),
            "portal_confirmed": bool(self.portal_confirmed),
        }


class PostInteractionRefreshGate:
    """Track the public observation/map barrier after a successful portal open."""

    def __init__(self, config: PostInteractionRefreshConfig) -> None:
        self.config = config
        self.clear()

    @property
    def active(self) -> bool:
        return self._active

    def clear(self) -> None:
        self._active = False
        self._episode_id = ""
        self._portal_id = ""
        self._started_at = 0.0
        self._baseline_sequence = 0
        self._baseline_graph_revision = 0
        self._baseline_observation_step: int | None = None

    def begin(
        self,
        candidate_snapshot: dict[str, Any],
        *,
        portal_id: str,
        now: float,
    ) -> PostInteractionRefreshStatus:
        """Start a barrier from the last pre-open decision snapshot."""

        self._episode_id = str(candidate_snapshot.get("episode_id") or "")
        self._portal_id = str(portal_id or "")
        self._started_at = float(now)
        self._baseline_sequence = int(
            _as_int(candidate_snapshot.get("sequence"), 0) or 0
        )
        self._baseline_graph_revision = int(
            _as_int(candidate_snapshot.get("graph_revision"), 0) or 0
        )
        self._baseline_observation_step = candidate_observation_step(
            candidate_snapshot
        )
        self._active = bool(self.config.enabled)
        return self.status(candidate_snapshot, now=now)

    def status(
        self, candidate_snapshot: dict[str, Any], *, now: float
    ) -> PostInteractionRefreshStatus:
        """Evaluate the latest candidate snapshot without mutating the barrier."""

        latest_sequence = int(_as_int(candidate_snapshot.get("sequence"), 0) or 0)
        latest_graph_revision = int(
            _as_int(candidate_snapshot.get("graph_revision"), 0) or 0
        )
        latest_observation_step = candidate_observation_step(candidate_snapshot)
        portal = portal_open_confirmation(candidate_snapshot, self._portal_id)
        candidate_updates = max(0, latest_sequence - self._baseline_sequence)
        graph_updated = latest_graph_revision > self._baseline_graph_revision
        observation_updated = (
            self._baseline_observation_step is not None
            and latest_observation_step is not None
            and latest_observation_step > self._baseline_observation_step
        )
        episode_changed = bool(
            self._episode_id
            and str(candidate_snapshot.get("episode_id") or "")
            and str(candidate_snapshot.get("episode_id") or "") != self._episode_id
        )
        generic_freshness_ready = (
            candidate_updates >= max(1, int(self.config.min_candidate_updates))
            and (
                not self.config.require_graph_revision
                or graph_updated
            )
            and (
                not self.config.require_observation_step
                or observation_updated
            )
        )
        portal_confirmed = bool(
            portal.ready or not self.config.require_portal_open_confirmation
        )
        freshness_ready = bool(generic_freshness_ready and portal_confirmed)
        timed_out = bool(
            self._active
            and not episode_changed
            and float(now) - self._started_at >= max(0.0, float(self.config.timeout_s))
        )
        if episode_changed:
            reason = "episode_changed"
        elif not self._active:
            reason = "disabled"
        elif freshness_ready:
            reason = (
                "fresh_opened_portal_graph_observation"
                if self.config.require_observation_step
                else "fresh_opened_portal_graph"
            )
        elif timed_out:
            reason = "post_open_portal_confirmation_timeout_fallback"
        elif candidate_updates < max(1, int(self.config.min_candidate_updates)):
            reason = "awaiting_candidate_updates"
        elif self.config.require_graph_revision and not graph_updated:
            reason = "awaiting_graph_revision"
        elif self.config.require_observation_step and not observation_updated:
            reason = "awaiting_observation_step"
        elif self.config.require_portal_open_confirmation and not portal.observed:
            reason = "awaiting_opened_portal"
        elif self.config.require_portal_open_confirmation and not portal.state_open:
            reason = "awaiting_portal_open_state"
        elif self.config.require_portal_open_confirmation and not portal.traversable:
            reason = "awaiting_portal_traversable"
        else:
            reason = "awaiting_portal_requires_interaction_clear"
        return PostInteractionRefreshStatus(
            active=bool(self._active),
            ready=bool(freshness_ready or timed_out),
            timed_out=timed_out,
            reason=reason,
            candidate_updates=candidate_updates,
            graph_updated=graph_updated,
            observation_updated=observation_updated,
            baseline_sequence=self._baseline_sequence,
            baseline_graph_revision=self._baseline_graph_revision,
            baseline_observation_step=self._baseline_observation_step,
            latest_sequence=latest_sequence,
            latest_graph_revision=latest_graph_revision,
            latest_observation_step=latest_observation_step,
            portal_id=self._portal_id,
            portal_observed=portal.observed,
            portal_state=portal.state,
            portal_open=portal.state_open,
            portal_traversable=portal.traversable,
            portal_requires_interaction_cleared=(
                portal.requires_interaction_cleared
            ),
            portal_confirmed=portal_confirmed,
        )


def portal_center_xy(graph: dict[str, Any], node_id: str) -> list[float] | None:
    """Return the public portal center available in the decision snapshot."""

    for node in graph.get("nodes") or []:
        if str(node.get("id") or "") != str(node_id or ""):
            continue
        attributes = node.get("attributes") or {}
        center = list(
            attributes.get("interaction_reference_aabb_center")
            or node.get("aabb_center")
            or node.get("centroid")
            or []
        )
        if len(center) < 2:
            return None
        return [float(center[0]), float(center[1])]
    return None


def portal_aabb_size_xy(graph: dict[str, Any], node_id: str) -> list[float] | None:
    """Return the interaction-reference portal footprint used for its axis."""

    for node in graph.get("nodes") or []:
        if str(node.get("id") or "") != str(node_id or ""):
            continue
        attributes = node.get("attributes") or {}
        size = list(
            attributes.get("interaction_reference_aabb_size")
            or node.get("aabb_size")
            or []
        )
        if len(size) < 2:
            return None
        try:
            return [abs(float(size[0])), abs(float(size[1]))]
        except (TypeError, ValueError):
            return None
    return None


def portal_clearance_aabb(
    graph: dict[str, Any], node_id: str
) -> tuple[list[float] | None, list[float] | None]:
    """Return the full node AABB used to reject unsafe traversal endpoints.

    A portal's interaction reference is often a compact handle/leaf proxy and
    need not share the full semantic node's AABB.  It remains appropriate for
    the crossing axis, while collision clearance must prefer the complete
    node geometry.
    """

    for node in graph.get("nodes") or []:
        if str(node.get("id") or "") != str(node_id or ""):
            continue
        attributes = node.get("attributes") or {}
        center = list(
            node.get("aabb_center")
            or attributes.get("interaction_reference_aabb_center")
            or node.get("centroid")
            or []
        )
        size = list(
            node.get("aabb_size")
            or attributes.get("interaction_reference_aabb_size")
            or []
        )
        if len(center) < 2 or len(size) < 2:
            return None, None
        try:
            return (
                [float(center[0]), float(center[1])],
                [abs(float(size[0])), abs(float(size[1]))],
            )
        except (TypeError, ValueError):
            return None, None
    return None, None


def _xyyaw(value: object) -> list[float] | None:
    """Normalize a public XY/Yaw payload without accepting malformed data."""

    if not isinstance(value, (list, tuple)):
        return None
    if len(value) < 2:
        return None
    try:
        return [
            float(value[0]),
            float(value[1]),
            float(value[2]) if len(value) > 2 else 0.0,
        ]
    except (TypeError, ValueError):
        return None


def _xy(value: object) -> list[float] | None:
    pose = _xyyaw(value)
    return pose[:2] if pose is not None else None


def _aabb_clearance_m(
    point_xy: Iterable[float],
    center_xy: Iterable[float],
    size_xy: Iterable[float] | None,
) -> float | None:
    """Return Euclidean clearance from an axis-aligned portal footprint.

    ``None`` means that the public graph did not publish a usable footprint;
    in that legacy case the caller retains geometric compatibility rather than
    manufacturing an AABB from the object centroid.
    """

    point = list(point_xy or [])
    center = list(center_xy or [])
    size = list(size_xy or [])
    if len(point) < 2 or len(center) < 2 or len(size) < 2:
        return None
    try:
        half_x = 0.5 * abs(float(size[0]))
        half_y = 0.5 * abs(float(size[1]))
        if half_x <= 1e-6 or half_y <= 1e-6:
            return None
        outside_x = abs(float(point[0]) - float(center[0])) - half_x
        outside_y = abs(float(point[1]) - float(center[1])) - half_y
    except (TypeError, ValueError):
        return None
    return math.hypot(max(0.0, outside_x), max(0.0, outside_y))


def _portal_axis_from_approach(
    approach_xyyaw: Iterable[float], portal_center: Iterable[float]
) -> tuple[float, float] | None:
    """Return the direction from the interacting side through the portal."""

    approach = list(approach_xyyaw or [])
    center = list(portal_center or [])
    if len(approach) < 2 or len(center) < 2:
        return None
    through_x = float(center[0]) - float(approach[0])
    through_y = float(center[1]) - float(approach[1])
    through_norm = math.hypot(through_x, through_y)
    if through_norm <= 1e-6:
        return None
    return through_x / through_norm, through_y / through_norm


def _deduplicated_xyyaw(values: Iterable[object]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for raw in values:
        pose = _xyyaw(raw)
        if pose is None:
            continue
        if any(
            math.hypot(pose[0] - previous[0], pose[1] - previous[1]) <= 1e-6
            and abs(math.atan2(math.sin(pose[2] - previous[2]), math.cos(pose[2] - previous[2])))
            <= 1e-6
            for previous in normalized
        ):
            continue
        normalized.append(pose)
    return normalized


def _post_open_opposite_portal_options(
    approach_xyyaw: Iterable[float],
    portal_center: Iterable[float],
    clearance_center_xy: Iterable[float] | None,
    clearance_size_xy: Iterable[float] | None,
    interaction_goal_candidates: Iterable[object],
    *,
    aabb_safety_margin_m: float = 0.05,
) -> list[list[float]]:
    """Select safe interaction poses on the opposite side of an opened door.

    The interaction candidate generator already emits both doorway sides.  A
    post-open continuation must not discard that information and recreate one
    radial point: a shifted or elongated AABB can put that reconstructed point
    in the door leaf.  Keep only candidates which actually cross the approach
    axis, reject candidates still within (or immediately against) the portal
    AABB, and prefer the straight/closest safe crossing first.
    """

    center = _xy(portal_center)
    axis = _portal_axis_from_approach(approach_xyyaw, portal_center)
    if center is None or axis is None:
        return []
    minimum_progress = max(1e-4, float(aabb_safety_margin_m))
    minimum_clearance = max(0.0, float(aabb_safety_margin_m))
    ordered: list[tuple[tuple[float, float, float, float], list[float]]] = []
    for candidate in _deduplicated_xyyaw(interaction_goal_candidates):
        relative_x = candidate[0] - center[0]
        relative_y = candidate[1] - center[1]
        axial_progress = relative_x * axis[0] + relative_y * axis[1]
        if axial_progress <= minimum_progress:
            continue
        lateral_offset = abs(relative_x * axis[1] - relative_y * axis[0])
        clearance = _aabb_clearance_m(
            candidate,
            clearance_center_xy or center,
            clearance_size_xy,
        )
        if clearance is not None and clearance < minimum_clearance:
            continue
        # The yaw of an interaction pose faces the door.  The continuation
        # should instead face through it so a final alignment cannot turn the
        # base back into the leaf.
        traversal_pose = [
            candidate[0],
            candidate[1],
            math.atan2(axis[1], axis[0]),
        ]
        axis_error = lateral_offset / max(axial_progress, 1e-6)
        # Rank a clean crossing before its tangential variants, then keep the
        # nearest candidate that has already cleared the AABB.
        ordering = (
            axis_error,
            axial_progress,
            lateral_offset,
            -(clearance if clearance is not None else 0.0),
        )
        ordered.append((ordering, traversal_pose))
    ordered.sort(key=lambda item: item[0])
    return _deduplicated_xyyaw(pose for _, pose in ordered)


def _translated_xyyaw_candidates(
    candidates: Iterable[object],
    source_center: Iterable[float] | None,
    destination_center: Iterable[float],
) -> list[list[float]]:
    """Move stored approach options with a refreshed portal center."""

    source = _xy(source_center)
    destination = _xy(destination_center)
    if source is None or destination is None:
        return _deduplicated_xyyaw(candidates)
    offset_x = destination[0] - source[0]
    offset_y = destination[1] - source[1]
    return [
        [pose[0] + offset_x, pose[1] + offset_y, pose[2]]
        for pose in _deduplicated_xyyaw(candidates)
    ]


def _traversal_goal_from_approach(
    approach_xyyaw: Iterable[float],
    portal_center: Iterable[float],
    traversal_distance_m: float,
) -> list[float] | None:
    """Project a far-side portal goal from an approach pose and current center."""

    approach = list(approach_xyyaw or [])
    center = list(portal_center or [])
    if len(approach) < 2 or len(center) < 2:
        return None
    axis = _portal_axis_from_approach(approach, center)
    if axis is None:
        return None
    unit_x, unit_y = axis
    traversal_distance = max(0.0, float(traversal_distance_m))
    return [
        float(center[0]) + unit_x * traversal_distance,
        float(center[1]) + unit_y * traversal_distance,
        math.atan2(unit_y, unit_x),
    ]


def build_post_interaction_traversal_candidate(
    active_candidate: dict[str, Any],
    feedback: dict[str, Any],
    *,
    robot_xy: Iterable[float] | None = None,
    traversal_distance_m: float = 0.9,
) -> BehaviorCandidate | None:
    """Build the immediate one-shot continuation of a successful portal open."""

    metadata = active_candidate.get("metadata") or {}
    command = active_candidate.get("interaction_command") or {}
    detail = feedback.get("detail") or {}
    if str(active_candidate.get("behavior_type") or "").upper() != "INTERACT":
        return None
    if str(metadata.get("node_type") or "").casefold() != "portal":
        return None
    status = str(feedback.get("status") or "").upper()
    success = feedback.get("success")
    if not (status == "SUCCEEDED" or (not status and success is True)):
        return None
    action = str(
        detail.get("action")
        or feedback.get("action")
        or command.get("action")
        or ""
    ).casefold()
    if action != "open":
        return None

    portal_id = str(
        detail.get("node_id")
        or feedback.get("node_id")
        or active_candidate.get("target_id")
        or command.get("node_id")
        or ""
    )
    if not portal_id:
        return None
    approach = list(
        detail.get("approach_goal_xyyaw")
        or command.get("interaction_approach_pose_xyyaw")
        or active_candidate.get("goal_xyyaw")
        or []
    )
    center = list(
        active_candidate.get("portal_center_xy")
        or metadata.get("portal_aabb_center_xy")
        or metadata.get("source_portal_aabb_center_xy")
        or []
    )
    portal_size = list(
        active_candidate.get("portal_aabb_size_xy")
        or metadata.get("portal_aabb_size_xy")
        or metadata.get("source_portal_aabb_size_xy")
        or []
    )
    clearance_center = list(
        active_candidate.get("portal_clearance_aabb_center_xy")
        or metadata.get("portal_clearance_aabb_center_xy")
        or metadata.get("source_portal_clearance_aabb_center_xy")
        or center
    )
    clearance_size = list(
        active_candidate.get("portal_clearance_aabb_size_xy")
        or metadata.get("portal_clearance_aabb_size_xy")
        or metadata.get("source_portal_clearance_aabb_size_xy")
        or portal_size
    )
    if len(approach) < 2 or len(center) < 2:
        return None

    traversal_distance = max(0.0, float(traversal_distance_m))
    source_goal_candidates = _deduplicated_xyyaw(
        metadata.get("goal_xyyaw_candidates") or []
    )
    traversal_options = _post_open_opposite_portal_options(
        approach,
        center,
        clearance_center,
        clearance_size,
        source_goal_candidates,
    )
    if traversal_options:
        goal = traversal_options[0]
        goal_source = "opposite_interaction_approach_candidates"
    else:
        goal = _traversal_goal_from_approach(
            approach,
            center,
            traversal_distance,
        )
        if goal is None:
            return None
        traversal_options = [goal]
        goal_source = "legacy_axis_projection"
    robot = list(robot_xy or [])
    distance_m = (
        math.hypot(goal[0] - float(robot[0]), goal[1] - float(robot[1]))
        if len(robot) >= 2
        else math.hypot(goal[0] - float(approach[0]), goal[1] - float(approach[1]))
    )
    event_id = str(
        detail.get("event_id")
        or feedback.get("event_id")
        or command.get("event_id")
        or active_candidate.get("decision_id")
        or "latest_open"
    )
    target_name = str(
        active_candidate.get("target_name")
        or detail.get("object_id")
        or command.get("object_id")
        or portal_id
    )
    confidence = float(
        (active_candidate.get("features") or {}).get("confidence", 1.0) or 1.0
    )
    return BehaviorCandidate(
        candidate_id=f"traverse:{portal_id}:{event_id}",
        behavior_type="NAVIGATE",
        source="post_interaction_feedback",
        target_id=portal_id,
        target_name=target_name,
        goal_xyyaw=goal,
        features={
            "exploration_gain": 1.25,
            "visibility_gain": 1.15,
            "semantic_gain": 1.0,
            "target_relevance": 0.0,
            "distance_m": distance_m,
            "interaction_cost": 0.0,
            "state_age_ratio": 0.0,
            "confidence": confidence,
            "priority": 1.0,
        },
        metadata={
            "node_type": "portal",
            "semantic_name": str(metadata.get("semantic_name") or "door"),
            "state": "open",
            "post_interaction_traversal": True,
            "opened_portal_id": portal_id,
            "source_interaction_event_id": event_id,
            "connected_room_ids": list(metadata.get("connected_room_ids") or []),
            "room_transition_required": True,
            "requires_approach": False,
            "verify_target_visibility": False,
            # These originate from the interaction's full two-sided approach
            # set, filtered to the opposite side and ordered for a straight,
            # AABB-clear crossing.  The executor preflights every option.
            "goal_xyyaw_candidates": traversal_options,
            "post_interaction_traversal_goal_source": goal_source,
            "post_interaction_traversal_option_count": len(traversal_options),
            "decision_local_transition": True,
            # The graph can refine this portal's reference geometry after the
            # action result.  Keep the physical approach anchor so the
            # decision node can project a fresh far-side goal at gate release
            # instead of retrying this initially frozen point.
            "source_interaction_approach_xyyaw": list(approach),
            "source_interaction_goal_xyyaw_candidates": source_goal_candidates,
            "source_portal_center_xy": [float(center[0]), float(center[1])],
            "source_portal_aabb_size_xy": (
                [float(portal_size[0]), float(portal_size[1])]
                if len(portal_size) >= 2
                else []
            ),
            "source_portal_clearance_aabb_center_xy": (
                [float(clearance_center[0]), float(clearance_center[1])]
                if len(clearance_center) >= 2
                else []
            ),
            "source_portal_clearance_aabb_size_xy": (
                [float(clearance_size[0]), float(clearance_size[1])]
                if len(clearance_size) >= 2
                else []
            ),
            "post_interaction_traversal_distance_m": traversal_distance,
        },
    )


def reproject_post_interaction_traversal_candidate(
    pending_candidate: dict[str, Any],
    candidate_snapshot: dict[str, Any],
    *,
    traversal_distance_m: float | None = None,
) -> dict[str, Any] | None:
    """Replace a cached traversal's frozen far-side goal with fresh geometry.

    A successful open publishes a new semantic planning map before the
    decision gate releases.  The portal center in that publication is the
    authoritative geometry for the next global plan; retaining the center
    captured before the action can put the continuation inside the door leaf
    or inflated wall.
    """

    portal_id = str(
        (pending_candidate.get("metadata") or {}).get("opened_portal_id")
        or pending_candidate.get("target_id")
        or ""
    )
    if not portal_id:
        return None
    metadata = pending_candidate.get("metadata") or {}
    approach = list(metadata.get("source_interaction_approach_xyyaw") or [])
    center = portal_center_xy(
        candidate_snapshot.get("graph_context") or {},
        portal_id,
    )
    if center is None:
        return None
    fresh_portal_size = portal_aabb_size_xy(
        candidate_snapshot.get("graph_context") or {},
        portal_id,
    )
    portal_size = (
        fresh_portal_size
        or _xy(metadata.get("source_portal_aabb_size_xy"))
        or _xy(metadata.get("portal_aabb_size_xy"))
        or []
    )
    fresh_clearance_center, fresh_clearance_size = portal_clearance_aabb(
        candidate_snapshot.get("graph_context") or {},
        portal_id,
    )
    clearance_center = (
        fresh_clearance_center
        or _xy(metadata.get("source_portal_clearance_aabb_center_xy"))
        or _xy(metadata.get("portal_clearance_aabb_center_xy"))
        or center
    )
    clearance_size = (
        fresh_clearance_size
        or _xy(metadata.get("source_portal_clearance_aabb_size_xy"))
        or _xy(metadata.get("portal_clearance_aabb_size_xy"))
        or portal_size
    )
    distance = (
        float(traversal_distance_m)
        if traversal_distance_m is not None
        else float(metadata.get("post_interaction_traversal_distance_m", 0.9))
    )
    source_center = (
        _xy(metadata.get("source_portal_center_xy"))
        or _xy(metadata.get("portal_aabb_center_xy"))
        or center
    )
    source_goal_candidates = list(
        metadata.get("source_interaction_goal_xyyaw_candidates") or []
    )
    translated_candidates = _translated_xyyaw_candidates(
        source_goal_candidates,
        source_center,
        center,
    )
    traversal_options = _post_open_opposite_portal_options(
        approach,
        center,
        clearance_center,
        clearance_size,
        translated_candidates,
    )
    if traversal_options:
        goal = traversal_options[0]
        goal_source = "refreshed_opposite_interaction_approach_candidates"
    else:
        goal = _traversal_goal_from_approach(
            approach,
            center,
            distance,
        )
        if goal is None:
            return None
        traversal_options = [goal]
        goal_source = "refreshed_legacy_axis_projection"

    projected = copy.deepcopy(pending_candidate)
    projected_metadata = projected.setdefault("metadata", {})
    projected_metadata["goal_xyyaw_candidates"] = traversal_options
    projected_metadata["post_interaction_traversal_goal_source"] = goal_source
    projected_metadata["post_interaction_traversal_option_count"] = len(
        traversal_options
    )
    projected_metadata["post_interaction_reprojected"] = True
    projected_metadata["post_interaction_reprojected_graph_revision"] = int(
        _as_int(candidate_snapshot.get("graph_revision"), 0) or 0
    )
    projected_metadata["post_interaction_reprojected_portal_center_xy"] = [
        float(center[0]),
        float(center[1]),
    ]
    if len(portal_size) >= 2:
        projected_metadata["post_interaction_reprojected_portal_aabb_size_xy"] = [
            float(portal_size[0]),
            float(portal_size[1]),
        ]
    if len(clearance_center) >= 2 and len(clearance_size) >= 2:
        projected_metadata["post_interaction_reprojected_clearance_aabb"] = {
            "center_xy": [float(clearance_center[0]), float(clearance_center[1])],
            "size_xy": [float(clearance_size[0]), float(clearance_size[1])],
        }
    projected["goal_xyyaw"] = goal
    robot = list(candidate_snapshot.get("robot_xy") or [])
    if len(robot) >= 2:
        projected.setdefault("features", {})["distance_m"] = math.hypot(
            goal[0] - float(robot[0]),
            goal[1] - float(robot[1]),
        )
    return projected


def inject_pending_traversal(
    candidate_snapshot: dict[str, Any],
    pending_candidate: dict[str, Any] | None,
) -> bool:
    """Inject a cached traversal unless the graph already published the same ID."""

    if not pending_candidate:
        return False
    metadata = pending_candidate.get("metadata") or {}
    pending_episode_id = str(metadata.get("source_episode_id") or "")
    snapshot_episode_id = str(candidate_snapshot.get("episode_id") or "")
    if (
        pending_episode_id
        and snapshot_episode_id
        and pending_episode_id != snapshot_episode_id
    ):
        return False
    candidate_id = str(pending_candidate.get("candidate_id") or "")
    if not candidate_id:
        return False
    candidates = list(candidate_snapshot.get("candidates") or [])
    if any(str(candidate.get("candidate_id") or "") == candidate_id for candidate in candidates):
        return False
    candidates.append(copy.deepcopy(pending_candidate))
    candidate_snapshot["candidates"] = candidates
    candidate_snapshot["candidate_count"] = len(candidates)
    return True


def pending_priority_candidate(
    candidates: Iterable[BehaviorCandidate], pending_candidate_id: str
) -> BehaviorCandidate | None:
    if not pending_candidate_id:
        return None
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.candidate_id == pending_candidate_id
            and bool((candidate.metadata or {}).get("post_interaction_traversal"))
        ),
        None,
    )


def is_terminal_post_interaction_traversal_failure(
    candidate_id: str,
    behavior_type: str,
    status: str,
    detail: dict[str, Any] | None = None,
) -> bool:
    """Identify a one-shot portal traversal that must not be retried unchanged."""

    normalized_status = str(status or "").upper()
    if (
        str(behavior_type or "").upper() != "NAVIGATE"
        or not str(candidate_id or "").startswith("traverse:")
        or normalized_status not in {"FAILED", "REJECTED", "CANCELED"}
    ):
        return False
    return not (
        normalized_status == "CANCELED"
        and str((detail or {}).get("reason") or "") == "preempted_by_target"
    )
